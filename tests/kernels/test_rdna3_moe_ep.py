# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Expert-parallel RDNA3 MoE: dispatch, the local layer, combine.

The reference is the same machine's single-rank path. Every rank builds the whole
expert set from one seed, keeps its own tokens, and computes the layer twice: once
with all experts local (EP=1, ``moe_forward``), once with only its shard and the
other ranks holding the rest (``ep_moe_forward``). The two must agree.

They should agree *exactly*, not approximately, which is the point of dispatching
one row per routed slot: a row's arithmetic does not depend on which rank or which
tile it lands in, only on its own activations and its expert's weights. A
mismatch past a rounding step means the exchange moved or paired something wrong,
which a loose tolerance would hide.

    pytest tests/kernels/test_rdna3_moe_ep.py            # launches torchrun
    torchrun --nproc_per_node=4 tests/kernels/test_rdna3_moe_ep.py   # directly
"""

import os
import shutil
import subprocess
import sys

import pytest
import torch

# RCCL maps one of these per rank at communicator init, and the size does not
# move with NCCL_BUFFSIZE or the channel count -- so a container started with
# Docker's default 64 MB /dev/shm cannot hold more than three ranks.
_NCCL_SHM_PER_RANK = 24 << 20

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_BUILD = os.path.join(_REPO, "build-fly", "python_packages")
if os.path.isdir(_BUILD) and _BUILD not in sys.path:
    sys.path.insert(0, _BUILD)


def _count_physical_gpus() -> int:
    """GPU count as the machine has it, not as HIP_VISIBLE_DEVICES filtered it."""
    env = {k: v for k, v in os.environ.items() if k != "HIP_VISIBLE_DEVICES"}
    try:
        out = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return int(out.stdout.strip().splitlines()[-1])
    except Exception:
        return 0


# ── worker ───────────────────────────────────────────────────────────────────


def _worker() -> int:
    import torch.distributed as dist

    from flydsl.runtime.device import get_rocm_arch
    from kernels.moe.rdna3_moe.dispatch_kernel import capacity_for, safe_capacity
    from kernels.moe.rdna3_moe.ep import ep_dispatch, ep_moe_forward
    from kernels.moe.rdna3_moe.forward import moe_forward, moe_gating

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    arch = str(get_rocm_arch() or "")
    if not arch.startswith("gfx11"):
        print(f"SKIP: RDNA3 MoE EP requires gfx11*, got {arch!r}")
        return 0

    dev = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=dev)
    rank, world = dist.get_rank(), dist.get_world_size()
    ok = True

    def check(name, got, ref, *, exact=True):
        nonlocal ok
        if got.shape != ref.shape:
            print(f"[rank {rank}] FAIL {name}: shape {tuple(got.shape)} != {tuple(ref.shape)}")
            ok = False
            return
        err = (got.float() - ref.float()).abs().max().item()
        bad = err > (0.0 if exact else 2e-2)
        print(f"[rank {rank}] {'FAIL' if bad else 'ok  '} {name}: max err {err:.3e}")
        ok = ok and not bad

    model_dim, inter_dim, topk = 256, 128, 2
    epr = 2
    experts = world * epr

    # One seed for the weights on every rank, so each holds the same global set
    # and can slice its own shard out of it. The tokens are per-rank, and the
    # counts differ so the exchange is asymmetric.
    gen = torch.Generator(device=dev).manual_seed(1234)
    w1 = (torch.randn(experts, 2 * inter_dim, model_dim, generator=gen, device=dev, dtype=torch.bfloat16) * 0.05).contiguous()
    w2 = (torch.randn(experts, model_dim, inter_dim, generator=gen, device=dev, dtype=torch.bfloat16) * 0.05).contiguous()

    tokens = 32 + 16 * rank
    tgen = torch.Generator(device=dev).manual_seed(100 + rank)
    x = (torch.randn(tokens, model_dim, generator=tgen, device=dev, dtype=torch.bfloat16) * 0.1).contiguous()
    logits = torch.randn(tokens, experts, generator=tgen, device=dev, dtype=torch.float32).contiguous()
    ids, weights = moe_gating(logits, topk=topk)

    shard = slice(rank * epr, (rank + 1) * epr)
    w1_local, w2_local = w1[shard].contiguous(), w2[shard].contiguous()

    for tile_m in (16, 32):
        ref = moe_forward(x, w1, w2, ids, weights, tile_m=tile_m)
        for plan in ("device", "host"):
            got = ep_moe_forward(x, w1_local, w2_local, ids, weights, tile_m=tile_m, plan_impl=plan, group=None)
            check(f"ep vs single-rank, tile_m={tile_m}, plan={plan}", got, ref)

    # The plan kernel against the torch calls it replaced: same permutation, same
    # counts, same metadata, not merely the same answer downstream.
    d_dev = ep_dispatch(x, ids, weights, experts_per_rank=epr, plan_impl="device")
    d_host = ep_dispatch(x, ids, weights, experts_per_rank=epr, plan_impl="host")
    if d_dev.exchange.send_counts != d_host.exchange.send_counts:
        print(f"[rank {rank}] FAIL plan counts: {d_dev.exchange.send_counts} != {d_host.exchange.send_counts}")
        ok = False
    if not torch.equal(d_dev.exchange.send_order, d_host.exchange.send_order):
        print(f"[rank {rank}] FAIL plan order: the kernel's permutation is not the stable sort's")
        ok = False
    check("plan payload", d_dev.x, d_host.x)
    check("plan expert ids", d_dev.topk_ids, d_host.topk_ids)
    check("plan weights", d_dev.topk_weights, d_host.topk_weights)

    # The dispatch itself: what arrived should be the rows the senders picked for
    # this rank's experts, and no others.
    disp = ep_dispatch(x, ids, weights, experts_per_rank=epr)
    mine = ((ids >= rank * epr) & (ids < (rank + 1) * epr)).sum().item()
    sent_home = int(disp.exchange.send_counts[rank])
    if sent_home != mine:
        print(f"[rank {rank}] FAIL dispatch: kept {sent_home} of its own rows, routing says {mine}")
        ok = False
    rows = int(disp.x.shape[0])
    if rows != sum(disp.exchange.recv_counts):
        print(f"[rank {rank}] FAIL dispatch: {rows} rows arrived, counts say {sum(disp.exchange.recv_counts)}")
        ok = False
    if rows and not bool(((disp.topk_ids >= 0) & (disp.topk_ids < epr)).all()):
        print(f"[rank {rank}] FAIL dispatch: an arriving row names an expert this rank does not own")
        ok = False
    # Every routed slot in the group is dispatched exactly once, so the rows
    # received across the group must total the slots sent across it.
    totals = torch.tensor([rows, tokens * topk], dtype=torch.int64, device=dev)
    dist.all_reduce(totals)
    if totals[0].item() != totals[1].item():
        print(f"[rank {rank}] FAIL dispatch: {totals[0].item()} rows received group-wide, {totals[1].item()} sent")
        ok = False

    # A rank with nothing routed to it still has to take part.
    empty_ids = torch.zeros(tokens, topk, dtype=torch.int32, device=dev)  # everything to expert 0, rank 0
    ref = moe_forward(x, w1, w2, empty_ids, weights, tile_m=16)
    for plan in ("device", "host"):
        got = ep_moe_forward(x, w1_local, w2_local, empty_ids, weights, tile_m=16, plan_impl=plan)
        check(f"one rank owns every token, plan={plan}", got, ref)

    # ── The fixed-capacity exchange ──────────────────────────────────────────
    # Every rank has to use the same capacity, and the token counts here are
    # ragged on purpose, so it comes from the largest of them rather than from
    # each rank's own -- which is what "auto" would do. The widest is the last
    # rank's, and every rank can work that out without asking.
    max_tokens = 32 + 16 * (world - 1)
    max_rows = max_tokens * topk
    safe = safe_capacity(tokens=max_tokens, topk=topk)
    roomy = capacity_for(tokens=max_tokens, topk=topk, world_size=world)
    for tile_m in (16, 32):
        ref = moe_forward(x, w1, w2, ids, weights, tile_m=tile_m)
        for cap, label in ((safe, f"safe={safe}"), (roomy, f"roomy={roomy}")):
            got = ep_moe_forward(x, w1_local, w2_local, ids, weights, tile_m=tile_m, capacity=cap, group=None)
            check(f"capacity vs single-rank, tile_m={tile_m}, {label}", got, ref)

    # Padding has to be invisible, not merely harmless downstream: what arrives
    # is world*capacity slots of which only some name an expert, and the rest
    # must carry the id the routing kernel drops.
    disp = ep_dispatch(x, ids, weights, experts_per_rank=epr, capacity=safe)
    if int(disp.x.shape[0]) != world * safe:
        print(f"[rank {rank}] FAIL capacity: {disp.x.shape[0]} slots arrived, capacity says {world * safe}")
        ok = False
    real = int((disp.topk_ids >= 0).sum().item())
    if real != sum(ep_dispatch(x, ids, weights, experts_per_rank=epr).exchange.recv_counts):
        print(f"[rank {rank}] FAIL capacity: {real} slots name an expert, the packed dispatch got another count")
        ok = False
    if not bool(((disp.topk_ids == -1) | ((disp.topk_ids >= 0) & (disp.topk_ids < epr))).all()):
        print(f"[rank {rank}] FAIL capacity: a slot is neither padding nor an expert this rank owns")
        ok = False

    # At safe_capacity the adversarial routing -- every token to rank 0 -- still
    # has room, because that capacity is the whole slot count.
    ref = moe_forward(x, w1, w2, empty_ids, weights, tile_m=16)
    got = ep_moe_forward(x, w1_local, w2_local, empty_ids, weights, tile_m=16, capacity=safe)
    check(f"one rank owns every token, capacity={safe}", got, ref)

    # And below it that routing overflows, which has to be heard about. The
    # detection is deferred by design, so it takes a few steps to surface.
    tight = max(1, max_rows // world)
    raised = False
    try:
        for _ in range(8):
            ep_moe_forward(x, w1_local, w2_local, empty_ids, weights, tile_m=16, capacity=tight)
        torch.cuda.synchronize()
        ep_moe_forward(x, w1_local, w2_local, empty_ids, weights, tile_m=16, capacity=tight)
    except RuntimeError as exc:
        raised = "dropped" in str(exc)
    print(f"[rank {rank}] {'ok  ' if raised else 'FAIL'} overflow at capacity={tight} is reported")
    ok = ok and raised

    flags = torch.tensor([1 if ok else 0], dtype=torch.int32, device=dev)
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    if rank == 0 and flags.item() == 1:
        print(f"All tests PASSED (world_size={world}, experts={experts}, epr={epr})")
    dist.barrier()
    dist.destroy_process_group()
    return 0 if flags.item() == 1 else 1


# ── launcher ─────────────────────────────────────────────────────────────────


@pytest.mark.multi_gpu
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_ep_moe_matches_the_single_rank_layer(world_size):
    have = _count_physical_gpus()
    if have < world_size:
        pytest.skip(f"needs >= {world_size} physical GPUs, found {have}")

    try:
        shm_free = shutil.disk_usage("/dev/shm").free
    except OSError:
        shm_free = 0
    need = world_size * _NCCL_SHM_PER_RANK
    if shm_free < need:
        pytest.skip(
            f"/dev/shm has {shm_free >> 20} MB free and RCCL wants about {need >> 20} MB for "
            f"{world_size} ranks; start the container with --shm-size=2g"
        )

    env = {k: v for k, v in os.environ.items() if k != "HIP_VISIBLE_DEVICES"}
    env["PYTHONPATH"] = _REPO + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            "torchrun",
            f"--nproc_per_node={world_size}",
            f"--master_port={29820 + world_size}",
            os.path.abspath(__file__),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    tail = f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}"
    assert result.returncode == 0, f"EP={world_size} failed (exit {result.returncode})\n{tail}"
    if "SKIP:" in result.stdout:
        pytest.skip(result.stdout[result.stdout.index("SKIP:") :].splitlines()[0])
    assert "All tests PASSED" in result.stdout, f"no success banner for EP={world_size}\n{tail}"


if __name__ == "__main__":
    raise SystemExit(_worker())
