#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""What expert parallelism costs the RDNA3 MoE layer, one node.

Splits the layer into the three things a rank does -- dispatch, the local layer,
combine -- and times each, so the exchange can be compared against the compute
it enables rather than against nothing.

    torchrun --nproc_per_node=2 scripts/bench_rdna3_moe_ep.py --tokens 1024

Rank 0 prints. The rows are per-rank wall time, so at EP=N the layer column is
roughly the single-card number divided by N (each rank holds 1/N of the experts
and, on a balanced routing, sees topk*tokens/N rows) while dispatch and combine
are new work that did not exist at EP=1.

Note what the timings include: dispatch ends with a host synchronise, because
``all_to_all_single`` needs its split sizes as host integers. That stall is real
and belongs in the number.
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_BUILD = os.path.join(_REPO, "build-fly", "python_packages")
if os.path.isdir(_BUILD) and _BUILD not in sys.path:
    sys.path.insert(0, _BUILD)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.moe.rdna3_moe.ep import ep_combine, ep_dispatch, ep_moe_forward  # noqa: E402
from kernels.moe.rdna3_moe.forward import moe_forward, moe_gating  # noqa: E402


def bench_us(fn, *, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=1024, help="tokens per rank")
    p.add_argument("--model-dim", type=int, default=2048)
    p.add_argument("--inter-dim", type=int, default=768)
    p.add_argument("--experts-per-rank", type=int, default=8)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--tiles", default="16,32,64")
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    arch = str(get_rocm_arch() or "")
    if not arch.startswith("gfx11"):
        print(f"ERROR: needs gfx11*, got {arch!r}")
        return 1

    dist.init_process_group(backend="nccl", device_id=dev)
    rank, world = dist.get_rank(), dist.get_world_size()
    epr = args.experts_per_rank
    experts = world * epr
    md, idim, topk, tokens = args.model_dim, args.inter_dim, args.topk, args.tokens

    gen = torch.Generator(device=dev).manual_seed(7)
    w1 = (torch.randn(epr, 2 * idim, md, generator=gen, device=dev, dtype=torch.bfloat16) * 0.05).contiguous()
    w2 = (torch.randn(epr, md, idim, generator=gen, device=dev, dtype=torch.bfloat16) * 0.05).contiguous()
    x = (torch.randn(tokens, md, generator=gen, device=dev, dtype=torch.bfloat16) * 0.1).contiguous()
    logits = torch.randn(tokens, experts, generator=gen, device=dev, dtype=torch.float32).contiguous()
    ids, weights = moe_gating(logits, topk=topk)

    # One rank's share of the work, at a balanced routing: it owns 1/world of the
    # experts, so it sees about that share of the group's routed rows.
    rows_group = tokens * topk * world
    flop_rank = 2.0 * (rows_group / world) * (2 * idim + md) * md

    if rank == 0:
        print(f"device={torch.cuda.get_device_name(0)} arch={arch} world_size={world}")
        print(f"tokens/rank={tokens} model={md} inter={idim} experts={experts} (epr={epr}) topk={topk}")
        print(
            f"\n{'tile':>5} {'plan':>7} {'dispatch':>9} {'layer':>9} {'combine':>9} "
            f"{'total':>9} {'TFLOP/s':>8} {'exch %':>7}"
        )
        print("-" * 70)

    for tile_m in (int(t) for t in args.tiles.split(",")):
        for plan_impl in ("device", "host"):
            _run_one(args, tile_m, plan_impl, x, ids, weights, w1, w2, epr, md, dev, rank, flop_rank)

    # What the transport costs on its own, so the dispatch column can be read
    # against a floor rather than against zero: one all-to-all of the same
    # payload, one of the counts, and the host sync the splits force.
    rows_send = tokens * topk
    src = torch.empty(rows_send, md, dtype=torch.bfloat16, device=dev)
    dst = torch.empty_like(src)
    t_payload = bench_us(lambda: dist.all_to_all_single(dst, src), iters=args.iters)
    counts = torch.ones(world, dtype=torch.int64, device=dev)
    counts_out = torch.empty_like(counts)
    t_counts = bench_us(lambda: dist.all_to_all_single(counts_out, counts), iters=args.iters)
    t_sync = bench_us(lambda: counts.tolist(), iters=args.iters)

    if rank == 0:
        bytes_out = rows_send * md * 2 * (world - 1) / world
        print("-" * 70)
        print(f"  payload off-rank per dispatch: {bytes_out / 1e6:.2f} MB (same again for combine)")
        print(f"  transport floor: payload all-to-all {t_payload:.1f} us")
        print(f"                   counts all-to-all  {t_counts:.1f} us + host sync {t_sync:.1f} us")

    dist.barrier()
    dist.destroy_process_group()
    return 0


def _run_one(args, tile_m, plan_impl, x, ids, weights, w1, w2, epr, md, dev, rank, flop_rank):
    """Time one (tile, plan) pair, in pieces and end to end."""
    disp = ep_dispatch(x, ids, weights, experts_per_rank=epr, plan_impl=plan_impl)
    rows = int(disp.x.shape[0])
    y_local = (
        moe_forward(disp.x, w1, w2, disp.topk_ids, disp.topk_weights, tile_m=tile_m)
        if rows
        else torch.empty(0, md, dtype=x.dtype, device=dev)
    )

    t_disp = bench_us(
        lambda: ep_dispatch(x, ids, weights, experts_per_rank=epr, plan_impl=plan_impl), iters=args.iters
    )
    t_layer = (
        bench_us(
            lambda: moe_forward(disp.x, w1, w2, disp.topk_ids, disp.topk_weights, tile_m=tile_m),
            iters=args.iters,
        )
        if rows
        else 0.0
    )
    t_comb = bench_us(lambda: ep_combine(y_local, disp.exchange), iters=args.iters)
    t_all = bench_us(
        lambda: ep_moe_forward(
            x, w1, w2, ids, weights, experts_per_rank=epr, tile_m=tile_m, plan_impl=plan_impl
        ),
        iters=args.iters,
    )

    # Slowest rank sets the pace, so report that one.
    stats = torch.tensor([t_disp, t_layer, t_comb, t_all], dtype=torch.float64, device=dev)
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    t_disp, t_layer, t_comb, t_all = (v.item() for v in stats)
    if rank == 0:
        tf = flop_rank / (t_all * 1e-6) / 1e12
        print(
            f"{tile_m:5d} {plan_impl:>7} {t_disp:9.1f} {t_layer:9.1f} {t_comb:9.1f} {t_all:9.1f} "
            f"{tf:8.2f} {100 * (t_disp + t_comb) / t_all:6.1f}%"
        )


if __name__ == "__main__":
    raise SystemExit(main())
