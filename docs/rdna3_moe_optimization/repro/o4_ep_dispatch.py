#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""o4: the expert-parallel dispatch plan, in one launch instead of twenty calls.

Two modes. The diagnosis runs on one GPU:

    python docs/rdna3_moe_optimization/repro/o4_ep_dispatch.py

and the A/B needs two ranks:

    rm -f /dev/shm/nccl-*
    torchrun --nproc_per_node=2 --master_port=29870 \
      docs/rdna3_moe_optimization/repro/o4_ep_dispatch.py

o4 is the same lesson as o1 and o2, moved to a different subsystem: at decode
sizes the cost was never the work, it was the number of times the host asked for
work. Building the dispatch permutation took roughly twenty torch calls, each
touching a few dozen elements, with a synchronising readback in the middle that
stopped the host from running ahead.

The single-GPU section reproduces the per-op profiling that found this, and it
is worth running on its own -- the torch traps it shows up are not specific to
MoE.
"""

from __future__ import annotations

import os

import torch

from common import bench_us, device_banner, header, section

MODEL_DIM, EXPERTS, TOPK = 2048, 16, 2


def torch_traps(tokens):
    """The per-op profile that decided what o4 should replace."""
    dev = "cuda"
    rows = tokens * TOPK
    world = 2
    gen = torch.Generator(device=dev).manual_seed(0)
    ids = torch.randint(0, EXPERTS, (rows,), generator=gen, device=dev, dtype=torch.int64)
    dest = torch.div(ids, EXPERTS // world, rounding_mode="floor")
    x = (torch.randn(rows, MODEL_DIM, generator=gen, device=dev, dtype=torch.bfloat16)).contiguous()

    print(f"  rows={rows} ({tokens} tokens x topk {TOPK}), payload {rows * MODEL_DIM * 2 / 1e6:.2f} MB")
    print()
    print(f"  {'op':<46} {'us':>8}  note")

    t = bench_us(lambda: torch.bincount(dest, minlength=world))
    print(f"  {'torch.bincount(dest)':<46} {t:8.1f}  counts {rows} elements")

    def sort_edges():
        order = torch.sort(dest, stable=True)
        return torch.searchsorted(order.values, torch.arange(world + 1, device=dev))

    t2 = bench_us(sort_edges)
    print(f"  {'torch.sort(stable) + searchsorted':<46} {t2:8.1f}  same counts, and the permutation too")
    if t2 < t:
        print(f"  {'':46} {'':>8}  -> sorting is CHEAPER than counting; bincount likely syncs")

    t = bench_us(lambda: dest[:world].tolist())
    print(f"  {'.tolist() on a few elements':<46} {t:8.1f}  a full device-to-host sync")

    t = bench_us(lambda: bool(((ids < 0) | (ids >= EXPERTS)).any()))
    print(f"  {'bool(((ids<0)|(ids>=E)).any())':<46} {t:8.1f}  a range check is a hidden sync")
    t = bench_us(lambda: ((ids < 0) | (ids >= EXPERTS)).sum().reshape(1))
    print(f"  {'  same check kept on device (.sum())':<46} {t:8.1f}  fold it into the one readback instead")

    idx = torch.argsort(dest, stable=True).to(torch.int32)
    t = bench_us(lambda: x.index_select(0, idx))
    gb = rows * MODEL_DIM * 2 / (t * 1e-6) / 1e9
    print(f"  {'x.index_select (the actual data movement)':<46} {t:8.1f}  {gb:.0f} GB/s -- NOT a trap, leave it alone")

    t = bench_us(lambda: torch.empty(rows, MODEL_DIM, dtype=torch.bfloat16, device=dev))
    print(f"  {'torch.empty()':<46} {t:8.1f}  allocation is not free either")


def single_gpu():
    header("o4  diagnosis: what the host dispatch plan was actually spending",
           "per-op profile -- run this before writing any kernel")
    device_banner()

    section("decode size: a few dozen elements per call")
    torch_traps(32)
    section("prefill size: the same ops, now with real data behind them")
    torch_traps(1024)

    section("what this says")
    print("  At decode the payload move is a rounding error and everything else is")
    print("  fixed cost per call. That is a launch-count problem, not a bandwidth")
    print("  problem, so the fix is to do the whole plan in one kernel -- which is")
    print("  o1's counting sort with the destination *rank* as the key instead of")
    print("  the expert, and no tile padding, because an all-to-all needs each")
    print("  destination's rows packed end to end.")
    print()
    print("  Three fixes landed before the kernel, just from this table:")
    print("   - bincount -> sort(stable) + searchsorted (the sort returns the")
    print("     permutation for free, so it replaces a later gather too)")
    print("   - two .tolist() calls merged into one torch.cat readback")
    print("   - the range check folded into that same readback instead of being")
    print("     its own sync")
    print()
    print("  For the two-rank A/B of the kernel itself:")
    print("    rm -f /dev/shm/nccl-*")
    print("    torchrun --nproc_per_node=2 --master_port=29870 \\")
    print("      docs/rdna3_moe_optimization/repro/o4_ep_dispatch.py")


def distributed():
    import torch.distributed as dist

    from kernels.moe.rdna3_moe.ep import ep_dispatch
    from kernels.moe.rdna3_moe.forward import moe_gating

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=dev)
    rank, world = dist.get_rank(), dist.get_world_size()
    epr = EXPERTS // world

    if rank == 0:
        header("o4  the A/B: host plan vs one-launch plan", f"EP={world}, experts_per_rank={epr}")
        device_banner()
        print(f"  {'tokens/rank':>12} {'plan_impl':>10} {'dispatch us':>12} {'delta':>8}")

    for tokens in (32, 1024):
        gen = torch.Generator(device=dev).manual_seed(7)
        x = (torch.randn(tokens, MODEL_DIM, generator=gen, device=dev, dtype=torch.bfloat16) * 0.1).contiguous()
        logits = torch.randn(tokens, EXPERTS, generator=gen, device=dev, dtype=torch.float32).contiguous()
        ids, wts = moe_gating(logits, topk=TOPK)

        # The kernel plan must reproduce the torch plan exactly, not merely give
        # an answer that happens to match downstream.
        d_host = ep_dispatch(x, ids, wts, experts_per_rank=epr, plan_impl="host")
        d_dev = ep_dispatch(x, ids, wts, experts_per_rank=epr, plan_impl="device")
        same = (
            torch.equal(d_host.x, d_dev.x)
            and torch.equal(d_host.topk_ids, d_dev.topk_ids)
            and torch.equal(d_host.topk_weights, d_dev.topk_weights)
        )

        times = {}
        for impl in ("host", "device"):
            t = bench_us(
                lambda impl=impl: ep_dispatch(x, ids, wts, experts_per_rank=epr, plan_impl=impl),
                warmup=5,
                iters=20,
            )
            stat = torch.tensor([t], dtype=torch.float64, device=dev)
            dist.all_reduce(stat, op=dist.ReduceOp.MAX)
            times[impl] = stat.item()
        if rank == 0:
            print(f"  {tokens:>12} {'host':>10} {times['host']:12.1f} {'':>8}")
            print(
                f"  {tokens:>12} {'device':>10} {times['device']:12.1f} "
                f"{100 * (times['device'] / times['host'] - 1):+7.1f}%"
            )
            print(f"  {'':>12} payload/x/meta bit-identical between the two plans: {same}")

    if rank == 0:
        section("what is left")
        print("  Each collective costs 45-50 us regardless of what it carries, and")
        print("  dispatch runs three of them (counts, payload, meta) plus a sync.")
        print("  At decode that floor is about half the remaining time. The two")
        print("  untaken levers are folding meta into the payload collective, and")
        print("  fixing rows per rank pair so the splits become compile-time")
        print("  constants -- which is also what graph capture would need.")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        distributed()
    else:
        single_gpu()
