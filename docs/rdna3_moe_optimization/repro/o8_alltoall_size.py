#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""An all-to-all's cost is not monotonic in its size, and the MoE payload sits in a hole.

    torchrun --nproc_per_node=4 --master_port=29911 \
      docs/rdna3_moe_optimization/repro/o8_alltoall_size.py

o8_ep_breakdown.py turned up something odd while sizing a different optimisation:
padding the dispatch payload's rows from 8192 to 8704 bytes made the collective
25% *faster* despite moving 6% more data. Rows are not a thing a collective can
see -- ``all_to_all_single`` moves one contiguous block per destination -- so the
row size can only matter through the block size it produces. At 1024 tokens a
rank sends 512 rows to each peer, and 512 x 8192 B is exactly 4 MB.

This isolates the effect: sweep the per-destination block size on its own,
nothing else in the picture. If the cost has holes at round sizes, then a MoE
payload -- whose size is model_dim x 2 bytes x a token count, all powers of two --
lands in one essentially by construction, and any of them can be stepped out of
for the price of a few percent more bytes.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist


def bench_us(fn, *, warmup=5, iters=30):
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
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--mb", type=float, default=4.0, help="per-destination block size to sweep around, MB")
    args = p.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    ws, rank = dist.get_world_size(), dist.get_rank()

    base = int(args.mb * 1024 * 1024) // 2  # bf16 elements per destination

    def run(n):
        send = torch.ones(ws * n, dtype=torch.bfloat16, device=dev)
        recv = torch.empty_like(send)
        us = bench_us(lambda: dist.all_to_all_single(recv, send), iters=args.iters)
        del send, recv
        return us, (ws - 1) * n * 2 / (us * 1e-6) / 1e9

    # Section 1: is the block's byte alignment what matters?
    fracs = [1.0, 1.008, 1.016, 1.031, 1.063, 1.094, 1.125, 1.19, 1.25, 1.5, 2.0, 0.75, 0.94, 0.5]
    coarse = []
    for n in sorted({int(base * f) for f in fracs}):
        us, gbs = run(n)
        coarse.append((n * 2, us, gbs))

    # Section 2: among blocks that *are* aligned, how much is left?
    step = 8  # bf16 elements, i.e. 16 bytes
    fine = []
    for n in range(int(base * 0.75) // step * step, int(base * 1.55), step * 4096):
        us, gbs = run(n)
        fine.append((n * 2, us, gbs))

    dist.barrier()
    if rank == 0:
        print()
        print("=" * 72)
        print(f"all_to_all_single cost vs per-destination block size, world_size={ws}")
        print("=" * 72)
        print()
        print("1. alignment: a block that is not a multiple of 16 bytes falls off a cliff")
        print()
        print(f"  {'block/dest':>12} {'16B mult':>9} {'us':>9} {'GB/s off-rank':>14}")
        for b, us, gbs in coarse:
            print(f"  {b / 1e6:9.3f} MB {str(b % 16 == 0):>9} {us:9.1f} {gbs:14.2f}")
        print()
        slow = [g for b, _, g in coarse if b % 16]
        fast = [g for b, _, g in coarse if b % 16 == 0]
        if slow:
            print(f"  unaligned: {min(slow):.2f}-{max(slow):.2f} GB/s     aligned: {min(fast):.2f}-{max(fast):.2f} GB/s")
        print()
        print("  That is a ~10x cliff, and it is a landmine for anyone widening the")
        print("  payload row: the block is rows x row_bytes and the row count varies")
        print("  with the routing, so only a row that is itself a multiple of 16 bytes")
        print("  is safe. A row of model_dim x 2 + 8 -- the obvious way to bolt two")
        print("  int32 of metadata onto a bf16 activation -- is not one.")
        print()
        print("2. among aligned blocks, size still matters")
        print()
        print(f"  {'block/dest':>12} {'us':>9} {'GB/s off-rank':>14}")
        for b, us, gbs in fine:
            print(f"  {b / 1e6:9.3f} MB {us:9.1f} {gbs:14.2f}")
        print()
        best = min(fine, key=lambda r: r[1])
        worst_gbs = min(fine, key=lambda r: r[2])
        print(f"  best {best[0] / 1e6:.3f} MB at {best[2]:.2f} GB/s, worst {worst_gbs[0] / 1e6:.3f} MB at {worst_gbs[2]:.2f} GB/s")
        print()
        print("  The MoE dispatch payload is tokens x topk / world_size rows of")
        print("  model_dim x 2 bytes -- powers of two all the way down, so the block")
        print("  lands on a round size by construction. Whether that is the good end of")
        print("  this curve or the bad one is not something the layer chose.")
        print()

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
