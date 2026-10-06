#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Does adding cards to the *same* batch still help?

    for n in 2 4 8; do
      HIP_VISIBLE_DEVICES=$(seq -s, 0 $((n-1))) torchrun --nproc_per_node=$n \
        --master_port=29993 docs/rdna3_moe_optimization/repro/ep_strong_scaling.py
    done

Every EP number in BENCHMARK.md §8 and in o9/o10 is **weak** scaling: tokens per
rank held fixed, so adding a card adds work. That measures whether the machine
can absorb more load, and the answer there is yes. It does not answer the
question people actually ask about a wall -- given one batch, does the eighth
card still buy anything? -- because in that regime each rank's share *shrinks*
while the exchange does not.

So this fixes the batch across the group and splits it: at EP=W each rank gets
``total / W``. Wall time is then comparable across W directly. The expert count
is fixed too, so it is the same model throughout and a rank simply holds fewer
of them.

Comparing across world sizes means comparing across processes, which §1.4 says
not to trust. The mitigation is the drift probe: a fixed local layer, the same
shape and the same work at every W. If the probe moves between runs, the machine
moved and the scaling numbers moved with it.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_BUILD = os.path.join(_REPO, "build-fly", "python_packages")
if os.path.isdir(_BUILD) and _BUILD not in sys.path:
    sys.path.insert(0, _BUILD)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from kernels.moe.rdna3_moe.ep import ep_combine, ep_dispatch, ep_moe_forward  # noqa: E402
from kernels.moe.rdna3_moe.forward import moe_forward  # noqa: E402


def bench_us(fn, *, warmup=3, iters=20):
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


def slowest(value, dev):
    """The pace is set by the last rank to finish, so report that one."""
    t = torch.tensor(float(value), dtype=torch.float64, device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prefill-tokens", type=int, default=8192, help="tokens for the whole group")
    p.add_argument("--decode-tokens", type=int, default=256, help="tokens for the whole group")
    p.add_argument("--model-dim", type=int, default=4096)
    p.add_argument("--inter-dim", type=int, default=14336)
    p.add_argument("--experts", type=int, default=8, help="total, so a rank holds experts // world")
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    world, rank = dist.get_world_size(), dist.get_rank()

    if args.experts % world:
        if rank == 0:
            print(f"SKIP: experts={args.experts} does not divide by world_size={world}")
        dist.destroy_process_group()
        return 0
    epr = args.experts // world
    md, idim, topk = args.model_dim, args.inter_dim, args.topk

    gen = torch.Generator(device=dev).manual_seed(67 + rank)
    w1 = (torch.randn(epr, 2 * idim, md, generator=gen, device=dev, dtype=torch.bfloat16) * 0.02).contiguous()
    w2 = (torch.randn(epr, md, idim, generator=gen, device=dev, dtype=torch.bfloat16) * 0.02).contiguous()

    # The drift probe: one expert, 512 rows, identical work at every world size.
    probe_w1 = (torch.randn(1, 2 * idim, md, generator=gen, device=dev, dtype=torch.bfloat16) * 0.02).contiguous()
    probe_w2 = (torch.randn(1, md, idim, generator=gen, device=dev, dtype=torch.bfloat16) * 0.02).contiguous()
    probe_x = (torch.randn(512, md, generator=gen, device=dev, dtype=torch.bfloat16) * 0.1).contiguous()
    probe_ids = torch.zeros(512, 1, dtype=torch.int32, device=dev)
    probe_wt = torch.ones(512, 1, dtype=torch.float32, device=dev)

    rows = []
    for label, total, tile_m, capacity in (
        ("prefill", args.prefill_tokens, 128, None),
        ("decode", args.decode_tokens, 64, None),
        ("decode+cap", args.decode_tokens, 64, "auto"),
    ):
        if total % world:
            continue
        tokens = total // world
        x = (torch.randn(tokens, md, generator=gen, device=dev, dtype=torch.bfloat16) * 0.1).contiguous()
        logits = torch.randn(tokens, args.experts, generator=gen, device=dev)
        weights, ids = logits.softmax(-1).topk(topk, dim=-1)
        ids, weights = ids.to(torch.int32), weights.float().contiguous()

        disp = ep_dispatch(x, ids, weights, experts_per_rank=epr, capacity=capacity)
        y = moe_forward(disp.x, w1, w2, disp.topk_ids, disp.topk_weights, tile_m=tile_m)

        totals, layers, exchanges = [], [], []
        for _ in range(args.repeats):
            totals.append(
                bench_us(
                    lambda c=capacity, x=x, i=ids, w=weights, t=tile_m: ep_moe_forward(
                        x, w1, w2, i, w, experts_per_rank=epr, tile_m=t, capacity=c
                    ),
                    iters=args.iters,
                )
            )
            layers.append(
                bench_us(
                    lambda t=tile_m: moe_forward(disp.x, w1, w2, disp.topk_ids, disp.topk_weights, tile_m=t),
                    iters=args.iters,
                )
            )
            exchanges.append(
                bench_us(
                    lambda c=capacity, x=x, i=ids, w=weights: ep_dispatch(
                        x, i, w, experts_per_rank=epr, capacity=c
                    ),
                    iters=args.iters,
                )
                + bench_us(lambda: ep_combine(y, disp.exchange), iters=args.iters)
            )
        rows.append(
            (
                label,
                tokens,
                slowest(statistics.median(totals), dev),
                slowest(statistics.median(layers), dev),
                slowest(statistics.median(exchanges), dev),
            )
        )

    probe = slowest(
        statistics.median(
            [
                bench_us(
                    lambda: moe_forward(probe_x, probe_w1, probe_w2, probe_ids, probe_wt, tile_m=128),
                    iters=args.iters,
                )
                for _ in range(args.repeats)
            ]
        ),
        dev,
    )

    if rank == 0:
        print()
        print("=" * 76)
        print(f"EP strong scaling: fixed batch, EP={world} (experts={args.experts}, a rank holds {epr})")
        print("=" * 76)
        print(f"  drift probe (512 rows, 1 expert, identical at every EP): {probe:.1f} us")
        print()
        print(f"  {'case':<12} {'tok/rank':>9} {'total':>10} {'layer':>10} {'exchange':>10} {'exch %':>7}")
        for label, tokens, total_us, layer_us, exch_us in rows:
            print(
                f"  {label:<12} {tokens:9d} {total_us:10.1f} {layer_us:10.1f} "
                f"{exch_us:10.1f} {100 * exch_us / total_us:6.1f}%"
            )
        print()
        print("  total is the whole batch's wall time, so it compares across EP directly:")
        print("  lower means the extra cards bought something.")
        print()

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
