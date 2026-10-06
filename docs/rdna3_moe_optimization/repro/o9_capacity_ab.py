#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""The fixed-capacity dispatch, against the packed one it is meant to replace.

    torchrun --nproc_per_node=4 --master_port=29961 \
      docs/rdna3_moe_optimization/repro/o9_capacity_ab.py --experts-per-rank 2
    # and with the graph, which is the other half of the point:
    torchrun --nproc_per_node=4 ... o9_capacity_ab.py --experts-per-rank 2 --capture

o9_decode_overhead.py priced this before it existed, by handing the real dispatch
its split sizes as constants -- which a benchmark may do, because its routing
never changes. That said -8.6% at EP=2 and -12.9% at EP=4 on the whole decode
layer, and another point and a half from a capture on top.

This is the built version, so it pays what that one did not: every rank sends
``capacity`` rows to every peer whether the routing filled them or not. The
question is whether the padding eats the win, so the capacities swept here run
from the tightest that fits a balanced routing up to ``tokens * topk``, which no
routing can overflow at all.

Correctness is the first row of the report and not an afterthought: padded slots
carry a local expert id of -1 and have to vanish -- the routing kernel must count
them into no expert, the GEMMs must see no tile of them, and combine must land
them on a scratch row. If any of that leaks, the answer moves.
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

from kernels.moe.rdna3_moe.dispatch_kernel import build_dispatch_plan  # noqa: E402
from kernels.moe.rdna3_moe.ep import ep_moe_forward  # noqa: E402


def bench_us(fn, *, warmup=3, iters=30):
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
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--model-dim", type=int, default=4096)
    p.add_argument("--inter-dim", type=int, default=14336)
    p.add_argument("--experts-per-rank", type=int, default=2)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--tile-m", type=int, default=64)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--capture", action="store_true", help="also time the fixed path inside a CUDA graph")
    args = p.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    ws, rank = dist.get_world_size(), dist.get_rank()

    md, idim, epr, topk, tile = args.model_dim, args.inter_dim, args.experts_per_rank, args.topk, args.tile_m
    tokens = args.tokens
    n_experts = ws * epr
    rows = tokens * topk
    torch.manual_seed(41 + rank)
    w1 = (torch.randn(epr, 2 * idim, md, dtype=torch.bfloat16, device=dev) * 0.02).contiguous()
    w2 = (torch.randn(epr, md, idim, dtype=torch.bfloat16, device=dev) * 0.02).contiguous()
    x = (torch.randn(tokens, md, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    logits = torch.randn(tokens, n_experts, device=dev)
    tw, ti = logits.softmax(-1).topk(topk, dim=-1)
    ti, tw = ti.to(torch.int32), tw.float().contiguous()

    balanced = (rows + ws - 1) // ws
    caps = sorted({balanced, 2 * balanced, rows})
    variants = [("packed", None)] + [(f"cap={c}", c) for c in caps]

    def layer(cap, out=None):
        return ep_moe_forward(
            x, w1, w2, ti, tw, experts_per_rank=epr, tile_m=tile, capacity=cap, out=out, group=None
        )

    ref = layer(None)

    # Whether a capacity is even a candidate is a question about the routing, so
    # ask it directly: the plan counts what it had to drop, and reading that here
    # costs a sync a benchmark's setup can afford. A capacity that drops rows is
    # not timed -- its answer is wrong, so its speed is not an argument.
    # Reduced over the group, and not because the number is nicer that way: the
    # verdict decides which variants get timed, so ranks that reach different
    # ones issue different collectives and the run deadlocks rather than
    # disagreeing. One rank overflowing is the whole group overflowing.
    def dropped(cap):
        plan = build_dispatch_plan(ti, tw, world_size=ws, experts_per_rank=epr, capacity=cap)
        n = plan.counts[2 * ws + 1 : 2 * ws + 2].clone()
        dist.all_reduce(n, op=dist.ReduceOp.MAX)
        return int(n.item())

    over = {name: dropped(cap) for name, cap in variants[1:]}
    exact = {name: (torch.equal(ref, layer(cap)) if not over[name] else False) for name, cap in variants[1:]}
    variants = [(n, c) for n, c in variants if n == "packed" or not over[n]]

    static_out = torch.empty(tokens, md, dtype=x.dtype, device=dev)
    cap_graph = 2 * balanced
    graph, capture_err, captured_ok = None, None, False
    if args.capture:
        for _ in range(5):
            layer(cap_graph, out=static_out)
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                layer(cap_graph, out=static_out)
            g.replay()
            torch.cuda.synchronize()
            graph, captured_ok = g, torch.equal(ref, static_out)
        except Exception as exc:  # noqa: BLE001
            capture_err = f"{type(exc).__name__}: {exc}"

    names = [n for n, _ in variants] + (["captured"] if graph is not None else [])
    runs = {n: [] for n in names}
    for _ in range(args.repeats):
        for name, cap in variants:
            runs[name].append(bench_us(lambda c=cap: layer(c), iters=args.iters))
        if graph is not None:
            runs["captured"].append(bench_us(graph.replay, iters=args.iters))

    dist.barrier()
    if rank == 0:
        payload = md * 2
        print()
        print("=" * 76)
        print(f"fixed-capacity dispatch, world_size={ws}, tokens/rank={tokens}, D={md}, tile_m={tile}")
        print("=" * 76)
        print(f"  routed slots a rank sends: {rows}; balanced share per peer: {balanced}")
        skipped = [f"{n} dropped {d}" for n, d in over.items() if d]
        if skipped:
            print(f"  overflowed, so not timed: {', '.join(skipped)}")
        if capture_err:
            print(f"  capture FAILED -- {capture_err}")
        elif graph is not None:
            print(f"  captured graph exact: {captured_ok}")
        print()
        print(f"    {'':<12} {'exact':>7} {'sent/peer':>10} {'bytes out':>11} {'decode layer':>28}")
        base = statistics.median(runs["packed"])
        for name in names:
            v = runs[name]
            med = statistics.median(v)
            if name == "packed":
                sent, ex, cap = f"{balanced}~", "ref", balanced
            elif name == "captured":
                sent, ex, cap = f"{cap_graph} g", str(captured_ok), cap_graph
            else:
                cap = dict(variants)[name]
                sent, ex = str(cap), str(exact[name])
            nbytes = cap * payload
            print(
                f"    {name:<12} {ex:>7} {sent:>10} {nbytes / 1e6:8.2f} MB "
                f"{med:8.1f} [{min(v):6.1f},{max(v):6.1f}] {100 * (med / base - 1):+6.1f}%"
            )
        print()
        print(f"  medians of {args.repeats} alternations of {args.iters} iterations; brackets are the spread.")
        print("  'bytes out' is per peer, so the padding cost is the column's growth.")
        print()

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
