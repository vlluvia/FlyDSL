#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""What decode's EP overhead is actually made of, and what removing it is worth.

    torchrun --nproc_per_node=4 --master_port=29951 \
      docs/rdna3_moe_optimization/repro/o9_decode_overhead.py --experts-per-rank 2

The number this starts from: at EP=1, with **zero bytes** leaving the rank, the
exchange still costs 409 us of a 4713 us decode. Nothing is being moved, so that
is all fixed cost, and it is the same class of problem as o1, o2 and o4 -- the
launches and the host, not the work.

Two things stand between that and a graph capture, which is what would remove it:

  the counts all-to-all             one collective, ~50 us whatever it carries
  the readback that follows it      ``.tolist()``, a host sync, ~25 us

``ep.py``'s docstring proposes fixing the rows per rank pair to get rid of both.
That needs a capacity, padding rows, and a combine that can skip them -- real
machinery. o7 and o8 both went wrong by building first and pricing after, so
this prices first, and it can do that honestly because a benchmark's routing does
not change between iterations: the counts can be computed once and passed in as
constants. The exchange is then numerically identical to the real one and simply
does not pay for learning its own split sizes.

So the three variants here are:

    today       ep_dispatch as shipped
    fixed       the same, splits handed in as host constants
    captured    ``fixed``, whole decode layer inside a CUDA graph

``today`` -> ``fixed`` is what a capacity buys. ``fixed`` -> ``captured`` is what
the capacity *unlocks*, which is the larger half of the prize and the reason to
want the capacity at all.
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
from kernels.moe.rdna3_moe.ep import Dispatch, Exchange, ep_combine, ep_dispatch  # noqa: E402
from kernels.moe.rdna3_moe.forward import moe_forward  # noqa: E402


def fixed_dispatch(x, topk_ids, topk_weights, *, experts_per_rank, send_counts, recv_counts):
    """``ep_dispatch`` with the split sizes already known.

    Same three collectives minus the counts exchange, and no readback -- so no
    host sync anywhere in it, which is the property a capture needs. Correct only
    because the caller guarantees these counts belong to this routing; a real
    implementation earns that guarantee with a capacity per rank pair and padding
    rows that the routing kernel drops.
    """
    ws = dist.get_world_size()
    tokens, model_dim = x.shape
    epr = int(experts_per_rank)
    dev = x.device
    rows_recv = int(sum(recv_counts))

    plan = build_dispatch_plan(topk_ids, topk_weights, world_size=ws, experts_per_rank=epr)
    x_send = x.index_select(0, plan.src_row).contiguous()

    x_recv = torch.empty(rows_recv, model_dim, dtype=x.dtype, device=dev)
    meta_recv = torch.empty(rows_recv, 2, dtype=torch.int32, device=dev)
    dist.all_to_all_single(x_recv, x_send, recv_counts, send_counts)
    dist.all_to_all_single(meta_recv, plan.meta, recv_counts, send_counts)

    return Dispatch(
        x=x_recv,
        topk_ids=meta_recv[:, 0].contiguous().view(rows_recv, 1),
        topk_weights=meta_recv[:, 1].contiguous().view(torch.float32).view(rows_recv, 1),
        exchange=Exchange(
            send_counts=send_counts,
            recv_counts=recv_counts,
            send_order=plan.send_order,
            tokens=int(tokens),
            topk=int(topk_ids.shape[1]),
            model_dim=int(model_dim),
        ),
    )


def run_layer(disp, w1, w2, tile_m, x, out=None):
    if disp.x.shape[0]:
        y = moe_forward(disp.x, w1, w2, disp.topk_ids, disp.topk_weights, tile_m=tile_m)
    else:
        y = torch.empty(0, disp.exchange.model_dim, dtype=x.dtype, device=x.device)
    return ep_combine(y, disp.exchange, out=out)


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
    p.add_argument(
        "--capture",
        action="store_true",
        help="also try a CUDA graph around the fixed path. Off by default: a "
        "capture that RCCL will not accept hangs rather than raising, so run it "
        "under `timeout` the first time on a new stack.",
    )
    args = p.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    ws, rank = dist.get_world_size(), dist.get_rank()

    md, idim, epr, topk, tile = args.model_dim, args.inter_dim, args.experts_per_rank, args.topk, args.tile_m
    tokens = args.tokens
    n_experts = ws * epr
    torch.manual_seed(31 + rank)
    w1 = (torch.randn(epr, 2 * idim, md, dtype=torch.bfloat16, device=dev) * 0.02).contiguous()
    w2 = (torch.randn(epr, md, idim, dtype=torch.bfloat16, device=dev) * 0.02).contiguous()
    x = (torch.randn(tokens, md, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    logits = torch.randn(tokens, n_experts, device=dev)
    tw, ti = logits.softmax(-1).topk(topk, dim=-1)
    ti, tw = ti.to(torch.int32), tw.float().contiguous()

    # The counts this routing produces, learned once the expensive way so the
    # fixed variants can be handed them.
    probe = ep_dispatch(x, ti, tw, experts_per_rank=epr)
    sc, rc = probe.exchange.send_counts, probe.exchange.recv_counts

    ref = run_layer(ep_dispatch(x, ti, tw, experts_per_rank=epr), w1, w2, tile, x)
    fixed_ok = torch.equal(
        ref, run_layer(fixed_dispatch(x, ti, tw, experts_per_rank=epr, send_counts=sc, recv_counts=rc), w1, w2, tile, x)
    )

    # A capture needs its output buffer to be the same one every replay, so the
    # layer writes into a static tensor rather than returning a fresh one.
    static_out = torch.empty(tokens, md, dtype=x.dtype, device=dev)

    def fixed_layer():
        disp = fixed_dispatch(x, ti, tw, experts_per_rank=epr, send_counts=sc, recv_counts=rc)
        return run_layer(disp, w1, w2, tile, x, out=static_out)

    graph, capture_err, captured_ok = None, None, False
    if args.capture:
        # Warm up eagerly on the default stream first. RCCL allocates its
        # channels and buffers on first use, and a capture that is the first
        # thing a communicator sees has nothing to record.
        for _ in range(5):
            fixed_layer()
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fixed_layer()
            g.replay()
            torch.cuda.synchronize()
            graph, captured_ok = g, torch.equal(ref, static_out)
        except Exception as exc:  # noqa: BLE001 -- reporting why is the point
            capture_err = f"{type(exc).__name__}: {exc}"

    variants = [("today", "dispatch+layer"), ("fixed", "dispatch+layer")]
    if graph is not None:
        variants.append(("captured", "layer"))

    runs = {(s, n): [] for s in ("dispatch", "layer") for n, _ in variants}
    for _ in range(args.repeats):
        runs[("dispatch", "today")].append(
            bench_us(lambda: ep_dispatch(x, ti, tw, experts_per_rank=epr), iters=args.iters)
        )
        runs[("layer", "today")].append(
            bench_us(lambda: run_layer(ep_dispatch(x, ti, tw, experts_per_rank=epr), w1, w2, tile, x), iters=args.iters)
        )
        runs[("dispatch", "fixed")].append(
            bench_us(
                lambda: fixed_dispatch(x, ti, tw, experts_per_rank=epr, send_counts=sc, recv_counts=rc),
                iters=args.iters,
            )
        )
        runs[("layer", "fixed")].append(bench_us(fixed_layer, iters=args.iters))
        if graph is not None:
            runs[("dispatch", "captured")].append(float("nan"))
            runs[("layer", "captured")].append(bench_us(graph.replay, iters=args.iters))

    dist.barrier()
    if rank == 0:
        print()
        print("=" * 76)
        print(f"decode EP overhead, world_size={ws}, tokens/rank={tokens}, D={md}, I={idim}, tile_m={tile}")
        print("=" * 76)
        print()
        print(f"  fixed splits give the same answer as today: {fixed_ok}")
        if capture_err:
            print(f"  capture FAILED -- {capture_err}")
        elif graph is not None:
            print(f"  captured graph gives the same answer as today: {captured_ok}")
        else:
            print("  capture not attempted (pass --capture)")
        print()
        print(f"    {'':<10} {'dispatch':>26} {'whole decode layer':>28}")
        base_d = statistics.median(runs[("dispatch", "today")])
        base_l = statistics.median(runs[("layer", "today")])
        for name, _ in variants:
            dv = [v for v in runs[("dispatch", name)] if v == v]
            lv = runs[("layer", name)]
            l_med = statistics.median(lv)
            if dv:
                d_med = statistics.median(dv)
                dcell = f"{d_med:8.1f} [{min(dv):6.1f},{max(dv):6.1f}] {100 * (d_med / base_d - 1):+6.1f}%"
            else:
                dcell = f"{'(in graph)':>26}"
            print(
                f"    {name:<10} {dcell} "
                f"{l_med:8.1f} [{min(lv):6.1f},{max(lv):6.1f}] {100 * (l_med / base_l - 1):+6.1f}%"
            )
        print()
        print(f"  medians of {args.repeats} alternations of {args.iters} iterations; brackets are the spread.")
        print("  The layer column is the one that decides -- dispatch is a fifth of it.")
        print()

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
