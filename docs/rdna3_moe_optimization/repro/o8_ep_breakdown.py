#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Where an EP dispatch's microseconds actually go, piece by piece.

    torchrun --nproc_per_node=4 --master_port=29901 \
      docs/rdna3_moe_optimization/repro/o8_ep_breakdown.py \
      --tokens 32 --model-dim 4096 --inter-dim 14336 --experts-per-rank 2

o4 cut dispatch's *planning* from fifteen torch launches to one kernel. What is
left is the exchange itself, and `ep.py` names two levers for it without having
measured either in isolation:

  fold the metadata into the payload's collective   saves one all-to-all,
                                                    costs a contiguous copy
  fix the rows per rank pair                        saves the counts exchange
                                                    and the host readback,
                                                    costs padding bandwidth

Both are priced in that docstring from the "an all-to-all costs 45-60 us before
it moves anything" rule of thumb. This times each step for real, at whatever
world size it is launched with, so the two levers can be sized before either is
built rather than after.

Needs a container with a big enough /dev/shm for the world size: RCCL takes
20.25 MB per communicator, so the default 64 MB caps out at three ranks. Run
with --shm-size=2g.
"""

from __future__ import annotations

import argparse
import os
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
from kernels.moe.rdna3_moe.ep import ep_dispatch  # noqa: E402


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
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--model-dim", type=int, default=4096)
    p.add_argument("--inter-dim", type=int, default=14336)
    p.add_argument("--experts-per-rank", type=int, default=2)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--iters", type=int, default=30)
    args = p.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    ws, rank = dist.get_world_size(), dist.get_rank()

    tokens, model_dim, epr, topk = args.tokens, args.model_dim, args.experts_per_rank, args.topk
    n_experts = ws * epr
    torch.manual_seed(1234 + rank)
    x = (torch.randn(tokens, model_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    logits = torch.randn(tokens, n_experts, device=dev)
    topk_weights, topk_ids = logits.softmax(-1).topk(topk, dim=-1)
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = topk_weights.float().contiguous()

    # One real dispatch, to size every buffer the steps below reuse.
    disp = ep_dispatch(x, topk_ids, topk_weights, experts_per_rank=epr)
    ex = disp.exchange
    rows_send, rows_recv = sum(ex.send_counts), sum(ex.recv_counts)
    plan = build_dispatch_plan(topk_ids, topk_weights, world_size=ws, experts_per_rank=epr)

    x_send = x.index_select(0, plan.src_row).contiguous()
    x_recv = torch.empty(rows_recv, model_dim, dtype=x.dtype, device=dev)
    meta_send = plan.meta
    meta_recv = torch.empty(rows_recv, 2, dtype=torch.int32, device=dev)
    sc, rc = ex.send_counts, ex.recv_counts

    # The merged buffer lever 1 would use: one row carries the activation and
    # the two metadata words, so a single collective moves both. int32 rather
    # than bf16 so the metadata needs no bit-punning to write.
    #
    # How many int32 the metadata region gets is not free to choose. Two is all
    # it needs, but that makes the row 8200 bytes where the payload's own row is
    # 8192, and a collective's copy is only as fast as its rows are aligned. So
    # sweep the padding rather than assume two is fine.
    half = model_dim // 2
    wide = {}
    for pad in PADS:
        wide[pad] = (
            torch.empty(rows_send, half + pad, dtype=torch.int32, device=dev),
            torch.empty(rows_recv, half + pad, dtype=torch.int32, device=dev),
        )

    def fill_wide(pad=2):
        ws_, _ = wide[pad]
        ws_[:, :half] = x.index_select(0, plan.src_row).view(torch.int32)
        ws_[:, half : half + 2] = meta_send

    def unpack_wide(pad=2):
        _, wr = wide[pad]
        return wr[:, :half].contiguous().view(torch.bfloat16), wr[:, half : half + 2].contiguous()

    steps = [
        ("plan kernel (o4)", lambda: build_dispatch_plan(topk_ids, topk_weights, world_size=ws, experts_per_rank=epr)),
        ("counts all-to-all", lambda: dist.all_to_all_single(plan.counts[ws : 2 * ws], plan.counts[:ws])),
        ("payload gather", lambda: x.index_select(0, plan.src_row).contiguous()),
        ("counts readback (.tolist)", lambda: plan.counts.tolist()),
        ("payload all-to-all", lambda: dist.all_to_all_single(x_recv, x_send, rc, sc)),
        ("meta all-to-all", lambda: dist.all_to_all_single(meta_recv, meta_send, rc, sc)),
        ("--- lever 1: merge meta into payload ---", None),
        ("  fill merged send buffer", fill_wide),
        ("  unpack to contiguous", unpack_wide),
    ]
    for pad in (2, 8, 32, 64):
        row_bytes = (half + pad) * 4
        steps.append(
            (
                f"  merged all-to-all, row {row_bytes}B",
                lambda p=pad: dist.all_to_all_single(wide[p][1], wide[p][0], rc, sc),
            )
        )
    steps += [
        ("--- whole thing ---", None),
        ("ep_dispatch total", lambda: ep_dispatch(x, topk_ids, topk_weights, experts_per_rank=epr)),
    ]

    timings = {}
    for name, fn in steps:
        if fn is not None:
            timings[name] = bench_us(fn, iters=args.iters)
    dist.barrier()

    if rank == 0:
        print()
        print("=" * 74)
        print(f"EP dispatch breakdown   world_size={ws}  tokens/rank={tokens}  D={model_dim}  epr={epr}")
        print("=" * 74)
        print(f"rows sent {rows_send}, received {rows_recv}; payload {rows_send * model_dim * 2 / 1e6:.2f} MB out")
        print()
        for name, fn in steps:
            if fn is None:
                print(f"  {name}")
                continue
            print(f"  {name:<30} {timings[name]:8.1f} us")
        print()

        today = (
            timings["plan kernel (o4)"]
            + timings["counts all-to-all"]
            + timings["payload gather"]
            + timings["counts readback (.tolist)"]
            + timings["payload all-to-all"]
            + timings["meta all-to-all"]
        )
        best_pad = min((2, 8, 32, 64), key=lambda p: timings[f"  merged all-to-all, row {(half + p) * 4}B"])
        merged = timings[f"  merged all-to-all, row {(half + best_pad) * 4}B"]
        lever1 = (
            timings["plan kernel (o4)"]
            + timings["counts all-to-all"]
            + timings["  fill merged send buffer"]
            + timings["counts readback (.tolist)"]
            + merged
            + timings["  unpack to contiguous"]
        )
        lever12 = lever1 - timings["counts all-to-all"] - timings["counts readback (.tolist)"]
        print(f"  {'sum of steps, as built today':<30} {today:8.1f} us")
        print(f"  {'measured ep_dispatch':<30} {timings['ep_dispatch total']:8.1f} us  (rest is host glue)")
        print()
        print(f"  best merged row is {(half + best_pad) * 4} B ({best_pad} int32 of metadata region)")
        print(f"  {'lever 1 (merge meta)':<30} {lever1:8.1f} us  {100 * (lever1 / today - 1):+6.1f}%")
        print(f"  {'lever 1 + 2 (fixed splits too)':<30} {lever12:8.1f} us  {100 * (lever12 / today - 1):+6.1f}%")
        print()
        print("  Lever 2 only removes the counts exchange and the readback; it cannot")
        print("  touch the payload, and it costs padding bandwidth this table does not")
        print("  charge it for. Lever 1's saving is the meta all-to-all minus the fill")
        print("  and unpack copies, so it shrinks as the payload grows.")
        print()

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
