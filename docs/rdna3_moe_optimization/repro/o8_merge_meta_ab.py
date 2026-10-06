#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Merging the metadata into the payload's collective: measured, and not taken.

    torchrun --nproc_per_node=4 --master_port=29931 \
      docs/rdna3_moe_optimization/repro/o8_merge_meta_ab.py --experts-per-rank 2

``ep.py`` sends four collectives a layer, two of them in dispatch: the
activations, and eight bytes a row of metadata beside them. An all-to-all on
these cards costs 45-60 us before it moves anything, so the small one costs
nearly what the big one does, and folding it into the big one's rows looks free.
o8_ep_breakdown.py priced that at -20% of dispatch at prefill by timing the
steps apart and adding them up.

It is not -20%. o7 is the standing warning against believing a sum of isolated
steps, and this is the same failure: the merge costs a copy to stage the send
rows and another to make the received activations contiguous again -- the GEMM
reads A with a leading dimension of exactly k_dim, so a row still carrying its
metadata cannot be fed to it -- and those two copies cost about what the
collective saved.

The merged path lives here rather than in ``ep.py`` because ``ep.py`` does not
have it. This is the evidence for why.

What it does show is a property of ``all_to_all_single`` worth knowing
separately, which o8_alltoall_size.py isolates: the cost steps around 4 MiB a
destination, and a dispatch payload lands on the round size by construction.
The ``merged+256B`` row is not carrying more metadata -- it is stepping the
block over that threshold, and at EP=4 prefill, where the unpadded block is
exactly 4 MiB, that is worth 7.8% where the merge itself is worth -3%.
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


def merged_dispatch(x, topk_ids, topk_weights, *, experts_per_rank, extra_words=0):
    """``ep_dispatch``, with the metadata riding in the payload's rows.

    ``extra_words`` pads the row beyond the minimum. The minimum is not two
    int32: the block a collective moves is ``rows x row_bytes`` for a row count
    the routing picks, and a block that is not a multiple of 16 bytes runs at a
    tenth of the bandwidth, so the row is rounded to 16 bytes first. 8200 bytes
    -- a D=4096 bf16 row plus two int32 -- is the trap, aligned for an even row
    count and not for an odd one.
    """
    ws = dist.get_world_size()
    tokens, model_dim = x.shape
    epr = int(experts_per_rank)
    dev = x.device
    half = model_dim * x.element_size() // 4
    words = (half + 2 + 3) // 4 * 4 + extra_words

    plan = build_dispatch_plan(topk_ids, topk_weights, world_size=ws, experts_per_rank=epr)
    dist.all_to_all_single(plan.counts[ws : 2 * ws], plan.counts[:ws])

    send_buf = torch.empty(plan.src_row.numel(), words, dtype=torch.int32, device=dev)
    torch.index_select(x.view(torch.int32), 0, plan.src_row, out=send_buf[:, :half])
    send_buf[:, half : half + 2] = plan.meta

    read_back = plan.counts.tolist()
    send_counts, recv_counts = read_back[:ws], read_back[ws : 2 * ws]
    rows_recv = int(sum(recv_counts))

    recv_buf = torch.empty(rows_recv, words, dtype=torch.int32, device=dev)
    dist.all_to_all_single(recv_buf, send_buf, recv_counts, send_counts)
    x_recv = recv_buf[:, :half].contiguous().view(x.dtype)
    meta = recv_buf[:, half : half + 2]

    return Dispatch(
        x=x_recv,
        topk_ids=meta[:, 0].contiguous().view(rows_recv, 1),
        topk_weights=meta[:, 1].contiguous().view(torch.float32).view(rows_recv, 1),
        exchange=Exchange(
            send_counts=send_counts,
            recv_counts=recv_counts,
            send_order=plan.send_order,
            tokens=int(tokens),
            topk=int(topk_ids.shape[1]),
            model_dim=int(model_dim),
        ),
    )


def layer(disp, w1, w2, tile_m, x):
    if disp.x.shape[0]:
        y = moe_forward(disp.x, w1, w2, disp.topk_ids, disp.topk_weights, tile_m=tile_m)
    else:
        y = torch.empty(0, disp.exchange.model_dim, dtype=x.dtype, device=x.device)
    return ep_combine(y, disp.exchange)


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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dim", type=int, default=4096)
    p.add_argument("--inter-dim", type=int, default=14336)
    p.add_argument("--experts-per-rank", type=int, default=2)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--repeats", type=int, default=5, help="alternations of the variants")
    args = p.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    ws, rank = dist.get_world_size(), dist.get_rank()

    md, idim, epr, topk = args.model_dim, args.inter_dim, args.experts_per_rank, args.topk
    n_experts = ws * epr
    torch.manual_seed(20 + rank)
    w1 = (torch.randn(epr, 2 * idim, md, dtype=torch.bfloat16, device=dev) * 0.02).contiguous()
    w2 = (torch.randn(epr, md, idim, dtype=torch.bfloat16, device=dev) * 0.02).contiguous()

    # ("split", -) is what ep.py ships; the rest are merged, with that many extra
    # int32 of row padding beyond the 16-byte minimum.
    variants = [("split", None), ("merged", 0), ("merged+256B", 64), ("merged+512B", 128)]
    out = {}
    for label, tokens, tile in [("decode", 32, 64), ("prefill", 1024, 128)]:
        x = (torch.randn(tokens, md, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
        logits = torch.randn(tokens, n_experts, device=dev)
        tw, ti = logits.softmax(-1).topk(topk, dim=-1)
        ti, tw = ti.to(torch.int32), tw.float().contiguous()

        def disp_of(extra):
            if extra is None:
                return ep_dispatch(x, ti, tw, experts_per_rank=epr)
            return merged_dispatch(x, ti, tw, experts_per_rank=epr, extra_words=extra)

        # The merged path has to be bit-identical, or its timing compares nothing.
        ref = layer(disp_of(None), w1, w2, tile, x)
        same = all(torch.equal(ref, layer(disp_of(e), w1, w2, tile, x)) for _, e in variants[1:])

        runs = {(s, n): [] for s in ("dispatch", "layer") for n, _ in variants}
        for _ in range(args.repeats):
            for name, extra in variants:
                runs[("dispatch", name)].append(bench_us(lambda e=extra: disp_of(e), iters=args.iters))
                runs[("layer", name)].append(
                    bench_us(lambda e=extra: layer(disp_of(e), w1, w2, tile, x), iters=args.iters)
                )
        out[label] = (
            same,
            tokens * topk / ws * md * 2,
            {k: (statistics.median(v), min(v), max(v)) for k, v in runs.items()},
        )

    dist.barrier()
    if rank == 0:
        print()
        print("=" * 78)
        print(f"merge metadata into the payload collective, world_size={ws}, D={md}, I={idim}")
        print("=" * 78)
        for label, (same, block, r) in out.items():
            print()
            print(f"  {label}   (every merged variant bit-identical to split: {same})")
            print(f"  unpadded block per destination: {block / 1e6:.3f} MB")
            print(f"    {'':<14} {'dispatch':>26} {'layer':>28}")
            for name, _ in variants:
                d_med, d_lo, d_hi = r[("dispatch", name)]
                l_med, l_lo, l_hi = r[("layer", name)]
                dd = 100 * (d_med / r[("dispatch", "split")][0] - 1)
                ll = 100 * (l_med / r[("layer", "split")][0] - 1)
                print(
                    f"    {name:<14} {d_med:8.1f} [{d_lo:6.1f},{d_hi:6.1f}] {dd:+6.1f}% "
                    f"{l_med:9.1f} [{l_lo:7.1f},{l_hi:7.1f}] {ll:+6.1f}%"
                )
        print()
        print(f"  medians of {args.repeats} alternations of {args.iters} iterations; brackets are the spread.")
        print("  A change smaller than the spread is not a change -- which is most of")
        print("  this table, and the reason ep.py still sends two collectives.")
        print()

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
