#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""o6: the same tiles, dispatched in a different order.

    python docs/rdna3_moe_optimization/repro/o6_grid_order.py

Nothing about the arithmetic changes in this phase. The grid was
``(N_tiles, M_tiles)`` with the N index innermost; o6 makes it ``(M_tiles,
N_tiles)`` for one class of shape. Which order is faster is purely a question of
what survives in cache.

A workgroup reads one A tile (BLOCK_M x k_dim) and one B slab (BLOCK_N x k_dim).
Whichever index moves innermost is the one whose operand gets reused out of
cache while the other is re-read from memory:

    N innermost   one A tile is reused across the whole N sweep;
                  each expert's B slab is re-read once per M tile it owns
    M innermost   each expert's B slab is reused across its own M tiles;
                  the A tiles have to stay live across the M sweep instead

So the answer depends on which operand is bigger and whether the weights fit in
cache at all. This script measures all four combinations.
"""

from __future__ import annotations

import torch

# common must be imported before anything under kernels/: it is what puts the
# repo root on sys.path. Keep these as from-imports so the sort order holds.
from common import bench_us, device_banner, header, section
from kernels.moe.rdna3_moe import host
from kernels.moe.rdna3_moe.grouped_gemm import create_grouped_gemm_module
from kernels.moe.rdna3_moe.routing_kernel import build_routing_device

EXPERTS, TOPK, TOKENS, TILE_M = 8, 2, 1024, 128
WIDE = (4096, 14336)  # weights far beyond any cache
BASE = (2048, 768)  # whole weight set fits in Infinity Cache


def time_stage(stage, model_dim, inter_dim, cfg, m_major):
    """Build one stage with an explicit tile and grid order, and time it."""
    reg_m, reg_n, reg_k, waves_m, waves_n = cfg
    k_dim, n_out = (model_dim, inter_dim) if stage in ("linear", "gate_up") else (inter_dim, model_dim)
    rows = TOKENS * TOPK
    launch, bm, bn, bk = create_grouped_gemm_module(
        k_dim=k_dim, n_out=n_out, experts=EXPERTS, stage=stage, topk=TOPK,
        doweight=(stage == "down"),
        reg_m=reg_m, reg_n=reg_n, reg_k=reg_k, waves_m=waves_m, waves_n=waves_n,
        m_major=m_major,
    )
    dev = "cuda"
    torch.manual_seed(0)
    a_rows = rows if stage == "down" else TOKENS
    a = (torch.randn(a_rows, k_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    w_rows = (2 if stage == "gate_up" else 1) * n_out
    w = (torch.randn(EXPERTS, w_rows, k_dim, dtype=torch.bfloat16, device=dev) * 0.05).contiguous()
    ids = torch.randint(0, EXPERTS, (TOKENS, TOPK), dtype=torch.int32, device=dev)
    wts = torch.rand(TOKENS, TOPK, dtype=torch.float32, device=dev)
    r = build_routing_device(ids, experts=EXPERTS, tile_m=bm, exact=True, reuse=False)
    c = torch.zeros(TOKENS, TOPK, n_out, dtype=torch.bfloat16, device=dev)
    empty = torch.empty(0, dtype=torch.float32, device=dev)
    stream = torch.cuda.current_stream()

    def run():
        launch(c, a, w, r.sorted_ids, r.expert_ids, wts if stage == "down" else empty, TOKENS, r.num_blocks, stream)

    run()
    torch.cuda.synchronize()
    us = bench_us(run, iters=30)
    tf = 2.0 * rows * w_rows * k_dim / (us * 1e-6) / 1e12
    return us, tf, c.clone(), (bm, bn, bk)


def operand_sizes(stage, model_dim, inter_dim, bm, bn):
    k_dim = model_dim if stage in ("linear", "gate_up") else inter_dim
    return bm * k_dim * 2, bn * k_dim * 2


def main():
    header("o6  workgroup dispatch order", "same tiles, same arithmetic, different cache behaviour")
    device_banner()

    for model_dim, inter_dim in (WIDE, BASE):
        w1_b = EXPERTS * 2 * inter_dim * model_dim * 2
        w2_b = EXPERTS * model_dim * inter_dim * 2
        section(f"D{model_dim}/I{inter_dim}   total weights {(w1_b + w2_b) / 1e9:.2f} GB")
        fits = (w1_b + w2_b) < 96e6
        print(f"  {'fits' if fits else 'does NOT fit'} in a 96 MB Infinity Cache"
              + ("  -> B re-reads are hits whatever the order" if fits else "  -> a B re-read is a real trip to memory"))
        print()
        print("  The tile is held fixed across all four rows so the only variable is")
        print("  the order; it is not each shape's best tile.")
        print()
        print(f"  {'stage':>8} {'block':>14} {'A tile':>9} {'B slab':>9} "
              f"{'N innermost':>12} {'M innermost':>12} {'delta':>8} {'bitwise same':>13}")
        for stage, cfg in (("gate_up", (4, 2, 2, 2, 2)), ("down", (4, 4, 2, 2, 2))):
            us_n, tf_n, out_n, (bm, bn, bk) = time_stage(stage, model_dim, inter_dim, cfg, False)
            us_m, tf_m, out_m, _ = time_stage(stage, model_dim, inter_dim, cfg, True)
            a_b, b_b = operand_sizes(stage, model_dim, inter_dim, bm, bn)
            same = torch.equal(out_n, out_m)
            print(
                f"  {stage:>8} {f'{bm}x{bn}x{bk}':>14} {a_b / 1e6:8.1f}M {b_b / 1e6:8.1f}M "
                f"{tf_n:11.2f}T {tf_m:11.2f}T {100 * (tf_m / tf_n - 1):+7.1f}% {str(same):>13}"
            )

    section("reading the table")
    print("  gate_up on the wide shape is the one case that wants M innermost: its")
    print("  weights miss cache, and its A tile is the small operand so keeping the")
    print("  M sweep live is cheap.")
    print()
    print("  down never wants it. Its k_dim is the intermediate size, so its A tile")
    print("  is several times larger than gate_up's while its N sweep is short --")
    print("  N innermost was already giving it good reuse.")
    print()
    print("  The base shape never wants it either: its whole weight set sits in")
    print("  cache, so the re-reads the order would save were never costing anything.")
    print()
    print(f"  Which is why this is a per-shape table, not a default:")
    for key in sorted(host._SHAPE_M_MAJOR):
        print(f"    host._SHAPE_M_MAJOR: {key}")

    section("the padding that is left")
    print("  Run roofline.py: at tile_m=128 this shape spends 24% of its MACs on")
    print("  rows the routing padded in, which looks like the next structural lever.")
    print("  o7 went after it and found it is not: those MACs are only about 11% of")
    print("  the time, and the obvious way at them -- a main tile plus a short tail")
    print("  tile per expert -- measured 9.5% slower, because a second launch has to")
    print("  stream all the weights again. See o7_padding_measure.py.")


if __name__ == "__main__":
    main()
