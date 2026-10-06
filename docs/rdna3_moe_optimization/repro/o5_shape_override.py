#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""o5: one shape was losing to eager torch, and the tile table was why.

    python docs/rdna3_moe_optimization/repro/o5_shape_override.py

o3 tuned the tile table at D=2048/I=768. A Mixtral-sized D=4096/I=14336 has a
twenty-times wider intermediate dimension, which changes both the N grid and how
much weight traffic each tile carries -- and it was the one shape in the sweep
running slower than eager torch.

The A/B works by emptying host._SHAPE_TILES, which is the whole of the o5
change. Everything else stays as it is.

Note the ordering of the two fixes, because it is the useful lesson: most of the
win came from asking for a *taller tile_m*, and only the remainder from
re-tuning the block shape underneath it.
"""

from __future__ import annotations

import torch

# common must be imported before anything under kernels/: it is what puts the
# repo root on sys.path. Keep these as from-imports so the sort order holds.
from common import bench_us, device_banner, header, layer_flops, make_layer_inputs, section, torch_moe_layer
from kernels.moe.rdna3_moe import host
from kernels.moe.rdna3_moe.forward import moe_forward

MODEL_DIM, INTER_DIM, EXPERTS, TOPK, TOKENS = 4096, 14336, 8, 2, 1024


class override:
    """Turn the o5 shape table on or off around a block of code."""

    def __init__(self, enabled):
        self.enabled = enabled

    def __enter__(self):
        self.saved = host._SHAPE_TILES
        if not self.enabled:
            host._SHAPE_TILES = {}
        host.compile_grouped_gemm.cache_clear()
        return self

    def __exit__(self, *exc):
        host._SHAPE_TILES = self.saved
        host.compile_grouped_gemm.cache_clear()
        return False


def geometry(tile_m):
    """What block shape the table hands this stage, as (BLOCK_M, N, K, accs)."""
    g1, bm1, bn1, bk1 = host.compile_moe_gemm1(
        model_dim=MODEL_DIM, inter_dim=INTER_DIM, experts=EXPERTS, topk=TOPK, tile_m=tile_m
    )
    g2, bm2, bn2, bk2 = host.compile_moe_gemm2(
        model_dim=MODEL_DIM, inter_dim=INTER_DIM, experts=EXPERTS, topk=TOPK, tile_m=tile_m
    )
    return f"{bm1}x{bn1}x{bk1} (accs {g1.acc_vectors})", f"{bm2}x{bn2}x{bk2} (accs {g2.acc_vectors})"


def main():
    header("o5  an exact-shape tile override", f"D{MODEL_DIM}/I{INTER_DIM}, {TOKENS} tokens, E={EXPERTS}, topk={TOPK}")
    device_banner()

    x, w1, w2, ids, wts = make_layer_inputs(
        tokens=TOKENS, model_dim=MODEL_DIM, inter_dim=INTER_DIM, experts=EXPERTS, topk=TOPK
    )
    flop = layer_flops(tokens=TOKENS, model_dim=MODEL_DIM, inter_dim=INTER_DIM, topk=TOPK)
    print(f"  weights: {(w1.numel() + w2.numel()) * 2 / 1e9:.2f} GB   useful FLOPs: {flop / 1e9:.1f} G")

    section("0. the baseline this shape was losing to")
    y_ref = torch_moe_layer(x, w1, w2, ids, wts)
    t_torch = bench_us(lambda: torch_moe_layer(x, w1, w2, ids, wts), warmup=3, iters=10)
    print(f"  eager torch, one matmul per expert: {t_torch / 1000:7.2f} ms  {flop / (t_torch * 1e-6) / 1e12:5.2f} TFLOP/s")
    print()
    print("  This reference is a plain loop, a little slower than the tuned baseline")
    print("  in scripts/bench_rdna3_moe.py, so the speedup column here reads high.")
    print("  Use the bench script for headline numbers; use this one for the A/B.")

    # ── 1. Symptom and the tile_m ladder ─────────────────────────────────
    section("1. with the generic table only: which tile_m does this shape want?")
    print("  tile_m is the caller's choice -- it sets both the routing padding and")
    print("  the block tile. The old benchmark row used 64, the prefill default.")
    print()
    print(f"  {'tile_m':>7} {'gate_up tile':>22} {'down tile':>22} {'layer':>9} {'TFLOP/s':>8} {'vs torch':>9}")
    with override(False):
        for tile_m in (32, 64, 128):
            g1, g2 = geometry(tile_m)
            t = bench_us(lambda tm=tile_m: moe_forward(x, w1, w2, ids, wts, tile_m=tm), warmup=3, iters=10)
            print(
                f"  {tile_m:>7} {g1:>22} {g2:>22} {t / 1000:8.2f}m {flop / (t * 1e-6) / 1e12:8.2f} "
                f"{t_torch / t:8.2f}x"
            )
    print()
    print("  Most of the o5 win is already here: just asking for tile_m=128 instead")
    print("  of 64 crosses the torch line, before any tile re-tuning.")

    # ── 2. The override ──────────────────────────────────────────────────
    section("2. re-tuning the block shape underneath tile_m=128")
    print(f"  {'_SHAPE_TILES':>13} {'gate_up tile':>22} {'down tile':>22} {'layer':>9} {'TFLOP/s':>8} {'vs torch':>9}")
    results = {}
    for enabled in (False, True):
        with override(enabled):
            g1, g2 = geometry(128)
            y = moe_forward(x, w1, w2, ids, wts, tile_m=128)
            t = bench_us(lambda: moe_forward(x, w1, w2, ids, wts, tile_m=128), warmup=3, iters=10)
            results[enabled] = t
            print(
                f"  {'on' if enabled else 'off':>13} {g1:>22} {g2:>22} {t / 1000:8.2f}m "
                f"{flop / (t * 1e-6) / 1e12:8.2f} {t_torch / t:8.2f}x"
            )
    diff = (y.float() - y_ref.float()).abs().max().item()
    print(f"\n  maxdiff vs the eager reference: {diff:.2e}  (bf16 accumulation, expected)")
    print(f"  override is worth {100 * (1 - results[True] / results[False]):.1f}% on top of the tile_m change")
    print("  (o6's grid order is on in both rows -- this isolates the tile choice only)")

    # ── 3. What was tried and rejected ───────────────────────────────────
    section("3. rejected: tile_m=256")
    rows_per_expert = TOKENS * TOPK / EXPERTS
    print(f"  This shape averages {rows_per_expert:.0f} routed rows per expert, so a 256-row tile")
    print("  looked like it would give each expert exactly one tile and stop the")
    print("  repeated weight loads. It was swept and rejected: gate_up improved but")
    print("  down got slower, and the padding cost rises steeply -- see roofline.py,")
    print("  which quantifies both sides of that trade.")
    print()
    print(f"  (tile_m=256 is not in host._TILES at all: {sorted(host.SUPPORTED_TILE_M)})")


if __name__ == "__main__":
    main()
