#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Measure the ceiling before claiming a speedup, then build a model that predicts.

    python docs/rdna3_moe_optimization/repro/roofline.py

This is the method that produced o6, and it is the most portable thing in this
folder: none of it is specific to MoE. The order matters.

  1. Measure what the machine can actually do -- dense peak and memory
     bandwidth. Without this, "62 TFLOP/s" is a number with no meaning.
  2. Count the work you are actually doing, including the work you are wasting.
     Here that is the padding rows, and they turn out to be 24% of the MACs.
  3. Write down a model that predicts runtime from those two numbers, and check
     it against measurements. A model that predicts is a model you can optimise
     against; a model that does not means you have the wrong bottleneck.

The payoff in o6 was step 2 producing a contradiction: correcting for padding
put the kernel's real throughput *above* the dense library, which meant the
library was not the ceiling and there was more room than it looked.

One warning about step 3, which this script's own model gets wrong. It prices
work at a single throughput, so a wasted MAC costs the same as a useful one. That
is what makes the 24% padding figure below look like 24% of the runtime, and o7
measured it at about a tenth of that -- a padded row costs roughly a third of an
average row, because most of a launch's time does not scale with rows at all. The
model is still worth having; it predicted the tile_m crossover to two decimal
places. But a share of the *work* is not a share of the *time*, and the gap
between them is exactly where o7's mixed-tile scheme died. Read
o7_padding_predict.py and o7_padding_measure.py before acting on section 5.
"""

from __future__ import annotations

import torch

from common import bench_us, device_banner, header, layer_flops, make_layer_inputs, section
from kernels.moe.rdna3_moe.forward import moe_forward

MODEL_DIM, INTER_DIM, EXPERTS, TOPK, TOKENS = 4096, 14336, 8, 2, 1024


def measure_ceilings():
    section("1. what this machine can do at all")
    peak = 0.0
    for n in (4096, 8192):
        a = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
        t = bench_us(lambda: a @ b, warmup=3, iters=10) * 1e-6
        tf = 2 * n**3 / t / 1e12
        peak = max(peak, tf)
        print(f"  dense bf16 {n}^3 (rocBLAS)      {t * 1e3:8.2f} ms  {tf:7.2f} TFLOP/s")
        del a, b

    buf = torch.empty(int(1.5e9) // 2, dtype=torch.bfloat16, device="cuda")
    t = bench_us(lambda: buf.sum(), warmup=3, iters=10) * 1e-6
    bw = buf.numel() * 2 / t
    print(f"  streaming read {buf.numel() * 2 / 1e9:.2f} GB       {t * 1e3:8.2f} ms  {bw / 1e9:7.1f} GB/s")
    del buf
    torch.cuda.empty_cache()

    section("2. the same library on this problem's actual shapes")
    rows_per_expert = TOKENS * TOPK // EXPERTS
    for name, m, k, n in (
        ("gemm1 per expert", rows_per_expert, MODEL_DIM, 2 * INTER_DIM),
        ("gemm2 per expert", rows_per_expert, INTER_DIM, MODEL_DIM),
    ):
        a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(k, n, dtype=torch.bfloat16, device="cuda")
        t = bench_us(lambda: a @ b, warmup=5, iters=20) * 1e-6
        print(f"  {name} {m}x{n}x{k}   {t * 1e3:7.3f} ms  {2 * m * n * k / t / 1e12:7.2f} TFLOP/s"
              f"   (x{EXPERTS} = {t * EXPERTS * 1e3:.2f} ms)")
        del a, b
    torch.cuda.empty_cache()
    print()
    print("  A skinny M like this is where the library gives up throughput, and it")
    print("  is exactly the shape MoE creates: the batch is split across experts.")
    return peak, bw


def padding_stats():
    section("3. the work that is not in the answer")
    print("  Routing pads each expert's rows up to a multiple of tile_m, and every")
    print("  padded row costs a full row of MACs. With E=8 and 2048 routed rows, an")
    print("  expert averages 256 rows -- sitting exactly on the 128 boundary, which")
    print("  is the worst case: about half the experts spill into a nearly empty")
    print("  third tile.")
    print()
    rows = TOKENS * TOPK
    gen = torch.Generator(device="cuda").manual_seed(0)
    out = {}
    print(f"  {'tile_m':>7} {'avg tiles':>10} {'padded rows':>12} {'wasted MACs':>12}")
    for tile_m in (32, 64, 128, 256):
        tot = 0.0
        trials = 200
        for _ in range(trials):
            logits = torch.randn(TOKENS, EXPERTS, generator=gen, device="cuda")
            sel = logits.topk(TOPK, dim=-1).indices.reshape(-1)
            counts = torch.bincount(sel, minlength=EXPERTS)
            blocks = torch.clamp((counts + tile_m - 1) // tile_m, min=1)
            tot += float(blocks.sum())
        nb = tot / trials
        padded = nb * tile_m
        out[tile_m] = nb
        print(f"  {tile_m:>7} {nb:10.1f} {padded:12.0f} {100 * (padded / rows - 1):11.1f}%")
    return out


def model_vs_measured(peak_tf, bw, blocks):
    section("4. a model, and whether it predicts")
    print("  Every M tile re-reads its expert's whole B slab, so weight traffic is")
    print("  proportional to the tile count, while compute is proportional to the")
    print("  padded row count. The two pull in opposite directions:")
    print()
    print("    time ~ max( B_bytes / bandwidth , padded_FLOPs / throughput )")
    print()
    useful = layer_flops(tokens=TOKENS, model_dim=MODEL_DIM, inter_dim=INTER_DIM, topk=TOPK)
    rows = TOKENS * TOPK

    # The kernel's real MAC throughput, backed out of a measurement rather than
    # assumed: run the layer, then divide the padded FLOPs by the time.
    x, w1, w2, ids, wts = make_layer_inputs(
        tokens=TOKENS, model_dim=MODEL_DIM, inter_dim=INTER_DIM, experts=EXPERTS, topk=TOPK
    )
    measured = {}
    for tile_m in (64, 128):
        t = bench_us(lambda tm=tile_m: moe_forward(x, w1, w2, ids, wts, tile_m=tm), warmup=3, iters=10)
        measured[tile_m] = t / 1000.0
    thru = useful * (blocks[128] * 128 / rows) / (measured[128] * 1e-3) / 1e12
    print(f"  measured at tile_m=128: {measured[128]:.2f} ms")
    print(f"  padded FLOPs / that time = {thru:.1f} TFLOP/s of real MAC throughput")
    print(f"  against a dense library peak of {peak_tf:.1f} TFLOP/s on the same card")
    if thru > peak_tf:
        print("  -> the kernel is FASTER than the library once padding is accounted for,")
        print("     so the library number was never the ceiling")
    print()
    print(f"  {'tile_m':>7} {'B traffic':>11} {'DRAM ms':>9} {'compute ms':>11} {'model ms':>9} {'measured':>9}")
    for tile_m, nb in blocks.items():
        b_bytes = nb * (2 * INTER_DIM * MODEL_DIM * 2) + nb * (MODEL_DIM * INTER_DIM * 2)
        dram = b_bytes / bw * 1e3
        comp = useful * (nb * tile_m / rows) / (thru * 1e12) * 1e3
        got = measured.get(tile_m)
        print(
            f"  {tile_m:>7} {b_bytes / 1e9:9.2f} GB {dram:9.2f} {comp:11.2f} {max(dram, comp):9.2f} "
            f"{(f'{got:.2f}' if got else '-'):>9}"
        )
    print()
    print("  tile_m=128 wins not because it is good but because it sits exactly on")
    print("  the crossover: smaller tiles drown in repeated weight reads, larger")
    print("  ones drown in padding.")
    print()
    print("  If the measured tile_m=64 beats the model, that is o6 working: the DRAM")
    print("  column assumes every M tile re-reads its B slab, which is what the")
    print("  N-innermost order did. o6 removes part of that for gate_up, so reality")
    print("  now sits below the pre-o6 model. A model you can falsify is the point.")

    section("5. so what is actually fixable")
    once = (EXPERTS * 2 * INTER_DIM * MODEL_DIM * 2 + EXPERTS * MODEL_DIM * INTER_DIM * 2) / bw * 1e3
    print(f"  If the weights were read exactly once, DRAM would cost {once:.2f} ms and")
    print("  every row of the table above would be compute-bound. Then the best")
    print("  tile_m would be the *smallest* one, because padding is all that is left:")
    for tile_m, nb in blocks.items():
        comp = useful * (nb * tile_m / rows) / (thru * 1e12) * 1e3
        print(f"    tile_m={tile_m:3d} -> {comp:5.2f} ms")
    print()
    print("  o6 chases the first half of this: the repeated reads are a dispatch-order")
    print("  artifact, not a law.")
    print()
    print("  The second half -- the padding -- looks like the bigger prize here and is")
    print("  not. This table prices every row the same, so it reads 24% of the MACs as")
    print("  24% of the time. o7 measured the padding directly, by running a routing")
    print("  that needs none: the whole layer's padding is worth 1.16 ms of 10.80, or")
    print("  about 11%, because a padded row costs a third of what an average row does.")
    print("  Worse, splitting a stage into a main and a tail launch -- the obvious way")
    print("  to shed padding -- buys a second full pass over the weights, 2.26 ms, which")
    print("  is more than the entire budget. See o7_padding_measure.py.")


def main():
    header("roofline: measure the ceiling before optimising",
           f"D{MODEL_DIM}/I{INTER_DIM}, {TOKENS} tokens, E={EXPERTS}, topk={TOPK}")
    device_banner()
    peak, bw = measure_ceilings()
    blocks = padding_stats()
    model_vs_measured(peak, bw, blocks)


if __name__ == "__main__":
    main()
