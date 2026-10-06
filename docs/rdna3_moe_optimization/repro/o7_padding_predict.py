#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""o7, part 1: the case for cutting the routing's padding, and how it is built.

    python docs/rdna3_moe_optimization/repro/o7_padding_predict.py

Read this together with o7_padding_measure.py, which runs what this predicts and
finds the opposite. This half is kept because the way it goes wrong is the
lesson, not a detour on the way to one.

roofline.py ends on a number: at tile_m=128 the wide shape spends 24.4% of its
MACs on rows the routing padded in, and calls that the next structural lever. The
obvious way at it is to stop padding every expert up to 128 -- give each one as
many full 128-row tiles as its rows fill, then one short tail tile for the
remainder:

    n_main = c // 128              main tiles, exactly full, no padding
    n_tail = ceil((c % 128) / 64)  tail tiles, padded to 64

An expert with 302 rows becomes 2 main + 1 tail = 320 padded rows where padding
to 128 costs 384. Across the routing that takes the waste from 24% to about 12%,
and to 6% with a 32-row tail.

Whether less padding is less time is a different question, and this script tries
to answer it before anything is built. Padding is not the only thing tile height
decides: a short tile has less work per wave, so the rows moved into it may cost
more each than the padded rows they replace. That effect cannot be modelled, only
measured -- so measure the per-row cost at each height, then use it to price the
scheme:

    t(scheme) = sum over heights of  padded_rows(height) / rate(height)

The flaw is in that formula, and it is not visible from here.
"""

from __future__ import annotations

import contextlib

import numpy as np
import torch

# common must be imported before anything under kernels/: it is what puts the
# repo root on sys.path. Keep these as from-imports so the sort order holds.
from common import bench_us, device_banner, header, make_layer_inputs, section

from kernels.moe.rdna3_moe import host
from kernels.moe.rdna3_moe.forward import moe_forward
from kernels.moe.rdna3_moe.grouped_gemm import create_grouped_gemm_module
from kernels.moe.rdna3_moe.routing_kernel import build_routing_device

EXPERTS, TOPK, TOKENS = 8, 2, 1024
MODEL_DIM, INTER_DIM = 4096, 14336
HEIGHTS = (16, 32, 64, 128)
STAGES = (("gate_up", MODEL_DIM, INTER_DIM), ("down", INTER_DIM, MODEL_DIM))

# BLOCK_M = reg_m*16*waves_m and BLOCK_N = reg_n*16*waves_n, so a tile height can
# be spent on more rows per wave or on a wider N. The production table picks each
# height's config for the token count that height is meant to serve: a decode
# wants a narrow N so there are enough workgroups to fill the card. A tail tile is
# not that case -- it rides along in a prefill whose N grid is already wide, so it
# may want the wide, register-heavy config only the 128 height uses today. Sweep,
# rather than assume the production entry is the height's best.
CANDIDATES = {
    16: ((1, 2, 4, 1, 2), (1, 4, 2, 1, 2)),
    32: ((2, 1, 4, 1, 2), (2, 2, 2, 1, 2), (2, 4, 2, 1, 2), (2, 2, 2, 1, 4), (1, 4, 2, 2, 2)),
    64: ((2, 1, 4, 2, 2), (2, 2, 2, 2, 2), (2, 4, 2, 2, 2), (4, 2, 2, 1, 2), (4, 4, 2, 1, 2)),
    128: ((4, 1, 2, 2, 2), (4, 2, 2, 2, 2), (4, 4, 2, 2, 2), (2, 4, 2, 4, 2)),
}

# What the sweep below picks for the 64 height, so section 4 can try it for real.
SWEPT_64 = {
    ("gate_up", MODEL_DIM, INTER_DIM, 64): ((2, 4, 2, 2, 2),),
    ("down", INTER_DIM, MODEL_DIM, 64): ((4, 4, 2, 1, 2),),
}


def measure_rate(stage, k_dim, n_out, height, cfg):
    """Time one stage at one tile and back out its MAC rate.

    The rate counts padded rows, not real ones: it is what a row costs to push
    through this tile, which is what a scheme mixing heights needs priced.
    """
    reg_m, reg_n, reg_k, waves_m, waves_n = cfg
    launch, bm, bn, _ = create_grouped_gemm_module(
        k_dim=k_dim,
        n_out=n_out,
        experts=EXPERTS,
        stage=stage,
        topk=TOPK,
        doweight=(stage == "down"),
        reg_m=reg_m,
        reg_n=reg_n,
        reg_k=reg_k,
        waves_m=waves_m,
        waves_n=waves_n,
        m_major=(stage, k_dim, n_out) in host._SHAPE_M_MAJOR,
    )
    assert bm == height, f"{cfg} gives BLOCK_M={bm}, not {height}"
    dev = "cuda"
    torch.manual_seed(0)
    a_rows = TOKENS * TOPK if stage == "down" else TOKENS
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
    padded = r.num_blocks * bm
    return us, 2.0 * padded * w_rows * k_dim / (us * 1e-6) / 1e12, r.num_blocks, bn


def expert_counts(trials=400):
    """The routing's rows-per-expert distribution, which is what padding acts on."""
    gen = torch.Generator(device="cuda").manual_seed(0)
    out = []
    for _ in range(trials):
        logits = torch.randn(TOKENS, EXPERTS, generator=gen, device="cuda")
        sel = logits.topk(TOPK, dim=-1).indices.reshape(-1)
        out.append(torch.bincount(sel, minlength=EXPERTS).cpu().numpy())
    return np.stack(out)


def plan_uniform(counts, height):
    """Rows and tiles under today's scheme: every expert padded up to height."""
    blocks = np.maximum((counts + height - 1) // height, 1)
    return {height: blocks.sum(axis=1).mean() * height}, blocks.sum(axis=1).mean()


def plan_mixed(counts, main, tail):
    """Rows and tiles under full main tiles plus a tail tile per expert."""
    n_main = counts // main
    rest = counts - n_main * main
    n_tail = (rest + tail - 1) // tail
    rows = {main: n_main.sum(axis=1).mean() * main, tail: n_tail.sum(axis=1).mean() * tail}
    return rows, n_main.sum(axis=1).mean() + n_tail.sum(axis=1).mean()


def price(rows, rates, stage, n_out, k_dim):
    """Turn a row plan into a time, each height at its own measured rate."""
    return sum(2.0 * n * n_out * k_dim / (rates[(stage, h)] * 1e12) * 1e3 for h, n in rows.items())


@contextlib.contextmanager
def swept_64(on):
    saved = {k: host._SHAPE_TILES.get(k) for k in SWEPT_64}
    if on:
        host._SHAPE_TILES.update(SWEPT_64)
    host.compile_grouped_gemm.cache_clear()
    try:
        yield
    finally:
        for k, v in saved.items():
            host._SHAPE_TILES.pop(k, None) if v is None else host._SHAPE_TILES.update({k: v})
        host.compile_grouped_gemm.cache_clear()


def sweep_heights():
    section("1. what a row costs at each tile height")
    print("  Same stage, same shape; only the tile changes. The rate counts padded")
    print("  rows, so it is the cost of pushing a row through this tile. Each height")
    print("  gets a config sweep, not just its production entry: the question is")
    print("  whether a short tile is slow because it is short, or only because its")
    print("  production config also narrows N.")
    print()
    rates, best = {}, {}
    for stage, k_dim, n_out in STAGES:
        prod = {h: host._TILES[stage][h][0] for h in HEIGHTS}
        prod[128] = host._SHAPE_TILES.get((stage, k_dim, n_out, 128), (prod[128],))[0]
        print(f"  {stage}")
        print(f"    {'tile':>10} {'tiles':>6} {'padded':>7} {'ms':>8} {'MAC TFLOP/s':>12}")
        for height in HEIGHTS:
            for cfg in CANDIDATES[height]:
                try:
                    us, tf, nb, bn = measure_rate(stage, k_dim, n_out, height, cfg)
                except Exception as e:
                    print(f"    {f'{height}x?':>10} skipped: {str(e).splitlines()[0][:44]}")
                    continue
                if tf > rates.get((stage, height), 0):
                    rates[(stage, height)], best[(stage, height)] = tf, (height, bn)
                tag = "  <- production" if cfg == prod[height] else ""
                print(f"    {f'{height}x{bn}':>10} {nb:>6} {nb * height:>7} {us / 1000:8.3f} {tf:12.2f}{tag}")
            print()
    print("  Config matters more than height does. gate_up's 64 height goes from 60 to")
    print("  75 TFLOP/s once N is widened, so a tail tile is not doomed to be slow --")
    print("  which is what makes the scheme look worth trying.")
    return rates, best


def main():
    header(
        "o7 part 1  predicting what cutting the padding is worth",
        f"D{MODEL_DIM}/I{INTER_DIM}, {TOKENS} tokens, E={EXPERTS}, topk={TOPK}",
    )
    device_banner()
    rates, best = sweep_heights()

    section("2. how much padding each scheme actually pays for")
    counts = expert_counts()
    real = TOKENS * TOPK
    print(f"  rows to place: {real}   experts: {EXPERTS}   mean rows/expert: {counts.mean():.1f}")
    print(f"  over {counts.shape[0]} routings the per-expert count runs {counts.min()} to {counts.max()}")
    print()
    schemes = [("uniform", h, None) for h in HEIGHTS] + [("mixed", 128, t) for t in (16, 32, 64)]
    plans = {}
    print(f"  {'scheme':>16} {'tiles':>7} {'padded rows':>12} {'waste':>8}")
    for kind, main_h, tail in schemes:
        plan, tiles = plan_uniform(counts, main_h) if tail is None else plan_mixed(counts, main_h, tail)
        name = f"{main_h}" if tail is None else f"{main_h}+{tail}"
        plans[name] = plan
        padded = sum(plan.values())
        print(f"  {name:>16} {tiles:7.1f} {padded:12.0f} {100 * (padded / real - 1):7.1f}%")
    print()
    print("  The padding does fall the way it was supposed to. A 64-row tail halves it")
    print("  and a 16-row tail all but removes it.")

    section("3. the prediction")
    print("  Each scheme's rows, priced at the best rate its height reached:")
    for stage, k_dim, n_out in STAGES:
        w_rows = (2 if stage == "gate_up" else 1) * n_out
        print(f"  {stage}   " + "  ".join(f"{h}x{best[(stage, h)][1]}:{rates[(stage, h)]:.1f}T" for h in HEIGHTS))
        base = price(plans["128"], rates, stage, w_rows, k_dim)
        for name, plan in plans.items():
            ms = price(plan, rates, stage, w_rows, k_dim)
            print(f"    {name:>10} {ms:8.3f} ms {100 * (base / ms - 1):+7.1f}%")
        print()
    print("  So: mixed 128+64 is worth about +6% on gate_up, and every mixed variant")
    print("  loses on down, whose short tiles are much worse because its k_dim is the")
    print("  intermediate size. Stage1 is roughly two thirds of the layer, so the")
    print("  prediction for the layer is about +3.7% -- from a stage1-only change.")

    section("4. one reason to distrust all of the above")
    print("  Section 1 timed one stage, at one token count, in isolation. Nothing")
    print("  guarantees those numbers survive a whole layer. Its 64-height winner is")
    print("  cheap to check, so check it: put the swept config in the shape table and")
    print("  run the real layer at tile_m=64.")
    print()
    print(f"  {'tokens':>7} {'generic 64x32':>14} {'swept 64x128':>14} {'delta':>8} {'tile_m=128':>12}")
    for tokens in (128, 256, 512, 1024):
        x, w1, w2, ids, wts = make_layer_inputs(
            tokens=tokens, model_dim=MODEL_DIM, inter_dim=INTER_DIM, experts=EXPERTS, topk=TOPK
        )
        got = {}
        for on in (False, True):
            with swept_64(on):
                moe_forward(x, w1, w2, ids, wts, tile_m=64)
                torch.cuda.synchronize()
                got[on] = bench_us(lambda: moe_forward(x, w1, w2, ids, wts, tile_m=64), warmup=3, iters=20)
        t128 = bench_us(lambda: moe_forward(x, w1, w2, ids, wts, tile_m=128), warmup=3, iters=20)
        print(
            f"  {tokens:>7} {got[False] / 1000:13.3f}m {got[True] / 1000:13.3f}m "
            f"{100 * (got[False] / got[True] - 1):+7.1f}% {t128 / 1000:11.3f}m"
        )
    print()
    print("  The swept config only wins at 512 and 1024 tokens -- exactly where the")
    print("  layer would have picked the 128 height anyway, and where 128 beats both.")
    print("  Where the 64 height is actually used, the generic narrow-N entry is about")
    print("  20% faster. A per-stage number at one token count did not transfer.")
    print()
    print("  Which is a reason to build the mixed scheme and measure it rather than")
    print("  ship the prediction: o7_padding_measure.py.")


if __name__ == "__main__":
    main()
