#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""o7 probe: is a mixed tile size worth it, before writing any of it.

    python docs/rdna3_moe_optimization/repro/o7_probe.py

roofline.py ended on a number: at tile_m=128 the wide shape spends 24.4% of its
MACs on rows the routing padded in. The obvious fix is to stop padding every
expert to 128 -- give each one as many 128-row tiles as its rows fill, then a
small tail tile for the remainder. That drops the waste to about the tail size
over the average expert.

But padding is not the only thing a tile size decides, so the fix is not
obviously a win. Two things push back:

  more tiles       a 32-row tail tile re-reads its expert's whole B slab, the
                   same as a 128-row tile does. Tile count is what drives weight
                   traffic, so cutting padding buys compute by spending DRAM.
  worse tiles      a short tile has less work per wave to hide latency behind.
                   Its achieved MAC rate is lower, so the tail rows cost more
                   each than the padded rows they replace.

The first is in the roofline model already. The second is not, and cannot be --
it has to be measured. So this script measures the padding-inclusive MAC rate at
each tile size, then uses it to predict the mixed scheme before implementing it:

    t(scheme) = sum over tile sizes of  padded_MACs(size) / rate(size)

If the prediction says no, that is the cheapest possible answer.
"""

from __future__ import annotations

import numpy as np
import torch

# common must be imported before anything under kernels/: it is what puts the
# repo root on sys.path. Keep these as from-imports so the sort order holds.
from common import bench_us, device_banner, header, section

from kernels.moe.rdna3_moe import host
from kernels.moe.rdna3_moe.grouped_gemm import create_grouped_gemm_module
from kernels.moe.rdna3_moe.routing_kernel import build_routing_device

EXPERTS, TOPK, TOKENS = 8, 2, 1024
MODEL_DIM, INTER_DIM = 4096, 14336
TILES = (16, 32, 64, 128)
STAGES = (("gate_up", MODEL_DIM, INTER_DIM), ("down", INTER_DIM, MODEL_DIM))

# BLOCK_M = reg_m*16*waves_m and BLOCK_N = reg_n*16*waves_n, so a given tile
# height can be spent on more rows per wave or on a wider N. The production
# table picks each height's config for the token count that height is meant to
# serve -- a decode wants a narrow N so there are enough workgroups to fill the
# card. A tail tile is not that case: it rides along in a prefill where the N
# grid is already wide, so it can afford the wide, register-heavy config that
# only the 128 row height uses today. These are the candidates per height.
CANDIDATES = {
    16: ((1, 2, 4, 1, 2), (1, 4, 2, 1, 2), (1, 2, 2, 1, 4), (1, 4, 2, 1, 4)),
    32: ((2, 1, 4, 1, 2), (2, 2, 2, 1, 2), (2, 4, 2, 1, 2), (2, 2, 2, 1, 4), (1, 4, 2, 2, 2)),
    64: ((2, 1, 4, 2, 2), (2, 2, 2, 2, 2), (2, 4, 2, 2, 2), (4, 2, 2, 1, 2), (4, 4, 2, 1, 2)),
    128: ((4, 1, 2, 2, 2), (4, 2, 2, 2, 2), (4, 4, 2, 2, 2), (2, 4, 2, 4, 2)),
}


def measure_rate(stage, k_dim, n_out, tile_m, cfg):
    """Time one stage at one tile size and config, and back out its MAC rate.

    The rate counts the padded rows, not the real ones: it is what a row costs
    to push through this tile, which is the thing that extrapolates to a scheme
    that mixes sizes.
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
    assert bm == tile_m, f"{cfg} gives BLOCK_M={bm}, not {tile_m}"
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
    macs = padded * w_rows * k_dim
    return us, 2.0 * macs / (us * 1e-6) / 1e12, r.num_blocks, bn


def expert_counts(trials=400):
    """The routing's row-per-expert distribution, which is what padding acts on."""
    gen = torch.Generator(device="cuda").manual_seed(0)
    out = []
    for _ in range(trials):
        logits = torch.randn(TOKENS, EXPERTS, generator=gen, device="cuda")
        sel = logits.topk(TOPK, dim=-1).indices.reshape(-1)
        out.append(torch.bincount(sel, minlength=EXPERTS).cpu().numpy())
    return np.stack(out)


def plan_uniform(counts, tile_m):
    """Rows and tiles under today's scheme: every expert padded up to tile_m."""
    blocks = np.maximum((counts + tile_m - 1) // tile_m, 1)
    return {tile_m: blocks.sum(axis=1).mean() * tile_m}, blocks.sum(axis=1).mean()


def plan_mixed(counts, main, tail):
    """Rows and tiles under a main tile plus a tail tile per expert."""
    n_main = counts // main
    rest = counts - n_main * main
    n_tail = np.maximum((rest + tail - 1) // tail, 1)  # keep a tile per expert
    rows = {main: n_main.sum(axis=1).mean() * main, tail: n_tail.sum(axis=1).mean() * tail}
    return rows, n_main.sum(axis=1).mean() + n_tail.sum(axis=1).mean()


def predict(rows, rates, stage, k_dim, n_out):
    """Turn a row plan into a time, at each size's own measured rate."""
    w_rows = (2 if stage == "gate_up" else 1) * n_out
    return sum(2.0 * n * w_rows * k_dim / (rates[(stage, size)] * 1e12) * 1e3 for size, n in rows.items())


def main():
    header("o7 probe  should padding be the next thing to touch?", f"D{MODEL_DIM}/I{INTER_DIM}, {TOKENS} tokens, E={EXPERTS}, topk={TOPK}")
    device_banner()

    section("1. what a row costs at each tile height")
    print("  Same stage, same shape, only the tile changes. The rate counts padded")
    print("  rows, so it is the cost of pushing a row through this tile -- and a")
    print("  short tile being worse at it is the effect the roofline model has no")
    print("  way to know about.")
    print()
    print("  Each height gets a config sweep, not just the production entry: the")
    print("  question is whether a short tile is slow because it is short, or only")
    print("  because its production config also narrows N.")
    print()
    rates, best_cfg = {}, {}
    for stage, k_dim, n_out in STAGES:
        prod = dict(zip(TILES, (host._TILES[stage][t][0] for t in TILES)))
        prod[128] = host._SHAPE_TILES.get((stage, k_dim, n_out, 128), (prod[128],))[0]
        print(f"  {stage}")
        print(f"    {'tile':>10} {'tiles':>6} {'padded':>7} {'us':>8} {'MAC TFLOP/s':>12}  {'':>4}")
        for tile_m in TILES:
            for cfg in CANDIDATES[tile_m]:
                try:
                    us, tf, nb, bn = measure_rate(stage, k_dim, n_out, tile_m, cfg)
                except Exception as e:  # a config the compiler will not take
                    print(f"    {f'{tile_m}x?':>10} {str(e)[:52]}")
                    continue
                tag = "  <- production" if cfg == prod[tile_m] else ""
                if tf > rates.get((stage, tile_m), 0):
                    rates[(stage, tile_m)], best_cfg[(stage, tile_m)] = tf, cfg
                print(f"    {f'{tile_m}x{bn}':>10} {nb:>6} {nb * tile_m:>7} {us:8.1f} {tf:12.2f}{tag}")
            print()

    section("2. how much padding each scheme actually pays for")
    counts = expert_counts()
    rows = TOKENS * TOPK
    print(f"  rows to place: {rows}   experts: {EXPERTS}   mean rows/expert: {counts.mean():.1f}")
    print(f"  spread across {counts.shape[0]} routings: min {counts.min()}, max {counts.max()}")
    print()
    schemes = [("uniform", t, None) for t in TILES] + [("mixed", 128, t) for t in (16, 32, 64)]
    print(f"  {'scheme':>16} {'tiles':>7} {'padded rows':>12} {'waste':>8}")
    plans = {}
    for kind, main, tail in schemes:
        plan, tiles = plan_uniform(counts, main) if tail is None else plan_mixed(counts, main, tail)
        name = f"{main}" if tail is None else f"{main}+{tail}"
        plans[(kind, main, tail)] = (plan, tiles, name)
        padded = sum(plan.values())
        print(f"  {name:>16} {tiles:7.1f} {padded:12.0f} {100 * (padded / rows - 1):7.1f}%")

    section("3. the prediction")
    print("  Padding falls the way it was supposed to. Whether that is a speedup")
    print("  depends on what the replacement rows cost, at the best config each")
    print("  height can reach:")
    print()
    for stage, k_dim, n_out in STAGES:
        print(f"  {stage}  best per height: " + "  ".join(f"{t}:{rates[(stage, t)]:.1f}T" for t in TILES))
        print(f"    {'scheme':>16} {'ms':>8} {'vs 128':>8}")
        base = predict(plans[("uniform", 128, None)][0], rates, stage, k_dim, n_out)
        for key, (plan, _, name) in plans.items():
            ms = predict(plan, rates, stage, k_dim, n_out)
            print(f"    {name:>16} {ms:8.3f} {100 * (base / ms - 1):+7.1f}%")
        print()


if __name__ == "__main__":
    main()
