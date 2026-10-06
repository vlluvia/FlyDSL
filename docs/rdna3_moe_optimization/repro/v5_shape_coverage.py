#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""v5: does the tile table hold up on the shapes real models actually have?

    python docs/rdna3_moe_optimization/repro/v5_shape_coverage.py
    python docs/rdna3_moe_optimization/repro/v5_shape_coverage.py --shapes kimi-k3
    python docs/rdna3_moe_optimization/repro/v5_shape_coverage.py --rows 2048 --no-check

o3 tuned the generic table at D2048/I768 and o5 added an exact-shape override for
D4096/I14336. Those are Qwen3-30B-A3B and Mixtral-8x7B, so coverage is not zero
-- but between them they span one regime, and the current generation is not in
it. Mixtral has 8 experts of width 14336; Qwen3.5-397B has 512 of width 1024 and
Kimi-K3 has 896 of width 3072. Two things move at once:

  * ``inter_dim`` collapses. ``gate_up``'s ``n_out`` is ``2*inter_dim``, so it
    goes from 28672 to 1024-6144, and ``down``'s ``k_dim`` *is* ``inter_dim``,
    which is the operand o6 weighed when it decided the dispatch order. Both
    tables are indexed by exactly these numbers.

  * rows per expert collapses with it. This is the one that is easy to miss.
    The tile is chosen for how many rows an expert gets, and in EP the expert
    card holds ``E / world`` of them -- 112 for Kimi-K3 at EP=8, against
    Mixtral's 8. At 2048 arriving rows that is 18 rows an expert, so a 128-row
    tile is 86% padding. o5's answer at this shape ("ask for a taller tile_m")
    inverts.

So this measures the axis those two tables are indexed on, over the shapes that
matter, and reports where the tuned choice is missing.

Configured the way EP actually feeds the layer: **the expert card sees topk=1**.
v3 dispatches one row per routed slot, so a row arrives carrying a single expert
and the local ``topk`` is always 1 no matter what the model's is (see the note in
the plan's v3 section). ``--rows`` is therefore the arriving row count, which for
a rank holding a slice of a topk=8 model at 1024 tokens is 8192, not 1024.

Shapes are the MoE block's, not the attention block's, and are the routed expert
width -- Kimi-K3 routes in a 3584-wide latent rather than its 7168 hidden, so
3584 is the number its grouped GEMM sees.
"""

from __future__ import annotations

import argparse

import torch

# common must be imported before anything under kernels/: it is what puts the
# repo root on sys.path. Keep these as from-imports so the sort order holds.
from common import bench_us, device_banner, header, layer_flops, section, torch_moe_layer
from kernels.moe.rdna3_moe import host
from kernels.moe.rdna3_moe.forward import moe_forward

# (key, label, model_dim, inter_dim, experts, topk, source)
#
# model_dim/inter_dim are what the grouped GEMM contracts over; experts/topk are
# the model's globals, which this script divides by --world to get a card's.
SHAPES = [
    (
        "qwen3-30b",
        "Qwen3-30B-A3B",
        2048,
        768,
        128,
        8,
        "o3 tuned the generic table here",
    ),
    (
        "mixtral-8x7b",
        "Mixtral-8x7B",
        4096,
        14336,
        8,
        2,
        "o5/o6 override lives here",
    ),
    (
        "qwen3.5-35b",
        "Qwen3.5-35B-A3B",
        2048,
        512,
        256,
        8,
        "hidden 2048, moe_intermediate 512, 256 experts + 1 shared",
    ),
    (
        "qwen3.5-397b",
        "Qwen3.5-397B-A17B",
        4096,
        1024,
        512,
        10,
        "hidden 4096, moe_intermediate 1024, 512 experts",
    ),
    (
        "deepseek-v4",
        "DeepSeek-V4",
        4096,
        2048,
        256,
        6,
        "hidden 4096, moe_intermediate 2048, 256 routed + 1 shared",
    ),
    (
        "kimi-k3",
        "Kimi-K3",
        3584,
        3072,
        896,
        16,
        "latent MoE dim 3584 (not hidden 7168), moe_intermediate 3072, 896 experts",
    ),
]

TILE_MS = (16, 32, 64, 128)

# The ceilings the roofline column is a fraction of, all measured on this
# gfx1100 and all already in the docs: BENCHMARK.md §1 for the 829 GB/s, and
# rdna3_int8_gemm's docstring for the 100 TFLOP/s (a loop of back-to-back WMMA
# out of registers -- the 122.6 on the spec sheet assumes a boost clock the card
# does not hold). MALL is Navi31's 96 MB last-level cache.
HBM_GBPS = 829.0
WMMA_TFLOPS = 100.0
MALL_BYTES = 96 * 1024 * 1024


def stage_floor(*, counts, tile_m, k_dim, n_out):
    """One grouped-GEMM stage's ``(bytes moved, MACs issued)`` floor.

    Shared with ``v5_stage_gap.py`` so the two scripts cannot drift into
    disagreeing models. See ``predict_us`` for what the B term assumes and where
    it is known to be wrong.
    """
    tiles = counts.add(tile_m - 1).div(tile_m, rounding_mode="floor")
    padded_rows = int((tiles * tile_m).sum())
    slab = n_out * k_dim * 2
    # One B pass if the slab caches, otherwise one per M tile that reads it.
    b_passes = int(tiles.sum()) if slab > MALL_BYTES else counts.numel()
    nbytes = b_passes * slab
    # A is held across the N sweep (N is innermost unless _SHAPE_M_MAJOR says
    # otherwise), so it is read once per M tile, not once per N tile.
    nbytes += padded_rows * k_dim * 2
    nbytes += padded_rows * n_out * 2
    return nbytes, 2.0 * padded_rows * n_out * k_dim


def predict_us(*, counts, tile_m, model_dim, inter_dim, bw_gbps, mac_tflops):
    """o6's model, per stage: ``max(bytes / bandwidth, padded MACs / ceiling)``.

    o6 established that a stage's floor is whichever of the two binds, and that
    the B term is what the dispatch order moves: an expert's weight slab is
    re-read once per M tile it owns. What o6 did not have to model is the case
    this script is full of -- a slab small enough that the re-reads are cache
    hits. Mixtral's is 352 MB against a 96 MB MALL so every re-read is a real
    trip; Kimi-K3 at EP=8 has 112 experts of 66 MB, which fit. So the B term
    counts the re-reads only for the shapes that cannot cache them.

    Where this is known to be wrong: when the slab misses MALL *and* an expert
    owns many tiles, it over-predicts badly -- Mixtral at tile_m=16 measures
    20.5 ms against a predicted 54.7 ms. A floor the measurement beats is a
    falsified model, not a fast kernel, so the caller marks those rows. The B
    term there is all-or-nothing (cached or not) while the truth is that L2
    catches part of an N sweep even when the slab as a whole does not fit. The
    column is trustworthy for the mem-bound shapes with cacheable slabs and for
    the mac-bound rows, which is where this script uses it.

    Returns ``(microseconds, which_term)``.
    """
    total_bytes = 0.0
    total_mac = 0.0
    for k_dim, n_out in ((model_dim, 2 * inter_dim), (inter_dim, model_dim)):
        nbytes, macs = stage_floor(counts=counts, tile_m=tile_m, k_dim=k_dim, n_out=n_out)
        total_bytes += nbytes
        total_mac += macs

    mem_us = total_bytes / (bw_gbps * 1e9) * 1e6
    mac_us = total_mac / (mac_tflops * 1e12) * 1e6
    return (max(mem_us, mac_us), "mem" if mem_us >= mac_us else "mac")


def table_keys(model_dim: int, inter_dim: int):
    """``(gate_up, down)`` as the two shape tables index them.

    Both tables are keyed on the ``n_out`` the *builder* is given, and for
    ``gate_up`` that is ``inter_dim``: the kernel doubles it itself to run the
    gate and up B streams over one A tile. So the key is not the 2*inter_dim
    the stage actually writes -- see ``compile_moe_gemm1``.
    """
    return ("gate_up", model_dim, inter_dim), ("down", inter_dim, model_dim)


def tuned_state(model_dim: int, inter_dim: int) -> str:
    """Which of the two shape-indexed tables already name this shape."""
    g1, d = table_keys(model_dim, inter_dim)
    named = [k for k in (g1, d) if any(t[:3] == k for t in host._SHAPE_TILES)]
    order = [k for k in (g1, d) if k in host._SHAPE_M_MAJOR]
    parts = []
    if named:
        parts.append("_SHAPE_TILES(" + ",".join(k[0] for k in named) + ")")
    if order:
        parts.append("_SHAPE_M_MAJOR(" + ",".join(k[0] for k in order) + ")")
    return ", ".join(parts) or "generic"


def geometry(model_dim, inter_dim, experts, tile_m):
    """``(gate_up, down)`` block shapes the tables hand this stage, or an error."""
    try:
        g1, bm1, bn1, bk1 = host.compile_moe_gemm1(
            model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=1, tile_m=tile_m
        )
        g2, bm2, bn2, bk2 = host.compile_moe_gemm2(
            model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=1, tile_m=tile_m
        )
    except ValueError as exc:
        return None, str(exc)
    return (f"{bm1}x{bn1}x{bk1}", f"{bm2}x{bn2}x{bk2}"), None


def make_inputs(*, rows, model_dim, inter_dim, experts, seed=0):
    """One expert card's inputs: ``rows`` arriving rows, each naming one expert.

    Not ``make_layer_inputs``, which routes through ``moe_gating`` -- that kernel
    reduces inside a power-of-two lane group and rules out some expert counts,
    and anyway the gating runs on the token's home card at the model's global
    expert count, not here. What arrives here is already routed.
    """
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(seed)
    x = (torch.randn(rows, model_dim, generator=gen, device=dev, dtype=torch.bfloat16) * 0.1).contiguous()
    w1 = (
        torch.randn(experts, 2 * inter_dim, model_dim, generator=gen, device=dev, dtype=torch.bfloat16) * 0.05
    ).contiguous()
    w2 = (
        torch.randn(experts, model_dim, inter_dim, generator=gen, device=dev, dtype=torch.bfloat16) * 0.05
    ).contiguous()
    ids = torch.randint(0, experts, (rows, 1), generator=gen, device=dev, dtype=torch.int32).contiguous()
    wts = torch.rand(rows, 1, generator=gen, device=dev, dtype=torch.float32).contiguous()
    return x, w1, w2, ids, wts


def run_shape(key, label, model_dim, inter_dim, experts, topk, source, *, world, rows, check, iters):
    epr = max(1, experts // world)
    section(f"{label}   D{model_dim}/I{inter_dim}, E={experts} topk={topk}")
    print(f"  {source}")
    print(
        f"  a card at EP={world} holds {epr} experts; {rows} arriving rows is "
        f"{rows / epr:.1f} rows an expert"
    )
    print(
        f"  gate_up: writes {2 * inter_dim} cols over k={model_dim} (tables key it as n_out={inter_dim});"
        f" down: n_out={model_dim} k={inter_dim}"
    )
    print(f"  shape tables say: {tuned_state(model_dim, inter_dim)}")

    weight_gb = epr * (2 * inter_dim * model_dim + model_dim * inter_dim) * 2 / 1e9
    print(f"  weights on the card: {weight_gb:.2f} GB")
    print()

    try:
        x, w1, w2, ids, wts = make_inputs(rows=rows, model_dim=model_dim, inter_dim=inter_dim, experts=epr)
    except torch.OutOfMemoryError:
        print("  SKIP: the expert slice does not fit this card")
        return None

    flop = layer_flops(tokens=rows, model_dim=model_dim, inter_dim=inter_dim, topk=1)
    counts = torch.bincount(ids.reshape(-1).long(), minlength=epr)
    t_torch = bench_us(lambda: torch_moe_layer(x, w1, w2, ids, wts), warmup=2, iters=3)

    print(
        f"  {'tile_m':>7} {'gate_up':>14} {'down':>14} {'layer':>10} {'TFLOP/s':>8} "
        f"{'vs torch':>9} {'pad':>5} {'floor':>9} {'bound':>6} {'of floor':>9}"
    )
    best = None
    for tile_m in TILE_MS:
        geo, err = geometry(model_dim, inter_dim, epr, tile_m)
        if geo is None:
            print(f"  {tile_m:>7} {'no tile fits':>14} {'':>14}   {err[:44]}")
            continue
        try:
            t = bench_us(lambda tm=tile_m: moe_forward(x, w1, w2, ids, wts, tile_m=tm), warmup=3, iters=iters)
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            print(f"  {tile_m:>7} {geo[0]:>14} {geo[1]:>14}   FAILED: {str(exc)[:40]}")
            continue
        # Padding is per expert: a tile only ever holds one expert's rows, so
        # every expert rounds its own count up to a whole tile. Taken from the
        # actual histogram rather than rows/epr, because the routing is not
        # balanced and the rounding is what the imbalance shows up in.
        padded = int((counts.add(tile_m - 1).div(tile_m, rounding_mode="floor") * tile_m).sum())
        pad = 1 - rows / max(padded, 1)
        floor, bound = predict_us(
            counts=counts,
            tile_m=tile_m,
            model_dim=model_dim,
            inter_dim=inter_dim,
            bw_gbps=HBM_GBPS,
            mac_tflops=WMMA_TFLOPS,
        )
        # A floor above the measurement means the model is wrong here, not that
        # the kernel is fast; see predict_us.
        flag = " <- model falsified" if floor > t else ""
        print(
            f"  {tile_m:>7} {geo[0]:>14} {geo[1]:>14} {t:9.1f}u {flop / (t * 1e-6) / 1e12:8.2f} "
            f"{t_torch / t:8.2f}x {100 * pad:4.0f}% {floor:8.1f}u {bound:>6} {100 * floor / t:8.0f}%{flag}"
        )
        if best is None or t < best[1]:
            best = (tile_m, t, floor, bound)

    if best is None:
        print("\n  no tile_m works on this shape")
        return None

    print(f"\n  eager torch: {t_torch:.1f}us {flop / (t_torch * 1e-6) / 1e12:.2f} TFLOP/s")
    print(f"  best tile_m={best[0]} at {flop / (best[1] * 1e-6) / 1e12:.2f} TFLOP/s, {t_torch / best[1]:.2f}x torch")
    print(
        f"  that is {100 * best[2] / best[1]:.0f}% of its {best[3]}-bound floor, so the headroom "
        f"left to tuning is {best[1] - best[2]:.0f}us"
    )

    if check:
        y = moe_forward(x, w1, w2, ids, wts, tile_m=best[0])
        ref = torch_moe_layer(x, w1, w2, ids, wts)
        diff = (y.float() - ref.float()).abs().max().item()
        scale = ref.float().abs().max().item()
        print(f"  correctness at tile_m={best[0]}: max abs err {diff:.2e} on a {scale:.2e} output")

    del x, w1, w2, ids, wts
    torch.cuda.empty_cache()
    return (
        key,
        label,
        best[0],
        flop / (best[1] * 1e-6) / 1e12,
        t_torch / best[1],
        rows / epr,
        100 * best[2] / best[1],
        best[3],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", nargs="*", default=None, help="keys to run, default all")
    p.add_argument("--world", type=int, default=8, help="EP size the expert slice is sized for")
    p.add_argument("--rows", type=int, default=2048, help="arriving rows, i.e. tokens x model topk")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--no-check", action="store_true", help="skip the torch correctness compare")
    args = p.parse_args()

    header(
        "v5  the tile table on real model shapes",
        f"{args.rows} arriving rows, one card of an EP={args.world} group, local topk=1",
    )
    device_banner()
    print()
    print("  Rows per expert is the column to watch: it is what picks tile_m, and")
    print("  it falls as the expert count rises. See the module docstring.")

    wanted = SHAPES if not args.shapes else [s for s in SHAPES if s[0] in set(args.shapes)]
    if not wanted:
        raise SystemExit(f"no shape matched {args.shapes}; keys are {[s[0] for s in SHAPES]}")

    results = []
    for shape in wanted:
        got = run_shape(
            *shape, world=args.world, rows=args.rows, check=not args.no_check, iters=args.iters
        )
        if got:
            results.append(got)

    section("summary")
    print(
        f"  {'model':>20} {'rows/expert':>12} {'best tile_m':>12} {'TFLOP/s':>9} "
        f"{'vs torch':>9} {'of floor':>9} {'bound by':>9}"
    )
    for _, label, tile_m, tflops, speedup, rpe, of_floor, bound in results:
        print(
            f"  {label:>20} {rpe:12.1f} {tile_m:12d} {tflops:9.2f} {speedup:8.2f}x "
            f"{of_floor:8.0f}% {bound:>9}"
        )
    print()
    print("  TFLOP/s falls with rows per expert, but read the 'of floor' column before")
    print("  calling that a tuning gap: a shape whose weights dwarf its work cannot")
    print("  reach Mixtral's number no matter how the tile is shaped.")


if __name__ == "__main__":
    main()
