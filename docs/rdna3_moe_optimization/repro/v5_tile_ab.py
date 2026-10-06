#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""v5-2b: a deeper BLOCK_K for ``down``, measured on the whole layer.

    python docs/rdna3_moe_optimization/repro/v5_tile_ab.py
    python docs/rdna3_moe_optimization/repro/v5_tile_ab.py --shapes deepseek-v4
    python docs/rdna3_moe_optimization/repro/v5_tile_ab.py --raw 2048,768,8,2,1024

This is the reproduction of what ``host._BUCKET_TILES`` is worth: it empties that
table for one arm and leaves it for the other, the same way ``o5_shape_override``
empties ``_SHAPE_TILES``.

``sweep_rdna3_moe_tiles.py`` times one stage with a pre-built routing, which is
the right tool for picking a tile and the wrong one for deciding whether to ship
it, for two reasons. A stage that is 22% of the layer can win 20% and move the
layer by 4. And it builds a *different kernel* than production: it goes through
``exact`` routing with ``bounded_blocks=False``, while ``moe_forward`` takes o2's
``bounded_blocks=True``, where the grid is an upper bound and the surplus
workgroups exit early. Those tails do not rank tiles the same -- at one shape the
stage said the generic tile won by 5.3% and the layer said the candidate won by
1.2%, opposite signs. So the layer is what decides.

What the sweeps found, and why it is worth a table entry:

  * ``gate_up`` has no headroom at any of these shapes. The generic table's
    first choice *is* the sweep's best -- 64x32x64 at both v5 targets, and
    128x32x64 at DeepSeek-V4 within 0.2 us of the generic 128x32x32. That is a
    negative result and it is the more useful half: the 45-50% of roofline
    ``v5_shape_coverage.py`` reported at these shapes is not a tile gap, so no
    amount of sweeping gate_up will close it.

  * ``down`` does, and o3's own rule is what mispredicts it. o3 halved BLOCK_K
    from 64 to 32 to fit a second workgroup per CU, worth most of what the
    single-B stages gained at D2048/I768. But BLOCK_K=32 doubles the k-step
    count, and each step carries a barrier. When ``k_dim`` is the *intermediate*
    size -- which for ``down`` it is, and which the current generation has
    collapsed to 512-2048 -- there are too few steps left to amortise them, and
    the trade inverts. Measured on the stage, D4096: 18.7% at I=768, 25.3% at
    I=1024, 23.8% at I=2048, all going the other way from o3.

Only ``down`` got an entry, because only ``down`` had one to earn. And the entry
is keyed on how many tiles an expert owns rather than on the shape, because the
same D2048/I768 wants opposite things at 16 experts and at 8 -- pass ``--raw`` to
see that, since without it every config is the EP one (``topk=1``) and the
regressing case is o3's ``topk=2``.
"""

from __future__ import annotations

import argparse

import torch

# common must be imported before anything under kernels/: it is what puts the
# repo root on sys.path. Keep these as from-imports so the sort order holds.
from common import bench_us, device_banner, header, layer_flops, section
from kernels.moe.rdna3_moe import host
from kernels.moe.rdna3_moe.forward import moe_forward
from v5_shape_coverage import SHAPES, table_keys

DEFAULT_KEYS = ("qwen3-30b", "qwen3.5-35b", "qwen3.5-397b", "deepseek-v4", "kimi-k3", "mixtral-8x7b")


class bucket_table:
    """Turn ``host._BUCKET_TILES`` off or on around a block, o5's pattern.

    Emptying the table is the whole of the v5-2 change, the same way emptying
    ``_SHAPE_TILES`` was the whole of o5's.
    """

    def __init__(self, enabled):
        self.enabled = enabled

    def __enter__(self):
        self.saved = dict(host._BUCKET_TILES)
        if not self.enabled:
            host._BUCKET_TILES.clear()
        host.compile_grouped_gemm.cache_clear()
        return self

    def __exit__(self, *exc):
        host._BUCKET_TILES.clear()
        host._BUCKET_TILES.update(self.saved)
        host.compile_grouped_gemm.cache_clear()
        return False


def geometry(model_dim, inter_dim, epr, tile_m):
    _, bm, bn, bk = host.compile_moe_gemm2(
        model_dim=model_dim, inter_dim=inter_dim, experts=epr, topk=1, tile_m=tile_m
    )
    return f"{bm}x{bn}x{bk}"


def run_shape(label, model_dim, inter_dim, epr, *, tokens, topk, tile_m, iters):
    _, down_key = table_keys(model_dim, inter_dim)
    where = f"D{model_dim}/I{inter_dim}, experts={epr}, topk={topk}, {tokens} tokens, tile_m={tile_m}"
    if down_key + (tile_m,) in host._SHAPE_TILES:
        section(f"{label}   {where}  [exact-shape override outranks the bucket, skipped]")
        return None

    rows_per_expert = tokens * topk / epr
    bucket = host.tile_bucket_for(rows_per_expert, tile_m)
    section(f"{label}   {where}   ({rows_per_expert:.0f} rows an expert, bucket {bucket})")

    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(0)
    x = (torch.randn(tokens, model_dim, generator=gen, device=dev, dtype=torch.bfloat16) * 0.1).contiguous()
    w1 = (
        torch.randn(epr, 2 * inter_dim, model_dim, generator=gen, device=dev, dtype=torch.bfloat16) * 0.05
    ).contiguous()
    w2 = (
        torch.randn(epr, model_dim, inter_dim, generator=gen, device=dev, dtype=torch.bfloat16) * 0.05
    ).contiguous()
    ids = torch.randint(0, epr, (tokens, topk), generator=gen, device=dev, dtype=torch.int32).contiguous()
    wts = torch.rand(tokens, topk, generator=gen, device=dev, dtype=torch.float32).contiguous()
    flop = layer_flops(tokens=tokens, model_dim=model_dim, inter_dim=inter_dim, topk=topk)

    arms = (("bucket off", False), ("bucket on", True))
    # Two passes, and only the second is reported. Whichever arm is timed first
    # absorbs the process's warm-up -- clock ramp, first-touch of the weights --
    # and on a small layer that was worth up to 10%, which is the same size as
    # the effect being measured. It showed up as the two arms disagreeing at a
    # shape where the table already holds the candidate's tile, i.e. where they
    # are the same kernel and the honest answer is zero.
    out, geos, diffs = {}, {}, {}
    ref = None
    for reported in (False, True):
        for name, enabled in arms:
            with bucket_table(enabled):
                geos[name] = geometry(model_dim, inter_dim, epr, tile_m)
                y = moe_forward(x, w1, w2, ids, wts, tile_m=tile_m)
                torch.cuda.synchronize()
                t = bench_us(lambda: moe_forward(x, w1, w2, ids, wts, tile_m=tile_m), warmup=5, iters=iters)
            if not reported:
                continue
            out[name] = t
            # Same values whichever tile computes them; the first one is the check.
            if ref is None:
                ref = y.clone()
                diffs[name] = 0.0
            else:
                diffs[name] = (y.float() - ref.float()).abs().max().item()

    print(f"  {'arm':>11} {'down tile':>14} {'layer':>10} {'TFLOP/s':>8} {'delta':>8}")
    for name, _ in arms:
        t = out[name]
        delta = "" if name == "bucket off" else f"{100 * (1 - t / out['bucket off']):+7.1f}%"
        print(
            f"  {name:>11} {geos[name]:>14} {t:9.1f}u {flop / (t * 1e-6) / 1e12:8.2f} "
            f"{delta:>8}  maxdiff {diffs[name]:.1e}"
        )

    del x, w1, w2, ids, wts
    torch.cuda.empty_cache()
    return label, tile_m, rows_per_expert, bucket, out["bucket off"], out["bucket on"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", nargs="*", default=list(DEFAULT_KEYS))
    p.add_argument("--world", type=int, default=8)
    p.add_argument("--rows", type=int, default=2048)
    p.add_argument("--tile-m", type=int, nargs="*", default=[64, 128])
    p.add_argument("--iters", type=int, default=100)
    p.add_argument(
        "--raw",
        action="append",
        default=None,
        metavar="D,I,E,topk,tokens",
        help="an arbitrary config instead of the model table, e.g. o3's 2048,768,8,2,1024",
    )
    args = p.parse_args()

    header(
        "v5-2b  a deeper BLOCK_K for down, on the whole layer",
        "--raw configs as given; otherwise one card of an EP group, local topk=1",
    )
    device_banner()

    if args.raw:
        cases = []
        for spec in args.raw:
            d, i, e, k, t = (int(v) for v in spec.split(","))
            cases.append((f"D{d}/I{i} E{e} k{k}", d, i, e, k, t))
    else:
        wanted = [s for s in SHAPES if s[0] in set(args.shapes)]
        if not wanted:
            raise SystemExit(f"no shape matched {args.shapes}; keys are {[s[0] for s in SHAPES]}")
        cases = [
            (label, md, idim, max(1, experts // args.world), 1, args.rows)
            for _k, label, md, idim, experts, _tk, _s in wanted
        ]

    results = []
    for label, model_dim, inter_dim, epr, topk, tokens in cases:
        for tile_m in args.tile_m:
            got = run_shape(
                label,
                model_dim,
                inter_dim,
                epr,
                tokens=tokens,
                topk=topk,
                tile_m=tile_m,
                iters=args.iters,
            )
            if got:
                results.append(got)

    section("summary: layer, _BUCKET_TILES off vs on")
    print(f"  {'model':>20} {'tile_m':>7} {'rows/exp':>9} {'bucket':>7} {'off':>10} {'on':>10} {'delta':>8}")
    for label, tile_m, rpe, bucket, base, cand in results:
        print(
            f"  {label:>20} {tile_m:7d} {rpe:9.0f} {bucket:7d} {base:9.1f}u {cand:9.1f}u "
            f"{100 * (1 - cand / base):+7.1f}%"
        )
    print()
    print("  Bucket 4 rows should read ~0%: nothing is entered there, because that is")
    print("  where the deeper BLOCK_K regresses. Buckets 1 and 2 at tile_m<=64 are the")
    print("  entries. A stage win shrinks by its share of the layer, so compare the")
    print("  delta against the run-to-run band in BENCHMARK.md §1 before believing it.")


if __name__ == "__main__":
    main()
