#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""v5-2f: does o6's dispatch order still pay when an expert owns one tile?

    python docs/rdna3_moe_optimization/repro/v5_m_major_ab.py
    python docs/rdna3_moe_optimization/repro/v5_m_major_ab.py --shapes kimi-k3
    python docs/rdna3_moe_optimization/repro/v5_m_major_ab.py --raw 4096,14336,8,2,1024

o6 put the M tiles innermost so that one expert's tiles are dispatched back to
back and its weight slab is read from memory once instead of once per tile. It
was worth 62.1 -> 69.8 TFLOP/s on ``gate_up`` at D4096/I14336, and
``_SHAPE_M_MAJOR`` names exactly that shape.

Two of o6's premises need rechecking on the v5 shapes, and they point opposite
ways:

  * The one that says *do* it. o6 excluded ``down`` at its own shape because
    ``down``'s ``k_dim`` is the intermediate size, so its A tile was 3.7 MB
    against ``gate_up``'s 1 MB -- holding one per M tile cost more than the B
    re-reads it saved. At I=512 that A tile is 64 KB. The reason for the
    exclusion is gone, so ``down`` may want M-innermost now. o6 also gated on
    whether the weight set fits cache, and every v5 shape misses: 0.20 GB at
    Qwen3.5-35B up to 7.40 GB at Kimi-K3, against a 96 MB MALL.

  * The one that says it *cannot matter*. The re-reads o6 is saving only exist
    if an expert owns more than one M tile. An EP card holds ``E / world``
    experts, so Kimi-K3 at EP=8 gives an expert 18 rows -- one tile at any
    ``tile_m`` the table offers. No second tile, no re-read, nothing for the
    order to fix. That is the same axis ``_BUCKET_TILES`` had to be keyed on,
    and ``_SHAPE_M_MAJOR`` is keyed on ``(stage, k_dim, n_out)``, which does not
    carry it either.

So the prediction going in is: no effect wherever the bucket is 1, whatever the
shape says. The ``bucket`` column is printed so that is falsifiable rather than
assumed, and ``--rows`` is there to walk a shape across the boundary.

Arms toggle one stage at a time, against an empty table rather than the shipped
one, so Mixtral's existing entries are part of what gets measured.
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

DEFAULT_KEYS = ("qwen3-30b", "mixtral-8x7b", "qwen3.5-35b", "qwen3.5-397b", "deepseek-v4", "kimi-k3")


class m_major:
    """Replace ``host._SHAPE_M_MAJOR`` for a block, o5's override pattern."""

    def __init__(self, keys):
        self.keys = set(keys)

    def __enter__(self):
        self.saved = set(host._SHAPE_M_MAJOR)
        host._SHAPE_M_MAJOR.clear()
        host._SHAPE_M_MAJOR.update(self.keys)
        host.compile_grouped_gemm.cache_clear()
        return self

    def __exit__(self, *exc):
        host._SHAPE_M_MAJOR.clear()
        host._SHAPE_M_MAJOR.update(self.saved)
        host.compile_grouped_gemm.cache_clear()
        return False


def run_shape(label, model_dim, inter_dim, epr, *, tokens, topk, tile_m, iters):
    g1_key, down_key = table_keys(model_dim, inter_dim)
    rows_per_expert = tokens * topk / epr
    bucket = host.tile_bucket_for(rows_per_expert, tile_m)
    weight_gb = epr * (2 * inter_dim * model_dim + model_dim * inter_dim) * 2 / 1e9
    slab_mb = 2 * inter_dim * model_dim * 2 / 1e6

    section(
        f"{label}   D{model_dim}/I{inter_dim}, experts={epr}, tile_m={tile_m}"
        f"   ({rows_per_expert:.0f} rows an expert, bucket {bucket})"
    )
    print(
        f"  weights {weight_gb:.2f} GB ({'misses' if weight_gb * 1e3 > 96 else 'fits'} the 96 MB MALL), "
        f"one gate_up slab {slab_mb:.1f} MB"
    )

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

    arms = (
        ("N innermost", ()),
        ("gate_up", (g1_key,)),
        ("down", (down_key,)),
        ("both", (g1_key, down_key)),
    )
    # Two passes, second one reported: whichever arm runs first absorbs the
    # process's warm-up, which on a small layer is worth as much as the effect.
    # See v5_tile_ab.py, where that bias produced a confident +9.9% between two
    # runs of the same kernel.
    out, diffs = {}, {}
    ref = None
    for reported in (False, True):
        for name, keys in arms:
            with m_major(keys):
                y = moe_forward(x, w1, w2, ids, wts, tile_m=tile_m)
                torch.cuda.synchronize()
                t = bench_us(lambda: moe_forward(x, w1, w2, ids, wts, tile_m=tile_m), warmup=5, iters=iters)
            if not reported:
                continue
            out[name] = t
            # The order changes which workgroup computes what, not what it
            # computes, so every arm owes the same bits.
            if ref is None:
                ref = y.clone()
                diffs[name] = 0.0
            else:
                diffs[name] = (y.float() - ref.float()).abs().max().item()

    base = out["N innermost"]
    print(f"  {'M innermost for':>16} {'layer':>10} {'TFLOP/s':>8} {'delta':>8}")
    for name, _ in arms:
        t = out[name]
        delta = "" if name == "N innermost" else f"{100 * (1 - t / base):+7.1f}%"
        print(
            f"  {name:>16} {t:9.1f}u {flop / (t * 1e-6) / 1e12:8.2f} {delta:>8}  maxdiff {diffs[name]:.1e}"
        )

    del x, w1, w2, ids, wts
    torch.cuda.empty_cache()
    return label, tile_m, bucket, base, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", nargs="*", default=list(DEFAULT_KEYS))
    p.add_argument("--world", type=int, default=8)
    p.add_argument("--rows", type=int, default=2048)
    p.add_argument("--tile-m", type=int, nargs="*", default=[32, 64])
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--raw", action="append", default=None, metavar="D,I,E,topk,tokens")
    args = p.parse_args()

    header(
        "v5-2f  o6's dispatch order, rechecked where an expert owns one tile",
        "M innermost per stage, against N innermost everywhere",
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
            results.append(
                run_shape(
                    label,
                    model_dim,
                    inter_dim,
                    epr,
                    tokens=tokens,
                    topk=topk,
                    tile_m=tile_m,
                    iters=args.iters,
                )
            )

    section("summary: delta from N innermost")
    print(f"  {'model':>20} {'tile_m':>7} {'bucket':>7} {'gate_up':>9} {'down':>9} {'both':>9}")
    for label, tile_m, bucket, base, out in results:
        cols = "".join(f"{100 * (1 - out[k] / base):+8.1f}%" for k in ("gate_up", "down", "both"))
        print(f"  {label:>20} {tile_m:7d} {bucket:7d}{'':>1}{cols}")
    print()
    print("  Prediction to check against: nothing should move at bucket 1, because the")
    print("  B re-reads the order saves only exist when an expert owns a second tile.")
    print("  Read any delta under ~4% as noise -- see BENCHMARK.md §13.6 for that floor.")


if __name__ == "__main__":
    main()
