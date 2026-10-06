#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""v5-2a: before sweeping tiles, find out whether the gap is even in the GEMMs.

    python docs/rdna3_moe_optimization/repro/v5_stage_gap.py
    python docs/rdna3_moe_optimization/repro/v5_stage_gap.py --shapes qwen3.5-35b

``v5_shape_coverage.py`` found two shapes with real headroom: Qwen3-30B-A3B at
45% of a mac-bound floor and Qwen3.5-35B-A3B at 50% of a mem-bound one. The
obvious next move is to sweep block tiles at those shapes. This script exists to
stop that being the *first* move.

The reason is arithmetic. Qwen3.5-35B's layer is 574 us against a 285 us floor,
and the layer is five launches -- so the 290 us gap is 58 us a launch, which is
the shape of a fixed per-launch cost, not of a badly shaped tile. And there is a
known candidate: ``epr`` is 32 here, and o1 measured its routing kernel at 52 us
for E=32 (its comment says the design "runs out" past a few tens of experts).
If most of the gap is routing, a tile sweep would spend a day moving 1%.

So this times each launch on its own, against its own floor, and prints what
share of the layer each one is. o7/o8 are the standing warning that summing
components misleads -- the sum here does not have to match the assembled layer,
and the assembled number is printed alongside precisely so the discrepancy is
visible rather than assumed away.

Configured like ``v5_shape_coverage.py``: one card of an EP=8 group, local
topk=1, ``--rows`` arriving rows.
"""

from __future__ import annotations

import argparse

import torch

# common must be imported before anything under kernels/: it is what puts the
# repo root on sys.path. Keep these as from-imports so the sort order holds.
from common import bench_us, device_banner, header, section
from kernels.moe.rdna3_moe import host
from kernels.moe.rdna3_moe.forward import moe_forward, moe_reduce
from kernels.moe.rdna3_moe.routing_kernel import build_routing_device
from v5_shape_coverage import HBM_GBPS, SHAPES, WMMA_TFLOPS, stage_floor

# The two v5_shape_coverage found headroom at. Others can be asked for by key.
DEFAULT_KEYS = ("qwen3-30b", "qwen3.5-35b")


def floor_us(*, counts, tile_m, k_dim, n_out):
    nbytes, macs = stage_floor(counts=counts, tile_m=tile_m, k_dim=k_dim, n_out=n_out)
    mem = nbytes / (HBM_GBPS * 1e9) * 1e6
    mac = macs / (WMMA_TFLOPS * 1e12) * 1e6
    return (max(mem, mac), "mem" if mem >= mac else "mac")


def run_shape(label, model_dim, inter_dim, experts, source, *, world, rows, tile_m, iters):
    epr = max(1, experts // world)
    section(f"{label}   D{model_dim}/I{inter_dim}, epr={epr}, tile_m={tile_m}")

    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(0)
    x = (torch.randn(rows, model_dim, generator=gen, device=dev, dtype=torch.bfloat16) * 0.1).contiguous()
    w1 = (
        torch.randn(epr, 2 * inter_dim, model_dim, generator=gen, device=dev, dtype=torch.bfloat16) * 0.05
    ).contiguous()
    w2 = (
        torch.randn(epr, model_dim, inter_dim, generator=gen, device=dev, dtype=torch.bfloat16) * 0.05
    ).contiguous()
    ids = torch.randint(0, epr, (rows, 1), generator=gen, device=dev, dtype=torch.int32).contiguous()
    wts = torch.rand(rows, 1, generator=gen, device=dev, dtype=torch.float32).contiguous()
    counts = torch.bincount(ids.reshape(-1).long(), minlength=epr)

    gemm1, *_ = host.compile_moe_gemm1(
        model_dim=model_dim, inter_dim=inter_dim, experts=epr, topk=1, tile_m=tile_m, bounded_blocks=True
    )
    gemm2, *_ = host.compile_moe_gemm2(
        model_dim=model_dim, inter_dim=inter_dim, experts=epr, topk=1, tile_m=tile_m, bounded_blocks=True
    )

    stream = torch.cuda.current_stream()
    routing = build_routing_device(ids, experts=epr, tile_m=tile_m, reuse=False)
    a2 = torch.empty(rows, 1, inter_dim, dtype=x.dtype, device=dev)
    y = torch.empty(rows, 1, model_dim, dtype=x.dtype, device=dev)
    unused = torch.empty(0, dtype=torch.float32, device=dev)
    nb, nbd = routing.num_blocks, routing.num_blocks_device

    def do_routing():
        build_routing_device(ids, experts=epr, tile_m=tile_m, reuse=True)

    def do_gemm1():
        gemm1(a2, x, w1, routing.sorted_ids, routing.expert_ids, unused, rows, nb, stream, nbd)

    def do_gemm2():
        gemm2(y, a2, w2, routing.sorted_ids, routing.expert_ids, wts, rows, nb, stream, nbd)

    def do_reduce():
        moe_reduce(y, stream=stream)

    stages = [
        ("routing", do_routing, None),
        ("gemm1 gate_up", do_gemm1, (model_dim, 2 * inter_dim)),
        ("gemm2 down", do_gemm2, (inter_dim, model_dim)),
        ("reduce", do_reduce, None),
    ]

    print(f"  {'stage':>14} {'solo':>9} {'floor':>9} {'bound':>6} {'of floor':>9} {'gap':>9} {'share':>7}")
    solo, gaps = {}, {}
    for name, fn, dims in stages:
        t = bench_us(fn, warmup=5, iters=iters)
        solo[name] = t
        if dims is None:
            print(f"  {name:>14} {t:8.1f}u {'-':>9} {'-':>6} {'-':>9} {t:8.1f}u")
            gaps[name] = t
            continue
        fl, bound = floor_us(counts=counts, tile_m=tile_m, k_dim=dims[0], n_out=dims[1])
        gaps[name] = t - fl
        print(f"  {name:>14} {t:8.1f}u {fl:8.1f}u {bound:>6} {100 * fl / t:8.0f}% {t - fl:8.1f}u")

    total_solo = sum(solo.values())
    assembled = bench_us(lambda: moe_forward(x, w1, w2, ids, wts, tile_m=tile_m), warmup=5, iters=iters)
    total_gap = sum(gaps.values())

    print()
    print(f"  {'sum of the four':>14} {total_solo:8.1f}u")
    print(f"  {'assembled layer':>14} {assembled:8.1f}u   ({assembled - total_solo:+.1f}u vs the sum)")
    print()
    print("  where the gap is, as a share of it:")
    for name, g in sorted(gaps.items(), key=lambda kv: -kv[1]):
        print(f"    {name:>14} {g:8.1f}u {100 * g / total_gap:5.0f}%")

    del x, w1, w2, ids, wts, a2, y
    torch.cuda.empty_cache()
    return label, epr, assembled, gaps, total_gap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", nargs="*", default=list(DEFAULT_KEYS))
    p.add_argument("--world", type=int, default=8)
    p.add_argument("--rows", type=int, default=2048)
    p.add_argument("--tile-m", type=int, default=64, help="the tile v5_shape_coverage found best")
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    header(
        "v5-2a  is the gap in the GEMMs at all?",
        f"{args.rows} arriving rows, one card of an EP={args.world} group, local topk=1",
    )
    device_banner()

    wanted = [s for s in SHAPES if s[0] in set(args.shapes)]
    if not wanted:
        raise SystemExit(f"no shape matched {args.shapes}; keys are {[s[0] for s in SHAPES]}")

    rows_out = []
    for key, label, model_dim, inter_dim, experts, _topk, source in wanted:
        rows_out.append(
            run_shape(
                label,
                model_dim,
                inter_dim,
                experts,
                source,
                world=args.world,
                rows=args.rows,
                tile_m=args.tile_m,
                iters=args.iters,
            )
        )

    section("verdict")
    print(f"  {'model':>20} {'epr':>5} {'layer':>9} {'routing':>9} {'of gap':>7} {'GEMM gap':>9} {'of gap':>7}")
    for label, epr, assembled, gaps, total_gap in rows_out:
        gemm = gaps["gemm1 gate_up"] + gaps["gemm2 down"]
        print(
            f"  {label:>20} {epr:5d} {assembled:8.1f}u {gaps['routing']:8.1f}u "
            f"{100 * gaps['routing'] / total_gap:6.0f}% {gemm:8.1f}u {100 * gemm / total_gap:6.0f}%"
        )
    print()
    print("  If routing owns most of the gap, v5-2's tile sweep is the wrong lever and")
    print("  the work belongs in v5-3 (the routing kernel at epr in the tens).")


if __name__ == "__main__":
    main()
