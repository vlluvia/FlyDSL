#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Sweep block tiles for one RDNA3 MoE stage, and say what limits each one.

``host.py`` carries one tile per ``tile_m``, taken from the dense kernel's
autotune. That table is wrong for ``gate_up``: a fused gate/up workgroup carries
*two* accumulator sets over one A tile, so at the same block shape it needs twice
the registers, and the dense kernel's 128x128x32 tile spills 769 times on it.
This is how the replacements were picked.

Three numbers explain most of the results, and they are printed next to each:

  ``LDS``    bytes per workgroup. gfx11 gives a workgroup up to 64 KB and a CU
             64 KB total, so anything over 32 KB means one workgroup per CU.
  ``waves``  waves resident per SIMD, if LDS is what limits it. The current
             table sits at 2 of a possible 16, which is very little to hide
             memory latency with.
  ``accs``   accumulator vectors per thread, eight VGPRs each. Past about 16 the
             kernel is near the 256-VGPR cap and starts spilling, at which point
             nothing else about the tile matters.

    python scripts/sweep_rdna3_moe_tiles.py --stage gate_up --tokens 1024
    python scripts/sweep_rdna3_moe_tiles.py --stage down --tokens 32 --tile-m 16,32

``accs`` is a floor on the register count, not the count itself, so when a tile
is slow for no visible reason the next thing to read is the ISA:

    FLYDSL_RUNTIME_ENABLE_CACHE=0 FLYDSL_DUMP_IR=1 FLYDSL_DEBUG_DUMP_ASM=1 \\
      FLYDSL_DUMP_DIR=/tmp/isa python scripts/sweep_rdna3_moe_tiles.py ...
    grep -E 'vgpr_count|vgpr_spill_count' /tmp/isa/grouped_gemm_kernel_0/*_isa.s

which is how the 128x128x32 gate_up tile was caught: 256 VGPRs, the cap, and 769
spills. Its replacement uses 201 and spills none.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_BUILD = os.path.join(_REPO, "build-fly", "python_packages")
if os.path.isdir(_BUILD) and _BUILD not in sys.path:
    sys.path.insert(0, _BUILD)

import torch  # noqa: E402

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.gemm.rdna3_f16_gemm import WMMA_M, WMMA_N  # noqa: E402
from kernels.moe.rdna3_moe.grouped_gemm import create_grouped_gemm_module  # noqa: E402
from kernels.moe.rdna3_moe.routing_kernel import build_routing_device  # noqa: E402

LDS_PER_CU = 64 * 1024
MAX_WAVES_PER_SIMD = 16
SIMDS_PER_CU = 2

REG_M = (1, 2, 4)
REG_N = (1, 2, 4)
REG_K = (2, 4)
WAVES_M = (1, 2, 4)
WAVES_N = (1, 2, 4)


def bench_us(fn, *, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters


def waves_per_simd(lds_bytes, threads):
    """Waves per SIMD once LDS has decided how many workgroups fit."""
    per_cu = max(1, LDS_PER_CU // max(lds_bytes, 1))
    return min(MAX_WAVES_PER_SIMD, per_cu * (threads // 32) / SIMDS_PER_CU)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="gate_up", choices=("linear", "gate_up", "down"))
    p.add_argument("--tokens", type=int, default=1024)
    p.add_argument("--model-dim", type=int, default=2048)
    p.add_argument("--inter-dim", type=int, default=768)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--tile-m", default="16,32,64,128")
    p.add_argument("--m-major", action="store_true", help="dispatch M tiles innermost")
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()

    arch = str(get_rocm_arch() or "")
    if not arch.startswith("gfx11"):
        print(f"ERROR: needs gfx11*, got {arch!r}")
        return 1

    stage, tokens, topk, E = args.stage, args.tokens, args.topk, args.experts
    md, idim = args.model_dim, args.inter_dim
    want_tiles = {int(t) for t in args.tile_m.split(",") if t.strip()}
    # stage1 contracts model_dim into inter_dim; stage2 the other way round.
    k_dim, n_out = (md, idim) if stage in ("linear", "gate_up") else (idim, md)
    rows = tokens * topk

    dev = "cuda"
    torch.manual_seed(0)
    a_rows = rows if stage == "down" else tokens
    a = (torch.randn(a_rows, k_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    w_rows = (2 if stage == "gate_up" else 1) * n_out
    w = (torch.randn(E, w_rows, k_dim, dtype=torch.bfloat16, device=dev) * 0.05).contiguous()
    topk_ids = torch.randint(0, E, (tokens, topk), dtype=torch.int32, device=dev)
    weights = torch.rand(tokens, topk, dtype=torch.float32, device=dev)
    stream = torch.cuda.current_stream()
    flop = 2.0 * rows * w_rows * k_dim

    print(f"device={torch.cuda.get_device_name(0)} arch={arch} stage={stage}")
    print(f"tokens={tokens} rows={rows} k_dim={k_dim} n_out={n_out} experts={E} topk={topk}")
    print(f"\n{'tile':>16} {'thr':>4} {'accs':>5} {'LDS KB':>7} {'waves':>6} {'us':>8} {'TFLOP/s':>8}  {'ref':>8}")
    print("-" * 78)

    ref = None
    results = []
    for reg_m, reg_n, reg_k, waves_m, waves_n in itertools.product(REG_M, REG_N, REG_K, WAVES_M, WAVES_N):
        block_m = WMMA_M * reg_m * waves_m
        if block_m not in want_tiles:
            continue
        if WMMA_N * reg_n * waves_n > n_out:
            continue
        try:
            launch, bm, bn, bk = create_grouped_gemm_module(
                k_dim=k_dim,
                n_out=n_out,
                experts=E,
                stage=stage,
                topk=topk,
                doweight=(stage == "down"),
                reg_m=reg_m,
                reg_n=reg_n,
                reg_k=reg_k,
                waves_m=waves_m,
                waves_n=waves_n,
                m_major=args.m_major,
            )
        except (ValueError, RuntimeError) as exc:
            print(f"{f'{reg_m},{reg_n},{reg_k},{waves_m},{waves_n}':>16} skipped: {str(exc)[:52]}")
            continue

        r = build_routing_device(topk_ids, experts=E, tile_m=bm, exact=True, reuse=False)
        out_shape = (r.rows, n_out) if stage == "linear" else (tokens, topk, n_out)
        c = torch.zeros(*out_shape, dtype=torch.bfloat16, device=dev)
        wt = weights if stage == "down" else torch.empty(0, dtype=torch.float32, device=dev)

        def run(launch=launch, c=c, r=r, wt=wt):
            launch(c, a, w, r.sorted_ids, r.expert_ids, wt, tokens, r.num_blocks, stream)

        run()
        torch.cuda.synchronize()
        # Every tile computes the same values into the same [token, slot] layout,
        # so the first one that builds is the reference for the rest. ``linear``
        # is excluded: its output is in padded routing order, which the tile
        # changes.
        if stage != "linear":
            if ref is None:
                ref = c.clone()
                diff = 0.0
            else:
                diff = (c.float() - ref.float()).abs().max().item()
        else:
            diff = float("nan")

        us = bench_us(run, iters=args.iters)
        tf = flop / (us * 1e-6) / 1e12
        results.append((tf, bm, bn, bk, (reg_m, reg_n, reg_k, waves_m, waves_n), us, launch))
        print(
            f"{f'{reg_m},{reg_n},{reg_k},{waves_m},{waves_n}':>16} {launch.threads:4d} "
            f"{launch.acc_vectors:5d} {launch.lds_bytes / 1024:7.1f} "
            f"{waves_per_simd(launch.lds_bytes, launch.threads):6.1f} {us:8.1f} {tf:8.2f}  "
            f"{'' if diff != diff else f'{diff:.1e}'}"
        )

    print("-" * 78)
    print("best per tile_m:")
    for tile in sorted(want_tiles):
        got = [r for r in results if r[1] == tile]
        if not got:
            continue
        tf, bm, bn, bk, cfg, us, launch = max(got)
        print(
            f"  tile_m={tile:3d}  {bm}x{bn}x{bk}  reg/waves={cfg}  "
            f"{us:.1f} us  {tf:.2f} TFLOP/s  LDS {launch.lds_bytes / 1024:.1f} KB  accs {launch.acc_vectors}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
