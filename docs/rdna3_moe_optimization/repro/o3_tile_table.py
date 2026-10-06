#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""o3: why tile_m=128 collapsed, and why the first explanation was wrong.

    python docs/rdna3_moe_optimization/repro/o3_tile_table.py

The symptom was a 6x cliff: gate_up at tile_m=128 ran at a fraction of tile_m=64
on the same shape. The comment in host.py blamed LDS and occupancy. This script
reproduces the measurement, then reproduces the arithmetic that shows the LDS
story cannot be true, then shows the number that does explain it.

The proof is in the ISA, which needs environment variables set before the
process starts -- see o3_isa_dump.py and the command in the README.

Two independent levers came out of this phase:
  A. accumulators -- buy BLOCK_N from *waves*, not from registers
  B. LDS -- halve BLOCK_K so a second workgroup fits the CU
"""

from __future__ import annotations

import torch

from common import bench_us, device_banner, header, section
from kernels.gemm.rdna3_f16_gemm import WMMA_K, WMMA_M, WMMA_N
from kernels.moe.rdna3_moe.grouped_gemm import create_grouped_gemm_module
from kernels.moe.rdna3_moe.routing_kernel import build_routing_device

MODEL_DIM, INTER_DIM, EXPERTS, TOPK, TOKENS = 2048, 768, 8, 2, 1024
LDS_PER_CU = 64 * 1024
MAX_WAVES_PER_SIMD = 16
SIMDS_PER_CU = 2


def waves_per_simd(lds_bytes, threads):
    """Waves resident per SIMD, once LDS has decided how many workgroups fit."""
    per_cu = max(1, LDS_PER_CU // max(lds_bytes, 1))
    return min(MAX_WAVES_PER_SIMD, per_cu * (threads // 32) / SIMDS_PER_CU)


def build_and_time(stage, cfg, *, tokens=TOKENS, model_dim=MODEL_DIM, inter_dim=INTER_DIM):
    """Build one explicit tile and time it on a real routing."""
    reg_m, reg_n, reg_k, waves_m, waves_n = cfg
    k_dim, n_out = (model_dim, inter_dim) if stage in ("linear", "gate_up") else (inter_dim, model_dim)
    rows = tokens * TOPK
    launch, bm, bn, bk = create_grouped_gemm_module(
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
    )

    dev = "cuda"
    torch.manual_seed(0)
    a_rows = rows if stage == "down" else tokens
    a = (torch.randn(a_rows, k_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    w_rows = (2 if stage == "gate_up" else 1) * n_out
    w = (torch.randn(EXPERTS, w_rows, k_dim, dtype=torch.bfloat16, device=dev) * 0.05).contiguous()
    ids = torch.randint(0, EXPERTS, (tokens, TOPK), dtype=torch.int32, device=dev)
    wts = torch.rand(tokens, TOPK, dtype=torch.float32, device=dev)
    r = build_routing_device(ids, experts=EXPERTS, tile_m=bm, exact=True, reuse=False)
    c = torch.zeros(tokens, TOPK, n_out, dtype=torch.bfloat16, device=dev)
    empty = torch.empty(0, dtype=torch.float32, device=dev)
    stream = torch.cuda.current_stream()

    def run():
        launch(c, a, w, r.sorted_ids, r.expert_ids, wts if stage == "down" else empty, tokens, r.num_blocks, stream)

    run()
    torch.cuda.synchronize()
    us = bench_us(run, iters=30)
    tf = 2.0 * rows * w_rows * k_dim / (us * 1e-6) / 1e12
    return dict(
        cfg=cfg,
        shape=f"{bm}x{bn}x{bk}",
        threads=launch.threads,
        accs=launch.acc_vectors,
        lds=launch.lds_bytes,
        waves=waves_per_simd(launch.lds_bytes, launch.threads),
        us=us,
        tf=tf,
    )


def show(rows, note=""):
    print(f"  {'reg/waves':>13} {'block':>14} {'thr':>4} {'accs':>5} {'LDS KB':>7} {'waves':>6} {'us':>9} {'TFLOP/s':>8}")
    for r in rows:
        print(
            f"  {','.join(str(v) for v in r['cfg']):>13} {r['shape']:>14} {r['threads']:>4} {r['accs']:>5} "
            f"{r['lds'] / 1024:7.1f} {r['waves']:6.1f} {r['us']:9.1f} {r['tf']:8.2f}"
        )
    if note:
        print(f"  {note}")


def main():
    header("o3  the tile table, and a wrong diagnosis", f"D{MODEL_DIM}/I{INTER_DIM}, {TOKENS} tokens, E={EXPERTS}")
    device_banner()
    print(f"  WMMA atom {WMMA_M}x{WMMA_N}x{WMMA_K};  BLOCK_M=16*reg_m*waves_m, BLOCK_N=16*reg_n*waves_n, BLOCK_K=16*reg_k")

    # ── 1. The cliff ─────────────────────────────────────────────────────
    section("1. symptom: the dense kernel's 128 tile falls off a cliff on gate_up")
    dense_128 = build_and_time("gate_up", (4, 4, 2, 2, 2))  # 128x128x32, what the dense autotune picked
    dense_64 = build_and_time("gate_up", (2, 2, 4, 2, 2))  # 64x64x64
    show([dense_64, dense_128], note=f"-> the 128 tile is {dense_64['tf'] / dense_128['tf']:.1f}x SLOWER than the 64 one")

    # ── 2. The explanation that cannot be true ───────────────────────────
    section("2. the LDS story does not survive its own arithmetic")
    print(f"  gfx11 gives a workgroup 64 KB of LDS and a CU 64 KB in total, so any")
    print(f"  tile over 32 KB means one workgroup per CU. Both of these are over:")
    print(f"    64  tile: {dense_64['lds'] / 1024:5.1f} KB -> {dense_64['waves']:.0f} waves/SIMD")
    print(f"    128 tile: {dense_128['lds'] / 1024:5.1f} KB -> {dense_128['waves']:.0f} waves/SIMD")
    print(f"  Same occupancy. Occupancy cannot explain a {dense_64['tf'] / dense_128['tf']:.1f}x difference.")

    # ── 3. The number that does explain it ───────────────────────────────
    section("3. accumulators: gate_up carries one set per B stream")
    print("  A fused gate/up workgroup runs two B streams over one A tile, so it")
    print("  holds 2*reg_m*reg_n accumulator vectors of 8 VGPRs each. gfx11 caps a")
    print("  thread at 256 VGPRs.")
    print()
    for r in (dense_64, dense_128):
        print(
            f"    {','.join(str(v) for v in r['cfg']):>13}  accs={r['accs']:2d} "
            f"-> {8 * r['accs']:3d} VGPRs of accumulator alone"
            + ("   <- at the cap, before anything else" if 8 * r["accs"] >= 256 else "")
        )
    print()
    print("  Prove it in the ISA (see README / o3_isa_dump.py):")
    print("    grep -E 'vgpr_count|vgpr_spill_count' .../21_final_isa.s")
    print("  The 128x128x32 tile reports vgpr_spill_count: 769. The fix reports 0.")

    # ── 4. Lever A: buy N from waves ─────────────────────────────────────
    section("4. lever A: same BLOCK_N, a quarter of the accumulators")
    print("  reg_n=1 with waves_n=2 covers the same N as reg_n=2 with waves_n=1,")
    print("  but N bought from waves costs no accumulators at all.")
    print()
    fixed_128 = build_and_time("gate_up", (4, 1, 2, 2, 2))  # 128x32x32, what the table holds now
    mid_128 = build_and_time("gate_up", (4, 2, 2, 2, 2))  # 128x64x32, what o5 picks for the wide shape
    show([dense_128, fixed_128, mid_128])
    print(f"  -> {dense_128['tf']:.1f} to {fixed_128['tf']:.1f} TFLOP/s, same tile_m, same LDS budget class")

    # ── 5. Lever B: BLOCK_K and the second workgroup ─────────────────────
    section("5. lever B: halve BLOCK_K, fit a second workgroup per CU")
    print("  The pipeline double-buffers, so BLOCK_K=64 puts a stage over 32 KB and")
    print("  only one workgroup fits. BLOCK_K=32 halves the buffer and admits two.")
    print()
    down_old = build_and_time("down", (2, 2, 4, 2, 2))  # 64x64x64
    down_new = build_and_time("down", (2, 2, 2, 2, 4))  # 64x128x32, the table's choice
    show([down_old, down_new])
    print("  -> this is where most of the single-B stages' gain came from")

    # ── 6. The bug the sweep found ───────────────────────────────────────
    section("6. a build-time rejection that came out of this phase")
    print("  reg_m=reg_n=1 on a single-B stage leaves exactly one accumulator. The")
    print("  K loop's carried state then comes back as a bare value instead of a")
    print("  length-1 sequence, and state[:n_acc] raises TypeError. Rather than")
    print("  normalising that corner, the builder now refuses the tile -- one WMMA")
    print("  per thread cannot fill the pipeline anyway.")
    print()
    try:
        create_grouped_gemm_module(
            k_dim=MODEL_DIM, n_out=INTER_DIM, experts=EXPERTS, stage="linear", topk=TOPK,
            reg_m=1, reg_n=1, reg_k=2, waves_m=1, waves_n=1,
        )
        print("  UNEXPECTED: the builder accepted a single-accumulator tile")
    except ValueError as exc:
        print(f"  builder refuses it: {exc}")


if __name__ == "__main__":
    main()
