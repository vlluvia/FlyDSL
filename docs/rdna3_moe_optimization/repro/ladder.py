#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""The o1-o6 ladder: turn each optimisation on in order, on one shape, one run.

    python docs/rdna3_moe_optimization/repro/ladder.py

Every step here is a real toggle in the current code, so this is a genuine
cumulative measurement rather than a retelling:

    routing_impl        host -> device-exact (o1) -> device (o2)
    host._TILES         the pre-o3 table -> the current one (o3)
    host._SHAPE_TILES   empty -> the D4096/I14336 entries (o5)
    host._SHAPE_M_MAJOR empty -> the wide-stage1 entry (o6)

Two caveats worth reading before you trust the numbers:

  * **The pre-o3 table is a reconstruction.** o3 replaced one shared table with a
    stage-aware one, and the original is gone. What is used below is the
    before-column of o3's own write-up, which covers every cell these shapes
    touch. It reproduces the documented 128-tile collapse, so it is at least
    the right table.
  * **o4 is not in this ladder at all.** It is expert-parallel dispatch, which
    needs two ranks and is not part of a single-card layer. See
    o4_ep_dispatch.py.

o5 and o6 only have entries for D4096/I14336, so on any other shape those two
rows are no-ops by construction, and the script says so rather than pretending.
"""

from __future__ import annotations

import importlib.util
import os

from common import REPO, bench_us, device_banner, header, layer_flops, make_layer_inputs, section
from kernels.moe.rdna3_moe import host
from kernels.moe.rdna3_moe.forward import moe_forward

# Use the benchmark script's own torch baseline, so these numbers can be put
# next to the ones scripts/bench_rdna3_moe.py prints.
_spec = importlib.util.spec_from_file_location("_bench_ref", os.path.join(REPO, "scripts", "bench_rdna3_moe.py"))
_bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bench)
torch_layer = _bench.torch_layer

# Before o3 there was a single table for every stage: the dense GEMM's autotuned
# tiles. These five cells are the "original tile" column of o3's write-up.
PRE_O3 = {
    16: (1, 2, 4, 1, 2),  # 16x64x64
    32: (2, 2, 4, 1, 2),  # 32x64x64
    64: (2, 2, 4, 2, 2),  # 64x64x64
    128: (4, 4, 2, 2, 2),  # 128x128x32  <- the one that spills 769 times
}

# (label, routing_impl, pre_o3_tiles, shape_override, m_major)
STEPS = [
    ("v3 (起点)", "host", True, False, False),
    ("+o1 routing 核", "device-exact", True, False, False),
    ("+o2 免同步", "device", True, False, False),
    ("+o3 tile 表", "device", False, False, False),
    ("+o5 shape override", "device", False, True, False),
    ("+o6 派发顺序", "device", False, True, True),
]

SHAPES = [
    # (model_dim, inter_dim, experts, topk, tokens, tile_m, label)
    (2048, 768, 8, 2, 32, 16, "D2048/I768 decode"),
    (2048, 768, 8, 2, 1024, 128, "D2048/I768 prefill"),
    (4096, 14336, 8, 2, 1024, 128, "D4096/I14336 prefill"),
]


class configured:
    """Put the module globals into the state a given rung of the ladder had."""

    def __init__(self, *, pre_o3, override, m_major):
        self.pre_o3, self.override, self.m_major = pre_o3, override, m_major

    def __enter__(self):
        self.saved = (host._TILES, host._SHAPE_TILES, host._SHAPE_M_MAJOR)
        if self.pre_o3:
            # Keep the current lists as fallbacks so an odd n_out still builds;
            # the old tile is first, so it is what gets used when it fits.
            host._TILES = {
                stage: {tm: (PRE_O3[tm],) + cfgs for tm, cfgs in per_tile.items()}
                for stage, per_tile in host._TILES.items()
            }
        if not self.override:
            host._SHAPE_TILES = {}
        if not self.m_major:
            host._SHAPE_M_MAJOR = set()
        host.compile_grouped_gemm.cache_clear()
        return self

    def __exit__(self, *exc):
        host._TILES, host._SHAPE_TILES, host._SHAPE_M_MAJOR = self.saved
        host.compile_grouped_gemm.cache_clear()
        return False


def geometry(model_dim, inter_dim, experts, topk, tile_m):
    g1, bm1, bn1, bk1 = host.compile_moe_gemm1(
        model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk, tile_m=tile_m
    )
    g2, bm2, bn2, bk2 = host.compile_moe_gemm2(
        model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk, tile_m=tile_m
    )
    return f"{bn1}x{bk1}/{bn2}x{bk2}", g1.m_major


def run_shape(model_dim, inter_dim, experts, topk, tokens, tile_m, label):
    section(f"{label}   E={experts} topk={topk} tokens={tokens} tile_m={tile_m}")
    x, w1, w2, ids, wts = make_layer_inputs(
        tokens=tokens, model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk
    )
    flop = layer_flops(tokens=tokens, model_dim=model_dim, inter_dim=inter_dim, topk=topk)
    iters = 10 if tokens >= 1024 and inter_dim > 4096 else 30

    t_torch = bench_us(lambda: torch_layer(x, w1, w2, ids, wts), warmup=3, iters=iters)
    print(f"  torch (eager, 每专家一次 matmul): {t_torch / 1000:8.3f} ms  {flop / (t_torch * 1e-6) / 1e12:6.2f} TFLOP/s")
    print()
    print(f"  {'配置':<22} {'gate_up/down 块':>16} {'M内层':>6} {'layer+route':>12} {'TFLOP/s':>8} "
          f"{'vs torch':>9} {'累积':>7}")

    first = None
    for lbl, impl, pre_o3, override, m_major in STEPS:
        with configured(pre_o3=pre_o3, override=override, m_major=m_major):
            geo, mm = geometry(model_dim, inter_dim, experts, topk, tile_m)
            moe_forward(x, w1, w2, ids, wts, tile_m=tile_m, routing_impl=impl)
            us = bench_us(
                lambda impl=impl: moe_forward(x, w1, w2, ids, wts, tile_m=tile_m, routing_impl=impl),
                warmup=3,
                iters=iters,
            )
        first = first or us
        print(
            f"  {lbl:<22} {geo:>16} {str(mm):>6} {us / 1000:11.3f}m {flop / (us * 1e-6) / 1e12:8.2f} "
            f"{t_torch / us:8.2f}x {first / us:6.2f}x"
        )
    return first


def main():
    header("o1-o6 累积阶梯", "每一级都是当前代码里真实存在的开关，同一次运行内测完")
    device_banner()
    print("  layer+route = 一层 GEMM + 每次调用重建 routing，也就是一个 decode step 真正付的钱")
    print("  TFLOP/s 只算有用的 FLOP（不给 padding 行记功）")

    for shape in SHAPES:
        run_shape(*shape)

    section("读这张表要注意的")
    print("  o1/o2 只动 routing，不改 GEMM，所以它们在 layer+route 上有效、在纯 GEMM 上是 0。")
    print("  o3/o5/o6 只动 GEMM 的 tile 与派发，对 routing 没有影响。")
    print("  o5/o6 的表里只有 D4096/I14336 的条目，所以在别的 shape 上这两级按定义就是空操作。")
    print("  o4 是 EP dispatch，需要两张卡，不在这张表里 —— 见 o4_ep_dispatch.py。")
    print()
    print("  pre-o3 的 tile 表是重建的（见本文件开头）。它复现出了文档里记的 128 tile")
    print("  塌陷，所以至少是对的那张表，但不要把它当成逐位还原的历史代码。")


if __name__ == "__main__":
    main()
