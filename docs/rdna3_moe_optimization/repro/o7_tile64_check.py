#!/usr/bin/env python3
"""Scratch: does the 64-height config the o7 sweep found help a real layer?"""

from __future__ import annotations

import contextlib

import torch
from common import bench_us, device_banner, header, make_layer_inputs, section

from kernels.moe.rdna3_moe import host
from kernels.moe.rdna3_moe.forward import moe_forward

EXPERTS, TOPK = 8, 2
MODEL_DIM, INTER_DIM = 4096, 14336
NEW = {
    ("gate_up", MODEL_DIM, INTER_DIM, 64): ((2, 4, 2, 2, 2),),  # 64x128x32, vs 64x32 generic
    ("down", INTER_DIM, MODEL_DIM, 64): ((4, 4, 2, 1, 2),),  # 64x128x32, vs 64x32 generic
}


@contextlib.contextmanager
def overrides(on):
    saved = {k: host._SHAPE_TILES.get(k) for k in NEW}
    if on:
        host._SHAPE_TILES.update(NEW)
    host.compile_grouped_gemm.cache_clear()
    try:
        yield
    finally:
        for k, v in saved.items():
            host._SHAPE_TILES.pop(k, None) if v is None else host._SHAPE_TILES.update({k: v})
        host.compile_grouped_gemm.cache_clear()


def main():
    header("o7 byproduct  the 64 height's config at this shape", f"D{MODEL_DIM}/I{INTER_DIM}")
    device_banner()
    section("full layer at tile_m=64, generic config vs the swept one")
    print(f"  {'tokens':>7} {'generic 64x32':>14} {'swept 64x128':>14} {'delta':>8} {'tile_m=128':>12}")
    for tokens in (128, 256, 512, 1024):
        x, w1, w2, ids, wts = make_layer_inputs(
            tokens=tokens, model_dim=MODEL_DIM, inter_dim=INTER_DIM, experts=EXPERTS, topk=TOPK
        )
        got = {}
        for on in (False, True):
            with overrides(on):
                moe_forward(x, w1, w2, ids, wts, tile_m=64)
                torch.cuda.synchronize()
                got[on] = bench_us(lambda: moe_forward(x, w1, w2, ids, wts, tile_m=64), warmup=3, iters=20)
        t128 = bench_us(lambda: moe_forward(x, w1, w2, ids, wts, tile_m=128), warmup=3, iters=20)
        print(
            f"  {tokens:>7} {got[False] / 1000:13.3f}m {got[True] / 1000:13.3f}m "
            f"{100 * (got[False] / got[True] - 1):+7.1f}% {t128 / 1000:11.3f}m"
        )


if __name__ == "__main__":
    main()
