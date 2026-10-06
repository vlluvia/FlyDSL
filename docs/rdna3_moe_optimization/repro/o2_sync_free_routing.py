#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""o2: stop reading the tile count back to the host.

    python docs/rdna3_moe_optimization/repro/o2_sync_free_routing.py

o1 left the routing kernel at ~10us with a ~30us host round trip bolted to it,
because the GEMM grid needed the exact tile count. o2's observation is that the
grid does not need to be exact -- an upper bound computable from the row count
alone works, if the kernel skips the tiles past the real end.

The three ``routing_impl`` values are all still in the code, so the whole phase
is a three-way A/B you can run:

    host          the original numpy builder
    device-exact  o1: kernel + readback
    device        o2: kernel, bounded grid, no readback

Part 3 shows why the padded tiles were already safe to run, which is the reason
the early exit could be an optimisation rather than a correctness fix.
"""

from __future__ import annotations

import torch

from common import bench_us, device_banner, header, make_layer_inputs, section
from kernels.moe.rdna3_moe.forward import moe_forward
from kernels.moe.rdna3_moe.routing_kernel import build_routing_device, max_blocks_for

MODEL_DIM, INTER_DIM, EXPERTS, TOPK = 2048, 768, 8, 2
IMPLS = ("host", "device-exact", "device")


def main():
    header("o2  a bounded grid instead of a synchronising one", "max_blocks_for + bounded_blocks early exit")
    device_banner()

    # ── 1. The bound ─────────────────────────────────────────────────────
    section("1. the grid does not need the exact count, only a bound")
    print("  ceil(c/t) < c/t + 1 per expert, and an expert with no rows still gets")
    print("  its one tile -- so experts + ceil(rows/tile_m) bounds the sum, and it")
    print("  is computable before the gating has even finished.")
    print()
    print(f"  {'tokens':>7} {'tile_m':>7} {'bound':>7} {'actual':>7} {'wasted tiles':>13}")
    for tokens, tile_m in [(32, 16), (256, 16), (1024, 64), (1024, 128)]:
        _, _, _, ids, _ = make_layer_inputs(
            tokens=tokens, model_dim=MODEL_DIM, inter_dim=INTER_DIM, experts=EXPERTS, topk=TOPK
        )
        bound = max_blocks_for(tokens=tokens, topk=TOPK, experts=EXPERTS, tile_m=tile_m)
        exact = build_routing_device(ids, experts=EXPERTS, tile_m=tile_m, exact=True, reuse=False).num_blocks
        print(f"  {tokens:>7} {tile_m:>7} {bound:>7} {exact:>7} {bound - exact:>13}")

    # ── 2. The three-way comparison ──────────────────────────────────────
    section("2. what each routing_impl costs the layer")
    for tokens, tile_m in [(32, 16), (1024, 64)]:
        x, w1, w2, ids, wts = make_layer_inputs(
            tokens=tokens, model_dim=MODEL_DIM, inter_dim=INTER_DIM, experts=EXPERTS, topk=TOPK
        )
        r_pre = build_routing_device(ids, experts=EXPERTS, tile_m=tile_m, exact=True, reuse=False)
        bare = bench_us(lambda: moe_forward(x, w1, w2, ids, wts, tile_m=tile_m, routing=r_pre))
        print()
        print(f"  tokens={tokens} tile_m={tile_m}   layer with routing prebuilt: {bare:.1f} us")
        print(f"    {'routing_impl':>14} {'layer+route':>12} {'routing adds':>13}")
        ref = None
        for impl in IMPLS:
            y = moe_forward(x, w1, w2, ids, wts, tile_m=tile_m, routing_impl=impl)
            t = bench_us(lambda impl=impl: moe_forward(x, w1, w2, ids, wts, tile_m=tile_m, routing_impl=impl))
            if ref is None:
                ref, diff = y.clone(), 0.0
            else:
                diff = (y.float() - ref.float()).abs().max().item()
            print(f"    {impl:>14} {t:11.1f}u {t - bare:12.1f}u   maxdiff vs host {diff:.1e}")

    # ── 3. Why the padded tiles were always safe ─────────────────────────
    section("3. the early exit is a saving, not a correctness fix")
    print("  A tile past the real end reads sentinel routing ids. The A gather is a")
    print("  bounds-checked buffer descriptor, so a sentinel row reads zero; the")
    print("  scatter is bounds-checked too, so it stores nothing. The tile computes")
    print("  garbage into registers and throws it away.")
    print()
    tokens, tile_m = 1024, 64
    x, w1, w2, ids, wts = make_layer_inputs(
        tokens=tokens, model_dim=MODEL_DIM, inter_dim=INTER_DIM, experts=EXPERTS, topk=TOPK
    )
    y_exact = moe_forward(x, w1, w2, ids, wts, tile_m=tile_m, routing_impl="device-exact")
    y_bound = moe_forward(x, w1, w2, ids, wts, tile_m=tile_m, routing_impl="device")
    print(f"  exact grid vs bounded grid, bit-identical: {torch.equal(y_exact, y_bound)}")
    bound = max_blocks_for(tokens=tokens, topk=TOPK, experts=EXPERTS, tile_m=tile_m)
    exact = build_routing_device(ids, experts=EXPERTS, tile_m=tile_m, exact=True, reuse=False).num_blocks
    print(f"  the guard skips {bound - exact} of {bound} tiles ({100 * (bound - exact) / bound:.1f}% of the grid)")
    print()
    print("  Two details that made this work:")
    print("   - the guard sits at the *end* of the kernel; everything above it is")
    print("     address arithmetic that a padded tile makes well defined")
    print("   - the condition is the block index, so it is workgroup-uniform and")
    print("     the barriers inside stay matched")


if __name__ == "__main__":
    main()
