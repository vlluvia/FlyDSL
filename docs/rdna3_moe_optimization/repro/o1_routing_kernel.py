#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""o1: move routing off the host, and find out what that actually buys.

    python docs/rdna3_moe_optimization/repro/o1_routing_kernel.py

Three parts, in the order the original investigation ran them:

  1. Symptom -- at decode sizes the host routing builder costs more than the
     three GEMMs it feeds. That is the whole reason this phase exists.
  2. Correctness -- the device kernel must produce bit-identical buffers, not
     merely a routing that works. Ordering is part of the contract.
  3. The catch -- the kernel is ~10us but ``build_routing_device(exact=True)``
     is far more, because reading the tile count back to the host is a full
     round trip. Finding this is what created o2.

Part 4 reproduces the scaling limit that was left unfixed: one global load per
(expert, row) pair means cost grows with the expert count, and past about
E=64 the host builder wins again.
"""

from __future__ import annotations

import torch

from common import bench_us, device_banner, header, section
from kernels.moe.rdna3_moe.forward import moe_forward
from kernels.moe.rdna3_moe.routing import build_routing
from kernels.moe.rdna3_moe.routing_kernel import build_routing_device

MODEL_DIM, INTER_DIM, EXPERTS, TOPK = 2048, 768, 8, 2


def _ids(tokens, experts, topk, seed=0):
    """A routing drawn without replacement, the way real gating produces it."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(tokens, experts, generator=gen, device="cuda")
    return logits.topk(topk, dim=-1).indices.to(torch.int32).contiguous()


def main():
    header("o1  routing as one wave32 kernel", "counting sort in a single workgroup, stable, no LDS atomics")
    device_banner()

    # ── 1. The symptom ───────────────────────────────────────────────────
    section("1. symptom: at decode, routing costs more than the layer it feeds")
    from common import make_layer_inputs

    x, w1, w2, ids32, wts32 = make_layer_inputs(
        tokens=32, model_dim=MODEL_DIM, inter_dim=INTER_DIM, experts=EXPERTS, topk=TOPK
    )
    r_pre = build_routing_device(ids32, experts=EXPERTS, tile_m=16, exact=True)
    t_layer = bench_us(lambda: moe_forward(x, w1, w2, ids32, wts32, tile_m=16, routing=r_pre))
    t_host = bench_us(lambda: build_routing(ids32, experts=EXPERTS, tile_m=16))
    print(f"  layer alone (routing prebuilt)      {t_layer:8.1f} us")
    print(f"  host routing builder alone          {t_host:8.1f} us   ({t_host / t_layer:.2f}x the layer)")
    print()
    print("  Whatever the exact ratio on your machine, the point is the order of")
    print("  magnitude: deciding *which* rows to multiply costs about as much as")
    print("  multiplying them. That is not a GEMM problem, so no amount of tile")
    print("  tuning would have found it.")

    # ── 2. Correctness first ─────────────────────────────────────────────
    section("2. the device build must be bit-identical, ordering included")
    print(f"  {'tokens':>7} {'E':>4} {'topk':>5} {'tile_m':>7}  {'sorted_ids':>11} {'expert_ids':>11} {'num_blocks':>11}")
    for tokens, e, topk, tile_m in [(32, 8, 2, 16), (7, 4, 3, 32), (1024, 8, 2, 64), (256, 16, 4, 32)]:
        ids = _ids(tokens, e, topk)
        a = build_routing(ids, experts=e, tile_m=tile_m)
        b = build_routing_device(ids, experts=e, tile_m=tile_m, exact=True, reuse=False)
        same_s = torch.equal(a.sorted_ids, b.sorted_ids)
        same_e = torch.equal(a.expert_ids, b.expert_ids)
        same_n = a.num_blocks == b.num_blocks
        print(
            f"  {tokens:>7} {e:>4} {topk:>5} {tile_m:>7}  {str(same_s):>11} {str(same_e):>11} "
            f"{str(same_n):>11}"
        )

    # ── 3. The kernel is fast; the readback is not ───────────────────────
    section("3. where the time actually goes: the kernel vs the round trip")
    print(f"  {'tokens':>7} {'host numpy':>11} {'kernel+readback':>16} {'kernel only':>12}")
    for tokens in (32, 256, 1024, 4096):
        ids = _ids(tokens, EXPERTS, TOPK)
        t_h = bench_us(lambda: build_routing(ids, experts=EXPERTS, tile_m=16))
        t_exact = bench_us(lambda: build_routing_device(ids, experts=EXPERTS, tile_m=16, exact=True))
        # exact=False returns the bound instead of reading the count back, so
        # this is the launch on its own -- the same kernel, minus the wait.
        t_kern = bench_us(lambda: build_routing_device(ids, experts=EXPERTS, tile_m=16, exact=False))
        print(f"  {tokens:>7} {t_h:10.1f}u {t_exact:15.1f}u {t_kern:11.1f}u")
    print()
    print("  The gap between the last two columns is one device-to-host read of a")
    print("  single int32. That read is what o2 removes.")

    # ── 4. The limit that was left in ────────────────────────────────────
    section("4. the known limit: cost grows with the expert count")
    print("  A thread owns an (expert, slice) pair and walks its rows, one global")
    print("  load each. More experts means more passes over the same rows.")
    print()
    print(f"  {'E':>5} {'topk':>5} {'tokens':>7} {'host numpy':>11} {'kernel only':>12} {'kernel/E=8':>11}")
    base = None
    for e, topk, tokens in [(8, 2, 1024), (32, 4, 1024), (64, 4, 512), (128, 4, 512)]:
        ids = _ids(tokens, e, topk)
        t_h = bench_us(lambda: build_routing(ids, experts=e, tile_m=16))
        t_k = bench_us(lambda: build_routing_device(ids, experts=e, tile_m=16, exact=False, reuse=False))
        base = base or t_k
        flag = "   <- host wins" if t_k > t_h else ""
        print(f"  {e:>5} {topk:>5} {tokens:>7} {t_h:10.1f}u {t_k:11.1f}u {t_k / base:10.1f}x{flag}")
    print()
    print("  The kernel column grows with E while the host column does not. Where")
    print("  they cross depends on the machine and on topk, but the shape of the")
    print("  curve is the point: this design is right for 8-64 experts and wrong")
    print("  for hundreds.")
    print()
    print("  Fix (not taken): stage the routing ids in LDS and use an atomic")
    print("  histogram. Only the scatter needs the stable order, so the counting")
    print("  can go atomic. Written up in routing_kernel.py's docstring.")


if __name__ == "__main__":
    main()
