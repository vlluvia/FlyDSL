#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""v5-3a: what the routing kernel costs at hundreds of experts, before rewriting it.

    python docs/rdna3_moe_optimization/repro/v5_routing_scale.py
    python docs/rdna3_moe_optimization/repro/v5_routing_scale.py --rows 8192

o1 left a known scaling limit and named the fix: a thread owns an (expert, slice)
pair, so every row is read from global memory by all E threads that might claim
it, and the walk is ``E * rows / 256`` loads. It measured 89 us at E=128 against
a host builder that does not grow with E, and o1's own conclusion was that this
design is right up to a few tens of experts and wrong past that. Current models
are past that: 256 experts at Qwen3.5-35B, 896 at Kimi-K3.

Two things that 89 us does not settle, which is what this script is for.

Whether it is worth rewriting. 89 us is a lot next to a decode layer and nothing
next to a 10 ms Kimi-K3 prefill layer, and the expert count that makes routing
slow is also the one that makes the layer big -- so the fraction may not grow
even though the microsecond count does. Part 2 measures routing against the layer
it feeds, which is the number that decides.

Whether the limit is a slope or a wall. ``E <= BLOCK`` was a hard reject in
``create_routing_module``: past 256 there is no slice left to give a thread, so
the kernel did not build at all. Part 3 walks E up to and past it. A shape that
cannot run is a different kind of problem from one that runs slowly, and only
the first one had to be fixed -- which is what v5-3 did, leaving the slope where
part 2 found it.
"""

from __future__ import annotations

import argparse

import torch

# common must be imported before anything under kernels/: it is what puts the
# repo root on sys.path. Keep these as from-imports so the sort order holds.
from common import bench_us, device_banner, header, section
from kernels.moe.rdna3_moe.forward import moe_forward
from kernels.moe.rdna3_moe.routing import build_routing
from kernels.moe.rdna3_moe.routing_kernel import BLOCK, build_routing_device
from v5_shape_coverage import SHAPES


def _ids(tokens, experts, topk, seed=0):
    """A routing drawn without replacement, the way real gating produces it."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(tokens, experts, generator=gen, device="cuda")
    return logits.topk(topk, dim=-1).indices.to(torch.int32).contiguous()


def part1_pairs(rows, tile_m):
    """The cost model on its own: hold the pair count still and vary only E."""
    section("1. it is the (expert, row) pairs, not the experts")
    print("  Same rows throughout, so the counting sort's real work is fixed and")
    print("  only the redundant reads move. E*rows is what the walk pays for.")
    print()
    print(f"  {'E':>5} {'rows':>7} {'pairs':>10} {'loads/thread':>13} {'kernel':>9} {'host':>9}")
    for e in (8, 32, 64, 128, 256):
        ids = _ids(rows, e, 1)
        t_k = bench_us(lambda: build_routing_device(ids, experts=e, tile_m=tile_m, exact=False, reuse=False))
        t_h = bench_us(lambda: build_routing(ids, experts=e, tile_m=tile_m))
        pairs = e * rows
        print(f"  {e:>5} {rows:>7} {pairs:>10} {pairs // BLOCK:>13} {t_k:8.1f}u {t_h:8.1f}u")
        del ids
    print()
    print("  A thread's loads are E*rows/256 because SLICES=256/E shrinks as E grows:")
    print("  at E=256 one thread walks every row of its expert alone.")


def part2_fraction(shapes, world, rows, tile_m):
    """The number that decides whether this is worth rewriting."""
    section(f"2. routing against the layer it feeds (EP={world}, {rows} rows)")
    print(f"  {'model':>20} {'E/card':>7} {'routing':>9} {'layer':>10} {'share':>7}")
    for _key, label, model_dim, inter_dim, experts, _topk, _src in shapes:
        epr = max(1, experts // world)
        ids = _ids(rows, epr, 1)
        dev = "cuda"
        gen = torch.Generator(device=dev).manual_seed(0)
        x = (torch.randn(rows, model_dim, generator=gen, device=dev, dtype=torch.bfloat16) * 0.1).contiguous()
        w1 = (
            torch.randn(epr, 2 * inter_dim, model_dim, generator=gen, device=dev, dtype=torch.bfloat16) * 0.05
        ).contiguous()
        w2 = (
            torch.randn(epr, model_dim, inter_dim, generator=gen, device=dev, dtype=torch.bfloat16) * 0.05
        ).contiguous()
        wts = torch.rand(rows, 1, generator=gen, device=dev, dtype=torch.float32).contiguous()

        t_r = bench_us(lambda: build_routing_device(ids, experts=epr, tile_m=tile_m, exact=False))
        moe_forward(x, w1, w2, ids, wts, tile_m=tile_m)
        torch.cuda.synchronize()
        t_l = bench_us(lambda: moe_forward(x, w1, w2, ids, wts, tile_m=tile_m), warmup=5, iters=30)
        print(f"  {label:>20} {epr:>7} {t_r:8.1f}u {t_l:9.1f}u {100 * t_r / t_l:6.1f}%")

        del x, w1, w2, wts, ids
        torch.cuda.empty_cache()


def part3_wall(rows, tile_m):
    """Past 256 the kernel used not to get slower, it stopped existing."""
    section("3. the limit was a wall, not a slope")
    print("  A thread needs a slice of its own, and there are 256 of them, so")
    print("  E > 256 was a build failure rather than a slow build. It now takes")
    print("  one pass per block of experts; the walk is unchanged, which part 2")
    print("  says is the right trade.")
    print()
    print(f"  {'E':>5} {'chunks':>7} {'kernel':>9} {'host':>9}  builds?")
    for e in (128, 256, 384, 512, 896):
        chunks = -(-e // BLOCK)
        try:
            ids = _ids(min(rows, 4096), e, 1)
            t_k = bench_us(lambda: build_routing_device(ids, experts=e, tile_m=tile_m, exact=False, reuse=False))
            t_h = bench_us(lambda: build_routing(ids, experts=e, tile_m=tile_m))
            cells, verdict = f"{t_k:8.1f}u {t_h:8.1f}u", "yes"
            del ids
        except (ValueError, RuntimeError) as exc:
            cells, verdict = f"{'':>9} {'':>9}", f"no -- {str(exc).splitlines()[0][:50]}"
        print(f"  {e:>5} {chunks:>7} {cells}  {verdict}")
    print()
    print("  Qwen3.5-397B's 512 experts are the reachable case: 12.9 GB of weights")
    print("  fit a 24 GB card, so EP=1 is a configuration somebody has, and it had")
    print("  no routing. Kimi-K3's 896 would need 59 GB, so its own EP=1 is out of")
    print("  reach for other reasons -- it is here because EP=2 and EP=4 are not.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=2048)
    p.add_argument("--world", type=int, default=8)
    p.add_argument("--tile-m", type=int, default=32)
    p.add_argument("--skip-layer", action="store_true", help="parts 1 and 3 only; they need no weights")
    args = p.parse_args()

    header(
        "v5-3a  the routing kernel at hundreds of experts",
        "o1's known limit, measured before it gets rewritten",
    )
    device_banner()

    part1_pairs(args.rows, args.tile_m)
    if not args.skip_layer:
        part2_fraction(SHAPES, args.world, args.rows, args.tile_m)
    part3_wall(args.rows, args.tile_m)


if __name__ == "__main__":
    main()
