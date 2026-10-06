#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""o7 prototype: mixed tile heights, measured rather than predicted.

    python docs/rdna3_moe_optimization/repro/o7_mixed_prototype.py

o7_probe.py predicted that giving stage1 a 128-row main tile plus a 64-row tail
tile is worth about +5.7% on that stage, and that every mixed variant loses on
stage2. This runs it, because a prediction that is not checked is not a result.

The scheme, per expert with ``c`` rows:

    n_main = c // 128            main tiles, exactly full, no padding at all
    n_tail = ceil((c % 128) / 64) tail tiles, padded to 64

so an expert with 302 rows becomes 2 main + 1 tail = 320 padded rows, where
padding everything to 128 would have cost 384.

Two things make this cheap to build. The tiles do not have to be one launch:
BLOCK_M is a compile-time constant, so "mixed heights" is just the existing
kernel launched twice over complementary row sets, and no kernel changes at all.
And the two stages do not have to share a routing: stage1 addresses its output
by packed token id, not by routing row, so stage2 can read that output through a
completely different row order. That decoupling is what lets stage1 take the
mixed scheme while stage2 keeps the uniform 128 it prefers.
"""

from __future__ import annotations

import contextlib

import numpy as np
import torch

# common must be imported before anything under kernels/: it is what puts the
# repo root on sys.path. Keep these as from-imports so the sort order holds.
from common import bench_us, device_banner, header, make_layer_inputs, section, torch_moe_layer

from kernels.moe.rdna3_moe import host
from kernels.moe.rdna3_moe.forward import moe_forward, moe_reduce
from kernels.moe.rdna3_moe.host import compile_moe_gemm1, compile_moe_gemm2
from kernels.moe.rdna3_moe.routing import SLOT_SHIFT, Routing, build_routing

EXPERTS, TOPK, TOKENS = 8, 2, 1024
MODEL_DIM, INTER_DIM = 4096, 14336
MAIN, TAIL = 128, 64

# The tail tile's config, from o7_probe.py's sweep. The production entry for the
# 64 height is 64x32, tuned for a decode that needs many workgroups; a tail tile
# rides along in a prefill whose N grid is already wide, so it wants 64x128
# instead -- 74.8 against 60.2 TFLOP/s. Without this the tail is too slow to pay
# for itself and the whole scheme regresses.
TAIL_CFG = (2, 4, 2, 2, 2)  # 64x128x32


def build_mixed_routing(topk_ids, *, experts, main=MAIN, tail=TAIL):
    """Split the routing into full main tiles and a padded tail per expert.

    Returns ``(main_routing, tail_routing)``, either of which may have zero
    tiles. Between them every ``(token, slot)`` appears exactly once, which is
    what makes launching over both equivalent to launching over one padded
    order.
    """
    tokens, topk = topk_ids.shape
    flat = topk_ids.reshape(-1).to(torch.int32).cpu().numpy()
    rows_in = flat.size
    order = np.argsort(flat, kind="stable")
    counts = np.bincount(flat, minlength=experts)[:experts]

    n_main = counts // main
    main_rows = n_main * main
    rest = counts - main_rows
    n_tail = (rest + tail - 1) // tail
    tail_rows = n_tail * tail

    # Where each expert's rows start: in the sorted order, and in each buffer.
    src = np.cumsum(counts) - counts
    main_base = np.cumsum(main_rows) - main_rows
    tail_base = np.cumsum(tail_rows) - tail_rows

    # A sorted row's index within its own expert decides which buffer it lands
    # in: the first main_rows of an expert fill its main tiles, the rest spill
    # into its tail.
    owner = flat[order]
    within = np.arange(rows_in) - src[owner]
    is_main = within < main_rows[owner]
    packed = ((order % topk) << SLOT_SHIFT) | (order // topk)

    sentinel = (topk << SLOT_SHIFT) | tokens
    ids_main = np.full(int(main_rows.sum()), sentinel, dtype=np.int32)
    ids_tail = np.full(int(tail_rows.sum()), sentinel, dtype=np.int32)
    ids_main[main_base[owner[is_main]] + within[is_main]] = packed[is_main]
    ids_tail[tail_base[owner[~is_main]] + within[~is_main] - main_rows[owner[~is_main]]] = packed[~is_main]

    dev = topk_ids.device

    def pack(ids, blocks, tile_m):
        return Routing(
            sorted_ids=torch.from_numpy(ids).to(dev),
            expert_ids=torch.from_numpy(np.repeat(np.arange(experts, dtype=np.int32), blocks)).to(dev),
            num_blocks=int(blocks.sum()),
            tile_m=tile_m,
            tokens=int(tokens),
            topk=int(topk),
        )

    return pack(ids_main, n_main, main), pack(ids_tail, n_tail, tail)


@contextlib.contextmanager
def tail_config():
    """Put the tail tile's config in front of the production table."""
    key = ("gate_up", MODEL_DIM, INTER_DIM, TAIL)
    saved = host._SHAPE_TILES.get(key)
    host._SHAPE_TILES[key] = (TAIL_CFG,)
    host.compile_grouped_gemm.cache_clear()
    try:
        yield
    finally:
        if saved is None:
            host._SHAPE_TILES.pop(key, None)
        else:
            host._SHAPE_TILES[key] = saved
        host.compile_grouped_gemm.cache_clear()


def mixed_layer(x, w1, w2, wts, r_main, r_tail, r_down):
    """The layer with a mixed stage1 and a uniform stage2."""
    tokens, model_dim = x.shape
    experts, two_inter, _ = w1.shape
    inter_dim = two_inter // 2
    topk = wts.shape[1]
    common = dict(model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk, bounded_blocks=False)
    g1_main, *_ = compile_moe_gemm1(tile_m=r_main.tile_m, **common)
    g1_tail, *_ = compile_moe_gemm1(tile_m=r_tail.tile_m, **common)
    g2, *_ = compile_moe_gemm2(doweight=True, tile_m=r_down.tile_m, **common)

    stream = torch.cuda.current_stream()
    a2 = torch.empty(tokens, topk, inter_dim, dtype=x.dtype, device=x.device)
    y = torch.empty(tokens, topk, model_dim, dtype=x.dtype, device=x.device)
    unused = torch.empty(0, dtype=torch.float32, device=x.device)

    def run():
        for g, r in ((g1_main, r_main), (g1_tail, r_tail)):
            if r.num_blocks:
                g(a2, x, w1, r.sorted_ids, r.expert_ids, unused, tokens, r.num_blocks, stream)
        g2(y, a2, w2, r_down.sorted_ids, r_down.expert_ids, wts, tokens, r_down.num_blocks, stream)
        return moe_reduce(y, stream=stream)

    return run


def main():
    header("o7 prototype  mixed tile heights on stage1", f"D{MODEL_DIM}/I{INTER_DIM}, {TOKENS} tokens, E={EXPERTS}, topk={TOPK}")
    device_banner()

    x, w1, w2, ids, wts = make_layer_inputs(
        tokens=TOKENS, model_dim=MODEL_DIM, inter_dim=INTER_DIM, experts=EXPERTS, topk=TOPK
    )
    r_down = build_routing(ids, experts=EXPERTS, tile_m=MAIN)
    r_main, r_tail = build_mixed_routing(ids, experts=EXPERTS)

    section("1. what the split looks like on this routing")
    counts = np.bincount(ids.reshape(-1).cpu().numpy(), minlength=EXPERTS)
    print(f"  rows per expert: {list(counts)}")
    print(f"  uniform 128:  {r_down.num_blocks:3d} tiles, {r_down.rows:5d} padded rows")
    print(f"  main 128:     {r_main.num_blocks:3d} tiles, {r_main.rows:5d} rows (no padding by construction)")
    print(f"  tail 64:      {r_tail.num_blocks:3d} tiles, {r_tail.rows:5d} rows")
    mixed_rows = r_main.rows + r_tail.rows
    real = TOKENS * TOPK
    print(f"  mixed total:  {r_main.num_blocks + r_tail.num_blocks:3d} tiles, {mixed_rows:5d} padded rows")
    print()
    print(f"  padding waste: {100 * (r_down.rows / real - 1):.1f}% uniform -> {100 * (mixed_rows / real - 1):.1f}% mixed")

    section("2. does it compute the same thing")
    ref = torch_moe_layer(x, w1, w2, ids, wts)
    base = moe_forward(x, w1, w2, ids, wts, tile_m=MAIN, routing=r_down)
    with tail_config():
        run = mixed_layer(x, w1, w2, wts, r_main, r_tail, r_down)
        got = run()
    torch.cuda.synchronize()
    for name, out in (("uniform 128", base), ("mixed 128+64", got)):
        err = (out.float() - ref.float()).abs().max().item()
        rel = err / ref.float().abs().max().item()
        print(f"  {name:>14}  max abs err {err:.4e}  relative {rel:.2e}")
    print(f"  {'mixed vs uniform':>14}  max abs diff {(got.float() - base.float()).abs().max().item():.4e}")
    print()
    print("  The two orders sum the same products in a different tile order, so they")
    print("  are not expected to be bitwise equal -- only to agree with torch.")

    section("3. is it faster")
    t_base = bench_us(lambda: moe_forward(x, w1, w2, ids, wts, tile_m=MAIN, routing=r_down), warmup=3, iters=20)
    with tail_config():
        run = mixed_layer(x, w1, w2, wts, r_main, r_tail, r_down)
        t_mixed = bench_us(run, warmup=3, iters=20)
    print(f"  {'uniform 128':>14} {t_base / 1000:8.3f} ms")
    print(f"  {'mixed 128+64':>14} {t_mixed / 1000:8.3f} ms   {100 * (t_base / t_mixed - 1):+.1f}%")
    print()
    print("  o7_probe.py predicted +5.7% on stage1, which is about +3.7% on a layer")
    print("  where stage1 is two thirds of the time. Anything far from that means the")
    print("  model is missing a cost, so the next section takes stage1 apart.")

    section("4. where the prediction went wrong")
    print("  Time each stage1 launch on its own. The probe measured a rate per row at")
    print("  each height and assumed it was a property of the height. If a launch has")
    print("  a fixed cost that does not shrink with its row count, that assumption is")
    print("  wrong, and the tail -- which carries few rows -- is where it shows.")
    print()
    common = dict(model_dim=MODEL_DIM, inter_dim=INTER_DIM, experts=EXPERTS, topk=TOPK, bounded_blocks=False)
    stream = torch.cuda.current_stream()
    a2 = torch.empty(TOKENS, TOPK, INTER_DIM, dtype=x.dtype, device=x.device)
    unused = torch.empty(0, dtype=torch.float32, device=x.device)
    w_rows = 2 * INTER_DIM
    # An expert's B slab, read once per launch at best: this is the floor a
    # launch pays whatever it carries.
    slab_gb = EXPERTS * w_rows * MODEL_DIM * 2 / 1e9

    print(f"  {'launch':>16} {'block':>10} {'tiles':>6} {'rows':>6} {'ms':>7} {'TFLOP/s':>9} {'GB/s of B':>10}")
    with tail_config():
        for name, r in (("uniform 128", r_down), ("main 128", r_main), ("tail 64", r_tail)):
            g, bm, bn, _ = compile_moe_gemm1(tile_m=r.tile_m, **common)

            def once(g=g, r=r):
                g(a2, x, w1, r.sorted_ids, r.expert_ids, unused, TOKENS, r.num_blocks, stream)

            us = bench_us(once, warmup=3, iters=20)
            tf = 2.0 * r.rows * w_rows * MODEL_DIM / (us * 1e-6) / 1e12
            print(
                f"  {name:>16} {f'{bm}x{bn}':>10} {r.num_blocks:>6} {r.rows:>6} "
                f"{us / 1000:7.3f} {tf:9.2f} {slab_gb / (us * 1e-6):10.0f}"
            )
    print()
    print(f"  Each row's weight pass is {slab_gb:.2f} GB. A launch cannot read less than")
    print("  that, so the fewer rows it carries the worse its rate looks -- the cost is")
    print("  per launch, not per row.")

    section("5. the ceiling on any padding fix at all")
    print("  Worth knowing before trying a third scheme: how much is the padding")
    print("  actually worth? Feed the same kernel a routing that happens to need no")
    print("  padding -- every expert given exactly 256 rows, which is two full 128")
    print("  tiles -- and the difference against the real routing is the entire prize,")
    print("  whatever mechanism goes after it.")
    print()
    rng = np.random.default_rng(0)
    flat = np.repeat(np.arange(EXPERTS, dtype=np.int32), TOKENS * TOPK // EXPERTS)
    rng.shuffle(flat)  # keep the gather scattered, as a real routing's is
    balanced = torch.from_numpy(flat.reshape(TOKENS, TOPK)).to(ids.device)
    r_bal = build_routing(balanced, experts=EXPERTS, tile_m=MAIN)

    y = torch.empty(TOKENS, TOPK, MODEL_DIM, dtype=x.dtype, device=x.device)
    g1, *_ = compile_moe_gemm1(tile_m=MAIN, **common)
    g2, *_ = compile_moe_gemm2(doweight=True, tile_m=MAIN, **common)
    runners = {
        "gate_up": (2 * INTER_DIM, MODEL_DIM, lambda r: g1(a2, x, w1, r.sorted_ids, r.expert_ids, unused, TOKENS, r.num_blocks, stream)),
        "down": (MODEL_DIM, INTER_DIM, lambda r: g2(y, a2, w2, r.sorted_ids, r.expert_ids, wts, TOKENS, r.num_blocks, stream)),
    }
    budget = 0.0
    for stage, (n_rows, k_dim, call) in runners.items():
        print(f"  {stage}")
        print(f"    {'routing':>10} {'tiles':>6} {'rows':>6} {'waste':>7} {'ms':>7} {'useful TF/s':>12} {'us per padded row':>18}")
        times = {}
        for name, r in (("real", r_down), ("balanced", r_bal)):
            us = bench_us(lambda r=r: call(r), warmup=3, iters=20)
            times[name] = us
            tf = 2.0 * real * n_rows * k_dim / (us * 1e-6) / 1e12  # useful rows, not padded
            print(
                f"    {name:>10} {r.num_blocks:>6} {r.rows:>6} {100 * (r.rows / real - 1):6.1f}% "
                f"{us / 1000:7.3f} {tf:12.2f} {us / r.rows:18.2f}"
            )
        gap = times["real"] - times["balanced"]
        budget += gap
        pad_rows = r_down.rows - r_bal.rows
        print(f"    the {pad_rows} padded rows are worth {gap / 1000:.3f} ms, i.e. {gap / pad_rows:.2f} us each")
        print(f"    against an average row's {times['real'] / r_down.rows:.2f} us -- a padded row costs a")
        print("    fraction of a real one, because most of a launch's cost does not")
        print("    scale with its row count at all")
        print()
    print(f"  Whole-layer budget for any padding fix: {budget / 1000:.3f} ms of {t_base / 1000:.3f} ms")
    print(f"  = {100 * budget / t_base:.1f}%. That is the ceiling, not the expected gain. The")
    print("  two-launch scheme spends more than the entire budget on its second")
    print("  weight pass before it saves a single row.")


if __name__ == "__main__":
    main()
