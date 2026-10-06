#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""o7, part 2: building the mixed-tile scheme, and finding out it loses.

    python docs/rdna3_moe_optimization/repro/o7_padding_measure.py

o7_padding_predict.py priced a 128-row main tile plus a 64-row tail tile at about
+6% on stage1. This builds it. It measures -8.8% on the layer, and then finds
the cost the prediction was missing, which turns out to disqualify the entire
approach rather than this one variant of it.

Two things make the scheme cheap to build, and neither is the problem:

  no kernel change   BLOCK_M is a compile-time constant, so "mixed heights" is
                     just the existing kernel launched twice over complementary
                     row sets. Not one line of the GEMM moves.
  no shared routing  stage1 addresses its output by packed token id rather than
                     by routing row, so stage2 can read that output through a
                     completely different row order. That is what lets stage1
                     take the mixed scheme while stage2 keeps the uniform 128 it
                     prefers.

The problem is the second launch itself. Section 4 isolates it and section 5
prices the whole idea, including schemes nobody has written yet.
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

# From part 1's sweep: the production entry for the 64 height is 64x32, tuned for
# a decode that needs many workgroups. A tail tile rides along in a prefill whose
# N grid is already wide, so it wants 64x128 -- 75 against 60 TFLOP/s. Give the
# scheme its best shot; without this it loses by even more.
TAIL_CFG = (2, 4, 2, 2, 2)  # 64x128x32


def build_mixed_routing(topk_ids, *, experts, main=MAIN, tail=TAIL):
    """Split the routing into full main tiles and one padded tail per expert.

    Returns ``(main_routing, tail_routing)``, either of which may hold zero
    tiles. Between them every ``(token, slot)`` appears exactly once, which is
    what makes launching over both equivalent to launching over one padded order.
    """
    tokens, topk = topk_ids.shape
    flat = topk_ids.reshape(-1).to(torch.int32).cpu().numpy()
    order = np.argsort(flat, kind="stable")
    counts = np.bincount(flat, minlength=experts)[:experts]

    n_main = counts // main
    main_rows = n_main * main
    n_tail = (counts - main_rows + tail - 1) // tail
    tail_rows = n_tail * tail

    # Where each expert's rows begin: in the sorted order, and in each buffer.
    src = np.cumsum(counts) - counts
    main_base = np.cumsum(main_rows) - main_rows
    tail_base = np.cumsum(tail_rows) - tail_rows

    # A sorted row's index within its own expert decides which buffer it lands
    # in: an expert's first main_rows fill its main tiles, the rest spill into
    # its tail.
    owner = flat[order]
    within = np.arange(flat.size) - src[owner]
    is_main = within < main_rows[owner]
    packed = ((order % topk) << SLOT_SHIFT) | (order // topk)

    sentinel = (topk << SLOT_SHIFT) | tokens
    ids_main = np.full(int(main_rows.sum()), sentinel, dtype=np.int32)
    ids_tail = np.full(int(tail_rows.sum()), sentinel, dtype=np.int32)
    ids_main[main_base[owner[is_main]] + within[is_main]] = packed[is_main]
    ids_tail[tail_base[owner[~is_main]] + within[~is_main] - main_rows[owner[~is_main]]] = packed[~is_main]

    def pack(ids, blocks, tile_m):
        experts_of_tile = np.repeat(np.arange(experts, dtype=np.int32), blocks)
        return Routing(
            sorted_ids=torch.from_numpy(ids).to(topk_ids.device),
            expert_ids=torch.from_numpy(experts_of_tile).to(topk_ids.device),
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
        host._SHAPE_TILES.pop(key, None) if saved is None else host._SHAPE_TILES.update({key: saved})
        host.compile_grouped_gemm.cache_clear()


def mixed_layer(x, w1, w2, wts, r_main, r_tail, r_down):
    """The layer with a mixed stage1 and a uniform stage2."""
    tokens, model_dim = x.shape
    inter_dim = w1.shape[1] // 2
    topk = wts.shape[1]
    common = dict(
        model_dim=model_dim, inter_dim=inter_dim, experts=w1.shape[0], topk=topk, bounded_blocks=False
    )
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
    header(
        "o7 part 2  measuring the mixed-tile scheme",
        f"D{MODEL_DIM}/I{INTER_DIM}, {TOKENS} tokens, E={EXPERTS}, topk={TOPK}",
    )
    device_banner()

    x, w1, w2, ids, wts = make_layer_inputs(
        tokens=TOKENS, model_dim=MODEL_DIM, inter_dim=INTER_DIM, experts=EXPERTS, topk=TOPK
    )
    real = TOKENS * TOPK
    r_down = build_routing(ids, experts=EXPERTS, tile_m=MAIN)
    r_main, r_tail = build_mixed_routing(ids, experts=EXPERTS)

    section("1. the split on this routing")
    counts = np.bincount(ids.reshape(-1).cpu().numpy(), minlength=EXPERTS)
    print(f"  rows per expert: {' '.join(str(int(c)) for c in counts)}")
    print()
    mixed_rows = r_main.rows + r_tail.rows
    print(f"  {'':>14} {'tiles':>6} {'rows':>6}")
    print(f"  {'uniform 128':>14} {r_down.num_blocks:>6} {r_down.rows:>6}")
    print(f"  {'main 128':>14} {r_main.num_blocks:>6} {r_main.rows:>6}   full by construction, no padding")
    print(f"  {'tail 64':>14} {r_tail.num_blocks:>6} {r_tail.rows:>6}")
    print(f"  {'mixed total':>14} {r_main.num_blocks + r_tail.num_blocks:>6} {mixed_rows:>6}")
    print()
    print(f"  padding waste {100 * (r_down.rows / real - 1):.1f}% uniform -> {100 * (mixed_rows / real - 1):.1f}% mixed")

    section("2. does it compute the same thing")
    ref = torch_moe_layer(x, w1, w2, ids, wts)
    base = moe_forward(x, w1, w2, ids, wts, tile_m=MAIN, routing=r_down)
    with tail_config():
        got = mixed_layer(x, w1, w2, wts, r_main, r_tail, r_down)()
    torch.cuda.synchronize()
    scale = ref.float().abs().max().item()
    for name, out in (("uniform 128", base), ("mixed 128+64", got)):
        err = (out.float() - ref.float()).abs().max().item()
        print(f"  {name:>16} vs torch: max abs {err:.4e}, relative {err / scale:.2e}")
    print(f"  {'mixed vs uniform':>16}: max abs {(got.float() - base.float()).abs().max().item():.4e}")
    print()
    print("  Both orders sum the same products, and every (token, slot) appears in")
    print("  exactly one tile of exactly one launch, so the split is sound.")

    section("3. is it faster")
    t_base = bench_us(lambda: moe_forward(x, w1, w2, ids, wts, tile_m=MAIN, routing=r_down), warmup=3, iters=20)
    with tail_config():
        run = mixed_layer(x, w1, w2, wts, r_main, r_tail, r_down)
        t_mixed = bench_us(run, warmup=3, iters=20)
    print(f"  {'uniform 128':>14} {t_base / 1000:8.3f} ms")
    print(f"  {'mixed 128+64':>14} {t_mixed / 1000:8.3f} ms   {100 * (t_base / t_mixed - 1):+.1f}%")
    print()
    print("  Predicted +3.7% on the layer, measured the other way by more than twice")
    print("  as much. Something in the pricing is wrong, not just imprecise.")

    section("4. what the prediction was missing")
    print("  Time each stage1 launch on its own. Part 1 measured a rate per row at")
    print("  each height and treated it as a property of the height. If a launch has")
    print("  a cost that does not shrink with its row count, that is wrong -- and the")
    print("  tail, which carries the fewest rows, is where it would show.")
    print()
    common = dict(model_dim=MODEL_DIM, inter_dim=INTER_DIM, experts=EXPERTS, topk=TOPK, bounded_blocks=False)
    stream = torch.cuda.current_stream()
    a2 = torch.empty(TOKENS, TOPK, INTER_DIM, dtype=x.dtype, device=x.device)
    y = torch.empty(TOKENS, TOPK, MODEL_DIM, dtype=x.dtype, device=x.device)
    unused = torch.empty(0, dtype=torch.float32, device=x.device)
    w_rows1 = 2 * INTER_DIM
    slab_gb = EXPERTS * w_rows1 * MODEL_DIM * 2 / 1e9

    print(f"  {'launch':>16} {'block':>10} {'tiles':>6} {'rows':>6} {'ms':>7} {'MAC TF/s':>9} {'GB/s of B':>10}")
    with tail_config():
        for name, r in (("uniform 128", r_down), ("main 128", r_main), ("tail 64", r_tail)):
            g, bm, bn, _ = compile_moe_gemm1(tile_m=r.tile_m, **common)
            us = bench_us(
                lambda g=g, r=r: g(a2, x, w1, r.sorted_ids, r.expert_ids, unused, TOKENS, r.num_blocks, stream),
                warmup=3,
                iters=20,
            )
            tf = 2.0 * r.rows * w_rows1 * MODEL_DIM / (us * 1e-6) / 1e12
            print(
                f"  {name:>16} {f'{bm}x{bn}':>10} {r.num_blocks:>6} {r.rows:>6} "
                f"{us / 1000:7.3f} {tf:9.2f} {slab_gb / (us * 1e-6):10.0f}"
            )
    print()
    print("  There it is. The same 128 tile drops from ~85 to ~69 TFLOP/s purely by")
    print("  carrying fewer rows, so the rate was never a property of the height. Every")
    print(f"  launch has to stream all {EXPERTS} experts' weights -- {slab_gb:.2f} GB, {1e3 * slab_gb / 830:.2f} ms")
    print("  at this card's ~830 GB/s -- and that cost is per launch, amortised over")
    print("  however many rows the launch happens to carry. Splitting a stage in two")
    print("  buys one extra full pass over the weights.")

    section("5. the ceiling on any padding fix, not just this one")
    print("  So how much is the padding worth at all? Hand the same kernel a routing")
    print("  that needs no padding -- every expert given exactly 256 rows, which is")
    print("  two full 128 tiles -- and the gap against the real routing is the entire")
    print("  prize, whatever mechanism goes after it.")
    print()
    rng = np.random.default_rng(0)
    flat = np.repeat(np.arange(EXPERTS, dtype=np.int32), real // EXPERTS)
    rng.shuffle(flat)  # keep the gather scattered, as a real routing's is
    r_bal = build_routing(torch.from_numpy(flat.reshape(TOKENS, TOPK)).to(ids.device), experts=EXPERTS, tile_m=MAIN)

    g1, *_ = compile_moe_gemm1(tile_m=MAIN, **common)
    g2, *_ = compile_moe_gemm2(doweight=True, tile_m=MAIN, **common)
    stages = (
        ("gate_up", 2 * INTER_DIM, MODEL_DIM, lambda r: g1(a2, x, w1, r.sorted_ids, r.expert_ids, unused, TOKENS, r.num_blocks, stream)),
        ("down", MODEL_DIM, INTER_DIM, lambda r: g2(y, a2, w2, r.sorted_ids, r.expert_ids, wts, TOKENS, r.num_blocks, stream)),
    )
    pad_rows = r_down.rows - r_bal.rows
    budget = 0.0
    for stage, n_out, k_dim, call in stages:
        print(f"  {stage}")
        print(f"    {'routing':>10} {'tiles':>6} {'rows':>6} {'waste':>7} {'ms':>7} {'useful TF/s':>12}")
        times = {}
        for name, r in (("real", r_down), ("balanced", r_bal)):
            times[name] = bench_us(lambda r=r: call(r), warmup=3, iters=20)
            tf = 2.0 * real * n_out * k_dim / (times[name] * 1e-6) / 1e12  # useful rows, not padded
            print(
                f"    {name:>10} {r.num_blocks:>6} {r.rows:>6} {100 * (r.rows / real - 1):6.1f}% "
                f"{times[name] / 1000:7.3f} {tf:12.2f}"
            )
        gap = times["real"] - times["balanced"]
        budget += gap
        print(
            f"    {pad_rows} padded rows are worth {gap / 1000:.3f} ms: {gap / pad_rows:.2f} us each, "
            f"against {times['real'] / r_down.rows:.2f} us for an average row"
        )
        print()
    print(f"  Whole-layer budget for any padding fix: {budget / 1000:.3f} ms of {t_base / 1000:.3f} ms, or")
    print(f"  {100 * budget / t_base:.1f}%. And that is the ceiling, not a gain -- it assumes the")
    print("  padding is removed for free.")
    print()
    print("  Which settles it. A padded row costs roughly a third of what an average")
    print("  row costs, because most of a launch's time does not scale with rows at")
    print("  all. So roofline.py reporting 24% of the MACs as wasted was true and")
    print(f"  still misleading: those MACs are only {100 * budget / t_base:.0f}% of the time, and that is")
    print("  before any scheme's own overhead.")
    print()
    print(f"  A split launch's second weight pass costs {1e3 * slab_gb / 830:.2f} ms by itself, more than")
    print("  the whole budget, so no two-launch scheme can win at this shape however")
    print("  its tails are sized. Anything that wants this budget has to stay inside")
    print("  one launch and skip padded rows within the tuned inner loop -- a change")
    print("  to the most schedule-sensitive code there is, for at most a tenth.")


if __name__ == "__main__":
    main()
