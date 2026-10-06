#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Stage breakdown of the RDNA3 MoE layer at Qwen3.8-Flash-Next's TP=8 decode shape.

    python3 scripts/bench_qwen38_moe_decode.py --tokens 1,2,4,8,16,760

The layer is five launches and the whole-layer number does not say which one is
slow, so this times them apart and prints what each one *has* to move: the tile
count the routing produced against the tiles that carry a real row, and the
weight bytes those tiles read against the bytes the routing asked for.

At E=513 that gap is the story. Every expert owns a tile whether or not a token
chose it, so a one-token step builds 513 tiles to do 11 rows of work, and each
tile streams its expert's weight slab. Run with ``--check`` to also verify the
answer against eager torch.

``--vllm`` adds vLLM's Triton ``fused_moe`` on the same inputs and the same
clock, and prints the tile config it chose -- which is also how you see whether
it found a tuned table (``VLLM_TUNED_CONFIG_FOLDER``) or fell back to the
default heuristic. It needs to run somewhere vLLM is importable, i.e. the
serving image rather than the FlyDSL dev container.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# The in-tree build only goes on the path when nothing else supplies flydsl.
# The serving image ships its own, built against a newer glibc than the tree's
# ``libFlyPythonCAPI.so``, so prepending the tree there shadows a working
# install with one that cannot load -- and the serving image is exactly where
# this has to run to reach vLLM.
try:
    import flydsl  # noqa: F401
except ImportError:
    _build = os.path.join(_REPO_ROOT, "build-fly", "python_packages")
    if os.path.isdir(_build):
        sys.path.insert(0, _build)

from kernels.moe.rdna3_moe.forward import moe_reduce  # noqa: E402
from kernels.moe.rdna3_moe.host import compile_moe_gemm1, compile_moe_gemm2  # noqa: E402
from kernels.moe.rdna3_moe.routing_kernel import (  # noqa: E402
    build_routing_device,
    compile_routing,
    max_blocks_for,
)
from kernels.moe.rdna3_moe.vllm_adapter import tile_m_for  # noqa: E402

# Qwen3.8-Flash-Next's MoE block at TP=8, with the shared expert fused on as
# expert E and riding an extra routing slot.
HIDDEN = 2560
INTER_TP8 = 80
ROUTED_EXPERTS = 512
TOPK_ROUTED = 10


def _bench(fn, *, warmup=20, iters=100, rounds=3) -> float:
    """Steady-state microseconds per call, best of ``rounds``.

    Best rather than mean because the card is shared with whatever else is on it
    and the contention only ever adds. Wall time around a synced loop, so this is
    what the caller pays including the launch -- which at these sizes is a real
    part of the bill.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters * 1e6)
    return best


def _bench_graph(fn, *, inner=10, iters=20, rounds=3) -> float | None:
    """The same, with the calls replayed from a CUDA graph.

    vLLM runs decode under a graph, so this is the number that transfers: no
    per-launch host work, only what the GPU does.

    ``inner`` calls go into one graph because ``replay()`` itself costs the host
    several microseconds. At these sizes that is not a rounding error -- timing
    one kernel per graph made the four stages sum to more than the layer they
    are stages of. It is also what ``benchmark_moe.py`` does, so the two sides
    of the comparison are clocked the same way.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            for _ in range(inner):
                fn()
    except Exception as exc:  # noqa: BLE001 -- a shape that will not capture is a result
        print(f"   (graph capture failed: {exc})")
        return None
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(iters):
            graph.replay()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / (iters * inner) * 1e6)
    graph.reset()
    return best


def _us(v) -> str:
    return f"{v:7.1f} us" if v is not None else "      - us"


def _sum_us(*vals) -> str:
    return f"{sum(v for v in vals if v is not None):.1f} us"


def _route(tokens, experts, topk, shared_id, dev="cuda", seed=0):
    """A router draw plus the always-on shared slot, as the adapter sees it."""
    gen = torch.Generator(device=dev).manual_seed(seed + 1)
    logits = torch.randn(tokens, experts, generator=gen, device=dev, dtype=torch.float32)
    w, ids = torch.topk(torch.softmax(logits, dim=-1), k=topk, dim=-1)
    w = w / w.sum(dim=-1, keepdim=True)
    shared_ids = torch.full((tokens, 1), shared_id, dtype=torch.int32, device=dev)
    shared_w = torch.sigmoid(torch.randn(tokens, 1, generator=gen, device=dev, dtype=torch.float32))
    ids = torch.cat([ids.to(torch.int32), shared_ids], dim=1).contiguous()
    w = torch.cat([w.to(torch.float32), shared_w], dim=1).contiguous()
    return ids, w


def _vllm_fused_moe(x, w13, w2, topk_ids, topk_weights, *, experts, stream):
    """vLLM's Triton ``fused_moe`` on the same inputs, timed the same way.

    The point is the *same harness*: a number quoted from vLLM's own benchmark
    is not comparable to one from here (different iteration counts, different
    graph handling, gating included or not). This runs only the expert half --
    what the FlyDSL layer replaces -- so the two columns line up.

    Returns ``(eager_us, graphed_us, config)``; ``config`` is what the kernel
    actually picked, which is how you can see whether a tuned table was found.
    """
    from vllm.model_executor.layers.fused_moe.fused_moe import (  # noqa: PLC0415
        fused_experts,
        try_get_optimal_moe_config,
    )

    tokens = x.shape[0]
    config = try_get_optimal_moe_config(
        w13.shape, w2.shape, topk_ids.shape[1], None, tokens
    )
    def _call():
        fused_experts(x, w13, w2, topk_weights, topk_ids, global_num_experts=experts)

    return _bench(_call), _bench_graph(_call), config


def _torch_moe(x, w13, w2, topk_ids, topk_weights):
    tokens, topk = topk_ids.shape
    inter = w13.shape[1] // 2
    out = torch.zeros(tokens, x.shape[1], dtype=torch.float32, device=x.device)
    flat_ids, flat_w = topk_ids.reshape(-1), topk_weights.reshape(-1)
    for e in torch.unique(flat_ids).tolist():
        sel = (flat_ids == e).nonzero(as_tuple=True)[0]
        tok = torch.div(sel, topk, rounding_mode="floor")
        xe = x.index_select(0, tok)
        gate = xe @ w13[e][:inter].t()
        up = xe @ w13[e][inter:].t()
        h = (torch.nn.functional.silu(gate.float()) * up.float()).to(x.dtype)
        out.index_add_(0, tok, (h @ w2[e].t()).float() * flat_w[sel].unsqueeze(1))
    return out


def run(tokens: int, *, w13, w2, experts: int, topk: int, inter: int, hidden: int, tile_m: int,
        check: bool, vllm: bool = False):
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(0)
    x = (torch.randn(tokens, hidden, generator=gen, device=dev, dtype=torch.bfloat16) * 0.1).contiguous()
    topk_ids, topk_weights = _route(tokens, experts - 1, topk - 1, experts - 1, dev=dev)

    # Re-read per call rather than binding once: capture runs on its own stream,
    # and a launch queued to the stream that was current beforehand records
    # nothing into the graph.
    stream = torch.cuda.current_stream

    rows_per_expert = tokens * topk / experts
    gemm1, bm1, bn1, bk1 = compile_moe_gemm1(
        model_dim=hidden, inter_dim=inter, experts=experts, topk=topk, tile_m=tile_m,
        bounded_blocks=True, rows_per_expert=rows_per_expert,
    )
    gemm2, bm2, bn2, bk2 = compile_moe_gemm2(
        model_dim=hidden, inter_dim=inter, experts=experts, topk=topk, doweight=True, tile_m=tile_m,
        bounded_blocks=True, rows_per_expert=rows_per_expert,
    )

    routing = build_routing_device(topk_ids, experts=experts, tile_m=tile_m, stream=stream())
    torch.cuda.synchronize()
    real_blocks = int(routing.num_blocks_device[0])
    bound = max_blocks_for(tokens=tokens, topk=topk, experts=experts, tile_m=tile_m)
    # Tiles that carry at least one routed row, which is what the work would be
    # if an expert with no rows got no tile.
    active_experts = int(torch.unique(topk_ids).numel())
    counts = torch.bincount(topk_ids.reshape(-1).long(), minlength=experts)
    useful_blocks = int(((counts + tile_m - 1) // tile_m).sum())

    a2 = torch.empty(tokens, topk, inter, dtype=x.dtype, device=dev)
    y = torch.empty(tokens, topk, hidden, dtype=x.dtype, device=dev)
    out = torch.empty(tokens, hidden, dtype=x.dtype, device=dev)
    unused = torch.empty(0, dtype=torch.float32, device=dev)
    nb_dev = routing.num_blocks_device

    def _routing():
        build_routing_device(topk_ids, experts=experts, tile_m=tile_m, stream=stream())

    def _g1():
        gemm1(a2, x, w13, routing.sorted_ids, routing.expert_ids, unused, tokens, routing.num_blocks, stream(), nb_dev)

    def _g2():
        gemm2(y, a2, w2, routing.sorted_ids, routing.expert_ids, topk_weights, tokens, routing.num_blocks, stream(), nb_dev)

    def _red():
        moe_reduce(y, out=out, stream=stream())

    # The routing kernel with the per-call host work (the build's cache lookups,
    # the workspace lookup, the contiguous check) taken out, so the two costs are
    # separable -- a decode step under a CUDA graph pays only the first.
    raw_routing = compile_routing(experts=experts, topk=topk, tile_m=tile_m)
    ws_ids, ws_eids, ws_aux = routing.sorted_ids, routing.expert_ids, routing.num_blocks_device

    def _routing_raw():
        raw_routing(topk_ids, ws_ids, ws_eids, ws_aux, tokens, bound, stream())

    def _layer():
        _routing(); _g1(); _g2(); _red()

    def _layer_raw():
        _routing_raw(); _g1(); _g2(); _red()

    t_route, t_raw = _bench(_routing), _bench(_routing_raw)
    t_g1, t_g2, t_red = _bench(_g1), _bench(_g2), _bench(_red)
    t_layer = _bench(_layer)
    t_graph = _bench_graph(_layer_raw)
    # The eager stage numbers each carry a host launch, which at these sizes is
    # most of them. Graphing a stage on its own is what it costs the GPU.
    gr_route = _bench_graph(_routing_raw)
    gr_g1, gr_g2, gr_red = _bench_graph(_g1), _bench_graph(_g2), _bench_graph(_red)

    # Weight traffic: one tile streams its expert's whole slab out of DRAM, once
    # per N-tile of the grid, and the L2 is far too small to hold 630 MB.
    w_bytes = (2 * inter * hidden + hidden * inter) * 2
    graph = f"{t_graph:8.1f}" if t_graph is not None else "       -"
    print(
        f"tokens={tokens:<5} tile_m={tile_m}  tiles: {real_blocks} built / {useful_blocks} useful "
        f"({active_experts} active experts, bound {bound})\n"
        f"   blocks g1 {bn1}x{bk1} grid=({inter // bn1}, {routing.num_blocks})   "
        f"g2 {bn2}x{bk2} grid=({hidden // bn2}, {routing.num_blocks})\n"
        f"   routing {t_route:8.1f} us ({t_raw:6.1f} raw)   gemm1 {t_g1:7.1f} us   gemm2 {t_g2:7.1f} us   "
        f"reduce {t_red:6.1f} us   (eager)\n"
        f"   routing {_us(gr_route)}      gemm1 {_us(gr_g1)}   gemm2 {_us(gr_g2)}   "
        f"reduce {_us(gr_red)}   (graphed, sum {_sum_us(gr_route, gr_g1, gr_g2, gr_red)})\n"
        f"   layer {t_layer:9.1f} us eager   {graph} us graphed\n"
        f"   weight read {real_blocks * w_bytes / 2**20:8.1f} MiB built vs "
        f"{useful_blocks * w_bytes / 2**20:7.1f} MiB useful"
    )

    if check:
        _routing(); _g1(); _g2(); _red()
        torch.cuda.synchronize()
        ref = _torch_moe(x, w13, w2, topk_ids, topk_weights)
        err = (out.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
        print(f"   max rel err {err:.3e}  {'OK' if err < 0.05 else 'FAIL'}")

    if vllm:
        v_eager, v_graph, cfg = _vllm_fused_moe(
            x, w13, w2, topk_ids, topk_weights, experts=experts, stream=stream
        )
        v_g = f"{v_graph:8.1f}" if v_graph is not None else "       -"
        print(
            f"   vllm  {v_eager:9.1f} us eager   {v_g} us graphed   "
            f"BLOCK M{cfg['BLOCK_SIZE_M']}xN{cfg['BLOCK_SIZE_N']}xK{cfg['BLOCK_SIZE_K']} "
            f"warps={cfg['num_warps']}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="1,2,4,8,16,760")
    ap.add_argument("--experts", type=int, default=ROUTED_EXPERTS + 1)
    ap.add_argument("--topk", type=int, default=TOPK_ROUTED + 1)
    ap.add_argument("--inter", type=int, default=INTER_TP8)
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--tile-m", type=int, default=0, help="0 picks the adapter's")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--vllm", action="store_true",
                    help="also time vLLM's Triton fused_moe on the same inputs")
    args = ap.parse_args()

    # 630 MB at the real shape, and the card is usually already holding a server,
    # so the weights are built once and every token count reads the same pair.
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(0)
    e, inter, hidden = args.experts, args.inter, args.hidden
    w13 = (torch.randn(e, 2 * inter, hidden, generator=gen, device=dev, dtype=torch.bfloat16) * 0.05).contiguous()
    w2 = (torch.randn(e, hidden, inter, generator=gen, device=dev, dtype=torch.bfloat16) * 0.05).contiguous()

    for tokens in [int(t) for t in args.tokens.split(",")]:
        tile_m = args.tile_m or tile_m_for(tokens=tokens, topk=args.topk, experts=args.experts)
        run(
            tokens,
            w13=w13,
            w2=w2,
            experts=args.experts,
            topk=args.topk,
            inter=args.inter,
            hidden=args.hidden,
            tile_m=tile_m,
            check=args.check,
            vllm=args.vllm,
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
