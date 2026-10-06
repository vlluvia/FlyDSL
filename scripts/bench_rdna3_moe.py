#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Benchmark the RDNA3 MoE kernels against a torch baseline, stage by stage.

The baseline is the same arithmetic in eager torch: group the routed rows by
expert, then one ``matmul`` per expert. That is what a MoE layer looks like
before anyone writes a grouped GEMM, and it is the thing the kernels have to
beat. Its per-expert python loop is part of the cost, which is fair -- the loop
is why the baseline is slow -- but it does mean the baseline gets worse as the
expert count grows at fixed token count.

FLOP counts are the *useful* ones: ``2 * tokens * topk * N * K``, counting the
routed rows only. The padding a tile needs is work the kernel does and does not
get credit for, so a shape whose experts each land a few rows past a tile
boundary reads low here -- which is the honest way round.

The topk reduce and the routing build are reported separately, in microseconds:
neither is a GEMM, and rolling them into a TFLOP/s figure would only hide them.
``layer`` is the GEMMs with a routing handed to them; ``layer+route`` builds one
per call, either on the host or with the o1 kernel, and is what a decode step
pays.

    python scripts/bench_rdna3_moe.py
    python scripts/bench_rdna3_moe.py --tokens 32,1024 --experts 8 --topk 2
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_BUILD = os.path.join(_REPO, "build-fly", "python_packages")
if os.path.isdir(_BUILD) and _BUILD not in sys.path:
    sys.path.insert(0, _BUILD)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.moe.rdna3_moe.forward import moe_forward, moe_gating, moe_reduce  # noqa: E402
from kernels.moe.rdna3_moe.host import SUPPORTED_TILE_M, compile_grouped_gemm, compile_moe_gemm1, compile_moe_gemm2  # noqa: E402
from kernels.moe.rdna3_moe.routing import build_routing  # noqa: E402
from kernels.moe.rdna3_moe.routing_kernel import build_routing_device  # noqa: E402


def bench_us(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters


def tflops(flop: float, us: float) -> float:
    return flop / (us * 1e-6) / 1e12


# ── The torch baseline: group by expert, one matmul each ─────────────────────


def _grouped(topk_ids: torch.Tensor, experts: int):
    """Row order and per-expert counts, the same grouping the kernel walks."""
    topk = topk_ids.shape[1]
    flat = topk_ids.reshape(-1).to(torch.int64)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=experts)[:experts].tolist()
    return order, torch.div(order, topk, rounding_mode="floor"), counts


def torch_linear(x, w, rows, counts):
    out = torch.empty(rows.numel(), w.shape[1], dtype=x.dtype, device=x.device)
    xg = x[rows]
    off = 0
    for e, c in enumerate(counts):
        if c:
            out[off : off + c] = xg[off : off + c] @ w[e].t()
        off += c
    return out


def torch_gemm1(x, w1, rows, counts):
    inter = w1.shape[1] // 2
    gu = torch_linear(x, w1, rows, counts)
    return F.silu(gu[:, :inter].float()).to(x.dtype) * gu[:, inter:]


def torch_gemm2(a2_rows, w2, weights_rows, counts):
    """``a2_rows`` is already in routing order, one row per routed slot."""
    out = torch.empty(a2_rows.shape[0], w2.shape[1], dtype=a2_rows.dtype, device=a2_rows.device)
    off = 0
    for e, c in enumerate(counts):
        if c:
            out[off : off + c] = a2_rows[off : off + c] @ w2[e].t()
        off += c
    return out * weights_rows.to(out.dtype).unsqueeze(1)


def torch_layer(x, w1, w2, topk_ids, topk_weights):
    experts = w1.shape[0]
    order, rows, counts = _grouped(topk_ids, experts)
    h = torch_gemm1(x, w1, rows, counts)
    y = torch_gemm2(h, w2, topk_weights.reshape(-1)[order], counts)
    out = torch.zeros(x.shape[0], w2.shape[1], dtype=x.dtype, device=x.device)
    return out.index_add_(0, rows, y)


# ── One shape, every stage, every tile ───────────────────────────────────────


def run_shape(*, tokens, model_dim, inter_dim, experts, topk, dtype, tiles, warmup, iters, check):
    dev = "cuda"
    torch.manual_seed(0)
    x = (torch.randn(tokens, model_dim, dtype=dtype, device=dev) * 0.1).contiguous()
    w1 = (torch.randn(experts, 2 * inter_dim, model_dim, dtype=dtype, device=dev) * 0.05).contiguous()
    w2 = (torch.randn(experts, model_dim, inter_dim, dtype=dtype, device=dev) * 0.05).contiguous()
    logits = (torch.randn(tokens, experts, dtype=dtype, device=dev) * 2.0).contiguous()

    topk_ids, topk_weights = moe_gating(logits, topk=topk)
    torch.cuda.synchronize()
    # ``order`` is the flat (token, slot) index of each routing row, so it is both
    # the baseline's gather and the permutation that puts a kernel's scattered
    # output back into routing order.
    order, rows_idx, counts = _grouped(topk_ids, experts)
    w_gate = w1[:, :inter_dim].contiguous()
    empty = torch.empty(0, dtype=torch.float32, device=dev)
    stream = torch.cuda.current_stream()

    routed = tokens * topk
    flop_lin = 2.0 * routed * inter_dim * model_dim
    flop_g1 = 2.0 * routed * (2 * inter_dim) * model_dim
    flop_g2 = 2.0 * routed * model_dim * inter_dim

    # Stage inputs in routing order, so both sides see the same rows.
    a2_rows = torch_gemm1(x, w1, rows_idx, counts)
    w_rows = topk_weights.reshape(-1)[order]
    a2 = torch.empty(tokens, topk, inter_dim, dtype=dtype, device=dev)
    a2.reshape(-1, inter_dim)[order] = a2_rows

    results = []  # (stage, impl, tile_m, us, flop, maxdiff)

    def record(stage, impl, tile_m, fn, flop, got=None, ref=None):
        us = bench_us(fn, warmup=warmup, iters=iters)
        diff = float("nan")
        if check and got is not None and ref is not None:
            diff = (got().float() - ref.float()).abs().max().item()
        results.append((stage, impl, tile_m, us, flop, diff))

    # Baselines first: they define the reference values the kernels are checked on.
    ref_lin = torch_linear(x, w_gate, rows_idx, counts)
    ref_g2 = torch_gemm2(a2_rows, w2, w_rows, counts)
    ref_layer = torch_layer(x, w1, w2, topk_ids, topk_weights)
    record("v1 linear", "torch", None, lambda: torch_linear(x, w_gate, rows_idx, counts), flop_lin)
    record("v2 gemm1", "torch", None, lambda: torch_gemm1(x, w1, rows_idx, counts), flop_g1)
    record("v3 gemm2", "torch", None, lambda: torch_gemm2(a2_rows, w2, w_rows, counts), flop_g2)
    record("layer", "torch", None, lambda: torch_layer(x, w1, w2, topk_ids, topk_weights), flop_g1 + flop_g2)
    # The baseline builds its grouping every call, so it is the baseline for both
    # the routing-excluded and the routing-included row.
    record("layer+route", "torch", None, lambda: torch_layer(x, w1, w2, topk_ids, topk_weights), flop_g1 + flop_g2)

    y_k = None
    for tile_m in tiles:
        try:
            r = build_routing(topk_ids, experts=experts, tile_m=tile_m)
            lin, _, _, _ = compile_grouped_gemm(
                k_dim=model_dim, n_out=inter_dim, experts=experts, stage="linear", tile_m=tile_m
            )
            g1, _, _, _ = compile_moe_gemm1(
                model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk, tile_m=tile_m
            )
            g2, _, _, _ = compile_moe_gemm2(
                model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk, doweight=True, tile_m=tile_m
            )
        except (ValueError, RuntimeError) as exc:
            print(f"  tile_m={tile_m}: skipped ({exc})", flush=True)
            continue

        c_lin = torch.empty(r.rows, inter_dim, dtype=dtype, device=dev)
        a2_k = torch.empty(tokens, topk, inter_dim, dtype=dtype, device=dev)
        y_k = torch.empty(tokens, topk, model_dim, dtype=dtype, device=dev)
        out_k = torch.empty(tokens, model_dim, dtype=dtype, device=dev)

        valid = (r.sorted_ids & 0xFFFFFF) < tokens

        record(
            "v1 linear",
            "flydsl",
            tile_m,
            lambda: lin(c_lin, x, w_gate, r.sorted_ids, r.expert_ids, empty, tokens, r.num_blocks, stream),
            flop_lin,
            got=lambda: c_lin[valid],
            ref=ref_lin,
        )
        record(
            "v2 gemm1",
            "flydsl",
            tile_m,
            lambda: g1(a2_k, x, w1, r.sorted_ids, r.expert_ids, empty, tokens, r.num_blocks, stream),
            flop_g1,
            got=lambda: a2_k.reshape(-1, inter_dim)[order],
            ref=a2_rows,
        )
        record(
            "v3 gemm2",
            "flydsl",
            tile_m,
            lambda: g2(y_k, a2, w2, r.sorted_ids, r.expert_ids, topk_weights, tokens, r.num_blocks, stream),
            flop_g2,
            got=lambda: y_k.reshape(-1, model_dim)[order],
            ref=ref_g2,
        )
        record(
            "layer",
            "flydsl",
            tile_m,
            lambda: moe_forward(x, w1, w2, topk_ids, topk_weights, tile_m=tile_m, routing=r, out=out_k),
            flop_g1 + flop_g2,
            got=lambda: out_k,
            ref=ref_layer,
        )
        # Same layer, routing built per call: what a decode step actually pays,
        # and the only row where o1 shows up.
        for impl, kind in (("dev-rt", "device"), ("exact", "device-exact"), ("host-rt", "host")):
            record(
                "layer+route",
                impl,
                tile_m,
                lambda kind=kind: moe_forward(
                    x, w1, w2, topk_ids, topk_weights, tile_m=tile_m, routing_impl=kind, out=out_k
                ),
                flop_g1 + flop_g2,
                got=lambda: out_k,
                ref=ref_layer,
            )

    # The two launches that are not GEMMs, plus the routing build. Reported in
    # microseconds because a TFLOP/s figure for them would be meaningless.
    aux = [
        ("gating", lambda: moe_gating(logits, topk=topk)),
        (f"route/host@{tiles[0]}", lambda: build_routing(topk_ids, experts=experts, tile_m=tiles[0])),
        (
            f"route/dev@{tiles[0]}",
            lambda: build_routing_device(topk_ids, experts=experts, tile_m=tiles[0]),
        ),
    ]
    if y_k is not None:
        aux.append(("reduce", lambda: moe_reduce(y_k, out=out_k)))
    aux_us = [(name, bench_us(fn, warmup=warmup, iters=iters)) for name, fn in aux]
    return results, aux_us


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="32,256,1024")
    p.add_argument("--model-dim", type=int, default=2048)
    p.add_argument("--inter-dim", type=int, default=768)
    p.add_argument("--experts", type=int, default=8, help="experts on this rank")
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--dtype", default="bf16", choices=("bf16", "f16"))
    p.add_argument("--tiles", default=",".join(str(t) for t in SUPPORTED_TILE_M))
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--no-check", action="store_true", help="skip the against-torch check")
    args = p.parse_args()

    arch = str(get_rocm_arch() or "")
    if not arch.startswith("gfx11"):
        print(f"ERROR: rdna3_moe needs gfx11*, got {arch!r}")
        return 1
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    tiles = [int(t) for t in args.tiles.split(",") if t.strip()]

    print(f"device={torch.cuda.get_device_name(0)} arch={arch} dtype={args.dtype}")
    print(
        f"model_dim={args.model_dim} inter_dim={args.inter_dim} experts={args.experts} "
        f"topk={args.topk} (this rank)"
    )
    print("baseline: eager torch, rows grouped by expert, one matmul per expert")

    for tokens in (int(t) for t in args.tokens.split(",") if t.strip()):
        print(f"\n{'=' * 78}\ntokens={tokens}  routed rows={tokens * args.topk}\n{'=' * 78}")
        results, aux_us = run_shape(
            tokens=tokens,
            model_dim=args.model_dim,
            inter_dim=args.inter_dim,
            experts=args.experts,
            topk=args.topk,
            dtype=dtype,
            tiles=tiles,
            warmup=args.warmup,
            iters=args.iters,
            check=not args.no_check,
        )

        base = {stage: us for stage, impl, _, us, _, _ in results if impl == "torch"}
        print(f"{'stage':12s} {'impl':7s} {'tile_m':>6s} {'us':>9s} {'TFLOP/s':>8s} {'vs torch':>9s}   maxdiff")
        print("-" * 78)
        for stage, impl, tile_m, us, flop, diff in results:
            tile = "-" if tile_m is None else str(tile_m)
            rel = "-" if impl == "torch" else f"{base[stage] / us:.2f}x"
            d = "" if diff != diff else f"{diff:.2e}"
            print(f"{stage:12s} {impl:7s} {tile:>6s} {us:9.1f} {tflops(flop, us):8.2f} {rel:>9s}   {d}")
        print("-" * 78)
        print("  " + "   ".join(f"{name}={us:.1f}us" for name, us in aux_us))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
