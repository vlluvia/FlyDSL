# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Shared helpers for the o1-o6 reproduction scripts.

Every script in this folder is standalone -- run it directly, no arguments
needed for the default case. This module only holds the things they all need:
the import path, a timing loop, and the inputs a MoE layer takes.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# repro/ -> rdna3_moe_optimization/ -> docs/ -> repo root
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_BUILD = os.path.join(_REPO, "build-fly", "python_packages")
if os.path.isdir(_BUILD) and _BUILD not in sys.path:
    sys.path.insert(0, _BUILD)

import torch  # noqa: E402

from flydsl.runtime.device import get_rocm_arch  # noqa: E402

REPO = _REPO


def require_gfx11() -> str:
    """These kernels are wave32 WMMA; refuse to report numbers from anything else."""
    arch = str(get_rocm_arch() or "")
    if not arch.startswith("gfx11"):
        raise SystemExit(f"ERROR: these kernels need gfx11* (RDNA3), got {arch!r}")
    return arch


def bench_us(fn, *, warmup=10, iters=50):
    """Microseconds per call, timed with CUDA events.

    Events sit on the stream, so a host-bound step still shows up: the GPU idles
    waiting for the next launch and the gap lands between the two events. That is
    what makes this usable for the routing comparisons, where the thing being
    measured is partly CPU work.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters


def make_layer_inputs(*, tokens, model_dim, inter_dim, experts, topk, device="cuda", seed=0):
    """The tensors one MoE layer takes, with a routing drawn from real gating.

    Returns ``(x, w1, w2, topk_ids, topk_weights)``. ``w1`` is gate-half-first at
    ``[experts, 2*inter_dim, model_dim]`` and ``w2`` is ``[experts, model_dim,
    inter_dim]``; both are TN, which is what the kernel's B stream expects.
    """
    from kernels.moe.rdna3_moe.forward import moe_gating

    gen = torch.Generator(device=device).manual_seed(seed)
    x = (torch.randn(tokens, model_dim, generator=gen, device=device, dtype=torch.bfloat16) * 0.1).contiguous()
    w1 = (
        torch.randn(experts, 2 * inter_dim, model_dim, generator=gen, device=device, dtype=torch.bfloat16) * 0.05
    ).contiguous()
    w2 = (
        torch.randn(experts, model_dim, inter_dim, generator=gen, device=device, dtype=torch.bfloat16) * 0.05
    ).contiguous()
    logits = torch.randn(tokens, experts, generator=gen, device=device, dtype=torch.float32).contiguous()
    topk_ids, topk_weights = moe_gating(logits, topk=topk)
    return x, w1, w2, topk_ids, topk_weights


def layer_flops(*, tokens, model_dim, inter_dim, topk):
    """Useful FLOPs in one MoE layer: gate+up, then down.

    "Useful" means the routed rows only. Rows the routing padded a tile out with
    are real work for the GPU but not part of the answer, which is exactly the
    gap o6 is about -- see roofline.py.
    """
    rows = tokens * topk
    return 2.0 * rows * (2 * inter_dim) * model_dim + 2.0 * rows * model_dim * inter_dim


def torch_moe_layer(x, w1, w2, topk_ids, topk_weights):
    """The eager baseline: group rows by expert, one matmul per expert.

    This is what "vs torch" means in the tables -- not a fused library kernel,
    but the loop a framework runs when nobody has written a MoE kernel. ``w1`` is
    gate-half-first, and the routing weight is folded in at the end, matching
    where the FlyDSL kernel folds it.
    """
    tokens, topk = topk_ids.shape
    experts, two_i, model_dim = w1.shape
    inter = two_i // 2
    out = torch.zeros(tokens, model_dim, dtype=torch.float32, device=x.device)
    flat_ids = topk_ids.reshape(-1)
    flat_w = topk_weights.reshape(-1)
    for e in range(experts):
        sel = (flat_ids == e).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        tok = torch.div(sel, topk, rounding_mode="floor")
        xe = x.index_select(0, tok)
        gate = xe @ w1[e][:inter].t()
        up = xe @ w1[e][inter:].t()
        h = (torch.nn.functional.silu(gate.float()) * up.float()).to(x.dtype)
        y = h @ w2[e].t()
        out.index_add_(0, tok, y.float() * flat_w[sel].unsqueeze(1))
    return out.to(x.dtype)


def header(title, subtitle=""):
    print()
    print("=" * 78)
    print(title)
    if subtitle:
        print(subtitle)
    print("=" * 78)


def section(title):
    print()
    print(f"--- {title} " + "-" * max(0, 73 - len(title)))


def device_banner():
    arch = require_gfx11()
    print(f"device={torch.cuda.get_device_name(0)}  arch={arch}")
    return arch
