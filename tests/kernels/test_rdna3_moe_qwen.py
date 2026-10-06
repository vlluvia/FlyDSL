#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""The RDNA3 MoE layer at Qwen3.8-Flash-Next's shapes, under vLLM's parallelism.

Qwen4Exp routes 10 of 512 experts per token through a 640-wide intermediate, with
a 640-wide shared expert on every token, at hidden 2560. vLLM shards that MoE by
tensor parallelism rather than expert parallelism: every rank holds all 512
experts and a ``640/TP`` slice of each one's intermediate. At TP=8 the per-rank
intermediate is 80.

80 is the whole difficulty. It is five WMMA tiles, so it divides no tuned block
tile -- those are 32 or 64 wide in N and 32 or 64 deep in K -- and until
``host.py`` learned to narrow generically, neither stage would build at this
shape. These tests pin that: the tables still hand the tuned shapes their tuned
tiles, and the TP-sharded shape gets *a* tile at every ``tile_m`` rather than an
error.

The shared expert is the other half. It is the same width as a routed expert, so
it does not need a second kernel or a second pair of GEMMs: stack it on as expert
``E`` and give every token an extra routing slot pointing at it, weighted by its
sigmoid gate. ``test_fused_shared_expert_matches_the_qwen_formula`` is the check
that this is the same arithmetic Qwen's block describes.

The expert count is cut down in most tests here. It does not enter the tile
choice -- that is indexed by ``(stage, k_dim, n_out, tile_m)`` -- but it does
unroll the routing kernel's expert scan, and 512 experts is minutes of MLIR per
build. One test runs the real 512 to prove the multi-pass routing carries it.
"""

import os
import sys
from enum import Enum

import pytest
import torch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for _p in (os.path.join(_REPO_ROOT, "build-fly", "python_packages"), _REPO_ROOT):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available.", allow_module_level=True)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402

_ARCH = str(get_rocm_arch())
if not _ARCH.startswith("gfx11"):
    pytest.skip(f"RDNA3 MoE requires gfx11*, got {_ARCH}", allow_module_level=True)

from kernels.moe.rdna3_moe import host  # noqa: E402
from kernels.moe.rdna3_moe.vllm_adapter import (  # noqa: E402
    DECODE_TILE_M,
    PREFILL_TILE_M,
    add_shared_slot,
    check_layer_supported,
    fuse_shared_expert,
    moe_layer_forward,
    tile_m_for,
)
from tests.test_common import verify_output  # noqa: E402

# Qwen3.8-Flash-Next's text config, as the MoE block sees it.
HIDDEN = 2560
MOE_INTERMEDIATE = 640
SHARED_INTERMEDIATE = 640
NUM_EXPERTS = 512
TOPK = 10


def _shard(tp: int) -> int:
    """The per-rank intermediate. 640/8 = 80, which is the case under test."""
    assert MOE_INTERMEDIATE % tp == 0
    return MOE_INTERMEDIATE // tp


def _weights(experts, inter, hidden=HIDDEN, seed=0, dev="cuda"):
    """``w13[E, 2*inter, hidden]`` gate-half-first and ``w2[E, hidden, inter]``."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    w13 = (torch.randn(experts, 2 * inter, hidden, generator=gen, device=dev, dtype=torch.bfloat16) * 0.05).contiguous()
    w2 = (torch.randn(experts, hidden, inter, generator=gen, device=dev, dtype=torch.bfloat16) * 0.05).contiguous()
    return w13, w2


def _route(tokens, experts, topk, seed=0, dev="cuda"):
    """A top-k routing with renormalised weights, as vLLM's router emits."""
    gen = torch.Generator(device=dev).manual_seed(seed + 1)
    logits = torch.randn(tokens, experts, generator=gen, device=dev, dtype=torch.float32)
    weights, ids = torch.topk(torch.softmax(logits, dim=-1), k=topk, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return ids.to(torch.int32).contiguous(), weights.to(torch.float32).contiguous()


def _torch_moe(x, w13, w2, topk_ids, topk_weights):
    """The eager layer: one matmul per expert over the rows routed to it.

    Gate half first, silu on the gate, routing weight folded in at the end --
    the same places the kernel puts them.
    """
    tokens, topk = topk_ids.shape
    experts, two_inter, _ = w13.shape
    inter = two_inter // 2
    out = torch.zeros(tokens, x.shape[1], dtype=torch.float32, device=x.device)
    flat_ids = topk_ids.reshape(-1)
    flat_w = topk_weights.reshape(-1)
    for e in range(experts):
        sel = (flat_ids == e).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        tok = torch.div(sel, topk, rounding_mode="floor")
        xe = x.index_select(0, tok)
        gate = xe @ w13[e][:inter].t()
        up = xe @ w13[e][inter:].t()
        h = (torch.nn.functional.silu(gate.float()) * up.float()).to(x.dtype)
        y = h @ w2[e].t()
        out.index_add_(0, tok, y.float() * flat_w[sel].unsqueeze(1))
    return out


# ── The tile tables at a TP-sharded intermediate ─────────────────────────────


@pytest.mark.parametrize("tile_m", host.SUPPORTED_TILE_M)
@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_the_tp_sharded_intermediate_builds(tp, tile_m):
    """Every TP degree Qwen4Exp can run at has a tile, at every routing height.

    TP=8 is the one that needs the generic narrowing: 80 divides no tuned
    BLOCK_N and no tuned BLOCK_K, so before ``_narrowings`` both stages raised.
    The others are here to show the tuned path is still reachable.
    """
    inter = _shard(tp)
    _, bm1, bn1, bk1 = host.compile_moe_gemm1(
        model_dim=HIDDEN, inter_dim=inter, experts=8, topk=TOPK, tile_m=tile_m
    )
    _, bm2, bn2, bk2 = host.compile_moe_gemm2(
        model_dim=HIDDEN, inter_dim=inter, experts=8, topk=TOPK, tile_m=tile_m
    )

    assert bm1 == tile_m and bm2 == tile_m
    # stage1 writes the intermediate and contracts the hidden; stage2 the other
    # way round. Both divisions have to be exact or the tile reads off the end.
    assert inter % bn1 == 0, f"gate_up BLOCK_N={bn1} does not divide inter_dim={inter}"
    assert HIDDEN % bk1 == 0
    assert HIDDEN % bn2 == 0
    assert inter % bk2 == 0, f"down BLOCK_K={bk2} does not divide inter_dim={inter}"
    # The prefetch pipeline needs two k-tiles to have anything to overlap.
    assert inter // bk2 >= 2


def test_the_tuned_shapes_keep_their_tuned_tiles():
    """Narrowing is a fallback, not a replacement.

    The generic enumeration can satisfy the shapes o3 tuned as well, so this
    pins that it stays behind the measured table rather than in front of it.
    """
    for stage, k_dim, n_out, tile_m, want in [
        ("gate_up", 2048, 768, 16, (16, 64, 64)),
        ("gate_up", 2048, 768, 64, (64, 32, 64)),
        ("down", 768, 2048, 16, (16, 64, 32)),
        ("down", 768, 2048, 64, (64, 128, 32)),
    ]:
        _, bm, bn, bk = host.compile_grouped_gemm(
            k_dim=k_dim, n_out=n_out, experts=8, stage=stage, topk=2, doweight=(stage == "down"), tile_m=tile_m
        )
        assert (bm, bn, bk) == want, f"{stage} {k_dim}x{n_out} tile_m={tile_m}"


def test_a_shape_no_tile_divides_is_still_an_error():
    """Narrowing bottoms out at the WMMA shape, and says so.

    24 is not a multiple of 16, so no block tile can cover it and the fallback
    must not quietly pick one that reads past the row.
    """
    with pytest.raises(ValueError, match="no tile for stage"):
        host.compile_grouped_gemm(k_dim=HIDDEN, n_out=24, experts=2, stage="gate_up", topk=2, tile_m=16)


# ── The layer at those shapes ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "tokens, tile_m",
    [pytest.param(32, 16, id="decode"), pytest.param(512, 64, id="prefill")],
)
def test_the_layer_matches_torch_at_tp8(tokens, tile_m):
    """The TP=8 shape computes the layer, not just a kernel that builds.

    A narrowed tile is still a tile the tables never measured, and BLOCK_N=16
    for stage1 means the scatter epilogue runs at its narrowest -- one channel
    group per lane pass. Worth checking the answer and not only the build.
    """
    experts, topk, inter = 8, 4, _shard(8)
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(0)
    x = (torch.randn(tokens, HIDDEN, generator=gen, device=dev, dtype=torch.bfloat16) * 0.1).contiguous()
    w13, w2 = _weights(experts, inter)
    topk_ids, topk_weights = _route(tokens, experts, topk)

    got = moe_layer_forward(x, w13, w2, topk_ids, topk_weights, tile_m=tile_m)
    ref = _torch_moe(x, w13, w2, topk_ids, topk_weights)

    assert got.shape == (tokens, HIDDEN)
    assert verify_output(got.float(), ref, atol=0.05, rtol=0.05)


def test_the_real_expert_count_routes():
    """512 experts, which is two routing passes, at the real hidden and shard.

    The routing kernel pairs one thread with one expert, so past 256 it sweeps
    the experts in blocks and each sweep has to start its tiles after the last
    one's. Everything else here runs a cut-down expert count to keep the MLIR
    cheap, so this is the only test that exercises the shape as deployed.
    """
    tokens, topk, inter = 8, TOPK, _shard(8)
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(0)
    x = (torch.randn(tokens, HIDDEN, generator=gen, device=dev, dtype=torch.bfloat16) * 0.1).contiguous()
    w13, w2 = _weights(NUM_EXPERTS, inter)
    topk_ids, topk_weights = _route(tokens, NUM_EXPERTS, topk)

    got = moe_layer_forward(x, w13, w2, topk_ids, topk_weights, tile_m=DECODE_TILE_M)
    ref = _torch_moe(x, w13, w2, topk_ids, topk_weights)

    assert verify_output(got.float(), ref, atol=0.05, rtol=0.05)


# ── The shared expert, as a routing slot ─────────────────────────────────────


def test_fused_shared_expert_matches_the_qwen_formula():
    """``sum_k w_k * expert_k(x) + sigmoid(gate(x)) * shared(x)``, in one launch.

    Qwen's shared expert is the same width as a routed one, so fusing it is a
    stack and an extra slot rather than a second pair of GEMMs. The reference
    keeps the two halves apart and adds them, which is what the block does.
    """
    tokens, experts, topk, inter = 32, 8, 4, _shard(8)
    assert SHARED_INTERMEDIATE == MOE_INTERMEDIATE, "the fusion needs the two widths to agree"
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(0)
    x = (torch.randn(tokens, HIDDEN, generator=gen, device=dev, dtype=torch.bfloat16) * 0.1).contiguous()
    w13, w2 = _weights(experts, inter, seed=0)
    shared_w13, shared_w2 = _weights(1, inter, seed=7)
    shared_w13, shared_w2 = shared_w13[0], shared_w2[0]
    topk_ids, topk_weights = _route(tokens, experts, topk)
    shared_gate = torch.sigmoid(torch.randn(tokens, 1, generator=gen, device=dev, dtype=torch.float32))

    fused13, fused2 = fuse_shared_expert(w13, w2, shared_w13, shared_w2)
    assert fused13.shape == (experts + 1, 2 * inter, HIDDEN)
    assert fused2.shape == (experts + 1, HIDDEN, inter)

    ids, weights = add_shared_slot(
        topk_ids, topk_weights, shared_expert_id=experts, shared_weight=shared_gate
    )
    assert ids.shape == (tokens, topk + 1)
    got = moe_layer_forward(x, fused13, fused2, ids, weights, tile_m=DECODE_TILE_M)

    routed = _torch_moe(x, w13, w2, topk_ids, topk_weights)
    gate = x.float() @ shared_w13[:inter].float().t()
    up = x.float() @ shared_w13[inter:].float().t()
    h = (torch.nn.functional.silu(gate) * up).to(x.dtype)
    shared = (h @ shared_w2.t()).float() * shared_gate

    assert verify_output(got.float(), routed + shared, atol=0.05, rtol=0.05)


def test_an_ungated_shared_slot_weighs_one():
    """No gate means weight 1, which is the plain "always on" shared expert."""
    topk_ids = torch.zeros(4, 2, dtype=torch.int32, device="cuda")
    topk_weights = torch.full((4, 2), 0.5, dtype=torch.float32, device="cuda")
    ids, weights = add_shared_slot(topk_ids, topk_weights, shared_expert_id=9)

    assert torch.equal(ids[:, -1], torch.full((4,), 9, dtype=torch.int32, device="cuda"))
    assert torch.equal(weights[:, -1], torch.ones(4, device="cuda"))
    assert torch.equal(weights[:, :-1], topk_weights)


def test_fusing_checks_the_shared_expert_is_expert_shaped():
    """A shared expert of a different width cannot be stacked, and must say so.

    Qwen4Exp happens to have ``shared_expert_intermediate_size ==
    moe_intermediate_size``; a model where they differ needs the separate path,
    not a silent reshape.
    """
    w13, w2 = _weights(4, 80)
    with pytest.raises(ValueError, match="shared_w13 must be"):
        fuse_shared_expert(w13, w2, torch.zeros(2 * 96, HIDDEN, device="cuda"), w2[0])
    with pytest.raises(ValueError, match="shared_w2 must be"):
        fuse_shared_expert(w13, w2, w13[0], torch.zeros(HIDDEN, 96, device="cuda"))


# ── Picking the routing height ───────────────────────────────────────────────


def test_the_tile_height_follows_rows_per_expert_not_batch_size():
    """512 experts keeps a decode-sized batch on the short tile.

    The tall tile only pays once an expert has the rows to fill it, and with
    this many experts that does not happen until the batch is in the thousands
    -- a 256-token batch is half a row per expert, where a 64-row tile would be
    99% padding.
    """
    assert tile_m_for(tokens=1, topk=TOPK, experts=NUM_EXPERTS) == DECODE_TILE_M
    assert tile_m_for(tokens=256, topk=TOPK, experts=NUM_EXPERTS) == DECODE_TILE_M
    assert tile_m_for(tokens=8192, topk=TOPK, experts=NUM_EXPERTS) == PREFILL_TILE_M
    # Few experts fill a tall tile much sooner.
    assert tile_m_for(tokens=512, topk=2, experts=8) == PREFILL_TILE_M


# ── What the adapter refuses ─────────────────────────────────────────────────


class _Layer:
    """The attributes ``check_layer_supported`` reads off a vLLM RoutedExperts.

    A stand-in rather than the real class so this runs without vLLM installed;
    the names are checked against the real one by the method module importing it.
    """

    def __init__(self, **kw):
        self.expert_map = None
        self.activation = "silu"
        self.apply_router_weight_on_input = False
        self.w13_bias = None
        self.w2_bias = None
        self.__dict__.update(kw)


def test_a_tp_only_layer_is_accepted():
    check_layer_supported(_Layer(), torch.bfloat16)
    check_layer_supported(_Layer(), torch.float16)


def test_vllm_silu_enum_is_accepted():
    """vLLM carries the activation as ``MoEActivation.SILU``, not a string."""

    class _Activation(Enum):
        SILU = "silu"

    check_layer_supported(_Layer(activation=_Activation.SILU), torch.bfloat16)


def test_expert_parallelism_is_refused_rather_than_miscomputed():
    """An expert map means some routed experts are on another rank.

    The grouped GEMM gathers every routing row through a local weight slab, so
    a global id it does not own would silently read the wrong expert. This is
    the one refusal that protects a wrong answer rather than a crash.
    """
    layer = _Layer(expert_map=torch.zeros(8, dtype=torch.int32))
    with pytest.raises(NotImplementedError, match="tensor-parallel only"):
        check_layer_supported(layer, torch.bfloat16)


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"activation": "gelu"}, "silu"),
        ({"apply_router_weight_on_input": True}, "apply_router_weight_on_input"),
        ({"w13_bias": torch.zeros(1)}, "no bias term"),
        ({"w2_bias": torch.zeros(1)}, "no bias term"),
    ],
)
def test_unsupported_layer_options_are_refused(kw, match):
    with pytest.raises(NotImplementedError, match=match):
        check_layer_supported(_Layer(**kw), torch.bfloat16)


def test_fp32_activations_are_refused():
    """The WMMA operand types are 16-bit; f32 in would need a different mma."""
    with pytest.raises(NotImplementedError, match="bf16 or f16"):
        check_layer_supported(_Layer(), torch.float32)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
