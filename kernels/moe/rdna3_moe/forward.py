# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""The single-rank RDNA3 MoE layer: gating, routing, both GEMMs, the reduce.

Five launches per layer:

    gating   logits          -> topk_ids, topk_weights     (optional)
    routing  topk_ids        -> sorted_ids, expert_ids      counting sort
    gemm1    x               -> a2[tokens, topk, inter]     silu(gate) * up
    gemm2    a2              -> y [tokens, topk, model]     down, weighted
    reduce   y               -> out[tokens, model]          sum over topk

The topk sum is its own launch because gfx11 has no bf16 atomic, so gemm2 cannot
accumulate into a ``[tokens, model_dim]`` output the way the CDNA kernels do. It
folds the routing weight in instead and leaves an unweighted sum, which is what
``compile_moe_reduction`` computes -- one of the two CDNA MoE kernels that runs
here unchanged. The gating kernel is the other.

Everything is this rank's: ``w1``/``w2`` carry this rank's experts and ``x`` is
whatever rows a dispatch left here. At EP=1 the dispatch is the identity, and
that is the only case wired up -- an EP>1 caller does its own dispatch and
combine around these calls.
"""

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from kernels.common.tensor_shim import _run_compiled
from kernels.moe.moe_gemm_2stage import compile_moe_reduction
from kernels.moe.rdna3_moe.host import compile_moe_gemm1, compile_moe_gemm2
from kernels.moe.rdna3_moe.routing import Routing, build_routing
from kernels.moe.rdna3_moe.routing_kernel import build_routing_device
from kernels.moe.topk_gating_softmax_kernel import build_topk_gating_softmax_module

_REDUCE_DTYPES = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}


@functools.lru_cache(maxsize=64)
def _reduce_exe(topk: int, model_dim: int, dtype_str: str):
    return compile_moe_reduction(topk=topk, model_dim=model_dim, dtype_str=dtype_str)


@functools.lru_cache(maxsize=64)
def _gating_launch(experts: int, topk: int, dtype_str: str, renormalize: bool):
    return build_topk_gating_softmax_module(experts, topk, dtype_str, renormalize)


def _ptr(t: torch.Tensor):
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


def moe_gating(gating_logits: torch.Tensor, *, topk: int, renormalize: bool = True, stream=None):
    """``[tokens, experts]`` logits -> ``(topk_ids int32, topk_weights f32)``.

    This is the CDNA package's fused softmax + top-K kernel. It reduces inside a
    lane group of ``experts // VPT`` lanes with ``shuffle_xor``, so it is
    wave-size agnostic and runs here as built -- but the lane group has to be a
    power of two no wider than a wave, which on wave32 rules out some expert
    counts. A caller with one of those computes its top-K itself and calls
    ``moe_forward`` directly.
    """
    if gating_logits.dim() != 2:
        raise ValueError(f"gating_logits must be [tokens, experts], got {tuple(gating_logits.shape)}")
    tokens, experts = gating_logits.shape
    dtype_str = _REDUCE_DTYPES.get(gating_logits.dtype)
    if dtype_str is None:
        raise ValueError(f"gating_logits dtype must be bf16/f16/f32, got {gating_logits.dtype}")

    launch = _gating_launch(int(experts), int(topk), dtype_str, bool(renormalize))
    dev = gating_logits.device
    weights = torch.empty(tokens, topk, dtype=torch.float32, device=dev)
    ids = torch.empty(tokens, topk, dtype=torch.int32, device=dev)
    # The kernel also emits vLLM's token_expert_indices, which nothing here reads.
    tei = torch.empty(tokens, topk, dtype=torch.int32, device=dev)
    # Via the CompiledFunction, for the reason in ``grouped_gemm.launch``.
    _run_compiled(
        launch,
        gating_logits.contiguous(),
        weights,
        ids,
        tei,
        fx.Int32(int(tokens)),
        stream if stream is not None else torch.cuda.current_stream(),
    )
    return ids, weights


def moe_reduce(y: torch.Tensor, *, out: torch.Tensor | None = None, stream=None) -> torch.Tensor:
    """Sum ``[tokens, topk, model_dim] -> [tokens, model_dim]``.

    Unweighted: gemm2 already folded the routing weight in.
    """
    if y.dim() != 3:
        raise ValueError(f"y must be [tokens, topk, model_dim], got {tuple(y.shape)}")
    tokens, topk, model_dim = y.shape
    dtype_str = _REDUCE_DTYPES.get(y.dtype)
    if dtype_str is None:
        raise ValueError(f"y dtype must be bf16/f16/f32, got {y.dtype}")

    if out is None:
        out = torch.empty(tokens, model_dim, dtype=y.dtype, device=y.device)
    exe = _reduce_exe(int(topk), int(model_dim), dtype_str)
    # The reducer takes an expert mask and the routing ids for the EP case where
    # a slot can be invalid. Here every slot is valid, and it reads neither.
    unused = torch.empty(0, dtype=torch.int32, device=y.device)
    _run_compiled(
        exe,
        _ptr(y),
        _ptr(out),
        _ptr(unused),
        _ptr(unused),
        int(tokens),
        stream if stream is not None else torch.cuda.current_stream(),
    )
    return out


def moe_forward(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    tile_m: int = 16,
    routing: Routing | None = None,
    routing_impl: str = "device",
    out: torch.Tensor | None = None,
    stream=None,
) -> torch.Tensor:
    """One MoE layer from a routing that is already decided.

    ``x`` is ``[tokens, model_dim]``, ``w1`` is ``[experts, 2*inter_dim,
    model_dim]`` gate-half-first, ``w2`` is ``[experts, model_dim, inter_dim]``,
    and ``topk_weights`` is f32 because the kernel folds it in as f32.

    ``tile_m`` is the caller's: it decides both the routing padding and the block
    tile, and the right choice depends on how many rows an expert gets. See the
    table in ``host.py`` -- a prefill wants a taller tile than a decode, and the
    two differ by more than the measurement noise. How many rows an expert gets
    is also passed down to the tile choice itself, since for ``down`` the shape
    and ``tile_m`` together do not determine the best tile; see
    ``host.tile_bucket_for``.

    ``routing`` may be passed to reuse a routing already built for this
    ``topk_ids`` and ``tile_m``, which is the whole reason the two GEMMs take the
    same buffers. Otherwise ``routing_impl`` picks who builds it:

      ``"device"``        the routing kernel, and nothing waits for it: the grid
                          is an upper bound on the tile count and the GEMMs read
                          the real one off the device.
      ``"device-exact"``  the same kernel, with the count read back to the host.
      ``"host"``          the numpy builder, which is the faster of the two once
                          the expert count is past a few tens -- see
                          ``routing_kernel``'s note on where the kernel runs out.
    """
    if x.dim() != 2:
        raise ValueError(f"x must be [tokens, model_dim], got {tuple(x.shape)}")
    if w1.dim() != 3 or w2.dim() != 3:
        raise ValueError(f"weights must be 3-D, got w1={tuple(w1.shape)} w2={tuple(w2.shape)}")

    tokens, model_dim = x.shape
    experts, two_inter, w1_k = w1.shape
    inter_dim = two_inter // 2
    topk = topk_ids.shape[1]

    if two_inter != 2 * inter_dim:
        raise ValueError(f"w1's row count must be 2*inter_dim, got {two_inter}")
    if w1_k != model_dim:
        raise ValueError(f"w1 contracts {w1_k}, but x has model_dim={model_dim}")
    if tuple(w2.shape) != (experts, model_dim, inter_dim):
        raise ValueError(f"w2 must be {(experts, model_dim, inter_dim)}, got {tuple(w2.shape)}")
    if topk_ids.shape[0] != tokens or topk_weights.shape != topk_ids.shape:
        raise ValueError(
            f"routing must be [tokens={tokens}, topk], got ids={tuple(topk_ids.shape)} "
            f"weights={tuple(topk_weights.shape)}"
        )
    if topk_weights.dtype != torch.float32:
        raise ValueError(f"topk_weights must be f32, got {topk_weights.dtype}")

    if routing is None:
        if routing_impl == "device":
            routing = build_routing_device(topk_ids, experts=experts, tile_m=tile_m, stream=stream)
        elif routing_impl == "device-exact":
            routing = build_routing_device(topk_ids, experts=experts, tile_m=tile_m, exact=True, stream=stream)
        elif routing_impl == "host":
            routing = build_routing(topk_ids, experts=experts, tile_m=tile_m)
        else:
            raise ValueError(f"routing_impl must be 'device', 'device-exact' or 'host', got {routing_impl!r}")
    elif (routing.tile_m, routing.tokens, routing.topk) != (tile_m, tokens, topk):
        raise ValueError(
            f"the routing was built for tile_m={routing.tile_m}, tokens={routing.tokens}, "
            f"topk={routing.topk}, not tile_m={tile_m}, tokens={tokens}, topk={topk}"
        )

    in_dtype = _REDUCE_DTYPES.get(x.dtype)
    if in_dtype not in ("bf16", "f16"):
        raise ValueError(f"x dtype must be bf16 or f16, got {x.dtype}")

    # A routing that only knows a bound on its tile count needs the build that
    # reads the real one off the device; an exact one does not, and gets the
    # build without the extra scalar load.
    bounded = not routing.exact
    # How many rows an expert averages, which is the axis the tile tables cannot
    # read off the shape -- see host.tile_bucket_for. The average, not the real
    # per-expert counts: those live on the device and waiting for them would undo
    # o2. It only has to land in the right bucket of three, and a routing skewed
    # enough to cross a bucket boundary is skewed enough that its tail dominates
    # anyway.
    rows_per_expert = tokens * topk / max(experts, 1)
    gemm1, _, _, _ = compile_moe_gemm1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        in_dtype=in_dtype,
        out_dtype=in_dtype,
        bounded_blocks=bounded,
        rows_per_expert=rows_per_expert,
    )
    gemm2, _, _, _ = compile_moe_gemm2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        doweight=True,
        tile_m=tile_m,
        in_dtype=in_dtype,
        out_dtype=in_dtype,
        bounded_blocks=bounded,
        rows_per_expert=rows_per_expert,
    )

    if stream is None:
        stream = torch.cuda.current_stream()
    # Both intermediates are fully written: every (token, slot) appears exactly
    # once in the routing, so no row is left holding whatever was in the
    # allocation. Only the padded rows go unstored, and they address nothing.
    a2 = torch.empty(tokens, topk, inter_dim, dtype=x.dtype, device=x.device)
    y = torch.empty(tokens, topk, model_dim, dtype=x.dtype, device=x.device)
    unused = torch.empty(0, dtype=torch.float32, device=x.device)

    nb_dev = routing.num_blocks_device if bounded else None
    gemm1(a2, x, w1, routing.sorted_ids, routing.expert_ids, unused, tokens, routing.num_blocks, stream, nb_dev)
    gemm2(y, a2, w2, routing.sorted_ids, routing.expert_ids, topk_weights, tokens, routing.num_blocks, stream, nb_dev)
    return moe_reduce(y, out=out, stream=stream)


def moe_forward_from_logits(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_logits: torch.Tensor,
    *,
    topk: int,
    tile_m: int = 16,
    renormalize: bool = True,
    out: torch.Tensor | None = None,
    stream=None,
) -> torch.Tensor:
    """``moe_forward`` with the gating in front of it."""
    topk_ids, topk_weights = moe_gating(gating_logits, topk=topk, renormalize=renormalize, stream=stream)
    return moe_forward(
        x, w1, w2, topk_ids, topk_weights, tile_m=tile_m, out=out, stream=stream
    )
