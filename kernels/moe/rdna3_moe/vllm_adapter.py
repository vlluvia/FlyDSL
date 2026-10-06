# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Running the RDNA3 MoE layer under vLLM, for Qwen3.8-Flash-Next (Qwen4Exp).

The kernel in ``forward.py`` already computes what a vLLM MoE layer computes, so
this module is mostly a translation of names and layouts rather than new
arithmetic. What it has to reconcile is three things.

**The weight layout is already right.** vLLM stacks the routed experts as
``w13_weight[E, 2*I, H]`` with the gate half first and ``w2_weight[E, H, I]``,
which is exactly ``moe_forward``'s ``w1``/``w2``. Nothing is transposed or
copied; the adapter hands the parameters straight through.

**Tensor parallelism shrinks the intermediate, not the expert count.** Qwen4Exp
is ``moe_intermediate_size=640`` over 512 experts, and vLLM's default MoE
parallelism for this model is TP, not EP -- every rank holds all 512 experts and
a 1/TP slice of each one's intermediate. At TP=8 that is ``I=80`` per rank, which
divides none of the tuned block tiles (they are all 32 or 64 wide in N). The
generic narrowing in ``host.py`` is what makes that build; see ``_narrowings``.
It builds at every ``tile_m``, at ``BLOCK_N=16`` for stage1 and ``BLOCK_K=16``
for stage2, both of which divide 80.

**The shared expert is a routing slot.** Qwen's block computes

    out = sum_k w_k * expert_k(x)  +  sigmoid(gate(x)) * shared(x)

and ``shared_expert_intermediate_size`` equals ``moe_intermediate_size`` (640),
so the shared expert has the same shape as a routed one. vLLM already knows how
to exploit that -- ``maybe_fuse_shared_experts`` stacks it on as expert ``E`` at
load time and the combined gate emits ``E+1`` logits -- and when that path is on,
the quant method is handed ``E+1`` experts and ``topk+1`` slots and there is
nothing left for this adapter to special-case: the grouped GEMM runs the shared
expert as one more row per token, and ``moe_reduce`` sums it in with the rest.
That is the configuration this adapter is written for, and
``fuse_shared_expert`` below does the same stacking for a caller outside vLLM.

When the fused path is *off*, vLLM passes a separate ``SharedExperts`` module.
The runner has already launched that module before it calls the adapter, then
reads its output afterward, adds the two, and performs one all-reduce. The
adapter therefore only runs the routed half. Either way it never all-reduces:
its output is this rank's routed partial, which is what the runner expects.

What this does not do
---------------------
Expert parallelism. ``topk_ids`` under EP name global experts and some of them
live elsewhere, which the grouped GEMM has no way to express -- it gathers every
routing row through a local weight slab. ``kernels/moe/rdna3_moe/ep.py`` is the
expert-parallel path and it owns its own dispatch; the two are not composed here.
The adapter raises rather than silently computing a wrong answer.

A note on where this is efficient
---------------------------------
What decides the cost of a decode here is how many tiles the routing produces,
because every tile streams its expert's weight slab out of DRAM whether or not
it has rows to multiply by it. One token over 11 slots touches 11 experts, so
the honest bill is 11 slabs -- 12.9 MiB, about 16 us at this card's read
bandwidth -- and the measured layer is 40-50 us, the rest being four kernel
launches over data far too small to fill the machine.

It was not always: routing used to give every expert a tile whether or not a
token chose it, on the grounds that a tile of sentinels gathers zeros and stores
nothing. It does, but it reads the weights first, and at 513 experts a one-token
step was 513 tiles for 11 rows of work -- 601 MiB against 12.9, and a layer of
1030 us. Dropping the empty tiles is in ``routing_kernel``'s ``_tiles``; the
bound the caller allocates and launches over follows it in ``max_blocks_for``.

``tile_m=16`` is the decode default because it is the shortest the WMMA shape
allows, and with this many experts an expert has one or two rows at decode
whatever the batch -- see ``tile_m_for``.
"""

from __future__ import annotations

import torch

from kernels.moe.rdna3_moe.forward import moe_forward

__all__ = [
    "PREFILL_TILE_M",
    "DECODE_TILE_M",
    "add_shared_slot",
    "check_layer_supported",
    "fuse_shared_expert",
    "moe_layer_forward",
    "tile_m_for",
]

SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


def check_layer_supported(layer, dtype: torch.dtype) -> None:
    """Refuse what the grouped GEMM cannot express, rather than miscompute it.

    Takes a vLLM ``RoutedExperts`` but only reads attributes off it, so it is
    here rather than next to the method that calls it -- this is a statement
    about the kernel, and it should be checkable without importing vLLM.
    """
    if getattr(layer, "expert_map", None) is not None:
        # Under EP a routing id names a global expert and some of them are on
        # another rank; the grouped GEMM gathers every row through a local slab
        # and has no way to skip one. ep.py is the expert-parallel path, and it
        # does its own dispatch rather than composing with this.
        raise NotImplementedError(
            "the RDNA3 MoE kernel is tensor-parallel only; run with expert parallelism off, "
            "or use kernels.moe.rdna3_moe.ep for an expert-parallel layer"
        )
    if dtype not in SUPPORTED_DTYPES:
        raise NotImplementedError(f"the RDNA3 MoE kernel takes bf16 or f16 activations, got {dtype}")
    activation = getattr(layer, "activation", "silu")
    activation_name = getattr(activation, "value", activation)
    if activation_name != "silu":
        raise NotImplementedError(
            f"the stage1 epilogue is silu(gate)*up; layer.activation is "
            f"{activation!r}"
        )
    if getattr(layer, "apply_router_weight_on_input", False):
        # The kernel folds the routing weight into stage2's epilogue, which is
        # the end with fewer output elements and the only one wired up.
        raise NotImplementedError("apply_router_weight_on_input is not implemented by the RDNA3 MoE kernel")
    if getattr(layer, "w13_bias", None) is not None or getattr(layer, "w2_bias", None) is not None:
        raise NotImplementedError("the RDNA3 MoE kernel has no bias term")

# Which routing granularity to pad to. The tables in ``host.py`` were measured
# with a prefill wanting a tall tile and a decode a short one; the middle is
# flat. The crossover is in rows per expert rather than tokens, so it is
# computed rather than thresholded on the batch -- see ``tile_m_for``.
DECODE_TILE_M = 16
PREFILL_TILE_M = 64

# Below this many routed rows per expert a taller tile is all padding.
_ROWS_PER_EXPERT_FOR_TALL_TILE = 64


def tile_m_for(*, tokens: int, topk: int, experts: int) -> int:
    """The routing tile this shape should pad to.

    An expert averages ``tokens*topk/experts`` rows, and a tile taller than that
    is padding the kernel still has to multiply. With 512 experts that average
    stays under one tile until the batch is in the thousands, so this returns the
    decode tile for every decode-sized batch and only steps up once a tall tile
    would actually be filled.
    """
    rows_per_expert = tokens * topk / max(int(experts), 1)
    return PREFILL_TILE_M if rows_per_expert >= _ROWS_PER_EXPERT_FOR_TALL_TILE else DECODE_TILE_M


def fuse_shared_expert(
    w13: torch.Tensor,
    w2: torch.Tensor,
    shared_w13: torch.Tensor,
    shared_w2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack a shared expert onto the routed ones as expert ``E``.

    ``shared_w13`` is ``[2*I, H]`` gate-half-first and ``shared_w2`` is
    ``[H, I]`` -- one expert's worth, the same layout the routed stack uses per
    slab. The result is ``[E+1, ...]``, and a caller then routes every token to
    expert ``E`` on an extra slot whose weight is the shared gate.

    This is the same fusion vLLM's ``maybe_fuse_shared_experts`` does at load
    time; it exists here for callers driving the kernel directly. It copies, so
    do it once at load, not per step.
    """
    if shared_w13.shape != w13.shape[1:]:
        raise ValueError(f"shared_w13 must be {tuple(w13.shape[1:])}, got {tuple(shared_w13.shape)}")
    if shared_w2.shape != w2.shape[1:]:
        raise ValueError(f"shared_w2 must be {tuple(w2.shape[1:])}, got {tuple(shared_w2.shape)}")
    return (
        torch.cat([w13, shared_w13.unsqueeze(0).to(w13.dtype)], dim=0).contiguous(),
        torch.cat([w2, shared_w2.unsqueeze(0).to(w2.dtype)], dim=0).contiguous(),
    )


def add_shared_slot(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    shared_expert_id: int,
    shared_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append the always-on slot that a fused shared expert rides in.

    ``shared_weight`` is ``[tokens]`` or ``[tokens, 1]`` -- Qwen's
    ``sigmoid(shared_expert_gate(x))``. Passing ``None`` gives it weight 1, which
    is the ungated case.
    """
    tokens = topk_ids.shape[0]
    ids = torch.full((tokens, 1), int(shared_expert_id), dtype=topk_ids.dtype, device=topk_ids.device)
    if shared_weight is None:
        w = torch.ones(tokens, 1, dtype=torch.float32, device=topk_weights.device)
    else:
        w = shared_weight.reshape(tokens, 1).to(torch.float32)
    return torch.cat([topk_ids, ids], dim=1), torch.cat([topk_weights.to(torch.float32), w], dim=1)


def moe_layer_forward(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    tile_m: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """One rank's MoE output, before any all-reduce.

    ``x`` is ``[tokens, hidden]``, the weights are vLLM's ``w13_weight`` and
    ``w2_weight`` for this rank, and the routing is whatever the router produced
    -- including the shared expert's slot if it was fused in. The return is this
    rank's partial sum over the slots; combining it across TP ranks is the
    caller's, because under TP each rank holds a slice of every expert's
    intermediate and the sum is only complete after the reduce.
    """
    if x.dim() != 2:
        raise ValueError(f"x must be [tokens, hidden], got {tuple(x.shape)}")
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)
    if topk_weights.dtype != torch.float32:
        topk_weights = topk_weights.to(torch.float32)

    tokens, topk = topk_ids.shape
    if tile_m is None:
        tile_m = tile_m_for(tokens=tokens, topk=topk, experts=w13.shape[0])

    return moe_forward(
        x.contiguous(),
        w13,
        w2,
        topk_ids.contiguous(),
        topk_weights.contiguous(),
        tile_m=tile_m,
        out=out,
    )
