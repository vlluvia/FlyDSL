#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Correctness tests for the RDNA3 (gfx11*) MoE kernels.

The kernel under test is a grouped GEMM in three modes. All of them walk the
padded routing order and take each row through the expert that row was routed to,
so a reference is one matmul per row and the interesting cases are the ones where
the *routing*, not the arithmetic, can go wrong: a row gathered from the wrong
token, a tile reading the wrong expert's weight, a result scattered to the wrong
slot, or a padded row reading past the end of its input. Several tests here look
at rows and slots the caller does not care about for exactly that reason.

A sum over ``topk`` hides a scatter that permutes slots, so the two MoE stages
each get a per-slot check as well as an against-the-reference one.

The routing buffers have three builders, and they are checked against each other
in that order: the python loop in ``build_routing`` below, the host's vectorised
``routing.build_routing``, and the wave32 kernel in ``routing_kernel``. All three
emit the ids the CDNA sorting kernel emits (``(slot << 24) | token``, sentinel
``(topk << 24) | tokens``), which is why that kernel's consumers -- and these --
do not care which one ran. The per-stage tests use the loop.
"""

import os
import sys

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

from kernels.moe.rdna3_moe.forward import (  # noqa: E402
    moe_forward,
    moe_forward_from_logits,
    moe_gating,
    moe_reduce,
)
from kernels.moe.rdna3_moe.grouped_gemm import create_grouped_gemm_module  # noqa: E402
from kernels.moe.rdna3_moe import host  # noqa: E402
from kernels.moe.rdna3_moe.host import compile_grouped_gemm, compile_moe_gemm1, compile_moe_gemm2  # noqa: E402
from kernels.moe.rdna3_moe.routing import build_routing as build_routing_torch  # noqa: E402
from kernels.moe.rdna3_moe.routing_kernel import (  # noqa: E402
    MAX_EXPERTS,
    build_routing_device,
    compile_routing,
    max_blocks_for,
)
from tests.kernels.test_ref import torch_moe_gemm1, torch_moe_gemm2  # noqa: E402
from tests.test_common import verify_output  # noqa: E402

_DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}
_NO_WEIGHTS = None  # lazily allocated empty tensor for the stages that ignore it


def _no_weights():
    global _NO_WEIGHTS
    if _NO_WEIGHTS is None:
        _NO_WEIGHTS = torch.empty(0, dtype=torch.float32, device="cuda")
    return _NO_WEIGHTS


def build_routing(topk_ids, experts, tile_m, tokens):
    """Group rows by expert and pad each expert up to ``tile_m``.

    Returns ``(sorted_ids, expert_ids, num_blocks)``. ``sorted_ids[i]`` is the
    packed ``(slot << 24) | token`` of routing row ``i``; padded rows carry the
    sentinel, whose token field is ``tokens`` and so addresses one row past the
    input. ``expert_ids[b]`` names the expert of tile ``b``.
    """
    topk = topk_ids.shape[1]
    sentinel = (topk << 24) | tokens
    sorted_ids, expert_ids = [], []
    for e in range(experts):
        token, slot = torch.where(topk_ids == e)
        rows = ((slot.to(torch.int32) << 24) | token.to(torch.int32)).tolist()
        blocks = -(-len(rows) // tile_m)
        sorted_ids += rows + [sentinel] * (blocks * tile_m - len(rows))
        expert_ids += [e] * blocks
    dev = topk_ids.device
    return (
        torch.tensor(sorted_ids, dtype=torch.int32, device=dev),
        torch.tensor(expert_ids, dtype=torch.int32, device=dev),
        len(expert_ids),
    )


def _canonical(routing, tile_m):
    """``sorted_ids`` with each expert's slot range sorted into one order.

    Two builders that group the same rows under the same experts may still
    disagree on the order inside an expert -- the device kernel hands out slots
    with an atomic, the host builder with a stable argsort -- and nothing
    downstream can tell. This is the form in which they have to agree.
    """
    ids = routing.sorted_ids
    labels = routing.expert_ids.repeat_interleave(tile_m)
    out = ids.clone()
    for e in torch.unique(labels).tolist():
        mask = labels == e
        out[mask] = torch.sort(ids[mask]).values
    return out


def _route(tokens, experts, topk, seed=0):
    """A routing with distinct experts per token, and its weights."""
    torch.manual_seed(seed)
    score = torch.rand(tokens, experts, device="cuda")
    weights, ids = torch.topk(score, k=topk, dim=1)
    return ids.to(torch.int32), weights.contiguous()


def _reduce_topk(y):
    """Sum ``[tokens, topk, d] -> [tokens, d]`` with the existing MoE reducer.

    ``moe_reduce`` wraps the CDNA package's ``compile_moe_reduction``, which is
    wave-size agnostic (256 threads, no MFMA, no atomics) and so runs here
    unchanged. Stage2 folds the routing weight in, which is why an unweighted sum
    is the right reduction.
    """
    out = moe_reduce(y)
    torch.cuda.synchronize()
    return out


# ── The linear mode: the gather and the per-expert weight, on their own ───────


def _reference_linear(X, W, sorted_ids, expert_ids, tile_m, tokens):
    """Per-row matmul through the routed expert; padded rows stay zero."""
    rows = sorted_ids.numel()
    ref = torch.zeros(rows, W.shape[1], dtype=torch.float32, device=X.device)
    token = (sorted_ids & 0xFFFFFF).to(torch.long).tolist()
    eids = expert_ids.to(torch.long).tolist()
    for i in range(rows):
        if token[i] < tokens:
            ref[i] = X[token[i]].float() @ W[eids[i // tile_m]].float().T
    return ref


def _run_linear(*, tokens, k_dim, n_out, experts, topk, tile_m, in_dtype="bf16", out_dtype="bf16", seed=0, c_extra=0):
    """Compile, route, launch the linear mode. Returns ``(C_backing, ref, rows)``.

    ``c_extra`` appends rows to the C allocation so a caller can check the kernel
    did not store into them.
    """
    dev = "cuda"
    in_t, out_t = _DTYPES[in_dtype], _DTYPES[out_dtype]
    launch, block_m, _, _ = compile_grouped_gemm(
        k_dim=k_dim, n_out=n_out, experts=experts, stage="linear", tile_m=tile_m, in_dtype=in_dtype, out_dtype=out_dtype
    )
    assert block_m == tile_m

    torch.manual_seed(seed)
    X = (torch.randn(tokens, k_dim, dtype=in_t, device=dev) * 0.1).contiguous()
    W = (torch.randn(experts, n_out, k_dim, dtype=in_t, device=dev) * 0.1).contiguous()
    topk_ids, _ = _route(tokens, experts, topk, seed=seed)
    sorted_ids, expert_ids, num_blocks = build_routing(topk_ids, experts, tile_m, tokens)
    rows = num_blocks * tile_m
    assert sorted_ids.numel() == rows

    backing = torch.zeros(rows + c_extra, n_out, dtype=out_t, device=dev)
    if c_extra:
        backing[rows:] = 7.0
    C = backing[:rows]

    launch(C, X, W, sorted_ids, expert_ids, _no_weights(), tokens, num_blocks, torch.cuda.current_stream())
    torch.cuda.synchronize()
    return backing, _reference_linear(X, W, sorted_ids, expert_ids, tile_m, tokens), rows


@pytest.mark.parametrize(
    "tokens, k_dim, n_out, experts, topk",
    [
        pytest.param(8, 256, 64, 8, 1, id="t8-k256x64-e8k1"),
        pytest.param(32, 256, 128, 8, 2, id="t32-k256x128-e8k2"),
        # 128 tokens over 4 experts is ~64 routing rows each, so an expert spans
        # several tiles and the per-expert weight has to hold across them.
        pytest.param(128, 512, 128, 4, 2, id="t128-k512x128-e4k2"),
        pytest.param(64, 128, 256, 16, 4, id="t64-k128x256-e16k4"),
    ],
)
@pytest.mark.parametrize("tile_m", [16, 32])
def test_linear(tokens, k_dim, n_out, experts, topk, tile_m):
    """Every routing row takes its own token through its own expert's weight."""
    got, ref, rows = _run_linear(
        tokens=tokens, k_dim=k_dim, n_out=n_out, experts=experts, topk=topk, tile_m=tile_m
    )
    assert verify_output(got[:rows].float(), ref, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("tile_m", [16, 32, 64, 128])
def test_linear_tiles(tile_m):
    """Each tile in the host table builds and computes the same answer."""
    got, ref, rows = _run_linear(tokens=64, k_dim=256, n_out=128, experts=4, topk=2, tile_m=tile_m)
    assert verify_output(got[:rows].float(), ref, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("in_dtype, out_dtype", [("bf16", "bf16"), ("f16", "f16"), ("bf16", "f32"), ("f16", "bf16")])
def test_linear_dtypes(in_dtype, out_dtype):
    got, ref, rows = _run_linear(
        tokens=32, k_dim=256, n_out=128, experts=8, topk=2, tile_m=16, in_dtype=in_dtype, out_dtype=out_dtype
    )
    assert verify_output(got[:rows].float(), ref, atol=0.05, rtol=0.05)


def test_padded_rows_read_as_zero():
    """A padded row's sentinel must read zero, not whatever follows A.

    Routing pads each expert up to the tile, so with 8 tokens and topk=1 over 8
    experts most tiles are almost all padding. The kernel does not branch on the
    sentinel; it bounds the A descriptor and asks the hardware to enforce it,
    which on RDNA is opt-in -- with the default descriptor mode these rows come
    back as garbage from past the end of A.
    """
    tokens, tile_m = 8, 16
    got, _, rows = _run_linear(tokens=tokens, k_dim=256, n_out=64, experts=8, topk=1, tile_m=tile_m)
    topk_ids, _ = _route(tokens, 8, 1)
    sorted_ids, _, _ = build_routing(topk_ids, 8, tile_m, tokens)

    padded = (sorted_ids & 0xFFFFFF) >= tokens
    assert int(padded.sum()) > 0, "this shape was supposed to be mostly padding"
    assert torch.equal(got[:rows][padded], torch.zeros_like(got[:rows][padded]))


def test_does_not_store_past_c():
    """The C allocation ends at the last routing row; nothing may be written after it."""
    got, ref, rows = _run_linear(tokens=32, k_dim=256, n_out=128, experts=8, topk=2, tile_m=16, c_extra=32)
    assert verify_output(got[:rows].float(), ref, atol=0.05, rtol=0.05)
    canary = torch.full_like(got[rows:], 7.0)
    assert torch.equal(got[rows:], canary), "the kernel stored past the last routing row"


def test_expert_id_selects_the_weight():
    """Rotating expert_ids must change the answer.

    A kernel that ignored ``expert_ids`` and always read expert 0 would pass
    every test above on a single-expert shape and most of them on any shape where
    the reference is built from the same ids. This pins the weight slab to the id
    rather than to the tile index.
    """
    tokens, k_dim, n_out, experts, topk, tile_m = 32, 256, 128, 8, 2, 16
    dev = "cuda"
    launch, _, _, _ = compile_grouped_gemm(
        k_dim=k_dim, n_out=n_out, experts=experts, stage="linear", tile_m=tile_m
    )

    torch.manual_seed(0)
    X = (torch.randn(tokens, k_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    W = (torch.randn(experts, n_out, k_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    topk_ids, _ = _route(tokens, experts, topk)
    sorted_ids, expert_ids, num_blocks = build_routing(topk_ids, experts, tile_m, tokens)
    rows = num_blocks * tile_m

    def run(eids):
        C = torch.zeros(rows, n_out, dtype=torch.bfloat16, device=dev)
        launch(C, X, W, sorted_ids, eids, _no_weights(), tokens, num_blocks, torch.cuda.current_stream())
        torch.cuda.synchronize()
        return C.float()

    rotated = (expert_ids + 1) % experts
    assert verify_output(
        run(expert_ids), _reference_linear(X, W, sorted_ids, expert_ids, tile_m, tokens), atol=0.05, rtol=0.05
    )
    assert verify_output(
        run(rotated), _reference_linear(X, W, sorted_ids, rotated, tile_m, tokens), atol=0.05, rtol=0.05
    )


def test_token_id_selects_the_row():
    """Reversing the gathered token of every row must permute the output the same way.

    The gather is the one place a gemm can be right on average and wrong per row,
    so this compares against the same kernel run on a permuted routing rather
    than against a matmul.
    """
    tokens, k_dim, n_out, tile_m = 16, 256, 64, 16
    dev = "cuda"
    launch, _, _, _ = compile_grouped_gemm(k_dim=k_dim, n_out=n_out, experts=1, stage="linear", tile_m=tile_m)

    torch.manual_seed(0)
    X = (torch.randn(tokens, k_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    W = (torch.randn(1, n_out, k_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    expert_ids = torch.zeros(1, dtype=torch.int32, device=dev)

    def run(ids):
        C = torch.zeros(tile_m, n_out, dtype=torch.bfloat16, device=dev)
        launch(C, X, W, ids, expert_ids, _no_weights(), tokens, 1, torch.cuda.current_stream())
        torch.cuda.synchronize()
        return C.float()

    ident = torch.arange(tokens, dtype=torch.int32, device=dev)
    straight = run(ident)
    reversed_ = run(torch.flip(ident, dims=[0]).contiguous())
    assert torch.equal(reversed_, torch.flip(straight, dims=[0]))


def test_rejects_unknown_tile_m():
    with pytest.raises(ValueError, match="tile_m must be one of"):
        compile_grouped_gemm(k_dim=256, n_out=128, experts=4, tile_m=48)


def test_rejects_doweight_outside_stage2():
    with pytest.raises(ValueError, match="doweight is only implemented"):
        compile_grouped_gemm(k_dim=256, n_out=128, experts=4, stage="gate_up", topk=2, doweight=True)


def test_rejects_a_tile_of_one_accumulator():
    # The K loop carries the accumulators as loop state, and one value comes back
    # unwrapped -- so the builder rules the tile out rather than special-casing it.
    with pytest.raises(ValueError, match="one accumulator"):
        create_grouped_gemm_module(
            k_dim=256, n_out=128, experts=4, stage="linear", reg_m=1, reg_n=1, reg_k=2, waves_m=1, waves_n=1
        )


@pytest.mark.parametrize("stage,tile_m", [("gate_up", 64), ("down", 64), ("down", 32), ("linear", 128)])
def test_the_tile_table_narrows_itself_for_a_small_shape(stage, tile_m):
    """A shape too small for the tuned tile gets a narrower one, not an error.

    The table is measured at LLM-sized dimensions, where a wide BLOCK_N has the
    blocks to fill. n_out=32 divides none of the preferred tiles.
    """
    launch, block_m, block_n, block_k = compile_grouped_gemm(
        k_dim=64, n_out=32, experts=2, stage=stage, topk=2, doweight=(stage == "down"), tile_m=tile_m
    )
    assert block_m == tile_m
    assert 32 % block_n == 0
    assert 64 % block_k == 0
    assert launch.lds_bytes <= 64 * 1024


def test_reports_the_shape_no_tile_fits():
    with pytest.raises(ValueError, match="no tile for stage"):
        compile_grouped_gemm(k_dim=256, n_out=24, experts=2, stage="linear", tile_m=16)


# ── Stage 1: gate/up + silu + scatter ────────────────────────────────────────


def _run_stage1(
    *, tokens, model_dim, inter_dim, experts, topk, tile_m, in_dtype="bf16", out_dtype="bf16", seed=0, out_extra=0
):
    """Compile, route, launch stage1. Returns ``(out_backing, ref, rows, parts)``.

    ``out_extra`` appends rows to the output allocation so a caller can check the
    padded routing rows did not store into them. ``parts`` carries the inputs a
    follow-on stage2 needs.
    """
    dev = "cuda"
    in_t, out_t = _DTYPES[in_dtype], _DTYPES[out_dtype]
    launch, block_m, _, _ = compile_moe_gemm1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
    )
    assert block_m == tile_m

    torch.manual_seed(seed)
    X = (torch.randn(tokens, model_dim, dtype=in_t, device=dev) * 0.1).contiguous()
    # The gate half first, then the up half: w1[e, :inter] and w1[e, inter:].
    W1 = (torch.randn(experts, 2 * inter_dim, model_dim, dtype=in_t, device=dev) * 0.1).contiguous()
    topk_ids, topk_weights = _route(tokens, experts, topk, seed=seed)

    sorted_ids, expert_ids, num_blocks = build_routing(topk_ids, experts, tile_m, tokens)
    backing = torch.zeros(tokens * topk + out_extra, inter_dim, dtype=out_t, device=dev)
    if out_extra:
        backing[tokens * topk :] = 7.0
    out = backing[: tokens * topk].view(tokens, topk, inter_dim)

    launch(out, X, W1, sorted_ids, expert_ids, _no_weights(), tokens, num_blocks, torch.cuda.current_stream())
    torch.cuda.synchronize()

    ref = torch_moe_gemm1(X, W1, None, None, topk_ids, topk_weights, inter_dim, False)
    parts = dict(
        X=X, topk_ids=topk_ids, topk_weights=topk_weights, sorted_ids=sorted_ids, expert_ids=expert_ids,
        num_blocks=num_blocks, a2=out,
    )
    return backing, ref, tokens * topk, parts


@pytest.mark.parametrize(
    "tokens, model_dim, inter_dim, experts, topk",
    [
        pytest.param(8, 256, 64, 8, 1, id="t8-d256-i64-e8k1"),
        pytest.param(32, 256, 128, 8, 2, id="t32-d256-i128-e8k2"),
        pytest.param(128, 512, 128, 4, 2, id="t128-d512-i128-e4k2"),
        pytest.param(64, 128, 256, 16, 4, id="t64-d128-i256-e16k4"),
    ],
)
@pytest.mark.parametrize("tile_m", [16, 32])
def test_stage1_matches_torch(tokens, model_dim, inter_dim, experts, topk, tile_m):
    """silu(gate)*up, scattered back to [tokens, topk, inter], against the reference."""
    got, ref, rows, _ = _run_stage1(
        tokens=tokens, model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk, tile_m=tile_m
    )
    assert verify_output(got[:rows].float().reshape(ref.shape), ref, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("tile_m", [16, 32, 64, 128])
def test_stage1_tiles(tile_m):
    got, ref, rows, _ = _run_stage1(tokens=128, model_dim=256, inter_dim=128, experts=4, topk=2, tile_m=tile_m)
    assert verify_output(got[:rows].float().reshape(ref.shape), ref, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("in_dtype, out_dtype", [("bf16", "bf16"), ("f16", "f16"), ("bf16", "f32")])
def test_stage1_dtypes(in_dtype, out_dtype):
    got, ref, rows, _ = _run_stage1(
        tokens=32,
        model_dim=256,
        inter_dim=128,
        experts=8,
        topk=2,
        tile_m=16,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
    )
    assert verify_output(got[:rows].float().reshape(ref.shape), ref, atol=0.05, rtol=0.05)


def test_stage1_drops_padded_row_stores():
    """A padded routing row must store nothing.

    Its sentinel decodes to output row ``tokens*topk + topk``, past the end of a
    ``[tokens, topk, inter]`` buffer, and the only thing stopping the store is
    the bounded descriptor. With 8 tokens at topk=1 over 8 experts every tile is
    mostly padding, so a descriptor that did not enforce its bound would write
    well past the allocation.
    """
    got, ref, rows, _ = _run_stage1(
        tokens=8, model_dim=256, inter_dim=64, experts=8, topk=1, tile_m=16, out_extra=64
    )
    assert verify_output(got[:rows].float().reshape(ref.shape), ref, atol=0.05, rtol=0.05)
    canary = torch.full_like(got[rows:], 7.0)
    assert torch.equal(got[rows:], canary), "a padded routing row stored past the output"


def test_stage1_scatter_lands_on_the_routed_slot():
    """The (token, slot) a row came from is the (token, slot) it lands on.

    Every routed row here gets the same activation, so the gemm cannot
    distinguish them and only the scatter decides where each result goes. Rows the
    routing never produced have to stay untouched, which is what catches a scatter
    that drops the slot from the address or transposes it with the token.
    """
    tokens, model_dim, inter_dim, experts, topk, tile_m = 4, 256, 64, 4, 2, 16
    dev = "cuda"
    launch, _, _, _ = compile_moe_gemm1(
        model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk, tile_m=tile_m
    )

    torch.manual_seed(0)
    X = torch.ones(tokens, model_dim, dtype=torch.bfloat16, device=dev) * 0.05
    W1 = (torch.randn(experts, 2 * inter_dim, model_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()

    # Route token t to expert t%experts at slot 0 only -- slot 1 is never routed.
    sentinel = (topk << 24) | tokens
    sorted_ids, expert_ids = [], []
    for e in range(experts):
        rows = [(0 << 24) | t for t in range(tokens) if t % experts == e]
        sorted_ids += rows + [sentinel] * (tile_m - len(rows))
        expert_ids.append(e)
    sorted_ids = torch.tensor(sorted_ids, dtype=torch.int32, device=dev)
    expert_ids = torch.tensor(expert_ids, dtype=torch.int32, device=dev)

    out = torch.full((tokens, topk, inter_dim), -1.0, dtype=torch.bfloat16, device=dev)
    launch(out, X, W1, sorted_ids, expert_ids, _no_weights(), tokens, len(expert_ids), torch.cuda.current_stream())
    torch.cuda.synchronize()

    assert torch.equal(out[:, 1, :], torch.full_like(out[:, 1, :], -1.0)), "wrote an unrouted slot"
    for t in range(tokens):
        gate = X[t].float() @ W1[t % experts, :inter_dim].float().T
        up = X[t].float() @ W1[t % experts, inter_dim:].float().T
        assert verify_output(
            out[t, 0, :].float(), torch.nn.functional.silu(gate) * up, atol=0.05, rtol=0.05
        ), f"token {t} slot 0"


# ── Stage 2: down-projection + routing weight + scatter, then reduce ─────────


def _reference_stage2_slots(a2, w2, topk_ids, topk_weights, doweight):
    """Per-slot stage2, before the topk sum. A sum would hide a slot permutation."""
    tokens, topk, _ = a2.shape
    out = torch.zeros(tokens, topk, w2.shape[1], dtype=torch.float32, device=a2.device)
    for t in range(tokens):
        for s in range(topk):
            e = int(topk_ids[t, s])
            y = a2[t, s].float() @ w2[e].float().T
            out[t, s] = y * float(topk_weights[t, s]) if doweight else y
    return out


def _run_stage2(
    *, tokens, model_dim, inter_dim, experts, topk, tile_m, doweight=True, out_dtype="bf16", seed=0, out_extra=0
):
    """Compile, route, launch stage2 on a random A2. Returns ``(backing, parts)``."""
    dev = "cuda"
    out_t = _DTYPES[out_dtype]
    launch, block_m, _, _ = compile_moe_gemm2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        doweight=doweight,
        tile_m=tile_m,
        out_dtype=out_dtype,
    )
    assert block_m == tile_m

    torch.manual_seed(seed)
    A2 = (torch.randn(tokens, topk, inter_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    W2 = (torch.randn(experts, model_dim, inter_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    topk_ids, topk_weights = _route(tokens, experts, topk, seed=seed)
    sorted_ids, expert_ids, num_blocks = build_routing(topk_ids, experts, tile_m, tokens)

    backing = torch.zeros(tokens * topk + out_extra, model_dim, dtype=out_t, device=dev)
    if out_extra:
        backing[tokens * topk :] = 7.0
    out = backing[: tokens * topk].view(tokens, topk, model_dim)

    launch(out, A2, W2, sorted_ids, expert_ids, topk_weights, tokens, num_blocks, torch.cuda.current_stream())
    torch.cuda.synchronize()
    return backing, dict(A2=A2, W2=W2, topk_ids=topk_ids, topk_weights=topk_weights, out=out)


@pytest.mark.parametrize(
    "tokens, model_dim, inter_dim, experts, topk",
    [
        pytest.param(8, 128, 256, 8, 1, id="t8-d128-i256-e8k1"),
        pytest.param(32, 256, 128, 8, 2, id="t32-d256-i128-e8k2"),
        pytest.param(128, 128, 512, 4, 2, id="t128-d128-i512-e4k2"),
        pytest.param(64, 256, 128, 16, 4, id="t64-d256-i128-e16k4"),
    ],
)
@pytest.mark.parametrize("tile_m", [16, 32])
def test_stage2_per_slot(tokens, model_dim, inter_dim, experts, topk, tile_m):
    """Each routed slot's down-projection lands on that slot, weighted.

    Stage2 gathers A row ``token*topk + slot`` rather than ``token``, so this is
    also the only test of that addressing.
    """
    backing, p = _run_stage2(
        tokens=tokens, model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk, tile_m=tile_m
    )
    ref = _reference_stage2_slots(p["A2"], p["W2"], p["topk_ids"], p["topk_weights"], doweight=True)
    assert verify_output(p["out"].float(), ref, atol=0.05, rtol=0.05)
    canary = torch.full_like(backing[tokens * topk :], 7.0)
    assert torch.equal(backing[tokens * topk :], canary)


@pytest.mark.parametrize("tile_m", [16, 32, 64])
def test_stage2_tiles(tile_m):
    _, p = _run_stage2(tokens=128, model_dim=256, inter_dim=256, experts=4, topk=2, tile_m=tile_m)
    ref = _reference_stage2_slots(p["A2"], p["W2"], p["topk_ids"], p["topk_weights"], doweight=True)
    assert verify_output(p["out"].float(), ref, atol=0.05, rtol=0.05)


def test_stage2_doweight_off_leaves_the_weight_out():
    """Without doweight the result is the unweighted projection.

    Pinning both directions is what keeps the weight from being silently applied
    twice once the reduce is unweighted.
    """
    _, p = _run_stage2(tokens=32, model_dim=256, inter_dim=128, experts=8, topk=2, tile_m=16, doweight=False)
    ref = _reference_stage2_slots(p["A2"], p["W2"], p["topk_ids"], p["topk_weights"], doweight=False)
    assert verify_output(p["out"].float(), ref, atol=0.05, rtol=0.05)


def test_stage2_drops_padded_row_stores():
    backing, p = _run_stage2(
        tokens=8, model_dim=128, inter_dim=256, experts=8, topk=1, tile_m=16, out_extra=64
    )
    ref = _reference_stage2_slots(p["A2"], p["W2"], p["topk_ids"], p["topk_weights"], doweight=True)
    assert verify_output(p["out"].float(), ref, atol=0.05, rtol=0.05)
    canary = torch.full_like(backing[8:], 7.0)
    assert torch.equal(backing[8:], canary), "a padded routing row stored past the output"


def test_stage2_plus_reduce_matches_torch():
    """stage2 then reduce is the canonical ``torch_moe_gemm2``.

    The reference folds the routing weight in and then sums the slots away, which
    is exactly the split here: ``doweight`` in the kernel, unweighted sum in
    ``moe_reduce``.
    """
    tokens, model_dim, inter_dim, experts, topk = 64, 256, 256, 8, 2
    _, p = _run_stage2(
        tokens=tokens, model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk, tile_m=16
    )
    got = _reduce_topk(p["out"])
    ref = torch_moe_gemm2(
        p["A2"], p["W2"], None, None, p["topk_ids"], p["topk_weights"], model_dim, doweight_stage2=True
    )
    assert verify_output(got.float(), ref, atol=0.05, rtol=0.05)


# ── The whole thing ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "tokens, model_dim, inter_dim, experts, topk, tile_m",
    [
        pytest.param(32, 256, 128, 8, 2, 16, id="decode"),
        pytest.param(256, 512, 256, 8, 2, 32, id="prefill"),
    ],
)
def test_end_to_end(tokens, model_dim, inter_dim, experts, topk, tile_m):
    """X -> stage1 -> stage2 -> reduce, against the two torch references chained.

    One routing drives all three launches, which is the property that matters for
    a real MoE layer: stage1 scatters to (token, slot), stage2 gathers the same
    (token, slot) back, and the reduce sums the slots away. A stage that agreed
    with its own reference but disagreed with its neighbour's row order would only
    show up here.
    """
    dev = "cuda"
    torch.manual_seed(0)
    X = (torch.randn(tokens, model_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    W1 = (torch.randn(experts, 2 * inter_dim, model_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    W2 = (torch.randn(experts, model_dim, inter_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    topk_ids, topk_weights = _route(tokens, experts, topk)
    sorted_ids, expert_ids, num_blocks = build_routing(topk_ids, experts, tile_m, tokens)
    stream = torch.cuda.current_stream()

    g1, _, _, _ = compile_moe_gemm1(
        model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk, tile_m=tile_m
    )
    g2, _, _, _ = compile_moe_gemm2(
        model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk, doweight=True, tile_m=tile_m
    )

    a2 = torch.zeros(tokens, topk, inter_dim, dtype=torch.bfloat16, device=dev)
    g1(a2, X, W1, sorted_ids, expert_ids, _no_weights(), tokens, num_blocks, stream)
    y = torch.zeros(tokens, topk, model_dim, dtype=torch.bfloat16, device=dev)
    g2(y, a2, W2, sorted_ids, expert_ids, topk_weights, tokens, num_blocks, stream)
    got = _reduce_topk(y)

    a2_ref = torch_moe_gemm1(X, W1, None, None, topk_ids, topk_weights, inter_dim, False)
    ref = torch_moe_gemm2(
        a2_ref.to(torch.bfloat16), W2, None, None, topk_ids, topk_weights, model_dim, doweight_stage2=True
    )
    assert verify_output(got.float(), ref, atol=0.05, rtol=0.05)


# ── The routing buffers, built with torch ────────────────────────────────────


@pytest.mark.parametrize(
    "tokens, experts, topk",
    [
        pytest.param(8, 8, 1, id="t8-e8k1"),
        pytest.param(32, 8, 2, id="t32-e8k2"),
        pytest.param(128, 4, 2, id="t128-e4k2"),
        pytest.param(64, 16, 4, id="t64-e16k4"),
        pytest.param(7, 8, 3, id="t7-e8k3"),
    ],
)
@pytest.mark.parametrize("tile_m", [16, 32, 64])
def test_routing_matches_the_loop_reference(tokens, experts, topk, tile_m):
    """The vectorised builder agrees with the obvious one, buffer for buffer.

    ``build_routing`` in this file is the obvious one: a python loop that asks
    ``torch.where`` for each expert's rows in turn. The shipped builder does the
    same thing in a sort and a couple of cumulative sums, which is the version
    that can run per decode step, so the two have to agree exactly -- not just up
    to a permutation, because the padding count depends on the grouping.
    """
    topk_ids, _ = _route(tokens, experts, topk)
    ref_ids, ref_experts, ref_blocks = build_routing(topk_ids, experts, tile_m, tokens)
    got = build_routing_torch(topk_ids, experts=experts, tile_m=tile_m)

    assert got.num_blocks == ref_blocks
    assert torch.equal(got.sorted_ids, ref_ids)
    assert torch.equal(got.expert_ids, ref_experts)


def test_routing_gives_an_unrouted_expert_no_tile():
    """An expert nobody picked is skipped, not padded out to an empty tile.

    A tile of nothing but sentinels gathers zeros and stores nothing, but it
    still streams its expert's weight slab first, and that is the whole cost of a
    sparse decode: at 512 experts and one token, a floor of one tile per expert
    is 500 slabs read to compute ten.
    """
    tokens, experts, topk, tile_m = 16, 8, 1, 16
    topk_ids = torch.zeros(tokens, topk, dtype=torch.int32, device="cuda")  # everyone picks expert 0
    r = build_routing_torch(topk_ids, experts=experts, tile_m=tile_m)

    assert r.num_blocks == 1, "only expert 0 has rows, so only expert 0 gets a tile"
    assert torch.equal(r.expert_ids, torch.zeros(1, dtype=torch.int32, device="cuda"))
    assert torch.equal(r.sorted_ids, torch.arange(tokens, dtype=torch.int32, device="cuda"))


def test_routing_rejects_an_expert_id_this_rank_does_not_have():
    """A row routed to a foreign expert has to fail loudly.

    ``expert_ids`` indexes this rank's experts, so an id past the end has no
    weight slab to read. Grouping would just leave the row out, and a dropped row
    is a token that leaves the layer scaled by less than one -- small enough to
    look like precision.
    """
    topk_ids = torch.tensor([[0, 1], [2, 5]], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="outside this rank's 4 experts"):
        build_routing_torch(topk_ids, experts=4, tile_m=16)


@pytest.mark.parametrize("tile_m", [16, 32])
def test_routing_places_every_row_once_under_one_expert(tile_m):
    """Each (token, slot) appears exactly once, in a tile that holds its expert.

    This is the invariant both GEMMs lean on: a tile spanning two experts would
    need two weight slabs, and a row appearing twice or not at all would show up
    as a token that was scaled twice or dropped.
    """
    tokens, experts, topk = 64, 8, 3
    topk_ids, _ = _route(tokens, experts, topk)
    r = build_routing_torch(topk_ids, experts=experts, tile_m=tile_m)
    assert r.sorted_ids.numel() == r.rows

    token = (r.sorted_ids & 0xFFFFFF).to(torch.long)
    slot = (r.sorted_ids >> 24).to(torch.long)
    real = token < tokens

    flat = (token[real] * topk + slot[real]).cpu()
    assert sorted(flat.tolist()) == list(range(tokens * topk)), "a routed row was dropped or duplicated"

    tile_expert = r.expert_ids.repeat_interleave(tile_m).to(torch.long)
    assert torch.equal(topk_ids[token[real], slot[real]].to(torch.long), tile_expert[real])


# ── The routing buffers, built by the wave32 kernel ──────────────────────────


@pytest.mark.parametrize(
    "tokens, experts, topk",
    [
        pytest.param(1, 2, 2, id="t1-e2k2"),  # fewer rows than threads
        pytest.param(8, 8, 1, id="t8-e8k1"),
        pytest.param(32, 8, 2, id="t32-e8k2"),
        pytest.param(7, 8, 3, id="t7-e8k3"),  # rows not a multiple of the block
        pytest.param(64, 16, 4, id="t64-e16k4"),
        pytest.param(256, 32, 2, id="t256-e32k2"),
        pytest.param(64, 256, 2, id="t64-e256k2"),  # the widest single-chunk scan
        pytest.param(600, 8, 4, id="t600-e8k4"),  # more rows than one step
    ],
)
@pytest.mark.parametrize("tile_m", [16, 32, 64])
def test_device_routing_matches_the_host_builder(tokens, experts, topk, tile_m):
    """The kernel's buffers are the host builder's, up to order within an expert.

    Not element for element: the kernel claims a slot with an LDS atomic, so
    whichever thread gets there first takes it, and the host's stable argsort
    puts each expert's rows in row order. The two agree on everything that has a
    meaning -- the tile count, which expert owns each tile, and which rows land
    in each expert's slot range -- and disagree only on the order inside that
    range, which no output value depends on because every row is its own dot
    product and carries the ``(token, slot)`` it scatters back to.

    ``_canonical`` sorts each expert's range to compare them. The sentinel is
    ``(topk << 24) | tokens``, larger than any real packed id, so sorting leaves
    the padding at the end of the range where it belongs.

    ``exact=True`` because that is the mode with the same contract as the host
    builder: buffers sliced to the tiles that exist. The default mode returns the
    bound instead and leaves the count on the device, which is a different
    promise and has its own test below.
    """
    topk_ids, _ = _route(tokens, experts, topk)
    want = build_routing_torch(topk_ids, experts=experts, tile_m=tile_m)
    got = build_routing_device(topk_ids, experts=experts, tile_m=tile_m, exact=True, reuse=False)
    torch.cuda.synchronize()

    assert got.num_blocks == want.num_blocks
    assert torch.equal(got.expert_ids, want.expert_ids)
    assert torch.equal(_canonical(got, tile_m), _canonical(want, tile_m))


def test_device_routing_defines_the_tail_it_did_not_need():
    """Past the last tile, the buffer is sentinel rows under expert 0.

    The host sizes the allocation from ``max_blocks_for``, which is an upper
    bound, so the tail between the real tile count and that bound belongs to
    nobody -- and is exactly what a grid of the bound launches over. Expert 0 is
    a real weight slab to read, and a tile of sentinel rows gathers zeros and
    stores nothing, so those tiles are correct even when the guard is off.
    """
    tokens, experts, topk, tile_m = 32, 8, 2, 16
    topk_ids, _ = _route(tokens, experts, topk)
    launch = compile_routing(experts=experts, topk=topk, tile_m=tile_m)
    max_blocks = max_blocks_for(tokens=tokens, topk=topk, experts=experts, tile_m=tile_m)

    sorted_ids = torch.full((max_blocks * tile_m,), -7, dtype=torch.int32, device="cuda")
    expert_ids = torch.full((max_blocks,), -7, dtype=torch.int32, device="cuda")
    aux = torch.zeros(1, dtype=torch.int32, device="cuda")
    launch(topk_ids, sorted_ids, expert_ids, aux, tokens, max_blocks, torch.cuda.current_stream())
    torch.cuda.synchronize()

    num_blocks = int(aux[0])
    assert 0 < num_blocks < max_blocks, "the bound has to be loose here for the test to mean anything"
    sentinel = (topk << 24) | tokens
    assert torch.all(expert_ids[num_blocks:] == 0)
    assert torch.all(sorted_ids[num_blocks * tile_m :] == sentinel)


def test_device_routing_bound_holds_over_random_routings():
    """``max_blocks_for`` never comes in under the tile count it is sizing.

    It is the allocation size and, later, the grid, and the host computes it
    without looking at the routing at all -- from the row count alone. One
    routing that beat it would be a buffer overrun.

    ``exact=True`` so that ``num_blocks`` is the count being bounded. The
    default mode reports the bound itself, against which this would hold by
    construction and check nothing.
    """
    for seed in range(32):
        tokens, experts, topk, tile_m = 33, 8, 2, 32
        topk_ids, _ = _route(tokens, experts, topk, seed=seed)
        r = build_routing_device(topk_ids, experts=experts, tile_m=tile_m, exact=True, reuse=False)
        bound = max_blocks_for(tokens=tokens, topk=topk, experts=experts, tile_m=tile_m)
        assert r.num_blocks <= bound, f"seed {seed}: {r.num_blocks} tiles against a bound of {bound}"


def test_device_routing_handles_an_expert_nobody_picked():
    """The whole batch on one expert produces one expert's tiles, not eight.

    This is also the case where a thread's slice is empty at both ends: expert 0
    owns every row and the other seven own none, so seven of the eight threads
    have to fall through the scatter and the naming without writing anything.
    """
    tokens, experts, topk, tile_m = 16, 8, 1, 16
    topk_ids = torch.zeros(tokens, topk, dtype=torch.int32, device="cuda")
    # exact=True: the claim is about the tiles the routing produced, not about
    # the bound the host sized the buffers with.
    r = build_routing_device(topk_ids, experts=experts, tile_m=tile_m, exact=True, reuse=False)
    torch.cuda.synchronize()

    assert r.num_blocks == 1
    assert torch.equal(r.expert_ids, torch.zeros(1, dtype=torch.int32, device="cuda"))
    assert torch.equal(r.sorted_ids, torch.arange(tokens, dtype=torch.int32, device="cuda"))


# ── Launching over a bound instead of the tile count ─────────────────────────


@pytest.mark.parametrize("tile_m", [16, 64])
def test_bounded_blocks_gemm_matches_the_exact_grid(tile_m):
    """A grid of the bound, guarded by the device count, is the exact grid's answer.

    This is the trade o2 makes: the host stops reading the tile count back (a 29
    us round trip against a 10 us kernel) and instead launches over
    ``max_blocks_for``, which it can compute from the row count alone. The tiles
    past the real count read the padding the routing kernel left them, and the
    guard is what keeps them from doing the arithmetic.
    """
    tokens, model_dim, inter_dim, experts, topk = 64, 256, 128, 8, 2
    dev = "cuda"
    torch.manual_seed(0)
    a = (torch.randn(tokens, model_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    w = (torch.randn(experts, 2 * inter_dim, model_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    topk_ids, _ = _route(tokens, experts, topk)

    exact = build_routing_device(topk_ids, experts=experts, tile_m=tile_m, exact=True, reuse=False)
    bound = build_routing_device(topk_ids, experts=experts, tile_m=tile_m, reuse=False)
    torch.cuda.synchronize()
    assert bound.num_blocks > exact.num_blocks, "the bound has to be loose here for the test to mean anything"

    out = []
    for r in (exact, bound):
        gemm, _, _, _ = compile_moe_gemm1(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            bounded_blocks=not r.exact,
        )
        c = torch.zeros(tokens, topk, inter_dim, dtype=torch.bfloat16, device=dev)
        gemm(
            c,
            a,
            w,
            r.sorted_ids,
            r.expert_ids,
            _no_weights(),
            tokens,
            r.num_blocks,
            torch.cuda.current_stream(),
            r.num_blocks_device if not r.exact else None,
        )
        torch.cuda.synchronize()
        out.append(c)
    assert torch.equal(out[0], out[1])


def test_bounded_blocks_gemm_needs_the_device_count():
    """The build that reads the count refuses to launch without it.

    Silently launching over the bound instead would compute the padded tiles --
    correct, but the wasted tiles are the whole reason the guard exists.
    """
    gemm, _, _, _ = compile_moe_gemm1(
        model_dim=256, inter_dim=128, experts=8, topk=2, tile_m=16, bounded_blocks=True
    )
    dev = "cuda"
    empty = torch.zeros(1, dtype=torch.int32, device=dev)
    with pytest.raises(ValueError, match="num_blocks_dev is required"):
        gemm(empty, empty, empty, empty, empty, empty, 8, 2, torch.cuda.current_stream())


def test_device_routing_reports_a_bound_and_leaves_the_count_on_the_device():
    tokens, experts, topk, tile_m = 32, 8, 2, 16
    topk_ids, _ = _route(tokens, experts, topk)
    bound = build_routing_device(topk_ids, experts=experts, tile_m=tile_m)
    exact = build_routing_device(topk_ids, experts=experts, tile_m=tile_m, exact=True)
    torch.cuda.synchronize()

    assert not bound.exact and exact.exact
    assert bound.num_blocks == max_blocks_for(tokens=tokens, topk=topk, experts=experts, tile_m=tile_m)
    assert int(bound.num_blocks_device[0]) == exact.num_blocks
    # The bound's buffers are the whole allocation; the exact one's are sliced to
    # the tiles that exist, and agree with the host builder there.
    assert bound.sorted_ids.numel() == bound.num_blocks * tile_m
    want = build_routing_torch(topk_ids, experts=experts, tile_m=tile_m)
    assert torch.equal(exact.sorted_ids, want.sorted_ids)
    assert torch.equal(bound.sorted_ids[: want.rows], want.sorted_ids)


def test_device_routing_reuses_one_workspace_per_shape():
    """Two builds of a shape share buffers, unless the caller asks not to.

    Three allocations measured 10-15 us, which at decode is the same order as the
    kernel filling them. Reuse is safe against the GEMMs reading them because the
    next routing launch queues behind them on the same stream; a caller who needs
    a routing to outlive the next step passes ``reuse=False``.
    """
    topk_ids, _ = _route(32, 8, 2)
    first = build_routing_device(topk_ids, experts=8, tile_m=16)
    second = build_routing_device(topk_ids, experts=8, tile_m=16)
    private = build_routing_device(topk_ids, experts=8, tile_m=16, reuse=False)
    torch.cuda.synchronize()

    assert first.sorted_ids.data_ptr() == second.sorted_ids.data_ptr()
    assert private.sorted_ids.data_ptr() != first.sorted_ids.data_ptr()
    assert torch.equal(private.sorted_ids, first.sorted_ids)


# ── The layer: gating, routing, both GEMMs, the reduce ───────────────────────


@pytest.mark.parametrize(
    "tokens, model_dim, inter_dim, experts, topk, tile_m",
    [
        pytest.param(32, 256, 128, 8, 2, 16, id="decode"),
        pytest.param(256, 512, 256, 8, 2, 32, id="prefill"),
        pytest.param(7, 256, 128, 4, 4, 16, id="ragged"),
    ],
)
def test_moe_forward_matches_torch(tokens, model_dim, inter_dim, experts, topk, tile_m):
    """``moe_forward`` is the four launches, and it agrees with the references.

    ``test_end_to_end`` chains the same four by hand. This is the same check
    through the entry point a caller uses, so it also covers the routing builder
    and the allocations the entry point makes for the intermediates.
    """
    dev = "cuda"
    torch.manual_seed(0)
    X = (torch.randn(tokens, model_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    W1 = (torch.randn(experts, 2 * inter_dim, model_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    W2 = (torch.randn(experts, model_dim, inter_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    topk_ids, topk_weights = _route(tokens, experts, topk)

    got = moe_forward(X, W1, W2, topk_ids, topk_weights, tile_m=tile_m)
    torch.cuda.synchronize()

    a2_ref = torch_moe_gemm1(X, W1, None, None, topk_ids, topk_weights, inter_dim, False)
    ref = torch_moe_gemm2(
        a2_ref.to(torch.bfloat16), W2, None, None, topk_ids, topk_weights, model_dim, doweight_stage2=True
    )
    assert verify_output(got.float(), ref, atol=0.05, rtol=0.05)


# v5: the shapes real models have, as opposed to the two the tables were tuned
# on. model_dim/inter_dim are the MoE block's -- Kimi-K3 routes in a 3584-wide
# latent rather than its 7168 hidden, so 3584 is what the grouped GEMM sees.
#
# Run with a card's slice of the experts and topk=1, which is what EP hands the
# layer: v3 dispatches one row per routed slot, so an arriving row names a single
# expert. The expert count is cut to 4 to keep the build cheap; it does not enter
# the tile choice, which is indexed by (stage, k_dim, n_out, tile_m).
_V5_SHAPES = [
    pytest.param(2048, 768, id="qwen3-30b-a3b"),
    pytest.param(4096, 14336, id="mixtral-8x7b"),
    pytest.param(2048, 512, id="qwen3.5-35b-a3b"),
    pytest.param(4096, 1024, id="qwen3.5-397b-a17b"),
    pytest.param(4096, 2048, id="deepseek-v4"),
    pytest.param(3584, 3072, id="kimi-k3"),
]


@pytest.mark.parametrize("model_dim, inter_dim", _V5_SHAPES)
@pytest.mark.parametrize("tile_m", [16, 128])
def test_the_layer_runs_on_the_shapes_real_models_have(model_dim, inter_dim, tile_m):
    """Every shape in the v5 matrix builds and agrees with torch, at both ends of tile_m.

    Guards the claim the coverage script measures: these compile and are correct.
    It says nothing about how *fast* they are -- the tuned tile lives in two
    shape-indexed tables that only name Mixtral, and
    ``repro/v5_shape_coverage.py`` is where that gap is quantified. This is here
    so a tile-table change cannot silently stop building one of them.

    tile_m=16 and 128 are the ends of the table, and they exercise different
    fallbacks: the wide shapes take the preferred tile, the narrow ones have to
    walk down the preference list to something their n_out divides.
    """
    dev = "cuda"
    experts, rows = 4, 64
    torch.manual_seed(0)
    X = (torch.randn(rows, model_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    W1 = (torch.randn(experts, 2 * inter_dim, model_dim, dtype=torch.bfloat16, device=dev) * 0.05).contiguous()
    W2 = (torch.randn(experts, model_dim, inter_dim, dtype=torch.bfloat16, device=dev) * 0.05).contiguous()
    topk_ids = torch.randint(0, experts, (rows, 1), dtype=torch.int32, device=dev).contiguous()
    topk_weights = torch.rand(rows, 1, dtype=torch.float32, device=dev).contiguous()

    got = moe_forward(X, W1, W2, topk_ids, topk_weights, tile_m=tile_m)
    torch.cuda.synchronize()

    a2_ref = torch_moe_gemm1(X, W1, None, None, topk_ids, topk_weights, inter_dim, False)
    ref = torch_moe_gemm2(
        a2_ref.to(torch.bfloat16), W2, None, None, topk_ids, topk_weights, model_dim, doweight_stage2=True
    )
    assert verify_output(got.float(), ref, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("experts,topk,tokens,tile_m", [(512, 1, 1024, 32), (896, 1, 2048, 32), (257, 2, 300, 16)])
def test_more_experts_than_threads_still_routes(experts, topk, tokens, tile_m):
    """896 experts is Kimi-K3 at EP=1, and it used to be a build failure.

    More experts than threads means the scan sweeps them in chunks and carries a
    running tile count between the sweeps, which is the part that can be off by
    a chunk without anything else noticing. 257 is there for the ragged chunk,
    where all but one thread is past the last expert.
    """
    topk_ids, _ = _route(tokens, experts, topk)
    want = build_routing_torch(topk_ids, experts=experts, tile_m=tile_m)
    got = build_routing_device(topk_ids, experts=experts, tile_m=tile_m, exact=True, reuse=False)
    torch.cuda.synchronize()

    assert got.num_blocks == want.num_blocks
    assert torch.equal(got.expert_ids, want.expert_ids)
    assert torch.equal(_canonical(got, tile_m), _canonical(want, tile_m))


def test_the_expert_count_is_still_bounded():
    """The histogram is one LDS counter per expert, so ``E`` is an LDS budget."""
    with pytest.raises(ValueError, match="experts must be in"):
        compile_routing(experts=MAX_EXPERTS + 1, topk=1, tile_m=32)


def test_moe_forward_takes_a_prebuilt_routing():
    """A routing built once can drive both GEMMs, and a mismatched one is refused.

    Reusing it is the point of the two stages taking the same buffers, and the
    check exists because a routing built for another tile or token count would
    otherwise be a silent out-of-bounds read.
    """
    tokens, model_dim, inter_dim, experts, topk, tile_m = 32, 256, 128, 8, 2, 16
    dev = "cuda"
    torch.manual_seed(0)
    X = (torch.randn(tokens, model_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    W1 = (torch.randn(experts, 2 * inter_dim, model_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    W2 = (torch.randn(experts, model_dim, inter_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    topk_ids, topk_weights = _route(tokens, experts, topk)

    r = build_routing_torch(topk_ids, experts=experts, tile_m=tile_m)
    with_routing = moe_forward(X, W1, W2, topk_ids, topk_weights, tile_m=tile_m, routing=r)
    without = moe_forward(X, W1, W2, topk_ids, topk_weights, tile_m=tile_m)
    torch.cuda.synchronize()
    assert torch.equal(with_routing, without)

    with pytest.raises(ValueError, match="the routing was built for"):
        moe_forward(X, W1, W2, topk_ids, topk_weights, tile_m=32, routing=r)


@pytest.mark.parametrize("tile_m", [16, 64])
def test_moe_forward_agrees_whichever_builder_routes_it(tile_m):
    """Same layer, same bits, however the routing got built.

    Three ways: the host builder, the kernel with its count read back, and the
    kernel with the count left on the device and the grid set to a bound.
    Bit-identical rather than close -- all three hand the GEMMs the same tiles in
    the same order, and the tiles past the bound contribute nothing, so there is
    no reassociation anywhere to explain a difference.
    """
    tokens, model_dim, inter_dim, experts, topk = 64, 256, 128, 8, 2
    dev = "cuda"
    torch.manual_seed(0)
    X = (torch.randn(tokens, model_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    W1 = (torch.randn(experts, 2 * inter_dim, model_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    W2 = (torch.randn(experts, model_dim, inter_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    topk_ids, topk_weights = _route(tokens, experts, topk)

    got = {
        impl: moe_forward(X, W1, W2, topk_ids, topk_weights, tile_m=tile_m, routing_impl=impl).clone()
        for impl in ("device", "device-exact", "host")
    }
    torch.cuda.synchronize()
    assert torch.equal(got["device"], got["host"])
    assert torch.equal(got["device-exact"], got["host"])

    with pytest.raises(ValueError, match="routing_impl must be"):
        moe_forward(X, W1, W2, topk_ids, topk_weights, tile_m=tile_m, routing_impl="gpu")


def test_moe_forward_writes_a_caller_supplied_out():
    tokens, model_dim, inter_dim, experts, topk = 32, 256, 128, 8, 2
    dev = "cuda"
    torch.manual_seed(0)
    X = (torch.randn(tokens, model_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    W1 = (torch.randn(experts, 2 * inter_dim, model_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    W2 = (torch.randn(experts, model_dim, inter_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    topk_ids, topk_weights = _route(tokens, experts, topk)

    out = torch.zeros(tokens, model_dim, dtype=torch.bfloat16, device=dev)
    got = moe_forward(X, W1, W2, topk_ids, topk_weights, tile_m=16, out=out)
    torch.cuda.synchronize()
    assert got.data_ptr() == out.data_ptr()
    assert out.abs().sum() > 0


@pytest.mark.parametrize("bad", ["w2_shape", "weights_dtype", "x_rank"])
def test_moe_forward_rejects_mismatched_shapes(bad):
    tokens, model_dim, inter_dim, experts, topk = 8, 256, 128, 4, 2
    dev = "cuda"
    X = torch.zeros(tokens, model_dim, dtype=torch.bfloat16, device=dev)
    W1 = torch.zeros(experts, 2 * inter_dim, model_dim, dtype=torch.bfloat16, device=dev)
    W2 = torch.zeros(experts, model_dim, inter_dim, dtype=torch.bfloat16, device=dev)
    topk_ids, topk_weights = _route(tokens, experts, topk)

    if bad == "w2_shape":
        W2 = W2.transpose(1, 2).contiguous()
        match = "w2 must be"
    elif bad == "weights_dtype":
        topk_weights = topk_weights.to(torch.bfloat16)
        match = "topk_weights must be f32"
    else:
        X = X.unsqueeze(0)
        match = r"x must be \[tokens, model_dim\]"

    with pytest.raises(ValueError, match=match):
        moe_forward(X, W1, W2, topk_ids, topk_weights, tile_m=16)


# ── Gating: the other CDNA kernel that runs here unchanged ───────────────────


@pytest.mark.parametrize("experts, topk", [(8, 2), (32, 4), (128, 8)])
def test_moe_gating_matches_torch(experts, topk):
    """The fused softmax + top-K agrees with ``torch.topk(softmax(...))``.

    It reduces over a lane group with ``shuffle_xor`` and takes its width from
    the wave size, so nothing in it is wave64 -- but nothing had checked on
    wave32 either, and the layout it picks (``experts // VPT`` lanes per token)
    is the part that differs there. Ties are compared on the weight rather than
    the expert id, since two experts with the same probability may legitimately
    come back in either order.
    """
    tokens = 64
    torch.manual_seed(0)
    logits = (torch.rand(tokens, experts, device="cuda", dtype=torch.float32) * 4.0 - 2.0).to(torch.bfloat16)
    ids, weights = moe_gating(logits, topk=topk)
    torch.cuda.synchronize()

    probs = torch.softmax(logits.float(), dim=1)
    ref_p, _ = torch.topk(probs, topk, dim=1)

    assert ids.min() >= 0 and ids.max() < experts
    selected = probs.gather(1, ids.to(torch.long))
    assert verify_output(selected, ref_p, atol=0.02, rtol=0.02), "the selected experts are not the top-K"
    assert verify_output(weights, ref_p / ref_p.sum(dim=1, keepdim=True), atol=0.02, rtol=0.02)


def test_moe_forward_from_logits_matches_torch():
    """Gating in front of the layer, against a reference built on its own routing.

    The reference has to use the routing the kernel chose: with random logits two
    experts can be within f32 noise of each other, and which of them a top-K
    picks is not something a reference can be expected to reproduce.
    """
    tokens, model_dim, inter_dim, experts, topk = 64, 256, 128, 8, 2
    dev = "cuda"
    torch.manual_seed(0)
    X = (torch.randn(tokens, model_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    W1 = (torch.randn(experts, 2 * inter_dim, model_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    W2 = (torch.randn(experts, model_dim, inter_dim, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    logits = (torch.randn(tokens, experts, dtype=torch.bfloat16, device=dev) * 2.0).contiguous()

    got = moe_forward_from_logits(X, W1, W2, logits, topk=topk, tile_m=16)
    ids, weights = moe_gating(logits, topk=topk)
    torch.cuda.synchronize()

    a2_ref = torch_moe_gemm1(X, W1, None, None, ids, weights, inter_dim, False)
    ref = torch_moe_gemm2(a2_ref.to(torch.bfloat16), W2, None, None, ids, weights, model_dim, doweight_stage2=True)
    assert verify_output(got.float(), ref, atol=0.05, rtol=0.05)


# ── The topk reduce, on gfx11 ────────────────────────────────────────────────


@pytest.mark.parametrize("dtype_str", ["bf16", "f16", "f32"])
@pytest.mark.parametrize("tokens, topk, model_dim", [(32, 2, 2048), (7, 8, 1024), (128, 4, 512)])
def test_moe_reduce_runs_on_gfx11(dtype_str, tokens, topk, model_dim):
    """``compile_moe_reduction`` has only ever been tested on CDNA.

    It is 256 threads with no MFMA, no wave64 lane arithmetic and no atomics, so
    it should be portable -- but nothing checked, and stage2 is built to feed it
    rather than to duplicate it. A non-power-of-two token count is in the set
    because the grid is one workgroup per token.
    """
    X = torch.randn(tokens, topk, model_dim, dtype=_DTYPES[dtype_str], device="cuda")
    got = _reduce_topk(X)
    assert verify_output(got.float(), X.float().sum(dim=1), atol=0.05, rtol=0.05)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
