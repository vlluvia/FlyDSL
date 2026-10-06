# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Host-side entry points for the RDNA3 MoE kernels.

The builder in ``grouped_gemm`` takes a block tile spelled the way
``rdna3_f16_gemm_autotune`` spells one, five numbers deep. What a caller actually
has is the tile its routing padded each expert to, so this module maps the one to
the other, caches the build, and names the two stages in MoE terms rather than
GEMM terms.
"""

import functools

from kernels.moe.rdna3_moe.grouped_gemm import create_grouped_gemm_module, validate_request

# Block tile per stage and routing tile_m, as
# (reg_m, reg_n, reg_k, waves_m, waves_n), measured with
# scripts/sweep_rdna3_moe_tiles.py on gfx1100 at model_dim=2048, inter_dim=768,
# 8 local experts, topk=2.
#
# MoE splits a batch across experts, so the rows one expert gets are few even in
# prefill and the short tiles are the ones that matter. 16 is the shortest the
# WMMA shape allows. Which tile_m to ask for is the caller's call -- a prefill
# wants 64, a decode wants 16, and the middle is flat -- but for a given tile_m
# the block shape underneath it is this module's problem.
#
# The table used to be the dense kernel's autotune output, reused unchanged on
# the theory that the grouped GEMM only changes where the operands come from.
# That is wrong for gate_up, and mildly wrong for the rest. Two things drive it:
#
# Registers. A fused gate/up workgroup runs two B streams over one A tile, so it
# carries two accumulator sets: 2 * reg_m * reg_n vectors of eight VGPRs. The
# dense 128x128x32 tile (reg_m=reg_n=4) therefore wants 32 of them, 256 VGPRs of
# accumulator alone, and gfx11 caps a thread at 256 -- the kernel spilled 769
# times and ran at 7.5 TFLOP/s. The fix is to buy width in N from *waves*
# instead of registers: reg_n=1 with waves_n=2 covers the same BLOCK_N at a
# quarter the accumulators.
#
# LDS. The pipeline double-buffers, so a buffer of BLOCK_K=64 puts every stage
# over 32 KB and only one workgroup fits a CU -- 2 waves per SIMD of a possible
# 16, which is very little to hide memory latency with. BLOCK_K=32 halves it and
# lets two workgroups in, which is most of what the single-B stages gained here.
#
# What that is worth at 1024 tokens (TFLOP/s, was -> now):
#
#     tile_m         16           32           64          128
#     gate_up    36.8 (kept)  42.0 -> 47.6  45.4 -> 56.6  7.6 -> 50.9
#     down        7.2 ->  8.0  ~   -> 49.5  51.9 -> 55.9   ~  -> 52.0
#
# (the gate_up 16 and down 16 columns are decode tiles, measured at 32 tokens.)
#
# Each entry is a preference list, not one tile. The measured tile comes first;
# the rest narrow BLOCK_N or BLOCK_K for a caller whose n_out or k_dim is too
# small to divide it, since the table is tuned for LLM-sized dimensions and a
# small shape should still build. `linear` is the unfused stage1 -- same pipeline
# as `down`, but its N is inter_dim rather than model_dim, and a narrow N does
# not have the blocks to spare for a wide tile.
_TILES = {
    "gate_up": {
        16: ((1, 2, 4, 1, 2), (1, 1, 4, 1, 2), (1, 2, 2, 1, 2), (1, 1, 2, 1, 2)),  # 16x64x64
        32: ((2, 1, 4, 1, 2), (2, 1, 2, 1, 2), (2, 1, 2, 1, 1)),  # 32x32x64
        64: ((2, 1, 4, 2, 2), (2, 1, 2, 2, 2), (2, 1, 2, 2, 1)),  # 64x32x64
        128: ((4, 1, 2, 2, 2), (2, 1, 2, 4, 2), (4, 1, 2, 2, 1)),  # 128x32x32
    },
    "down": {
        16: ((1, 2, 2, 1, 2), (1, 2, 2, 1, 1)),  # 16x64x32
        32: ((2, 2, 2, 1, 4), (2, 2, 2, 1, 2), (2, 2, 2, 1, 1)),  # 32x128x32
        64: ((2, 2, 2, 2, 4), (2, 2, 2, 2, 2), (2, 2, 2, 2, 1)),  # 64x128x32
        128: ((2, 2, 2, 4, 2), (2, 2, 2, 4, 1)),  # 128x64x32
    },
    "linear": {
        16: ((1, 2, 4, 1, 2), (1, 2, 2, 1, 2), (1, 2, 2, 1, 1)),  # 16x64x64
        32: ((2, 2, 4, 1, 2), (2, 2, 2, 1, 2), (2, 2, 2, 1, 1)),  # 32x64x64
        64: ((2, 2, 4, 2, 2), (2, 2, 2, 2, 2), (2, 2, 2, 2, 1)),  # 64x64x64
        128: ((4, 2, 4, 2, 2), (4, 2, 2, 2, 2), (2, 2, 2, 4, 1)),  # 128x64x64
    },
}

SUPPORTED_TILE_M = tuple(sorted(_TILES["linear"]))

# Exact-shape overrides (o5), keyed by ``(stage, k_dim, n_out, tile_m)``. A hit
# goes in *front* of the generic preference list, so the generic tile stays the
# fallback. Deliberately separate from ``_TILES``: promoting these wider tiles to
# the generic table's first choice regresses the D2048/I768 shape o3 tuned.
#
# Note the key's ``n_out`` is the one the *builder* is given, which for
# ``gate_up`` is inter_dim -- the kernel doubles it itself to run the gate and up
# B streams over one A tile.
_SHAPE_TILES = {
    ("gate_up", 4096, 14336, 128): ((4, 2, 2, 2, 2),),  # 128x64x32
    ("down", 14336, 4096, 128): ((4, 4, 2, 2, 2),),  # 128x128x32
    # o14. An inter_dim this narrow divides no BLOCK_N past 16, so ``gate_up``
    # is stuck with one wave32 per workgroup and the only axis left is K. The
    # enumerated fallback stops at BLOCK_K=64, but 2560 also divides 128, and
    # going that deep is worth ~20% at tile_m=16 (and ~8% at large M) because
    # it is what halves the barrier count on a workgroup that has nothing else
    # to overlap them with. 256 is past the turn -- the k-loop stops fitting.
    # ``down`` is flat across its BLOCK_N (it is on the bandwidth wall already);
    # 64 is entered only because it is worth ~1 us at tile_m=16.
    ("gate_up", 2560, 80, 16): ((1, 1, 8, 1, 1),),  # 16x16x128
    ("down", 80, 2560, 16): ((1, 4, 1, 1, 1),),  # 16x64x16
}

# Which shapes want the M tiles dispatched innermost (o6). A workgroup reads one
# A tile and one B slab, and whichever index moves innermost is the operand that
# gets reused out of cache. Only a stage1 wide enough that its N sweep evicts a B
# slab before the next M tile asks for it wants M innermost; ``down`` never does
# (its A tile is the large operand and its N sweep is short), and a shape whose
# whole weight set fits cache does not either, because the re-reads were hits.
_SHAPE_M_MAJOR = {
    ("gate_up", 4096, 14336),
    ("linear", 4096, 14336),
}

# How many tiles an expert owns, which is the axis neither the shape nor
# ``tile_m`` can express on its own -- the same D2048/I768 wants opposite things
# at 8 experts and at 16. Three buckets is as fine as this can usefully get: the
# caller only knows the *average* rows per expert (the real per-expert counts are
# on the device, and waiting for them would undo o2), and a routing skewed enough
# to cross a boundary is skewed enough that its tail dominates anyway.
_BUCKETS = ("partial", "few", "many")


def tile_bucket_for(rows_per_expert: float, tile_m: int) -> str:
    """Which of ``_BUCKETS`` an expert's row count falls in, in tiles."""
    tiles = float(rows_per_expert) / float(tile_m)
    if tiles <= 1.0:
        return "partial"
    if tiles <= 4.0:
        return "few"
    return "many"


# Bucket-indexed overrides (v5-2), keyed by ``(stage, bucket, tile_m)``, ranked
# between ``_SHAPE_TILES`` and ``_TILES``.
#
# Only ``down`` earns entries, and what they do is put BLOCK_K back to 64. o3
# halved it to 32 so a second workgroup fits a CU, which is most of what the
# single-B stages gained at D2048/I768 -- but BLOCK_K=32 doubles the k-step count
# and every step carries a barrier. For ``down`` the contraction is the
# *intermediate* size, which this generation has collapsed to 512-2048, so there
# are too few steps left to amortise the barriers and the trade inverts. It
# inverts only where occupancy was not the binding constraint to begin with,
# which is the buckets where an expert owns few enough tiles that the CU is not
# full either way -- hence the key.
#
# Transcribed from the v5-2 write-up rather than re-measured here; re-confirm on
# your card with ``docs/rdna3_moe_optimization/repro/v5_tile_ab.py``, which
# empties this table for one arm and leaves it for the other.
_BUCKET_TILES = {
    ("down", "partial", 16): ((1, 2, 4, 1, 2),),  # 16x64x64
    ("down", "few", 16): ((1, 2, 4, 1, 2),),
    ("down", "partial", 32): ((2, 2, 4, 1, 4),),  # 32x128x64
    ("down", "few", 32): ((2, 2, 4, 1, 4),),
    ("down", "partial", 64): ((2, 2, 4, 2, 4),),  # 64x128x64
    ("down", "few", 64): ((2, 2, 4, 2, 4),),
}

WMMA_TILE = 16  # the WMMA shape is 16x16x16, so every block dimension is a multiple
_REG_CHOICES = (8, 4, 2, 1)
_WAVE_CHOICES = (1, 2, 4, 8)
# Eight wave32 waves is 256 threads, the block size the tuned table tops out at.
_MAX_WAVES = 8


@functools.lru_cache(maxsize=64)
def _narrowings(tile_m: int) -> tuple[tuple[int, int, int, int, int], ...]:
    """Every block tile of this height, widest first.

    The tuned tables are measured at LLM-sized dimensions, where a wide BLOCK_N
    has the blocks to fill and a deep BLOCK_K the k-steps to amortise. A shape
    that divides none of them still has to build -- a tensor-parallel MoE is the
    normal way to get one, since sharding the expert intermediate over the ranks
    leaves a per-rank ``inter_dim`` that is small and need not be a power of two
    (Qwen4Exp at TP=8 is 640/8 = 80, which divides no tuned BLOCK_N but does
    divide 16). So the preference list ends with the full enumeration, ordered
    widest BLOCK_N first, then deepest BLOCK_K.

    Width is bought from waves before registers at equal BLOCK_N, which is the
    same rule the ``gate_up`` comment above gives: a fused gate/up workgroup
    carries an accumulator set per B stream, and ``reg_n`` is what multiplies
    them.
    """
    out = []
    for waves_m in _WAVE_CHOICES:
        span = WMMA_TILE * waves_m
        if tile_m % span:
            continue
        reg_m = tile_m // span
        if reg_m not in _REG_CHOICES:
            continue
        for waves_n in _WAVE_CHOICES:
            if waves_m * waves_n > _MAX_WAVES:
                continue
            for reg_n in _REG_CHOICES:
                for reg_k in (4, 2, 1):
                    out.append((reg_m, reg_n, reg_k, waves_m, waves_n))
    out.sort(key=lambda c: (-(c[1] * c[4]), -c[2], -c[4]))
    return tuple(dict.fromkeys(out))


def _candidates(stage: str, k_dim: int, n_out: int, tile_m: int, bucket: str | None):
    """The tile preference list for one build, most specific table first."""
    ranked = [
        _SHAPE_TILES.get((stage, k_dim, n_out, tile_m)),
        _BUCKET_TILES.get((stage, bucket, tile_m)) if bucket else None,
        _TILES[stage][tile_m],
        _narrowings(tile_m),
    ]
    seen, out = set(), []
    for table in ranked:
        for cfg in table or ():
            if cfg not in seen:
                seen.add(cfg)
                out.append(cfg)
    return out


@functools.lru_cache(maxsize=256)
def _build(
    *,
    k_dim: int,
    n_out: int,
    experts: int,
    stage: str,
    topk: int,
    doweight: bool,
    tile_m: int,
    in_dtype: str,
    out_dtype: str,
    bounded_blocks: bool,
    bucket: str | None,
):
    # Anything wrong with the request rather than the tile is raised here, where
    # it is not competing with the loop's "try the next tile" -- no tile fixes an
    # unknown stage or a doweight the epilogue does not implement.
    validate_request(stage=stage, in_dtype=in_dtype, out_dtype=out_dtype, doweight=doweight, topk=topk)

    m_major = (stage, k_dim, n_out) in _SHAPE_M_MAJOR
    tried = []
    for reg_m, reg_n, reg_k, waves_m, waves_n in _candidates(stage, k_dim, n_out, tile_m, bucket):
        # BLOCK_N has to divide the output width and BLOCK_K the contraction.
        # Checked here so the common narrowing is arithmetic rather than a
        # caught exception; everything else the tile has to satisfy (LDS, the
        # thread geometry, the accumulator floor) is the builder's to judge, so
        # that stays a try.
        if n_out % (WMMA_TILE * reg_n * waves_n) or k_dim % (WMMA_TILE * reg_k):
            continue
        try:
            launch, block_m, block_n, block_k = create_grouped_gemm_module(
                k_dim=k_dim,
                n_out=n_out,
                experts=experts,
                stage=stage,
                topk=topk,
                doweight=doweight,
                in_dtype=in_dtype,
                out_dtype=out_dtype,
                reg_m=reg_m,
                reg_n=reg_n,
                reg_k=reg_k,
                waves_m=waves_m,
                waves_n=waves_n,
                bounded_blocks=bounded_blocks,
                m_major=m_major,
            )
        except ValueError as exc:
            tried.append(f"{(reg_m, reg_n, reg_k, waves_m, waves_n)}: {exc}")
            continue
        assert block_m == tile_m, f"tile table disagrees with the kernel: {block_m} != {tile_m}"
        return launch, block_m, block_n, block_k

    detail = "\n  ".join(tried) if tried else "no candidate divided the shape"
    raise ValueError(
        f"no tile for stage={stage!r} at k_dim={k_dim}, n_out={n_out}, tile_m={tile_m}:\n  {detail}"
    )


def compile_grouped_gemm(
    *,
    k_dim: int,
    n_out: int,
    experts: int,
    stage: str = "linear",
    topk: int = 1,
    doweight: bool = False,
    tile_m: int = 16,
    in_dtype: str = "bf16",
    out_dtype: str = "bf16",
    bounded_blocks: bool = False,
    rows_per_expert: float | None = None,
):
    """Build (and cache) one stage for one shape and tile.

    ``tile_m`` must be the row granularity the caller's routing pads each expert
    to: a tile spanning two experts would need two weight slabs. Returns
    ``(launch, BLOCK_M, BLOCK_N, BLOCK_K)`` with ``BLOCK_M == tile_m``.

    ``rows_per_expert`` only picks a bucket (see ``tile_bucket_for``), so it is
    resolved before the cache key is formed -- a caller passing the live average
    would otherwise miss the cache on every step and recompile.
    """
    if tile_m not in SUPPORTED_TILE_M:
        raise ValueError(f"tile_m must be one of {SUPPORTED_TILE_M}, got {tile_m}")
    return _build(
        k_dim=int(k_dim),
        n_out=int(n_out),
        experts=int(experts),
        stage=stage,
        topk=int(topk),
        doweight=bool(doweight),
        tile_m=int(tile_m),
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        bounded_blocks=bool(bounded_blocks),
        bucket=None if rows_per_expert is None else tile_bucket_for(rows_per_expert, tile_m),
    )


# The repro scripts swap a table out and rebuild; the cache they have to drop is
# the one keyed on the shape, not this wrapper's argument marshalling.
compile_grouped_gemm.cache_clear = _build.cache_clear
compile_grouped_gemm.cache_info = _build.cache_info


def compile_moe_gemm1(
    *,
    model_dim,
    inter_dim,
    experts,
    topk,
    tile_m=16,
    in_dtype="bf16",
    out_dtype="bf16",
    bounded_blocks=False,
    rows_per_expert=None,
):
    """MoE stage1: ``silu(gate) * up`` into ``[tokens, topk, inter_dim]``.

    The weight is ``[experts, 2*inter_dim, model_dim]``, gate half first.
    """
    return compile_grouped_gemm(
        k_dim=model_dim,
        n_out=inter_dim,
        experts=experts,
        stage="gate_up",
        topk=topk,
        tile_m=tile_m,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        bounded_blocks=bounded_blocks,
        rows_per_expert=rows_per_expert,
    )


def compile_moe_gemm2(
    *,
    model_dim,
    inter_dim,
    experts,
    topk,
    doweight=True,
    tile_m=16,
    in_dtype="bf16",
    out_dtype="bf16",
    bounded_blocks=False,
    rows_per_expert=None,
):
    """MoE stage2 down-projection into ``[tokens, topk, model_dim]``.

    The weight is ``[experts, model_dim, inter_dim]``. ``doweight`` folds in the
    routing weight, which leaves the topk sum unweighted -- which is what
    ``moe_reduce`` does. Summing here instead would need an atomic accumulate,
    and gfx11 has no bf16 atomic.
    """
    return compile_grouped_gemm(
        k_dim=inter_dim,
        n_out=model_dim,
        experts=experts,
        stage="down",
        topk=topk,
        doweight=doweight,
        tile_m=tile_m,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        bounded_blocks=bounded_blocks,
        rows_per_expert=rows_per_expert,
    )
