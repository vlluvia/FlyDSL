#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Grouped WMMA GEMM for RDNA3 / RDNA3.5 MoE stage1 (gfx11*, wave32).

The MoE counterpart of ``rdna3_f16_gemm``. Same pipeline -- double-buffered LDS
ping-pong, v16-operand WMMA, parameterised block tile -- with the three changes
a grouped GEMM needs:

  * **B is per expert.** A workgroup's M-tile belongs to one expert, named by
    ``expert_ids[bid_m]``, and reads the ``[N, K]`` slab at that index of a
    ``[experts, N, K]`` weight. Which is why the tile has to be the unit the
    routing pads to: a tile spanning two experts would need two weights.

  * **A is gathered.** Row ``i`` of the tile is activation row
    ``sorted_ids[bid_m*BLOCK_M + i] & 0xFFFFFF``, so the row band is a list of
    indices rather than a slice and the tiled copy cannot address it. Each
    thread issues its own ``buffer_load`` per row instead, which is the same
    instruction the sliced path emits -- only the address differs.

  * **Padding rows read as zero.** Routing pads each expert's row count up to
    the tile, and the pad carries the sentinel ``token == tokens``. Rather than
    branch on it, the A descriptor is bounded at ``tokens`` rows and asked to
    enforce it, so the hardware returns zero for the sentinel's address. On RDNA
    that check is opt-in (``bounds_checked``): the default descriptor mode
    enforces nothing and the sentinel would read whatever follows the
    activations.

Two modes
---------
``gate_up=False`` -- the linear grouped GEMM. Weight ``[experts, n_out, K]``,
output ``[num_blocks*BLOCK_M, n_out]`` in the padded routing order, which is the
order the tiles already write, so the epilogue is the dense kernel's. This is
the mode a reference is one ``torch.matmul`` per row, and it is what makes the
gather and the per-expert weight testable on their own.

``gate_up=True`` -- the fused stage1. Weight ``[experts, 2*n_out, K]`` with the
gate half first, output ``out[token, slot, :] = silu(gate) * up`` of shape
``[tokens, topk, n_out]``. Both halves of a channel have to reach the same
accumulator for the multiply, so a workgroup runs two B streams over one A tile
and carries two accumulator sets -- the alternative, tiling N over ``2*n_out``,
lands gate and up in different workgroups.

The fused epilogue also has to *scatter*: its rows are routing rows and go back
to ``(token, slot)``, which the routing order does not keep contiguous. Storing
straight from the accumulator would be 2-byte stores at eight unrelated
addresses per lane, because a gfx11 lane holds ``D[2*si + lane/16][lane%16]`` --
eight different rows of one channel. So the epilogue stages the tile through
LDS and reads it back channel-contiguous, one 128-bit store per row chunk. The
same bounded-descriptor trick drops the padded rows' stores.

Expert-parallel contract
------------------------
Nothing here is global. ``experts`` is **this rank's** expert count and
``expert_ids`` indexes into it; ``X`` is whatever rows a dispatch left on this
rank and the packed token id is a row of that buffer. EP=1 is then just the case
where the dispatch is the identity, so the same build serves both and wiring up
dispatch later does not touch this file.

Addressing note: buffer offsets are 32-bit byte offsets, so one A buffer is
limited to 2 GiB (``tokens * model_dim * 2``) and one output buffer likewise.
Weight slabs are byte-offset in 64-bit before the descriptor is built, so only a
single expert's ``N*K`` has to fit that bound.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import vector
from flydsl.expr import as_ir_value, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.utils.arith import _to_raw as as_mlir_value
from flydsl.runtime.device import get_rocm_arch
from kernels.gemm.rdna3_f16_gemm import K_PAD, WAVE_SIZE, WMMA_K, WMMA_M, WMMA_N, _sched_plan

# LDS budget per workgroup on gfx11.
LDS_BYTES = 64 * 1024

# 8 bf16/f16 = one 128-bit GMEM/LDS access.
LOAD_VEC = 8

IN_DTYPES = {"bf16": fx.BFloat16, "f16": fx.Float16}
OUT_DTYPES = {"bf16": fx.BFloat16, "f16": fx.Float16, "f32": fx.Float32}

# Packed routing id: (slot << 24) | token, matching the CDNA MoE kernels and
# aiter's moe_sorting. The sentinel a padded row carries is (topk << 24) | tokens,
# whose token field addresses one row past the activations.
TOKEN_MASK = 0xFFFFFF
SLOT_SHIFT = 24

# -log2(e). silu(x) = x / (1 + e^-x) = x * rcp(1 + exp2(x * -log2 e)).
NEG_LOG2E = -1.4426950408889634


def create_moe_gemm1_module(
    *,
    model_dim: int,
    n_out: int,
    experts: int,
    gate_up: bool = False,
    topk: int = 1,
    in_dtype: str = "bf16",
    out_dtype: str = "bf16",
    # Block tile, named the way rdna3_f16_gemm_autotune names it. The default is
    # that module's TILE_16x64x64 decode tile: MoE splits the batch across
    # experts, so even a prefill hands each expert few enough rows that the
    # shortest tile the WMMA shape allows is the one that fits.
    reg_m: int = 1,
    reg_n: int = 2,
    reg_k: int = 4,
    waves_m: int = 1,
    waves_n: int = 2,
    lds_layout: str = "pad",
    sched_hint: bool = False,
):
    """Build the grouped GEMM. Returns ``(launch, BLOCK_M, BLOCK_N, BLOCK_K)``.

    ``BLOCK_M`` is the tile the caller's routing must pad each expert to.
    ``launch(out, x, w, sorted_ids, expert_ids, tokens, num_blocks, stream)``.

    ``n_out`` is the output channel count per row: the weight has ``n_out`` rows
    per expert when ``gate_up`` is off and ``2*n_out`` when it is on.
    """
    gpu_arch = str(get_rocm_arch() or "")
    if not gpu_arch.startswith("gfx11"):
        raise RuntimeError(
            f"rdna3_moe requires gfx11* (RDNA3 / RDNA3.5); current arch is {gpu_arch!r}. "
            "The CDNA MoE kernels live in kernels/moe/moe_gemm_2stage."
        )
    if in_dtype not in IN_DTYPES:
        raise ValueError(f"in_dtype must be one of {sorted(IN_DTYPES)}, got {in_dtype!r}")
    if out_dtype not in OUT_DTYPES:
        raise ValueError(f"out_dtype must be one of {sorted(OUT_DTYPES)}, got {out_dtype!r}")
    if gate_up and int(topk) < 1:
        raise ValueError(f"gate_up needs the routing topk to address its output rows, got {topk}")

    K = int(model_dim)
    N = int(n_out)
    E = int(experts)
    TOPK = int(topk)
    ld_a = K
    ld_b = K
    ld_c = N
    # Rows of weight per expert: the gate half then the up half.
    w_slab_rows = 2 * N if gate_up else N

    BLOCK_M = WMMA_M * reg_m * waves_m
    BLOCK_N = WMMA_N * reg_n * waves_n
    BLOCK_K = WMMA_K * reg_k
    THREADS_PER_BLOCK = waves_m * waves_n * WAVE_SIZE

    if N % BLOCK_N:
        raise ValueError(f"n_out={N} must be a multiple of BLOCK_N={BLOCK_N}")
    if K % BLOCK_K:
        raise ValueError(f"model_dim={K} must be a multiple of BLOCK_K={BLOCK_K}")
    # A row is read BLOCK_K at a time as 128-bit chunks, so the row length has to
    # keep every chunk 16-byte aligned.
    if ld_a % LOAD_VEC or ld_b % LOAD_VEC:
        raise ValueError(f"model_dim={K} must be a multiple of {LOAD_VEC} for the 128-bit copy")

    num_k_tiles = K // BLOCK_K
    if num_k_tiles < 2:
        raise ValueError(
            f"the prefetch pipeline needs at least 2 K-tiles; model_dim={K} over BLOCK_K={BLOCK_K} gives {num_k_tiles}"
        )

    # G2S thread geometry: thread (tk, tm) moves the 128-bit chunk at columns
    # [tk*LOAD_VEC, +LOAD_VEC) of row tm, and the copy repeats over M to cover
    # the tile -- so tm = tid // THRS_K and tk = tid % THRS_K.
    THRS_K = BLOCK_K // LOAD_VEC
    THRS_M = THREADS_PER_BLOCK // THRS_K
    if THRS_K * THRS_M != THREADS_PER_BLOCK:
        raise ValueError(f"BLOCK_K={BLOCK_K} does not divide {THREADS_PER_BLOCK} threads into whole 128-bit chunks")
    if BLOCK_M % THRS_M or BLOCK_N % THRS_M:
        raise ValueError(f"BLOCK_M={BLOCK_M} and BLOCK_N={BLOCK_N} must both be multiples of THRS_M={THRS_M}")
    REPS_M_A = BLOCK_M // THRS_M

    n_b_tiles = 2 if gate_up else 1
    G2S_CHUNKS = (BLOCK_M + n_b_tiles * BLOCK_N) * BLOCK_K // THREADS_PER_BLOCK // LOAD_VEC
    SCHED_PLAN = _sched_plan(reg_m, reg_n, reg_k, G2S_CHUNKS) if sched_hint else ()

    if lds_layout not in ("pad", "kblock"):
        raise ValueError(f"lds_layout must be 'pad' or 'kblock', got {lds_layout!r}")
    k_pad = 0 if lds_layout == "kblock" else K_PAD
    ROW_STRIDE_A = BLOCK_K + k_pad
    ROW_STRIDE_B = BLOCK_K + k_pad
    LDS_A_SIZE = BLOCK_M * ROW_STRIDE_A
    LDS_B_SIZE = BLOCK_N * ROW_STRIDE_B
    LDS_ONE_BUF = LDS_A_SIZE + n_b_tiles * LDS_B_SIZE
    if lds_layout == "pad" and ROW_STRIDE_A % LOAD_VEC:
        raise ValueError(
            f"K_PAD={k_pad} leaves an LDS row of {ROW_STRIDE_A} elements, "
            f"not a multiple of the {LOAD_VEC}-element vector store"
        )

    elem_dtype = IN_DTYPES[in_dtype]
    out_elem_cls = OUT_DTYPES[out_dtype]
    elem_bytes = elem_dtype.width // 8
    out_bytes = out_elem_cls.width // 8
    acc_size = 8 * reg_m * reg_n

    # ── CShuffle staging geometry (gate_up only) ─────────────────────────
    # The staged tile is read back one 128-bit chunk per row per thread. The pad
    # is what keeps those reads off each other's banks: a row of BLOCK_N 16-bit
    # channels is exactly 128 bytes, so without it every thread reading the same
    # channel group of a different row hits the same bank.
    SVEC = 128 // out_elem_cls.width
    C_PAD = LOAD_VEC
    C_ROW_STRIDE = BLOCK_N + C_PAD
    LANES_N = BLOCK_N // SVEC
    ROWS_PER_PASS = THREADS_PER_BLOCK // LANES_N if LANES_N else 0
    if gate_up:
        if BLOCK_N % SVEC or THREADS_PER_BLOCK % LANES_N:
            raise ValueError(f"BLOCK_N={BLOCK_N} does not split into {SVEC}-element chunks over the block's threads")
        if ROWS_PER_PASS > BLOCK_M or BLOCK_M % ROWS_PER_PASS:
            raise ValueError(
                f"the CShuffle readback covers {ROWS_PER_PASS} rows per pass, which does not tile BLOCK_M={BLOCK_M}"
            )
        C_REPS_M = BLOCK_M // ROWS_PER_PASS

    # The pipeline buffers and the staging tile are never live at once, so they
    # share one allocation.
    cshuf_elems = -(-BLOCK_M * C_ROW_STRIDE * out_bytes // elem_bytes) if gate_up else 0
    LDS_TOTAL = max(2 * LDS_ONE_BUF, cshuf_elems)
    if LDS_TOTAL * elem_bytes > LDS_BYTES:
        raise ValueError(
            f"tile needs {LDS_TOTAL * elem_bytes} B of LDS, over the {LDS_BYTES} B a gfx11 workgroup may allocate"
        )

    grid_n = N // BLOCK_N
    is_bf16 = in_dtype == "bf16"

    def _wmma_op(a_vec, b_vec, acc):
        # On gfx11 the WMMA intrinsic takes v16 inputs (and a v8 accumulator),
        # and represents bf16 as i16.
        if is_bf16:
            return rocdl.wmma_f32_16x16x16_bf16(acc.type, a_vec.bitcast(fx.Int16), b_vec.bitcast(fx.Int16), acc).result
        return rocdl.wmma_f32_16x16x16_f16(acc.type, a_vec, b_vec, acc).result

    @fx.struct
    class _SharedStorage:
        lds: fx.Array[elem_dtype, LDS_TOTAL, 16]

    @flyc.kernel
    def moe_gemm1_kernel(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_w: fx.Tensor,
        arg_sorted_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        i32_tokens: fx.Int32,
        tiled_mma: fx.TiledMma,
        tiled_copy_g2s: fx.TiledCopy,
    ):
        lds_storage = fx.SharedAllocator().allocate(_SharedStorage).peek()
        lds_ptr = lds_storage.lds.ptr

        def _v8_load(v8_idx):
            elem_off = fx.Int32(v8_idx * 8)
            ptr_off = fx.add_offset(lds_ptr, fx.make_int_tuple(elem_off))
            return fx.make_view(fx.recast_iter(elem_dtype, ptr_off), fx.make_layout(8, 1)).load()

        tid = gpu.thread_id("x")
        bid_n = fx.Int32(gpu.block_id("x"))
        bid_m = fx.Int32(gpu.block_id("y"))

        tid_i = fx.Int32(tid)
        wave_id = tid // WAVE_SIZE
        lane = tid % WAVE_SIZE
        # The gfx11 v16 ABI has lanes 16-31 mirror lanes 0-15, so a lane's row is
        # selected by ``lane % 16`` alone and it carries all 16 K-elements.
        lane16 = lane % 16
        wave_m = wave_id // waves_n
        wave_n = wave_id % waves_n

        thr_g2s = tiled_copy_g2s.get_slice(tid)
        thr_mma = tiled_mma.thr_slice(tid)

        # ── Routing metadata ────────────────────────────────────────────
        def _load_i32(base_iter, idx):
            """One i32 at an element offset. Offsetting the pointer rather than
            indexing a view keeps the read correct for any index: a view's layout
            would have to carry the array's length, which is a runtime value."""
            return fx.make_view(base_iter + idx, fx.make_layout(1, 1))[0]

        sid_iter = fx.recast_iter(fx.Int32, fx.get_iter(arg_sorted_ids))
        eid_iter = fx.recast_iter(fx.Int32, fx.get_iter(arg_expert_ids))
        expert_id = _load_i32(eid_iter, bid_m)
        row_base = bid_m * fx.Int32(BLOCK_M)

        # ── Gathered A ──────────────────────────────────────────────────
        # Bounded at the real row count and asked to enforce it, so a padded
        # row's sentinel token (== tokens) reads zero instead of running off the
        # end of the activations.
        a_buf = fx.rocdl.make_buffer_tensor(
            fx.make_view(fx.get_iter(arg_a), fx.make_layout(1, 1)),
            num_records_bytes=fx.Int64(i32_tokens) * fx.Int64(ld_a * elem_bytes),
            bounds_checked=True,
        )
        a_iter = fx.get_iter(a_buf)

        tm = tid_i // fx.Int32(THRS_K)
        tk = tid_i % fx.Int32(THRS_K)
        # Row offsets are k-independent, so the gather's addresses are resolved
        # once here rather than once per k-tile.
        a_row_off = []
        for r in range_constexpr(REPS_M_A):
            token = _load_i32(sid_iter, row_base + fx.Int32(r * THRS_M) + tm) & fx.Int32(TOKEN_MASK)
            a_row_off.append(token * fx.Int32(ld_a) + tk * fx.Int32(LOAD_VEC))

        buf_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)
        uni_copy = fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype)

        # ── Per-expert B ────────────────────────────────────────────────
        # 64-bit so a slab past 2 GiB into the weight still addresses; the
        # descriptor built on top of it only has to cover one expert.
        w_base = fx.get_iter(arg_w) + fx.Int64(expert_id) * fx.Int64(w_slab_rows * ld_b)

        def _b_stream(row_offset):
            view = fx.make_view(w_base + fx.Int64(row_offset), fx.make_layout((N, K), (ld_b, 1)))
            return thr_g2s.partition_S(
                fx.flat_divide(fx.rocdl.make_buffer_tensor(view), fx.make_tile(BLOCK_N, BLOCK_K))[
                    None, None, bid_n, None
                ]
            )

        # The gate half occupies the first N rows of the slab and the up half the
        # rest, so the two streams are the same partition against two bases.
        pB_g = [_b_stream(0)]
        if const_expr(gate_up):
            pB_g.append(_b_stream(N * ld_b))

        # ── GMEM -> registers -> LDS ────────────────────────────────────
        def _lds_dst(buf_offset, base, rows, row_stride):
            ptr = fx.add_offset(lds_ptr, fx.make_int_tuple(buf_offset + base))
            if const_expr(lds_layout == "kblock"):
                layout = fx.make_layout(
                    (rows, (LOAD_VEC, BLOCK_K // LOAD_VEC)),
                    (LOAD_VEC, (1, rows * LOAD_VEC)),
                )
            else:
                layout = fx.make_layout((rows, BLOCK_K), (row_stride, 1))
            return thr_g2s.partition_D(fx.make_view(fx.recast_iter(elem_dtype, ptr), layout))[None, None, None]

        def _pA_s(buf_offset):
            return _lds_dst(buf_offset, 0, BLOCK_M, ROW_STRIDE_A)

        def _pB_s(buf_offset, half):
            return _lds_dst(buf_offset, LDS_A_SIZE + half * LDS_B_SIZE, BLOCK_N, ROW_STRIDE_B)

        def _lds_elem(rows, row_stride, row, col):
            if const_expr(lds_layout == "kblock"):
                return (col // LOAD_VEC * rows + row) * LOAD_VEC + col % LOAD_VEC
            return row * row_stride + col

        frag_copy_A = fx.make_fragment_like(_pA_s(0))
        frag_copy_B = [fx.make_fragment_like(_pB_s(0, h)) for h in range_constexpr(n_b_tiles)]

        # One rank-1 view per row the gather fills. Slicing the fragment gives the
        # right register pointer but a rank-2 ``(LOAD_VEC,1):(1,0)`` memref, and a
        # copy into that shape survives regmem-to-vector-SSA promotion as a
        # leftover register pointer ("ub.poison ... register operand/result
        # remain"). Re-viewing the same pointer flat is the shape the promotion
        # handles, and it is what the LDS side of this fragment already uses.
        frag_a_slots = [
            fx.make_view(fx.get_iter(frag_copy_A[None, r, 0]), fx.make_layout(LOAD_VEC, 1))
            for r in range_constexpr(REPS_M_A)
        ]

        def _gmem_load(k_tile):
            k_off = fx.Int32(k_tile) * fx.Int32(BLOCK_K)
            for r in range_constexpr(REPS_M_A):
                mem = fx.make_view(a_iter + (a_row_off[r] + k_off), fx.make_layout(LOAD_VEC, 1))
                fx.copy(buf_copy, mem, frag_a_slots[r])
            for h in range_constexpr(n_b_tiles):
                fx.copy(buf_copy, pB_g[h][None, None, None, k_tile], frag_copy_B[h])

        def _lds_store(buf_offset):
            fx.copy(uni_copy, frag_copy_A, _pA_s(buf_offset))
            for h in range_constexpr(n_b_tiles):
                fx.copy(uni_copy, frag_copy_B[h], _pB_s(buf_offset, h))

        # ── LDS -> v16 operands ─────────────────────────────────────────
        # A lane carries 16 contiguous K-elements of its row, held as two
        # adjacent v8 chunks.
        _concat16_mask = list(range(16))

        def _load_b_from_lds(rk, buf_offset, half):
            vecs = []
            base = buf_offset + LDS_A_SIZE + half * LDS_B_SIZE
            for rn in range_constexpr(reg_n):
                row = wave_n * (reg_n * WMMA_N) + 16 * rn + lane16
                lo = _v8_load((base + _lds_elem(BLOCK_N, ROW_STRIDE_B, row, 16 * rk)) // 8)
                hi = _v8_load((base + _lds_elem(BLOCK_N, ROW_STRIDE_B, row, 16 * rk + 8)) // 8)
                vecs.append(lo.shuffle(hi, _concat16_mask))
            return vecs

        def _load_a_from_lds(rk, rm_val, buf_offset):
            row = wave_m * (reg_m * WMMA_M) + 16 * rm_val + lane16
            lo = _v8_load((buf_offset + _lds_elem(BLOCK_M, ROW_STRIDE_A, row, 16 * rk)) // 8)
            hi = _v8_load((buf_offset + _lds_elem(BLOCK_M, ROW_STRIDE_A, row, 16 * rk + 8)) // 8)
            return lo.shuffle(hi, _concat16_mask)

        def _barrier():
            # gfx11 barrier -- split signal/wait and s_wait_dscnt are gfx12+.
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string="s_waitcnt lgkmcnt(0)\ns_barrier",
                constraints="",
                has_side_effects=True,
            )

        def _compute_k_tile(accs_in, buf_offset):
            """All reg_k WMMA steps over one LDS buffer, for every B stream.

            One A fragment feeds every stream's reg_n WMMAs, which is the whole
            reason gate and up share a workgroup. Issuing the next A read ahead
            of its consumer gives the register allocator something to overlap the
            LDS wait with; see the same loop in rdna3_f16_gemm for what it is
            worth.
            """
            new_accs = list(accs_in)
            for rk in range_constexpr(reg_k):
                b_vecs = [_load_b_from_lds(rk, buf_offset, h) for h in range_constexpr(n_b_tiles)]
                a_next = _load_a_from_lds(rk, 0, buf_offset)
                for rm in range_constexpr(reg_m):
                    a_vec = a_next
                    if const_expr(rm + 1 < reg_m):
                        a_next = _load_a_from_lds(rk, rm + 1, buf_offset)
                    for h in range_constexpr(n_b_tiles):
                        for rn in range_constexpr(reg_n):
                            idx = h * (reg_m * reg_n) + rm * reg_n + rn
                            new_accs[idx] = _wmma_op(a_vec, b_vecs[h][rn], new_accs[idx])
            return new_accs

        def _sched_k_tile():
            emit = {
                "vmem": rocdl.sched_vmem,
                "mfma": rocdl.sched_mfma,
                "dsrd": rocdl.sched_dsrd,
                "dswr": rocdl.sched_dswr,
            }
            for group, count in SCHED_PLAN:
                emit[group](count)

        zero_acc = fx.full(8, 0.0, fx.Float32)
        n_acc = n_b_tiles * reg_m * reg_n
        c_lds_buf_stride = LDS_ONE_BUF

        def _accumulate():
            _gmem_load(fx.Int32(0))
            _lds_store(0)
            _barrier()

            init_state = [zero_acc for _ in range_constexpr(n_acc)]
            for iv, state in range(0, num_k_tiles - 1, 1, init=init_state):
                s_accs = list(state[:n_acc])
                _gmem_load(iv + 1)
                s_accs = _compute_k_tile(s_accs, iv % 2 * c_lds_buf_stride)
                _lds_store((1 - iv % 2) * c_lds_buf_stride)
                if const_expr(sched_hint):
                    _sched_k_tile()
                _barrier()
                results = yield list(s_accs)

            return _compute_k_tile(list(results[:n_acc]), ((num_k_tiles - 1) % 2) * c_lds_buf_stride)

        # ── Epilogues ───────────────────────────────────────────────────
        # frag_C flattens as si + 8*(rm + reg_m*rn), which is the order this walk
        # over the accumulators produces.
        def _acc_order(accs, half):
            off = half * (reg_m * reg_n)
            return [accs[off + rm * reg_n + rn] for rn in range_constexpr(reg_n) for rm in range_constexpr(reg_m)]

        def _fill_out_frag(frag_C, out_elems):
            if const_expr(out_elem_cls is fx.Float32):
                frag_out = frag_C
            else:
                frag_out = fx.make_fragment_like(frag_C, out_elem_cls.ir_type)
            frag_out.store(
                vector.from_elements(T.vec(acc_size, out_elem_cls.ir_type), [as_ir_value(e) for e in out_elems])
            )
            return frag_out

        def _convert(vals):
            if const_expr(out_elem_cls is fx.Float32):
                return vals
            return [v.to(out_elem_cls) for v in vals]

        def _store_linear(accs):
            """Rows are the padded routing order, which is the order the tile
            already writes, so this is the dense kernel's epilogue against a C
            pointer offset to the tile's row band."""
            copy_out = fx.make_copy_atom(fx.rocdl.BufferCopy(out_elem_cls.width), out_elem_cls)
            thr_r2g_C = fx.make_tiled_copy_C(copy_out, tiled_mma).get_slice(tid)
            c_view = fx.make_view(
                fx.get_iter(arg_c) + fx.Int64(bid_m) * fx.Int64(BLOCK_M * ld_c),
                fx.make_layout((BLOCK_M, N), (ld_c, 1)),
            )
            tC = fx.flat_divide(fx.rocdl.make_buffer_tensor(c_view), fx.make_tile(BLOCK_M, BLOCK_N))[
                None, None, 0, bid_n
            ]
            frag_C = thr_mma.make_fragment_C(tC)
            vals = [acc[si] for acc in _acc_order(accs, 0) for si in range_constexpr(8)]
            frag_out = _fill_out_frag(frag_C, _convert(vals))
            fx.copy(copy_out, thr_r2g_C.retile(frag_out), thr_r2g_C.partition_S(tC))

        def _silu_mul(accs):
            """silu(gate) * up over the accumulator pair, in f32.

            ``exp2`` is called raw: every argument is a routed activation times a
            weight, well inside the range where the bare hardware op is exact,
            and the guard ``math.exp2`` wraps it in costs three instructions per
            element for nothing.
            """
            gate = _acc_order(accs, 0)
            up = _acc_order(accs, 1)
            out = []
            for g_acc, u_acc in zip(gate, up):
                for si in range_constexpr(8):
                    g = fx.Float32(g_acc[si])
                    u = fx.Float32(u_acc[si])
                    e = fx.Float32(rocdl.exp2(T.f32, as_mlir_value(g * NEG_LOG2E)))
                    out.append(g * fx.Float32(rocdl.rcp(T.f32, as_mlir_value(fx.Float32(1.0) + e))) * u)
            return out

        def _store_scatter(accs):
            """Stage the tile through LDS, then scatter it by (token, slot).

            Straight from the accumulator each lane would hold eight rows of one
            channel, so the store would be eight 2-byte scatters per lane. LDS
            turns that into one 128-bit store per row chunk, and the readback is
            the only place the routing id is needed a second time.
            """
            lds_c = fx.make_view(
                fx.recast_iter(out_elem_cls, lds_ptr),
                fx.make_layout((BLOCK_M, BLOCK_N), (C_ROW_STRIDE, 1)),
            )
            cshuf_atom = fx.make_copy_atom(fx.UniversalCopy(out_elem_cls.width), out_elem_cls)
            thr_cshuf = fx.make_tiled_copy_C(cshuf_atom, tiled_mma).get_slice(tid)
            frag_out = _fill_out_frag(thr_mma.make_fragment_C(lds_c), _convert(_silu_mul(accs)))

            # The pipeline's last _compute_k_tile is still reading the buffer this
            # is about to overwrite.
            _barrier()
            fx.copy(cshuf_atom, thr_cshuf.retile(frag_out), thr_cshuf.partition_D(lds_c))
            _barrier()

            # out[token, slot, :] is row (token*topk + slot). Bounded at the real
            # row count so a padded row's sentinel -- token == tokens, slot ==
            # topk -- addresses past the end and the store is dropped.
            out_buf = fx.rocdl.make_buffer_tensor(
                fx.make_view(fx.get_iter(arg_c), fx.make_layout(1, 1)),
                num_records_bytes=fx.Int64(i32_tokens) * fx.Int64(TOPK * N * out_bytes),
                bounds_checked=True,
            )
            out_iter = fx.get_iter(out_buf)
            lds_c_ptr = fx.recast_iter(out_elem_cls, lds_ptr)
            uni_copy_out = fx.make_copy_atom(fx.UniversalCopy128b(), out_elem_cls)
            buf_copy_out = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), out_elem_cls)

            chan = (tid_i % fx.Int32(LANES_N)) * fx.Int32(SVEC)
            row_in_pass = tid_i // fx.Int32(LANES_N)
            for rep in range_constexpr(C_REPS_M):
                row = row_in_pass + fx.Int32(rep * ROWS_PER_PASS)
                packed = _load_i32(sid_iter, row_base + row)
                out_row = (packed & fx.Int32(TOKEN_MASK)) * fx.Int32(TOPK) + (packed >> fx.Int32(SLOT_SHIFT))

                src = fx.make_view(
                    fx.add_offset(lds_c_ptr, fx.make_int_tuple(row * fx.Int32(C_ROW_STRIDE) + chan)),
                    fx.make_layout(SVEC, 1),
                )
                staged = fx.make_fragment_like(src)
                fx.copy(uni_copy_out, src, staged)

                dst = fx.make_view(
                    out_iter + (out_row * fx.Int32(N) + bid_n * fx.Int32(BLOCK_N) + chan),
                    fx.make_layout(SVEC, 1),
                )
                fx.copy(buf_copy_out, staged, dst)

        if const_expr(gate_up):
            _store_scatter(_accumulate())
        else:
            _store_linear(_accumulate())

    @flyc.jit
    def launch_moe_gemm1(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_w: fx.Tensor,
        arg_sorted_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_num_blocks: fx.Int32,
        stream: fx.Stream,
    ):
        # 16x16x16 v16 WMMA atom over a waves_m x waves_n wave grid. The
        # permutation keeps wave wm on the contiguous row band
        # [wm*reg_m*16, +reg_m*16); the default stamping interleaves the repeats,
        # which measured 62% slower on the dense kernel.
        tiled_mma = fx.make_tiled_mma(
            fx.make_mma_atom(fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, elem_dtype, fx.Float32)),
            fx.make_layout((waves_m, waves_n, 1), (waves_n, 1, 0)),
            permutation=(
                fx.make_layout((WMMA_M, waves_m, reg_m), (1, WMMA_M * reg_m, WMMA_M)),
                fx.make_layout((WMMA_N, waves_n, reg_n), (1, WMMA_N * reg_n, WMMA_N)),
                WMMA_K,
            ),
        )
        tiled_copy_g2s = fx.make_tiled_copy(
            fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype),
            fx.make_layout(
                ((THRS_K, THRS_M), (1, LOAD_VEC)),
                ((THRS_M * LOAD_VEC, 1), (1, THRS_M)),
            ),
            fx.make_tile(THRS_M, BLOCK_K),
        )

        moe_gemm1_kernel(
            arg_c,
            arg_a,
            arg_w,
            arg_sorted_ids,
            arg_expert_ids,
            i32_tokens,
            tiled_mma,
            tiled_copy_g2s,
        ).launch(
            grid=(grid_n, fx.Int64(i32_num_blocks), 1),
            block=(THREADS_PER_BLOCK, 1, 1),
            stream=stream,
        )

    def launch(arg_c, arg_a, arg_w, arg_sorted_ids, arg_expert_ids, tokens, num_blocks, stream):
        if int(num_blocks) <= 0:
            return None
        return launch_moe_gemm1(
            arg_c,
            arg_a,
            arg_w,
            arg_sorted_ids,
            arg_expert_ids,
            int(tokens),
            int(num_blocks),
            stream,
        )

    launch.experts = E
    launch.gate_up = gate_up
    launch.topk = TOPK
    return launch, BLOCK_M, BLOCK_N, BLOCK_K
