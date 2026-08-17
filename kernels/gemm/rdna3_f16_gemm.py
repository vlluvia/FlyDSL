#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""WMMA GEMM kernel for RDNA3 / RDNA3.5 (gfx11*, wave32).

Ported from rdna_f16_gemm.py (gfx120x). Same algorithm (4-warp double-
buffered LDS ping-pong, 128x128x32 tiles, swizzled grid mapping) but
adapted for the legacy v16-operand WMMA ABI used by RDNA3/RDNA3.5:

  * Input operands (A, B) are vector<16> instead of vector<8>; each
    lane carries 16 contiguous K-elements of one M (or N) row. Lanes
    0-15 carry distinct rows; lanes 16-31 carry duplicates of the same
    rows lanes 0-15 read. We just have all lanes do the LDS loads —
    duplicate loads are wasted bandwidth but simpler than a wave-half
    broadcast.
    TODO(perf): lanes 16-31 could ``ds_swizzle_b32`` XOR 16 broadcast
    from lanes 0-15 to halve LDS read bandwidth.

  * Accumulator (C/D) is still vector<8>, but the per-lane row mapping
    differs from gfx12: lane L holds D[2*si + (L/16)][L%16], i.e. even
    rows in lanes 0-15 and odd rows in lanes 16-31. The store-back loop
    uses ``g_row = base + 2*si + klane`` instead of the gfx12
    ``g_row = base + 8*klane + si``.

Computes C[M,N] = A[M,K] @ B_T[N,K]^T (same interface as
``rdna_f16_gemm.create_wmma_gemm_module``).

The block tile is a parameter and defaults to 128x128x32. Deciding it from the
shape -- which is worth up to 3.0x on shapes too small to fill the grid -- is
the job of ``rdna3_f16_gemm_autotune``; this module only builds what it is told.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import vector
from flydsl.expr import as_ir_value, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from kernels.common.kernels_common import cvt_sr_f32_to_bf16

WMMA_M = 16
WMMA_N = 16
WMMA_K = 16
WAVE_SIZE = 32

# The k-padding both operands pay to break LDS bank conflicts. Also the default
# for a_k_pad/b_k_pad below, so a caller sizing a tile against the LDS budget can
# assume it without the two drifting apart.
K_PAD = 8


def _sched_plan(reg_m, reg_n, reg_k, g2s_chunks):
    """The order the k-tile body is meant to issue in, as (group, count) pairs.

    Left alone, LLVM hoists the whole ds_store block into the middle of the WMMA
    stream, so each store's ``s_waitcnt vmcnt`` stalls the wave while the WMMAs
    behind it wait. This spends the WMMA stream as cover instead: global loads
    go out first, the next k-step's LDS reads hide under this k-step's math, and
    the stores drain under the tail, which also leaves the ``lgkmcnt(0)`` in
    front of the barrier with almost nothing left to wait for.

    Built here, from Python ints, because the kernel body is traced and cannot
    branch on values. The counts are hints: a group the dependences cannot
    satisfy is dropped rather than honoured.
    """
    per_rk_wmma = reg_m * reg_n
    per_rk_dsrd = 2 * (reg_m + reg_n)  # each v16 operand is two 128-bit reads
    chunk = max(1, reg_n)

    plan = [("vmem", g2s_chunks), ("dsrd", per_rk_dsrd)]
    dsrd_left = per_rk_dsrd * (reg_k - 1)
    dsrd_step = chunk
    dswr_left = g2s_chunks
    dswr_chunk = max(1, g2s_chunks // 2)
    for _ in range(reg_k):
        issued = 0
        while issued < per_rk_wmma:
            n = min(chunk, per_rk_wmma - issued)
            plan.append(("mfma", n))
            issued += n
            if dsrd_left:
                take = min(dsrd_step, dsrd_left)
                plan.append(("dsrd", take))
                dsrd_left -= take
            elif dswr_left:
                take = min(dswr_chunk, dswr_left)
                plan.append(("dswr", take))
                dswr_left -= take
    if dswr_left:
        plan.append(("dswr", dswr_left))
    return tuple(plan)


def _group_width(grid_m, group_m):
    """Largest grouping width <= group_m that divides grid_m.

    ``_swizzle_tile_id`` derives bid_m from a fixed group width, so a grid_m that
    is not a multiple of it makes the final group address tiles past the end of
    the grid. That is reachable at the default 128x128 tile: on gfx1100 it writes
    a wrong C at M of 1152, 1280 and 1664, and faults outright at 1536 and 2560,
    depending on whether the address past the grid happens to be mapped.
    """
    return max(d for d in range(1, min(group_m, grid_m) + 1) if grid_m % d == 0)


def _swizzle_tile_id(pid, grid_n, group_width):
    """Linear workgroup id -> (bid_m, bid_n).

    Walks group_width tiles down M before stepping in N, so workgroups that run
    concurrently share B tiles in L2. Plain integer arithmetic, so it evaluates
    the same on a host int as on the kernel's block id.
    """
    num_pid_in_group = group_width * grid_n
    group_id = pid // num_pid_in_group
    pid_in_group = pid % num_pid_in_group
    return group_id * group_width + (pid_in_group % group_width), pid_in_group // group_width


def create_wmma_gemm_module(
    M: int,
    N: int,
    K: int,
    in_dtype="bf16",
    out_dtype="bf16",
    *,
    rounding="rn",  # "rn" (round to nearest) or "rs" (stochastic rounding)
    # 128x128x32. That tile is right once the problem is large enough to fill the
    # grid, but it cuts only 4 workgroups at 256x256 on a 96-CU part, so most CUs
    # idle. Choosing a tile from the shape is worth up to 3.0x there and lives in
    # rdna3_f16_gemm_autotune, which drives these arguments.
    reg_m=4,
    reg_n=4,
    reg_k=2,
    waves_m=2,
    waves_n=2,
    group_m=8,
    a_k_pad=K_PAD,
    b_k_pad=K_PAD,
    lds_layout="pad",
    sched_hint=False,
    stagger=0,
    stream_k=0,
    # Row strides of the operands, when the caller has room to make them
    # something other than tight. A tile reads BLOCK_K of a row at a time, which
    # at K=2048 in bf16 is a 64-byte run every 4096 bytes, and 4096 is exactly
    # the spacing that camps on one set of L2. rocBLAS gains 1.20x on this shape
    # from ld+32 alone (255.71 -> 212.86 us), so it is worth offering even though
    # stagger already covers part of the same ground.
    lda=None,
    ldb=None,
    ldc=None,
):
    ld_a = K if lda is None else int(lda)
    ld_b = K if ldb is None else int(ldb)
    ld_c = N if ldc is None else int(ldc)
    if ld_a < K or ld_b < K or ld_c < N:
        raise ValueError(
            f"leading dimensions must cover the operands: lda={ld_a}, ldb={ld_b} "
            f"need K={K}, ldc={ld_c} needs N={N}"
        )
    # The G2S copy moves 128 bits per thread, so a row has to start 16-byte
    # aligned or the vectorized load reads across the boundary. Silently
    # producing NaN is the failure mode, so it is refused instead.
    vec_elems = 16 // (2 if in_dtype in ("bf16", "f16") else 4)
    if ld_a % vec_elems or ld_b % vec_elems:
        raise ValueError(
            f"lda={ld_a} and ldb={ld_b} must be multiples of {vec_elems} to keep "
            f"each row 16-byte aligned for the 128-bit copy"
        )

    gpu_arch = str(get_rocm_arch() or "")
    if not gpu_arch.startswith("gfx11"):
        raise RuntimeError(
            f"rdna3_f16_gemm requires gfx11* (RDNA3 / RDNA3.5); current arch is {gpu_arch!r}. "
            "Use rdna_f16_gemm.create_wmma_gemm_module on gfx120* (RDNA4)."
        )

    BLOCK_M = WMMA_M * reg_m * waves_m  # 128
    BLOCK_N = WMMA_N * reg_n * waves_n  # 128
    BLOCK_K = WMMA_K * reg_k  # 32
    NUM_WAVES = waves_m * waves_n  # 4
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE  # 128

    # One WMMA K-step is 16 elements and the v16 operand reads it as two 8-wide
    # chunks, so BLOCK_K only has to be a multiple of 16. reg_k=1 (BLOCK_K=16) is
    # what keeps the LDS tile small enough to fit more than one workgroup per CU.
    assert reg_k >= 1
    assert rounding in ("rn", "rs"), f"rounding must be 'rn' or 'rs', got {rounding!r}"
    if rounding == "rs":
        assert out_dtype == "bf16", "stochastic rounding currently supports bf16 output only"

    LOAD_VEC = 8  # 8 bf16 = 128-bit GMEM/LDS load
    # G2S thread geometry: thread (tk, tm) moves the 128-bit chunk at columns
    # [tk*LOAD_VEC, +LOAD_VEC) of row tm, and the tiled copy repeats over M to
    # cover the tile. Same assignment the hand-rolled offset tables computed as
    # ``row = tid // THRS_K, col = (tid % THRS_K) * LOAD_VEC``.
    THRS_K = BLOCK_K // LOAD_VEC
    THRS_M = THREADS_PER_BLOCK // THRS_K
    # 128-bit chunks of A and B one thread moves per k-tile; equals both the
    # global-load and the ds_store count in the loop body.
    G2S_CHUNKS = (BLOCK_M + BLOCK_N) * BLOCK_K // THREADS_PER_BLOCK // LOAD_VEC
    SCHED_PLAN = _sched_plan(reg_m, reg_n, reg_k, G2S_CHUNKS) if sched_hint else ()
    assert THRS_K * THRS_M == THREADS_PER_BLOCK
    assert BLOCK_M % THRS_M == 0 and BLOCK_N % THRS_M == 0

    # Two ways to lay a tile out in LDS, and the choice is what decides which
    # macro tiles are reachable at all:
    #
    #   "pad"    row-major, BLOCK_K + 8 elements per row. The pad is what keeps
    #            the 128-bit reads off each other's banks, and it is the only
    #            pad that does: the row start has to stay 16-byte aligned, which
    #            leaves 0, 8, 16 and 24, and of those only 8 and 24 spread the
    #            16 rows a read touches over all 32 banks.
    #
    #   "kblock" k-major in groups of 8, so element (row, k) sits at
    #            ((k // 8) * rows + row) * 8 + k % 8. Consecutive rows are then
    #            16 bytes apart, so the 8 lanes the LDS services per cycle cover
    #            128 contiguous bytes -- every bank, once -- with no pad at all.
    #
    # Dropping the pad is worth 20% of the LDS budget, which is what brings a
    # 256x256x32 tile inside the 64 KB a workgroup may allocate.
    assert lds_layout in ("pad", "kblock")
    if lds_layout == "kblock":
        assert BLOCK_K % LOAD_VEC == 0
        a_k_pad = b_k_pad = 0
    ROW_STRIDE_A = BLOCK_K + a_k_pad
    ROW_STRIDE_B = BLOCK_K + b_k_pad
    LDS_A_SIZE = BLOCK_M * ROW_STRIDE_A
    LDS_B_SIZE = BLOCK_N * ROW_STRIDE_B
    LDS_ONE_BUF = LDS_A_SIZE + LDS_B_SIZE
    LDS_TOTAL = 2 * LDS_ONE_BUF

    assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0
    assert K % BLOCK_K == 0

    # The LDS row is written 128 bits at a time, so a pad that leaves the row
    # length off a vector boundary tears every write. The kernel still runs and
    # quietly returns NaN, which is a bad way to find out: a_k_pad=4 at
    # BLOCK_K=32 does exactly that.
    if lds_layout == "pad":
        for name, pad in (("a_k_pad", a_k_pad), ("b_k_pad", b_k_pad)):
            if (BLOCK_K + pad) % LOAD_VEC:
                raise ValueError(
                    f"{name}={pad} leaves an LDS row of {BLOCK_K + pad} elements, "
                    f"which is not a multiple of the {LOAD_VEC}-element vector "
                    f"store; use a multiple of {LOAD_VEC}"
                )

    num_k_tiles = K // BLOCK_K
    if num_k_tiles < 2:
        raise ValueError(f"Need at least 2 K-tiles for prefetch pipeline; got K={K}, BLOCK_K={BLOCK_K}")

    grid_m = M // BLOCK_M
    grid_n = N // BLOCK_N

    group_width = _group_width(grid_m, group_m)

    # Every workgroup walks the k-tiles in the same order, so at any instant the
    # whole machine is reading the same k-slice of A and of B. When the row
    # stride is a power of two those reads land on a narrow set of memory
    # channels and queue behind each other. Measured at 2048 cubed with M and N
    # pinned so the grid never changes: K=2048 (4096-byte rows) runs 67.4
    # TFLOP/s while 1920, 1984, 2112, 2176 and 2304 all sit at 70.0-71.9. The
    # dip is worth 6.5%.
    #
    # Starting each workgroup at a different k-tile and wrapping around
    # decorrelates them. ``stagger`` is the step in k-tiles between the starts
    # of consecutive workgroup ids; the swizzle hands consecutive ids adjacent
    # row bands of A, which are exactly the ones worth separating. rocBLAS
    # solves the same problem the same way and every solution it picks at this
    # shape carries StaggerU=32.
    #
    # Only wired up when the k-tile count is a power of two, so the wraparound
    # is a mask rather than a division on a runtime value.
    assert stagger >= 0
    stagger_step = int(stagger) if num_k_tiles & (num_k_tiles - 1) == 0 else 0

    # ── Stream-K ────────────────────────────────────────────────────────
    # One workgroup per output tile leaves the machine ragged whenever the tile
    # count is not a multiple of the slot count, and it never is here: 2048 is a
    # power of two, so a power-of-two tile cuts a power-of-two grid, and 96 CUs
    # is 2^5*3. 256 tiles on 96 slots is 2.67 rounds of work that has to run as
    # 3, and the tiles all take the same time so they stay in lockstep and the
    # last round really is two thirds empty. Measured by holding the tile and
    # the round count fixed and moving only the fill -- 2048x2304 (288 tiles,
    # 3 rounds, 100%) runs 82.15 TFLOP/s against 2048x2048 (256 tiles, 3 rounds,
    # 89%) at 73.34 -- it is worth 1.120x, against the 1.125x the fill predicts.
    #
    # Stream-K splits the tiles x k-tiles unit space evenly instead, so every
    # workgroup gets the same amount of work and there is no tail. The price is
    # that a tile whose units straddle a workgroup boundary is computed in two
    # pieces that have to be added: the later workgroup writes its piece to a
    # workspace and flags it, the earlier one waits and folds it in. Measured
    # separately, that traffic is 11.26 us against the 25.1 us above.
    #
    # ``stream_k`` is the number of persistent workgroups. Two constraints:
    #
    #   * It may not exceed what the machine can hold resident, because the
    #     waiter spins on a workgroup that must already be running. The producer
    #     writes its piece as its *first* tile and the consumer wants it as its
    #     *last*, so the wait is nearly always already satisfied, but a
    #     non-resident producer would deadlock outright.
    #   * num_tiles must be at least stream_k, which makes every unit range at
    #     least num_k_tiles long and so caps a tile at two contributors. That is
    #     what lets a workgroup write at most one partial and read at most one.
    num_tiles = grid_m * grid_n
    sk_wgs = int(stream_k)
    assert sk_wgs >= 0
    if sk_wgs:
        if sk_wgs > num_tiles:
            raise ValueError(
                f"stream_k={sk_wgs} needs at least that many tiles to keep a tile to two "
                f"contributors; this shape cuts {num_tiles}"
            )
    sk_units = num_tiles * num_k_tiles
    # When every range happens to land on a tile boundary nobody splits anything,
    # and the whole partial/flag apparatus is dead weight that still costs: it
    # spills 56 registers and makes the epilogue read a zero slot it does not
    # need. That case is worth having because it is the one that wins -- at 2048
    # cubed, 128 workgroups of two whole tiles each beat both the plain path and
    # a splitting Stream-K -- so it gets a body with none of the machinery in it.
    sk_no_split = bool(sk_wgs) and all(
        (w * sk_units // sk_wgs) % num_k_tiles == 0 for w in range(sk_wgs)
    )
    if sk_wgs and not sk_no_split and (sk_wgs - 1) * sk_units > 2**31 - 1:
        # The splitting path forms pid * num_tiles * num_k_tiles in i32 to find
        # its unit range. Overflowing that wraps to a negative tile id, which
        # faults rather than answering wrongly, so it is refused up front.
        raise ValueError(
            f"stream_k={sk_wgs} with {sk_units} units overflows the i32 unit "
            f"space for this shape; use a count that divides into whole tiles"
        )
    # Whole-tile wraparound is the wrong shape here, because a visit covers a
    # slice of one tile's k rather than all of it. The same decorrelation still
    # has to happen -- most visits are whole tiles that would otherwise all walk
    # k in step, which is the 1.055x the plain path measured -- so it becomes a
    # rotation inside the visit's own range instead.
    sk_rot_step = int(stagger) if sk_wgs else 0
    if sk_wgs:
        stagger_step = 0
    # One slot per workgroup, not per tile: a workgroup writes at most one
    # partial, so the workspace is fixed at ~6 MB whatever the problem size.
    # The extra slot on the end stays zero, so the epilogue can fold in a
    # partial unconditionally and there is only ever one copy of it.
    sk_zero_slot = sk_wgs
    sk_ws_floats = 0 if sk_no_split else (sk_wgs + 1) * THREADS_PER_BLOCK * 8 * reg_m * reg_n
    sk_flags = 0 if sk_no_split else sk_wgs

    is_bf16 = in_dtype == "bf16"

    def _wmma_op(a_vec, b_vec, acc):
        # On gfx11 the WMMA intrinsic takes v16 inputs (and v8 accumulator).
        if is_bf16:
            a_i16 = a_vec.bitcast(fx.Int16)
            b_i16 = b_vec.bitcast(fx.Int16)
            return rocdl.wmma_f32_16x16x16_bf16(acc.type, a_i16, b_i16, acc).result
        return rocdl.wmma_f32_16x16x16_f16(acc.type, a_vec, b_vec, acc).result

    elem_dtype = fx.BFloat16 if is_bf16 else fx.Float16
    out_elem_cls = {"bf16": fx.BFloat16, "f16": fx.Float16, "f32": fx.Float32}[out_dtype]
    acc_size = 8 * reg_m * reg_n  # accumulator f32 VGPRs per thread

    # ── Shared-memory storage for double-buffered A+B LDS tiles ──────────
    # One flat bf16/f16 array; v8 chunks are addressed by byte_offset // 2
    # (element-index = byte_offset / sizeof(elem)) inside the kernel.
    # 16-byte alignment so the underlying buffer is suitable for v8 loads
    # (8 * 2 bytes = 16 bytes).
    @fx.struct
    class _SharedStorage:
        lds: fx.Array[elem_dtype, LDS_TOTAL, 16]

    @flyc.kernel
    def wmma_gemm_kernel(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_bt: fx.Tensor,
        arg_ws: fx.Tensor,  # fp32 partials; only touched on the Stream-K path
        arg_flag: fx.Tensor,  # one i32 per persistent workgroup, ditto
        tiled_mma: fx.TiledMma,
        tiled_copy_g2s: fx.TiledCopy,
        sr_seed: fx.Int32,  # runtime seed; only read on the stochastic-rounding path
    ):
        lds_storage = fx.SharedAllocator().allocate(_SharedStorage).peek()
        lds_ptr = lds_storage.lds.ptr  # i8-base aliased as elem_dtype*

        # ── v8 load/store helpers — element-indexed (v8_idx = byte_offset // 2 // 8) ──
        # Mirrors fp8_gemm_utils.S2RLoader._vec_load_16xf8: byte-offset the
        # pointer, recast to the element dtype, project into a v8 view.
        def _v8_load(v8_idx):
            elem_off = fx.Int32(v8_idx * 8)  # v8 chunks are 8 elements wide
            ptr_off = fx.add_offset(lds_ptr, fx.make_int_tuple(elem_off))
            typed_ptr = fx.recast_iter(elem_dtype, ptr_off)
            return fx.make_view(typed_ptr, fx.make_layout(8, 1)).load()

        tid = gpu.thread_id("x")
        pid = gpu.block_id("x")

        wave_id = tid // 32
        lane = tid % 32
        # On gfx11 the v16 ABI has lanes 16-31 mirror lanes 0-15, so the
        # M (or N) row is selected by ``lane % 16`` only. No klane shift
        # in the K dimension — each lane carries all 16 K-elements.
        lane16 = lane % 16

        # Where this workgroup enters the k loop. Uniform across the block, so
        # it lives in scalar registers and the wraparound below costs one SALU
        # add and one SALU and per trip. Everything is forced to Int32 because
        # the block id is index-typed while the loop counter reaching _gmem_load
        # is not, and mixing the two fails the arith.addi verifier.
        if const_expr(stagger_step):
            k_first = fx.Int32(pid) * stagger_step % num_k_tiles
        else:
            k_first = 0

        def _k_tile(k_base, step, rot=None, n_iter=None):
            """Global k-tile index of the ``step``-th tile of a visit to a tile.

            Both staggered paths rotate the walk so concurrent workgroups sit at
            different k, they just wrap around different spans: the plain path
            around all of k, Stream-K around the visit's own slice of it. The
            slice length is only known at run time, so the wrap is a compare and
            a select rather than the plain path's mask.
            """
            if const_expr(stagger_step):
                return (k_first + fx.Int32(step)) % num_k_tiles
            if const_expr(sk_rot_step):
                kk = rot + fx.Int32(step)
                return k_base + (kk >= n_iter).select(kk - n_iter, kk)
            if const_expr(sk_wgs):
                return k_base + fx.Int32(step)
            return step

        wave_m = wave_id // waves_n
        wave_n = wave_id % waves_n

        # Wave wm owns the contiguous row band [wm*reg_m*16, +reg_m*16). A
        # tiled_mma stamps its wave grid across the tile instead, putting repeat
        # rm at row (rm*waves_m + wm)*16; measured on gfx1100 that interleaving
        # costs 62% at 3072x3072x1024 (269 -> 436 us) while gaining 3-8% on the
        # medium shapes, so a tiled_mma standing in for this loop has to carry a
        # permutation that restores the banding rather than adopt the default.

        # ============================================================
        # GMEM -> registers -> LDS, through the tiled copy
        # ============================================================
        thr_g2s = tiled_copy_g2s.get_slice(tid)
        thr_mma = tiled_mma.thr_slice(tid)
        copy_out = fx.make_copy_atom(fx.rocdl.BufferCopy(out_elem_cls.width), out_elem_cls)
        thr_r2g_C = fx.make_tiled_copy_C(copy_out, tiled_mma).get_slice(tid)

        def _tile_operands(bid_m, bid_n):
            """The A and B row bands this tile reads, partitioned per thread.

            Built per visit rather than once, because a Stream-K workgroup walks
            several tiles and each wants a different band. On the plain path
            there is exactly one visit and this folds back to what it was.
            """
            tA = fx.flat_divide(fx.rocdl.make_buffer_tensor(arg_a), fx.make_tile(BLOCK_M, BLOCK_K))[
                None, None, bid_m, None
            ]
            tB = fx.flat_divide(fx.rocdl.make_buffer_tensor(arg_bt), fx.make_tile(BLOCK_N, BLOCK_K))[
                None, None, bid_n, None
            ]
            return thr_g2s.partition_S(tA), thr_g2s.partition_S(tB)

        buf_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)
        uni_copy = fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype)

        # The destination buffer alternates on ``iv % 2``, a loop-carried value,
        # so it cannot be selected from a list of per-stage views at trace time.
        # The view is built from the running element offset into the single LDS
        # allocation instead, which is what the flat _v8_store did by hand.
        def _lds_dst(buf_offset, base, rows, row_stride):
            ptr = fx.add_offset(lds_ptr, fx.make_int_tuple(buf_offset + base))
            if const_expr(lds_layout == "kblock"):
                # (row, k) with k split as (k % 8, k // 8): the low part is
                # contiguous so the 128-bit copy atom still sees eight adjacent
                # elements, the high part strides a whole plane of rows.
                layout = fx.make_layout(
                    (rows, (LOAD_VEC, BLOCK_K // LOAD_VEC)),
                    (LOAD_VEC, (1, rows * LOAD_VEC)),
                )
            else:
                layout = fx.make_layout((rows, BLOCK_K), (row_stride, 1))
            view = fx.make_view(fx.recast_iter(elem_dtype, ptr), layout)
            return thr_g2s.partition_D(view)[None, None, None]

        def _pA_s(buf_offset):
            return _lds_dst(buf_offset, 0, BLOCK_M, ROW_STRIDE_A)

        def _pB_s(buf_offset):
            return _lds_dst(buf_offset, LDS_A_SIZE, BLOCK_N, ROW_STRIDE_B)

        def _lds_elem(rows, row_stride, row, col):
            """Element index of (row, col) inside one A- or B-tile of the buffer."""
            if const_expr(lds_layout == "kblock"):
                return (col // LOAD_VEC * rows + row) * LOAD_VEC + col % LOAD_VEC
            return row * row_stride + col

        frag_copy_A = fx.make_fragment_like(_pA_s(0))
        frag_copy_B = fx.make_fragment_like(_pB_s(0))

        def _gmem_load(pA_g, pB_g, k_tile):
            fx.copy(buf_copy, pA_g[None, None, None, k_tile], frag_copy_A)
            fx.copy(buf_copy, pB_g[None, None, None, k_tile], frag_copy_B)

        def _lds_store(buf_offset):
            fx.copy(uni_copy, frag_copy_A, _pA_s(buf_offset))
            fx.copy(uni_copy, frag_copy_B, _pB_s(buf_offset))

        # ============================================================
        # LDS read helpers — v16 by concatenating two v8 loads
        # ============================================================
        # gfx11's v16 operand has element layout: lane L (L%16) carries 16
        # contiguous K-elements of row (lane%16). So per WMMA K-tile we
        # need 16 K-elements, stored as two contiguous v8 chunks at
        # offsets ``col_lo = 16*rk`` and ``col_hi = 16*rk + 8``.
        _concat16_mask = list(range(16))  # shuffle mask for v8 ++ v8 → v16

        def _load_b_from_lds(rk, buf_offset):
            vecs = []
            col_lo = 16 * rk
            col_hi = 16 * rk + 8
            for rn in range_constexpr(reg_n):
                row = wave_n * (reg_n * WMMA_N) + 16 * rn + lane16
                lds_idx_lo = buf_offset + LDS_A_SIZE + _lds_elem(BLOCK_N, ROW_STRIDE_B, row, col_lo)
                lds_idx_hi = buf_offset + LDS_A_SIZE + _lds_elem(BLOCK_N, ROW_STRIDE_B, row, col_hi)
                v_lo = _v8_load(lds_idx_lo // 8)
                v_hi = _v8_load(lds_idx_hi // 8)
                vecs.append(v_lo.shuffle(v_hi, _concat16_mask))
            return vecs

        def _load_a_single_from_lds(rk, rm_val, buf_offset):
            col_lo = 16 * rk
            col_hi = 16 * rk + 8
            row = wave_m * (reg_m * WMMA_M) + 16 * rm_val + lane16
            lds_idx_lo = buf_offset + _lds_elem(BLOCK_M, ROW_STRIDE_A, row, col_lo)
            lds_idx_hi = buf_offset + _lds_elem(BLOCK_M, ROW_STRIDE_A, row, col_hi)
            v_lo = _v8_load(lds_idx_lo // 8)
            v_hi = _v8_load(lds_idx_hi // 8)
            return v_lo.shuffle(v_hi, _concat16_mask)

        def _barrier():
            # gfx11 barrier — split signal/wait and s_wait_dscnt are gfx12+.
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string="s_waitcnt lgkmcnt(0)\ns_barrier",
                constraints="",
                has_side_effects=True,
            )

        def _do_compute_rk(accs_in, rk, buf_offset, b_vecs):
            new_accs = list(accs_in)
            # Each A fragment feeds reg_n back-to-back WMMAs. Emitting its read
            # immediately before its first consumer lets the register allocator
            # give every rm the same quad, and then the read for rm+1 cannot
            # issue until the WMMAs on rm have retired -- the body ends up with
            # a full ``lgkmcnt(0)`` drain in front of each group of reg_n WMMAs
            # instead of a partial wait. Issuing one fragment ahead is what
            # breaks that: the read in flight and the one being consumed need
            # different registers, so the wait has something to overlap.
            #
            # Only survives together with sched_hint. On its own the machine
            # scheduler sinks the read back down onto its consumer and the ISA
            # comes out unchanged at any prefetch depth; the group barriers are
            # what hold the reads up front for the allocator to see. Measured as
            # a pair at 256x256x32, worth 5% (109.1 -> 103.9 ms at 16384 cubed)
            # and 9 lgkmcnt(0) drains per k-tile down to 4. Neither shows up in
            # an instruction histogram -- the counts and the register total are
            # the same, only the order changes.
            a_next = _load_a_single_from_lds(rk, 0, buf_offset)
            for rm in range_constexpr(reg_m):
                a_vec = a_next
                if const_expr(rm + 1 < reg_m):
                    a_next = _load_a_single_from_lds(rk, rm + 1, buf_offset)
                for rn in range_constexpr(reg_n):
                    idx = rm * reg_n + rn
                    new_accs[idx] = _wmma_op(
                        a_vec,
                        b_vecs[rn],
                        new_accs[idx],
                    )
            return new_accs

        def _compute_k_tile(accs_in, buf_offset):
            """All reg_k WMMA steps over one LDS buffer.

            Step rk reads B into the registers step rk-1 is still using, so its
            reads cannot issue until that step's WMMAs retire, and the wait in
            front of them is a full lgkmcnt(0) drain rather than a partial one.
            Reading every step's B up front instead does remove that dependence,
            and it loses: 2*reg_n*(reg_k-1) more live registers pushes the
            allocator into recycling the A fragments harder, and the drains go
            from 4 per k-tile to 8. Measured at 256x256x32, 104.0 -> 118.3 ms.
            """
            new_accs = list(accs_in)
            for rk in range_constexpr(reg_k):
                new_accs = _do_compute_rk(new_accs, rk, buf_offset,
                                          _load_b_from_lds(rk, buf_offset))
            return new_accs

        def _sched_k_tile():
            emit = {"vmem": rocdl.sched_vmem, "mfma": rocdl.sched_mfma,
                    "dsrd": rocdl.sched_dsrd, "dswr": rocdl.sched_dswr}
            for group, count in SCHED_PLAN:
                emit[group](count)

        zero_acc = fx.full(8, 0.0, fx.Float32)
        n_acc = reg_m * reg_n
        c_lds_buf_stride = LDS_ONE_BUF

        def _one_k_tile(pA_g, pB_g, s_accs, read_off, write_off, load_tile):
            """Prefetch the next k-tile, consume this one, hand over, barrier."""
            _gmem_load(pA_g, pB_g, load_tile)
            s_accs = _compute_k_tile(s_accs, read_off)
            _lds_store(write_off)
            if const_expr(sched_hint):
                _sched_k_tile()
            _barrier()
            return s_accs

        def _accumulate(pA_g, pB_g, k_base, n_iter, rot=None):
            """Run the double-buffered pipeline over ``n_iter`` k-tiles.

            ``n_iter`` is a Python int on the plain path, so the trip count and
            the closing buffer parity both fold at trace time and the loop comes
            out exactly as it did before Stream-K existed. On the Stream-K path
            it is a runtime value and both become ordinary SSA; a range of one
            k-tile is fine, the loop simply runs zero trips and the epilogue
            consumes what the prologue put in buffer 0.
            """
            if const_expr(sk_wgs):
                # A previous visit's closing _compute_k_tile may still be
                # reading the buffer this prologue is about to overwrite.
                _barrier()
            # --- PROLOGUE ---
            _gmem_load(pA_g, pB_g, _k_tile(k_base, fx.Int32(0), rot, n_iter))
            _lds_store(0)
            _barrier()

            init_state = [zero_acc for _ in range_constexpr(n_acc)]

            # The read buffer alternates with the trip counter, so both offsets
            # are values derived from ``iv`` and the body spends ~10 VALU and
            # SALU ops per trip on them. Stepping the loop by two fixes each
            # half's parity at trace time and folds the arithmetic into ds_load
            # immediates: overhead instructions drop from 0.93 to 0.31 per WMMA
            # and VGPRs from 212 to 204. It is 1.7% slower (104.0 -> 105.8 ms at
            # 16384 cubed). The loop is not issue-bound, so paying more
            # instructions is not what it costs.
            for iv, state in range(0, n_iter - 1, 1, init=init_state):
                s_accs = list(state[:n_acc])
                s_accs = _one_k_tile(pA_g, pB_g, s_accs, iv % 2 * c_lds_buf_stride,
                                     (1 - iv % 2) * c_lds_buf_stride,
                                     _k_tile(k_base, iv + 1, rot, n_iter))
                results = yield list(s_accs)

            return _compute_k_tile(list(results[:n_acc]),
                                   ((n_iter - 1) % 2) * c_lds_buf_stride)

        if const_expr(sk_wgs and not sk_no_split):
            # Offsets go through make_int_tuple as Int32, the way the LDS
            # helpers do: tid is index-typed and the slot is not, and handing
            # the mix straight to pointer arithmetic cannot be typed.
            ws_base = fx.get_iter(arg_ws)
            flag_base = fx.get_iter(arg_flag)
            tid32 = fx.Int32(tid)

            def _ws_slot(slot, g):
                """The 8 floats of accumulator group ``g`` for this thread."""
                off = fx.Int32(slot) * (THREADS_PER_BLOCK * acc_size) + (
                    fx.Int32(g * THREADS_PER_BLOCK) + tid32
                ) * 8
                ptr = fx.add_offset(ws_base, fx.make_int_tuple(off))
                return fx.make_view(fx.recast_iter(fx.Float32, ptr), fx.make_layout(8, 1))

            def _flag_ptr(slot):
                return fx.add_offset(flag_base, fx.make_int_tuple(fx.Int32(slot))).llvm_ptr

            def _flag_load(slot):
                return _llvm.LoadOp(
                    T.i32, _flag_ptr(slot), alignment=4,
                    ordering=_llvm.AtomicOrdering.acquire, syncscope="agent",
                ).result

            def _flag_store(slot, value, ordering):
                _llvm.StoreOp(
                    as_ir_value(fx.Int32(value)), _flag_ptr(slot), alignment=4,
                    ordering=ordering, syncscope="agent",
                )

        # ============================================================
        # Store results to GMEM through the tiled copy
        # ============================================================
        def _store_C(accs, bid_m, bid_n, seed_slot, merge_slot=None):
            # The gfx11 v8f32 accumulator (lane L holds D[2*si + L/16][L%16]) and
            # the wave banding are both encoded in the tiled_mma, so the row
            # arithmetic that used to live here is gone. What remains is the
            # value transform, which no copy atom can express.
            #
            # frag_C flattens as si + 8*(rm + reg_m*rn), so each run of 8 elements
            # is exactly one atom's accumulator, and one Philox draw still covers
            # one run.
            tC = fx.flat_divide(fx.rocdl.make_buffer_tensor(arg_c), fx.make_tile(BLOCK_M, BLOCK_N))[
                None, None, bid_m, bid_n
            ]
            frag_C = thr_mma.make_fragment_C(tC)
            pC_g = thr_r2g_C.partition_S(tC)
            if const_expr(out_elem_cls is fx.Float32):
                frag_C_out = frag_C
            else:
                frag_C_out = fx.make_fragment_like(frag_C, out_elem_cls.ir_type)
            frag_C_retile = thr_r2g_C.retile(frag_C_out)

            # Folding a Stream-K partial in belongs here rather than at the call
            # site. Done there it straddles a runtime branch, so the incoming
            # and the summed accumulators are both live across the region --
            # 2 x 16 v8f32 on a kernel already sitting at 256 VGPRs, which cost
            # 284 spilled registers and 0.83x. Here each group's sum kills the
            # value it came from, so only one set is ever live.
            if const_expr(merge_slot is not None):
                accs = [accs[g] + _ws_slot(merge_slot, g).load() for g in range_constexpr(n_acc)]

            ordered_accs = [accs[rm * reg_n + rn] for rn in range_constexpr(reg_n) for rm in range_constexpr(reg_m)]
            if const_expr(rounding == "rs"):
                # The 4 random words cover all 8 values, each taking a distinct
                # 16-bit slice (low/high of a word), so the f32 -> bf16 store is
                # unbiased in expectation without a per-element draw. Keying on
                # the thread's slot rather than the output coordinate keeps the
                # draw independent of where the tiled copy lands the fragment.
                out_elems = []
                for g, acc in enumerate(ordered_accs):
                    base_off = (seed_slot * THREADS_PER_BLOCK + tid) * acc_size + 8 * g
                    words = fx.random.randint4x(fx.Uint32(sr_seed), fx.Uint32(base_off))
                    for si in range_constexpr(8):
                        word = words[si // 2]
                        rbits = word if si % 2 == 0 else (word >> fx.Uint32(16))
                        out_elems.append(cvt_sr_f32_to_bf16(acc[si], rbits))
            elif const_expr(out_elem_cls is fx.Float32):
                out_elems = [acc[si] for acc in ordered_accs for si in range_constexpr(8)]
            else:
                out_elems = [acc[si].to(out_elem_cls) for acc in ordered_accs for si in range_constexpr(8)]

            frag_C_out.store(
                vector.from_elements(T.vec(acc_size, out_elem_cls.ir_type), [as_ir_value(e) for e in out_elems])
            )
            fx.copy(copy_out, frag_C_retile, pC_g)

        if const_expr(not sk_wgs):
            bid_m, bid_n = _swizzle_tile_id(pid, grid_n, group_width)
            pA_g, pB_g = _tile_operands(bid_m, bid_n)
            accs = _accumulate(pA_g, pB_g, 0, num_k_tiles)
            _store_C(accs, bid_m, bid_n, pid)
        else:
            # ── Stream-K ────────────────────────────────────────────
            # Workgroup w owns units [w*U/W, (w+1)*U/W) of the tiles x k-tiles
            # space. Every range is at least num_k_tiles long, so a tile has at
            # most two contributors: the workgroup holding its k=0 owns it, and
            # the next workgroup along may hold a suffix. That caps this
            # workgroup at one partial written (its first tile, if it starts
            # mid-tile) and one partial read (its last, if it ends mid-tile).
            pid32 = fx.Int32(pid)
            if const_expr(sk_no_split):
                # Whole tiles only, so the range can be taken in tile space and
                # the unit space never has to be formed. It also must not be:
                # pid * num_tiles * num_k_tiles passes 2^31 at 12288 square,
                # which wrapped to a negative tile id and faulted the GPU.
                t_first = pid32 * num_tiles // sk_wgs
                t_last = (pid32 + 1) * num_tiles // sk_wgs - 1
            else:
                unit_begin = pid32 * sk_units // sk_wgs
                unit_end = (pid32 + 1) * sk_units // sk_wgs
                t_first = unit_begin // num_k_tiles
                t_last = (unit_end - 1) // num_k_tiles
                k_lo_first = unit_begin - t_first * num_k_tiles
                k_hi_last = unit_end - t_last * num_k_tiles

            for t, _carry in range(t_first, t_last + 1, 1, init=[fx.Int32(0)]):
                t32 = fx.Int32(t)
                bid_m, bid_n = _swizzle_tile_id(t32, grid_n, group_width)
                pA_g, pB_g = _tile_operands(bid_m, bid_n)
                if const_expr(sk_no_split):
                    # Whole tiles only, so the trip count and the closing buffer
                    # parity are Python ints again and the pipeline comes out
                    # exactly as it does on the plain path.
                    k_lo, n_iter = fx.Int32(0), num_k_tiles
                else:
                    k_lo = (t32 == t_first).select(k_lo_first, fx.Int32(0))
                    k_hi = (t32 == t_last).select(k_hi_last, fx.Int32(num_k_tiles))
                    n_iter = k_hi - k_lo
                if const_expr(sk_rot_step):
                    rot = pid32 * sk_rot_step % fx.Int32(n_iter)
                else:
                    rot = None
                accs = _accumulate(pA_g, pB_g, k_lo, n_iter, rot)

                if const_expr(sk_no_split):
                    _store_C(accs, bid_m, bid_n, t32)
                elif k_lo > fx.Int32(0):
                    # Someone else owns this tile's k=0. Park the piece in our
                    # own slot -- thread tid writes what thread tid of the owner
                    # will want, so the fragment never has to be reshaped -- and
                    # release. vmcnt drains this thread's stores, the barrier
                    # covers the rest of the block, and only then is the flag
                    # allowed to become visible.
                    #
                    # One thread raises it, because the flag is also the thing
                    # that has to come back down: with all 128 raising it, a
                    # straggler wave can re-raise it after the consumer has
                    # already lowered it, and the launch ends with it stuck up.
                    for g in range_constexpr(n_acc):
                        _ws_slot(pid32, g).store(accs[g])
                    rocdl.s_waitcnt(vmcnt=0)
                    _barrier()
                    if tid32 == fx.Int32(0):
                        _flag_store(pid32, 1, _llvm.AtomicOrdering.release)
                else:
                    # We own this tile's k=0. If we do not also own its tail,
                    # the producer is the next workgroup, which computed this as
                    # its *first* tile, so the flag is nearly always already up
                    # by the time we reach the end of our own range.
                    #
                    # The epilogue then folds that partial in unconditionally,
                    # reading a slot that is the producer's or, when there is
                    # nothing to fold, a spare slot left at zero. Branching to
                    # two epilogues instead costs more than the wasted adds: it
                    # doubles ~1400 lines of the tile loop's body and spills 80
                    # registers, and the wasted reads all hit the one zero slot,
                    # so they stay in L2.
                    nxt = pid32 + fx.Int32(1)
                    need_merge = k_hi < fx.Int32(num_k_tiles)
                    if need_merge:
                        # One thread waits and then lowers the flag again for the
                        # next launch, and the barrier is what lets the other
                        # waves past. Waiting in all 128 instead is what hung:
                        # whichever wave got there first lowered the flag out
                        # from under the ones still reading it, and those spun
                        # for a producer that had long since finished. The
                        # acquire and its cache invalidate are also what make
                        # the partial visible, and they only have to happen once
                        # per workgroup because a workgroup is one CU's worth of
                        # waves sharing one L0.
                        if tid32 == fx.Int32(0):
                            flag_val = _flag_load(nxt)
                            while flag_val == fx.Int32(0):
                                flag_val = _flag_load(nxt)
                            _flag_store(nxt, 0, _llvm.AtomicOrdering.monotonic)
                        _barrier()
                    _store_C(accs, bid_m, bid_n, t32,
                             merge_slot=need_merge.select(nxt, fx.Int32(sk_zero_slot)))
                _ = yield [fx.Int32(0)]

    @flyc.jit
    def launch_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_bt: fx.Tensor,
        arg_ws: fx.Tensor,
        arg_flag: fx.Tensor,
        stream: fx.Stream,
        sr_seed: fx.Int32 = 0,
    ):
        # 16x16x16 v16 WMMA atom (gfx11.wmma) over a waves_m x waves_n wave grid.
        # The permutation spans the whole block tile and remaps the natural
        # (atom, wave, repeat) coordinate so wave wm keeps the contiguous band
        # [wm*reg_m*16, +reg_m*16); the default stamping interleaves the repeats
        # instead, which measured 62% slower at 3072x3072x1024.
        mma_atom = fx.make_mma_atom(fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, elem_dtype, fx.Float32))
        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((waves_m, waves_n, 1), (waves_n, 1, 0)),
            permutation=(
                fx.make_layout((WMMA_M, waves_m, reg_m), (1, WMMA_M * reg_m, WMMA_M)),
                fx.make_layout((WMMA_N, waves_n, reg_n), (1, WMMA_N * reg_n, WMMA_N)),
                WMMA_K,
            ),
        )
        # G2S tiled copy: thread (tk, tm) moves one 128-bit contiguous chunk of
        # row tm, repeated over M to cover the block tile.
        tiled_copy_g2s = fx.make_tiled_copy(
            fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype),
            fx.make_layout(
                ((THRS_K, THRS_M), (1, LOAD_VEC)),
                ((THRS_M * LOAD_VEC, 1), (1, THRS_M)),
            ),
            fx.make_tile(THRS_M, BLOCK_K),
        )

        arg_a_2d = fx.make_view(fx.get_iter(arg_a), fx.make_layout((M, K), (ld_a, 1)))
        arg_bt_2d = fx.make_view(fx.get_iter(arg_bt), fx.make_layout((N, K), (ld_b, 1)))
        arg_c_2d = fx.make_view(fx.get_iter(arg_c), fx.make_layout((M, N), (ld_c, 1)))

        ws_1d = fx.make_view(fx.get_iter(arg_ws), fx.make_layout(max(sk_ws_floats, 1), 1))
        flag_1d = fx.make_view(fx.get_iter(arg_flag), fx.make_layout(max(sk_flags, 1), 1))

        c1 = 1
        total_blocks = sk_wgs if sk_wgs else grid_m * grid_n
        bk = THREADS_PER_BLOCK

        launcher = wmma_gemm_kernel(
            arg_c_2d, arg_a_2d, arg_bt_2d, ws_1d, flag_1d, tiled_mma, tiled_copy_g2s, sr_seed
        )
        launcher.launch(
            grid=(total_blocks, c1, c1),
            block=(bk, c1, c1),
            stream=stream,
        )

    if not sk_ws_floats:
        # Nothing to thread through, either because this is the plain path or
        # because no range splits a tile, but the kernel signature carries the
        # two arguments either way, so hand it something valid and untouched.
        def launch(arg_c, arg_a, arg_bt, stream, sr_seed=0):
            return launch_gemm(arg_c, arg_a, arg_bt, arg_c, arg_c, stream, sr_seed)

        return launch, BLOCK_M, BLOCK_N, BLOCK_K

    # Stream-K's scratch is sized by the persistent grid, not by the problem, so
    # one allocation serves every call this module makes. The flags have to
    # start at zero and every launch leaves them that way again, and so does the
    # spare slot on the end of the workspace, which nothing ever writes.
    state = {}

    def launch(arg_c, arg_a, arg_bt, stream, sr_seed=0):
        if not state:
            import torch

            dev = arg_c.device
            state["ws"] = torch.zeros(sk_ws_floats, device=dev, dtype=torch.float32)
            state["flag"] = torch.zeros(sk_wgs, device=dev, dtype=torch.int32)
        return launch_gemm(arg_c, arg_a, arg_bt, state["ws"], state["flag"], stream, sr_seed)

    return launch, BLOCK_M, BLOCK_N, BLOCK_K
