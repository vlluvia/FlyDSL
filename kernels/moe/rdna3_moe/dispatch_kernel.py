# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""o4: the dispatch plan in one wave32 workgroup.

``ep.py``'s first cut built the plan out of torch calls: divide the expert ids by
the experts per rank, stable-sort that, take the bucket edges for the counts,
gather the payload, build the metadata, check the range. Fifteen or so launches,
and at decode every one of them works on 64 elements -- so the cost is not the
work, it is the calling. Measured at 32 tokens a rank, dispatch took 531 us
against a transport floor of about 165. This is that plan as one kernel, which
brings dispatch to 353 us at decode and 503 from 588 at 1024 tokens a rank -- the
rest being the three collectives and the host driving them, not the plan.

It is the same counting sort as ``routing_kernel.py``, with the destination rank
as the key instead of the expert, and without the tile padding -- an all-to-all
wants each destination's rows packed end to end, not rounded up to a tile. A
thread is a ``(rank, slice)`` pair, it counts how many of its slice's routed
slots are bound for its rank, and then writes exactly those. Slices own
contiguous ranges of rows, so the result is ordered by row within a destination:
the same stable order ``torch.sort`` produced, which is what lets combine undo
the permutation by index alone.

What the kernel emits, all in destination order:

    send_order  [rows] int64   which routed slot (token*topk + slot) this row was
    src_row     [rows] int32   send_order // topk, the token to gather from x
    meta        [rows, 2]      the local expert id, and the routing weight's bits
    counts      [2W+1] int32   rows to each rank; then a slot for the reply, and
                               the count of out-of-range expert ids

``src_row`` is separate from ``send_order`` only to spare the host a divide it
would otherwise launch a kernel for, and the two differ in width because of what
reads them: ``index_select`` takes an int32 index, ``index_copy_`` insists on
int64. So ``send_order`` is written as two dwords, the value and a zero -- a row
index is never negative, so the high half is always that.

The payload gather itself stays in torch: ``index_select`` moved 8.4 MB in
15.6 us, which is bandwidth, not overhead, and there is nothing to win by
rewriting it.

The counts buffer is laid out for one readback. The send half is what the kernel
counted, the reply half is where the counts all-to-all lands, and the last slot
carries the range check -- so the host reads its split sizes and validates its
input in a single copy, instead of a ``cat`` and three separate syncs.

Out-of-range ids are clamped into the last rank rather than dropped, so the row
counts still add up and the exchange stays consistent; the host raises after it
reads the flag.
"""

import functools
from typing import NamedTuple

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, ptrtoint, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from kernels.common.tensor_shim import _run_compiled

BLOCK = 256
UNROLL = 4  # rows read per step of the two row walks


def _ld(ptr, idx):
    return fx.ptr_load(ptr + fx.Int64(idx))


def _st(ptr, val, idx):
    fx.ptr_store(val, ptr + fx.Int64(idx))


def create_dispatch_plan_module(*, world_size: int, experts_per_rank: int, topk: int, capacity: int | None = None):
    """Build the plan kernel for one (world_size, experts_per_rank, topk).

    Returns ``launch(topk_ids, topk_weights, send_order, src_row, meta, counts,
    rows, stream)``. The caller owns the buffers; see ``build_dispatch_plan``.

    ``capacity`` decides where a destination's rows go. Without it they are
    packed, each destination starting where the last one ended, so the block a
    collective sends is exactly as long as the routing made it -- and its length
    is therefore something the host has to learn before it can call
    ``all_to_all_single``, which costs a collective and a sync.

    With it, destination ``d``'s rows start at ``d * capacity`` whatever the
    routing did. Every block is the same length, so the split sizes are constants
    and neither the counts exchange nor the readback happens; the exchange stops
    needing a host at all, which is what lets a graph capture swallow the whole
    decode step. The slots a destination does not fill are given a local expert
    id of -1, and the routing kernel drops those for free -- its histogram and
    its scatter both test ``id == e`` for ``e`` in ``[0, experts)``, so a row
    outside that range joins no tile and reaches no GEMM. The padding costs
    bandwidth on the wire and nothing after it.

    A destination that wants more than ``capacity`` rows loses the excess, and
    the count lands in ``counts[2*world_size + 1]``. Reading that is a sync, so
    it is the caller's to check when it can afford to; ``build_dispatch_plan``'s
    default capacity is chosen so it cannot happen.
    """
    W, EPR, TOPK = int(world_size), int(experts_per_rank), int(topk)
    CAP = None if capacity is None else int(capacity)
    if not str(get_rocm_arch() or "").startswith("gfx11"):
        raise RuntimeError(f"rdna3_moe requires gfx11*, got {get_rocm_arch()!r}")
    if not 1 <= W <= BLOCK:
        raise ValueError(f"world_size must be in [1, {BLOCK}], got {W}")
    if EPR < 1:
        raise ValueError(f"experts_per_rank must be positive, got {EPR}")
    if TOPK < 1:
        raise ValueError(f"topk must be positive, got {TOPK}")
    if CAP is not None and CAP < 1:
        raise ValueError(f"capacity must be positive, got {CAP}")

    SLICES = BLOCK // W
    N_EXPERTS = W * EPR

    @fx.struct
    class _Shared:
        # Per (slice, rank) count, indexed by thread so the threads past
        # W*SLICES have a slot nobody reads.
        cnt: fx.Array[fx.Int32, BLOCK, 16]
        # Per rank row count, so every thread can scan all ranks.
        tot: fx.Array[fx.Int32, W, 16]
        # Out-of-range ids, summed over the block.
        bad: fx.Array[fx.Int32, BLOCK, 16]

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def dispatch_plan_kernel(
        TopkIds: fx.Pointer,
        Weights: fx.Pointer,
        SendOrder: fx.Pointer,
        SrcRow: fx.Pointer,
        Meta: fx.Pointer,
        Counts: fx.Pointer,
        i32_rows: fx.Int32,
    ):
        i32pt = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4)
        ids = fx.inttoptr(i32pt, fx.Int64(ptrtoint(TopkIds)))
        # The weight is copied as its bits, so it never needs to be an f32 here.
        wts = fx.inttoptr(i32pt, fx.Int64(ptrtoint(Weights)))
        order = fx.inttoptr(i32pt, fx.Int64(ptrtoint(SendOrder)))
        srow = fx.inttoptr(i32pt, fx.Int64(ptrtoint(SrcRow)))
        meta = fx.inttoptr(i32pt, fx.Int64(ptrtoint(Meta)))
        counts = fx.inttoptr(i32pt, fx.Int64(ptrtoint(Counts)))

        lds = fx.SharedAllocator().allocate(_Shared).peek()
        cnt_mr, tot_mr, bad_mr = lds.cnt.ptr, lds.tot.ptr, lds.bad.ptr

        tid = gpu.thread_idx.x
        c0, c1 = fx.Int32(0), fx.Int32(1)
        c_epr, c_topk = fx.Int32(EPR), fx.Int32(TOPK)
        c_last = fx.Int32(W - 1)
        d = tid % fx.Int32(W)
        s = tid // fx.Int32(W)

        rows = i32_rows
        step = fx.Int32(SLICES * UNROLL)
        chunk = ((rows + step - c1) // step) * fx.Int32(UNROLL)
        lo = s * chunk
        walk = ArithValue(chunk // fx.Int32(UNROLL)).index_cast(T.index)

        def _read(row):
            """The UNROLL expert ids at ``row``, their ranks, and which exist."""
            inb = [row + fx.Int32(u) < rows for u in range_constexpr(UNROLL)]
            safe = [inb[u].select(row + fx.Int32(u), c0) for u in range_constexpr(UNROLL)]
            eids = [_ld(ids, safe[u]) for u in range_constexpr(UNROLL)]
            # Clamp rather than drop: a row that named a nonexistent expert still
            # has to be counted somewhere or the send and receive counts stop
            # agreeing. The host raises on the flag below.
            ranks = []
            for u in range_constexpr(UNROLL):
                r = eids[u] // c_epr
                r = (r < c0).select(c0, r)
                ranks.append((r > c_last).select(c_last, r))
            return safe, inb, eids, ranks

        # ── Histogram: how many of my rows are bound for my rank ─────────────
        mine = c0
        bad = c0
        for _i in range(fx.Index(0), walk, fx.Index(1)):
            row = lo + fx.Int32(_i) * fx.Int32(UNROLL)
            _, inb, eids, ranks = _read(row)
            for u in range_constexpr(UNROLL):
                mine = (inb[u] & (ranks[u] == d)).select(mine + c1, mine)
                # Counted once per row rather than once per (row, rank): the
                # thread that claims the row is the one that sees it.
                off = inb[u] & (ranks[u] == d) & ((eids[u] < c0) | (eids[u] >= fx.Int32(N_EXPERTS)))
                bad = off.select(bad + c1, bad)
        _st(cnt_mr, mine, tid)
        _st(bad_mr, bad, tid)
        gpu.barrier()

        # ── Scan the slices of my rank: my start, and my rank's total ─────────
        start = c0
        total = c0
        for sl in range_constexpr(SLICES):
            v = _ld(cnt_mr, fx.Int32(sl * W) + d)
            start = (fx.Int32(sl) < s).select(start + v, start)
            total = total + v
        _st(tot_mr, total, d)
        gpu.barrier()

        # ── Where my rank's rows begin ───────────────────────────────────────
        if const_expr(CAP is None):
            # Packed: scan the ranks before mine and start after them.
            first = c0
            for rr in range_constexpr(W):
                t = _ld(tot_mr, fx.Int32(rr))
                first = (fx.Int32(rr) < d).select(first + t, first)
        else:
            # Fixed: my rank's slots are mine whether I fill them or not, so
            # there is nothing to scan and nothing for the host to learn.
            first = d * fx.Int32(CAP)

        if s == c0:
            _st(counts, total, d)
        if tid == c0:
            acc = c0
            for b in range_constexpr(BLOCK):
                acc = acc + _ld(bad_mr, fx.Int32(b))
            _st(counts, acc, fx.Int32(2 * W))

        # ── Scatter my rows, in row order ────────────────────────────────────
        dst = first + start
        limit = first + fx.Int32(CAP if CAP is not None else 0)
        over = c0
        for _i in range(fx.Index(0), walk, fx.Index(1)):
            row = lo + fx.Int32(_i) * fx.Int32(UNROLL)
            safe, inb, eids, ranks = _read(row)
            wbits = [_ld(wts, safe[u]) for u in range_constexpr(UNROLL)]
            for u in range_constexpr(UNROLL):
                hit = inb[u] & (ranks[u] == d)
                # Past the capacity there is no slot to write, so the row is
                # dropped and counted. Without a capacity nothing is past it.
                if const_expr(CAP is None):
                    fits = hit
                else:
                    fits = hit & (dst < limit)
                if fits:
                    # send_order is int64, so two dwords: the row, then the zero
                    # high half.
                    _st(order, safe[u], dst * fx.Int32(2))
                    _st(order, c0, dst * fx.Int32(2) + c1)
                    _st(srow, safe[u] // c_topk, dst)
                    _st(meta, eids[u] % c_epr, dst * fx.Int32(2))
                    _st(meta, wbits[u], dst * fx.Int32(2) + c1)
                if const_expr(CAP is not None):
                    over = (hit & (dst >= limit)).select(over + c1, over)
                dst = hit.select(dst + c1, dst)

        if const_expr(CAP is not None):
            # ── Define the slots my rank did not fill ────────────────────────
            # A local expert id of -1 is what makes these free: the routing
            # kernel counts and scatters only ids in [0, experts), so a padded
            # row joins no tile. ``src_row`` still has to address something for
            # the gather, and ``send_order`` points at the scratch slot combine
            # keeps one past the real rows.
            rem = fx.Int32(CAP) - total
            rem = (rem > c0).select(rem, c0)
            pad = ArithValue((rem + fx.Int32(SLICES - 1)) // fx.Int32(SLICES)).index_cast(T.index)
            for _p in range(fx.Index(0), pad, fx.Index(1)):
                slot = total + s + fx.Int32(_p) * fx.Int32(SLICES)
                if slot < fx.Int32(CAP):
                    base = first + slot
                    _st(order, rows, base * fx.Int32(2))
                    _st(order, c0, base * fx.Int32(2) + c1)
                    _st(srow, c0, base)
                    _st(meta, fx.Int32(-1), base * fx.Int32(2))
                    _st(meta, c0, base * fx.Int32(2) + c1)
            # Overflow, summed over the block the same way the range check is.
            _st(bad_mr, over, tid)
            gpu.barrier()
            if tid == c0:
                acc = c0
                for b in range_constexpr(BLOCK):
                    acc = acc + _ld(bad_mr, fx.Int32(b))
                _st(counts, acc, fx.Int32(2 * W + 1))

    @flyc.jit
    def launch_plan(
        TopkIds: fx.Pointer,
        Weights: fx.Pointer,
        SendOrder: fx.Pointer,
        SrcRow: fx.Pointer,
        Meta: fx.Pointer,
        Counts: fx.Pointer,
        i32_rows: fx.Int32,
        stream: fx.Stream,
    ):
        dispatch_plan_kernel(TopkIds, Weights, SendOrder, SrcRow, Meta, Counts, i32_rows).launch(
            grid=(1, 1, 1), block=(BLOCK, 1, 1), stream=stream
        )

    def launch(topk_ids, topk_weights, send_order, src_row, meta, counts, rows, stream):
        def ptr(t):
            return flyc.from_c_void_p(fx.Uint8, t.data_ptr())

        return _run_compiled(
            launch_plan,
            ptr(topk_ids),
            ptr(topk_weights),
            ptr(send_order),
            ptr(src_row),
            ptr(meta),
            ptr(counts),
            fx.Int32(int(rows)),
            stream,
        )

    launch.world_size = W
    launch.experts_per_rank = EPR
    launch.topk = TOPK
    launch.slices = SLICES
    return launch


@functools.lru_cache(maxsize=256)
def compile_dispatch_plan(*, world_size: int, experts_per_rank: int, topk: int):
    """Build (and cache) the plan kernel for one shape."""
    return create_dispatch_plan_module(world_size=world_size, experts_per_rank=experts_per_rank, topk=topk)


class Plan(NamedTuple):
    """One dispatch's plan, all of it still on the device except ``counts``."""

    send_order: torch.Tensor
    src_row: torch.Tensor
    meta: torch.Tensor
    counts: torch.Tensor


class _Workspace(NamedTuple):
    send_order: torch.Tensor
    src_row: torch.Tensor
    meta: torch.Tensor
    counts: torch.Tensor


_WORKSPACES: dict[tuple, _Workspace] = {}


def _workspace(*, rows: int, world_size: int, device) -> _Workspace:
    """The plan's four buffers, kept per shape and reused.

    Same bargain as ``routing_kernel._workspace``: four ``torch.empty`` calls
    cost about as much as the kernel does at decode, and the point of the kernel
    is to leave nothing to do per step. Reuse is safe against the collectives
    that read them for the same reason -- they are stream-ordered behind the
    previous step -- and a caller who needs to hold a plan asks for its own.
    """
    key = (int(rows), int(world_size), str(device))
    ws = _WORKSPACES.get(key)
    if ws is None:
        ws = _Workspace(
            send_order=torch.empty(rows, dtype=torch.int32, device=device),
            src_row=torch.empty(rows, dtype=torch.int32, device=device),
            meta=torch.empty(rows, 2, dtype=torch.int32, device=device),
            counts=torch.empty(2 * world_size + 1, dtype=torch.int32, device=device),
        )
        _WORKSPACES[key] = ws
    return ws


def build_dispatch_plan(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    world_size: int,
    experts_per_rank: int,
    reuse: bool = True,
    stream=None,
) -> Plan:
    """Group the routed slots by destination rank, in one launch.

    ``topk_ids`` is ``[tokens, topk]`` of global expert ids and ``topk_weights``
    matches it. Nothing is read back here; the caller reads ``counts`` once, when
    it needs the split sizes.
    """
    if topk_ids.dim() != 2:
        raise ValueError(f"topk_ids must be [tokens, topk], got {tuple(topk_ids.shape)}")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(f"topk_weights must match topk_ids, got {tuple(topk_weights.shape)}")
    if topk_weights.dtype != torch.float32:
        raise ValueError(f"topk_weights must be f32, got {topk_weights.dtype}")

    tokens, topk = topk_ids.shape
    rows = int(tokens) * int(topk)
    launch = compile_dispatch_plan(
        world_size=int(world_size), experts_per_rank=int(experts_per_rank), topk=int(topk)
    )

    dev = topk_ids.device
    if reuse:
        ws = _workspace(rows=rows, world_size=int(world_size), device=dev)
    else:
        ws = _Workspace(
            send_order=torch.empty(rows, dtype=torch.int32, device=dev),
            src_row=torch.empty(rows, dtype=torch.int32, device=dev),
            meta=torch.empty(rows, 2, dtype=torch.int32, device=dev),
            counts=torch.empty(2 * int(world_size) + 1, dtype=torch.int32, device=dev),
        )

    st = stream if stream is not None else torch.cuda.current_stream()
    launch(
        topk_ids.contiguous(),
        topk_weights.contiguous(),
        ws.send_order,
        ws.src_row,
        ws.meta,
        ws.counts,
        rows,
        st,
    )
    return Plan(send_order=ws.send_order, src_row=ws.src_row, meta=ws.meta, counts=ws.counts)
