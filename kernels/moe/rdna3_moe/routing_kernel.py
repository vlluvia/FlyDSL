# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""o1: the routing buffers in one wave32 workgroup.

``routing.py`` builds the same buffers on the host, and measured 103 us at 32
tokens -- more than the three GEMMs of a decode layer put together. This is that
work as one kernel launch, which costs 6-13 us over every shape measured.

Moving the work was only half of it (o1). The other half was not waiting for the
answer (o2): a caller that reads the tile count back to size its grid pays 29 us
of round trip for it, three times the kernel. So the count stays on the device,
the grid is ``max_blocks_for``'s bound, and the GEMM skips the tiles past the end
itself -- see ``build_routing_device`` and ``grouped_gemm``'s ``bounded_blocks``.
Routing then costs the decode layer 11 us, against 115 with the read-back and the
per-step allocations.

Why one workgroup
-----------------
The buffer is a counting sort: histogram, prefix sum, scatter. The prefix sum is
over experts, so it is inherently a single cooperative step, and the whole thing
touches a few thousand ints -- there is nothing here to spread across CUs. What
matters is that it is *one launch* instead of fifteen.

The four phases
---------------
A thread owns routing rows, strided: ``tid``, ``tid + 256``, and so on. It reads
each of its rows exactly twice, once per row walk.

    histogram   one LDS atomic increment per row, into a counter per expert
    scan        over the per-expert tile counts, giving each expert the slot its
                rows start at -- written back over the counter, so the same
                array becomes the scatter's cursor
    scatter     one LDS atomic increment per row, and the slot it returns is
                where that row's packed id goes
    tail        the padding: each expert's last tile, and the tiles between the
                real count and the bound the caller allocated

The scan is over ``E`` counters and the block has 256 threads, so it sweeps the
experts in ``ceil(E / 256)`` chunks, each an inclusive Hillis-Steele scan of
``log2(256) = 8`` steps with a running carry between chunks.

What this replaced, and what it cost
------------------------------------
The first version paired a thread with an ``(expert, slice)`` and had it scan its
slice of the rows looking for its expert. That made the sort stable for free --
slices are contiguous and prefixed in order, so each expert's rows came out in
row order, bit-identical to the host builder's argsort -- but it examined every
``(expert, row)`` pair, so the walk was ``E * rows / 256`` global loads per
thread. Fine at eight experts. At Qwen4Exp's 513 and a 760-token prefill it is
16.7 M pair checks, and the kernel measured **1649 us** against a layer that is
otherwise 1.1 ms.

Reading each row once instead needs a counter that any row can reach, which
means an LDS atomic, and that is what gives the stability up: whichever thread
wins the atomic takes the slot. Row order inside an expert does not change any
output value -- each row is its own dot product, and the packed id carries the
``(token, slot)`` it scatters back to -- so the layer's output is unchanged and
still deterministic. What changed is that ``sorted_ids`` is only equal to the
host builder's up to a permutation within each expert; see
``test_device_routing_matches_the_host_builder``, which compares them canonically.

Measured per launch at E=513, topk=11, tile_m=16, against the pair walk::

    tokens     pair walk    this
         1         16 us     6 us
        16         42 us     6 us
       760       1649 us    13 us

Nothing here is wave64: the only cross-thread communication is LDS plus barriers,
and the scan's step count is a constexpr.
"""

import functools
from typing import NamedTuple

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import gpu, ptrtoint, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from kernels.common.tensor_shim import _run_compiled
from kernels.moe.rdna3_moe.routing import SLOT_SHIFT, TOKEN_MASK, Routing

BLOCK = 256
# The histogram is one LDS counter per expert, so this is an LDS budget rather
# than a code-size one: E ints of counters plus the block's scan scratch.
MAX_EXPERTS = 4 * BLOCK


def _scan_offsets(width: int) -> tuple[int, ...]:
    """The Hillis-Steele step offsets covering ``width`` elements: 1, 2, 4, ...

    ``ceil(log2(width))`` of them, which is what a scan over a non-power-of-two
    width needs -- the last step reaches past the array and its lanes take the
    identity.
    """
    offs, off = [], 1
    while off < int(width):
        offs.append(off)
        off *= 2
    return tuple(offs)


def _lds_ld(ptr, idx):
    return fx.ptr_load(ptr + fx.Int64(idx))


def _lds_st(ptr, val, idx):
    fx.ptr_store(val, ptr + fx.Int64(idx))


def _lds_atomic_add(ptr, idx, val):
    """``old = atomicrmw add lds[idx], val`` at workgroup scope -- ``ds_add_rtn_u32``.

    The returned old value is the whole point: in the histogram it is discarded,
    but in the scatter it *is* the slot, so counting and claiming a place are the
    same instruction.
    """
    llvm_ptr = fx.to_llvm_ptr(fx.add_offset(ptr, fx.make_int_tuple(idx)))
    raw = val.ir_value() if hasattr(val, "ir_value") else val
    return fx.Int32(
        _llvm.AtomicRMWOp(
            _llvm.AtomicBinOp.add,
            llvm_ptr,
            raw,
            _llvm.AtomicOrdering.monotonic,
            syncscope="workgroup",
            alignment=4,
        ).result
    )


def create_routing_module(*, experts: int, topk: int, tile_m: int):
    """Build the routing kernel for one (experts, topk, tile_m).

    Returns ``launch(topk_ids, sorted_ids, expert_ids, aux, tokens, max_blocks,
    stream)``. The caller owns the buffers; see ``build_routing_device``.
    """
    E, TOPK, TILE = int(experts), int(topk), int(tile_m)
    if not str(get_rocm_arch() or "").startswith("gfx11"):
        raise RuntimeError(f"rdna3_moe requires gfx11*, got {get_rocm_arch()!r}")
    if not 1 <= E <= MAX_EXPERTS:
        raise ValueError(f"experts must be in [1, {MAX_EXPERTS}], got {E}")
    if not 1 <= TOPK <= 0xFF:
        raise ValueError(f"the packed id has 8 bits of slot, so topk must be in [1, 255], got {TOPK}")
    if TILE < 1:
        raise ValueError(f"tile_m must be positive, got {TILE}")

    # Expert chunks the scan sweeps, one block-wide scan apiece.
    CHUNKS = -(-E // BLOCK)

    @fx.struct
    class _Shared:
        # One entry per expert, and it is three things in turn: the histogram's
        # row count, then the slot that expert's rows start at, then the cursor
        # the scatter bumps as it fills them.
        cnt: fx.Array[fx.Int32, E, 16]
        # One entry per thread, holding the chunk of tile counts being scanned.
        scan: fx.Array[fx.Int32, BLOCK, 16]

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def routing_kernel(
        TopkIds: fx.Pointer,
        SortedIds: fx.Pointer,
        ExpertIds: fx.Pointer,
        Aux: fx.Pointer,
        i32_tokens: fx.Int32,
        i32_max_blocks: fx.Int32,
    ):
        i32pt = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4)
        ids = fx.inttoptr(i32pt, fx.Int64(ptrtoint(TopkIds)))
        out = fx.inttoptr(i32pt, fx.Int64(ptrtoint(SortedIds)))
        eout = fx.inttoptr(i32pt, fx.Int64(ptrtoint(ExpertIds)))
        aux = fx.inttoptr(i32pt, fx.Int64(ptrtoint(Aux)))

        lds = fx.SharedAllocator().allocate(_Shared).peek()
        cnt_mr, scan_mr = lds.cnt.ptr, lds.scan.ptr

        tid = gpu.thread_idx.x
        c0, c1 = fx.Int32(0), fx.Int32(1)
        c_tile = fx.Int32(TILE)
        c_e = fx.Int32(E)

        def _tiles(count):
            """``ceil(count / TILE)`` -- an expert with no rows owns no tile.

            The floor used to be one tile per expert, on the grounds that a tile
            of nothing but sentinels gathers zeros and stores nothing. It does,
            but it still streams that expert's whole weight slab out of DRAM
            first, and at Qwen4Exp's 513 experts a one-token decode is 502 empty
            tiles against 11 real ones -- 601 MiB of weight read to do 13 MiB of
            work. See ``max_blocks_for`` for what the bound becomes.
            """
            return (count + c_tile - c1) // c_tile

        rows = i32_tokens * fx.Int32(TOPK)
        sentinel = (fx.Int32(TOPK) << fx.Int32(SLOT_SHIFT)) | i32_tokens
        # Every thread walks the same number of steps -- the trip count is
        # uniform and the tail is masked, which keeps the loop off the divergent
        # path.
        steps = ArithValue((rows + fx.Int32(BLOCK - 1)) // fx.Int32(BLOCK)).index_cast(T.index)

        def _my_row(_i):
            """This thread's row at step ``_i``, and whether it is one at all."""
            row = tid + fx.Int32(_i) * fx.Int32(BLOCK)
            return row, row < rows

        def _expert_of(row, inb):
            """The expert a row named, and whether this rank has it.

            A row naming an expert this rank does not hold is dropped rather than
            counted -- the same thing the pair walk did by never matching it, and
            the host builder raises on it before it gets here.
            """
            eid = _lds_ld(ids, inb.select(row, c0))
            return eid, inb & (eid >= c0) & (eid < c_e)

        # ── Histogram: one atomic per row, into a counter per expert ─────────
        for _c in range_constexpr(CHUNKS):
            e = fx.Int32(_c * BLOCK) + tid
            if e < c_e:
                _lds_st(cnt_mr, c0, e)
        gpu.barrier()

        for _i in range(fx.Index(0), steps, fx.Index(1)):
            row, inb = _my_row(_i)
            eid, keep = _expert_of(row, inb)
            if keep:
                _lds_atomic_add(cnt_mr, eid, c1)
        gpu.barrier()

        # ── Scan the tile counts, and name the tiles ─────────────────────────
        # Experts are numbered in order, so a chunk's tiles all follow the
        # previous chunk's; ``nblocks`` is the running total, identical in every
        # thread, and after the last chunk it is what the caller reads out of
        # ``aux``.
        nblocks = c0
        for _c in range_constexpr(CHUNKS):
            e = fx.Int32(_c * BLOCK) + tid
            inr = e < c_e
            count = inr.select(_lds_ld(cnt_mr, inr.select(e, c0)), c0)
            mine = _tiles(count)
            _lds_st(scan_mr, mine, tid)
            gpu.barrier()

            # Inclusive Hillis-Steele over the block: eight steps of one LDS read
            # and one write, against the EC serial dependent loads a per-thread
            # walk of the counters would take.
            cur = mine
            for off in _scan_offsets(BLOCK):
                src = tid - fx.Int32(off)
                # A read per step whatever the index, so the trip count stays
                # uniform; the out-of-range lanes take the identity.
                prev = _lds_ld(scan_mr, (src >= c0).select(src, c0))
                # Split the read from the write: the next step's read is of this
                # step's values, and there is one array.
                gpu.barrier()
                cur = (src >= c0).select(cur + prev, cur)
                _lds_st(scan_mr, cur, tid)
                gpu.barrier()

            # ``cur`` is the inclusive prefix, so the exclusive one -- the tile
            # this expert's rows start at -- is it less this expert's own tiles.
            first = nblocks + cur - mine
            row0 = first * c_tile
            if inr:
                # The counter becomes the cursor the scatter bumps.
                _lds_st(cnt_mr, row0, e)

            # An expert with no rows has ``mine == 0``, and so does a thread past
            # the end of the last chunk: both loops fall through without a guard.
            named = ArithValue(mine).index_cast(T.index)
            for _b in range(fx.Index(0), named, fx.Index(1)):
                _lds_st(eout, e, first + fx.Int32(_b))
            # The pad slots of this expert's last tile. Disjoint from every slot
            # the scatter will claim, which is why the two need no ordering.
            pad = ArithValue(mine * c_tile - count).index_cast(T.index)
            for _p in range(fx.Index(0), pad, fx.Index(1)):
                _lds_st(out, sentinel, row0 + count + fx.Int32(_p))

            nblocks = nblocks + _lds_ld(scan_mr, fx.Int32(BLOCK - 1))
            # The next chunk overwrites the scan array.
            gpu.barrier()

        # ── Scatter: the slot the cursor hands back is where the row goes ────
        for _i in range(fx.Index(0), steps, fx.Index(1)):
            row, inb = _my_row(_i)
            eid, keep = _expert_of(row, inb)
            safe = inb.select(row, c0)
            packed = ((safe % fx.Int32(TOPK)) << fx.Int32(SLOT_SHIFT)) | (safe // fx.Int32(TOPK))
            if keep:
                _lds_st(out, packed, _lds_atomic_add(cnt_mr, eid, c1))

        if tid == c0:
            _lds_st(aux, nblocks, c0)

        # ── Define the tail the caller allocated but the routing did not fill ─
        # ``max_blocks`` is the host's sync-free upper bound. o1 reads the exact
        # count back and slices, so this only keeps the buffer well defined; o2
        # is what makes a launch over the whole tail correct.
        tail_b = ArithValue((i32_max_blocks - nblocks - tid + fx.Int32(BLOCK - 1)) // fx.Int32(BLOCK)).index_cast(
            T.index
        )
        for _t in range(fx.Index(0), tail_b, fx.Index(1)):
            blk = nblocks + tid + fx.Int32(_t) * fx.Int32(BLOCK)
            if blk < i32_max_blocks:
                _lds_st(eout, c0, blk)
        tail_r = ArithValue(
            ((i32_max_blocks - nblocks) * c_tile - tid + fx.Int32(BLOCK - 1)) // fx.Int32(BLOCK)
        ).index_cast(T.index)
        for _t in range(fx.Index(0), tail_r, fx.Index(1)):
            slot = nblocks * c_tile + tid + fx.Int32(_t) * fx.Int32(BLOCK)
            if slot < i32_max_blocks * c_tile:
                _lds_st(out, sentinel, slot)

    @flyc.jit
    def launch_routing(
        TopkIds: fx.Pointer,
        SortedIds: fx.Pointer,
        ExpertIds: fx.Pointer,
        Aux: fx.Pointer,
        i32_tokens: fx.Int32,
        i32_max_blocks: fx.Int32,
        stream: fx.Stream,
    ):
        routing_kernel(TopkIds, SortedIds, ExpertIds, Aux, i32_tokens, i32_max_blocks).launch(
            grid=(1, 1, 1), block=(BLOCK, 1, 1), stream=stream
        )

    def launch(topk_ids, sorted_ids, expert_ids, aux, tokens, max_blocks, stream):
        def ptr(t):
            return flyc.from_c_void_p(fx.Uint8, t.data_ptr())

        return _run_compiled(
            launch_routing,
            ptr(topk_ids),
            ptr(sorted_ids),
            ptr(expert_ids),
            ptr(aux),
            fx.Int32(int(tokens)),
            fx.Int32(int(max_blocks)),
            stream,
        )

    launch.experts = E
    launch.topk = TOPK
    launch.tile_m = TILE
    return launch


@functools.lru_cache(maxsize=256)
def compile_routing(*, experts: int, topk: int, tile_m: int):
    """Build (and cache) the routing kernel for one shape."""
    return create_routing_module(experts=experts, topk=topk, tile_m=tile_m)


def max_blocks_for(*, tokens: int, topk: int, experts: int, tile_m: int) -> int:
    """An upper bound on the tile count, known without looking at the routing.

    An expert with rows contributes ``ceil(c/t) = 1 + (c-1)//t`` tiles and one
    with none contributes nothing, so over ``n`` non-empty experts the sum is at
    most ``n + rows//t`` -- and ``n`` is at most ``min(E, rows)``, since a
    non-empty expert costs a row. The point of the bound is that the host can
    size the buffers, and later pick a grid, without waiting for the gating to
    finish.

    The ``min`` is what makes a decode cheap. At Qwen4Exp's 513 experts a
    one-token step routes 11 rows, so the bound is 11 tiles rather than the 524
    it would be if every expert had to be allowed one.
    """
    rows = int(tokens) * int(topk)
    # A routing with no rows still wants somewhere to point; one tile of padding
    # is the smallest well-defined buffer.
    return max(1, min(int(experts), rows) + rows // int(tile_m))


class _Workspace(NamedTuple):
    sorted_ids: torch.Tensor
    expert_ids: torch.Tensor
    aux: torch.Tensor


_WORKSPACES: dict[tuple, _Workspace] = {}


def _workspace(*, max_blocks: int, tile_m: int, device) -> _Workspace:
    """The three buffers one routing needs, kept per shape and reused.

    Three ``torch.empty`` calls measured 10-15 us together, which at decode is
    the same order as the kernel that fills them. They are reused rather than
    cached-and-freed because the point is to have nothing to do per step.

    Reuse is safe against the GEMMs that read them for the reason ordering on one
    stream is safe: the next step's routing kernel is queued behind this step's
    GEMMs. A caller who wants to keep a routing alive across steps, or run two
    layers on two streams, asks for its own with ``reuse=False``.
    """
    key = (int(max_blocks), int(tile_m), str(device))
    ws = _WORKSPACES.get(key)
    if ws is None:
        ws = _Workspace(
            sorted_ids=torch.empty(max_blocks * tile_m, dtype=torch.int32, device=device),
            expert_ids=torch.empty(max_blocks, dtype=torch.int32, device=device),
            aux=torch.empty(1, dtype=torch.int32, device=device),
        )
        _WORKSPACES[key] = ws
    return ws


def build_routing_device(
    topk_ids: torch.Tensor,
    *,
    experts: int,
    tile_m: int,
    exact: bool = False,
    reuse: bool = True,
    stream=None,
) -> Routing:
    """The device build of ``routing.build_routing``, with the same contract.

    ``exact=True`` reads the tile count back, so the result is what the host
    builder returns: buffers sliced to the tiles that exist, and a
    ``num_blocks`` a GEMM can use as its grid directly. That read is a 29 us
    round trip, against a kernel that takes 10 -- so the default is ``False``,
    which returns the *bound* as ``num_blocks`` and leaves the count on the
    device in ``num_blocks_device``. The buffers then carry padding past the last
    real tile, and it is the GEMM (built with ``bounded_blocks=True``) that skips
    them. Nothing about the layer's output differs; what differs is that the host
    never waits.
    """
    if topk_ids.dim() != 2:
        raise ValueError(f"topk_ids must be [tokens, topk], got {tuple(topk_ids.shape)}")
    tokens, topk = topk_ids.shape
    if tokens > TOKEN_MASK:
        raise ValueError(f"the packed id has 24 bits of token, so tokens must be < {TOKEN_MASK}, got {tokens}")

    launch = compile_routing(experts=int(experts), topk=int(topk), tile_m=int(tile_m))
    max_blocks = max_blocks_for(tokens=tokens, topk=topk, experts=experts, tile_m=tile_m)

    dev = topk_ids.device
    if reuse:
        ws = _workspace(max_blocks=max_blocks, tile_m=int(tile_m), device=dev)
    else:
        ws = _Workspace(
            sorted_ids=torch.empty(max_blocks * tile_m, dtype=torch.int32, device=dev),
            expert_ids=torch.empty(max_blocks, dtype=torch.int32, device=dev),
            aux=torch.empty(1, dtype=torch.int32, device=dev),
        )

    st = stream if stream is not None else torch.cuda.current_stream()
    launch(topk_ids.contiguous(), ws.sorted_ids, ws.expert_ids, ws.aux, int(tokens), max_blocks, st)

    if not exact:
        return Routing(
            sorted_ids=ws.sorted_ids,
            expert_ids=ws.expert_ids,
            num_blocks=max_blocks,
            tile_m=int(tile_m),
            tokens=int(tokens),
            topk=int(topk),
            exact=False,
            num_blocks_device=ws.aux,
        )
    # Order the read after the launch's own stream rather than trusting the
    # current one.
    st.synchronize()
    num_blocks = int(ws.aux[0])
    return Routing(
        sorted_ids=ws.sorted_ids[: num_blocks * tile_m],
        expert_ids=ws.expert_ids[:num_blocks],
        num_blocks=num_blocks,
        tile_m=int(tile_m),
        tokens=int(tokens),
        topk=int(topk),
        exact=True,
        num_blocks_device=ws.aux,
    )
