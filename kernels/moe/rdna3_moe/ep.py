# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Expert-parallel dispatch and combine over RCCL, one node.

``forward.py`` is written for one rank: ``w1``/``w2`` hold this rank's experts
and ``x`` is whatever rows arrived here. This module is what makes rows arrive.
Gating happens on the rank the token lives on, dispatch sends each routed slot to
the rank that owns its expert, that rank runs the layer, and combine sends the
results home to be summed.

The unit of exchange is a **routed slot**, not a token: a token with topk=2 sends
two rows, one per expert, even when both experts live on the same rank. That is
one copy of the activation more than the minimum -- deduplicating a token whose
slots share a destination would save it, at a 1/world_size hit rate -- but it
buys something worth more than the bandwidth: every dispatched row carries
exactly one expert, so the expert rank sees ``topk=1`` and runs the single-rank
layer completely unchanged. No routing, GEMM or reduce code knows EP exists.

Four collectives per layer:

    counts     [world_size] int64      who is sending whom how many rows
    dispatch   [rows, model_dim]       the activations
    dispatch   [rows, 2] int32         local expert id, routing weight bits
    combine    [rows, model_dim]       the results, splits reversed

The expert id and the weight ride together in one int32 pair rather than in two
collectives of their own. Eight bytes a row against the payload's four thousand
is nothing as bandwidth, but on this machine an all-to-all costs 30-60 us before
it moves anything, so the collective that carries them is worth as much as the
one that carries the activations. Packing them saves one.

The routing weight travels with the row and is folded in by gemm2, exactly as at
EP=1, rather than being applied on the home rank at combine time. Both are
correct -- the home rank has the weights either way -- but folding it in gemm2
keeps the arithmetic bit-identical to the single-rank path, which is what the
tests compare against, and avoids needing a weighted reduce that
``compile_moe_reduction`` does not provide.

``all_to_all_single`` takes its split sizes as host integers, so learning how
many rows are coming means reading the counts back and synchronising -- the same
stall o2 removed from routing, back again once per dispatch. ``capacity`` is the
way out and is the third collective above as well: give every rank pair a fixed
number of slots and both the counts exchange and the readback stop being
necessary, because there is nothing left to learn. That is the decode path, and
it is opt-in for a reason -- see below.

Where this stands, measured at EP=2 on gfx1100 with
``scripts/bench_rdna3_moe_ep.py`` (model_dim=2048, inter_dim=768, 8 experts a
rank, topk=2, microseconds a rank):

    tokens/rank   plan     dispatch   layer   combine   total
      1024        device      503      378      352     1172
      1024        host        588      379      352     1322
        32        device      353      104      137      587
        32        host        531      101      110      816

The prefill rows are mostly honest work: 4.19 MB leaves each rank per dispatch
and an all-to-all on these cards runs at 12-15 GB/s over PCIe, so 305 us of the
503 is the wire. The decode rows are not. Nothing moves at 32 tokens -- 0.13 MB
-- and dispatch still costs three times the layer it feeds.

The ``host`` rows are what the plan cost as torch calls, and the gap between the
two is ``dispatch_kernel.py`` (o4): fifteen launches over tensors of 64 elements
became one. What is left at decode is the collectives themselves. Three of them
-- counts, payload, metadata -- at 45-50 us each whatever they carry, plus the
readback, is about 165 us of the 353, and the rest is the host driving them.

Folding the metadata into the payload's collective was the obvious lever here
and it was tried; ``repro/o8_merge_meta_ab.py`` is the A/B. It is not taken,
because it does not pay: one collective fewer, against a copy to stage the send
rows and another to make the received activations contiguous for a GEMM whose
leading dimension is fixed at k_dim. Measured at D=4096/I=14336, against the two
collectives below, on dispatch alone: +3 to +5% at prefill, -7% at decode with
the spreads overlapping. On the layer, nothing outside the noise.

Two things found while measuring it are worth keeping, both about
``all_to_all_single`` rather than about MoE:

  A block that is not a multiple of 16 bytes runs at about a tenth of the
  bandwidth of one that is -- 2 GB/s against 21. The block is
  ``rows x row_bytes`` and the routing picks the row count, so only a row that
  is itself 16-byte aligned is safe. Widening a bf16 row by two int32 of
  metadata gives 8200 bytes at D=4096, which is aligned for an even row count
  and not for an odd one; the first attempt at the merge above was ten times
  slower than the thing it replaced, for that reason alone.

  Cost steps around 4 MiB a destination: 16.7 GB/s at or below, 20.7 just
  above. The dispatch payload is ``tokens x topk / world_size`` rows of
  ``model_dim x 2`` bytes, all powers of two in a real model, so it lands on
  the round size by construction -- at EP=4 with 1024 tokens a rank and
  D=4096 it is exactly 4 MiB, on the slow side. Padding the row to step over
  cuts dispatch 7.8% there. It is not done, because the band is narrow and
  which side of it a layer sits on moves with the world size: the same padding
  costs 6.6% at EP=2, where the block is already 8 MiB.

``repro/o8_alltoall_size.py`` measures both.

**The capacity path (o9).** ``capacity=`` on ``ep_dispatch`` gives each rank pair
that many slots whether the routing fills them or not, which makes every block
the same length, which makes the split sizes constants: no counts exchange, no
readback, no host in the step at all. Priced first this time, by handing the real
dispatch its counts as constants -- a benchmark's routing does not change, so
that is exact and needs none of the machinery. At EP=4 decode it said the whole
layer had 13.5% in it, and 15.3% with a graph capture on top, against 4.7% from
summing the two steps being removed. **The assembled gain was larger than the
parts**, the opposite way round from o7 and o8: the readback was not costing its
own 22 us, it was stopping the queue and preventing the dispatch from overlapping
the layer's launches.

Built, it comes to 10.9% at EP=4 decode and 11.6% captured, the difference being
padding on the wire. Three things about it that are not obvious:

  The padding is free after the wire, and that is what makes it viable rather
  than a detail. Padded slots carry a local expert id of -1, and the routing
  kernel's histogram and scatter both test ``id == e`` over ``[0, experts)``, so
  those rows join no tile and reach no GEMM. If they did, a ``tile_m=64`` tile at
  decode is 22.5 GFLOP and 450 us, and a factor of two would mean half again as
  many tiles -- the whole idea would be dead.

  The capacity that cannot overflow is not the capacity to use. ``tokens*topk``
  is what no routing can exceed, and it measured **+7.0%** at EP=4 decode: four
  times the payload, over two collectives, gives back everything the readback
  saved. ``capacity_for``'s factor of two is where the padding stops mattering.

  It loses at prefill, +4.7% at a factor of two, because that exchange is
  bandwidth-bound and doubling the payload costs more than the fixed overhead it
  removes. So ``capacity=None`` stays the default and this is a decode path.

What it also buys, and what may matter more than the 11% for a server: the
spread. At EP=4 decode the packed path measures over a ~100 us band, the fixed
capacity over ~10, and the captured graph over ~3.

``repro/o9_decode_overhead.py`` is the pricing and ``repro/o9_capacity_ab.py``
the built version with the capacity sweep.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.distributed as dist

from kernels.moe.rdna3_moe.dispatch_kernel import build_dispatch_plan
from kernels.moe.rdna3_moe.forward import moe_forward, moe_reduce


class Exchange(NamedTuple):
    """What combine needs to undo a dispatch.

    ``send_order`` is the permutation dispatch applied: sent row ``i`` was slot
    ``send_order[i] % topk`` of token ``send_order[i] // topk``. Combine scatters
    the returning rows back through it.
    """

    send_counts: list[int]
    recv_counts: list[int]
    send_order: torch.Tensor
    tokens: int
    topk: int
    model_dim: int
    # Set when the exchange ran at a fixed capacity per rank pair. Then every
    # block is ``capacity`` long, the counts above are all that, and
    # ``send_order`` has an entry per *slot* rather than per row -- the unused
    # ones addressing a scratch slot combine keeps past the real rows.
    capacity: int | None = None


class Dispatch(NamedTuple):
    """The rows that landed on this rank, shaped for ``moe_forward``.

    ``topk_ids`` is ``[rows, 1]`` of *local* expert indices and ``topk_weights``
    is ``[rows, 1]``, so the layer runs at topk=1 over whatever arrived.
    """

    x: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    exchange: Exchange


def _group_size(group) -> tuple[int, int]:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("EP needs an initialised process group; call dist.init_process_group first")
    return dist.get_world_size(group), dist.get_rank(group)


_OVERFLOW_WATCH: dict[tuple, tuple[torch.Tensor, list[int]]] = {}


def _check_overflow(plan, world_size: int, capacity: int) -> None:
    """Raise if a capacity dispatch dropped rows -- a step or two after it did.

    The count is on the device, and reading it there costs the sync the capacity
    exists to avoid. So it comes back the only way that stays free: an async copy
    into pinned memory, stream-ordered behind the plan, read on a later call by
    which time some earlier step's value has landed. Nothing waits, a capture
    records the copy like any other stream op, and a routing that overflows is
    still heard from within a step or two rather than never.

    What that buys is a loud failure instead of a quiet one. It is not a fence:
    the step that overflowed has already gone out, with the excess rows missing.
    A caller who cannot have that runs at ``safe_capacity``, where the routing
    has no way to overflow, and pays the bytes.
    """
    flag = plan.counts[2 * world_size + 1 : 2 * world_size + 2]
    key = (str(plan.counts.device),)
    seen = _OVERFLOW_WATCH.get(key)
    if seen is None:
        seen = (torch.zeros(1, dtype=torch.int32, pin_memory=True), [0])
        _OVERFLOW_WATCH[key] = seen
    host, armed = seen
    dropped = int(host[0])
    if armed[0] and dropped:
        host[0] = 0
        raise RuntimeError(
            f"a dispatch at capacity={capacity} dropped {dropped} row(s): some rank pair was "
            f"routed more than {capacity} of the {plan.send_order.shape[0] // world_size * world_size} "
            f"slots. Raise the capacity (see capacity_for's factor) or use safe_capacity, which "
            f"no routing can overflow. Detected one or more steps late -- see _check_overflow."
        )
    host.copy_(flag, non_blocking=True)
    armed[0] = 1


def ep_dispatch(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    experts_per_rank: int,
    group=None,
    plan_impl: str = "device",
    capacity: int | None = None,
    stream=None,
) -> Dispatch:
    """Send each routed slot to the rank owning its expert.

    ``x`` is ``[tokens, model_dim]`` and ``topk_ids`` is ``[tokens, topk]`` of
    *global* expert ids -- expert ``e`` lives on rank ``e // experts_per_rank``
    as local expert ``e % experts_per_rank``. Every rank in the group must call
    this, including one with no tokens.

    ``plan_impl`` picks who groups the slots by destination: ``"device"`` is
    ``dispatch_kernel``'s one launch, ``"host"`` the torch calls it replaced,
    kept as the reference the tests compare against.

    ``capacity`` gives each rank pair that many slots, so the split sizes are
    constants and this function needs no host: no counts exchange, no readback,
    nothing to sync on. That is what makes the step capturable, and at decode it
    is worth more than the collective it removes -- the sync was not costing its
    own 25 us, it was stopping the queue. ``"auto"`` takes ``capacity_for``'s
    default, which is a balanced share doubled; a routing lopsided enough to
    overflow it loses the excess rows and ``_check_overflow`` says so a step or
    two later. See this module's docstring. Requires ``plan_impl="device"``.

    **Every rank must pass the same capacity.** It is what both sides of the
    exchange use to split the buffer, so ranks that disagree do not agree on
    where a block ends -- and because nothing is exchanged to find out, nothing
    catches it either: the collective reads the length it was told. A capacity
    belongs to a deployment the way a maximum batch size does. ``"auto"`` derives
    it from *this* rank's token count and is therefore only the same number
    everywhere if the token counts are, which is the decode case and not the
    general one; ranks with ragged batches pass the integer themselves.
    """
    world_size, _ = _group_size(group)
    if x.dim() != 2:
        raise ValueError(f"x must be [tokens, model_dim], got {tuple(x.shape)}")
    if topk_ids.shape[0] != x.shape[0] or topk_weights.shape != topk_ids.shape:
        raise ValueError(
            f"routing must be [tokens={x.shape[0]}, topk], got ids={tuple(topk_ids.shape)} "
            f"weights={tuple(topk_weights.shape)}"
        )
    if topk_weights.dtype != torch.float32:
        raise ValueError(f"topk_weights must be f32, got {topk_weights.dtype}")

    tokens, model_dim = x.shape
    topk = topk_ids.shape[1]
    epr = int(experts_per_rank)
    n_experts = world_size * epr
    dev = x.device

    if plan_impl == "device":
        plan = build_dispatch_plan(topk_ids, topk_weights, world_size=world_size, experts_per_rank=epr, stream=stream)
        send_order = plan.send_order
        meta_send = plan.meta
        # The counts buffer holds the send half, the reply half, and the range
        # check, so all of it comes back in one copy. The two halves are disjoint
        # slices of one allocation, which is all the collective asks.
        dist.all_to_all_single(plan.counts[world_size : 2 * world_size], plan.counts[:world_size], group=group)
        x_send = x.index_select(0, plan.src_row).contiguous()
        read_back = plan.counts.tolist()
    elif plan_impl == "host":
        ids = topk_ids.reshape(-1).to(torch.int64)
        # Group the slots by destination. A stable sort keeps a destination's rows
        # in (token, slot) order, which is what lets combine index back through
        # ``send_order`` without carrying the token id in the payload. Sorting the
        # destinations rather than argsorting them hands back the sorted keys too,
        # and the counts follow from where each destination's run begins -- which
        # is worth doing over ``torch.bincount``, whose 2048-element call costs
        # twice what the sort does.
        dest = torch.div(ids, epr, rounding_mode="floor").clamp_(0, world_size - 1)
        dest_sorted, send_order = torch.sort(dest, stable=True)
        edges = torch.searchsorted(dest_sorted, torch.arange(world_size + 1, dtype=dest.dtype, device=dev))
        send_counts_t = edges[1:] - edges[:-1]

        recv_counts_t = torch.empty_like(send_counts_t)
        dist.all_to_all_single(recv_counts_t, send_counts_t, group=group)

        # Everything that does not depend on the split sizes is queued before the
        # readback, so the wait covers the exchange and the gathers at once, and
        # the range check rides along in it rather than paying for a sync of its
        # own.
        src_row = torch.div(send_order, topk, rounding_mode="floor")
        x_send = x.index_select(0, src_row).contiguous()
        # One row of metadata: the local expert id, and the routing weight carried
        # as its bits so both fit one int32 tensor.
        meta_send = torch.empty(int(send_order.numel()), 2, dtype=torch.int32, device=dev)
        meta_send[:, 0] = (ids.index_select(0, send_order) % epr).to(torch.int32)
        meta_send[:, 1] = topk_weights.reshape(-1).index_select(0, send_order).view(torch.int32)
        out_of_range = ((ids < 0) | (ids >= n_experts)).sum().reshape(1)
        read_back = torch.cat([send_counts_t, recv_counts_t, out_of_range]).tolist()
    else:
        raise ValueError(f"plan_impl must be 'device' or 'host', got {plan_impl!r}")

    # The stall this module's docstring warns about: the split sizes have to be
    # host integers.
    send_counts = read_back[:world_size]
    recv_counts = read_back[world_size : 2 * world_size]
    if read_back[2 * world_size]:
        raise ValueError(f"topk_ids must be global expert ids in [0, {n_experts}) for this group")
    rows_recv = int(sum(recv_counts))

    x_recv = torch.empty(rows_recv, model_dim, dtype=x.dtype, device=dev)
    meta_recv = torch.empty(rows_recv, 2, dtype=torch.int32, device=dev)
    dist.all_to_all_single(x_recv, x_send, recv_counts, send_counts, group=group)
    dist.all_to_all_single(meta_recv, meta_send, recv_counts, send_counts, group=group)

    return Dispatch(
        x=x_recv,
        topk_ids=meta_recv[:, 0].contiguous().view(rows_recv, 1),
        topk_weights=meta_recv[:, 1].contiguous().view(torch.float32).view(rows_recv, 1),
        exchange=Exchange(
            send_counts=send_counts,
            recv_counts=recv_counts,
            send_order=send_order,
            tokens=int(tokens),
            topk=int(topk),
            model_dim=int(model_dim),
        ),
    )


def ep_combine(
    y: torch.Tensor,
    exchange: Exchange,
    *,
    group=None,
    out: torch.Tensor | None = None,
    stream=None,
) -> torch.Tensor:
    """Send expert results home and sum each token's slots.

    ``y`` is ``[rows_recv, model_dim]``, one row per row this rank received.
    Returns ``[tokens, model_dim]`` on the rank the tokens came from.
    """
    world_size, _ = _group_size(group)
    ex = exchange
    rows_recv = int(sum(ex.recv_counts))
    if tuple(y.shape) != (rows_recv, ex.model_dim):
        raise ValueError(f"y must be {(rows_recv, ex.model_dim)} to combine this dispatch, got {tuple(y.shape)}")
    if len(ex.recv_counts) != world_size:
        raise ValueError(f"this exchange was made for a group of {len(ex.recv_counts)}, not {world_size}")

    rows_send = int(sum(ex.send_counts))
    y_back = torch.empty(rows_send, ex.model_dim, dtype=y.dtype, device=y.device)
    if ex.capacity is None:
        dist.all_to_all_single(y_back, y.contiguous(), ex.send_counts, ex.recv_counts, group=group)
    else:
        # Every block is the same length, so the split sizes need not be said --
        # which is the point: saying them would mean knowing them on the host.
        dist.all_to_all_single(y_back, y.contiguous(), group=group)

    # Undo the dispatch permutation into [token, slot] order and sum the slots.
    # Every slot was sent, so every row of the staging buffer is written -- the
    # reduce never reads an uninitialised one.
    #
    # At a fixed capacity ``y_back`` also carries the slots no row was routed
    # into, holding whatever the layer left there. One extra staging row absorbs
    # them: the plan pointed every padded slot at it, and nothing reads it.
    rows = ex.tokens * ex.topk
    staged = torch.empty(rows + (0 if ex.capacity is None else 1), ex.model_dim, dtype=y.dtype, device=y.device)
    staged.index_copy_(0, ex.send_order, y_back)
    return moe_reduce(staged[:rows].view(ex.tokens, ex.topk, ex.model_dim), out=out, stream=stream)


def ep_moe_forward(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    experts_per_rank: int | None = None,
    tile_m: int = 16,
    routing_impl: str = "device",
    plan_impl: str = "device",
    group=None,
    out: torch.Tensor | None = None,
    stream=None,
) -> torch.Tensor:
    """One expert-parallel MoE layer: dispatch, the local layer, combine.

    ``topk_ids`` is global; ``w1``/``w2`` hold this rank's shard of the experts,
    which is where ``experts_per_rank`` defaults from. Returns this rank's
    ``[tokens, model_dim]``.
    """
    epr = int(w1.shape[0]) if experts_per_rank is None else int(experts_per_rank)
    disp = ep_dispatch(x, topk_ids, topk_weights, experts_per_rank=epr, group=group, plan_impl=plan_impl, stream=stream)

    if disp.x.shape[0]:
        y = moe_forward(
            disp.x,
            w1,
            w2,
            disp.topk_ids,
            disp.topk_weights,
            tile_m=tile_m,
            routing_impl=routing_impl,
            stream=stream,
        )
    else:
        # Nobody routed anything here. The collectives still have to run, so this
        # rank contributes an empty send rather than skipping combine.
        y = torch.empty(0, disp.exchange.model_dim, dtype=x.dtype, device=x.device)

    return ep_combine(y, disp.exchange, group=group, out=out, stream=stream)
