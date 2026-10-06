# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Routing buffers for the RDNA3 grouped GEMM.

The grouped GEMM walks a padded routing order: rows grouped by expert, each
expert's group padded up to the block tile so that no tile spans two experts and
a workgroup can read one weight slab. This module builds that order.

The CDNA sorting kernel (``kernels/moe/moe_sorting_kernel.py``) does the same job
in one launch, but it is wave64: its prefix sum has a DPP path that assumes 64
lanes, and it only builds for gfx94*/gfx95*. Until a wave32 version of it exists
this is host code, for a reason worth writing down.

A torch version of this is about fifteen elementwise and scan launches over a few
thousand elements. On gfx1100 that measured ~307 us whatever the token count --
all of it launch overhead, and four times the three GEMMs of a decode layer put
together. The same arithmetic in numpy, plus the copies, measured 82 us at 32
tokens and 162 us at 1024. So the buffers are built on the host and copied down.

``routing_kernel.py`` is the wave32 one-shot version, and it is what the layer
calls now: 10 us of launch against this module's 103 at 32 tokens. This one stays
as the reference the kernel is tested against, and as the faster of the two once
the expert count gets past a few tens -- see that module for why.

The buffers are the ones the CDNA kernels take, so either builder drops in:
packed id ``(slot << 24) | token``, sentinel ``(topk << 24) | tokens`` for padded
rows, and one entry of ``expert_ids`` per tile naming that tile's expert.
"""

from typing import NamedTuple

import numpy as np
import torch

TOKEN_MASK = 0xFFFFFF
SLOT_SHIFT = 24


class Routing(NamedTuple):
    """One padded routing order, ready to launch.

    ``sorted_ids`` has ``num_blocks * tile_m`` entries and ``expert_ids`` has
    ``num_blocks``. ``num_blocks`` is a python int because it is the grid's Y
    extent; the buffers are on the device.

    ``exact`` says whether that count is the tile count or an upper bound on it.
    A bound is what a builder returns when it will not stop to read the real
    count off the device: the extra tiles are there, filled with padding, and it
    is the *kernel* that skips them by comparing its block index against
    ``num_blocks_device``. A GEMM built without that comparison must be handed an
    exact routing, or it will do the extra tiles' arithmetic for nothing -- which
    is still correct, since a padded tile gathers zeros and stores nothing.
    """

    sorted_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_blocks: int
    tile_m: int
    tokens: int
    topk: int
    exact: bool = True
    num_blocks_device: torch.Tensor | None = None

    @property
    def rows(self) -> int:
        return self.num_blocks * self.tile_m


def build_routing(topk_ids: torch.Tensor, *, experts: int, tile_m: int) -> Routing:
    """Group ``[tokens, topk]`` expert ids by expert and pad each group to ``tile_m``.

    ``topk_ids[t, s]`` is the expert token ``t`` routed to at slot ``s``, indexed
    in **this rank's** experts. An expert no token chose gets no tile, so
    ``num_blocks`` follows the routing rather than having a floor of ``E``; a
    caller reusing one allocation across steps sizes it from
    ``routing_kernel.max_blocks_for`` instead.
    """
    if topk_ids.dim() != 2:
        raise ValueError(f"topk_ids must be [tokens, topk], got {tuple(topk_ids.shape)}")
    tokens, topk = topk_ids.shape
    if topk > 0xFF:
        raise ValueError(f"the packed id has 8 bits of slot, so topk must be < 256, got {topk}")
    if tokens > TOKEN_MASK:
        raise ValueError(f"the packed id has 24 bits of token, so tokens must be < {TOKEN_MASK}, got {tokens}")

    flat = topk_ids.reshape(-1).to(torch.int32).cpu().numpy()
    rows_in = flat.size

    # Rows in expert order. A stable sort keeps each expert's rows in (token,
    # slot) order, which is not required but makes the buffer readable and the
    # gather's access pattern monotonic.
    order = np.argsort(flat, kind="stable")
    counts = np.bincount(flat, minlength=experts)[:experts]
    if int(counts.sum()) != rows_in:
        raise ValueError(
            f"topk_ids has {rows_in - int(counts.sum())} entries outside this rank's {experts} experts"
        )

    blocks = (counts + (tile_m - 1)) // tile_m
    cum_blocks = np.cumsum(blocks)
    num_blocks = int(cum_blocks[-1])

    # Per expert: where its rows start in the padded buffer, less where they
    # start in the sorted one, so that adding a sorted row's own index lands it
    # at its padded slot.
    row_base = (cum_blocks - blocks) * tile_m - (np.cumsum(counts) - counts)
    dest = row_base[flat[order]] + np.arange(rows_in)

    sentinel = (topk << SLOT_SHIFT) | tokens
    sorted_ids = np.full(num_blocks * tile_m, sentinel, dtype=np.int32)
    sorted_ids[dest] = ((order % topk) << SLOT_SHIFT) | (order // topk)
    expert_ids = np.repeat(np.arange(experts, dtype=np.int32), blocks)

    dev = topk_ids.device
    return Routing(
        sorted_ids=torch.from_numpy(sorted_ids).to(dev, non_blocking=True),
        expert_ids=torch.from_numpy(expert_ids).to(dev, non_blocking=True),
        num_blocks=num_blocks,
        tile_m=tile_m,
        tokens=int(tokens),
        topk=int(topk),
    )
