# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""RDNA3 (gfx11*, wave32) MoE kernels.

WMMA grouped GEMM for expert-parallel MoE, plus the single-rank layer that chains
it: ``moe_forward``.

The CDNA package ``kernels/moe/moe_gemm_2stage`` cannot be retargeted here: it is
wave64 MFMA with fp8/int8 weights, and gfx11 has neither MFMA nor fp8 WMMA. Two
CDNA kernels do run here unchanged, and the layer calls them rather than
duplicating them: the topk-reduce (``compile_moe_reduction``, 256 threads, no
MFMA, no atomics) and the gating softmax (which reduces with ``shuffle_xor`` over
a lane group). The sorting kernel is not one of them -- its prefix sum is wave64
-- so the routing buffers get their own wave32 kernel in ``routing_kernel.py``,
with ``routing.py``'s host builder kept as the reference the tests compare to.

The kernels only ever see **this rank's** experts and the rows that already
landed on this rank, so the same build serves EP=1 and EP>1; see
``kernels/moe/rdna3_moe/grouped_gemm.py`` for the contract. ``ep.py`` is what
makes rows land: RCCL all-to-all dispatch and combine around that same layer,
one node.
"""

from kernels.moe.rdna3_moe.dispatch_kernel import (
    Plan,
    build_dispatch_plan,
    compile_dispatch_plan,
)
from kernels.moe.rdna3_moe.ep import (
    Dispatch,
    Exchange,
    ep_combine,
    ep_dispatch,
    ep_moe_forward,
)
from kernels.moe.rdna3_moe.forward import (
    moe_forward,
    moe_forward_from_logits,
    moe_gating,
    moe_reduce,
)
from kernels.moe.rdna3_moe.grouped_gemm import STAGES, create_grouped_gemm_module
from kernels.moe.rdna3_moe.host import (
    SUPPORTED_TILE_M,
    compile_grouped_gemm,
    compile_moe_gemm1,
    compile_moe_gemm2,
)
from kernels.moe.rdna3_moe.routing import Routing, build_routing
from kernels.moe.rdna3_moe.routing_kernel import (
    build_routing_device,
    compile_routing,
    max_blocks_for,
)

__all__ = [
    "Dispatch",
    "Exchange",
    "Plan",
    "Routing",
    "STAGES",
    "SUPPORTED_TILE_M",
    "build_dispatch_plan",
    "build_routing",
    "build_routing_device",
    "compile_routing",
    "max_blocks_for",
    "compile_dispatch_plan",
    "compile_grouped_gemm",
    "compile_moe_gemm1",
    "compile_moe_gemm2",
    "create_grouped_gemm_module",
    "ep_combine",
    "ep_dispatch",
    "ep_moe_forward",
    "moe_forward",
    "moe_forward_from_logits",
    "moe_gating",
    "moe_reduce",
]
