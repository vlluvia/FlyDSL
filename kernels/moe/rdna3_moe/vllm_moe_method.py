# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""The vLLM binding: a fused-MoE method backed by the RDNA3 grouped GEMM.

Imports vLLM, so it is not pulled in by ``kernels.moe.rdna3_moe``'s package
init -- import it explicitly from a vLLM process.

The seam is ``FusedMoEMethodBase.forward_cuda``, which is handed this rank's
weights and the router's decision and is asked for this rank's partial sum. That
is precisely ``moe_forward``'s contract, so the override is a layout check and a
call; see ``vllm_adapter`` for why nothing has to be transposed and how the
shared expert gets folded in.

Everything above the seam is left to vLLM: the router, the shared expert when it
is not fused, and the all-reduce that finishes the tensor-parallel sum.
"""

from __future__ import annotations

import torch
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

from kernels.moe.rdna3_moe.vllm_adapter import check_layer_supported, moe_layer_forward

__all__ = ["RDNA3FusedMoEMethod", "install_rdna3_moe"]


class RDNA3FusedMoEMethod(UnquantizedFusedMoEMethod):
    """``UnquantizedFusedMoEMethod`` with the routed experts on the RDNA3 kernel.

    Only ``forward_cuda`` changes. Weight creation, loading and padding stay the
    base class's, except for ROCm's optional inter-slab padding: the grouped
    GEMM computes the expert base as ``expert_id * slab_elements`` and therefore
    requires a tightly packed expert dimension.
    """

    def _maybe_pad_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Keep expert slabs contiguous.

        vLLM's ROCm padding returns a strided view whose logical shape is
        unchanged but whose expert stride contains an extra 512-byte gap. The
        RDNA3 kernel receives only a pointer and the logical dimensions, so it
        cannot discover that gap and would read the wrong expert after expert
        zero.
        """
        return weight

    def _setup_kernel(
        self,
        layer,
        w13: torch.Tensor,
        w2: torch.Tensor,
    ) -> None:
        """Leave weights in vLLM's native ``[E, N, K]`` layout.

        The base implementation prepares a second modular kernel and may
        transform weights for its backend. This method itself is the backend,
        and ``moe_layer_forward`` consumes the native tensors directly.
        """
        if not w13.is_contiguous() or not w2.is_contiguous():
            raise ValueError(
                "the RDNA3 MoE kernel requires contiguous w13_weight and "
                "w2_weight expert slabs"
            )
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        self.moe_kernel = None

    def forward_cuda(
        self,
        layer,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        check_layer_supported(layer, x.dtype)

        # SharedExperts is owned by MoERunner: it is executed before this call,
        # read back after it, then added to this routed partial before the one
        # TP all-reduce. Calling it here would execute it twice and violate its
        # single-pending-output state machine.

        hidden = x.shape[-1]
        out = moe_layer_forward(
            x.view(-1, hidden),
            layer.w13_weight,
            layer.w2_weight,
            topk_ids,
            topk_weights,
        )
        return out.view(x.shape)

    def forward_native(
        self,
        layer,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        """Use the RDNA3 path when vLLM dispatches the native method.

        CUDA Graph / torch.compile warmup can select ``forward_native`` even
        though the runtime backend is ROCm. The adapter intentionally leaves
        ``moe_kernel`` unset, so forwarding here avoids the base class'
        unrelated kernel assertion.
        """
        return self.forward_cuda(
            layer,
            x,
            topk_weights,
            topk_ids,
            shared_experts,
            shared_experts_input,
        )


def install_rdna3_moe(module: torch.nn.Module) -> int:
    """Swap every eligible unquantized fused-MoE method for the RDNA3 one.

    Returns how many layers were swapped, so a caller can log it and notice when
    the answer is zero. Mirrors ``enable_qwen4_exp_low_latency_gemm``: walk the
    built model and replace a method in place, rather than reaching into how the
    layers were constructed.

    A no-op off gfx11 rather than an error, so a caller can install
    unconditionally: the kernel is wave32 WMMA and its builder refuses anything
    else, and the CDNA MoE kernels live in ``kernels/moe/moe_gemm_2stage``.
    """
    from flydsl.runtime.device import get_rocm_arch

    if not str(get_rocm_arch() or "").startswith("gfx11"):
        return 0

    swapped = 0
    for child in module.modules():
        method = getattr(child, "quant_method", None)
        # `type() is` and not isinstance: a subclass is some other backend's
        # method and replacing it would drop whatever it does.
        if type(method) is not UnquantizedFusedMoEMethod:
            continue
        if not hasattr(child, "w13_weight") or not hasattr(child, "w2_weight"):
            continue
        child.quant_method = _rebind(method)
        swapped += 1
    return swapped


def _rebind(method: UnquantizedFusedMoEMethod) -> RDNA3FusedMoEMethod:
    """Re-type a built method in place, keeping the state it already holds.

    The base method is constructed with a MoE config and then has backend
    selection and kernel setup run against it; rebuilding one from the config
    would redo that and can pick a different backend. Only the class changes.
    """
    method.__class__ = RDNA3FusedMoEMethod
    # ``CustomOp.__init__`` cached the dispatch method before this instance
    # was retyped. Refresh it so ROCm eager execution and Graph warmup both
    # reach the adapter's HIP/CUDA implementation instead of the base native
    # implementation.
    method._forward_method = method.forward_hip
    return method
