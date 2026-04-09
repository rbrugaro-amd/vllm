# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Inductor post-grad pass (PatternMatcher variant): fuse paired
q_a_layernorm + kv_a_layernorm in DeepSeek-V3 / Kimi-K2 MLA attention
into aiter's fused_qk_rmsnorm HIP kernel.

Uses torch._inductor.pattern_matcher.register_replacement to match the
split -> rms_norm topology as a connected subgraph.

Target FX-graph pattern (unfused, vllm_ir stage):
    gemm -> split_with_sizes([q_dim, kv_dim])
        +-- q_c     -> vllm_ir.rms_norm(q_c, q_w, eps)
        +-- kv_lora -> split_with_sizes([kv_c_dim, k_pe_dim])
                        +-- kv_c -> vllm_ir.rms_norm(kv_c, kv_w, eps)
                        +-- k_pe

Replacement:
    gemm -> split_with_sizes([q_dim, kv_dim])
        +-- q_c  -+
        +-- kv_lora -> split_with_sizes([kv_c_dim, k_pe_dim])
                        +-- kv_c -+-> fused_mla_dual_rms_norm(q_c, q_w,
                        +-- k_pe                               kv_c, kv_w,
                                                               eps, eps)
"""
from __future__ import annotations

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._inductor.pattern_matcher import PatternMatcherPass

import vllm.ir.ops
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Custom-op registration via direct_register_custom_op
# ---------------------------------------------------------------------------

_OP_REGISTERED = False


def _fused_mla_dual_rms_norm_impl(
    x1: torch.Tensor,
    x1_weight: torch.Tensor,
    x2: torch.Tensor,
    x2_weight: torch.Tensor,
    x1_epsilon: float,
    x2_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.ops.fused_qk_norm_rope_cache_quant import fused_qk_rmsnorm

    return fused_qk_rmsnorm(
        q=x1,
        q_weight=x1_weight,
        q_eps=x1_epsilon,
        k=x2,
        k_weight=x2_weight,
        k_eps=x2_epsilon,
    )


def _fused_mla_dual_rms_norm_fake(
    x1: torch.Tensor,
    x1_weight: torch.Tensor,
    x2: torch.Tensor,
    x2_weight: torch.Tensor,
    x1_epsilon: float,
    x2_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.empty_like(x1), torch.empty_like(x2))


def _ensure_op_registered() -> None:
    global _OP_REGISTERED
    if _OP_REGISTERED:
        return
    _OP_REGISTERED = True

    direct_register_custom_op(
        op_name="fused_mla_dual_rms_norm",
        op_func=_fused_mla_dual_rms_norm_impl,
        mutates_args=[],
        fake_impl=_fused_mla_dual_rms_norm_fake,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEVICE = torch.device("cuda")


def _empty_bf16(*shape: int) -> torch.Tensor:
    return torch.empty(*shape, dtype=torch.bfloat16, device=_DEVICE)


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------


class MLADualRMSNormPattern:
    """
    Match the paired q/kv RMS norm topology in MLA attention and replace
    with a single fused_mla_dual_rms_norm call.

    The pattern covers the connected subgraph rooted at the first
    split_with_sizes (which produces q_c and kv_lora), through the
    two rms_norm calls, and the k_pe passthrough.
    """

    def __init__(
        self,
        q_dim: int,
        kv_c_dim: int,
        k_pe_dim: int,
        epsilon: float,
    ) -> None:
        self.q_dim = q_dim
        self.kv_dim = kv_c_dim + k_pe_dim
        self.kv_c_dim = kv_c_dim
        self.k_pe_dim = k_pe_dim
        self.epsilon = epsilon

    def get_inputs(self) -> list[torch.Tensor]:
        T = 5
        projected = _empty_bf16(T, self.q_dim + self.kv_dim)
        q_weight = _empty_bf16(self.q_dim)
        kv_weight = _empty_bf16(self.kv_c_dim)
        return [projected, q_weight, kv_weight]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        q_dim = self.q_dim
        kv_dim = self.kv_dim
        kv_c_dim = self.kv_c_dim
        k_pe_dim = self.k_pe_dim
        eps = self.epsilon

        def pattern(
            projected: torch.Tensor,
            q_weight: torch.Tensor,
            kv_weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            q_c, kv_lora = projected.split([q_dim, kv_dim], dim=-1)
            kv_c, k_pe = kv_lora.split([kv_c_dim, k_pe_dim], dim=-1)
            q_normed = vllm.ir.ops.rms_norm(q_c, q_weight, eps)
            kv_normed = vllm.ir.ops.rms_norm(kv_c, kv_weight, eps)
            return q_normed, kv_normed, k_pe

        def replacement(
            projected: torch.Tensor,
            q_weight: torch.Tensor,
            kv_weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            q_c, kv_lora = projected.split([q_dim, kv_dim], dim=-1)
            kv_c, k_pe = kv_lora.split([kv_c_dim, k_pe_dim], dim=-1)
            result = torch.ops.vllm.fused_mla_dual_rms_norm(
                q_c, q_weight, kv_c, kv_weight, eps, eps,
            )
            return result[0], result[1], k_pe

        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            pm.fwd_only,
            pm_pass,
        )


# ---------------------------------------------------------------------------
# Pass
# ---------------------------------------------------------------------------

# DeepSeek-V3 / Kimi-K2 MLA geometry
_MLA_Q_DIM = 1536
_MLA_KV_C_DIM = 512
_MLA_K_PE_DIM = 64


class MLADualRMSNormFusionPass(VllmPatternMatcherPass):
    """
    Post-grad PatternMatcher pass that fuses paired q / kv RMS norms in
    MLA attention into ``fused_mla_dual_rms_norm`` backed by aiter's
    ``fused_qk_rmsnorm`` HIP kernel.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        _ensure_op_registered()

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="mla_dual_rms_norm_fusion_pass"
        )

        for epsilon in [1e-5, 1e-6]:
            MLADualRMSNormPattern(
                q_dim=_MLA_Q_DIM,
                kv_c_dim=_MLA_KV_C_DIM,
                k_pe_dim=_MLA_K_PE_DIM,
                epsilon=epsilon,
            ).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        if self.matched_count > 0:
            logger.info(
                "MLADualRMSNormFusionPass: fused %d q/kv norm pair(s)",
                self.matched_count,
            )

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(self, MLADualRMSNormPattern)
