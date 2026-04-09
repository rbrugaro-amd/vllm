# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Inductor post-grad pass: fuse paired q_a_layernorm + kv_a_layernorm in
DeepSeek-V3 / Kimi-K2 MLA attention into aiter's fused_qk_rmsnorm HIP
kernel.

Target FX-graph pattern (unfused):
    gemm_with_dynamic_quant -> split_with_sizes([q_dim, kv_dim])
        +-- q_c     -> rocm_aiter_rms_norm(q_c, q_w, eps)     -> gemm (q_b_proj)
        +-- kv_lora -> split_with_sizes([kv_c_dim, k_pe_dim])
                        +-- kv_c -> rocm_aiter_rms_norm(kv_c, kv_w, eps)
                        +-- k_pe

Replacement:
    Both norms -> fused_mla_dual_rms_norm(q_c, q_w, kv_c, kv_w, eps1, eps2)
    backed by aiter fused_qk_rmsnorm at runtime.
"""
from __future__ import annotations

import operator
from typing import Optional

import torch
from torch import fx

from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Custom-op registration
# ---------------------------------------------------------------------------

_vllm_lib: Optional[torch.library.Library] = None


def _ensure_op_registered() -> None:
    global _vllm_lib
    if _vllm_lib is not None:
        return

    _vllm_lib = torch.library.Library("vllm", "FRAGMENT")

    _vllm_lib.define(
        "fused_mla_dual_rms_norm("
        "  Tensor x1, Tensor x1_weight,"
        "  Tensor x2, Tensor x2_weight,"
        "  float x1_epsilon, float x2_epsilon"
        ") -> (Tensor, Tensor)"
    )

    @torch.library.impl(_vllm_lib, "fused_mla_dual_rms_norm", "CUDA")
    def _cuda_impl(
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

    @torch.library.impl(_vllm_lib, "fused_mla_dual_rms_norm", "Meta")
    def _meta_impl(
        x1: torch.Tensor,
        x1_weight: torch.Tensor,
        x2: torch.Tensor,
        x2_weight: torch.Tensor,
        x1_epsilon: float,
        x2_epsilon: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.empty_like(x1), torch.empty_like(x2))


# ---------------------------------------------------------------------------
# FX graph pass
# ---------------------------------------------------------------------------

# Matches both the ROCm/aiter lowered op and the vllm_ir intermediate op.
# vLLM >=0.19 uses vllm_ir.rms_norm at the post-grad stage; IR lowering
# to rocm_aiter_rms_norm happens in a later pass.
_RMS_NORM_TARGET_NAMES = frozenset([
    "vllm::rocm_aiter_rms_norm",
    "vllm_ir::rms_norm",
])


class MLADualRMSNormFusionPass(VllmInductorPass):
    """
    Post-grad FX pass that fuses paired q / kv RMS norms in MLA attention
    into ``fused_mla_dual_rms_norm`` backed by aiter's
    ``fused_qk_rmsnorm`` HIP kernel.
    """

    def __init__(self, config: VllmConfig):
        super().__init__(config)
        _ensure_op_registered()
        self.matched_count: int = 0

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        count = self._fuse_mla_rms_norms(graph)
        self.matched_count += count
        if count > 0:
            graph.lint()
            logger.info(
                "MLADualRMSNormFusionPass: fused %d q/kv norm pair(s)", count
            )

    def uuid(self) -> str:
        return self.hash_source(self)

    # ------------------------------------------------------------------
    # Core rewrite loop
    # ------------------------------------------------------------------

    def _fuse_mla_rms_norms(self, graph: fx.Graph) -> int:
        split_target = torch.ops.aten.split_with_sizes.default

        fused_count = 0
        processed: set[fx.Node] = set()
        to_erase: list[fx.Node] = []

        for node in list(graph.nodes):
            if not self._is_rms_norm(node) or node in processed:
                continue

            pair = self._match_mla_pair(node, split_target)
            if pair is None:
                continue

            q_norm, kv_norm, q_c, kv_c = pair
            if kv_norm in processed:
                continue

            self._rewrite(graph, q_norm, kv_norm, q_c, kv_c)

            processed.update({q_norm, kv_norm})
            to_erase.extend([q_norm, kv_norm])
            fused_count += 1

        for node in reversed(to_erase):
            graph.erase_node(node)

        return fused_count

    # ------------------------------------------------------------------
    # Pattern matching helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_rms_norm(node: fx.Node) -> bool:
        if node.op != "call_function":
            return False
        name = getattr(node.target, "_name", None)
        return name is not None and name in _RMS_NORM_TARGET_NAMES

    @staticmethod
    def _is_getitem(node: fx.Node, *, index: int) -> bool:
        return (
            node.op == "call_function"
            and node.target is operator.getitem
            and node.args[1] == index
        )

    @staticmethod
    def _find_getitem_user(node: fx.Node, *, index: int) -> Optional[fx.Node]:
        for u in node.users:
            if (
                u.op == "call_function"
                and u.target is operator.getitem
                and u.args[1] == index
            ):
                return u
        return None

    def _match_mla_pair(
        self,
        q_norm: fx.Node,
        split_target,
    ) -> Optional[tuple[fx.Node, fx.Node, fx.Node, fx.Node]]:
        """Walk the graph from *q_norm* to verify the MLA dual-norm pattern."""
        q_c = q_norm.args[0]

        if not self._is_getitem(q_c, index=0):
            return None
        q_split = q_c.args[0]
        if q_split.op != "call_function" or q_split.target is not split_target:
            return None

        kv_lora = self._find_getitem_user(q_split, index=1)
        if kv_lora is None:
            return None

        kv_split = None
        for u in kv_lora.users:
            if u.op == "call_function" and u.target is split_target:
                kv_split = u
                break
        if kv_split is None:
            return None

        kv_c = self._find_getitem_user(kv_split, index=0)
        if kv_c is None:
            return None

        kv_norm = None
        for u in kv_c.users:
            if self._is_rms_norm(u):
                kv_norm = u
                break
        if kv_norm is None:
            return None

        return (q_norm, kv_norm, q_c, kv_c)

    # ------------------------------------------------------------------
    # Graph rewrite
    # ------------------------------------------------------------------

    def _rewrite(
        self,
        graph: fx.Graph,
        q_norm: fx.Node,
        kv_norm: fx.Node,
        q_c: fx.Node,
        kv_c: fx.Node,
    ) -> None:
        kv_split = kv_c.args[0]
        kv_lora = kv_split.args[0]
        q_split = q_c.args[0]

        # Hoist nodes to ensure topological order:
        # q_split -> q_c -> kv_lora -> kv_split -> kv_c -> [fused]
        q_split.append(q_c)
        q_c.append(kv_lora)
        kv_lora.append(kv_split)
        kv_split.append(kv_c)

        with graph.inserting_after(kv_c):
            fused = graph.call_function(
                torch.ops.vllm.fused_mla_dual_rms_norm.default,
                args=(
                    q_c,
                    q_norm.args[1],   # q_weight
                    kv_c,
                    kv_norm.args[1],  # kv_weight
                    q_norm.args[2],   # q_epsilon
                    kv_norm.args[2],  # kv_epsilon
                ),
            )
        with graph.inserting_after(fused):
            q_normed = graph.call_function(
                operator.getitem, args=(fused, 0)
            )
        with graph.inserting_after(q_normed):
            kv_normed = graph.call_function(
                operator.getitem, args=(fused, 1)
            )

        q_norm.replace_all_uses_with(q_normed)
        kv_norm.replace_all_uses_with(kv_normed)
