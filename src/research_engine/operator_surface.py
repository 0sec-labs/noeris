"""Canonical operator surface definitions.

The full registry includes every operator spec that can be addressed by shared
infrastructure. Not every registered spec is ready for the generic
``triton-iterate`` CLI: some have dedicated benchmark paths or need additional
shape parsing before they can safely use the generic result recorder.
"""

from __future__ import annotations


TRITON_ITERATE_OPERATORS: tuple[str, ...] = (
    "matmul",
    "rmsnorm",
    "softmax",
    "layernorm",
    "cross_entropy",
    "attention",
    "attention_v2",
    "rotary",
    "geglu",
    "fused_norm_linear",
)

TRITON_ITERATE_WORKFLOW_OPERATORS: tuple[str, ...] = (
    "matmul",
    "rmsnorm",
    "softmax",
    "layernorm",
    "cross_entropy",
    "attention",
    "rotary",
    "geglu",
)

PUBLIC_PATCH_OPERATORS: tuple[str, ...] = (
    "rmsnorm",
    "geglu",
)

REGISTERED_INTERNAL_OPERATORS: tuple[str, ...] = (
    "attention_decode",
    "cuda_qk_norm_rope",
    "gelu",
    "grouped_gemm",
    "kv_shared_attention",
    "matmul_splitk",
    "moe_router",
    "ple_fusion",
    "ple_gather",
    "qk_norm_rope",
    "qk_norm_rope_bwd",
    "ssm_scan",
)
