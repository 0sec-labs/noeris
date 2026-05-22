# Operator Surface

Noeris keeps several operator surfaces intentionally separate. The registry is
the broad implementation surface; the CLI and workflow surfaces are narrower
because the generic runner also needs shape parsing, result recording, and
budgeted CI behavior.

## Counts

| Surface | Count | Meaning |
|---|---:|---|
| Registered operator specs | 22 | All names in `research_engine.TRITON_OPERATORS.names()` |
| Generic `triton-iterate` / `ablation` CLI | 10 | Operators supported by the shared search loop and result recorder |
| Default Triton Iterate workflow matrix | 8 | Budgeted GitHub Actions matrix for recurring/manual Modal search |
| Public `noeris.patch()` hooks | 2 | Drop-in HuggingFace patch surface in the `noeris` package |

## Registered Operator Specs

Core and normalization:

- `matmul`
- `matmul_splitk`
- `rmsnorm`
- `layernorm`
- `softmax`
- `cross_entropy`
- `gelu`

Attention and prologue:

- `attention`
- `attention_v2`
- `attention_decode`
- `rotary`
- `qk_norm_rope`
- `qk_norm_rope_bwd`
- `cuda_qk_norm_rope`
- `kv_shared_attention`

Fusion and runtime integration:

- `geglu`
- `fused_norm_linear`
- `ple_gather`
- `ple_fusion`

Routing and MoE:

- `moe_router`
- `grouped_gemm`

SSM:

- `ssm_scan`

## Generic Searchable Operators

These operators are accepted by `python -m research_engine.cli triton-iterate
--operator ...` and by the `ablation` command:

- `matmul`
- `rmsnorm`
- `softmax`
- `layernorm`
- `cross_entropy`
- `attention`
- `attention_v2`
- `rotary`
- `geglu`
- `fused_norm_linear`

Operators outside this list may still have benchmark scripts, dedicated
validation paths, or registered specs. They are not part of the generic search
CLI until their shape parsing and result-recording semantics are wired into the
shared path.

## Workflow Matrix

The default `.github/workflows/triton-iterate.yml` matrix runs:

- `matmul`
- `rmsnorm`
- `softmax`
- `layernorm`
- `cross_entropy`
- `attention`
- `rotary`
- `geglu`

`attention_v2` and `fused_norm_linear` are CLI-searchable but excluded from the
default workflow matrix to keep the Modal budget bounded.

## Public Patch API

The importable public package exposes `noeris.patch(model)` for:

- `rmsnorm`
- `geglu`

Lower-level kernels such as `qk_norm_rope`, `cross_entropy`, and attention
operators are available for custom integration and benchmarking, but they are
not generic `noeris.patch()` hooks.
