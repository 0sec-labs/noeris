# Results Index

Canonical latest artifacts (current public references):

- Gemma deeper-fusion full-layer main results:
  - `docs/results/gemma4-layer-bench-deeper-fusion-a100-after-geglu-retune.json`
  - `docs/results/gemma4-layer-bench-deeper-fusion-h100-after-geglu-retune.json`
- Gemma deeper-fusion stability reruns:
  - `docs/results/gemma4-layer-bench-deeper-fusion-a100-after-geglu-retune-repeat2.json`
  - `docs/results/gemma4-layer-bench-deeper-fusion-h100-after-geglu-retune-repeat3.json`
- Policy-routing sanity checks:
  - `docs/results/gemma4-layer-bench-deeper-fusion-a100-after-policy-routing-sanity.json`
  - `docs/results/gemma4-layer-bench-deeper-fusion-h100-after-policy-routing-sanity.json`

README headline claim artifacts:

- `docs/results/a100-sliding-window-showdown.json`
- `docs/results/a100-end-to-end-26layer.json`
- `docs/results/a100-19model-generalization.json`
- `docs/results/hardware-cross-learning-a100-to-h100.json`
- `docs/results/qk-norm-rope-a100-full.json`
- `docs/results/qk-norm-rope-h100-full.json`

Targeted QK-norm attention reruns:

- `docs/results/bandit-qknorm-attention-a100-v3.json`
- `docs/results/bandit-qknorm-attention-a100-v3.md`

FP8 baseline probes:

- `docs/results/fp8-hopper-probe.json`
- `docs/results/fp8-triton-matmul-probe-h100.json`
- `docs/results/fp8-triton-matmul-probe-h100.md`
- `docs/results/fp8-triton-matmul-autotune-h100.json`
- `docs/results/fp8-triton-matmul-autotune-h100.md`
- `docs/results/fp8-triton-matmul-autotune-h100-v2.json`
- `docs/results/fp8-triton-matmul-autotune-h100-v2.md`
- `docs/results/fp8-triton-matmul-autotune-h100-v3.json`
- `docs/results/fp8-triton-matmul-autotune-h100-v3.md`
- `docs/results/fp8-triton-matmul-autotune-h100-v4-splitk.json`
- `docs/results/fp8-triton-matmul-autotune-h100-v4-splitk.md`
- `docs/results/fp8-prepack-amortization-h100.json`
- `docs/results/fp8-prepack-amortization-h100.md`
- `docs/results/fp8-layout-reuse-policy-h100.json`
- `docs/results/fp8-layout-reuse-policy-h100.md`
- `docs/results/fp8-layout-runtime-integration-h100.json`
- `docs/results/fp8-layout-runtime-integration-h100.md`
- `docs/results/fp8-layout-runtime-cache-integration-h100.json`
- `docs/results/fp8-layout-runtime-cache-integration-h100.md`
- `docs/results/fp8-layout-runtime-integration-token-loop.json`
- `docs/results/fp8-layout-runtime-integration-token-loop.md`
- `docs/results/release-confidence-fp8-runtime-ci-local.json`
- `docs/results/release-confidence-fp8-runtime-ci-local.md`

Executor integration note:

- Live matmul benchmark payload now includes `fp8-runtime-layout-summary.json` when FP8 fixtures are present.

Speculative decoding verify+accept baseline:

- `docs/results/spec-decode-verify-accept-baseline.json`
- `docs/results/spec-decode-verify-accept-baseline.md`

Speculative decoding verify+accept fused v1:

- `docs/results/spec-decode-verify-accept-fused-v1.json`
- `docs/results/spec-decode-verify-accept-fused-v1.md`

Speculative decoding runtime-hook integration benchmark:

- `docs/results/spec-decode-verify-accept-runtime-integration.json`
- `docs/results/spec-decode-verify-accept-runtime-integration.md`

KV cache quantize-on-write fused v1:

- `docs/results/kv-quant-write-fused-v1.json`
- `docs/results/kv-quant-write-fused-v1.md`

KV cache quantize-on-write runtime integration:

- `docs/results/kv-quant-write-runtime-integration.json`
- `docs/results/kv-quant-write-runtime-integration.md`

Cross-vendor zero-shot scaffold (MI300X label, no target measurements):

- `docs/results/cross-vendor-zero-shot-scaffold-mi300x.json`
- `docs/results/cross-vendor-zero-shot-scaffold-mi300x.md`

Cold-shape cross-run learning ablation v2:

- `docs/results/cold-shape-cross-run-ablation-v2.json`
- `docs/results/cold-shape-cross-run-ablation-v2.md`

Kernel-aware NAS multi-hardware latency proxy:

- `docs/results/kernel-aware-nas-multihardware.json`
- `docs/results/kernel-aware-nas-multihardware.md`

Reproducible benchmark-pack command:

```bash
PYTHONPATH=src uv run --python 3.11 --no-project --with modal python3 scripts/gemma4_layer_benchmark_pack.py
```

This writes canonical pack outputs:

- `docs/results/gemma4-layer-bench-deeper-fusion-canonical-pack.json`
- `docs/results/gemma4-layer-bench-deeper-fusion-canonical-pack.md`

Current canonical pack artifacts:

- `docs/results/gemma4-layer-bench-deeper-fusion-canonical-pack.json`
- `docs/results/gemma4-layer-bench-deeper-fusion-canonical-pack.md`

Runtime metadata guidance:

Future paper-facing benchmark artifacts should include an `"environment"` block
captured by `research_engine.benchmark_metadata.collect_environment()`. This
records Python, PyTorch, Triton, CUDA runtime/driver versions, GPU name, git
commit, and the benchmark command so that results are auditable and
reproducible. Historical artifacts are not retroactively updated unless a
migration is clearly worthwhile.

Notes:

- Historical artifacts in this directory are retained for auditability and timeline context.
- README and paper should reference canonical latest artifacts above unless explicitly discussing historical progression.
