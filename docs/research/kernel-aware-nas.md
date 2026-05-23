# Kernel-Aware Architecture Search

Issue #81 tracks a research loop that flips the usual kernel workflow:
instead of only tuning kernels for a fixed model, use measured kernel behavior
to choose model dimensions that the kernels execute efficiently.

## Current implementation

- `src/research_engine/arch_cost_model.py` contains `ArchitectureCostModel`,
  a decoder-layer latency proxy for A100, T4, and H100.
- `ArchitectureCostModel.predict_layer_ms()` estimates per-kernel latency for
  RMSNorm, QKV projection, QK-norm+RoPE, attention, output projection,
  residual adds, and GeGLU MLP.
- `ArchitectureCostModel.rank_configs()` ranks candidate architectures by
  total layer latency or latency normalized by `hidden_dim * ffn_dim`.
- `ArchitectureCostModel.sweep_dimension()` exposes tile-efficiency cliffs for
  dimensions that do not land on the 128-wide matmul tile boundary.
- `generate_nas_candidates()` expands a base config across hidden width, head
  dimension, GQA ratio, FFN ratio, sliding window, and QK-norm placement while
  skipping invalid head/KV combinations.
- `scripts/nas_experiment.py` compares known model shapes against novel
  candidates, prints a fastest-first NAS ranking, and ranks the generated
  candidate space.

Run:

```bash
python scripts/nas_experiment.py --hardware a100
python scripts/nas_experiment.py --hardware t4
python scripts/nas_experiment.py --hardware h100
python scripts/nas_experiment.py --hardware a100 \
  --json-output docs/results/kernel-aware-nas-a100.json
```

## What this answers

The experiment can answer hardware-facing architecture questions such as:

- which candidate has the lowest predicted layer latency on a target GPU;
- which kernel dominates the layer latency for that candidate;
- which generated head-dim / GQA / FFN-ratio / QK-norm candidates rank fastest;
- whether a hidden size, head size, or FFN size falls off the 128-wide tile
  boundary; and
- whether the same ranking holds across A100, T4, and H100 profiles.

When `--json-output` is set, the script writes a deterministic artifact with
the candidate comparison table, fastest-first rankings, tile-cliff sweeps, and
summary fields. This is intended for CI comparisons and future checked-in
benchmark packs.

This is intentionally a latency proxy, not a quality proxy. The ranking does
not estimate perplexity, training stability, memory pressure, or downstream
accuracy. A full NAS loop should pair this latency ranking with a quality
constraint before selecting an architecture.

## Next steps

1. Calibrate the hardcoded effective throughput profiles from the persisted
   `.noeris` kernel performance database instead of manual constants.
2. Add a multi-hardware comparison command that writes A100, T4, and H100
   reports in one invocation.
3. Add candidate-quality constraints before selecting architectures.
4. Validate the top candidates with real Triton layer benchmarks.
