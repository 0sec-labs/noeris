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
- `scripts/nas_experiment.py --all-hardware` writes a deterministic
  A100/T4/H100 artifact pack with known models, seed candidates, generated
  candidates, a size-constrained latency ranking, and grouped knob winners.

Run:

```bash
python scripts/nas_experiment.py --hardware a100
python scripts/nas_experiment.py --hardware t4
python scripts/nas_experiment.py --hardware h100
python scripts/nas_experiment.py --hardware a100 \
  --json-output docs/results/kernel-aware-nas-a100.json
python scripts/nas_experiment.py --all-hardware
```

## What this answers

The experiment can answer hardware-facing architecture questions such as:

- which candidate has the lowest predicted layer latency on a target GPU;
- which kernel dominates the layer latency for that candidate;
- which generated head-dim / GQA / FFN-ratio / QK-norm/RoPE branch candidates
  rank fastest;
- which constrained value wins for each architecture knob across A100, T4, and
  H100 in the checked-in `knob_summary` block;
- whether a hidden size, head size, or FFN size falls off the 128-wide tile
  boundary; and
- whether the same ranking holds across A100, T4, and H100 profiles.

When `--json-output` is set, the script writes a deterministic artifact with
the candidate comparison table, fastest-first rankings, tile-cliff sweeps, and
summary fields. This is intended for CI comparisons and future checked-in
benchmark packs.

The canonical multi-hardware pack is:

- `docs/results/kernel-aware-nas-multihardware.json`
- `docs/results/kernel-aware-nas-multihardware.md`

## Measured vs Proxy

Measured inputs:

- A100 operator records in `.noeris/cost-model-training.json` are used when
  present to calibrate A100 RMSNorm, matmul, and attention constants.
- The calibration block in the JSON artifact records the source path, SHA-256,
  operator summaries, and any derived profile overrides.

Proxy-only pieces:

- T4 and H100 profiles currently use fixed effective-throughput constants.
- Layer latency is predicted from kernel-level throughput proxies, not measured
  with an end-to-end model benchmark.
- The `quality_constrained_latency` ranking uses size/capacity constraints:
  minimum hidden dimension, minimum `hidden_dim * ffn_dim` proxy, and an FFN
  ratio range. This prevents the search from simply choosing the smallest
  architecture, but it is not a learned quality model.
- The norm-placement knob in this proxy is the QK-norm/RoPE branch. It does
  not yet model full block-level norm variants such as pre-norm vs post-norm
  training behavior.

Still speculative:

- The ranking does not estimate perplexity, training stability, memory
  pressure, or downstream accuracy.
- The generated candidates have not been trained or evaluated.
- A full NAS loop must pair this latency proxy with real quality measurements
  before recommending an architecture.

## Next steps

1. Add real quality measurements for the top constrained candidates.
2. Validate top candidates with real Triton layer benchmarks.
3. Replace T4/H100 profile constants with measured persisted artifacts.
4. Expand candidate generation only after adding quality measurements, so the
   larger search does not optimize latency alone.
