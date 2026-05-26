# Cold-Shape Cross-Run Learning Ablation v2

Generated: 2026-05-26T19:28:24.566553+00:00
Database: `.noeris/cost-model-training.json`
Hardware: `NVIDIA A100-SXM4-80GB`

## Protocol

- Iterations: `6`
- Configs per iteration: `1`
- Seeds: `[0, 1, 2]`
- Curated configs excluded: `true`
- Held-out policy: `largest_shape_size_proxy_per_operator`

## Overall Summary

| Condition | mean final normalized | mean final regret % | median iter to 90% | 90% success |
|---|---:|---:|---:|---:|
| stateless_random | 0.9604 | 3.96 | 2.00 | 0.92 |
| database_seeded | 0.9604 | 3.96 | 1.00 | 0.75 |
| cost_model_ranking | 0.9971 | 0.29 | 1.00 | 1.00 |

## Held-Out Buckets

| Operator | Bucket | Candidates | Oracle best | Metric |
|---|---|---:|---:|---|
| cross_entropy | llama3_128k | 9 | 314.4100 | gb_per_s |
| layernorm | gpt_neox | 8 | 1338.0600 | gb_per_s |
| rmsnorm | llama_13b | 8 | 1409.0400 | gb_per_s |
| softmax | vocab_small | 9 | 1372.5500 | gb_per_s |

## Interpretation

The replay does not validate cross-run memory as the dominant selector under this v2 protocol. With curated starters removed, `cost_model_ranking` has the highest mean final normalized throughput; the paper should frame the persistent database as enabling replayable training data and selector inputs rather than claiming standalone compounding gains from database seeding.

Raw per-iteration histories are stored in the JSON artifact under `by_bucket`.
