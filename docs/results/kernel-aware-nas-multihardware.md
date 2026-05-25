# Kernel-Aware NAS Multi-Hardware Pack

This artifact is a deterministic latency proxy. It does not measure model quality, perplexity, or downstream accuracy.

Candidate count: `49`
Calibration status: `loaded`

| Hardware | unconstrained fastest | constrained fastest | constrained candidates |
|---|---|---|---:|
| A100 | deep_narrow | gen_h2048_hd128_gqa4_ffn4p0_w512 | 36 |
| T4 | deep_narrow | gen_h2048_hd128_gqa4_ffn4p0_w512 | 36 |
| H100 | deep_narrow | gen_h2048_hd128_gqa4_ffn4p0_w512 | 36 |

## Constraints

- Minimum hidden dimension: `2048`
- Minimum parameter proxy: `16.0M`
- FFN ratio range: `3.75` to `5.5`

## Limitations

- This is a latency proxy, not a measured model-quality or accuracy estimate.
- Quality constraints are size/capacity proxies until real training or eval measurements are added.
- Only A100 constants are calibrated from local persisted records when available.
