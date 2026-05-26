# Kernel-Aware NAS Multi-Hardware Pack

This artifact is a deterministic latency proxy. It does not measure model quality, perplexity, or downstream accuracy.

Candidate count: `89`
Calibration status: `loaded`

| Hardware | unconstrained fastest | constrained fastest | constrained candidates |
|---|---|---|---:|
| A100 | deep_narrow | gen_h2048_hd128_gqa4_ffn4p0_w512_rope | 68 |
| T4 | deep_narrow | gen_h2048_hd128_gqa4_ffn4p0_w512_rope | 68 |
| H100 | deep_narrow | gen_h2048_hd128_gqa4_ffn4p0_w512_rope | 68 |

## Architecture Knob Winners

Quality-constrained group winners use the same size/capacity floor as the constrained ranking.

| Hardware | knob | fastest value | fastest config | total ms | candidates |
|---|---|---|---|---:|---:|
| A100 | head_dim | 128 | gen_h2048_hd128_gqa4_ffn4p0_w512_rope | 1.282711 | 33 |
| A100 | gqa_group_size | 4 | gen_h2048_hd128_gqa4_ffn4p0_w512_rope | 1.282711 | 56 |
| A100 | ffn_ratio | 4.0 | gen_h2048_hd128_gqa4_ffn4p0_w512_rope | 1.282711 | 36 |
| A100 | window_size | 512 | gen_h2048_hd128_gqa4_ffn4p0_w512_rope | 1.282711 | 32 |
| A100 | use_qk_norm | rope_only | gen_h2048_hd128_gqa4_ffn4p0_w512_rope | 1.282711 | 32 |
| T4 | head_dim | 128 | gen_h2048_hd128_gqa4_ffn4p0_w512_rope | 9.931457 | 33 |
| T4 | gqa_group_size | 4 | gen_h2048_hd128_gqa4_ffn4p0_w512_rope | 9.931457 | 56 |
| T4 | ffn_ratio | 4.0 | gen_h2048_hd128_gqa4_ffn4p0_w512_rope | 9.931457 | 36 |
| T4 | window_size | 512 | gen_h2048_hd128_gqa4_ffn4p0_w512_rope | 9.931457 | 32 |
| T4 | use_qk_norm | rope_only | gen_h2048_hd128_gqa4_ffn4p0_w512_rope | 9.931457 | 32 |
| H100 | head_dim | 128 | gen_h2048_hd128_gqa4_ffn4p0_w512_rope | 0.607856 | 33 |
| H100 | gqa_group_size | 4 | gen_h2048_hd128_gqa4_ffn4p0_w512_rope | 0.607856 | 56 |
| H100 | ffn_ratio | 4.0 | gen_h2048_hd128_gqa4_ffn4p0_w512_rope | 0.607856 | 36 |
| H100 | window_size | 512 | gen_h2048_hd128_gqa4_ffn4p0_w512_rope | 0.607856 | 32 |
| H100 | use_qk_norm | rope_only | gen_h2048_hd128_gqa4_ffn4p0_w512_rope | 0.607856 | 32 |

## Constraints

- Minimum hidden dimension: `2048`
- Minimum parameter proxy: `16.0M`
- FFN ratio range: `3.75` to `5.5`

## Limitations

- This is a latency proxy, not a measured model-quality or accuracy estimate.
- Quality constraints are size/capacity proxies until real training or eval measurements are added.
- Only A100 constants are calibrated from local persisted records when available.
