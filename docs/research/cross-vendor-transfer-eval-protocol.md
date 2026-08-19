# Cross-Vendor Transfer Evaluation Protocol

This protocol defines how to validate issue #82 once AMD measurements are
available. Current repository status: AMD MI300X/MI250 hardware is unavailable
in this lane, so the cross-vendor AMD result remains scaffold-only.

## Inputs

- Prediction artifact from scaffold lane:
  - `docs/results/cross-vendor-zero-shot-scaffold-mi300x.json`
- Optional measurement template, not evidence:
  - `docs/results/cross-vendor-measured-mi300x-template.json`
- Real measured AMD artifact, to be produced only from MI300X/MI250 runs:
  - `docs/results/cross-vendor-measured-mi300x.json`

```json
{
  "is_measured_evidence": true,
  "measured": {
    "attention": {
      "bucket_name": [
        {"config_id": "cfg_1", "metric": 123.4, "latency_ms": 1.23}
      ]
    }
  }
}
```

The template is marked `is_measured_evidence=false` and uses placeholder
metric/latency values. `scripts/cross_vendor_transfer_eval.py` rejects template,
scaffold, deferred, sample, or non-positive measurement rows.

## Command

Template generation:

```bash
PYTHONPATH=src python3 scripts/cross_vendor_measured_pack_from_prediction.py \
  --prediction-json docs/results/cross-vendor-zero-shot-scaffold-mi300x.json \
  --output-json docs/results/cross-vendor-measured-mi300x-template.json \
  --top-k 5
```

Paper-facing evaluation, valid only after replacing the template with real AMD
measurements in `docs/results/cross-vendor-measured-mi300x.json`:

```bash
PYTHONPATH=src python3 scripts/cross_vendor_transfer_eval.py \
  --prediction-json docs/results/cross-vendor-zero-shot-scaffold-mi300x.json \
  --measured-json docs/results/cross-vendor-measured-mi300x.json \
  --top-k 5
```

## Metrics

- Spearman rank correlation between predicted and measured rankings.
- Top-k hit rate overlap between predicted and measured top-k config IDs.
- Latency regret of predicted-best config vs measured-best config.

## Outputs

- `docs/results/cross-vendor-transfer-eval.json`
- `docs/results/cross-vendor-transfer-eval.md`

These outputs are paper-facing evidence for cross-vendor ranking transfer
quality only when produced from a validator-accepted real AMD measured artifact.
Until then, public wording must describe the MI300X lane as scaffold-only.
