# Cross-Vendor Zero-Shot Scaffold (MI300X from NVIDIA data)

Issue: `#82`.

This scaffold predicts candidate config rankings for an unseen hardware label
(`AMD MI300X`) using only existing NVIDIA benchmark data in the local
ConfigDatabase.

Current status: scaffold-only. No AMD MI300X/MI250 benchmark measurements are
available in this repository lane.

## Run

```bash
PYTHONPATH=src uv run --python 3.11 --no-project python3 scripts/cross_vendor_zero_shot_scaffold.py
```

## Outputs

- `docs/results/cross-vendor-zero-shot-scaffold-mi300x.json`
- `docs/results/cross-vendor-zero-shot-scaffold-mi300x.md`
- `docs/results/cross-vendor-measured-mi300x-template.json` (placeholder
  capture sheet only, not measured evidence)

## What this is (and is not)

- **Is:** a reproducible zero-shot prediction scaffold producing per-operator,
  per-bucket top-k candidate configs for an unseen vendor label.
- **Is not:** a validated cross-vendor transfer result yet (no AMD measurements
  are used in this artifact). The measured-template JSON is not evidence and is
  rejected by the transfer evaluator.

## Next step to convert scaffold into result

Run the predicted candidates on real MI300X/MI250 hardware, write positive
measured `metric` and `latency_ms` rows to
`docs/results/cross-vendor-measured-mi300x.json`, then compute ranking transfer
metrics (Spearman rho, top-k overlap, latency regret).
