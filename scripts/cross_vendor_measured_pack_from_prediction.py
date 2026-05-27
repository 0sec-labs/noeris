#!/usr/bin/env python3
"""Create a measured-results template from cross-vendor prediction artifact."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-json",
        default="docs/results/cross-vendor-zero-shot-scaffold-mi300x-v2.json",
    )
    parser.add_argument(
        "--output-json",
        default="docs/results/cross-vendor-measured-mi300x-template.json",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)

    prediction_path = Path(args.prediction_json)
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    predictions = payload.get("predictions", {})

    measured: dict[str, dict[str, list[dict]]] = {}
    for operator, buckets in predictions.items():
        op_rows: dict[str, list[dict]] = {}
        for bucket, entry in buckets.items():
            rows = []
            for pred in entry.get("target_predicted_top", [])[: args.top_k]:
                cid = pred.get("config_id", "")
                if not cid:
                    continue
                rows.append(
                    {
                        "config_id": cid,
                        "metric": 0.0,
                        "latency_ms": 0.0,
                        "notes": "placeholder: replace with measured AMD metric and latency",
                    }
                )
            if rows:
                op_rows[bucket] = rows
        measured[operator] = op_rows

    out = {
        "artifact_type": "cross_vendor_measured_template",
        "measurement_status": "placeholder_not_measured",
        "is_measured_evidence": False,
        "hardware_access": "deferred_no_amd_hardware_in_repo_context",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_from_prediction": str(prediction_path),
        "instructions": (
            "Run the listed candidates on real MI300X/MI250 hardware, replace "
            "placeholder metric/latency rows with positive measured values, "
            "and save that as docs/results/cross-vendor-measured-mi300x.json "
            "before running cross_vendor_transfer_eval.py for paper-facing claims."
        ),
        "measured": measured,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
