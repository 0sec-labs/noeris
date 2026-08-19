#!/usr/bin/env python3
"""Evaluate cross-vendor zero-shot transfer predictions against measured AMD runs."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from research_engine.benchmark_metadata import collect_environment
from research_engine.cross_vendor_transfer import (
    latency_regret,
    spearman_rank_correlation,
    top_k_hit_rate,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _contains_marker(value: object, markers: tuple[str, ...]) -> str | None:
    text = str(value or "").lower()
    return next((marker for marker in markers if marker in text), None)


def _positive_finite(row: dict, field: str, label: str) -> None:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} is missing numeric {field}") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} has non-positive {field}: {value}")


def validate_measured_artifact(measured: dict, *, measured_path: Path | None = None) -> dict:
    """Reject scaffold/template artifacts before computing paper-facing metrics."""

    artifact_markers = ("placeholder", "template", "scaffold", "deferred", "sample")
    row_markers = ("placeholder", "fill", "replace", "not measured", "scaffold", "template")
    reasons: list[str] = []

    if measured.get("is_measured_evidence") is False:
        reasons.append("is_measured_evidence=false")

    for field in ("artifact_type", "measurement_status", "status"):
        marker = _contains_marker(measured.get(field), artifact_markers)
        if marker:
            reasons.append(f"{field} contains {marker!r}")

    if measured_path is not None:
        marker = _contains_marker(measured_path.name, artifact_markers)
        if marker:
            reasons.append(f"measured path contains {marker!r}")

    if reasons:
        raise ValueError(
            "measured AMD artifact is not eligible for paper-facing evaluation: "
            + "; ".join(reasons)
        )

    measured_rows = measured.get("measured")
    if not isinstance(measured_rows, dict):
        raise ValueError("measured AMD artifact must contain a measured object")

    row_count = 0
    for operator, buckets in measured_rows.items():
        if not isinstance(buckets, dict):
            raise ValueError(f"measured.{operator} must be an object of buckets")
        for bucket, rows in buckets.items():
            if not isinstance(rows, list):
                raise ValueError(f"measured.{operator}.{bucket} must be a list")
            for idx, row in enumerate(rows):
                label = f"measured.{operator}.{bucket}[{idx}]"
                if not isinstance(row, dict):
                    raise ValueError(f"{label} must be an object")
                marker = _contains_marker(row.get("notes"), row_markers)
                if marker:
                    raise ValueError(f"{label} notes contain placeholder marker {marker!r}")
                _positive_finite(row, "metric", label)
                _positive_finite(row, "latency_ms", label)
                row_count += 1

    if row_count == 0:
        raise ValueError("measured AMD artifact contains no measurement rows")

    return {"status": "passed_real_measurement_checks", "row_count": row_count}


def _to_md(report: dict) -> str:
    lines = [
        "# Cross-Vendor Transfer Evaluation",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        f"Prediction artifact: `{report['prediction_artifact']}`",
        f"Measured artifact: `{report['measured_artifact']}`",
    ]
    validation = report.get("measurement_validation", {})
    if validation:
        lines.append(
            f"Measurement validation: `{validation.get('status', 'passed')}` "
            f"({validation.get('row_count', 0)} rows)"
        )
    lines += [
        "",
        "| Operator | Buckets | mean spearman | mean top-k hit | mean latency regret |",
        "|---|---:|---:|---:|---:|",
    ]
    for op, summary in sorted(report["operator_summary"].items()):
        lines.append(
            f"| {op} | {summary['bucket_count']} | {summary['mean_spearman']:.4f} | {summary['mean_topk_hit_rate']:.4f} | {summary['mean_latency_regret']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _sort_desc(rows: list[dict], key: str) -> list[dict]:
    return sorted(rows, key=lambda r: float(r.get(key, 0.0)), reverse=True)


def evaluate_transfer(*, prediction: dict, measured: dict, top_k: int) -> dict:
    predictions = prediction.get("predictions", {})
    measured_rows = measured.get("measured", {})
    out: dict[str, dict[str, dict]] = {}
    op_summary: dict[str, dict] = {}

    for operator, buckets in predictions.items():
        op_out: dict[str, dict] = {}
        spearmans: list[float] = []
        hits: list[float] = []
        regrets: list[float] = []

        for bucket, row in buckets.items():
            pred_top = row.get("target_predicted_top", [])
            pred_with_metric = [r for r in pred_top if r.get("config_id")]
            measured_bucket = measured_rows.get(operator, {}).get(bucket, [])
            measured_sorted = _sort_desc(measured_bucket, "metric")

            if len(pred_with_metric) < 2 or len(measured_sorted) < 2:
                continue

            measured_map = {m.get("config_id", ""): float(m.get("metric", 0.0)) for m in measured_sorted}
            shared = [p for p in pred_with_metric if p.get("config_id") in measured_map]
            if len(shared) < 2:
                continue

            pred_scores = [float(p.get("predicted_metric", 0.0)) for p in shared]
            meas_scores = [measured_map[p.get("config_id", "")] for p in shared]
            shared_ids = [p.get("config_id", "") for p in shared]
            measured_ranked_ids = [r.get("config_id", "") for r in measured_sorted if r.get("config_id") in set(shared_ids)]

            rho = spearman_rank_correlation(pred_scores, meas_scores)
            hit = top_k_hit_rate(shared_ids, measured_ranked_ids, k=top_k)

            pred_best_id = shared_ids[0]
            meas_best_id = measured_sorted[0].get("config_id", "")
            pred_best_ms = next(
                (float(m.get("latency_ms", 0.0)) for m in measured_sorted if m.get("config_id") == pred_best_id),
                0.0,
            )
            meas_best_ms = next(
                (float(m.get("latency_ms", 0.0)) for m in measured_sorted if m.get("config_id") == meas_best_id),
                0.0,
            )
            regret = latency_regret(pred_best_ms, meas_best_ms)

            op_out[bucket] = {
                "shared_count": len(shared),
                "spearman": rho,
                "topk_hit_rate": hit,
                "latency_regret": regret,
                "predicted_best_config_id": pred_best_id,
                "measured_best_config_id": meas_best_id,
            }
            spearmans.append(rho)
            hits.append(hit)
            regrets.append(regret)

        out[operator] = op_out
        op_summary[operator] = {
            "bucket_count": len(op_out),
            "mean_spearman": (sum(spearmans) / len(spearmans)) if spearmans else 0.0,
            "mean_topk_hit_rate": (sum(hits) / len(hits)) if hits else 0.0,
            "mean_latency_regret": (sum(regrets) / len(regrets)) if regrets else 0.0,
        }

    return {"by_operator": out, "operator_summary": op_summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-json", required=True)
    parser.add_argument("--measured-json", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-json", default="docs/results/cross-vendor-transfer-eval.json")
    parser.add_argument("--output-md", default="docs/results/cross-vendor-transfer-eval.md")
    args = parser.parse_args()

    prediction_path = Path(args.prediction_json)
    measured_path = Path(args.measured_json)
    prediction = _load_json(prediction_path)
    measured = _load_json(measured_path)
    try:
        measurement_validation = validate_measured_artifact(measured, measured_path=measured_path)
    except ValueError as exc:
        parser.error(str(exc))

    eval_out = evaluate_transfer(prediction=prediction, measured=measured, top_k=args.top_k)
    cmd = f"python scripts/cross_vendor_transfer_eval.py --prediction-json {args.prediction_json} --measured-json {args.measured_json}"
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": collect_environment(command=cmd),
        "prediction_artifact": str(prediction_path),
        "measured_artifact": str(measured_path),
        "measurement_validation": measurement_validation,
        "top_k": args.top_k,
        **eval_out,
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_to_md(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
