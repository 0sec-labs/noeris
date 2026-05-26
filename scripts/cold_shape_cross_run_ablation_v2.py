#!/usr/bin/env python3
"""Offline cold-shape cross-run learning ablation v2.

This script replays measured ConfigDatabase rows instead of launching new GPU
benchmarks. It holds out cold shape buckets, trains priors on the remaining
database rows, and compares:

- stateless_random: random ordering with no cross-run memory,
- database_seeded: config ordering by historical normalized performance,
- cost_model_ranking: a CostModel trained without held-out buckets.

Curated starter configs are excluded by default so they cannot dominate both
conditions. The output includes raw per-iteration histories for every
condition, seed, and held-out bucket.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics as stats
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_engine.benchmark_metadata import collect_environment
from research_engine.cost_model import CostModel, encode_features, extract_features
from research_engine.triton_operators import REGISTRY


DEFAULT_OPERATORS = ("rmsnorm", "softmax", "layernorm", "cross_entropy")
DEFAULT_HARDWARE = "NVIDIA A100-SXM4-80GB"
CONDITIONS = ("stateless_random", "database_seeded", "cost_model_ranking")


@dataclass(slots=True)
class BucketRecord:
    key: str
    operator: str
    bucket: str
    hardware: str
    shape: dict[str, Any]
    metric_name: str
    candidates: dict[str, dict[str, Any]]
    oracle_best_metric: float
    oracle_best_config_id: str
    shape_size_proxy: float


@dataclass(slots=True)
class FeatureKnnModel:
    rows: list[tuple[str, list[float], float]]
    means: list[float]
    scales: list[float]
    k: int = 5

    def predict(
        self,
        *,
        shape: dict[str, Any],
        config: dict[str, Any],
        hardware: str,
        operator: str,
    ) -> float:
        features = encode_features(
            extract_features(
                shape=shape,
                config=config,
                hardware=hardware,
                operator=operator,
            )
        )
        scaled = _scale_features(features, self.means, self.scales)
        same_operator = [
            (distance, metric)
            for row_operator, row_features, metric in self.rows
            if row_operator == operator
            for distance in [_squared_distance(scaled, row_features)]
        ]
        if not same_operator:
            same_operator = [
                (_squared_distance(scaled, row_features), metric)
                for _, row_features, metric in self.rows
            ]
        nearest = sorted(same_operator, key=lambda item: item[0])[: self.k]
        if not nearest:
            return 0.0
        weighted_sum = 0.0
        weight_total = 0.0
        for distance, metric in nearest:
            weight = 1.0 / (distance + 1e-9)
            weighted_sum += weight * metric
            weight_total += weight
        return weighted_sum / weight_total if weight_total else 0.0


@dataclass(slots=True)
class RankingModel:
    method: str
    cost_model: CostModel
    knn_model: FeatureKnnModel | None = None

    def predict(
        self,
        *,
        shape: dict[str, Any],
        config: dict[str, Any],
        hardware: str,
        operator: str,
    ) -> float:
        if self.method == "gradient_boosted_regressor":
            return self.cost_model.predict(
                shape=shape,
                config=config,
                hardware=hardware,
                operator=operator,
            )
        if self.knn_model is None:
            return self.cost_model.predict(
                shape=shape,
                config=config,
                hardware=hardware,
                operator=operator,
            )
        return self.knn_model.predict(
            shape=shape,
            config=config,
            hardware=hardware,
            operator=operator,
        )


def _stable_seed(*parts: object) -> int:
    text = ":".join(str(part) for part in parts)
    value = 0
    for ch in text:
        value = (value * 131 + ord(ch)) % (2**32)
    return value


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    return stats.median(values) if values else 0.0


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _shape_size_proxy(shape: dict[str, Any]) -> float:
    values = [
        int(v)
        for k, v in shape.items()
        if k != "bucket" and isinstance(v, int) and not isinstance(v, bool) and v > 0
    ]
    if not values:
        return 0.0
    return float(math.prod(values))


def _squared_distance(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _scale_features(features: list[float], means: list[float], scales: list[float]) -> list[float]:
    return [
        (value - means[idx]) / scales[idx] if scales[idx] > 0.0 else 0.0
        for idx, value in enumerate(features)
    ]


def _split_record_key(key: str) -> tuple[str, str, str] | None:
    parts = key.split(":", 2)
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def _curated_ids_by_operator(operators: list[str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for operator in operators:
        try:
            spec = REGISTRY.get(operator)
        except KeyError:
            out[operator] = set()
            continue
        out[operator] = {spec.config_id_fn(config) for config in spec.curated_configs}
    return out


def _metric_name(operator: str) -> str:
    try:
        return REGISTRY.get(operator).metric_name
    except KeyError:
        return "metric"


def _collect_bucket_records(
    *,
    raw_records: dict[str, dict[str, Any]],
    operators: list[str],
    hardware: str,
    curated_ids: dict[str, set[str]],
    exclude_curated: bool,
    min_candidates: int,
) -> tuple[list[BucketRecord], dict[str, int]]:
    buckets: list[BucketRecord] = []
    excluded_curated_rows = 0
    excluded_nonpositive_rows = 0

    for key, record in raw_records.items():
        parsed = _split_record_key(key)
        if parsed is None:
            continue
        operator, bucket, hw = parsed
        if operator not in operators or hw != hardware:
            continue

        candidates: dict[str, dict[str, Any]] = {}
        for row in record.get("results", []):
            if not row.get("correct"):
                continue
            cid = str(row.get("config_id", ""))
            if not cid:
                continue
            if exclude_curated and cid in curated_ids.get(operator, set()):
                excluded_curated_rows += 1
                continue
            metric = float(row.get("tflops") or row.get("gb_per_s") or 0.0)
            if metric <= 0.0:
                excluded_nonpositive_rows += 1
                continue
            existing = candidates.get(cid)
            if existing is None or metric > existing["metric"]:
                candidates[cid] = {
                    "config_id": cid,
                    "config": row.get("config", {}),
                    "metric": metric,
                }

        if len(candidates) < min_candidates:
            continue

        best = max(candidates.values(), key=lambda item: item["metric"])
        shape = dict(record.get("shape", {}))
        buckets.append(
            BucketRecord(
                key=key,
                operator=operator,
                bucket=bucket,
                hardware=hw,
                shape=shape,
                metric_name=_metric_name(operator),
                candidates=candidates,
                oracle_best_metric=float(best["metric"]),
                oracle_best_config_id=str(best["config_id"]),
                shape_size_proxy=_shape_size_proxy(shape),
            )
        )

    diagnostics = {
        "excluded_curated_rows": excluded_curated_rows,
        "excluded_nonpositive_rows": excluded_nonpositive_rows,
    }
    return buckets, diagnostics


def _select_heldout_buckets(
    buckets: list[BucketRecord],
    *,
    heldout_per_operator: int,
) -> list[BucketRecord]:
    by_operator: dict[str, list[BucketRecord]] = {}
    for bucket in buckets:
        by_operator.setdefault(bucket.operator, []).append(bucket)

    heldout: list[BucketRecord] = []
    for operator in sorted(by_operator):
        ranked = sorted(
            by_operator[operator],
            key=lambda b: (-b.shape_size_proxy, b.bucket),
        )
        heldout.extend(ranked[:heldout_per_operator])
    return heldout


def _build_training_db(
    *,
    raw_records: dict[str, dict[str, Any]],
    heldout_keys: set[str],
    operators: list[str],
    hardware: str,
    curated_ids: dict[str, set[str]],
    exclude_curated: bool,
) -> tuple[dict[str, dict[str, Any]], int]:
    training: dict[str, dict[str, Any]] = {}
    row_count = 0
    for key, record in raw_records.items():
        parsed = _split_record_key(key)
        if parsed is None:
            continue
        operator, _, hw = parsed
        if key in heldout_keys or operator not in operators or hw != hardware:
            continue
        rows = []
        for row in record.get("results", []):
            cid = str(row.get("config_id", ""))
            if exclude_curated and cid in curated_ids.get(operator, set()):
                continue
            metric = float(row.get("tflops") or row.get("gb_per_s") or 0.0)
            if not row.get("correct") or metric <= 0.0:
                continue
            rows.append(row)
        if rows:
            clone = dict(record)
            clone["results"] = rows
            training[key] = clone
            row_count += len(rows)
    return training, row_count


def _train_cost_model(training_records: dict[str, dict[str, Any]]) -> tuple[CostModel, dict[str, Any]]:
    model = CostModel()
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        tmp_path = Path(f.name)
        json.dump({"records": training_records}, f)
    try:
        train_result = model.train_from_databases([tmp_path])
    finally:
        tmp_path.unlink(missing_ok=True)
    return model, train_result


def _build_knn_model(training_records: dict[str, dict[str, Any]], *, k: int = 5) -> FeatureKnnModel:
    unscaled_rows: list[tuple[str, list[float], float]] = []
    for key, record in training_records.items():
        parsed = _split_record_key(key)
        if parsed is None:
            continue
        operator, _, hardware = parsed
        shape = record.get("shape", {})
        for row in record.get("results", []):
            if not row.get("correct"):
                continue
            metric = float(row.get("tflops") or row.get("gb_per_s") or 0.0)
            if metric <= 0.0:
                continue
            features = encode_features(
                extract_features(
                    shape=shape,
                    config=row.get("config", {}),
                    hardware=hardware,
                    operator=operator,
                )
            )
            unscaled_rows.append((operator, features, metric))

    if not unscaled_rows:
        return FeatureKnnModel(rows=[], means=[], scales=[], k=k)

    width = len(unscaled_rows[0][1])
    means = [
        _mean([features[idx] for _, features, _ in unscaled_rows])
        for idx in range(width)
    ]
    scales = []
    for idx in range(width):
        values = [features[idx] for _, features, _ in unscaled_rows]
        mean = means[idx]
        variance = _mean([(value - mean) ** 2 for value in values])
        scales.append(math.sqrt(variance) if variance > 0.0 else 1.0)

    scaled_rows = [
        (operator, _scale_features(features, means, scales), metric)
        for operator, features, metric in unscaled_rows
    ]
    return FeatureKnnModel(rows=scaled_rows, means=means, scales=scales, k=k)


def _build_ranking_model(
    training_records: dict[str, dict[str, Any]],
) -> tuple[RankingModel, dict[str, Any]]:
    cost_model, train_result = _train_cost_model(training_records)
    if cost_model.regressor is not None:
        return (
            RankingModel(method="gradient_boosted_regressor", cost_model=cost_model),
            {
                "method": "gradient_boosted_regressor",
                "cost_model_train_result": train_result,
            },
        )

    knn = _build_knn_model(training_records)
    return (
        RankingModel(method="feature_knn_fallback", cost_model=cost_model, knn_model=knn),
        {
            "method": "feature_knn_fallback",
            "cost_model_train_result": train_result,
            "fallback_reason": "CostModel regressor unavailable; using pure-Python feature kNN ranking.",
            "fallback_k": knn.k,
            "fallback_training_rows": len(knn.rows),
        },
    )


def _database_seed_scores(training_records: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    per_operator_values: dict[str, dict[str, list[float]]] = {}
    for key, record in training_records.items():
        parsed = _split_record_key(key)
        if parsed is None:
            continue
        operator, _, _ = parsed
        best = max(
            (
                float(row.get("tflops") or row.get("gb_per_s") or 0.0)
                for row in record.get("results", [])
                if row.get("correct")
            ),
            default=0.0,
        )
        if best <= 0.0:
            continue
        for row in record.get("results", []):
            if not row.get("correct"):
                continue
            cid = str(row.get("config_id", ""))
            metric = float(row.get("tflops") or row.get("gb_per_s") or 0.0)
            if cid and metric > 0.0:
                per_operator_values.setdefault(operator, {}).setdefault(cid, []).append(metric / best)

    return {
        operator: {cid: _mean(values) for cid, values in config_values.items()}
        for operator, config_values in per_operator_values.items()
    }


def _ordered_config_ids(
    *,
    bucket: BucketRecord,
    condition: str,
    seed: int,
    db_scores: dict[str, dict[str, float]],
    ranking_model: RankingModel,
) -> list[str]:
    ids = list(bucket.candidates)
    rng = random.Random(_stable_seed(condition, seed, bucket.key))
    rng.shuffle(ids)

    if condition == "stateless_random":
        return ids

    if condition == "database_seeded":
        scores = db_scores.get(bucket.operator, {})
        tie_breaker = {cid: idx for idx, cid in enumerate(ids)}
        return sorted(ids, key=lambda cid: (-scores.get(cid, -1.0), tie_breaker[cid]))

    if condition == "cost_model_ranking":
        tie_breaker = {cid: idx for idx, cid in enumerate(ids)}
        scored = []
        for cid in ids:
            candidate = bucket.candidates[cid]
            pred = ranking_model.predict(
                shape=bucket.shape,
                config=candidate["config"],
                hardware=bucket.hardware,
                operator=bucket.operator,
            )
            scored.append((cid, pred, tie_breaker[cid]))
        scored.sort(key=lambda item: (-item[1], item[2]))
        return [cid for cid, _, _ in scored]

    raise ValueError(f"unknown condition: {condition}")


def _replay_bucket(
    *,
    bucket: BucketRecord,
    condition: str,
    seed: int,
    iterations: int,
    configs_per_iteration: int,
    db_scores: dict[str, dict[str, float]],
    ranking_model: RankingModel,
) -> dict[str, Any]:
    order = _ordered_config_ids(
        bucket=bucket,
        condition=condition,
        seed=seed,
        db_scores=db_scores,
        ranking_model=ranking_model,
    )
    best_metric = 0.0
    best_config_id = ""
    history = []
    iteration_to_90pct: int | None = None

    for iteration in range(1, iterations + 1):
        start = (iteration - 1) * configs_per_iteration
        stop = start + configs_per_iteration
        evaluated = order[start:stop]
        iteration_metrics = [
            bucket.candidates[cid]["metric"]
            for cid in evaluated
            if cid in bucket.candidates
        ]
        if iteration_metrics:
            iter_best_metric = max(iteration_metrics)
            iter_best_idx = iteration_metrics.index(iter_best_metric)
            iter_best_id = evaluated[iter_best_idx]
            if iter_best_metric > best_metric:
                best_metric = iter_best_metric
                best_config_id = iter_best_id
        else:
            iter_best_metric = 0.0

        normalized = best_metric / bucket.oracle_best_metric if bucket.oracle_best_metric else 0.0
        regret = bucket.oracle_best_metric - best_metric
        if iteration_to_90pct is None and normalized >= 0.9:
            iteration_to_90pct = iteration

        history.append(
            {
                "iteration": iteration,
                "evaluated_config_ids": evaluated,
                "iteration_best_metric": _round(iter_best_metric),
                "best_so_far_metric": _round(best_metric),
                "best_so_far_config_id": best_config_id,
                "normalized_best": _round(normalized),
                "regret": _round(regret),
                "regret_pct": _round((regret / bucket.oracle_best_metric) * 100.0),
                "reached_90pct_best": normalized >= 0.9,
            }
        )

    final_regret = bucket.oracle_best_metric - best_metric
    return {
        "condition": condition,
        "seed": seed,
        "order": order,
        "iterations_to_90pct_best": iteration_to_90pct,
        "final_best_metric": _round(best_metric),
        "final_best_config_id": best_config_id,
        "final_normalized_best": _round(best_metric / bucket.oracle_best_metric),
        "final_regret": _round(final_regret),
        "final_regret_pct": _round((final_regret / bucket.oracle_best_metric) * 100.0),
        "history": history,
    }


def _summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    finals = [float(run["final_best_metric"]) for run in runs]
    normalized = [float(run["final_normalized_best"]) for run in runs]
    regrets = [float(run["final_regret"]) for run in runs]
    regret_pcts = [float(run["final_regret_pct"]) for run in runs]
    to_90 = [
        int(run["iterations_to_90pct_best"])
        for run in runs
        if run["iterations_to_90pct_best"] is not None
    ]
    return {
        "runs": len(runs),
        "mean_final_metric": _round(_mean(finals)),
        "median_final_metric": _round(_median(finals)),
        "mean_final_normalized_best": _round(_mean(normalized)),
        "median_final_normalized_best": _round(_median(normalized)),
        "mean_final_regret": _round(_mean(regrets)),
        "mean_final_regret_pct": _round(_mean(regret_pcts)),
        "median_iterations_to_90pct_best": _round(_median([float(v) for v in to_90]), 2)
        if to_90
        else None,
        "success_rate_90pct": _round(len(to_90) / len(runs) if runs else 0.0),
    }


def _to_md(report: dict[str, Any]) -> str:
    lines = [
        "# Cold-Shape Cross-Run Learning Ablation v2",
        "",
        f"Generated: {report['generated_at_utc']}",
        f"Database: `{report['db_path']}`",
        f"Hardware: `{report['hardware']}`",
        "",
        "## Protocol",
        "",
        f"- Iterations: `{report['protocol']['iterations']}`",
        f"- Configs per iteration: `{report['protocol']['configs_per_iteration']}`",
        f"- Seeds: `{report['protocol']['seeds']}`",
        f"- Curated configs excluded: `{str(report['protocol']['exclude_curated']).lower()}`",
        f"- Held-out policy: `{report['protocol']['heldout_policy']}`",
        "",
        "## Overall Summary",
        "",
        "| Condition | mean final normalized | mean final regret % | median iter to 90% | 90% success |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        summary = report["summary"][condition]
        iter90 = summary["median_iterations_to_90pct_best"]
        iter90_text = "n/a" if iter90 is None else f"{iter90:.2f}"
        lines.append(
            f"| {condition} | {summary['mean_final_normalized_best']:.4f} | "
            f"{summary['mean_final_regret_pct']:.2f} | {iter90_text} | "
            f"{summary['success_rate_90pct']:.2f} |"
        )

    lines += [
        "",
        "## Held-Out Buckets",
        "",
        "| Operator | Bucket | Candidates | Oracle best | Metric |",
        "|---|---|---:|---:|---|",
    ]
    for bucket in report["heldout_buckets"]:
        lines.append(
            f"| {bucket['operator']} | {bucket['bucket']} | "
            f"{bucket['candidate_count']} | {bucket['oracle_best_metric']:.4f} | "
            f"{bucket['metric_name']} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        report["interpretation"],
        "",
        "Raw per-iteration histories are stored in the JSON artifact under `by_bucket`.",
        "",
    ]
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    db_path = Path(args.db_path)
    payload = json.loads(db_path.read_text(encoding="utf-8"))
    raw_records = payload.get("records", {})
    operators = list(args.operators)
    curated_ids = _curated_ids_by_operator(operators)

    buckets, diagnostics = _collect_bucket_records(
        raw_records=raw_records,
        operators=operators,
        hardware=args.hardware,
        curated_ids=curated_ids,
        exclude_curated=args.exclude_curated,
        min_candidates=args.min_candidates,
    )
    if not buckets:
        raise RuntimeError("no eligible measured buckets after filtering")

    heldout = _select_heldout_buckets(
        buckets,
        heldout_per_operator=args.heldout_per_operator,
    )
    heldout_keys = {bucket.key for bucket in heldout}
    training_records, training_row_count = _build_training_db(
        raw_records=raw_records,
        heldout_keys=heldout_keys,
        operators=operators,
        hardware=args.hardware,
        curated_ids=curated_ids,
        exclude_curated=args.exclude_curated,
    )
    ranking_model, ranking_meta = _build_ranking_model(training_records)
    db_scores = _database_seed_scores(training_records)

    by_bucket: dict[str, Any] = {}
    all_runs_by_condition: dict[str, list[dict[str, Any]]] = {name: [] for name in CONDITIONS}

    for bucket in heldout:
        bucket_runs: dict[str, dict[str, Any]] = {}
        for condition in CONDITIONS:
            runs = []
            for seed in args.seeds:
                run = _replay_bucket(
                    bucket=bucket,
                    condition=condition,
                    seed=int(seed),
                    iterations=args.iterations,
                    configs_per_iteration=args.configs_per_iteration,
                    db_scores=db_scores,
                    ranking_model=ranking_model,
                )
                runs.append(run)
                all_runs_by_condition[condition].append(run)
            bucket_runs[condition] = {
                "summary": _summarize_runs(runs),
                "runs": runs,
            }

        by_bucket[bucket.key] = {
            "operator": bucket.operator,
            "bucket": bucket.bucket,
            "hardware": bucket.hardware,
            "shape": bucket.shape,
            "metric_name": bucket.metric_name,
            "candidate_count": len(bucket.candidates),
            "oracle_best_metric": _round(bucket.oracle_best_metric),
            "oracle_best_config_id": bucket.oracle_best_config_id,
            "shape_size_proxy": bucket.shape_size_proxy,
            "conditions": bucket_runs,
        }

    summary = {
        condition: _summarize_runs(runs)
        for condition, runs in all_runs_by_condition.items()
    }

    best_condition = max(
        CONDITIONS,
        key=lambda name: summary[name]["mean_final_normalized_best"],
    )
    if best_condition == "database_seeded":
        interpretation = (
            "The replay supports the cross-run memory claim under this v2 protocol: "
            "database-seeded transfer has the highest mean final normalized throughput "
            "on held-out cold buckets with curated starters removed."
        )
    else:
        interpretation = (
            "The replay does not validate cross-run memory as the dominant selector "
            "under this v2 protocol. With curated starters removed, "
            f"`{best_condition}` has the highest mean final normalized throughput; "
            "the paper should frame the persistent database as enabling replayable "
            "training data and selector inputs rather than claiming standalone "
            "compounding gains from database seeding."
        )

    command = " ".join(sys.argv)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "cold_shape_cross_run_ablation_v2",
        "db_path": str(db_path),
        "hardware": args.hardware,
        "environment": collect_environment(command=command),
        "protocol": {
            "operators": operators,
            "conditions": list(CONDITIONS),
            "iterations": args.iterations,
            "configs_per_iteration": args.configs_per_iteration,
            "seeds": [int(seed) for seed in args.seeds],
            "heldout_per_operator": args.heldout_per_operator,
            "heldout_policy": "largest_shape_size_proxy_per_operator",
            "exclude_curated": args.exclude_curated,
            "min_candidates_after_filter": args.min_candidates,
            "metric": "higher_is_better",
        },
        "training": {
            "records_after_holdout": len(training_records),
            "rows_after_holdout": training_row_count,
            "ranking_model": ranking_meta,
            **diagnostics,
        },
        "heldout_buckets": [
            {
                "key": bucket.key,
                "operator": bucket.operator,
                "bucket": bucket.bucket,
                "hardware": bucket.hardware,
                "shape": bucket.shape,
                "candidate_count": len(bucket.candidates),
                "oracle_best_metric": _round(bucket.oracle_best_metric),
                "oracle_best_config_id": bucket.oracle_best_config_id,
                "metric_name": bucket.metric_name,
                "shape_size_proxy": bucket.shape_size_proxy,
            }
            for bucket in heldout
        ],
        "summary": summary,
        "interpretation": interpretation,
        "by_bucket": by_bucket,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default=".noeris/cost-model-training.json")
    parser.add_argument("--hardware", default=DEFAULT_HARDWARE)
    parser.add_argument("--operators", nargs="+", default=list(DEFAULT_OPERATORS))
    parser.add_argument("--heldout-per-operator", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--configs-per-iteration", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--min-candidates", type=int, default=6)
    parser.add_argument(
        "--include-curated",
        action="store_false",
        dest="exclude_curated",
        help="Include curated starter configs. By default they are excluded.",
    )
    parser.set_defaults(exclude_curated=True)
    parser.add_argument(
        "--output-json",
        default="docs/results/cold-shape-cross-run-ablation-v2.json",
    )
    parser.add_argument(
        "--output-md",
        default="docs/results/cold-shape-cross-run-ablation-v2.md",
    )
    args = parser.parse_args()

    report = build_report(args)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_to_md(report), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("experiment", "summary", "interpretation")}, indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
