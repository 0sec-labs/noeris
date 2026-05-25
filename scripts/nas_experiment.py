#!/usr/bin/env python3
"""NAS-style experiment: sweep architecture configs and predict layer latency.

Uses the kernel-aware ArchitectureCostModel to predict end-to-end decoder
layer latency from architecture hyperparameters.  Key goals:

1. Compare known architectures (Gemma 4 E2B, 31B, Llama 3 8B).
2. Explore novel configs (wider FFN, narrower heads, etc.).
3. Test "kernel cliff" hypothesis: do tile-unaligned dimensions (e.g.
   hidden_dim=4000 vs 4096) cause measurable slowdowns?

Usage::

    python scripts/nas_experiment.py [--hardware a100|t4|h100]
    python scripts/nas_experiment.py --hardware a100 --json-output /tmp/nas-a100.json
    python scripts/nas_experiment.py --all-hardware
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

# Direct import to avoid research_engine.__init__ pulling in torch
import importlib.util
_mod_path = Path(__file__).resolve().parent.parent / "src" / "research_engine" / "arch_cost_model.py"
_spec = importlib.util.spec_from_file_location("arch_cost_model", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ArchitectureCostModel = _mod.ArchitectureCostModel
_tile_efficiency = _mod._tile_efficiency


# -----------------------------------------------------------------------
# Architecture configs
# -----------------------------------------------------------------------

KNOWN_CONFIGS = [
    {
        "name": "gemma4_e2b",
        "hidden_dim": 1536, "num_heads": 8, "num_kv_heads": 1,
        "head_dim": 256, "ffn_dim": 6144, "seq_len": 2048, "batch_size": 1,
        "use_qk_norm": True, "window_size": 512,
    },
    {
        "name": "gemma4_31b",
        "hidden_dim": 5376, "num_heads": 32, "num_kv_heads": 16,
        "head_dim": 256, "ffn_dim": 21504, "seq_len": 2048, "batch_size": 1,
        "use_qk_norm": True, "window_size": 1024,
    },
    {
        "name": "llama3_8b",
        "hidden_dim": 4096, "num_heads": 32, "num_kv_heads": 8,
        "head_dim": 128, "ffn_dim": 14336, "seq_len": 2048, "batch_size": 1,
        "use_qk_norm": False, "window_size": None,
    },
    {
        "name": "llama3_70b",
        "hidden_dim": 8192, "num_heads": 64, "num_kv_heads": 8,
        "head_dim": 128, "ffn_dim": 28672, "seq_len": 2048, "batch_size": 1,
        "use_qk_norm": False, "window_size": None,
    },
]

NOVEL_CONFIGS = [
    {
        "name": "optimal_2b",
        "hidden_dim": 2048, "num_heads": 16, "num_kv_heads": 2,
        "head_dim": 128, "ffn_dim": 8192, "seq_len": 2048, "batch_size": 1,
        "use_qk_norm": True, "window_size": 1024,
    },
    {
        "name": "wide_ffn_2b",
        "hidden_dim": 1536, "num_heads": 8, "num_kv_heads": 1,
        "head_dim": 256, "ffn_dim": 8192, "seq_len": 2048, "batch_size": 1,
        "use_qk_norm": True, "window_size": 1024,
    },
    {
        "name": "narrow_heads",
        "hidden_dim": 2048, "num_heads": 32, "num_kv_heads": 4,
        "head_dim": 64, "ffn_dim": 8192, "seq_len": 2048, "batch_size": 1,
        "use_qk_norm": True, "window_size": 1024,
    },
    {
        "name": "deep_narrow",
        "hidden_dim": 1024, "num_heads": 8, "num_kv_heads": 1,
        "head_dim": 128, "ffn_dim": 4096, "seq_len": 2048, "batch_size": 1,
        "use_qk_norm": True, "window_size": 512,
    },
    {
        "name": "big_head_gqa",
        "hidden_dim": 2048, "num_heads": 8, "num_kv_heads": 1,
        "head_dim": 256, "ffn_dim": 8192, "seq_len": 2048, "batch_size": 1,
        "use_qk_norm": True, "window_size": 1024,
    },
]

SUPPORTED_HARDWARE = ("a100", "t4", "h100")

QUALITY_CONSTRAINTS = {
    "min_hidden_dim": 2048,
    "min_param_proxy_m": 16.0,
    "min_ffn_ratio": 3.75,
    "max_ffn_ratio": 5.5,
}


def _round_to_multiple(value: float, multiple: int) -> int:
    return int(round(value / multiple) * multiple)


def generate_candidate_configs() -> list[dict]:
    """Generate deterministic architecture candidates for constrained ranking."""
    configs: list[dict] = []
    seen_names: set[str] = set()
    for hidden_dim in (1536, 2048, 2560, 3072, 4096):
        for head_dim in (128, 256):
            if hidden_dim % head_dim != 0:
                continue
            num_heads = hidden_dim // head_dim
            for kv_group in (4,):
                num_kv_heads = max(1, num_heads // kv_group)
                if num_heads % num_kv_heads != 0:
                    continue
                for ffn_ratio_label, ffn_ratio in (("4p0", 4.0), ("5p33", 16.0 / 3.0)):
                    ffn_dim = _round_to_multiple(hidden_dim * ffn_ratio, 128)
                    for window_size in (512, 1024):
                        name = (
                            f"gen_h{hidden_dim}_hd{head_dim}_gqa{kv_group}_"
                            f"ffn{ffn_ratio_label}_w{window_size}"
                        )
                        if name in seen_names:
                            continue
                        seen_names.add(name)
                        configs.append(
                            {
                                "name": name,
                                "hidden_dim": hidden_dim,
                                "num_heads": num_heads,
                                "num_kv_heads": num_kv_heads,
                                "head_dim": head_dim,
                                "ffn_dim": ffn_dim,
                                "seq_len": 2048,
                                "batch_size": 1,
                                "use_qk_norm": True,
                                "window_size": window_size,
                            }
                        )
    return configs


def all_candidate_configs() -> list[dict]:
    """Known models, seed hand-written candidates, and generated candidates."""
    return KNOWN_CONFIGS + NOVEL_CONFIGS + generate_candidate_configs()


def _candidate_source(name: str) -> str:
    known = {cfg["name"] for cfg in KNOWN_CONFIGS}
    seed = {cfg["name"] for cfg in NOVEL_CONFIGS}
    if name in known:
        return "known_model"
    if name in seed:
        return "seed_candidate"
    return "generated_candidate"


def _candidate_param_proxy_m(config: dict) -> float:
    return config["hidden_dim"] * config["ffn_dim"] / 1e6


def _quality_proxy(config: dict) -> float:
    """Size/capacity proxy used only to constrain latency ranking."""
    param_proxy = _candidate_param_proxy_m(config)
    hidden_factor = min(1.0, config["hidden_dim"] / QUALITY_CONSTRAINTS["min_hidden_dim"])
    ffn_ratio = config["ffn_dim"] / config["hidden_dim"]
    ratio_factor = min(1.0, ffn_ratio / QUALITY_CONSTRAINTS["min_ffn_ratio"])
    return round(param_proxy * hidden_factor * ratio_factor, 6)


def _constraint_report(config: dict) -> dict:
    ffn_ratio = config["ffn_dim"] / config["hidden_dim"]
    param_proxy = _candidate_param_proxy_m(config)
    failed = []
    if config["hidden_dim"] < QUALITY_CONSTRAINTS["min_hidden_dim"]:
        failed.append("hidden_dim_below_floor")
    if param_proxy < QUALITY_CONSTRAINTS["min_param_proxy_m"]:
        failed.append("param_proxy_below_floor")
    if ffn_ratio < QUALITY_CONSTRAINTS["min_ffn_ratio"]:
        failed.append("ffn_ratio_below_floor")
    if ffn_ratio > QUALITY_CONSTRAINTS["max_ffn_ratio"]:
        failed.append("ffn_ratio_above_ceiling")
    return {
        "passes": not failed,
        "failed": failed,
        "param_proxy_m": round(param_proxy, 6),
        "ffn_ratio": round(ffn_ratio, 6),
        "quality_proxy": _quality_proxy(config),
    }


def _candidate_catalog(configs: list[dict]) -> list[dict]:
    return [
        {
            "name": cfg["name"],
            "source": _candidate_source(cfg["name"]),
            "config": _prediction_config(cfg),
            "constraints": _constraint_report(cfg),
        }
        for cfg in configs
    ]


def _prediction_config(config: dict) -> dict:
    return {key: value for key, value in config.items() if key != "name"}


def _ranking_rows(ranked: list[dict]) -> list[dict]:
    return [
        {
            "rank": row["rank"],
            "name": row["name"],
            "source": row.get("source", _candidate_source(row["name"])),
            "total_ms": row["total_ms"],
            "ms_per_mparam_proxy": row["ms_per_mparam_proxy"],
            "bottleneck": row["bottleneck"],
            "quality_proxy": row.get("quality_proxy"),
            "constraint_passes": row.get("constraints", {}).get("passes"),
        }
        for row in ranked
    ]


def _rank_constrained(comparisons: list[dict]) -> list[dict]:
    rows = [
        row
        for row in comparisons
        if row["constraints"]["passes"]
    ]
    rows = sorted(rows, key=lambda row: (row["total_ms"], -row["quality_proxy"], row["name"]))
    return [
        {
            "rank": index,
            "name": row["name"],
            "source": row["source"],
            "total_ms": row["total_ms"],
            "quality_proxy": row["quality_proxy"],
            "param_proxy_m": row["constraints"]["param_proxy_m"],
            "bottleneck": row["bottleneck"],
        }
        for index, row in enumerate(rows, start=1)
    ]


def build_report(
    model: ArchitectureCostModel,
    *,
    candidate_configs: list[dict] | None = None,
    calibration: dict | None = None,
) -> dict:
    """Build a machine-readable NAS report for one hardware profile."""
    all_configs = candidate_configs or all_candidate_configs()

    comparisons = []
    for cfg in all_configs:
        pred = model.predict_layer_ms(_prediction_config(cfg))
        pk = pred["per_kernel"]
        constraints = _constraint_report(cfg)
        comparisons.append({
            "name": cfg["name"],
            "source": _candidate_source(cfg["name"]),
            "config": _prediction_config(cfg),
            "total_ms": pred["total_ms"],
            "bottleneck": pred["bottleneck"],
            "per_kernel": pred["per_kernel"],
            "attention_ms": pk.get("attention", 0),
            "mlp_ms": pk.get("geglu_mlp", 0),
            "matmul_ms": pk.get("qkv_projection", 0) + pk.get("output_projection", 0),
            "ms_per_mparam_proxy": pred["total_ms"] / constraints["param_proxy_m"],
            "quality_proxy": constraints["quality_proxy"],
            "constraints": constraints,
            "tile_penalties": pred["tile_penalties"],
        })

    hidden_base = {
        "num_heads": 32, "num_kv_heads": 8, "head_dim": 128,
        "ffn_dim": 14336, "seq_len": 2048, "batch_size": 1,
        "use_qk_norm": False, "window_size": None,
    }
    ffn_base = {
        "hidden_dim": 2048, "num_heads": 16, "num_kv_heads": 2,
        "head_dim": 128, "seq_len": 2048, "batch_size": 1,
        "use_qk_norm": True, "window_size": 1024,
    }
    hidden_sweep = model.sweep_dimension(
        hidden_base,
        "hidden_dim",
        list(range(3840, 4352, 32)),
    )
    ffn_sweep = model.sweep_dimension(
        ffn_base,
        "ffn_dim",
        list(range(7936, 8448, 64)),
    )

    fastest = model.rank_configs(all_configs, metric="total_ms")
    most_efficient = model.rank_configs(all_configs, metric="ms_per_mparam_proxy")
    fastest_by_name = {row["name"]: row for row in fastest}
    for row in fastest:
        row["source"] = _candidate_source(row["name"])
        row["constraints"] = _constraint_report(row["config"])
        row["quality_proxy"] = row["constraints"]["quality_proxy"]
    for row in most_efficient:
        row["source"] = _candidate_source(row["name"])
        row["constraints"] = _constraint_report(row["config"])
        row["quality_proxy"] = row["constraints"]["quality_proxy"]

    constrained = _rank_constrained(comparisons)
    source_counts: dict[str, int] = {}
    for cfg in all_configs:
        source = _candidate_source(cfg["name"])
        source_counts[source] = source_counts.get(source, 0) + 1

    return {
        "schema_version": 1,
        "experiment": "kernel_aware_nas",
        "hardware": model.hardware,
        "candidate_count": len(all_configs),
        "candidate_source_counts": source_counts,
        "quality_constraints": QUALITY_CONSTRAINTS,
        "calibration": calibration or {"status": "not_requested"},
        "comparison": comparisons,
        "rankings": {
            "total_ms": _ranking_rows(fastest),
            "ms_per_mparam_proxy": _ranking_rows(most_efficient),
            "quality_constrained_latency": constrained,
        },
        "kernel_cliffs": {
            "hidden_dim": {
                "base_config": hidden_base,
                "values": hidden_sweep,
            },
            "ffn_dim": {
                "base_config": ffn_base,
                "values": ffn_sweep,
            },
        },
        "summary": {
            "fastest_config": fastest[0]["name"],
            "fastest_total_ms": fastest[0]["total_ms"],
            "most_efficient_config": most_efficient[0]["name"],
            "most_efficient_ms_per_mparam_proxy": most_efficient[0]["ms_per_mparam_proxy"],
            "quality_constrained_fastest_config": constrained[0]["name"] if constrained else None,
            "quality_constrained_fastest_total_ms": constrained[0]["total_ms"] if constrained else None,
            "quality_constrained_candidate_count": len(constrained),
            "deep_narrow_unconstrained_rank": fastest_by_name["deep_narrow"]["rank"],
        },
    }


def build_calibration_metadata(path: Path | None) -> dict:
    """Summarize and derive practical A100 profile overrides from persisted data."""
    if path is None:
        return {"status": "disabled", "profile_overrides": {}}
    if not path.exists():
        return {
            "status": "unavailable",
            "source_path": str(path),
            "profile_overrides": {},
        }

    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", {})
    operator_hardware: dict[str, dict[str, list[float]]] = {}
    for key, record in records.items():
        parts = key.split(":")
        operator = parts[0] if parts else record.get("operator", "unknown")
        hardware_label = record.get("shape_key", key).split(":")[-1]
        value = float(record.get("best_tflops", 0.0) or 0.0)
        operator_hardware.setdefault(operator, {}).setdefault(hardware_label, []).append(value)

    profile_overrides: dict[str, dict[str, float]] = {}
    a100_label = "NVIDIA A100-SXM4-80GB"
    a100_values = {
        operator: values_by_hw.get(a100_label, [])
        for operator, values_by_hw in operator_hardware.items()
    }
    a100_override: dict[str, float] = {}
    if a100_values.get("rmsnorm"):
        a100_override["rmsnorm_gbps"] = round(max(a100_values["rmsnorm"]), 4)
    if a100_values.get("matmul"):
        a100_override["matmul_tflops"] = round(max(a100_values["matmul"]), 4)
    if a100_values.get("attention"):
        a100_override["attention_tflops"] = round(max(a100_values["attention"]), 4)
    if a100_override:
        profile_overrides["a100"] = a100_override

    summary = {}
    for operator, values_by_hw in sorted(operator_hardware.items()):
        summary[operator] = {}
        for hardware_label, values in sorted(values_by_hw.items()):
            summary[operator][hardware_label] = {
                "record_count": len(values),
                "median_best_metric": round(statistics.median(values), 4),
                "max_best_metric": round(max(values), 4),
            }

    return {
        "status": "loaded",
        "source_path": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "profile_overrides": profile_overrides,
        "summary": summary,
        "notes": [
            "A100 rmsnorm/matmul/attention constants are overridden when matching persisted records exist.",
            "Other hardware profiles remain hardcoded proxy constants until measured artifacts are available.",
        ],
    }


def _model_for_hardware(hardware: str, calibration: dict) -> ArchitectureCostModel:
    overrides = calibration.get("profile_overrides", {}).get(hardware, {})
    return ArchitectureCostModel(hardware=hardware, profile_overrides=overrides)


def build_multi_hardware_report(
    *,
    calibration_db: Path | None = Path(".noeris/cost-model-training.json"),
) -> dict:
    """Build one deterministic pack containing A100, T4, and H100 NAS reports."""
    calibration = build_calibration_metadata(calibration_db)
    candidates = all_candidate_configs()
    reports = {
        hardware: build_report(
            _model_for_hardware(hardware, calibration),
            candidate_configs=candidates,
            calibration=calibration,
        )
        for hardware in SUPPORTED_HARDWARE
    }
    summary = {
        hardware: {
            "fastest_config": report["summary"]["fastest_config"],
            "quality_constrained_fastest_config": report["summary"][
                "quality_constrained_fastest_config"
            ],
            "quality_constrained_candidate_count": report["summary"][
                "quality_constrained_candidate_count"
            ],
        }
        for hardware, report in reports.items()
    }
    return {
        "schema_version": 1,
        "experiment": "kernel_aware_nas_multi_hardware",
        "hardware_profiles": list(SUPPORTED_HARDWARE),
        "candidate_count": len(candidates),
        "candidate_catalog": _candidate_catalog(candidates),
        "quality_constraints": QUALITY_CONSTRAINTS,
        "calibration": calibration,
        "reports": reports,
        "summary": summary,
        "limitations": [
            "This is a latency proxy, not a measured model-quality or accuracy estimate.",
            "Quality constraints are size/capacity proxies until real training or eval measurements are added.",
            "Only A100 constants are calibrated from local persisted records when available.",
        ],
    }


def write_report(report: dict, output_path: Path) -> None:
    """Write a deterministic JSON artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_multi_hardware_markdown(pack: dict) -> str:
    lines = [
        "# Kernel-Aware NAS Multi-Hardware Pack",
        "",
        "This artifact is a deterministic latency proxy. It does not measure model quality, perplexity, or downstream accuracy.",
        "",
        f"Candidate count: `{pack['candidate_count']}`",
        f"Calibration status: `{pack['calibration']['status']}`",
        "",
        "| Hardware | unconstrained fastest | constrained fastest | constrained candidates |",
        "|---|---|---|---:|",
    ]
    for hardware in pack["hardware_profiles"]:
        row = pack["summary"][hardware]
        lines.append(
            f"| {hardware.upper()} | {row['fastest_config']} | "
            f"{row['quality_constrained_fastest_config']} | "
            f"{row['quality_constrained_candidate_count']} |"
        )
    lines.extend([
        "",
        "## Constraints",
        "",
        f"- Minimum hidden dimension: `{pack['quality_constraints']['min_hidden_dim']}`",
        f"- Minimum parameter proxy: `{pack['quality_constraints']['min_param_proxy_m']}M`",
        f"- FFN ratio range: `{pack['quality_constraints']['min_ffn_ratio']}` to `{pack['quality_constraints']['max_ffn_ratio']}`",
        "",
        "## Limitations",
        "",
    ])
    for item in pack["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def run_comparison(model: ArchitectureCostModel) -> None:
    """Compare all configs and print a table."""
    all_configs = KNOWN_CONFIGS + NOVEL_CONFIGS

    print(f"\n{'='*90}")
    print(f"  Layer Latency Predictions  ({model.hardware.upper()})")
    print(f"{'='*90}")
    header = f"{'Config':20s} {'Total ms':>9s} {'Bottleneck':>18s} {'Attn ms':>8s} {'MLP ms':>8s} {'MatMul ms':>9s}"
    print(header)
    print("-" * 90)

    results = []
    for cfg in all_configs:
        name = cfg["name"]
        pred = model.predict_layer_ms(_prediction_config(cfg))

        pk = pred["per_kernel"]
        attn_ms = pk.get("attention", 0)
        mlp_ms = pk.get("geglu_mlp", 0)
        matmul_ms = pk.get("qkv_projection", 0) + pk.get("output_projection", 0)

        print(f"{name:20s} {pred['total_ms']:9.3f} {pred['bottleneck']:>18s} "
              f"{attn_ms:8.3f} {mlp_ms:8.3f} {matmul_ms:9.3f}")
        results.append((name, pred))

    # Efficiency-normalized: ms per parameter proxy (hidden_dim * ffn_dim)
    print(f"\n{'='*90}")
    print("  Efficiency: ms per M-params proxy  (hidden * ffn / 1e6)")
    print(f"{'='*90}")
    for cfg in all_configs:
        name = cfg["name"]
        pred = [r for n, r in results if n == name][0]
        param_proxy = cfg["hidden_dim"] * cfg["ffn_dim"] / 1e6
        eff = pred["total_ms"] / param_proxy
        print(f"  {name:20s}  {eff:.4f} ms/M-param-proxy  "
              f"(tile_eff D={_tile_efficiency(cfg['hidden_dim']):.2f} "
              f"FFN={_tile_efficiency(cfg['ffn_dim']):.2f})")

    print(f"\n{'='*90}")
    print("  Fastest-first NAS ranking")
    print(f"{'='*90}")
    for row in model.rank_configs(all_configs):
        print(f"  #{row['rank']:02d} {row['name']:20s}  {row['total_ms']:7.3f} ms  "
              f"bottleneck={row['bottleneck']}")


def run_kernel_cliff_test(model: ArchitectureCostModel) -> None:
    """Test whether tile-unaligned dimensions cause performance cliffs."""
    print(f"\n{'='*90}")
    print("  Kernel Cliff Test: hidden_dim sweep around 4096")
    print(f"{'='*90}")

    base = {
        "num_heads": 32, "num_kv_heads": 8, "head_dim": 128,
        "ffn_dim": 14336, "seq_len": 2048, "batch_size": 1,
        "use_qk_norm": False, "window_size": None,
    }

    dims = list(range(3840, 4352, 32))
    results = model.sweep_dimension(base, "hidden_dim", dims)

    print(f"  {'hidden_dim':>10s} {'total_ms':>9s} {'aligned':>8s} {'tile_eff':>9s} {'bottleneck':>18s}")
    print("  " + "-" * 60)
    for r in results:
        eff = r["tile_efficiency"]
        aligned = "yes" if r["aligned_128"] else "NO"
        marker = " ***" if not r["aligned_128"] else ""
        print(f"  {r['hidden_dim']:10d} {r['total_ms']:9.3f} {aligned:>8s} "
              f"{eff:9.3f} {r['bottleneck']:>18s}{marker}")

    # Also sweep ffn_dim
    print(f"\n  Kernel Cliff Test: ffn_dim sweep around 8192")
    print("  " + "-" * 60)

    base2 = {
        "hidden_dim": 2048, "num_heads": 16, "num_kv_heads": 2,
        "head_dim": 128, "seq_len": 2048, "batch_size": 1,
        "use_qk_norm": True, "window_size": 1024,
    }

    ffn_dims = list(range(7936, 8448, 64))
    results2 = model.sweep_dimension(base2, "ffn_dim", ffn_dims)

    print(f"  {'ffn_dim':>10s} {'total_ms':>9s} {'aligned':>8s} {'tile_eff':>9s}")
    print("  " + "-" * 60)
    for r in results2:
        eff = r["tile_efficiency"]
        aligned = "yes" if r["aligned_128"] else "NO"
        marker = " ***" if not r["aligned_128"] else ""
        print(f"  {r['ffn_dim']:10d} {r['total_ms']:9.3f} {aligned:>8s} {eff:9.3f}{marker}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NAS architecture latency experiment")
    parser.add_argument("--hardware", default="a100", choices=["a100", "t4", "h100"],
                        help="Target GPU for predictions (default: a100)")
    parser.add_argument("--all-hardware", action="store_true",
                        help="Write one deterministic A100/T4/H100 NAS artifact pack")
    parser.add_argument("--json-output", type=Path,
                        help="Optional path for a machine-readable JSON report")
    parser.add_argument("--md-output", type=Path,
                        help="Optional path for a markdown summary")
    parser.add_argument("--calibration-db", type=Path,
                        default=Path(".noeris/cost-model-training.json"),
                        help="Optional persisted Noeris training DB for profile calibration")
    args = parser.parse_args()

    if args.all_hardware:
        report = build_multi_hardware_report(calibration_db=args.calibration_db)
        json_output = args.json_output or Path("docs/results/kernel-aware-nas-multihardware.json")
        md_output = args.md_output or json_output.with_suffix(".md")
        write_report(report, json_output)
        md_output.parent.mkdir(parents=True, exist_ok=True)
        md_output.write_text(render_multi_hardware_markdown(report), encoding="utf-8")
        print(f"Wrote JSON artifact: {json_output}")
        print(f"Wrote markdown summary: {md_output}")
        return

    model = ArchitectureCostModel(hardware=args.hardware)
    report = build_report(model)

    run_comparison(model)
    run_kernel_cliff_test(model)

    print(f"\n{'='*90}")
    print("  Key takeaways:")
    print("  - MLP (GeGLU) dominates for large models; attention dominates for long seq + many heads")
    print("  - Tile-unaligned dims (not multiple of 128) pay a measurable penalty")
    print("  - GQA (low num_kv_heads) saves attention time but QKV projection is still large")
    print("  - NAS should prefer hidden_dim, ffn_dim that are multiples of 128 (or 256)")
    print(f"{'='*90}\n")

    if args.json_output:
        write_report(report, args.json_output)
        print(f"Wrote JSON artifact: {args.json_output}")


if __name__ == "__main__":
    main()
