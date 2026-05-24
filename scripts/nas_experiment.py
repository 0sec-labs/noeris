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
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Direct import to avoid research_engine.__init__ pulling in torch
import importlib.util
_mod_path = Path(__file__).resolve().parent.parent / "src" / "research_engine" / "arch_cost_model.py"
_spec = importlib.util.spec_from_file_location("arch_cost_model", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ArchitectureCostModel = _mod.ArchitectureCostModel
generate_nas_candidates = _mod.generate_nas_candidates
_tile_efficiency = _mod._tile_efficiency

_meta_path = Path(__file__).resolve().parent.parent / "src" / "research_engine" / "benchmark_metadata.py"
_meta_spec = importlib.util.spec_from_file_location("benchmark_metadata", _meta_path)
_meta_mod = importlib.util.module_from_spec(_meta_spec)
_meta_spec.loader.exec_module(_meta_mod)
collect_environment = _meta_mod.collect_environment


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


def _prediction_config(config: dict) -> dict:
    return {key: value for key, value in config.items() if key != "name"}


def _ranking_rows(ranked: list[dict]) -> list[dict]:
    return [
        {
            "rank": row["rank"],
            "name": row["name"],
            "total_ms": row["total_ms"],
            "ms_per_mparam_proxy": row["ms_per_mparam_proxy"],
            "bottleneck": row["bottleneck"],
        }
        for row in ranked
    ]


def generated_candidate_configs() -> list[dict]:
    """Generate a compact 2B-class candidate space for kernel-aware NAS."""
    base = {
        "hidden_dim": 2048,
        "num_heads": 16,
        "num_kv_heads": 2,
        "head_dim": 128,
        "ffn_dim": 8192,
        "seq_len": 2048,
        "batch_size": 1,
        "use_qk_norm": True,
        "window_size": 1024,
    }
    return generate_nas_candidates(
        base,
        hidden_dims=[1536, 2048, 2560],
        head_dims=[64, 128, 256],
        ffn_ratios=[3.0, 4.0, 5.333],
        kv_head_counts=[1, 2, 4, 8],
        window_sizes=[512, 1024, None],
        qk_norm_options=[True, False],
    )


def build_report(model: ArchitectureCostModel) -> dict:
    """Build a machine-readable NAS report for one hardware profile."""
    all_configs = KNOWN_CONFIGS + NOVEL_CONFIGS

    comparisons = []
    for cfg in all_configs:
        pred = model.predict_layer_ms(_prediction_config(cfg))
        pk = pred["per_kernel"]
        param_proxy_m = cfg["hidden_dim"] * cfg["ffn_dim"] / 1e6
        comparisons.append({
            "name": cfg["name"],
            "config": _prediction_config(cfg),
            "total_ms": pred["total_ms"],
            "bottleneck": pred["bottleneck"],
            "per_kernel": pred["per_kernel"],
            "attention_ms": pk.get("attention", 0),
            "mlp_ms": pk.get("geglu_mlp", 0),
            "matmul_ms": pk.get("qkv_projection", 0) + pk.get("output_projection", 0),
            "ms_per_mparam_proxy": pred["total_ms"] / param_proxy_m,
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
    generated = generated_candidate_configs()
    generated_fastest = model.rank_configs(generated, metric="total_ms")
    generated_efficient = model.rank_configs(generated, metric="ms_per_mparam_proxy")

    return {
        "schema_version": 1,
        "experiment": "kernel_aware_nas",
        "environment": collect_environment(
            command=f"python scripts/nas_experiment.py --hardware {model.hardware}",
        ),
        "hardware": model.hardware,
        "candidate_count": len(all_configs),
        "generated_candidate_count": len(generated),
        "comparison": comparisons,
        "rankings": {
            "total_ms": _ranking_rows(fastest),
            "ms_per_mparam_proxy": _ranking_rows(most_efficient),
        },
        "generated_search": {
            "total_ms_top": _ranking_rows(generated_fastest[:25]),
            "ms_per_mparam_proxy_top": _ranking_rows(generated_efficient[:25]),
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
            "most_efficient_ms_per_mparam_proxy": (
                most_efficient[0]["ms_per_mparam_proxy"]
            ),
            "fastest_generated_config": generated_fastest[0]["name"],
            "fastest_generated_total_ms": generated_fastest[0]["total_ms"],
        },
    }


def write_report(report: dict, output_path: Path) -> None:
    """Write a deterministic JSON artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def run_generated_search(model: ArchitectureCostModel, top_n: int = 10) -> None:
    """Generate and rank a broader kernel-aware architecture candidate set."""
    generated = generated_candidate_configs()
    ranked = model.rank_configs(generated)

    print(f"\n{'='*90}")
    print(f"  Generated NAS candidates: top {top_n} of {len(generated)}")
    print(f"{'='*90}")
    print(f"  {'Rank':>4s} {'Config':38s} {'Total ms':>9s} {'Eff':>8s} {'Bottleneck':>14s}")
    print("  " + "-" * 82)
    for row in ranked[:top_n]:
        print(f"  #{row['rank']:02d}  {row['name'][:38]:38s} "
              f"{row['total_ms']:9.3f} {row['ms_per_mparam_proxy']:8.4f} "
              f"{row['bottleneck']:>14s}")


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
    parser.add_argument("--json-output", type=Path,
                        help="Optional path for a machine-readable JSON report")
    args = parser.parse_args()

    model = ArchitectureCostModel(hardware=args.hardware)
    report = build_report(model)

    run_comparison(model)
    run_generated_search(model)
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
