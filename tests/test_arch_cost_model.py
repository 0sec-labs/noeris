from __future__ import annotations

import copy
import importlib.util
import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tests import _pathfix  # noqa: F401

from research_engine.arch_cost_model import (
    HARDWARE_PROFILES,
    ArchitectureCostModel,
    generate_nas_candidates,
    _is_tile_aligned,
    _tile_efficiency,
)


REPO = Path(__file__).resolve().parent.parent


BASE_CONFIG = {
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


def _load_nas_experiment_module():
    path = REPO / "scripts" / "nas_experiment.py"
    spec = importlib.util.spec_from_file_location("nas_experiment", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ArchitectureCostModelTests(unittest.TestCase):
    def test_hardware_profiles_are_supported(self) -> None:
        for hardware in HARDWARE_PROFILES:
            model = ArchitectureCostModel(hardware)
            self.assertEqual(model.hardware, hardware)

        with self.assertRaisesRegex(ValueError, "Unknown hardware"):
            ArchitectureCostModel("v100")

    def test_predict_layer_returns_consistent_breakdown(self) -> None:
        prediction = ArchitectureCostModel("a100").predict_layer_ms(BASE_CONFIG)

        self.assertGreater(prediction["total_ms"], 0.0)
        self.assertAlmostEqual(
            prediction["total_ms"],
            sum(prediction["per_kernel"].values()),
            places=9,
        )
        self.assertIn(prediction["bottleneck"], prediction["per_kernel"])
        self.assertEqual(
            set(prediction["tile_penalties"]),
            {"hidden_dim", "ffn_dim", "head_dim"},
        )

    def test_tile_alignment_reports_kernel_cliffs(self) -> None:
        self.assertTrue(_is_tile_aligned(4096))
        self.assertFalse(_is_tile_aligned(4000))
        self.assertEqual(_tile_efficiency(4096), 1.0)
        self.assertLess(_tile_efficiency(4000), 1.0)

        config = dict(
            BASE_CONFIG,
            hidden_dim=4000,
            num_heads=50,
            num_kv_heads=5,
            head_dim=80,
            ffn_dim=16192,
        )
        prediction = ArchitectureCostModel("a100").predict_layer_ms(config)
        hidden_penalty = prediction["tile_penalties"]["hidden_dim"]
        ffn_penalty = prediction["tile_penalties"]["ffn_dim"]

        self.assertFalse(hidden_penalty["aligned_128"])
        self.assertFalse(ffn_penalty["aligned_128"])
        self.assertGreater(hidden_penalty["wasted_work_pct"], 0.0)
        self.assertGreater(ffn_penalty["wasted_work_pct"], 0.0)

    def test_sliding_window_reduces_attention_cost(self) -> None:
        model = ArchitectureCostModel("a100")
        full_attention = model.predict_layer_ms(dict(BASE_CONFIG, window_size=None))
        window_attention = model.predict_layer_ms(dict(BASE_CONFIG, window_size=512))

        self.assertGreater(
            full_attention["per_kernel"]["attention"],
            window_attention["per_kernel"]["attention"],
        )

    def test_h100_profile_predicts_lower_latency_than_a100(self) -> None:
        a100 = ArchitectureCostModel("a100").predict_layer_ms(BASE_CONFIG)
        h100 = ArchitectureCostModel("h100").predict_layer_ms(BASE_CONFIG)

        self.assertLess(h100["total_ms"], a100["total_ms"])
        self.assertLess(h100["per_kernel"]["geglu_mlp"], a100["per_kernel"]["geglu_mlp"])

    def test_rank_configs_sorts_and_preserves_inputs(self) -> None:
        configs = [
            dict(
                BASE_CONFIG,
                name="wide",
                hidden_dim=4096,
                num_heads=32,
                num_kv_heads=8,
                ffn_dim=14336,
            ),
            dict(BASE_CONFIG, name="compact", hidden_dim=1024, num_heads=8, ffn_dim=4096),
            dict(BASE_CONFIG, name="base"),
        ]
        original = copy.deepcopy(configs)

        ranked = ArchitectureCostModel("a100").rank_configs(configs)

        self.assertEqual(configs, original)
        self.assertEqual([row["rank"] for row in ranked], [1, 2, 3])
        self.assertEqual(ranked[0]["name"], "compact")
        self.assertLessEqual(ranked[0]["total_ms"], ranked[1]["total_ms"])

    def test_rank_configs_validates_metric(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown ranking metric"):
            ArchitectureCostModel("a100").rank_configs([BASE_CONFIG], metric="accuracy")

    def test_sweep_dimension_includes_tile_efficiency(self) -> None:
        base = {
            "num_heads": 16,
            "num_kv_heads": 2,
            "head_dim": 128,
            "ffn_dim": 8192,
            "seq_len": 2048,
            "batch_size": 1,
            "use_qk_norm": True,
            "window_size": 1024,
        }
        results = ArchitectureCostModel("a100").sweep_dimension(
            base, "hidden_dim", [2048, 2016]
        )

        self.assertTrue(results[0]["aligned_128"])
        self.assertEqual(results[0]["tile_efficiency"], 1.0)
        self.assertFalse(results[1]["aligned_128"])
        self.assertLess(results[1]["tile_efficiency"], 1.0)

    def test_generate_nas_candidates_varies_architecture_knobs(self) -> None:
        candidates = generate_nas_candidates(
            BASE_CONFIG,
            hidden_dims=[2048],
            head_dims=[64, 128],
            ffn_ratios=[3.0, 4.0],
            kv_head_counts=[1, 2, 64],
            window_sizes=[512, None],
            qk_norm_options=[True, False],
        )

        names = [cfg["name"] for cfg in candidates]
        self.assertEqual(len(names), len(set(names)))
        self.assertGreater(len(candidates), 1)
        self.assertTrue(all(cfg["ffn_dim"] % 128 == 0 for cfg in candidates))
        self.assertTrue(
            all(cfg["num_heads"] % cfg["num_kv_heads"] == 0 for cfg in candidates)
        )
        self.assertTrue(
            all(cfg["num_kv_heads"] <= cfg["num_heads"] for cfg in candidates)
        )
        self.assertEqual({cfg["head_dim"] for cfg in candidates}, {64, 128})
        self.assertEqual({cfg["use_qk_norm"] for cfg in candidates}, {True, False})
        self.assertNotIn(64, {cfg["num_kv_heads"] for cfg in candidates})

    def test_generated_candidates_are_rankable(self) -> None:
        candidates = generate_nas_candidates(
            BASE_CONFIG,
            hidden_dims=[1536, 2048],
            head_dims=[128, 256],
            ffn_ratios=[3.0],
            kv_head_counts=[1, 2],
            window_sizes=[1024],
            qk_norm_options=[True],
        )
        ranked = ArchitectureCostModel("a100").rank_configs(candidates)

        self.assertEqual(len(ranked), len(candidates))
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertLessEqual(ranked[0]["total_ms"], ranked[-1]["total_ms"])


class NasExperimentTests(unittest.TestCase):
    def test_build_report_has_expected_schema_for_each_hardware(self) -> None:
        module = _load_nas_experiment_module()

        for hardware in ("a100", "t4", "h100"):
            report = module.build_report(ArchitectureCostModel(hardware))

            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(report["experiment"], "kernel_aware_nas")
            self.assertEqual(report["hardware"], hardware)
            self.assertEqual(report["candidate_count"], 9)
            self.assertEqual(len(report["comparison"]), report["candidate_count"])
            self.assertEqual(
                len(report["rankings"]["total_ms"]),
                report["candidate_count"],
            )
            self.assertEqual(
                len(report["rankings"]["ms_per_mparam_proxy"]),
                report["candidate_count"],
            )
            self.assertEqual(report["rankings"]["total_ms"][0]["rank"], 1)
            self.assertEqual(
                report["summary"]["fastest_config"],
                report["rankings"]["total_ms"][0]["name"],
            )
            self.assertGreater(report["generated_candidate_count"], 0)
            self.assertEqual(len(report["generated_search"]["total_ms_top"]), 25)
            self.assertEqual(
                report["summary"]["fastest_generated_config"],
                report["generated_search"]["total_ms_top"][0]["name"],
            )
            self.assertIn("hidden_dim", report["kernel_cliffs"])
            self.assertIn("ffn_dim", report["kernel_cliffs"])

    def test_build_report_preserves_cross_hardware_latency_ordering(self) -> None:
        module = _load_nas_experiment_module()
        reports = {
            hardware: module.build_report(ArchitectureCostModel(hardware))
            for hardware in ("a100", "t4", "h100")
        }

        def total_ms(hardware: str, name: str) -> float:
            rows = {
                row["name"]: row
                for row in reports[hardware]["comparison"]
            }
            return rows[name]["total_ms"]

        for name in ("gemma4_e2b", "optimal_2b", "llama3_8b"):
            self.assertLess(total_ms("h100", name), total_ms("a100", name))
            self.assertLess(total_ms("a100", name), total_ms("t4", name))

    def test_cli_writes_json_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = Path(temp_dir) / "nas-a100.json"
            result = subprocess.run(
                [
                    "python3",
                    str(REPO / "scripts" / "nas_experiment.py"),
                    "--hardware",
                    "a100",
                    "--json-output",
                    str(artifact),
                ],
                cwd=REPO,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(artifact.exists())
            payload = json.loads(artifact.read_text(encoding="utf-8"))

        self.assertIn("Wrote JSON artifact", result.stdout)
        self.assertEqual(payload["hardware"], "a100")
        self.assertEqual(payload["summary"]["fastest_config"], "deep_narrow")

    def test_run_comparison_does_not_mutate_config_lists(self) -> None:
        module = _load_nas_experiment_module()
        before = copy.deepcopy(module.KNOWN_CONFIGS + module.NOVEL_CONFIGS)

        with redirect_stdout(io.StringIO()):
            module.run_comparison(ArchitectureCostModel("a100"))

        self.assertEqual(module.KNOWN_CONFIGS + module.NOVEL_CONFIGS, before)

    def test_run_comparison_prints_nas_ranking(self) -> None:
        module = _load_nas_experiment_module()
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            module.run_comparison(ArchitectureCostModel("a100"))

        self.assertIn("Fastest-first NAS ranking", stdout.getvalue())

    def test_generated_search_prints_top_candidates(self) -> None:
        module = _load_nas_experiment_module()
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            module.run_generated_search(ArchitectureCostModel("a100"), top_n=3)

        output = stdout.getvalue()
        self.assertIn("Generated NAS candidates", output)
        self.assertIn("top 3", output)
        self.assertIn("#01", output)


if __name__ == "__main__":
    unittest.main()
