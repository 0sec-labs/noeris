from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path

from tests import _pathfix  # noqa: F401

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from cold_shape_cross_run_ablation_v2 import (  # noqa: E402
    CONDITIONS,
    DEFAULT_HARDWARE,
    DEFAULT_OPERATORS,
    _to_md,
    build_report,
)


class ColdShapeCrossRunAblationV2Tests(unittest.TestCase):
    def _default_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            db_path=str(REPO / ".noeris/cost-model-training.json"),
            hardware=DEFAULT_HARDWARE,
            operators=list(DEFAULT_OPERATORS),
            heldout_per_operator=1,
            iterations=6,
            configs_per_iteration=1,
            seeds=[0, 1, 2],
            min_candidates=6,
            exclude_curated=True,
        )

    def test_build_report_contains_raw_histories(self) -> None:
        report = build_report(self._default_args())

        self.assertEqual(report["experiment"], "cold_shape_cross_run_ablation_v2")
        self.assertEqual(report["protocol"]["conditions"], list(CONDITIONS))
        self.assertTrue(report["protocol"]["exclude_curated"])
        self.assertEqual(len(report["heldout_buckets"]), len(DEFAULT_OPERATORS))
        self.assertGreater(report["training"]["rows_after_holdout"], 0)
        self.assertIn("ranking_model", report["training"])

        for bucket in report["by_bucket"].values():
            for condition in CONDITIONS:
                runs = bucket["conditions"][condition]["runs"]
                self.assertEqual(len(runs), 3)
                for run in runs:
                    self.assertEqual(len(run["history"]), 6)
                    first = run["history"][0]
                    self.assertIn("evaluated_config_ids", first)
                    self.assertIn("best_so_far_metric", first)
                    self.assertIn("regret_pct", first)

    def test_markdown_summary_mentions_interpretation(self) -> None:
        report = build_report(self._default_args())
        md = _to_md(report)

        self.assertIn("Cold-Shape Cross-Run Learning Ablation v2", md)
        self.assertIn("mean final normalized", md)
        self.assertIn("Interpretation", md)

    def test_checked_in_artifact_has_histories(self) -> None:
        artifact = REPO / "docs/results/cold-shape-cross-run-ablation-v2.json"
        data = json.loads(artifact.read_text(encoding="utf-8"))

        self.assertEqual(data["protocol"]["iterations"], 6)
        self.assertEqual(data["protocol"]["seeds"], [0, 1, 2])
        self.assertEqual(set(data["summary"]), set(CONDITIONS))
        sample_bucket = next(iter(data["by_bucket"].values()))
        sample_run = sample_bucket["conditions"]["stateless_random"]["runs"][0]
        self.assertEqual(len(sample_run["history"]), 6)


if __name__ == "__main__":
    unittest.main()
