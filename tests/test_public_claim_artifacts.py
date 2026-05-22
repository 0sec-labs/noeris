from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts/check_public_claim_artifacts.py"


def _load_checker_module():
    spec = importlib.util.spec_from_file_location("check_public_claim_artifacts", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load check_public_claim_artifacts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PublicClaimArtifactTests(unittest.TestCase):
    def test_checker_passes_on_current_public_docs(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("headline claims match artifacts", result.stdout)

    def test_checker_scans_canonical_results_index(self) -> None:
        checker = _load_checker_module()
        docs = {path.relative_to(REPO).as_posix() for path in checker.PUBLIC_DOCS}

        self.assertIn("README.md", docs)
        self.assertIn("docs/paper/noeris.md", docs)
        self.assertIn("docs/results/README.md", docs)

    def test_checker_tracks_headline_claim_artifacts(self) -> None:
        checker = _load_checker_module()
        artifacts = {path.relative_to(REPO).as_posix() for path in checker.CLAIM_ARTIFACTS}

        expected = {
            "docs/results/a100-sliding-window-showdown.json",
            "docs/results/a100-end-to-end-26layer.json",
            "docs/results/a100-19model-generalization.json",
            "docs/results/hardware-cross-learning-a100-to-h100.json",
            "docs/results/qk-norm-rope-a100-full.json",
            "docs/results/qk-norm-rope-h100-full.json",
            "docs/results/gemma4-layer-bench-deeper-fusion-a100-after-geglu-retune.json",
            "docs/results/gemma4-layer-bench-deeper-fusion-h100-after-geglu-retune.json",
        }
        self.assertLessEqual(expected, artifacts)


if __name__ == "__main__":
    unittest.main()
