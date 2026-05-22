from __future__ import annotations

import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
CONTRIBUTING = REPO / "CONTRIBUTING.md"
README = REPO / "README.md"
CI_DOC = REPO / "docs/system/CI.md"


class ContributorSetupDocsTests(unittest.TestCase):
    def test_contributing_documents_expected_setup_paths(self) -> None:
        text = CONTRIBUTING.read_text(encoding="utf-8")

        required = [
            "CPU-Only Setup",
            "Linux CUDA Setup",
            "macOS arm64 and `uv`",
            "PYTHONPATH",
            "scripts/ci_local.sh",
            "MODAL_TOKEN_ID",
            "AZURE_OPENAI_API_KEY",
        ]
        for phrase in required:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)

    def test_setup_docs_are_linked_from_readme_and_ci_docs(self) -> None:
        for path in (README, CI_DOC):
            with self.subTest(path=path):
                self.assertIn("CONTRIBUTING.md", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
