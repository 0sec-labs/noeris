from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


class PackageMetadataTests(unittest.TestCase):
    def test_generated_egg_info_metadata_is_not_tracked(self) -> None:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        )
        tracked_egg_info = [
            path
            for path in result.stdout.splitlines()
            if (REPO / path).exists()
            and any(part.endswith(".egg-info") for part in Path(path).parts)
        ]

        self.assertEqual(tracked_egg_info, [])

    def test_gitignore_ignores_generated_egg_info_metadata(self) -> None:
        ignored_patterns = {
            line.strip()
            for line in (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

        self.assertIn("*.egg-info/", ignored_patterns)


if __name__ == "__main__":
    unittest.main()
