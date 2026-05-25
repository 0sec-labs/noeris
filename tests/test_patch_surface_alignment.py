"""Tests that the supported patch kernel list and failure messages stay aligned
between public code, research-engine code, README, and OPERATOR_SURFACE docs.

Closes https://github.com/0sec-labs/noeris/issues/113
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


class TestPatchSurfaceAlignment(unittest.TestCase):
    """Ensure a single authoritative public patch surface across code and docs."""

    # ------------------------------------------------------------------
    # Source-of-truth: the public noeris package
    # ------------------------------------------------------------------

    def _public_patch_kernels(self) -> frozenset[str]:
        from noeris.patch import SUPPORTED_PATCH_KERNELS
        return SUPPORTED_PATCH_KERNELS

    # ------------------------------------------------------------------
    # operator_surface.py agrees
    # ------------------------------------------------------------------

    def test_operator_surface_public_patch_matches_noeris_package(self) -> None:
        from research_engine.operator_surface import PUBLIC_PATCH_OPERATORS

        self.assertEqual(
            set(PUBLIC_PATCH_OPERATORS),
            self._public_patch_kernels(),
            "PUBLIC_PATCH_OPERATORS in operator_surface.py must match "
            "SUPPORTED_PATCH_KERNELS in noeris/patch.py",
        )

    # ------------------------------------------------------------------
    # OPERATOR_SURFACE.md agrees on count and kernel names
    # ------------------------------------------------------------------

    def test_operator_surface_doc_patch_count(self) -> None:
        doc = (REPO / "docs/system/OPERATOR_SURFACE.md").read_text(encoding="utf-8")
        expected_count = len(self._public_patch_kernels())

        self.assertIn(
            f"Public `noeris.patch()` hooks | {expected_count}",
            doc,
            "OPERATOR_SURFACE.md patch hook count must match SUPPORTED_PATCH_KERNELS",
        )

    def test_operator_surface_doc_lists_only_supported_patch_kernels(self) -> None:
        doc = (REPO / "docs/system/OPERATOR_SURFACE.md").read_text(encoding="utf-8")
        patch_section = doc.split("## Public Patch API")[1]

        for kernel in self._public_patch_kernels():
            with self.subTest(kernel=kernel):
                self.assertIn(
                    f"- `{kernel}`",
                    patch_section,
                    f"OPERATOR_SURFACE.md Public Patch API must list {kernel}",
                )

    # ------------------------------------------------------------------
    # README.md agrees on the public patch hook count
    # ------------------------------------------------------------------

    def test_readme_patch_hook_count(self) -> None:
        readme = (REPO / "README.md").read_text(encoding="utf-8")
        expected_count = len(self._public_patch_kernels())

        self.assertIn(
            f"Public `noeris.patch()` hooks | {expected_count}",
            readme,
            "README.md patch hook count must match SUPPORTED_PATCH_KERNELS",
        )

    # ------------------------------------------------------------------
    # Docs must NOT claim unsupported generic hooks
    # ------------------------------------------------------------------

    _UNSUPPORTED_GENERIC_HOOKS = (
        "qk_norm_rope",
        "qk_norm",
        "cross_entropy",
        "QK-RMSNorm+RoPE",
    )

    def test_readme_does_not_claim_unsupported_drop_in_hooks(self) -> None:
        readme = (REPO / "README.md").read_text(encoding="utf-8")
        quick_start = readme.split("## Quick start")[1].split("##")[0]

        for hook in self._UNSUPPORTED_GENERIC_HOOKS:
            with self.subTest(hook=hook):
                # These should NOT appear as if they are drop-in patch() args
                self.assertNotIn(
                    f'kernels=["{hook}"',
                    quick_start,
                    f"README quick-start must not show {hook} as a drop-in patch kernel",
                )

    def test_operator_surface_doc_does_not_claim_unsupported_drop_in_hooks(self) -> None:
        doc = (REPO / "docs/system/OPERATOR_SURFACE.md").read_text(encoding="utf-8")
        patch_section = doc.split("## Public Patch API")[1]

        for hook in self._UNSUPPORTED_GENERIC_HOOKS:
            with self.subTest(hook=hook):
                self.assertNotIn(
                    f"- `{hook}`",
                    patch_section,
                    f"OPERATOR_SURFACE.md Public Patch API must not list {hook} "
                    f"as a drop-in hook",
                )

    # ------------------------------------------------------------------
    # research_engine/patch.py module docstring must not overclaim
    # ------------------------------------------------------------------

    def test_research_engine_patch_docstring_does_not_overclaim_drop_in(self) -> None:
        src = (REPO / "src/research_engine/patch.py").read_text(encoding="utf-8")
        # The module-level docstring should not claim QK-RoPE is drop-in
        # Look at the first docstring (module-level)
        match = re.search(r'^"""(.*?)"""', src, re.DOTALL)
        self.assertIsNotNone(match, "research_engine/patch.py must have a module docstring")
        docstring = match.group(1)

        # Must not say "replaces ... QK-RoPE" or similar overclaiming phrasing
        self.assertNotIn(
            "QK-RoPE, GeGLU",
            docstring,
            "research_engine/patch.py docstring must not claim QK-RoPE as drop-in",
        )
        self.assertNotIn(
            "RMSNorm, QK-RoPE",
            docstring,
            "research_engine/patch.py docstring must not claim QK-RoPE as drop-in",
        )

    # ------------------------------------------------------------------
    # noeris.patch() rejects unsupported kernels with a clear message
    # ------------------------------------------------------------------

    def test_patch_rejects_unsupported_kernels_with_clear_error(self) -> None:
        from noeris.patch import _normalize_patch_kernels

        for kernel in self._UNSUPPORTED_GENERIC_HOOKS:
            with self.subTest(kernel=kernel):
                with self.assertRaises(ValueError) as ctx:
                    _normalize_patch_kernels([kernel])
                self.assertIn("does not support drop-in", str(ctx.exception))

    # ------------------------------------------------------------------
    # README quick-start code comment must say RMSNorm + gated MLP only
    # ------------------------------------------------------------------

    def test_readme_quick_start_comment_is_accurate(self) -> None:
        readme = (REPO / "README.md").read_text(encoding="utf-8")
        quick_start = readme.split("## Quick start")[1].split("##")[0]

        # The inline comment on noeris.patch() should not claim QK-RoPE/cross-entropy
        # are part of the drop-in surface
        patch_lines = [
            line for line in quick_start.splitlines()
            if "noeris.patch(" in line
        ]
        for line in patch_lines:
            self.assertNotIn("QK-RoPE", line)
            self.assertNotIn("cross-entropy", line)
            self.assertNotIn("cross_entropy", line)


if __name__ == "__main__":
    unittest.main()
