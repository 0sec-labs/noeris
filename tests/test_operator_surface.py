from __future__ import annotations

import argparse
import re
import unittest
from pathlib import Path

from research_engine import TRITON_OPERATORS
from research_engine.cli import build_parser
from research_engine.operator_surface import (
    PUBLIC_PATCH_OPERATORS,
    REGISTERED_INTERNAL_OPERATORS,
    TRITON_ITERATE_OPERATORS,
    TRITON_ITERATE_WORKFLOW_OPERATORS,
)


REPO = Path(__file__).resolve().parent.parent


class OperatorSurfaceTests(unittest.TestCase):
    def test_operator_surface_partitions_registry(self) -> None:
        registered = tuple(TRITON_OPERATORS.names())
        searchable = set(TRITON_ITERATE_OPERATORS)
        internal = set(REGISTERED_INTERNAL_OPERATORS)

        self.assertEqual(tuple(sorted(searchable | internal)), registered)
        self.assertEqual(searchable & internal, set())
        self.assertLessEqual(set(TRITON_ITERATE_WORKFLOW_OPERATORS), searchable)
        self.assertLessEqual(set(PUBLIC_PATCH_OPERATORS), set(registered))

    def test_cli_operator_choices_match_searchable_surface(self) -> None:
        parser = build_parser()
        subparsers = _subparsers(parser)

        for command in ("triton-iterate", "ablation"):
            with self.subTest(command=command):
                command_parser = subparsers.choices[command]
                operator_action = _operator_action(command_parser)
                self.assertEqual(tuple(operator_action.choices), TRITON_ITERATE_OPERATORS)

    def test_workflow_matrix_matches_declared_workflow_surface(self) -> None:
        workflow = (REPO / ".github/workflows/triton-iterate.yml").read_text(encoding="utf-8")
        match = re.search(r"operator: \[(?P<operators>[^\]]+)\]", workflow)
        self.assertIsNotNone(match)
        operators = tuple(
            item.strip()
            for item in match.group("operators").split(",")
            if item.strip()
        )

        self.assertEqual(operators, TRITON_ITERATE_WORKFLOW_OPERATORS)

    def test_operator_docs_include_current_registry_names_and_counts(self) -> None:
        readme = (REPO / "README.md").read_text(encoding="utf-8")
        surface_doc = (REPO / "docs/system/OPERATOR_SURFACE.md").read_text(encoding="utf-8")
        registered = TRITON_OPERATORS.names()

        self.assertIn(f"operators-{len(registered)}", readme)
        self.assertIn(f"Registered operator specs | {len(registered)}", surface_doc)

        for name in registered:
            with self.subTest(name=name):
                self.assertIn(f"`{name}`", surface_doc)


def _subparsers(parser: argparse.ArgumentParser) -> argparse._SubParsersAction:
    return next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )


def _operator_action(parser: argparse.ArgumentParser) -> argparse.Action:
    return next(action for action in parser._actions if action.dest == "operator")


if __name__ == "__main__":
    unittest.main()
