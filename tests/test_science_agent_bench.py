from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests import _pathfix  # noqa: F401

from research_engine.science_agent_bench import (
    STRATEGY_DECOMPOSE,
    STRATEGY_RETRIEVE,
    DeterministicScienceRunner,
    ExperimentDatabase,
    ScienceStrategyRouter,
    ScienceTaskSpec,
    TaskRegistry,
    build_local_fixture_registry,
    run_local_fixture_task,
)


class ScienceAgentBenchScaffoldTests(unittest.TestCase):
    def test_register_choose_record_and_extract_memory(self) -> None:
        registry = build_local_fixture_registry()
        database = ExperimentDatabase()
        runner = DeterministicScienceRunner(registry=registry, database=database)

        result = runner.run_task("toy_bio_normalize_total")

        self.assertEqual(result.strategy, STRATEGY_DECOMPOSE)
        self.assertTrue(result.success)
        self.assertEqual(result.output, {"normalized": [4.0, 8.0, 12.0]})
        self.assertEqual(database.results_for(task_id="toy_bio_normalize_total"), [result])

        memory = database.memory_items(
            discipline="bioinformatics",
            task_type="normalization",
        )
        self.assertEqual(len(memory), 1)
        self.assertIn("target_sum / sum(counts)", memory[0].content)

    def test_router_uses_retrieval_after_memory_exists(self) -> None:
        registry = build_local_fixture_registry()
        database = ExperimentDatabase()
        runner = DeterministicScienceRunner(registry=registry, database=database)

        first = runner.run_task("toy_bio_normalize_total")
        second_strategy = ScienceStrategyRouter().choose_strategy(
            registry.get("toy_bio_normalize_total"),
            database,
        )
        second = runner.run_task("toy_bio_normalize_total")

        self.assertEqual(first.strategy, STRATEGY_DECOMPOSE)
        self.assertEqual(second_strategy, STRATEGY_RETRIEVE)
        self.assertEqual(second.strategy, STRATEGY_RETRIEVE)
        self.assertTrue(second.success)
        self.assertTrue(
            any("reused prior normalization memory" in note for note in second.notes)
        )

    def test_database_round_trip(self) -> None:
        registry = build_local_fixture_registry()
        task = registry.get("toy_bio_normalize_total")
        result = run_local_fixture_task(task, strategy=STRATEGY_DECOMPOSE)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "science-experiments.json"
            database = ExperimentDatabase(path)
            database.record_result(result)
            loaded = ExperimentDatabase(path)

        self.assertEqual(len(loaded.results), 1)
        self.assertEqual(loaded.results[0].task_id, result.task_id)
        self.assertEqual(
            loaded.memory_items()[0].memory_id,
            result.reusable_memory[0].memory_id,
        )

    def test_registry_rejects_duplicates(self) -> None:
        registry = TaskRegistry()
        task = ScienceTaskSpec(
            task_id="dup",
            name="duplicate",
            discipline="chemistry",
            task_type="formatting",
            input_payload={},
            expected_output={},
        )
        registry.register(task)

        with self.assertRaises(ValueError):
            registry.register(task)


if __name__ == "__main__":
    unittest.main()
