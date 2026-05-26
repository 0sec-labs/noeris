"""Experimental deterministic scaffold for ScienceAgentBench-style tasks.

This module is deliberately local-only. It mirrors the shape of the Noeris
kernel loop -- registry, strategy routing, result database, and reusable
memory -- without downloading ScienceAgentBench, calling an LLM, or using paid
compute.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STRATEGY_DECOMPOSE = "decompose_subtasks"
STRATEGY_DIRECT = "direct_solve"
STRATEGY_RETRIEVE = "retrieve_and_adapt"

DEFAULT_STRATEGIES = [
    STRATEGY_DECOMPOSE,
    STRATEGY_DIRECT,
    STRATEGY_RETRIEVE,
]


@dataclass(frozen=True, slots=True)
class ScienceTaskSpec:
    """Minimal task descriptor for a deterministic science experiment."""

    task_id: str
    name: str
    discipline: str
    task_type: str
    input_payload: dict[str, Any]
    expected_output: dict[str, Any]
    strategies: list[str] = field(default_factory=lambda: list(DEFAULT_STRATEGIES))
    description: str = ""


@dataclass(frozen=True, slots=True)
class ScienceMemoryItem:
    """Reusable cross-task memory extracted from a completed task."""

    memory_id: str
    discipline: str
    task_type: str
    strategy: str
    content: str
    evidence_task_id: str
    confidence: str = "medium"


@dataclass(frozen=True, slots=True)
class ScienceExperimentResult:
    """Recorded result from running one strategy on one task."""

    task_id: str
    discipline: str
    task_type: str
    strategy: str
    success: bool
    score: float
    output: dict[str, Any]
    expected_output: dict[str, Any]
    reusable_memory: list[ScienceMemoryItem] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class TaskRegistry:
    """Registry for experimental science task specs."""

    def __init__(self) -> None:
        self._tasks: dict[str, ScienceTaskSpec] = {}

    def register(self, task: ScienceTaskSpec) -> ScienceTaskSpec:
        if task.task_id in self._tasks:
            raise ValueError(f"Science task already registered: {task.task_id}")
        if not task.strategies:
            raise ValueError("Science task must define at least one strategy")
        self._tasks[task.task_id] = task
        return task

    def get(self, task_id: str) -> ScienceTaskSpec:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise KeyError(f"Unknown science task: {task_id}") from exc

    def list_tasks(
        self,
        *,
        discipline: str | None = None,
        task_type: str | None = None,
    ) -> list[ScienceTaskSpec]:
        tasks = list(self._tasks.values())
        if discipline is not None:
            tasks = [task for task in tasks if task.discipline == discipline]
        if task_type is not None:
            tasks = [task for task in tasks if task.task_type == task_type]
        return sorted(tasks, key=lambda task: task.task_id)


class ExperimentDatabase:
    """Small JSON-backed result and memory database for science tasks."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.results: list[ScienceExperimentResult] = []
        if self.path is not None and self.path.exists():
            self._load()

    def record_result(self, result: ScienceExperimentResult) -> ScienceExperimentResult:
        self.results.append(result)
        if self.path is not None:
            self.save()
        return result

    def results_for(
        self,
        *,
        task_id: str | None = None,
        discipline: str | None = None,
        task_type: str | None = None,
    ) -> list[ScienceExperimentResult]:
        results = self.results
        if task_id is not None:
            results = [result for result in results if result.task_id == task_id]
        if discipline is not None:
            results = [result for result in results if result.discipline == discipline]
        if task_type is not None:
            results = [result for result in results if result.task_type == task_type]
        return list(results)

    def memory_items(
        self,
        *,
        discipline: str | None = None,
        task_type: str | None = None,
    ) -> list[ScienceMemoryItem]:
        by_id: dict[str, ScienceMemoryItem] = {}
        for result in self.results_for(discipline=discipline, task_type=task_type):
            for item in result.reusable_memory:
                by_id[item.memory_id] = item
        return [by_id[key] for key in sorted(by_id)]

    def strategy_success_rate(
        self,
        *,
        discipline: str,
        task_type: str,
        strategy: str,
    ) -> float | None:
        rows = [
            result
            for result in self.results_for(discipline=discipline, task_type=task_type)
            if result.strategy == strategy
        ]
        if not rows:
            return None
        return sum(1.0 for row in rows if row.success) / len(rows)

    def save(self) -> None:
        if self.path is None:
            raise ValueError("ExperimentDatabase has no path")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "results": [_result_to_dict(result) for result in self.results],
        }
        self.path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _load(self) -> None:
        if self.path is None:
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.results = [
            _result_from_dict(item) for item in payload.get("results", [])
        ]


class ScienceStrategyRouter:
    """Deterministic strategy chooser for the local scaffold.

    The router mirrors the role of the existing adaptive router, but keeps v0
    free of optional numeric dependencies and model cost estimates. It prefers
    retrieval when compatible memory exists, otherwise picks the historically
    best strategy for the task type, falling back to task order for cold start.
    """

    def choose_strategy(
        self,
        task: ScienceTaskSpec,
        database: ExperimentDatabase,
    ) -> str:
        if (
            STRATEGY_RETRIEVE in task.strategies
            and database.memory_items(
                discipline=task.discipline,
                task_type=task.task_type,
            )
        ):
            return STRATEGY_RETRIEVE

        scored: list[tuple[float, int, str]] = []
        for index, strategy in enumerate(task.strategies):
            rate = database.strategy_success_rate(
                discipline=task.discipline,
                task_type=task.task_type,
                strategy=strategy,
            )
            if rate is not None:
                scored.append((rate, -index, strategy))
        if scored:
            return max(scored)[2]
        return task.strategies[0]


class DeterministicScienceRunner:
    """Run local fixture tasks and record results."""

    def __init__(
        self,
        *,
        registry: TaskRegistry,
        database: ExperimentDatabase | None = None,
        router: ScienceStrategyRouter | None = None,
    ) -> None:
        self.registry = registry
        self.database = database or ExperimentDatabase()
        self.router = router or ScienceStrategyRouter()

    def run_task(self, task_id: str) -> ScienceExperimentResult:
        task = self.registry.get(task_id)
        strategy = self.router.choose_strategy(task, self.database)
        result = run_local_fixture_task(
            task,
            strategy=strategy,
            memory=self.database.memory_items(
                discipline=task.discipline,
                task_type=task.task_type,
            ),
        )
        self.database.record_result(result)
        return result


def build_local_fixture_registry() -> TaskRegistry:
    """Return a registry with one deterministic toy science task."""

    registry = TaskRegistry()
    registry.register(
        ScienceTaskSpec(
            task_id="toy_bio_normalize_total",
            name="Toy bioinformatics normalization",
            discipline="bioinformatics",
            task_type="normalization",
            description=(
                "Normalize a count vector to a fixed target sum. This is a "
                "local fixture only, not a ScienceAgentBench task."
            ),
            input_payload={
                "counts": [2.0, 4.0, 6.0],
                "target_sum": 24.0,
            },
            expected_output={
                "normalized": [4.0, 8.0, 12.0],
            },
        )
    )
    return registry


def run_local_fixture_task(
    task: ScienceTaskSpec,
    *,
    strategy: str,
    memory: list[ScienceMemoryItem] | None = None,
) -> ScienceExperimentResult:
    """Run a supported local fixture task with the requested strategy."""

    if strategy not in task.strategies:
        raise ValueError(f"Strategy {strategy!r} is not enabled for {task.task_id}")
    if task.task_type != "normalization":
        raise ValueError(f"No local fixture runner for task type: {task.task_type}")

    counts = [float(value) for value in task.input_payload["counts"]]
    target_sum = float(task.input_payload["target_sum"])
    output: dict[str, Any]
    notes: list[str] = []

    if strategy == STRATEGY_DIRECT:
        output = {"total": round(sum(counts), 6)}
        notes.append("direct_solve computes a raw total and intentionally misses normalization")
    elif strategy == STRATEGY_RETRIEVE and _has_normalization_memory(memory or []):
        output = {"normalized": _normalize_total(counts, target_sum)}
        notes.append("retrieve_and_adapt reused prior normalization memory")
    else:
        output = {"normalized": _normalize_total(counts, target_sum)}
        notes.append("decompose_subtasks split load, scale, and format steps")

    success = output == task.expected_output
    score = 1.0 if success else 0.0
    reusable_memory = _extract_memory(task, strategy, success)
    return ScienceExperimentResult(
        task_id=task.task_id,
        discipline=task.discipline,
        task_type=task.task_type,
        strategy=strategy,
        success=success,
        score=score,
        output=output,
        expected_output=task.expected_output,
        reusable_memory=reusable_memory,
        notes=notes,
    )


def _normalize_total(counts: list[float], target_sum: float) -> list[float]:
    total = sum(counts)
    if total == 0:
        return [0.0 for _ in counts]
    scale = target_sum / total
    return [round(value * scale, 6) for value in counts]


def _has_normalization_memory(memory: list[ScienceMemoryItem]) -> bool:
    return any(item.task_type == "normalization" for item in memory)


def _extract_memory(
    task: ScienceTaskSpec,
    strategy: str,
    success: bool,
) -> list[ScienceMemoryItem]:
    if not success:
        return []
    if task.task_type != "normalization":
        return []
    return [
        ScienceMemoryItem(
            memory_id=f"{task.discipline}:{task.task_type}:normalize-total",
            discipline=task.discipline,
            task_type=task.task_type,
            strategy=strategy,
            content=(
                "For normalization tasks, scale each count by "
                "target_sum / sum(counts) and preserve output order."
            ),
            evidence_task_id=task.task_id,
            confidence="high",
        )
    ]


def _result_to_dict(result: ScienceExperimentResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["reusable_memory"] = [asdict(item) for item in result.reusable_memory]
    return payload


def _result_from_dict(payload: dict[str, Any]) -> ScienceExperimentResult:
    memory = [
        ScienceMemoryItem(**item) for item in payload.get("reusable_memory", [])
    ]
    return ScienceExperimentResult(
        task_id=payload["task_id"],
        discipline=payload["discipline"],
        task_type=payload["task_type"],
        strategy=payload["strategy"],
        success=bool(payload["success"]),
        score=float(payload["score"]),
        output=payload.get("output", {}),
        expected_output=payload.get("expected_output", {}),
        reusable_memory=memory,
        notes=payload.get("notes", []),
        created_at_utc=payload.get("created_at_utc", ""),
    )
