"""Deterministic file boundary from Noeris into the 0brain controller."""

from __future__ import annotations

import json
import math
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from .kernel_candidates import (
    build_kernel_improvement_challengers,
    propose_kernel_configs,
)
from .triton_operators import REGISTRY
from .world_model import ConfigHypothesis, WorldModel
from .zero_research import GeneratorIdentity


def export_kernel_challengers(
    *,
    world_model_path: str | Path,
    source_refs_path: str | Path,
    baseline_path: str | Path,
    shape_path: str | Path,
    operator: str,
    hardware: str,
    generator: GeneratorIdentity,
    max_candidates: int = 5,
) -> list[dict[str, object]]:
    """Load sealed inputs and return a bounded controller-ready JSON array."""

    if not hardware.strip():
        raise ValueError("hardware must be non-empty")
    world_model_raw = _read_json(world_model_path, "world model")
    if not isinstance(world_model_raw, list):
        raise ValueError("world model must be a JSON array")
    hypotheses: list[ConfigHypothesis] = []
    for index, item in enumerate(world_model_raw):
        if not isinstance(item, dict):
            raise ValueError(f"world model[{index}] must be an object")
        _validate_hypothesis(item, index)
        try:
            hypotheses.append(ConfigHypothesis.from_dict(item))
        except TypeError as exc:
            raise ValueError(f"world model[{index}] has an invalid schema") from exc

    refs_raw = _read_json(source_refs_path, "source refs")
    if not isinstance(refs_raw, dict):
        raise ValueError("source refs must be a JSON object")
    refs: dict[str, tuple[str, ...]] = {}
    for key, value in refs_raw.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(value, list) or not value or not all(
            isinstance(ref, str) and bool(ref.strip()) for ref in value
        ):
            raise ValueError("source refs values must be non-empty arrays of non-empty strings")
        refs[key] = tuple(value)

    baseline = _integer_object(_read_json(baseline_path, "baseline"), "baseline")
    shape = _object(_read_json(shape_path, "shape"), "shape")
    spec = REGISTRY.get(operator)
    proposals = propose_kernel_configs(
        spec=spec,
        baseline_config=baseline,
        shape=shape,
        hardware=hardware,
        world_model=WorldModel(hypotheses, include_builtins=False),
        source_refs_by_hypothesis=refs,
        max_candidates=max_candidates,
    )
    if not proposals:
        raise ValueError("no valid kernel challengers were generated")
    return build_kernel_improvement_challengers(
        proposals,
        generator=generator,
        max_candidates=max_candidates,
    )


def canonical_json(value: object) -> str:
    """Match the deliberately narrow canonical JSON used by 0brain."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_canonical_json(path: str | Path, value: object) -> None:
    """Atomically write one canonical artifact without following an output symlink."""

    destination = Path(path)
    if destination.is_symlink():
        raise ValueError("output path must not be a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _read_json(path: str | Path, label: str) -> Any:
    source = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(source, flags)
    except OSError as exc:
        raise ValueError(f"{label} must be a readable regular file") from exc
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 16 * 1024 * 1024:
            raise ValueError(f"{label} must be a regular file no larger than 16 MiB")
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            fd = -1
            return json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    finally:
        if fd >= 0:
            os.close(fd)


def _object(value: Any, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _integer_object(value: Any, label: str) -> dict[str, int]:
    raw = _object(value, label)
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in raw.values()):
        raise ValueError(f"{label} values must be integers")
    return raw  # type: ignore[return-value]


def _validate_hypothesis(value: dict[str, Any], index: int) -> None:
    expected = {
        "description",
        "conditions",
        "predicted_effect",
        "evidence_for",
        "evidence_against",
        "confidence",
        "source",
    }
    if set(value) != expected:
        raise ValueError(f"world model[{index}] has an invalid schema")
    for field in ("description", "predicted_effect", "source"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"world model[{index}].{field} must be non-empty text")
    conditions = value["conditions"]
    if not isinstance(conditions, dict) or not all(
        isinstance(key, str) and bool(key.strip()) for key in conditions
    ):
        raise ValueError(f"world model[{index}].conditions must be an object with non-empty keys")
    for field in ("evidence_for", "evidence_against"):
        count = value[field]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"world model[{index}].{field} must be a non-negative integer")
    confidence = value["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError(f"world model[{index}].confidence must be finite and in [0, 1]")
