from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_engine.cli import main
from research_engine.zero_research_export import export_kernel_challengers
from research_engine.zero_research import GeneratorIdentity


GENERATOR = GeneratorIdentity(
    id="noeris.world-model-v1",
    digest=f"sha256:{'a' * 64}",
)


def inputs(tmp_path: Path, *, conditions: dict[str, object] | None = None) -> dict[str, Path]:
    payloads = {
        "world-model": [
            {
                "description": "deep K split",
                "conditions": conditions or {"operator": "matmul", "num_stages": [3, 4]},
                "predicted_effect": "higher measured throughput",
                "evidence_for": 9,
                "evidence_against": 1,
                "confidence": 0.9,
                "source": "sealed-controller-input",
            }
        ],
        "source-refs": {"deep K split": ["artifact:world-model:sha256:abc"]},
        "baseline": {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 2,
        },
        "shape": {"M": 64, "N": 64, "K": 4096},
    }
    paths = {}
    for name, value in payloads.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path
    return paths


def export(paths: dict[str, Path], **overrides):
    return export_kernel_challengers(
        world_model_path=paths["world-model"],
        source_refs_path=paths["source-refs"],
        baseline_path=paths["baseline"],
        shape_path=paths["shape"],
        operator="matmul",
        hardware="H100",
        generator=overrides.pop("generator", GENERATOR),
        max_candidates=overrides.pop("max_candidates", 5),
        **overrides,
    )


def test_export_is_deterministic_complete_bounded_and_generator_only(tmp_path: Path) -> None:
    paths = inputs(tmp_path)
    first = export(paths)
    second = export(paths)
    assert first == second
    assert 1 <= len(first) <= 5
    assert all(set(item) == {
        "schemaVersion", "kind", "id", "targetProject", "generator", "hypothesis", "change"
    } for item in first)
    assert all(set(item["change"]["knobs"]) == {
        "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "num_warps", "num_stages"
    } for item in first)
    forbidden = {"budget", "evaluation", "score", "verdict", "promotion", "authority"}
    rendered = json.dumps(first)
    assert all(field not in rendered for field in forbidden)


def test_reordered_input_json_produces_identical_output(tmp_path: Path) -> None:
    paths = inputs(tmp_path)
    expected = export(paths)
    for path in paths.values():
        value = json.loads(path.read_text())
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    assert export(paths) == expected


def test_fails_closed_on_zero_candidates_and_bad_generator(tmp_path: Path) -> None:
    paths = inputs(tmp_path, conditions={"operator": "softmax", "num_stages": 3})
    with pytest.raises(ValueError, match="no valid kernel challengers"):
        export(paths)
    paths = inputs(tmp_path)
    with pytest.raises(ValueError, match="generator digest"):
        export(paths, generator=GeneratorIdentity("noeris.world-model-v1", "moving-tag"))


def test_cli_writes_canonical_array_and_rejects_more_than_five(tmp_path: Path) -> None:
    paths = inputs(tmp_path)
    output = tmp_path / "challengers.json"
    common = [
        "0research-export",
        "--world-model", str(paths["world-model"]),
        "--source-refs", str(paths["source-refs"]),
        "--baseline", str(paths["baseline"]),
        "--shape", str(paths["shape"]),
        "--operator", "matmul",
        "--hardware", "H100",
        "--generator-id", GENERATOR.id,
        "--generator-digest", GENERATOR.digest,
    ]
    assert main([*common, "--output", str(output)]) == 0
    assert output.read_text().endswith("\n")
    assert isinstance(json.loads(output.read_text()), list)
    assert main([*common, "--max-candidates", "6"]) == 2
