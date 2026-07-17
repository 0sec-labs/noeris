from __future__ import annotations

from copy import deepcopy

import pytest

from research_engine.kernel_candidates import (
    build_kernel_improvement_challengers,
    propose_kernel_configs,
)
from research_engine.triton_operators import TritonOperatorSpec
from research_engine.world_model import ConfigHypothesis, WorldModel
from research_engine.zero_research import GeneratorIdentity


def spec(shared_memory_check=lambda config: True) -> TritonOperatorSpec:
    return TritonOperatorSpec(
        name="matmul",
        param_space={"BLOCK_SIZE": [64, 128, 256], "SPLIT_K": [1, 2, 4], "num_warps": [1, 2, 4]},
        curated_configs=[],
        shape_buckets=[],
        metric_name="tflops",
        config_id_fn=lambda config: ":".join(f"{key}={config[key]}" for key in sorted(config)),
        shape_bucket_fn=lambda shape: "test",
        benchmark_script_fn=lambda configs, shapes: "",
        grid_generator_fn=lambda **kwargs: [],
        shared_memory_check_fn=shared_memory_check,
    )


def hypothesis(
    description: str,
    conditions: dict[str, object],
    *,
    confidence: float = 0.9,
) -> ConfigHypothesis:
    return ConfigHypothesis(
        description=description,
        conditions=conditions,
        predicted_effect=f"Measured evidence predicts {description}",
        evidence_for=9,
        evidence_against=1,
        confidence=confidence,
        source="discovered",
    )


BASELINE = {"BLOCK_SIZE": 64, "SPLIT_K": 1, "num_warps": 4}


def propose(model: WorldModel, **kwargs):
    refs = {
        item.description: (f"artifact:sealed-world-model:{index}",)
        for index, item in enumerate(model.hypotheses)
    }
    return propose_kernel_configs(
        spec=kwargs.pop("spec", spec()),
        baseline_config=kwargs.pop("baseline_config", BASELINE),
        shape=kwargs.pop("shape", {"M": 64, "N": 64, "K": 4096}),
        hardware=kwargs.pop("hardware", "H100"),
        world_model=model,
        source_refs_by_hypothesis=kwargs.pop("source_refs_by_hypothesis", refs),
        **kwargs,
    )


def test_expands_lists_over_complete_baseline_and_filters_shape_keys() -> None:
    model = WorldModel(
        [hypothesis("deep K", {"operator": "matmul", "K": 4096, "SPLIT_K": [2, 4]})],
        include_builtins=False,
    )
    proposals = propose(model)
    assert [dict(item.config) for item in proposals] == [
        {"BLOCK_SIZE": 64, "SPLIT_K": 2, "num_warps": 4},
        {"BLOCK_SIZE": 64, "SPLIT_K": 4, "num_warps": 4},
    ]
    assert all("K" not in item.config for item in proposals)


def test_is_deterministic_bounded_deduplicated_and_non_mutating() -> None:
    duplicate = hypothesis("same family", {"BLOCK_SIZE": [128, 256]})
    model = WorldModel(
        [duplicate, hypothesis("duplicate config", {"BLOCK_SIZE": 128}, confidence=0.8)],
        include_builtins=False,
    )
    before_model = deepcopy(model.hypotheses)
    before_baseline = deepcopy(BASELINE)
    first = propose(model, max_candidates=1)
    second = propose(model, max_candidates=1)
    assert [dict(item.config) for item in first] == [dict(item.config) for item in second]
    assert len(first) == 1
    assert model.hypotheses == before_model
    assert BASELINE == before_baseline


def test_rejects_invalid_baseline_and_shared_memory_candidates() -> None:
    model = WorldModel([hypothesis("wide", {"BLOCK_SIZE": [128, 256]})], include_builtins=False)
    with pytest.raises(ValueError, match="complete operator config"):
        propose(model, baseline_config={"BLOCK_SIZE": 64})
    with pytest.raises(ValueError, match="outside the operator space"):
        propose(model, baseline_config={**BASELINE, "BLOCK_SIZE": 999})
    proposals = propose(
        model,
        spec=spec(shared_memory_check=lambda config: config["BLOCK_SIZE"] < 256),
    )
    assert [item.config["BLOCK_SIZE"] for item in proposals] == [128]


def test_skips_unknown_and_out_of_domain_parameters() -> None:
    model = WorldModel(
        [
            hypothesis("shape only", {"K": 4096}),
            hypothesis("unsupported tile", {"BLOCK_SIZE": 512}),
        ],
        include_builtins=False,
    )
    assert propose(model) == []


def test_requires_durable_refs_for_every_usable_hypothesis() -> None:
    model = WorldModel([hypothesis("wide", {"BLOCK_SIZE": 128})], include_builtins=False)
    with pytest.raises(ValueError, match="lacks durable source refs"):
        propose(model, source_refs_by_hypothesis={})


def test_serializes_truthful_kernel_config_challengers() -> None:
    model = WorldModel([hypothesis("wide", {"BLOCK_SIZE": [128, 256]})], include_builtins=False)
    proposals = propose(model)
    challengers = build_kernel_improvement_challengers(
        proposals,
        generator=GeneratorIdentity(
            id="noeris.world-model-v1",
            digest=f"sha256:{'a' * 64}",
        ),
    )
    assert len(challengers) == 2
    challenger = challengers[0]
    assert challenger["targetProject"] == "noeris"
    assert challenger["change"] == {
        "kind": "kernel_config",
        "knobs": {"BLOCK_SIZE": 128, "SPLIT_K": 1, "num_warps": 4},
    }
    assert challenger["hypothesis"]["sourceRefs"] == ["artifact:sealed-world-model:0"]
