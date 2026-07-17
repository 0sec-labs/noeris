"""Deterministic, generator-only kernel configuration proposals.

This module turns world-model hypotheses into complete executable configuration
challengers. It never benchmarks, scores, updates the world model, or promotes a
candidate; those responsibilities remain with independent controller/evaluator
layers.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from typing import Mapping

from .models import Hypothesis
from .triton_operators import TritonOperatorSpec
from .world_model import ConfigHypothesis, WorldModel
from .zero_research import (
    CandidateProposal,
    GeneratorIdentity,
    build_improvement_challengers,
)


@dataclass(frozen=True, slots=True)
class KernelConfigProposal:
    """One complete config plus the provenance for why it was proposed."""

    config: Mapping[str, int]
    hypothesis: str
    rationale: str
    confidence: float
    hypothesis_source: str
    source_refs: tuple[str, ...]


def propose_kernel_configs(
    *,
    spec: TritonOperatorSpec,
    baseline_config: Mapping[str, int],
    shape: Mapping[str, object],
    hardware: str,
    world_model: WorldModel,
    source_refs_by_hypothesis: Mapping[str, tuple[str, ...]],
    max_candidates: int = 5,
) -> list[KernelConfigProposal]:
    """Expand matching hypotheses into at most five valid complete configs."""

    if not 1 <= max_candidates <= 5:
        raise ValueError("max_candidates must be in [1, 5]")
    baseline = _validate_baseline(spec, baseline_config)
    proposals: list[KernelConfigProposal] = []
    seen_configs: set[str] = set()

    hypotheses = sorted(
        world_model.hypotheses,
        key=lambda item: (-item.confidence, item.description, item.source),
    )
    for hypothesis in hypotheses:
        if not hypothesis.matches_context(
            operator=spec.name,
            shape=dict(shape),
            hardware=hardware,
        ):
            continue
        overlays = _expand_valid_overlays(spec, hypothesis)
        if not overlays:
            continue
        source_refs = source_refs_by_hypothesis.get(hypothesis.description)
        if not source_refs or any(not ref.strip() for ref in source_refs):
            raise ValueError(
                f"usable hypothesis lacks durable source refs: {hypothesis.description!r}"
            )
        if len(source_refs) != len(set(source_refs)):
            raise ValueError(
                f"hypothesis source refs contain duplicates: {hypothesis.description!r}"
            )
        if not math.isfinite(hypothesis.confidence) or not 0 <= hypothesis.confidence <= 1:
            raise ValueError(
                f"hypothesis confidence must be finite and in [0, 1]: {hypothesis.description!r}"
            )

        for overlay in overlays:
            config = {**baseline, **overlay}
            if config == baseline or not spec.shared_memory_check_fn(config):
                continue
            canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
            if canonical in seen_configs:
                continue
            seen_configs.add(canonical)
            proposals.append(
                KernelConfigProposal(
                    config=config,
                    hypothesis=hypothesis.description,
                    rationale=hypothesis.predicted_effect,
                    confidence=hypothesis.confidence,
                    hypothesis_source=hypothesis.source,
                    source_refs=tuple(source_refs),
                )
            )
            if len(proposals) >= max_candidates:
                return proposals
    return proposals


def build_kernel_improvement_challengers(
    proposals: list[KernelConfigProposal],
    *,
    generator: GeneratorIdentity,
    max_candidates: int = 5,
) -> list[dict[str, object]]:
    """Serialize kernel proposals through the common Noeris boundary."""

    hypotheses_by_title: dict[str, Hypothesis] = {}
    candidate_proposals: list[CandidateProposal] = []
    for proposal in proposals:
        existing = hypotheses_by_title.get(proposal.hypothesis)
        generated = Hypothesis(
            title=proposal.hypothesis,
            rationale=proposal.rationale,
            novelty_reason="World-model-guided configuration challenger.",
            expected_signal="Independent held-out benchmark improvement.",
            priority_score=proposal.confidence,
            ranking_rationale=f"source={proposal.hypothesis_source}",
        )
        if existing is not None and existing != generated:
            raise ValueError(
                f"kernel proposals disagree about hypothesis metadata: {proposal.hypothesis!r}"
            )
        hypotheses_by_title[proposal.hypothesis] = generated
        candidate_proposals.append(
            CandidateProposal(
                hypothesis_title=proposal.hypothesis,
                change_kind="kernel_config",
                knobs=dict(proposal.config),
                source_refs=proposal.source_refs,
            )
        )
    return build_improvement_challengers(
        list(hypotheses_by_title.values()),
        candidate_proposals,
        target_project="noeris",
        generator=generator,
        max_candidates=max_candidates,
    )


def _validate_baseline(
    spec: TritonOperatorSpec,
    baseline_config: Mapping[str, int],
) -> dict[str, int]:
    expected = set(spec.param_space)
    actual = set(baseline_config)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"baseline must be a complete operator config; missing={missing!r}, unknown={unknown!r}"
        )
    baseline: dict[str, int] = {}
    for name in sorted(spec.param_space):
        value = baseline_config[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"baseline parameter {name!r} must be an integer")
        if value not in spec.param_space[name]:
            raise ValueError(f"baseline parameter {name!r} is outside the operator space")
        baseline[name] = value
    if not spec.shared_memory_check_fn(baseline):
        raise ValueError("baseline fails the operator shared-memory constraint")
    return baseline


def _expand_valid_overlays(
    spec: TritonOperatorSpec,
    hypothesis: ConfigHypothesis,
) -> list[dict[str, int]]:
    suggestion = hypothesis.config_suggestion()
    keys = sorted(set(suggestion) & set(spec.param_space))
    if not keys:
        return []
    values: list[list[int]] = []
    for key in keys:
        raw_values = suggestion[key] if isinstance(suggestion[key], list) else [suggestion[key]]
        accepted: list[int] = []
        for value in raw_values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"hypothesis parameter {key!r} must contain only integer values"
                )
            if value not in spec.param_space[key]:
                continue
            if value not in accepted:
                accepted.append(value)
        if not accepted:
            return []
        values.append(sorted(accepted))
    return [dict(zip(keys, combination, strict=True)) for combination in itertools.product(*values)]
