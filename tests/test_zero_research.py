from __future__ import annotations

from dataclasses import replace

import pytest

from research_engine.models import Hypothesis
from research_engine.zero_research import (
    CandidateProposal,
    GeneratorIdentity,
    build_improvement_challengers,
)


GENERATOR = GeneratorIdentity(
    id="noeris.world-model-v1",
    digest=f"sha256:{'a' * 64}",
)


def hypothesis(title: str, priority: float) -> Hypothesis:
    return Hypothesis(
        title=title,
        rationale=f"Evidence-backed rationale for {title}",
        novelty_reason="Targets a confirmed failure cluster.",
        expected_signal="Higher held-out success without more noise.",
        priority_score=priority,
    )


def proposal(title: str, text: str = "Inspect parser seeds first.") -> CandidateProposal:
    return CandidateProposal(
        hypothesis_title=title,
        change_kind="prompt",
        knobs={"source_audit.hypothesis": text},
        source_refs=("artifact:failure-cluster:abc123",),
    )


def build(hypotheses: list[Hypothesis], proposals: list[CandidateProposal], **kwargs):
    return build_improvement_challengers(
        hypotheses,
        proposals,
        target_project="pwnkit",
        generator=GENERATOR,
        **kwargs,
    )


def test_emits_only_generator_owned_fields_in_priority_order() -> None:
    candidates = build(
        [hypothesis("lower", 0.2), hypothesis("higher", 0.9)],
        [proposal("lower"), proposal("higher")],
        max_candidates=1,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["targetProject"] == "pwnkit"
    assert candidate["hypothesis"]["statement"] == "higher"
    assert set(candidate) == {
        "schemaVersion", "kind", "id", "targetProject", "generator", "hypothesis", "change"
    }
    assert not ({"budget", "evaluation", "authority", "score"} & set(candidate))


def test_challenger_id_binds_all_generator_owned_content() -> None:
    first = build([hypothesis("focused", 1.0)], [proposal("focused")])[0]
    second = build([hypothesis("focused", 1.0)], [proposal("focused")])[0]
    changed = build(
        [replace(hypothesis("focused", 1.0), rationale="Different rationale")],
        [proposal("focused")],
    )[0]
    assert first["id"] == second["id"]
    assert first["id"] != changed["id"]
    assert len(first["id"].removeprefix("imp_pwnkit_")) == 64


def test_emits_diverse_challengers_and_caps_batch() -> None:
    candidates = build(
        [hypothesis("parser cluster", 1.0)],
        [
            proposal("parser cluster", f"Intervention {index}")
            for index in range(7)
        ],
    )
    assert len(candidates) == 5
    assert len({candidate["id"] for candidate in candidates}) == 5


def test_rejects_duplicate_interventions() -> None:
    with pytest.raises(ValueError, match="duplicate candidate intervention"):
        build(
            [hypothesis("focused", 1.0)],
            [proposal("focused"), proposal("focused")],
        )


def test_rejects_unknown_hypothesis_or_unbounded_change() -> None:
    with pytest.raises(ValueError, match="unknown hypothesis"):
        build([hypothesis("known", 1.0)], [proposal("invented")])
    with pytest.raises(ValueError, match="unsupported change kind"):
        build(
            [hypothesis("known", 1.0)],
            [replace(proposal("known"), change_kind="code_rewrite")],
        )


@pytest.mark.parametrize("value", [1.5, float("nan"), 2**53])
def test_rejects_non_interoperable_number(value: object) -> None:
    invalid = CandidateProposal(
        hypothesis_title="focused",
        change_kind="prompt",
        knobs={"source_audit.limit": value},
        source_refs=("artifact:failure.json",),
    )
    with pytest.raises(ValueError, match="safe integer|string, boolean"):
        build([hypothesis("focused", 1.0)], [invalid])


def test_rejects_unpinned_generator_or_oversized_policy() -> None:
    with pytest.raises(ValueError, match="lowercase sha256"):
        build_improvement_challengers(
            [hypothesis("focused", 1.0)],
            [proposal("focused")],
            target_project="pwnkit",
            generator=replace(GENERATOR, digest="sha256:not-real"),
        )
    with pytest.raises(ValueError, match="max_candidates"):
        build([hypothesis("focused", 1.0)], [proposal("focused")], max_candidates=6)
