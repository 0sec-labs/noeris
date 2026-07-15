from __future__ import annotations

from dataclasses import replace

import pytest

from research_engine.models import Hypothesis
from research_engine.zero_research import (
    CandidateProposal,
    ImprovementEnvelope,
    build_improvement_candidates,
    freeze_allowed_knobs,
)


def hypothesis(title: str, priority: float) -> Hypothesis:
    return Hypothesis(
        title=title,
        rationale=f"Evidence-backed rationale for {title}",
        novelty_reason="Targets a confirmed failure cluster.",
        expected_signal="Higher held-out success without more noise.",
        priority_score=priority,
    )


def envelope() -> ImprovementEnvelope:
    return ImprovementEnvelope(
        project="pwnkit",
        manifest_id="cybergym-v1",
        evaluator_digest="sha256:evaluator",
        development_corpus_digest="sha256:development",
        held_out_corpus_digest="sha256:held-out",
        negative_control_corpus_digest="sha256:foxguard-controls",
        development_case_ids=("dev-1", "dev-2"),
        held_out_case_ids=("held-1", "held-2"),
        negative_control_case_ids=("control-1",),
        min_held_out_cases=20,
        min_negative_control_cases=20,
        min_success_rate_delta=0.1,
        max_false_positive_rate_delta=0.0,
        max_cost_per_success_increase_ratio=0.2,
        max_inconclusive_rate=0.1,
        require_significance=True,
        max_runs=40,
        max_usd=100,
        max_wall_clock_minutes=180,
        allowed_knobs_by_kind=freeze_allowed_knobs(
            {"prompt": {"source_audit.hypothesis"}, "feature_flag": {"web_search"}}
        ),
        max_candidates=1,
    )


def proposal(title: str, text: str = "Inspect parser seeds first.") -> CandidateProposal:
    return CandidateProposal(
        hypothesis_title=title,
        change_kind="prompt",
        knobs={"source_audit.hypothesis": text},
        source_refs=("github:0sec-labs/0sec#1026",),
    )


def test_emits_highest_priority_candidate_in_0brain_schema() -> None:
    candidates = build_improvement_candidates(
        [hypothesis("lower", 0.2), hypothesis("higher", 0.9)],
        [proposal("lower"), proposal("higher")],
        envelope(),
        created_at="2026-07-15T21:00:00Z",
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["project"] == "pwnkit"
    assert candidate["hypothesis"]["statement"] == "higher"
    assert candidate["evaluation"]["evaluatorDigest"] == "sha256:evaluator"
    assert candidate["authority"] == {
        "mode": "draft_pr_only",
        "evaluatorChangesAllowed": False,
        "autoMergeAllowed": False,
        "externalPublicationAllowed": False,
    }


def test_candidate_id_is_deterministic_for_same_intervention() -> None:
    args = ([hypothesis("focused", 1.0)], [proposal("focused")], envelope())
    first = build_improvement_candidates(
        *args, created_at="2026-07-15T21:00:00Z"
    )[0]
    second = build_improvement_candidates(
        *args, created_at="2026-07-16T21:00:00Z"
    )[0]
    assert first["id"] == second["id"]


def test_emits_diverse_challengers_for_the_same_hypothesis() -> None:
    env = replace(envelope(), max_candidates=5)
    candidates = build_improvement_candidates(
        [hypothesis("parser cluster", 1.0)],
        [
            proposal("parser cluster", "Inspect parser seeds first."),
            proposal("parser cluster", "Trace state transitions first."),
        ],
        env,
        created_at="2026-07-15T21:00:00Z",
    )
    assert len(candidates) == 2
    assert len({candidate["id"] for candidate in candidates}) == 2


def test_rejects_duplicate_interventions() -> None:
    with pytest.raises(ValueError, match="duplicate candidate intervention"):
        build_improvement_candidates(
            [hypothesis("focused", 1.0)],
            [proposal("focused"), proposal("focused")],
            replace(envelope(), max_candidates=5),
            created_at="2026-07-15T21:00:00Z",
        )


def test_rejects_knob_not_allowlisted_by_evaluator() -> None:
    unsafe = CandidateProposal(
        hypothesis_title="focused",
        change_kind="prompt",
        knobs={"evaluator.threshold": 0},
        source_refs=("artifact:failure.json",),
    )
    with pytest.raises(ValueError, match="unallowlisted knobs"):
        build_improvement_candidates(
            [hypothesis("focused", 1.0)],
            [unsafe],
            envelope(),
            created_at="2026-07-15T21:00:00Z",
        )


def test_rejects_corpus_leakage() -> None:
    leaking = replace(envelope(), held_out_case_ids=("dev-1",))
    with pytest.raises(ValueError, match="partitions must be disjoint"):
        build_improvement_candidates(
            [hypothesis("focused", 1.0)],
            [proposal("focused")],
            leaking,
            created_at="2026-07-15T21:00:00Z",
        )


def test_rejects_unknown_hypothesis() -> None:
    with pytest.raises(ValueError, match="unknown hypothesis"):
        build_improvement_candidates(
            [hypothesis("known", 1.0)],
            [proposal("invented")],
            envelope(),
            created_at="2026-07-15T21:00:00Z",
        )


def test_rejects_non_finite_knob_value() -> None:
    invalid = CandidateProposal(
        hypothesis_title="focused",
        change_kind="prompt",
        knobs={"source_audit.hypothesis": float("nan")},
        source_refs=("artifact:failure.json",),
    )
    with pytest.raises(ValueError, match="must be finite"):
        build_improvement_candidates(
            [hypothesis("focused", 1.0)],
            [invalid],
            envelope(),
            created_at="2026-07-15T21:00:00Z",
        )
