"""Bounded adapter from Noeris hypotheses to 0research improvement candidates.

Noeris proposes hypotheses and allowlisted knob values.  The caller (0brain)
owns the immutable evaluation envelope, budgets, corpus partition, and
promotion authority.  Keeping those responsibilities separate prevents a
candidate generator from weakening the test it is expected to pass.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Mapping

from .models import Hypothesis


ALLOWED_PROJECTS = frozenset({"pwnkit", "0verse", "foxguard", "noeris", "0brain"})
ALLOWED_CHANGE_KINDS = frozenset(
    {"prompt", "feature_flag", "routing", "ranking", "scheduler"}
)


@dataclass(frozen=True, slots=True)
class CandidateProposal:
    """A proposed bounded intervention for one known Noeris hypothesis."""

    hypothesis_title: str
    change_kind: str
    knobs: Mapping[str, str | int | float | bool]
    source_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ImprovementEnvelope:
    """Evaluator-owned constraints which candidate generation cannot alter."""

    project: str
    manifest_id: str
    evaluator_digest: str
    development_corpus_digest: str
    held_out_corpus_digest: str
    negative_control_corpus_digest: str
    development_case_ids: tuple[str, ...]
    held_out_case_ids: tuple[str, ...]
    negative_control_case_ids: tuple[str, ...]
    min_held_out_cases: int
    min_negative_control_cases: int
    min_success_rate_delta: float
    max_false_positive_rate_delta: float
    max_cost_per_success_increase_ratio: float
    max_inconclusive_rate: float
    require_significance: bool
    max_runs: int
    max_usd: float
    max_wall_clock_minutes: int
    allowed_knobs_by_kind: Mapping[str, frozenset[str]]
    max_candidates: int = 5


def freeze_allowed_knobs(
    value: Mapping[str, set[str] | frozenset[str]],
) -> Mapping[str, frozenset[str]]:
    """Return an immutable allowlist suitable for an ImprovementEnvelope."""

    return MappingProxyType({kind: frozenset(knobs) for kind, knobs in value.items()})


def build_improvement_candidates(
    hypotheses: list[Hypothesis],
    proposals: list[CandidateProposal],
    envelope: ImprovementEnvelope,
    *,
    created_at: str,
) -> list[dict[str, object]]:
    """Emit schema-v1 0research candidates ordered by hypothesis priority.

    This function deliberately does not invent evaluator settings, corpus
    membership, authority, or budget.  Invalid or unallowlisted proposals fail
    closed rather than being silently broadened.
    """

    _validate_envelope(envelope)
    _validate_created_at(created_at)
    by_title = _hypotheses_by_title(hypotheses)
    seen_interventions: set[str] = set()
    ranked: list[tuple[Hypothesis, CandidateProposal]] = []

    for proposal in proposals:
        hypothesis = by_title.get(proposal.hypothesis_title)
        if hypothesis is None:
            raise ValueError(
                f"proposal references unknown hypothesis: {proposal.hypothesis_title!r}"
            )
        _validate_proposal(proposal, envelope)
        intervention = _intervention_key(proposal)
        if intervention in seen_interventions:
            raise ValueError("duplicate candidate intervention")
        seen_interventions.add(intervention)
        ranked.append((hypothesis, proposal))

    ranked.sort(
        key=lambda pair: (
            -pair[0].priority_score,
            pair[0].title,
            pair[1].change_kind,
            _intervention_key(pair[1]),
        )
    )
    selected = ranked[: envelope.max_candidates]
    return [
        _candidate(hypothesis, proposal, envelope, created_at)
        for hypothesis, proposal in selected
    ]


def _candidate(
    hypothesis: Hypothesis,
    proposal: CandidateProposal,
    envelope: ImprovementEnvelope,
    created_at: str,
) -> dict[str, object]:
    canonical_change = json.dumps(
        {
            "project": envelope.project,
            "hypothesis": hypothesis.title,
            "kind": proposal.change_kind,
            "knobs": dict(proposal.knobs),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical_change.encode()).hexdigest()[:12]
    return {
        "schemaVersion": 1,
        "id": f"imp_{envelope.project}_{digest}",
        "project": envelope.project,
        "createdAt": created_at,
        "hypothesis": {
            "statement": hypothesis.title,
            "rationale": hypothesis.rationale,
            "sourceRefs": list(proposal.source_refs),
        },
        "change": {
            "kind": proposal.change_kind,
            "knobs": dict(proposal.knobs),
        },
        "budget": {
            "maxRuns": envelope.max_runs,
            "maxUsd": envelope.max_usd,
            "maxWallClockMinutes": envelope.max_wall_clock_minutes,
        },
        "evaluation": {
            "manifestId": envelope.manifest_id,
            "evaluatorDigest": envelope.evaluator_digest,
            "developmentCorpusDigest": envelope.development_corpus_digest,
            "heldOutCorpusDigest": envelope.held_out_corpus_digest,
            "negativeControlCorpusDigest": envelope.negative_control_corpus_digest,
            "developmentCaseIds": list(envelope.development_case_ids),
            "heldOutCaseIds": list(envelope.held_out_case_ids),
            "negativeControlCaseIds": list(envelope.negative_control_case_ids),
            "minHeldOutCases": envelope.min_held_out_cases,
            "minNegativeControlCases": envelope.min_negative_control_cases,
            "minSuccessRateDelta": envelope.min_success_rate_delta,
            "maxFalsePositiveRateDelta": envelope.max_false_positive_rate_delta,
            "maxCostPerSuccessIncreaseRatio": (
                envelope.max_cost_per_success_increase_ratio
            ),
            "maxInconclusiveRate": envelope.max_inconclusive_rate,
            "requireSignificance": envelope.require_significance,
        },
        "authority": {
            "mode": "draft_pr_only",
            "evaluatorChangesAllowed": False,
            "autoMergeAllowed": False,
            "externalPublicationAllowed": False,
        },
    }


def _intervention_key(proposal: CandidateProposal) -> str:
    return json.dumps(
        {
            "hypothesis": proposal.hypothesis_title,
            "kind": proposal.change_kind,
            "knobs": dict(proposal.knobs),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _hypotheses_by_title(hypotheses: list[Hypothesis]) -> dict[str, Hypothesis]:
    result: dict[str, Hypothesis] = {}
    for hypothesis in hypotheses:
        title = hypothesis.title.strip()
        if not title:
            raise ValueError("hypothesis title must not be empty")
        if title in result:
            raise ValueError(f"duplicate hypothesis title: {title!r}")
        if not math.isfinite(hypothesis.priority_score):
            raise ValueError(f"hypothesis priority must be finite: {title!r}")
        result[title] = hypothesis
    return result


def _validate_proposal(
    proposal: CandidateProposal,
    envelope: ImprovementEnvelope,
) -> None:
    if proposal.change_kind not in ALLOWED_CHANGE_KINDS:
        raise ValueError(f"unsupported change kind: {proposal.change_kind!r}")
    allowed = envelope.allowed_knobs_by_kind.get(proposal.change_kind)
    if allowed is None:
        raise ValueError(f"change kind is not enabled by evaluator: {proposal.change_kind!r}")
    if not proposal.knobs:
        raise ValueError("proposal knobs must not be empty")
    unknown = set(proposal.knobs) - allowed
    if unknown:
        raise ValueError(f"proposal contains unallowlisted knobs: {sorted(unknown)!r}")
    for name, value in proposal.knobs.items():
        if not name.strip():
            raise ValueError("proposal knob name must not be empty")
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"proposal knob {name!r} must be scalar")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"proposal knob {name!r} must be finite")
        if isinstance(value, str) and len(value) > 4_000:
            raise ValueError(f"proposal knob {name!r} exceeds 4000 characters")
    if not proposal.source_refs or any(not ref.strip() for ref in proposal.source_refs):
        raise ValueError("proposal source_refs must contain non-empty evidence references")
    if len(set(proposal.source_refs)) != len(proposal.source_refs):
        raise ValueError("proposal source_refs contains duplicates")


def _validate_envelope(envelope: ImprovementEnvelope) -> None:
    if envelope.project not in ALLOWED_PROJECTS:
        raise ValueError(f"unsupported target project: {envelope.project!r}")
    for label, value in (
        ("manifest_id", envelope.manifest_id),
        ("evaluator_digest", envelope.evaluator_digest),
        ("development_corpus_digest", envelope.development_corpus_digest),
        ("held_out_corpus_digest", envelope.held_out_corpus_digest),
        ("negative_control_corpus_digest", envelope.negative_control_corpus_digest),
    ):
        if not value.strip():
            raise ValueError(f"{label} must not be empty")
    if not 1 <= envelope.max_candidates <= 5:
        raise ValueError("max_candidates must be in [1, 5]")
    if envelope.max_runs <= 0 or envelope.max_wall_clock_minutes <= 0:
        raise ValueError("run and wall-clock budgets must be positive")
    if not math.isfinite(envelope.max_usd) or envelope.max_usd < 0:
        raise ValueError("cost budget must be non-negative")
    if envelope.min_held_out_cases <= 0:
        raise ValueError("min_held_out_cases must be positive")
    if envelope.min_negative_control_cases <= 0:
        raise ValueError("min_negative_control_cases must be positive")
    for label, value in (
        ("min_success_rate_delta", envelope.min_success_rate_delta),
        ("max_false_positive_rate_delta", envelope.max_false_positive_rate_delta),
        ("max_inconclusive_rate", envelope.max_inconclusive_rate),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{label} must be in [0, 1]")
    if (
        not math.isfinite(envelope.max_cost_per_success_increase_ratio)
        or envelope.max_cost_per_success_increase_ratio < 0
    ):
        raise ValueError("max_cost_per_success_increase_ratio must be non-negative")
    _validate_partition("development", envelope.development_case_ids)
    _validate_partition("held-out", envelope.held_out_case_ids)
    _validate_partition("negative-control", envelope.negative_control_case_ids)
    development = set(envelope.development_case_ids)
    held_out = set(envelope.held_out_case_ids)
    controls = set(envelope.negative_control_case_ids)
    if development & held_out or controls & (development | held_out):
        raise ValueError("evaluation corpus partitions must be disjoint")
    invalid_kinds = set(envelope.allowed_knobs_by_kind) - ALLOWED_CHANGE_KINDS
    if invalid_kinds:
        raise ValueError(f"allowlist contains unsupported change kinds: {sorted(invalid_kinds)!r}")
    if any(not knobs for knobs in envelope.allowed_knobs_by_kind.values()):
        raise ValueError("each enabled change kind must allow at least one knob")


def _validate_partition(label: str, values: tuple[str, ...]) -> None:
    if not values or any(not value.strip() for value in values):
        raise ValueError(f"{label} case ids must be non-empty")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} case ids contain duplicates")


def _validate_created_at(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("created_at must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError("created_at must include a timezone")
