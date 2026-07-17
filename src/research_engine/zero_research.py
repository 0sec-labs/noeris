"""Generator-only boundary from Noeris into the 0research controller.

Noeris may propose a bounded intervention and cite why it is worth testing. It
must not choose its budget, evaluator, corpora, scores, or promotion authority.
Those fields are injected and validated by the trusted 0brain controller.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Mapping

from .models import Hypothesis


ALLOWED_PROJECTS = frozenset({"pwnkit", "0verse", "foxguard", "noeris", "0brain"})
ALLOWED_CHANGE_KINDS = frozenset(
    {"prompt", "feature_flag", "routing", "ranking", "scheduler", "kernel_config"}
)
MAX_SAFE_INTEGER = 2**53 - 1
SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
SAFE_GENERATOR_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")


@dataclass(frozen=True, slots=True)
class GeneratorIdentity:
    """Controller-pinned identity for the code producing challengers."""

    id: str
    digest: str


@dataclass(frozen=True, slots=True)
class CandidateProposal:
    """A proposed bounded intervention for one known Noeris hypothesis."""

    hypothesis_title: str
    change_kind: str
    knobs: Mapping[str, str | int | bool]
    source_refs: tuple[str, ...]


def build_improvement_challengers(
    hypotheses: list[Hypothesis],
    proposals: list[CandidateProposal],
    *,
    target_project: str,
    generator: GeneratorIdentity,
    max_candidates: int = 5,
) -> list[dict[str, object]]:
    """Emit at most five content-addressed, generator-owned challengers.

    The result intentionally has no budget, evaluation, measured outcome, or
    authority fields. Invalid proposals fail closed rather than being widened.
    """

    _validate_boundary(target_project, generator, max_candidates)
    by_title = _hypotheses_by_title(hypotheses)
    seen_interventions: set[str] = set()
    ranked: list[tuple[Hypothesis, CandidateProposal]] = []

    for proposal in proposals:
        hypothesis = by_title.get(proposal.hypothesis_title)
        if hypothesis is None:
            raise ValueError(
                f"proposal references unknown hypothesis: {proposal.hypothesis_title!r}"
            )
        _validate_proposal(proposal)
        intervention = _canonical_json(
            {
                "hypothesis": proposal.hypothesis_title,
                "kind": proposal.change_kind,
                "knobs": dict(proposal.knobs),
            }
        )
        if intervention in seen_interventions:
            raise ValueError("duplicate candidate intervention")
        seen_interventions.add(intervention)
        ranked.append((hypothesis, proposal))

    ranked.sort(
        key=lambda pair: (
            -pair[0].priority_score,
            pair[0].title,
            pair[1].change_kind,
            _canonical_json(dict(pair[1].knobs)),
        )
    )
    return [
        _challenger(hypothesis, proposal, target_project, generator)
        for hypothesis, proposal in ranked[:max_candidates]
    ]


def _challenger(
    hypothesis: Hypothesis,
    proposal: CandidateProposal,
    target_project: str,
    generator: GeneratorIdentity,
) -> dict[str, object]:
    content: dict[str, object] = {
        "schemaVersion": 1,
        "kind": "improvement_challenger",
        "targetProject": target_project,
        "generator": {"id": generator.id, "digest": generator.digest},
        "hypothesis": {
            "statement": hypothesis.title,
            "rationale": hypothesis.rationale,
            "sourceRefs": list(proposal.source_refs),
        },
        "change": {
            "kind": proposal.change_kind,
            "knobs": dict(proposal.knobs),
        },
    }
    digest = hashlib.sha256(_canonical_json(content).encode()).hexdigest()
    return {**content, "id": f"imp_{target_project}_{digest}"}


def _canonical_json(value: object) -> str:
    """Canonical JSON shared with 0brain for the deliberately narrow schema."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


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
        if not hypothesis.rationale.strip():
            raise ValueError(f"hypothesis rationale must not be empty: {title!r}")
        result[title] = hypothesis
    return result


def _validate_proposal(proposal: CandidateProposal) -> None:
    if proposal.change_kind not in ALLOWED_CHANGE_KINDS:
        raise ValueError(f"unsupported change kind: {proposal.change_kind!r}")
    if not proposal.knobs:
        raise ValueError("proposal knobs must not be empty")
    for name, value in proposal.knobs.items():
        if not name.strip():
            raise ValueError("proposal knob name must not be empty")
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            if abs(value) > MAX_SAFE_INTEGER:
                raise ValueError(f"proposal knob {name!r} exceeds the safe integer range")
            continue
        if isinstance(value, str):
            if len(value) > 4_000:
                raise ValueError(f"proposal knob {name!r} exceeds 4000 characters")
            continue
        raise ValueError(f"proposal knob {name!r} must be a string, boolean, or safe integer")
    if not proposal.source_refs or any(not ref.strip() for ref in proposal.source_refs):
        raise ValueError("proposal source_refs must contain non-empty evidence references")
    if len(set(proposal.source_refs)) != len(proposal.source_refs):
        raise ValueError("proposal source_refs contains duplicates")


def _validate_boundary(
    target_project: str,
    generator: GeneratorIdentity,
    max_candidates: int,
) -> None:
    if target_project not in ALLOWED_PROJECTS:
        raise ValueError(f"unsupported target project: {target_project!r}")
    if not SAFE_GENERATOR_ID.fullmatch(generator.id):
        raise ValueError("generator id must be lowercase and filesystem safe")
    if not SHA256_DIGEST.fullmatch(generator.digest):
        raise ValueError("generator digest must be a lowercase sha256 digest")
    if not 1 <= max_candidates <= 5:
        raise ValueError("max_candidates must be in [1, 5]")
