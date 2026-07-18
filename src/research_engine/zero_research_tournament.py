"""Untrusted Noeris kernel tournament proposal builder.

This dependency-injected seam exists for adapter development and tests. Its
output is deliberately a proposal schema that 0brain must never accept as
evidence. A separate fixed-policy verifier will eventually promote verified
raw artifacts and receipts into an accepted evidence contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping


DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{1,127}$")
AUTHORITY = {
    "promotionAllowed": False,
    "githubWriteAllowed": False,
    "autoMergeAllowed": False,
    "deploymentAllowed": False,
    "externalPublicationAllowed": False,
}


@dataclass(frozen=True, slots=True)
class ArmMeasurement:
    """Raw output from a controller-supplied in-process GPU runner."""

    reference_digest: str
    output_digests: tuple[str, ...]
    timings_ms: tuple[float, ...]
    max_absolute_error: float
    max_relative_error: float
    warmups_completed: int


TournamentRunner = Callable[[str, Mapping[str, int], Mapping[str, int], int, int, int, float, float], ArmMeasurement]
SignatureVerifier = Callable[[bytes, str, str, str, str], None]
EvidenceAttestor = Callable[[bytes, str, str], Mapping[str, str]]


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256(value: str | bytes) -> str:
    payload = value.encode() if isinstance(value, str) else value
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def domain(name: str, value: object) -> str:
    return sha256(f"{name}\0{canonical_json(value)}\n")


def verify_ssh_signature(material: bytes, signature: str, principal: str, namespace: str, policy: str) -> None:
    supplied = Path(policy).absolute()
    policy_path = Path(policy).resolve(strict=True)
    if supplied != policy_path:
        raise ValueError("controller signer policy must be a canonical path")
    descriptor = os.open(policy_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before_info = os.fstat(descriptor)
        if not stat.S_ISREG(before_info.st_mode) or before_info.st_nlink != 1 or before_info.st_uid != 0 or before_info.st_size == 0 or before_info.st_size > 64 * 1024 or before_info.st_mode & 0o022:
            raise ValueError("controller signer policy must be a bounded root-owned canonical file")
        policy_bytes = os.read(descriptor, before_info.st_size + 1)
        after_info = os.fstat(descriptor)
        if (before_info.st_dev, before_info.st_ino, before_info.st_size, before_info.st_mtime_ns) != (after_info.st_dev, after_info.st_ino, after_info.st_size, after_info.st_mtime_ns):
            raise ValueError("controller signer policy changed while read")
    finally:
        os.close(descriptor)
    parent = policy_path.parent
    parent_info = parent.lstat()
    if not parent.is_dir() or parent.is_symlink() or parent_info.st_uid != 0 or parent_info.st_mode & 0o022:
        raise ValueError("controller signer policy parent must be root-owned and non-writable")
    with tempfile.TemporaryDirectory(prefix="noeris-signature-") as directory:
        signature_path = Path(directory) / "signature.ssh"
        policy_snapshot = Path(directory) / "allowed_signers"
        signature_path.write_text(signature, encoding="utf-8")
        policy_snapshot.write_bytes(policy_bytes)
        result = subprocess.run(
            ["/usr/bin/ssh-keygen", "-Y", "verify", "-f", str(policy_snapshot), "-I", principal, "-n", namespace, "-s", str(signature_path)],
            input=material, capture_output=True, timeout=10, check=False,
        )
        if result.returncode != 0:
            raise ValueError("controller signature is invalid")


def _build_untrusted_tournament_proposal(
    *,
    plan_value: object,
    authorization_value: object,
    candidate_value: object,
    allocation_id: str,
    environment_value: object,
    worker_identity_value: object,
    worker_usage_value: object,
    controller_policy: str,
    worker_policy: str,
    runner: TournamentRunner,
    attestor: EvidenceAttestor,
    verifier: SignatureVerifier = verify_ssh_signature,
) -> dict[str, object]:
    """Build a signed but categorically unaccepted adapter-development proposal."""

    candidate = _object(candidate_value, "candidate")
    plan = _object(plan_value, "tournament plan")
    authorization = _object(authorization_value, "controller authorization")
    _validate_candidate(candidate)
    _validate_authorization(authorization)
    _exact(plan, ["arms", "authority", "budget", "candidateDigest", "candidateId", "contract", "controllerAuthorization", "evaluator", "generator", "hardware", "manifest", "novelty", "operator", "planDigest", "planId", "randomizationDigest", "repository", "rounds", "schemaVersion"], "tournament plan")
    if plan.get("schemaVersion") != 1 or plan.get("contract") != "noeris-kernel-tournament-plan-v1" or plan.get("authority") != AUTHORITY:
        raise ValueError("tournament plan identity or authority is invalid")
    plan_body = {key: value for key, value in plan.items() if key not in {"planDigest", "planId"}}
    expected_plan_digest = sha256(canonical_json(plan_body))
    if plan.get("planDigest") != expected_plan_digest or plan.get("planId") != f"noeris-{expected_plan_digest[7:]}":
        raise ValueError("tournament plan digest is invalid")
    candidate_digest = sha256(canonical_json(candidate))
    if plan.get("candidateDigest") != candidate_digest or plan.get("candidateId") != candidate.get("id"):
        raise ValueError("tournament plan does not bind the exact candidate")
    change = _object(candidate.get("change"), "candidate change")
    if candidate.get("project") != "noeris" or change.get("kind") != "kernel_config":
        raise ValueError("candidate is not a Noeris kernel configuration")

    controller = _object(plan.get("controllerAuthorization"), "controller authorization ref")
    _exact(controller, ["namespace", "principal", "ref", "sha256"], "controller authorization ref")
    authorization_digest = sha256(canonical_json(authorization))
    if controller.get("sha256") != authorization_digest or controller.get("ref") != f"0research-noeris-tournament-controller-envelope-v1:{authorization_digest}":
        raise ValueError("signed controller authorization bytes drift from the plan")
    signature = _text(authorization.get("signatureSsh"), "controller signature")
    principal = _text(authorization.get("controllerPrincipal"), "controller principal")
    if principal != controller.get("principal") or controller.get("namespace") != "0research-noeris-tournament-plan-v1":
        raise ValueError("controller signature identity drifts from the plan")
    unsigned = {key: value for key, value in authorization.items() if key != "signatureSsh"}
    verifier(canonical_json(unsigned).encode(), signature, principal, "0research-noeris-tournament-plan-v1", controller_policy)
    if authorization.get("candidateDigest") != candidate_digest:
        raise ValueError("signed controller authorization does not bind the candidate")

    _cross_check_authorization(plan, authorization, candidate)
    _validate_plan_components(plan, authorization, candidate)
    rounds = plan.get("rounds")
    if not isinstance(rounds, list):
        raise ValueError("tournament rounds must be an array")
    selected = [item for item in rounds if isinstance(item, dict) and item.get("allocationId") == allocation_id]
    if len(selected) != 1:
        raise ValueError("allocation id must select exactly one planned round")
    round_value = selected[0]
    seed = _positive_integer(round_value.get("seed"), "round seed")
    manifest = _object(plan.get("manifest"), "manifest")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest cases must be non-empty")
    orders = round_value.get("armOrders")
    if not isinstance(orders, list) or len(orders) != len(cases):
        raise ValueError("round arm orders do not cover the exact manifest")
    arms = _object(plan.get("arms"), "arms")
    evaluator = _object(plan.get("evaluator"), "evaluator")
    samples = _positive_integer(evaluator.get("samples"), "evaluator samples")
    warmups = _positive_integer(evaluator.get("warmups"), "evaluator warmups")
    environment = _environment(environment_value, evaluator, str(plan.get("hardware")))
    worker_identity = _worker_identity(worker_identity_value, plan)

    started = time.monotonic()
    results: list[dict[str, object]] = []
    for case, order_value in zip(cases, orders, strict=True):
        case_object = _object(case, "manifest case")
        order_object = _object(order_value, "arm order")
        if order_object.get("caseId") != case_object.get("id") or sorted(order_object.get("order", [])) != ["challenger", "champion"]:
            raise ValueError("arm order is invalid or drifts from the manifest")
        shape = _integer_object(case_object.get("shape"), "case shape")
        case_results: list[dict[str, object]] = []
        for arm_id in order_object["order"]:
            arm = _object(arms.get(arm_id), f"{arm_id} arm")
            measurement = runner(str(arm_id), _integer_object(arm.get("config"), f"{arm_id} config"), shape, _case_seed(seed, _positive_integer(case_object.get("tensorSeed"), "tensor seed"), str(case_object.get("id"))), warmups, samples, float(evaluator["absoluteTolerance"]), float(evaluator["relativeTolerance"]))
            case_results.append(_measurement(str(case_object.get("id")), str(case_object.get("lane")), str(arm_id), measurement, samples, warmups, float(evaluator["absoluteTolerance"]), float(evaluator["relativeTolerance"])))
        if len({item["referenceDigest"] for item in case_results}) != 1:
            raise ValueError("champion and challenger did not use the same exact reference input")
        results.extend(case_results)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    usage_input = worker_usage_value(elapsed_ms) if callable(worker_usage_value) else worker_usage_value
    worker_usage = _worker_usage(usage_input)
    budget = _object(plan.get("budget"), "budget")
    measured_runs = len(results)
    if budget.get("maxUsd") != 0 or worker_usage["costUsd"] != 0 or measured_runs > _positive_integer(budget.get("maxRuns"), "run budget") or elapsed_ms > _positive_integer(budget.get("maxWallClockMinutes"), "wall-clock budget") * 60_000:
        raise ValueError("tournament round exceeded the zero-dollar controller budget")

    body = {
        "schemaVersion": 1, "contract": "noeris-kernel-tournament-proposal-v1", "acceptedBy0brain": False,
        "planId": plan["planId"], "planDigest": plan["planDigest"], "candidateId": plan["candidateId"], "candidateDigest": candidate_digest,
        "controllerAuthorization": controller, "allocationId": allocation_id, "seed": seed,
        "repository": plan["repository"], "operator": plan["operator"], "hardware": plan["hardware"], "environment": environment, "workerIdentity": worker_identity,
        "evaluator": evaluator, "manifest": {"id": manifest["id"], "digest": manifest["digest"], "corpusDigests": manifest["corpusDigests"]},
        "novelty": plan["novelty"], "arms": arms, "results": results,
        "usage": {"measuredArmCaseRuns": measured_runs, "timingSamples": measured_runs * samples, "elapsedMs": elapsed_ms, **worker_usage},
        "authority": AUTHORITY,
    }
    evidence_digest = sha256(canonical_json(body))
    signed_body = {**body, "evidenceDigest": evidence_digest}
    attestation = dict(attestor(canonical_json(signed_body).encode(), allocation_id, str(environment["deviceUuid"])))
    _exact(attestation, ["allocationId", "deviceUuid", "evidenceDigest", "namespace", "principal", "signatureSsh"], "allocation attestation")
    if attestation.get("allocationId") != allocation_id or attestation.get("deviceUuid") != environment["deviceUuid"] or attestation.get("evidenceDigest") != evidence_digest or attestation.get("namespace") != "0research-noeris-allocation-evidence-v1":
        raise ValueError("allocation attestation does not bind the exact evidence and device")
    _text(attestation.get("principal"), "allocation attestation principal")
    signature_value = _text(attestation.get("signatureSsh"), "allocation attestation signature")
    if not signature_value.startswith("-----BEGIN SSH SIGNATURE-----\n") or not signature_value.rstrip().endswith("-----END SSH SIGNATURE-----"):
        raise ValueError("allocation attestation signature is malformed")
    verifier(
        canonical_json(signed_body).encode(),
        signature_value,
        str(attestation["principal"]),
        "0research-noeris-allocation-evidence-v1",
        worker_policy,
    )
    return {**signed_body, "allocationAttestation": attestation}


def _cross_check_authorization(plan: dict[str, object], authorization: dict[str, object], candidate: dict[str, object]) -> None:
    for key in ("generator", "repository", "operator", "hardware", "evaluator"):
        if canonical_json(plan.get(key)) != canonical_json(authorization.get(key)):
            raise ValueError(f"plan {key} drifts from signed controller authorization")
    plan_manifest = _object(plan.get("manifest"), "plan manifest")
    authorization_manifest = _object(authorization.get("manifest"), "authorization manifest")
    for key in ("id", "digest", "cases"):
        if canonical_json(plan_manifest.get(key)) != canonical_json(authorization_manifest.get(key)):
            raise ValueError("plan manifest drifts from signed controller authorization")
    plan_arms = _object(plan.get("arms"), "plan arms")
    champion = _object(plan_arms.get("champion"), "champion arm")
    challenger = _object(plan_arms.get("challenger"), "challenger arm")
    candidate_change = _object(candidate.get("change"), "candidate change")
    if canonical_json(champion.get("config")) != canonical_json(authorization.get("championConfig")) or canonical_json(challenger.get("config")) != canonical_json(candidate_change.get("knobs")):
        raise ValueError("plan arms drift from signed champion or exact candidate")
    candidate_budget = _object(candidate.get("budget"), "candidate budget")
    plan_budget = _object(plan.get("budget"), "plan budget")
    for key in ("maxRuns", "maxUsd", "maxWallClockMinutes"):
        if plan_budget.get(key) != candidate_budget.get(key):
            raise ValueError("plan budget drifts from the exact candidate")
    auth_rounds = authorization.get("rounds")
    plan_rounds = plan.get("rounds")
    if not isinstance(auth_rounds, list) or not isinstance(plan_rounds, list) or [item.get("allocationId") for item in auth_rounds if isinstance(item, dict)] != [item.get("allocationId") for item in plan_rounds if isinstance(item, dict)]:
        raise ValueError("plan allocations drift from signed controller authorization")
    nonce = _text(authorization.get("randomizationNonce"), "randomization nonce")
    if plan.get("randomizationDigest") != sha256(nonce):
        raise ValueError("plan randomization digest drifts from signed controller authorization")
    cases = plan_manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("plan manifest cases are invalid")
    for index, (authorization_round, plan_round) in enumerate(zip(auth_rounds, plan_rounds, strict=True)):
        auth_item = _object(authorization_round, "authorization round")
        plan_item = _object(plan_round, "plan round")
        allocation = _text(auth_item.get("allocationId"), "allocation id")
        seed = int(hashlib.sha256(f"{nonce}\0{allocation}\0{index}".encode()).hexdigest()[:12], 16)
        expected_orders = []
        for case in cases:
            case_id = _text(_object(case, "manifest case").get("id"), "case id")
            first = hashlib.sha256(f"{nonce}\0{allocation}\0{case_id}".encode()).digest()[0] & 1
            expected_orders.append({"caseId": case_id, "order": ["champion", "challenger"] if first == 0 else ["challenger", "champion"]})
        if plan_item.get("seed") != seed or canonical_json(plan_item.get("armOrders")) != canonical_json(expected_orders):
            raise ValueError("plan seed or arm order is not controller-derived")


def _validate_candidate(candidate: dict[str, object]) -> None:
    _exact(candidate, ["authority", "budget", "change", "createdAt", "evaluation", "hypothesis", "id", "project", "schemaVersion"], "candidate")
    if candidate.get("schemaVersion") != 1 or candidate.get("project") != "noeris":
        raise ValueError("candidate identity is invalid")
    _timestamp(candidate.get("createdAt"), "candidate creation timestamp")
    if candidate.get("authority") != {"mode": "draft_pr_only", "evaluatorChangesAllowed": False, "autoMergeAllowed": False, "externalPublicationAllowed": False}:
        raise ValueError("candidate expands authority")


def _validate_authorization(authorization: dict[str, object]) -> None:
    _exact(authorization, ["candidateDigest", "championConfig", "controllerPrincipal", "evaluator", "generator", "hardware", "manifest", "novelty", "operator", "randomizationNonce", "repository", "rounds", "schemaVersion", "signatureSsh"], "controller authorization")
    if authorization.get("schemaVersion") != 1:
        raise ValueError("controller authorization schemaVersion must be 1")


def _validate_plan_components(plan: dict[str, object], authorization: dict[str, object], candidate: dict[str, object]) -> None:
    evaluator = _object(plan.get("evaluator"), "evaluator")
    _exact(evaluator, ["absoluteTolerance", "configDigest", "digest", "relativeTolerance", "samples", "softwareImageDigest", "warmups"], "evaluator")
    absolute = _bounded_float(evaluator.get("absoluteTolerance"), "absolute tolerance", 0, 0.01)
    relative = _bounded_float(evaluator.get("relativeTolerance"), "relative tolerance", 0, 0.01)
    samples = _bounded_integer(evaluator.get("samples"), "samples", 5, 100)
    warmups = _bounded_integer(evaluator.get("warmups"), "warmups", 1, 25)
    evaluator_config = {"absoluteTolerance": absolute, "relativeTolerance": relative, "samples": samples, "softwareImageDigest": _digest(evaluator.get("softwareImageDigest"), "software image digest"), "warmups": warmups}
    if evaluator.get("configDigest") != domain("noeris-kernel-evaluator-config-v1", evaluator_config):
        raise ValueError("evaluator config digest is not recomputable")

    manifest = _object(plan.get("manifest"), "manifest")
    _exact(manifest, ["cases", "corpusDigests", "digest", "id"], "manifest")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not 4 <= len(cases) <= 64:
        raise ValueError("manifest must contain 4 to 64 cases")
    normalized_cases = []
    for index, value in enumerate(cases):
        case = _object(value, f"case {index}")
        _exact(case, ["id", "lane", "shape", "tensorSeed"], f"case {index}")
        case_id = _safe_id(case.get("id"), f"case {index}.id")
        lane = _text(case.get("lane"), f"case {index}.lane")
        if lane not in {"development", "held_out", "negative_control"}:
            raise ValueError("manifest case lane is unsupported")
        normalized_cases.append({"id": case_id, "lane": lane, "shape": _integer_object(case.get("shape"), f"case {index}.shape"), "tensorSeed": _positive_integer(case.get("tensorSeed"), f"case {index}.tensorSeed")})
    if [item["id"] for item in normalized_cases] != sorted(item["id"] for item in normalized_cases) or len({item["id"] for item in normalized_cases}) != len(normalized_cases) or len({item["tensorSeed"] for item in normalized_cases}) != len(normalized_cases):
        raise ValueError("manifest case ids and tensor seeds must be unique and canonical")
    for lane in ("development", "held_out", "negative_control"):
        if not any(item["lane"] == lane for item in normalized_cases):
            raise ValueError(f"manifest lacks {lane} cases")
    corpus_digests = {
        "development": domain("noeris-kernel-corpus-v1", {"lane": "development", "cases": [item for item in normalized_cases if item["lane"] == "development"]}),
        "heldOut": domain("noeris-kernel-corpus-v1", {"lane": "held_out", "cases": [item for item in normalized_cases if item["lane"] == "held_out"]}),
        "negativeControl": domain("noeris-kernel-corpus-v1", {"lane": "negative_control", "cases": [item for item in normalized_cases if item["lane"] == "negative_control"]}),
    }
    if manifest.get("corpusDigests") != corpus_digests or manifest.get("digest") != domain("noeris-kernel-tournament-manifest-v1", {"id": manifest.get("id"), "corpusDigests": corpus_digests, "cases": normalized_cases}):
        raise ValueError("manifest or lane corpus digest is not recomputable")
    evaluation = _object(candidate.get("evaluation"), "candidate evaluation")
    expected_pins = {"manifestDigest": manifest.get("digest"), "developmentCorpusDigest": corpus_digests["development"], "heldOutCorpusDigest": corpus_digests["heldOut"], "negativeControlCorpusDigest": corpus_digests["negativeControl"], "evaluatorDigest": evaluator.get("digest")}
    for key, expected in expected_pins.items():
        if evaluation.get(key) != expected:
            raise ValueError("candidate evaluation pins drift from the exact tournament")

    novelty = _derive_novelty(_object(authorization.get("novelty"), "authorization novelty"), _object(candidate.get("change"), "candidate change"), str(plan.get("operator")), str(plan.get("hardware")), str(candidate.get("createdAt")))
    if plan.get("novelty") != novelty:
        raise ValueError("plan novelty is not derivable from signed history")
    rounds = plan.get("rounds")
    if not isinstance(rounds, list) or not 3 <= len(rounds) <= 5:
        raise ValueError("plan requires 3 to 5 rounds")
    allocation_ids = [_safe_id(_object(item, "round").get("allocationId"), "allocation id") for item in rounds]
    if len(set(allocation_ids)) != len(allocation_ids):
        raise ValueError("allocation ids must be unique")
    budget = _object(plan.get("budget"), "budget")
    _exact(budget, ["maxRuns", "maxUsd", "maxWallClockMinutes", "minimumRuns"], "budget")
    minimum_runs = len(normalized_cases) * 2 * len(rounds)
    if budget.get("minimumRuns") != minimum_runs or budget.get("maxUsd") != 0 or not minimum_runs <= _positive_integer(budget.get("maxRuns"), "maxRuns") <= 640 or not 1 <= _positive_integer(budget.get("maxWallClockMinutes"), "maxWallClockMinutes") <= 120 or len(normalized_cases) * len(rounds) * 2 * (samples + warmups) > 50_000:
        raise ValueError("plan budget is invalid or exceeds hard ceilings")
    arms = _object(plan.get("arms"), "arms")
    _exact(arms, ["challenger", "champion"], "arms")
    for arm_id in ("champion", "challenger"):
        arm = _object(arms.get(arm_id), f"{arm_id} arm")
        _exact(arm, ["config", "id"], f"{arm_id} arm")
        if arm.get("id") != arm_id:
            raise ValueError("arm identity is invalid")
        _integer_object(arm.get("config"), f"{arm_id} config")


def _derive_novelty(novelty: dict[str, object], change: dict[str, object], operator: str, hardware: str, created_at: str) -> dict[str, object]:
    _exact(novelty, ["classification", "generatorKnowledgeCutoff", "historySnapshot"], "authorization novelty")
    classification = _text(novelty.get("classification"), "novelty classification")
    if classification not in {"calibration", "prospective"}:
        raise ValueError("novelty classification is invalid")
    cutoff = _text(novelty.get("generatorKnowledgeCutoff"), "knowledge cutoff")
    cutoff_time = _timestamp(cutoff, "knowledge cutoff timestamp")
    created_time = _timestamp(created_at, "candidate creation timestamp")
    if cutoff_time >= created_time:
        raise ValueError("knowledge cutoff must strictly precede candidate creation")
    history = _object(novelty.get("historySnapshot"), "history snapshot")
    _exact(history, ["completeThrough", "configs", "contract", "digest", "schemaVersion"], "history snapshot")
    configs = history.get("configs")
    if history.get("schemaVersion") != 1 or history.get("contract") != "noeris-seen-config-history-v1" or history.get("completeThrough") != cutoff or not isinstance(configs, list) or len(configs) > 10_000:
        raise ValueError("history snapshot is invalid or incomplete")
    normalized = []
    for index, value in enumerate(configs):
        item = _object(value, f"seen config {index}")
        _exact(item, ["config", "hardware", "operator"], f"seen config {index}")
        normalized.append({"operator": _safe_id(item.get("operator"), "seen operator"), "hardware": _safe_id(item.get("hardware"), "seen hardware"), "config": _integer_object(item.get("config"), "seen config")})
    if normalized != sorted(normalized, key=canonical_json) or len({canonical_json(item) for item in normalized}) != len(normalized):
        raise ValueError("history snapshot configs must be unique and canonical")
    history_basis = {"completeThrough": cutoff, "configs": normalized, "contract": "noeris-seen-config-history-v1", "schemaVersion": 1}
    history_digest = domain("noeris-seen-config-history-v1", history_basis)
    if history.get("digest") != history_digest:
        raise ValueError("history snapshot digest is not recomputable")
    challenger = _integer_object(change.get("knobs"), "challenger knobs")
    previously_seen = any(item["operator"] == operator and item["hardware"] == hardware and item["config"] == challenger for item in normalized)
    if classification == "prospective" and previously_seen:
        raise ValueError("prospective candidate was already present in history")
    return {"classification": classification, "generatorKnowledgeCutoff": cutoff, "historyCheckpointDigest": history_digest, "seenConfigDigest": domain("noeris-seen-configs-v1", normalized), "seenConfigCount": len(normalized), "challengerPreviouslySeen": previously_seen}


def _case_seed(round_seed: int, tensor_seed: int, case_id: str) -> int:
    return int(hashlib.sha256(f"noeris-case-seed-v1\0{round_seed}\0{tensor_seed}\0{case_id}".encode()).hexdigest()[:12], 16)


def _measurement(case_id: str, lane: str, arm_id: str, value: ArmMeasurement, samples: int, warmups: int, absolute_tolerance: float, relative_tolerance: float) -> dict[str, object]:
    if not isinstance(value, ArmMeasurement) or len(value.output_digests) < 2 or len(set(value.output_digests)) != 1:
        raise ValueError("kernel measurement failed same-input determinism")
    _digest(value.reference_digest, "reference digest")
    for item in value.output_digests:
        _digest(item, "output digest")
    if len(value.timings_ms) != samples or any(not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(item) or item <= 0 for item in value.timings_ms):
        raise ValueError("kernel measurement timings are invalid or incomplete")
    max_absolute_error = _bounded_float(value.max_absolute_error, "max absolute error", 0, absolute_tolerance)
    max_relative_error = _bounded_float(value.max_relative_error, "max relative error", 0, relative_tolerance)
    if value.warmups_completed != warmups:
        raise ValueError("kernel measurement did not complete exact warmups")
    return {"caseId": case_id, "lane": lane, "armId": arm_id, "correct": True, "deterministic": True, "referenceDigest": value.reference_digest, "outputDigest": value.output_digests[0], "repeatOutputDigests": list(value.output_digests), "maxAbsoluteError": max_absolute_error, "maxRelativeError": max_relative_error, "warmupsCompleted": warmups, "timingsMs": list(value.timings_ms)}


def _environment(value: object, evaluator: dict[str, object], hardware: str) -> dict[str, object]:
    raw = _object(value, "environment")
    keys = ["cudaVersion", "deviceUuid", "driverVersion", "gpuName", "imageDigest", "pythonVersion", "torchVersion", "tritonVersion"]
    _exact(raw, keys, "environment")
    for key in keys:
        _text(raw.get(key), f"environment.{key}")
    if raw.get("imageDigest") != evaluator.get("softwareImageDigest"):
        raise ValueError("runtime image drifts from the evaluator pin")
    expected_gpu_token = {"t4": "t4", "a100": "a100", "h100": "h100"}.get(hardware)
    if expected_gpu_token is None or expected_gpu_token not in str(raw.get("gpuName")).lower():
        raise ValueError("runtime GPU does not match the planned hardware")
    return raw


def _worker_identity(value: object, plan: dict[str, object]) -> dict[str, object]:
    raw = _object(value, "worker identity")
    _exact(raw, ["evaluatorDigest", "repositoryCommitSha", "repositoryTreeDigest"], "worker identity")
    repository = _object(plan.get("repository"), "plan repository")
    evaluator = _object(plan.get("evaluator"), "plan evaluator")
    if raw.get("repositoryCommitSha") != repository.get("commitSha") or raw.get("repositoryTreeDigest") != repository.get("treeDigest") or raw.get("evaluatorDigest") != evaluator.get("digest"):
        raise ValueError("measured worker code identity drifts from the plan")
    return raw


def _worker_usage(value: object) -> dict[str, object]:
    raw = _object(value, "worker usage")
    _exact(raw, ["costUsd", "provider", "tier", "usageReceiptDigest"], "worker usage")
    if raw.get("provider") != "kaggle" or raw.get("tier") != "free" or raw.get("costUsd") != 0:
        raise ValueError("worker usage must prove Kaggle free-tier zero-dollar execution")
    _digest(raw.get("usageReceiptDigest"), "worker usage receipt digest")
    return raw


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _exact(value: dict[str, object], keys: list[str], label: str) -> None:
    if sorted(value) != sorted(keys):
        raise ValueError(f"{label} has unsupported or missing fields")


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _safe_id(value: object, label: str) -> str:
    parsed = _text(value, label)
    if not SAFE_ID.fullmatch(parsed) or parsed != parsed.lower():
        raise ValueError(f"{label} must be a lowercase safe id")
    return parsed


def _digest(value: object, label: str) -> str:
    parsed = _text(value, label)
    if not DIGEST.fullmatch(parsed):
        raise ValueError(f"{label} must be a lowercase sha256 digest")
    return parsed


def _timestamp(value: object, label: str) -> datetime:
    parsed = _text(value, label)
    if not parsed.endswith("Z"):
        raise ValueError(f"{label} must be a canonical UTC timestamp")
    try:
        moment = datetime.fromisoformat(f"{parsed[:-1]}+00:00")
    except ValueError as error:
        raise ValueError(f"{label} must be a canonical UTC timestamp") from error
    canonical_seconds = moment.isoformat(timespec="seconds").replace("+00:00", "Z")
    canonical_milliseconds = moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    if moment.tzinfo != timezone.utc or parsed not in {canonical_seconds, canonical_milliseconds}:
        raise ValueError(f"{label} must be a canonical UTC timestamp")
    return moment


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value >= 2**53:
        raise ValueError(f"{label} must be a positive safe integer")
    return value


def _bounded_integer(value: object, label: str, minimum: int, maximum: int) -> int:
    parsed = _positive_integer(value, label)
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{label} is outside bounded limits")
    return parsed


def _bounded_float(value: object, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not minimum <= float(value) <= maximum:
        raise ValueError(f"{label} is outside bounded limits")
    return float(value)


def _integer_object(value: object, label: str) -> dict[str, int]:
    raw = _object(value, label)
    if not raw or any(not isinstance(item, int) or isinstance(item, bool) or item < 1 for item in raw.values()):
        raise ValueError(f"{label} must contain positive integer values")
    return {key: int(raw[key]) for key in sorted(raw)}
