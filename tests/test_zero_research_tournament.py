from __future__ import annotations

import hashlib
import json

import pytest

from research_engine.zero_research_tournament import (
    AUTHORITY,
    ArmMeasurement,
    canonical_json,
    domain,
    _build_untrusted_tournament_proposal,
    sha256,
)


D = lambda value: f"sha256:{value * 64}"  # noqa: E731
SIGNATURE = "-----BEGIN SSH SIGNATURE-----\nTEST\n-----END SSH SIGNATURE-----\n"


def fixture():
    cases = [
        {"id": "dev-1", "lane": "development", "shape": {"K": 4096, "M": 64, "N": 64}, "tensorSeed": 11},
        {"id": "held-1", "lane": "held_out", "shape": {"K": 8192, "M": 64, "N": 64}, "tensorSeed": 12},
        {"id": "held-2", "lane": "held_out", "shape": {"K": 4096, "M": 128, "N": 64}, "tensorSeed": 13},
        {"id": "negative-1", "lane": "negative_control", "shape": {"K": 64, "M": 64, "N": 64}, "tensorSeed": 14},
    ]
    champion = {"BLOCK_SIZE_K": 32, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "GROUP_SIZE_M": 8, "num_stages": 2, "num_warps": 4}
    challenger = {**champion, "num_stages": 3}
    corpus_digests = {
        "development": domain("noeris-kernel-corpus-v1", {"lane": "development", "cases": [item for item in cases if item["lane"] == "development"]}),
        "heldOut": domain("noeris-kernel-corpus-v1", {"lane": "held_out", "cases": [item for item in cases if item["lane"] == "held_out"]}),
        "negativeControl": domain("noeris-kernel-corpus-v1", {"lane": "negative_control", "cases": [item for item in cases if item["lane"] == "negative_control"]}),
    }
    manifest_id = "noeris-kernel-heldout-v1"
    manifest_digest = domain("noeris-kernel-tournament-manifest-v1", {"id": manifest_id, "corpusDigests": corpus_digests, "cases": cases})
    evaluator_config = {"absoluteTolerance": 0.001, "relativeTolerance": 0.001, "samples": 5, "softwareImageDigest": D("9"), "warmups": 5}
    candidate = {
        "schemaVersion": 1, "id": f"imp_noeris_{'c' * 64}", "project": "noeris", "createdAt": "2026-07-18T00:00:00Z",
        "hypothesis": {"statement": "stage 3 helps", "rationale": "sealed", "sourceRefs": ["artifact:world-model:sha256:abc"]},
        "change": {"kind": "kernel_config", "knobs": {**challenger}},
        "budget": {"maxRuns": 24, "maxUsd": 0, "maxWallClockMinutes": 30},
        "evaluation": {"manifestDigest": manifest_digest, "developmentCorpusDigest": corpus_digests["development"], "heldOutCorpusDigest": corpus_digests["heldOut"], "negativeControlCorpusDigest": corpus_digests["negativeControl"], "evaluatorDigest": D("2")},
        "authority": {"mode": "draft_pr_only", "evaluatorChangesAllowed": False, "autoMergeAllowed": False, "externalPublicationAllowed": False},
    }
    candidate_digest = sha256(canonical_json(candidate))
    evaluator = {"digest": D("2"), "configDigest": domain("noeris-kernel-evaluator-config-v1", evaluator_config), **evaluator_config}
    repository = {"url": "0sec-labs/noeris", "commitSha": "a" * 40, "treeDigest": D("7")}
    manifest_auth = {"id": manifest_id, "digest": manifest_digest, "cases": cases}
    nonce = "f" * 64
    seen_configs = [{"operator": "matmul", "hardware": "t4", "config": champion}]
    history_basis = {"completeThrough": "2026-07-17T23:59:59.000Z", "configs": seen_configs, "contract": "noeris-seen-config-history-v1", "schemaVersion": 1}
    novelty_authorization = {"classification": "prospective", "generatorKnowledgeCutoff": history_basis["completeThrough"], "historySnapshot": {**history_basis, "digest": domain("noeris-seen-config-history-v1", history_basis)}}
    authorization = {
        "schemaVersion": 1, "candidateDigest": candidate_digest, "operator": "matmul", "hardware": "t4",
        "controllerPrincipal": "0research-controller", "signatureSsh": SIGNATURE,
        "generator": {"id": "noeris.world-model-v1", "digest": D("6")}, "randomizationNonce": nonce,
        "repository": repository, "championConfig": champion, "evaluator": evaluator, "manifest": manifest_auth,
        "novelty": novelty_authorization,
        "rounds": [{"allocationId": "kaggle-t4-001"}, {"allocationId": "kaggle-t4-002"}, {"allocationId": "kaggle-t4-003"}],
    }
    authorization_digest = sha256(canonical_json(authorization))
    plan_rounds = []
    for index, round_value in enumerate(authorization["rounds"]):
        allocation = round_value["allocationId"]
        seed = int(hashlib.sha256(f"{nonce}\0{allocation}\0{index}".encode()).hexdigest()[:12], 16)
        orders = []
        for case in cases:
            parity = hashlib.sha256(f"{nonce}\0{allocation}\0{case['id']}".encode()).digest()[0] & 1
            orders.append({"caseId": case["id"], "order": ["champion", "challenger"] if parity == 0 else ["challenger", "champion"]})
        plan_rounds.append({"allocationId": allocation, "seed": seed, "armOrders": orders})
    plan_body = {
        "schemaVersion": 1, "contract": "noeris-kernel-tournament-plan-v1", "candidateId": candidate["id"], "candidateDigest": candidate_digest,
        "controllerAuthorization": {"ref": f"0research-noeris-tournament-controller-envelope-v1:{authorization_digest}", "sha256": authorization_digest, "principal": "0research-controller", "namespace": "0research-noeris-tournament-plan-v1"},
        "generator": authorization["generator"], "repository": repository, "operator": "matmul", "hardware": "t4",
        "arms": {"champion": {"id": "champion", "config": {**champion}}, "challenger": {"id": "challenger", "config": {**challenger}}},
        "evaluator": evaluator,
        "manifest": {"id": manifest_auth["id"], "digest": manifest_auth["digest"], "corpusDigests": corpus_digests, "cases": cases},
        "novelty": {"classification": "prospective", "generatorKnowledgeCutoff": history_basis["completeThrough"], "historyCheckpointDigest": domain("noeris-seen-config-history-v1", history_basis), "seenConfigDigest": domain("noeris-seen-configs-v1", seen_configs), "seenConfigCount": 1, "challengerPreviouslySeen": False},
        "randomizationDigest": sha256(nonce), "rounds": plan_rounds,
        "budget": {"maxRuns": 24, "maxUsd": 0, "maxWallClockMinutes": 30, "minimumRuns": 24}, "authority": AUTHORITY,
    }
    plan_digest = sha256(canonical_json(plan_body))
    plan = {**plan_body, "planId": f"noeris-{plan_digest[7:]}", "planDigest": plan_digest}
    environment = {"cudaVersion": "12.4", "deviceUuid": "GPU-test", "driverVersion": "550", "gpuName": "Tesla T4", "imageDigest": D("9"), "pythonVersion": "3.11", "torchVersion": "2.5", "tritonVersion": "3.1"}
    worker_identity = {"repositoryCommitSha": repository["commitSha"], "repositoryTreeDigest": repository["treeDigest"], "evaluatorDigest": evaluator["digest"]}
    worker_usage = {"provider": "kaggle", "tier": "free", "costUsd": 0, "usageReceiptDigest": D("f")}
    return candidate, authorization, plan, environment, worker_identity, worker_usage


def runner(_arm, _config, _shape, _seed, warmups, samples, _absolute, _relative):
    return ArmMeasurement(D("1"), (D("2"), D("2")), tuple(1.0 + index / 100 for index in range(samples)), 0.0001, 0.0001, warmups)


def attestor(_material, allocation_id, device_uuid):
    return {"allocationId": allocation_id, "deviceUuid": device_uuid, "evidenceDigest": "PENDING", "namespace": "0research-noeris-allocation-evidence-v1", "principal": "kaggle-t4-runner", "signatureSsh": SIGNATURE}


def execute(overrides=None, verify=None, run=runner, attest=None):
    candidate, authorization, plan, environment, worker_identity, worker_usage = fixture()
    values = {"candidate_value": candidate, "authorization_value": authorization, "plan_value": plan, "environment_value": environment, "worker_identity_value": worker_identity, "worker_usage_value": worker_usage}
    if overrides:
        overrides(values)
    calls = []
    verifier = verify or (lambda material, signature, principal, namespace, policy: calls.append((principal, namespace, policy, len(material))))
    def bound_attestor(material, allocation_id, device_uuid):
        value = dict((attest or attestor)(material, allocation_id, device_uuid)); value["evidenceDigest"] = json.loads(material)["evidenceDigest"]
        return value
    result = _build_untrusted_tournament_proposal(allocation_id="kaggle-t4-001", controller_policy="/policy/controller", worker_policy="/policy/worker", runner=run, verifier=verifier, attestor=bound_attestor, **values)
    return result, calls


def rehash_plan(plan):
    body = {key: value for key, value in plan.items() if key not in {"planId", "planDigest"}}
    digest = sha256(canonical_json(body)); plan["planDigest"] = digest; plan["planId"] = f"noeris-{digest[7:]}"


def test_executes_exact_controller_order_and_retains_raw_evidence():
    result, calls = execute()
    assert result["contract"] == "noeris-kernel-tournament-proposal-v1"
    assert result["acceptedBy0brain"] is False
    assert result["usage"]["measuredArmCaseRuns"] == 8
    assert result["usage"]["timingSamples"] == 40
    assert len(result["results"]) == 8
    assert calls[0][:3] == ("0research-controller", "0research-noeris-tournament-plan-v1", "/policy/controller")
    assert calls[1][:3] == ("kaggle-t4-runner", "0research-noeris-allocation-evidence-v1", "/policy/worker")
    assert result["allocationAttestation"]["principal"] == "kaggle-t4-runner"
    assert result["authority"] == AUTHORITY


def test_rejects_candidate_and_authorization_substitution():
    with pytest.raises(ValueError, match="exact candidate"):
        execute(lambda values: values["candidate_value"]["change"]["knobs"].update({"num_stages": 4}))
    with pytest.raises(ValueError, match="authorization bytes"):
        execute(lambda values: values["authorization_value"].update({"hardware": "h100"}))


def test_rejects_rehashed_seed_or_arm_order_tampering():
    def tamper(values):
        values["plan_value"]["rounds"][0]["seed"] += 1
        rehash_plan(values["plan_value"])
    with pytest.raises(ValueError, match="controller-derived"):
        execute(tamper)


def test_rejects_incorrect_nondeterministic_or_incomplete_measurements():
    with pytest.raises(ValueError, match="absolute error"):
        execute(run=lambda *_: ArmMeasurement(D("1"), (D("2"), D("2")), (1, 1, 1, 1, 1), 1, 0, 5))
    with pytest.raises(ValueError, match="determinism"):
        execute(run=lambda *_: ArmMeasurement(D("1"), (D("2"), D("3")), (1, 1, 1, 1, 1), 0, 0, 5))
    with pytest.raises(ValueError, match="timings"):
        execute(run=lambda *_: ArmMeasurement(D("1"), (D("2"), D("2")), (1,), 0, 0, 5))


def test_rejects_runtime_image_drift_and_bad_signature():
    with pytest.raises(ValueError, match="runtime image"):
        execute(lambda values: values["environment_value"].update({"imageDigest": D("8")}))
    with pytest.raises(ValueError, match="signature rejected"):
        execute(verify=lambda *_: (_ for _ in ()).throw(ValueError("signature rejected")))


def test_rejects_rehashed_manifest_and_novelty_tampering():
    def tamper_manifest(values):
        values["plan_value"]["manifest"]["cases"][0]["shape"]["K"] = 2048
        rehash_plan(values["plan_value"])
    with pytest.raises(ValueError, match="signed controller authorization"):
        execute(tamper_manifest)

    def tamper_novelty(values):
        values["plan_value"]["novelty"]["seenConfigCount"] = 2
        rehash_plan(values["plan_value"])
    with pytest.raises(ValueError, match="signed history"):
        execute(tamper_novelty)


def test_rejects_reference_drift_between_arms():
    def mismatched_reference(arm, *_):
        reference = D("1") if arm == "champion" else D("2")
        return ArmMeasurement(reference, (D("3"), D("3")), (1, 1, 1, 1, 1), 0, 0, 5)
    with pytest.raises(ValueError, match="same exact reference"):
        execute(run=mismatched_reference)


def test_rejects_rehashed_budget_and_duplicate_case_tampering():
    def tamper_budget(values):
        values["plan_value"]["budget"]["minimumRuns"] = 23
        rehash_plan(values["plan_value"])
    with pytest.raises(ValueError, match="plan budget"):
        execute(tamper_budget)

    def duplicate_tensor_seed(values):
        cases = values["authorization_value"]["manifest"]["cases"]
        cases[1]["tensorSeed"] = cases[0]["tensorSeed"]
        values["plan_value"]["manifest"]["cases"] = cases
        values["plan_value"]["controllerAuthorization"]["sha256"] = sha256(canonical_json(values["authorization_value"]))
        values["plan_value"]["controllerAuthorization"]["ref"] = f"0research-noeris-tournament-controller-envelope-v1:{values['plan_value']['controllerAuthorization']['sha256']}"
        rehash_plan(values["plan_value"])
    with pytest.raises(ValueError, match="unique and canonical"):
        execute(duplicate_tensor_seed)


def test_rejects_hardware_worker_identity_and_paid_usage_drift():
    with pytest.raises(ValueError, match="planned hardware"):
        execute(lambda values: values["environment_value"].update({"gpuName": "NVIDIA A100"}))
    with pytest.raises(ValueError, match="worker code identity"):
        execute(lambda values: values["worker_identity_value"].update({"repositoryCommitSha": "b" * 40}))
    with pytest.raises(ValueError, match="zero-dollar execution"):
        execute(lambda values: values["worker_usage_value"].update({"tier": "paid", "costUsd": 1}))


def test_rejects_bad_allocation_attestation_and_authorization_shape():
    def wrong_attestor(_material, _allocation_id, device_uuid):
        return {"allocationId": "kaggle-t4-002", "deviceUuid": device_uuid, "evidenceDigest": "PENDING", "namespace": "0research-noeris-allocation-evidence-v1", "principal": "kaggle-t4-runner", "signatureSsh": SIGNATURE}
    with pytest.raises(ValueError, match="exact evidence and device"):
        execute(attest=wrong_attestor)

    with pytest.raises(ValueError, match="unsupported or missing fields"):
        execute(lambda values: values["authorization_value"].update({"unsealed": True}))

    def reject_worker(_material, _signature, _principal, namespace, _policy):
        if namespace == "0research-noeris-allocation-evidence-v1":
            raise ValueError("worker signature rejected")
    with pytest.raises(ValueError, match="worker signature rejected"):
        execute(verify=reject_worker)


def test_rejects_evaluator_tolerance_and_timestamp_tampering():
    def tamper_evaluator(values):
        values["authorization_value"]["evaluator"]["absoluteTolerance"] = 0.1
        values["plan_value"]["evaluator"]["absoluteTolerance"] = 0.1
        values["plan_value"]["controllerAuthorization"]["sha256"] = sha256(canonical_json(values["authorization_value"]))
        values["plan_value"]["controllerAuthorization"]["ref"] = f"0research-noeris-tournament-controller-envelope-v1:{values['plan_value']['controllerAuthorization']['sha256']}"
        rehash_plan(values["plan_value"])
    with pytest.raises(ValueError, match="absolute tolerance"):
        execute(tamper_evaluator)

    with pytest.raises(ValueError, match="timestamp"):
        execute(lambda values: values["candidate_value"].update({"createdAt": "not-a-timestamp"}))
