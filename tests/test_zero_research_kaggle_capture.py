from __future__ import annotations

import inspect
import json
import os
import struct
from datetime import datetime, timezone

import pytest

import research_engine.zero_research_kaggle_capture as capture_module
from research_engine.zero_research_kaggle_capture import CAPTURE_AUTHORITY, _run_capture_core
from research_engine.zero_research_tournament import canonical_json, sha256
from tests.test_zero_research_kaggle_worker import Backend
from tests.test_zero_research_tournament import fixture, rehash_plan


def environment():
    release = {
        "imageBuildDate": "20260319-213538",
        "imageGitCommit": "c292018b280631cbfe6f4f16fc6a84f2786b5f86",
        "cudaVersion": "12.4",
        "pythonVersion": "3.11.14",
        "torchVersion": "2.5.1",
        "tritonVersion": "3.1.0",
    }
    return {
        **release,
        "runtimeFingerprintDigest": sha256(f"noeris-kaggle-runtime-v1\0{canonical_json(release)}"),
        "deviceUuid": "GPU-capture-001",
        "driverVersion": "550.54",
        "gpuName": "Tesla T4",
    }


def run(tmp_path, *, backend=None, mutate=None, clock=None):
    candidate, authorization, plan, _old_environment, worker_identity, _usage = fixture()
    if mutate:
        mutate(candidate, authorization, plan, worker_identity)
    moments = iter(clock or [datetime(2026, 7, 18, 0, 0, tzinfo=timezone.utc), datetime(2026, 7, 18, 0, 1, tzinfo=timezone.utc)])
    output = tmp_path / "capture"
    result = _run_capture_core(
        candidate=candidate, authorization=authorization, plan_value=plan,
        allocation_id="kaggle-t4-001", output_directory=str(output),
        kernel_ref="zero-research/capture-001", environment=environment(),
        worker_identity=worker_identity, backend=backend or Backend(),
        now=lambda: next(moments), publisher=os.rename,
    )
    return result, output


def test_emits_unsigned_bounded_capture_with_observable_runtime_and_self_report(tmp_path):
    os.chmod(tmp_path, 0o700)
    backend = Backend()
    result, output = run(tmp_path, backend=backend)
    capture = result["capture"]
    assert capture["contract"] == "noeris-kaggle-allocation-capture-v1"
    assert capture["acceptedBy0brain"] is False
    assert capture["authority"] == CAPTURE_AUTHORITY
    assert "signatureSsh" not in canonical_json(capture)
    assert "imageDigest" not in capture["environment"]
    assert capture["environment"]["imageBuildDate"] == "20260319-213538"
    assert capture["environment"]["imageGitCommit"].startswith("c292018")
    assert len(backend.calls) == 8
    assert len(capture["results"]) == 8
    assert capture["captureDigest"] == sha256(canonical_json({key: value for key, value in capture.items() if key != "captureDigest"}))
    assert capture["usageSelfReport"]["path"] == "kaggle-t4-001/usage-self-report.json"
    usage = json.loads((output / capture["usageSelfReport"]["path"]).read_text())
    assert usage["independentlyObserved"] is False
    assert usage["tierClaim"] == "free"
    for item in capture["results"]:
        assert item["reference"]["path"] == f"kaggle-t4-001/raw/{item['caseId']}/reference.f64le"
        assert [value["path"] for value in item["outputs"]] == [
            f"kaggle-t4-001/raw/{item['caseId']}/{item['armId']}-output-1.f64le",
            f"kaggle-t4-001/raw/{item['caseId']}/{item['armId']}-output-2.f64le",
        ]


def test_capture_has_no_signer_policy_or_secret_surface():
    signature = inspect.signature(capture_module.run_fixed_kaggle_capture)
    assert set(signature.parameters) == {"candidate_path", "authorization_path", "plan_path", "allocation_id", "output_directory", "kernel_ref"}
    source = inspect.getsource(capture_module)
    assert "WORKER_KEY" not in source
    assert "allowed_signers" not in source
    assert "ssh-keygen" not in source


def test_rejects_nondeterminism_paid_plan_and_runtime_fingerprint_drift(tmp_path):
    os.chmod(tmp_path, 0o700)
    with pytest.raises(ValueError, match="nondeterministic"):
        run(tmp_path, backend=Backend(nondeterministic=True))
    assert not (tmp_path / "capture").exists()
    def paid(_candidate, _authorization, plan, _identity):
        plan["budget"].update({"maxUsd": 1}); rehash_plan(plan)
    with pytest.raises(ValueError, match="zero-dollar"):
        run(tmp_path, mutate=paid)
    drift = environment(); drift["torchVersion"] = "changed"
    candidate, authorization, plan, _old_environment, worker_identity, _usage = fixture()
    with pytest.raises(ValueError, match="fingerprint"):
        _run_capture_core(candidate=candidate, authorization=authorization, plan_value=plan, allocation_id="kaggle-t4-001", output_directory=str(tmp_path / "capture"), kernel_ref="zero-research/capture-001", environment=drift, worker_identity=worker_identity, backend=Backend(), publisher=os.rename)


def test_unsigned_capture_enforces_tournament_workload_ceilings_before_gpu_work(tmp_path):
    os.chmod(tmp_path, 0o700)

    def rebind(candidate, authorization, plan):
        candidate_digest = sha256(canonical_json(candidate))
        authorization["candidateDigest"] = candidate_digest
        authorization_digest = sha256(canonical_json(authorization))
        plan["candidateDigest"] = candidate_digest
        plan["controllerAuthorization"].update({
            "sha256": authorization_digest,
            "ref": f"0research-noeris-tournament-controller-envelope-v1:{authorization_digest}",
        })
        rehash_plan(plan)

    def too_many_cases(candidate, authorization, plan, _identity):
        plan["manifest"]["cases"] *= 17
        rebind(candidate, authorization, plan)

    backend = Backend()
    with pytest.raises(ValueError, match="4 to 64 cases"):
        run(tmp_path, backend=backend, mutate=too_many_cases)
    assert backend.calls == []

    def too_many_samples(_candidate, authorization, plan, _identity):
        plan["evaluator"]["samples"] = 101
        authorization["evaluator"]["samples"] = 101
        rebind(_candidate, authorization, plan)

    with pytest.raises(ValueError, match="samples is outside bounded limits"):
        run(tmp_path, backend=backend, mutate=too_many_samples)
    assert backend.calls == []

    def too_much_wall_time(candidate, authorization, plan, _identity):
        candidate["budget"]["maxWallClockMinutes"] = 121
        plan["budget"]["maxWallClockMinutes"] = 121
        rebind(candidate, authorization, plan)

    with pytest.raises(ValueError, match="hard ceilings"):
        run(tmp_path, backend=backend, mutate=too_much_wall_time)
    assert backend.calls == []


def test_restart_recovers_without_backend_and_rejects_raw_tampering(tmp_path):
    os.chmod(tmp_path, 0o700)
    first_backend = Backend()
    first, output = run(tmp_path, backend=first_backend)
    retry_backend = Backend()
    recovered, _output = run(tmp_path, backend=retry_backend)
    assert recovered["recovered"] is True
    assert recovered["capture"] == first["capture"]
    assert retry_backend.calls == []
    raw = next((output / "kaggle-t4-001" / "raw").rglob("*-output-1.f64le"))
    raw.write_bytes(struct.pack("<2d", 7.0, 8.0))
    with pytest.raises(ValueError, match="bytes or digest drift"):
        run(tmp_path, backend=Backend())


def test_restart_rejects_rehashed_plan_order_and_path_substitution(tmp_path):
    os.chmod(tmp_path, 0o700)
    _first, output = run(tmp_path)
    capture_path = output / "kaggle-t4-001" / "capture.json"
    capture = json.loads(capture_path.read_text())
    capture["results"][0]["armOrderIndex"] = 1 - capture["results"][0]["armOrderIndex"]
    body = {key: value for key, value in capture.items() if key != "captureDigest"}
    capture["captureDigest"] = sha256(canonical_json(body))
    capture_path.write_text(f"{canonical_json(capture)}\n")
    with pytest.raises(ValueError, match="drifts from the exact plan"):
        run(tmp_path, backend=Backend())


def test_restart_rejects_unprotected_nested_capture_directory(tmp_path):
    os.chmod(tmp_path, 0o700)
    _first, output = run(tmp_path)
    os.chmod(output / "kaggle-t4-001" / "raw", 0o755)
    with pytest.raises(ValueError, match="unprotected directory"):
        run(tmp_path, backend=Backend())


def test_fixed_wrapper_recovers_before_git_cuda_or_runtime_initialization(tmp_path, monkeypatch):
    os.chmod(tmp_path, 0o700)
    output = tmp_path / "capture"; output.mkdir(mode=0o700)
    candidate, authorization, plan, _environment, _identity, _usage = fixture()
    documents = {"candidate": candidate, "authorization": authorization, "plan": plan}
    monkeypatch.setattr(capture_module, "_stable_json", lambda path, _label: documents[path])
    monkeypatch.setattr(capture_module, "_recover_capture", lambda *_args: {"allocationId": "kaggle-t4-001", "recovered": True})
    monkeypatch.setattr(capture_module, "_capture_environment", lambda: pytest.fail("runtime initialized during recovery"))
    monkeypatch.setattr(capture_module, "TorchTritonBackend", lambda: pytest.fail("CUDA initialized during recovery"))
    monkeypatch.setattr(capture_module.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("git initialized during recovery"))
    result = capture_module.run_fixed_kaggle_capture("candidate", "authorization", "plan", "kaggle-t4-001", str(output), "zero-research/capture-001")
    assert result["recovered"] is True
