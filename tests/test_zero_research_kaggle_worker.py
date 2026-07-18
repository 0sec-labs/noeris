from __future__ import annotations

import json
import os
import stat
import struct
from datetime import datetime, timezone

import pytest

import research_engine.zero_research_kaggle_worker as worker_module
from research_engine.zero_research_kaggle_worker import RawMeasurement, _run_worker_core
from tests.test_zero_research_tournament import SIGNATURE, fixture


class Backend:
    def __init__(self, *, nondeterministic: bool = False, bad_timings: bool = False):
        self.calls = []
        self.nondeterministic = nondeterministic
        self.bad_timings = bad_timings

    def measure(self, config, shape, seed, warmups, samples):
        self.calls.append((dict(config), dict(shape), seed, warmups, samples))
        reference = struct.pack("<2d", 1.0, 2.0)
        second = struct.pack("<2d", 1.0, 3.0) if self.nondeterministic else reference
        timings = tuple(1_000_000 + index * 1_000 for index in range(samples - int(self.bad_timings)))
        return RawMeasurement(reference, (reference, second), timings, 0.0, 0.0, warmups)


def run(tmp_path, *, backend=None, mutate=None, signer=None, verifier=None, publisher=os.rename, clock=None):
    candidate, authorization, plan, environment, worker_identity, _usage = fixture()
    if mutate:
        mutate(candidate, authorization, plan, environment, worker_identity)
    output = tmp_path / "allocation"
    calls = []
    verify = verifier or (lambda material, signature, principal, namespace, policy: calls.append((len(material), principal, namespace, policy)))
    moments = iter(clock or [datetime(2026, 7, 18, 0, 0, 0, tzinfo=timezone.utc), datetime(2026, 7, 18, 0, 1, 0, tzinfo=timezone.utc)])
    result = _run_worker_core(
        candidate=candidate, authorization=authorization, plan_value=plan,
        allocation_id="kaggle-t4-001", output_directory=str(output),
        kernel_ref="zero-research/run-001", environment=environment,
        worker_identity=worker_identity, backend=backend or Backend(),
        signer=signer or (lambda _material, _namespace: SIGNATURE), verifier=verify,
        controller_policy="/policy/controller", worker_policy="/policy/worker",
        now=lambda: next(moments), publisher=publisher,
    )
    return result, calls, output


def test_emits_exact_proposal_raw_receipt_usage_and_distinct_outputs(tmp_path):
    os.chmod(tmp_path, 0o700)
    backend = Backend()
    result, calls, output = run(tmp_path, backend=backend)
    assert result["proposal"]["contract"] == "noeris-kernel-tournament-proposal-v1"
    assert result["proposal"]["acceptedBy0brain"] is False
    assert result["artifactReceipt"]["contract"] == "noeris-kernel-allocation-artifacts-v1"
    assert "oracleReceipt" not in result and "controllerObservation" not in result
    assert len(result["artifactReceipt"]["results"]) == 8
    assert len(backend.calls) == 8
    assert {item[2] for item in backend.calls} == {item[2] for item in backend.calls[::2]}
    first = result["artifactReceipt"]["results"][0]
    assert first["outputs"][0]["path"] != first["outputs"][1]["path"]
    assert first["outputs"][0]["sha256"] == first["outputs"][1]["sha256"]
    assert first["timingsNs"] == [1_000_000, 1_001_000, 1_002_000, 1_003_000, 1_004_000]
    allocation = output / "kaggle-t4-001"
    usage = json.loads((allocation / "usage.json").read_text())
    assert usage == {"accelerator": "gpu", "allocationId": "kaggle-t4-001", "completedAt": "2026-07-18T00:01:00.000Z", "contract": "noeris-kaggle-usage-v1", "costUsd": 0, "kernelRef": "zero-research/run-001", "provider": "kaggle", "schemaVersion": 1, "startedAt": "2026-07-18T00:00:00.000Z", "status": "complete", "tier": "free"}
    assert result["proposal"]["usage"]["usageReceiptDigest"] == result["artifactReceipt"]["usageArtifact"]["sha256"]
    assert result["artifactReceipt"]["usageArtifact"]["path"] == "kaggle-t4-001/usage.json"
    assert all(
        artifact["path"].startswith("kaggle-t4-001/raw/")
        for item in result["artifactReceipt"]["results"]
        for artifact in [item["reference"], *item["outputs"]]
    )
    assert {item[2] for item in calls} >= {"0research-noeris-tournament-plan-v1", "0research-noeris-allocation-evidence-v1", "0research-noeris-allocation-artifacts-v1"}
    for path in output.rglob("*"):
        if path.is_file(): assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not list(tmp_path.glob(".allocation-stage-*"))


def test_failure_leaves_no_published_or_staged_allocation(tmp_path):
    os.chmod(tmp_path, 0o700)
    with pytest.raises(ValueError, match="determinism"):
        run(tmp_path, backend=Backend(nondeterministic=True))
    assert not (tmp_path / "allocation").exists()
    assert not list(tmp_path.glob(".allocation-stage-*"))


def test_rejects_incomplete_raw_timings_before_publication(tmp_path):
    os.chmod(tmp_path, 0o700)
    with pytest.raises(ValueError, match="invalid raw outputs or timings"):
        run(tmp_path, backend=Backend(bad_timings=True))
    assert not (tmp_path / "allocation").exists()


def test_rejects_resource_or_config_expansion_before_backend(tmp_path):
    os.chmod(tmp_path, 0o700)
    backend = Backend()
    def mutate(_candidate, _authorization, plan, _environment, _identity):
        plan["arms"]["challenger"]["config"]["AUTOTUNE"] = 1
    with pytest.raises(ValueError, match="exact six integer matmul knobs"):
        run(tmp_path, backend=backend, mutate=mutate)
    assert backend.calls == []
    assert not (tmp_path / "allocation").exists()


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda plan: plan["arms"]["challenger"]["config"].__setitem__("num_stages", 99), "exact six integer matmul knobs"),
        (lambda plan: plan["manifest"]["cases"][0]["shape"].update({"M": 2048, "N": 2048}), "shape, memory, or FLOP ceilings"),
    ],
)
def test_rejects_disallowed_values_and_oversized_shapes_before_backend(tmp_path, mutate, message):
    os.chmod(tmp_path, 0o700)
    backend = Backend()
    with pytest.raises(ValueError, match=message):
        run(tmp_path, backend=backend, mutate=lambda _c, _a, plan, _e, _i: mutate(plan))
    assert backend.calls == []
    assert not (tmp_path / "allocation").exists()


def test_signature_failure_and_nonpositive_interval_fail_closed(tmp_path):
    os.chmod(tmp_path, 0o700)
    def reject(_material, _signature, _principal, namespace, _policy):
        if namespace == "0research-noeris-allocation-artifacts-v1": raise ValueError("receipt signature rejected")
    with pytest.raises(ValueError, match="receipt signature rejected"):
        run(tmp_path, verifier=reject)
    assert not (tmp_path / "allocation").exists()
    same = datetime(2026, 7, 18, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="completion timestamp"):
        run(tmp_path, clock=[same, same])
    assert not (tmp_path / "allocation").exists()


def test_wall_clock_budget_overrun_is_not_published(tmp_path):
    os.chmod(tmp_path, 0o700)
    start = datetime(2026, 7, 18, tzinfo=timezone.utc)
    completed = datetime(2026, 7, 18, 0, 31, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="wall-clock budget"):
        run(tmp_path, clock=[start, completed])
    assert not (tmp_path / "allocation").exists()


def test_no_replace_publication_never_overwrites_concurrent_target(tmp_path):
    os.chmod(tmp_path, 0o700)
    target = tmp_path / "allocation"
    def race(_stage, destination):
        destination.mkdir(mode=0o700)
        (destination / "marker").write_text("concurrent")
        raise FileExistsError("appeared")
    with pytest.raises(FileExistsError, match="appeared"):
        run(tmp_path, publisher=race)
    assert (target / "marker").read_text() == "concurrent"
    assert not (target / "kaggle-t4-001" / "proposal.json").exists()
    assert not list(tmp_path.glob(".allocation-stage-*"))


def test_restart_recovers_exact_retained_allocation_without_gpu_work(tmp_path):
    os.chmod(tmp_path, 0o700)
    first_backend = Backend()
    first, _calls, output = run(tmp_path, backend=first_backend)
    retry_backend = Backend()
    recovered, calls, _output = run(tmp_path, backend=retry_backend)
    assert len(first_backend.calls) == 8
    assert retry_backend.calls == []
    assert recovered["recovered"] is True
    assert recovered["proposal"] == first["proposal"]
    assert recovered["artifactReceipt"] == first["artifactReceipt"]
    assert {item[2] for item in calls} == {
        "0research-noeris-tournament-plan-v1",
        "0research-noeris-allocation-evidence-v1",
        "0research-noeris-allocation-artifacts-v1",
    }
    assert not list(tmp_path.glob(".allocation-stage-*"))


def test_restart_rejects_tampered_retained_raw_artifact_without_gpu_work(tmp_path):
    os.chmod(tmp_path, 0o700)
    _first, _calls, output = run(tmp_path)
    raw = next((output / "kaggle-t4-001" / "raw").rglob("*-output-1.f64le"))
    raw.write_bytes(struct.pack("<2d", 7.0, 8.0))
    retry_backend = Backend()
    with pytest.raises(ValueError, match="bytes or digest drift"):
        run(tmp_path, backend=retry_backend)
    assert retry_backend.calls == []


def test_restart_rejects_signed_raw_path_alias_without_gpu_work(tmp_path):
    os.chmod(tmp_path, 0o700)
    _first, _calls, output = run(tmp_path)
    receipt_path = output / "kaggle-t4-001" / "artifact-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["results"][0]["reference"]["path"] = "kaggle-t4-001/usage.json"
    receipt_path.write_text(f"{worker_module.canonical_json(receipt)}\n")
    retry_backend = Backend()
    with pytest.raises(ValueError, match="canonical allocation path"):
        run(tmp_path, backend=retry_backend)
    assert retry_backend.calls == []


def test_restart_rejects_unsupported_retained_files_without_gpu_work(tmp_path):
    os.chmod(tmp_path, 0o700)
    _first, _calls, output = run(tmp_path)
    extra = output / "unexpected.txt"
    extra.write_text("not part of the signed allocation")
    os.chmod(extra, 0o600)
    retry_backend = Backend()
    with pytest.raises(ValueError, match="unsupported files"):
        run(tmp_path, backend=retry_backend)
    assert retry_backend.calls == []


def test_fixed_wrapper_recovers_before_cuda_or_repository_initialization(tmp_path, monkeypatch):
    os.chmod(tmp_path, 0o700)
    output = tmp_path / "allocation"
    output.mkdir(mode=0o700)
    candidate, authorization, plan, _environment, _identity, _usage = fixture()
    documents = {"candidate": candidate, "authorization": authorization, "plan": plan}
    monkeypatch.setattr(worker_module, "_verify_fixed_trust_separation", lambda: None)
    monkeypatch.setattr(worker_module, "_stable_json", lambda path, _label: documents[path])
    monkeypatch.setattr(worker_module, "_runtime_environment", lambda _plan: pytest.fail("CUDA runtime initialized during recovery"))
    monkeypatch.setattr(worker_module, "TorchTritonBackend", lambda: pytest.fail("GPU backend initialized during recovery"))
    monkeypatch.setattr(worker_module.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("repository identity initialized during recovery"))
    monkeypatch.setattr(worker_module, "_recover_retained", lambda **_kwargs: {"allocationId": "kaggle-t4-001", "recovered": True})
    result = worker_module.run_fixed_kaggle_worker("candidate", "authorization", "plan", "kaggle-t4-001", str(output), "zero-research/run-001")
    assert result == {"allocationId": "kaggle-t4-001", "recovered": True}
