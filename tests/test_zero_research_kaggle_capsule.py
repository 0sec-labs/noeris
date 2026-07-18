from __future__ import annotations

import json
import os

import pytest

import research_engine.zero_research_kaggle_capture as capture_module
from research_engine.zero_research_kaggle_capsule import CAPSULE_AUTHORITY, load_execution_capsule
from research_engine.zero_research_kaggle_capture import _run_capture_core
from research_engine.zero_research_tournament import canonical_json, domain, sha256
from tests.test_zero_research_kaggle_capture import environment
from tests.test_zero_research_kaggle_worker import Backend
from tests.test_zero_research_tournament import SIGNATURE, fixture, rehash_plan


def write_capsule(path, capsule):
    body = {key: value for key, value in capsule.items() if key not in {"capsuleDigest", "signatureSsh"}}
    capsule["capsuleDigest"] = domain("0research-noeris-kaggle-execution-capsule-v1", body)
    path.write_text(f"{canonical_json(capsule)}\n")


def package(tmp_path):
    root = tmp_path / "package"; root.mkdir(mode=0o700, parents=True)
    run_bytes = b"# fixed generic offline template\n"
    (root / "run.py").write_bytes(run_bytes)
    template_body = {
        "schemaVersion": 1,
        "contract": "noeris-kaggle-execution-template-v1",
        "entrypoint": "run.py",
        "files": [{"path": "run.py", "bytes": len(run_bytes), "sha256": sha256(run_bytes)}],
    }
    template = {**template_body, "templateDigest": domain("noeris-kaggle-execution-template-v1", template_body)}
    candidate, authorization, plan, _environment, _identity, _usage = fixture()
    authorization["schemaVersion"] = 2
    authorization["rounds"] = [{"allocationId": item["allocationId"], "executionTemplateDigest": template["templateDigest"]} for item in authorization["rounds"]]
    authorization_digest = sha256(canonical_json(authorization))
    plan["schemaVersion"] = 2; plan["contract"] = "noeris-kernel-tournament-plan-v2"
    plan["controllerAuthorization"] = {
        "ref": f"0research-noeris-tournament-controller-envelope-v2:{authorization_digest}",
        "sha256": authorization_digest,
        "principal": "0research-controller",
        "namespace": "0research-noeris-tournament-plan-v2",
    }
    for item in plan["rounds"]:
        item["executionTemplateDigest"] = template["templateDigest"]
    rehash_plan(plan)
    kernel_ref = "zero-research/noeris-kaggle-t4-001"
    metadata = {"id": kernel_ref, "title": "noeris-kaggle-t4-001", "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": False, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (root / "kernel-metadata.json").write_text(f"{canonical_json(metadata)}\n")
    body = {
        "schemaVersion": 1,
        "contract": "0research-noeris-kaggle-execution-capsule-v1",
        "allocationId": "kaggle-t4-001",
        "kernelRef": kernel_ref,
        "candidate": candidate,
        "controllerEnvelope": authorization,
        "plan": plan,
        "executionTemplate": template,
        "controllerPrincipal": "0research-controller",
        "authority": CAPSULE_AUTHORITY,
    }
    capsule = {**body, "capsuleDigest": domain("0research-noeris-kaggle-execution-capsule-v1", body), "signatureSsh": SIGNATURE}
    capsule_path = root / "execution-capsule.json"; capsule_path.write_text(f"{canonical_json(capsule)}\n")
    return root, capsule_path, capsule


def test_verifies_canonical_capsule_and_exact_offline_template_tree(tmp_path):
    root, capsule_path, capsule = package(tmp_path)
    verified = load_execution_capsule(str(capsule_path), str(root))
    assert verified["capsuleDigest"] == capsule["capsuleDigest"]
    assert verified["executionTemplate"]["templateDigest"] == capsule["executionTemplate"]["templateDigest"]
    assert verified["authority"] == CAPSULE_AUTHORITY


def test_rejects_template_substitution_extra_files_symlinks_and_noncanonical_capsule(tmp_path):
    root, capsule_path, _capsule = package(tmp_path)
    (root / "run.py").write_text("# substituted\n")
    with pytest.raises(ValueError, match="bytes drift"):
        load_execution_capsule(str(capsule_path), str(root))

    root, capsule_path, _capsule = package(tmp_path / "extra")
    (root / "unexpected.py").write_text("pass\n")
    with pytest.raises(ValueError, match="file tree drifts"):
        load_execution_capsule(str(capsule_path), str(root))

    root, capsule_path, _capsule = package(tmp_path / "link")
    (root / "run.py").unlink(); (root / "run.py").symlink_to(capsule_path)
    with pytest.raises(ValueError, match="symlink"):
        load_execution_capsule(str(capsule_path), str(root))

    root, capsule_path, capsule = package(tmp_path / "encoding")
    capsule_path.write_text(json.dumps(capsule, indent=2))
    with pytest.raises(ValueError, match="canonical JSON"):
        load_execution_capsule(str(capsule_path), str(root))


def test_rejects_round_template_drift_before_runtime_initialization(tmp_path, monkeypatch):
    root, capsule_path, capsule = package(tmp_path)
    capsule["plan"]["rounds"][0]["executionTemplateDigest"] = f"sha256:{'f' * 64}"
    write_capsule(capsule_path, capsule)
    monkeypatch.setattr(capture_module, "_capture_environment", lambda: pytest.fail("runtime initialized before capsule verification"))
    with pytest.raises(ValueError, match="exact v2 plan round"):
        capture_module.run_fixed_kaggle_capsule(str(capsule_path), str(root), str(tmp_path / "output"))


def test_rejects_traversal_metadata_authority_and_capsule_digest_drift(tmp_path):
    root, capsule_path, capsule = package(tmp_path / "traversal")
    capsule["executionTemplate"]["files"][0]["path"] = "../run.py"
    write_capsule(capsule_path, capsule)
    with pytest.raises(ValueError, match="POSIX-relative"):
        load_execution_capsule(str(capsule_path), str(root))

    root, capsule_path, _capsule = package(tmp_path / "metadata")
    metadata = json.loads((root / "kernel-metadata.json").read_text()); metadata["enable_internet"] = True
    (root / "kernel-metadata.json").write_text(f"{canonical_json(metadata)}\n")
    with pytest.raises(ValueError, match="expands authority"):
        load_execution_capsule(str(capsule_path), str(root))

    root, capsule_path, capsule = package(tmp_path / "digest")
    capsule["capsuleDigest"] = f"sha256:{'0' * 64}"; capsule_path.write_text(f"{canonical_json(capsule)}\n")
    with pytest.raises(ValueError, match="capsule digest"):
        load_execution_capsule(str(capsule_path), str(root))


def test_v2_capture_binds_capsule_and_template_without_git_and_recovers(tmp_path):
    os.chmod(tmp_path, 0o700)
    root, capsule_path, capsule = package(tmp_path)
    verified = load_execution_capsule(str(capsule_path), str(root))
    plan = verified["plan"]
    repository, evaluator = plan["repository"], plan["evaluator"]
    identity = {"repositoryCommitSha": repository["commitSha"], "repositoryTreeDigest": repository["treeDigest"], "evaluatorDigest": evaluator["digest"], "executionTemplateDigest": verified["executionTemplate"]["templateDigest"]}
    moments = iter([capture_module.datetime(2026, 7, 18, 0, 0, tzinfo=capture_module.timezone.utc), capture_module.datetime(2026, 7, 18, 0, 1, tzinfo=capture_module.timezone.utc)])
    output = tmp_path / "output"
    first = _run_capture_core(candidate=verified["candidate"], authorization=verified["controllerEnvelope"], plan_value=plan, allocation_id=verified["allocationId"], output_directory=str(output), kernel_ref=verified["kernelRef"], environment=environment(), worker_identity=identity, backend=Backend(), execution_capsule_digest=verified["capsuleDigest"], execution_template_digest=verified["executionTemplate"]["templateDigest"], now=lambda: next(moments), publisher=os.rename)
    assert first["capture"]["contract"] == "noeris-kaggle-allocation-capture-v2"
    assert first["capture"]["executionCapsuleDigest"] == capsule["capsuleDigest"]
    assert first["capture"]["executionTemplateDigest"] == capsule["executionTemplate"]["templateDigest"]
    recovered = _run_capture_core(candidate=verified["candidate"], authorization=verified["controllerEnvelope"], plan_value=plan, allocation_id=verified["allocationId"], output_directory=str(output), kernel_ref=verified["kernelRef"], environment=pytest.fail, worker_identity=identity, backend=pytest.fail, execution_capsule_digest=verified["capsuleDigest"], execution_template_digest=verified["executionTemplate"]["templateDigest"])
    assert recovered["recovered"] is True


def test_fixed_capsule_wrapper_has_no_git_identity_dependency(tmp_path, monkeypatch):
    root, capsule_path, _capsule = package(tmp_path)
    monkeypatch.setattr(capture_module.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("git invoked by capsule wrapper"))
    monkeypatch.setattr(capture_module, "_capture_environment", lambda: environment())
    monkeypatch.setattr(capture_module, "TorchTritonBackend", Backend)
    captured = {}
    monkeypatch.setattr(capture_module, "_run_capture_core", lambda **kwargs: captured.update(kwargs) or {"allocationId": kwargs["allocation_id"]})
    result = capture_module.run_fixed_kaggle_capsule(str(capsule_path), str(root), str(tmp_path / "output"))
    assert result["allocationId"] == "kaggle-t4-001"
    assert captured["execution_capsule_digest"].startswith("sha256:")
    assert captured["worker_identity"]["executionTemplateDigest"] == captured["execution_template_digest"]


def test_private_core_and_legacy_wrapper_reject_v2_drift_before_gpu_or_git(tmp_path, monkeypatch):
    os.chmod(tmp_path, 0o700)
    root, capsule_path, _capsule = package(tmp_path)
    verified = load_execution_capsule(str(capsule_path), str(root)); plan = verified["plan"]
    repository, evaluator = plan["repository"], plan["evaluator"]
    identity = {"repositoryCommitSha": repository["commitSha"], "repositoryTreeDigest": repository["treeDigest"], "evaluatorDigest": evaluator["digest"], "executionTemplateDigest": verified["executionTemplate"]["templateDigest"]}
    backend = Backend()
    with pytest.raises(ValueError, match="exact execution capsule"):
        _run_capture_core(candidate=verified["candidate"], authorization=verified["controllerEnvelope"], plan_value=plan, allocation_id=verified["allocationId"], output_directory=str(tmp_path / "output"), kernel_ref=verified["kernelRef"], environment=environment(), worker_identity=identity, backend=backend, execution_capsule_digest="not-a-digest", execution_template_digest=verified["executionTemplate"]["templateDigest"], publisher=os.rename)
    assert backend.calls == []

    documents = {"candidate": verified["candidate"], "authorization": verified["controllerEnvelope"], "plan": plan}
    monkeypatch.setattr(capture_module, "_stable_json", lambda path, _label: documents[path])
    monkeypatch.setattr(capture_module.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("git invoked before legacy v2 rejection"))
    monkeypatch.setattr(capture_module, "_capture_environment", lambda: pytest.fail("runtime initialized before legacy v2 rejection"))
    with pytest.raises(ValueError, match="v1 plans only"):
        capture_module.run_fixed_kaggle_capture("candidate", "authorization", "plan", "kaggle-t4-001", str(tmp_path / "legacy"), verified["kernelRef"])
