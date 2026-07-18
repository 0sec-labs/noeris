"""Fixed-policy Kaggle worker for one controller-planned Noeris allocation.

The worker executes the exact randomized arm/case order, retains bounded raw
float64 reference and repeat-output bytes plus integer-nanosecond timings, and
signs both the untrusted proposal and its raw-artifact receipt. It cannot emit
the independent reference-oracle or controller usage-observer receipts and
therefore cannot create accepted 0brain series evidence by itself.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import time
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, Protocol

from .zero_research_tournament import (
    ArmMeasurement,
    canonical_json,
    sha256,
    _build_untrusted_tournament_proposal,
    verify_ssh_signature,
)


CONTROLLER_POLICY = "/etc/0research/noeris-controller.allowed_signers"
WORKER_POLICY = "/etc/0research/noeris-worker.allowed_signers"
WORKER_KEY = "/run/secrets/0research-noeris-worker-key"
WORKER_KEY_FINGERPRINT = "/etc/0research/noeris-worker-key.fingerprint"
IMAGE_DIGEST_FILE = "/etc/0research/software-image.digest"
WORKER_PRINCIPAL = "noeris-kaggle-worker"
SIGNATURE = re.compile(r"^-----BEGIN SSH SIGNATURE-----\n[\s\S]+\n-----END SSH SIGNATURE-----\n?$")
KERNEL_REF = re.compile(r"^[a-z0-9._-]+/[a-z0-9._-]+$")
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
MAX_RAW_BYTES = 32 * 1024 * 1024
MAX_VERIFIER_SERIES_BYTES = 256 * 1024 * 1024
CONFIG_KEYS = {"BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "num_warps", "num_stages"}
CONFIG_VALUES = {"BLOCK_SIZE_M": {32, 64, 128, 256}, "BLOCK_SIZE_N": {32, 64, 128, 256}, "BLOCK_SIZE_K": {32, 64, 128}, "GROUP_SIZE_M": {4, 8, 16}, "num_warps": {2, 4, 8}, "num_stages": {2, 3, 4, 5}}
GIT = "/usr/bin/git"


SignatureSigner = Callable[[bytes, str], str]


@dataclass(frozen=True, slots=True)
class RawMeasurement:
    reference_bytes: bytes
    output_bytes: tuple[bytes, bytes]
    timings_ns: tuple[int, ...]
    max_absolute_error: float
    max_relative_error: float
    warmups_completed: int


class MeasurementBackend(Protocol):
    def measure(
        self, config: Mapping[str, int], shape: Mapping[str, int], seed: int,
        warmups: int, samples: int,
    ) -> RawMeasurement: ...


def _timestamp(moment: datetime | None = None) -> str:
    value = (moment or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, object], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} has unsupported or missing fields")


def _protected_file(path_value: str, label: str, *, root_owned: bool = True, maximum: int = 64 * 1024) -> bytes:
    supplied = Path(path_value).absolute()
    path = Path(path_value).resolve(strict=True)
    if supplied != path:
        raise ValueError(f"{label} must be a canonical path")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        expected_uid = 0 if root_owned else os.getuid()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_uid != expected_uid or before.st_size < 1 or before.st_size > maximum or before.st_mode & 0o077:
            raise ValueError(f"{label} must be a bounded owner-only canonical file")
        value = os.read(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError(f"{label} changed while read")
        parent = path.parent; parent_info = parent.lstat()
        if parent.is_symlink() or parent_info.st_uid != expected_uid or (root_owned and parent_info.st_mode & 0o022):
            raise ValueError(f"{label} parent is not protected")
        return value
    finally:
        os.close(descriptor)


def _stable_json(path_value: str, label: str) -> object:
    try:
        return json.loads(_protected_file(path_value, label, root_owned=False, maximum=8 * 1024 * 1024))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error


def _fixed_sign(material: bytes, namespace: str) -> str:
    _protected_file(WORKER_KEY, "worker signing key", maximum=64 * 1024)
    expected_fingerprint = _protected_file(WORKER_KEY_FINGERPRINT, "worker key fingerprint", maximum=256).decode().strip()
    public = subprocess.run(["/usr/bin/ssh-keygen", "-y", "-f", WORKER_KEY], capture_output=True, timeout=10, check=True, env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"}).stdout
    fingerprint_result = subprocess.run(["/usr/bin/ssh-keygen", "-lf", "-", "-E", "sha256"], input=public, capture_output=True, text=False, timeout=10, check=True, env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"})
    fingerprint_fields = fingerprint_result.stdout.decode().strip().split()
    if len(fingerprint_fields) < 2 or fingerprint_fields[1] != expected_fingerprint or not expected_fingerprint.startswith("SHA256:"):
        raise ValueError("worker signing key fingerprint is not pinned")
    result = subprocess.run(
        ["/usr/bin/ssh-keygen", "-Y", "sign", "-f", WORKER_KEY, "-n", namespace],
        input=material, capture_output=True, timeout=10, check=False,
        env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    )
    signature = result.stdout.decode("utf-8", errors="strict")
    if result.returncode != 0 or not SIGNATURE.fullmatch(signature):
        raise ValueError("worker signing failed")
    verify_ssh_signature(material, signature, WORKER_PRINCIPAL, namespace, WORKER_POLICY)
    return signature


def _validate_stage(path_value: str) -> Path:
    path = Path(path_value).resolve(strict=True); info = path.lstat()
    if not path.is_dir() or path.is_symlink() or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("worker staging directory must be canonical, owner-only, and owned by the worker")
    return path


def _prepare_stage(path_value: str) -> tuple[Path, Path]:
    target = Path(path_value).absolute(); parent = target.parent.resolve(strict=True); info = parent.lstat()
    if parent.is_symlink() or info.st_uid != os.getuid() or info.st_mode & 0o077 or target.exists() or target.is_symlink():
        raise ValueError("worker output target must be absent under a canonical owner-only parent")
    stage = parent / f".{target.name}-stage-{uuid.uuid4().hex}"
    stage.mkdir(mode=0o700)
    return target, stage


def _retained_json(root: Path, relative: str, label: str) -> tuple[dict[str, object], bytes]:
    path = root / relative
    encoded = _protected_file(str(path), label, root_owned=False, maximum=8 * 1024 * 1024)
    try:
        value = _object(json.loads(encoded), label)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if encoded != f"{canonical_json(value)}\n".encode():
        raise ValueError(f"{label} is not canonical JSON")
    return value, encoded


def _retained_artifact(root: Path, value: object, label: str, *, float64: bool) -> bytes:
    artifact = _object(value, label)
    expected = {"bytes", "path", "sha256"} | ({"dtype", "elements"} if float64 else set())
    _exact(artifact, expected, label)
    relative = artifact.get("path")
    if not isinstance(relative, str) or not relative or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
        raise ValueError(f"{label} path is unsafe")
    path = (root / relative).resolve(strict=True)
    if path.parent != root.resolve(strict=True) and root.resolve(strict=True) not in path.parents:
        raise ValueError(f"{label} escapes the retained allocation")
    material = _protected_file(str(path), label, root_owned=False, maximum=MAX_RAW_BYTES)
    if artifact.get("bytes") != len(material) or artifact.get("sha256") != sha256(material):
        raise ValueError(f"{label} bytes or digest drift")
    if float64 and (artifact.get("dtype") != "float64-le" or artifact.get("elements") != len(material) // 8 or not material or len(material) % 8):
        raise ValueError(f"{label} float64 metadata is invalid")
    return material


def _recover_retained(
    *, target: Path, candidate: object, authorization: object, plan_value: object,
    allocation_id: str, kernel_ref: str, verifier, controller_policy: str,
    worker_policy: str,
) -> dict[str, object]:
    root = _validate_stage(str(target))
    proposal, _proposal_bytes = _retained_json(root, "proposal.json", "retained proposal")
    receipt, _receipt_bytes = _retained_json(root, "artifact-receipt.json", "retained artifact receipt")
    usage, usage_bytes = _retained_json(root, "usage.json", "retained usage receipt")

    proposal_keys = {
        "acceptedBy0brain", "allocationAttestation", "allocationId", "arms", "authority",
        "candidateDigest", "candidateId", "contract", "controllerAuthorization", "environment",
        "evaluator", "evidenceDigest", "hardware", "manifest", "novelty", "operator", "planDigest",
        "planId", "repository", "results", "schemaVersion", "seed", "usage", "workerIdentity",
    }
    _exact(proposal, proposal_keys, "retained proposal")
    plan = _object(plan_value, "tournament plan")
    _validate_resource_ceiling(plan)
    candidate_object = _object(candidate, "candidate")
    if (
        proposal.get("schemaVersion") != 1
        or proposal.get("contract") != "noeris-kernel-tournament-proposal-v1"
        or proposal.get("acceptedBy0brain") is not False
        or proposal.get("allocationId") != allocation_id
        or proposal.get("planId") != plan.get("planId")
        or proposal.get("planDigest") != plan.get("planDigest")
        or proposal.get("candidateId") != candidate_object.get("id")
        or proposal.get("candidateDigest") != sha256(canonical_json(candidate_object))
    ):
        raise ValueError("retained proposal does not bind the requested allocation, plan, and candidate")
    for key in ("arms", "authority", "evaluator", "hardware", "novelty", "operator", "repository"):
        if canonical_json(proposal.get(key)) != canonical_json(plan.get(key)):
            raise ValueError(f"retained proposal {key} drifts from the plan")
    plan_manifest = _object(plan.get("manifest"), "plan manifest")
    if canonical_json(proposal.get("manifest")) != canonical_json({key: plan_manifest[key] for key in ("id", "digest", "corpusDigests")}):
        raise ValueError("retained proposal manifest drifts from the plan")
    selected_rounds = [value for value in plan.get("rounds", []) if isinstance(value, dict) and value.get("allocationId") == allocation_id]
    if len(selected_rounds) != 1 or proposal.get("seed") != selected_rounds[0].get("seed"):
        raise ValueError("retained proposal seed drifts from the requested round")
    authorization_object = _object(authorization, "controller authorization")
    controller = _object(plan.get("controllerAuthorization"), "plan controller authorization")
    authorization_digest = sha256(canonical_json(authorization_object))
    if canonical_json(proposal.get("controllerAuthorization")) != canonical_json(controller) or controller.get("sha256") != authorization_digest or controller.get("ref") != f"0research-noeris-tournament-controller-envelope-v1:{authorization_digest}":
        raise ValueError("retained controller authorization binding is invalid")
    controller_principal = authorization_object.get("controllerPrincipal")
    controller_signature = authorization_object.get("signatureSsh")
    if controller_principal != controller.get("principal") or controller.get("namespace") != "0research-noeris-tournament-plan-v1" or not isinstance(controller_signature, str):
        raise ValueError("retained controller signature identity is invalid")
    unsigned_authorization = {key: value for key, value in authorization_object.items() if key != "signatureSsh"}
    verifier(canonical_json(unsigned_authorization).encode(), controller_signature, str(controller_principal), "0research-noeris-tournament-plan-v1", controller_policy)
    signed_body = {key: value for key, value in proposal.items() if key != "allocationAttestation"}
    evidence_body = {key: value for key, value in signed_body.items() if key != "evidenceDigest"}
    if proposal.get("evidenceDigest") != sha256(canonical_json(evidence_body)):
        raise ValueError("retained proposal evidence digest is invalid")
    attestation = _object(proposal.get("allocationAttestation"), "retained allocation attestation")
    _exact(attestation, {"allocationId", "deviceUuid", "evidenceDigest", "namespace", "principal", "signatureSsh"}, "retained allocation attestation")
    environment = _object(proposal.get("environment"), "retained environment")
    evaluator = _object(plan.get("evaluator"), "plan evaluator")
    worker_identity = _object(proposal.get("workerIdentity"), "retained worker identity")
    repository = _object(plan.get("repository"), "plan repository")
    if environment.get("imageDigest") != evaluator.get("softwareImageDigest") or "t4" not in str(environment.get("gpuName", "")).lower() or worker_identity != {"repositoryCommitSha": repository.get("commitSha"), "repositoryTreeDigest": repository.get("treeDigest"), "evaluatorDigest": evaluator.get("digest")}:
        raise ValueError("retained environment or worker code identity drifts from the plan")
    if (
        attestation.get("allocationId") != allocation_id
        or attestation.get("deviceUuid") != environment.get("deviceUuid")
        or attestation.get("evidenceDigest") != proposal.get("evidenceDigest")
        or attestation.get("namespace") != "0research-noeris-allocation-evidence-v1"
        or attestation.get("principal") != WORKER_PRINCIPAL
        or not isinstance(attestation.get("signatureSsh"), str)
        or not SIGNATURE.fullmatch(str(attestation.get("signatureSsh")))
    ):
        raise ValueError("retained allocation attestation binding is invalid")
    verifier(canonical_json(signed_body).encode(), str(attestation.get("signatureSsh")), WORKER_PRINCIPAL, "0research-noeris-allocation-evidence-v1", worker_policy)

    _exact(usage, {"accelerator", "allocationId", "completedAt", "contract", "costUsd", "kernelRef", "provider", "schemaVersion", "startedAt", "status", "tier"}, "retained usage receipt")
    if usage.get("schemaVersion") != 1 or usage.get("contract") != "noeris-kaggle-usage-v1" or usage.get("allocationId") != allocation_id or usage.get("kernelRef") != kernel_ref or usage.get("provider") != "kaggle" or usage.get("tier") != "free" or usage.get("costUsd") != 0 or usage.get("status") != "complete" or usage.get("accelerator") != "gpu":
        raise ValueError("retained usage receipt does not prove the requested zero-dollar allocation")
    try:
        started = datetime.fromisoformat(str(usage["startedAt"]).replace("Z", "+00:00"))
        completed = datetime.fromisoformat(str(usage["completedAt"]).replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("retained usage timestamps are invalid") from error
    if _timestamp(started) != usage.get("startedAt") or _timestamp(completed) != usage.get("completedAt") or completed <= started or (completed - started).total_seconds() * 1000 > int(_object(plan.get("budget"), "plan budget")["maxWallClockMinutes"]) * 60_000:
        raise ValueError("retained usage interval is invalid or over budget")

    receipt_keys = {"allocationId", "contract", "evidenceDigest", "results", "schemaVersion", "signatureSsh", "usageArtifact", "workerPrincipal"}
    _exact(receipt, receipt_keys, "retained artifact receipt")
    receipt_body = {key: value for key, value in receipt.items() if key != "signatureSsh"}
    if receipt.get("schemaVersion") != 1 or receipt.get("contract") != "noeris-kernel-allocation-artifacts-v1" or receipt.get("allocationId") != allocation_id or receipt.get("evidenceDigest") != proposal.get("evidenceDigest") or receipt.get("workerPrincipal") != WORKER_PRINCIPAL or not isinstance(receipt.get("signatureSsh"), str) or not SIGNATURE.fullmatch(str(receipt.get("signatureSsh"))):
        raise ValueError("retained artifact receipt binding is invalid")
    verifier(canonical_json(receipt_body).encode(), str(receipt.get("signatureSsh")), WORKER_PRINCIPAL, "0research-noeris-allocation-artifacts-v1", worker_policy)
    _retained_artifact(root, receipt.get("usageArtifact"), "retained usage artifact", float64=False)
    usage_artifact = _object(receipt.get("usageArtifact"), "retained usage artifact")
    proposal_usage = _object(proposal.get("usage"), "retained proposal usage")
    if usage_artifact.get("path") != "usage.json" or usage_artifact.get("sha256") != sha256(usage_bytes) or proposal_usage.get("usageReceiptDigest") != sha256(usage_bytes) or proposal_usage.get("provider") != "kaggle" or proposal_usage.get("tier") != "free" or proposal_usage.get("costUsd") != 0:
        raise ValueError("retained usage artifact is not bound to the proposal")

    raw_results = receipt.get("results")
    proposal_results = proposal.get("results")
    if not isinstance(raw_results, list) or not isinstance(proposal_results, list) or len(raw_results) != len(proposal_results):
        raise ValueError("retained raw results do not cover the proposal")
    for raw, measured in zip(raw_results, proposal_results, strict=True):
        raw_item, measured_item = _object(raw, "retained raw result"), _object(measured, "retained proposal result")
        _exact(raw_item, {"armId", "armOrderIndex", "caseId", "caseSeed", "outputs", "reference", "timingsNs", "warmupsCompleted"}, "retained raw result")
        outputs = raw_item.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != 2:
            raise ValueError("retained raw result requires two outputs")
        reference_bytes = _retained_artifact(root, raw_item.get("reference"), "retained reference artifact", float64=True)
        output_bytes = [_retained_artifact(root, output, "retained output artifact", float64=True) for output in outputs]
        timing_ns = raw_item.get("timingsNs")
        if (
            raw_item.get("caseId") != measured_item.get("caseId")
            or raw_item.get("armId") != measured_item.get("armId")
            or raw_item.get("warmupsCompleted") != measured_item.get("warmupsCompleted")
            or not isinstance(timing_ns, list)
            or len(timing_ns) != evaluator.get("samples")
            or any(not isinstance(item, int) or isinstance(item, bool) or item < 1 for item in timing_ns)
            or [item / 1_000_000 for item in timing_ns] != measured_item.get("timingsMs")
            or _object(raw_item.get("reference"), "retained reference").get("sha256") != measured_item.get("referenceDigest")
            or [_object(item, "retained output").get("sha256") for item in outputs] != measured_item.get("repeatOutputDigests")
            or len({_object(item, "retained output").get("path") for item in outputs}) != 2
            or len({_object(item, "retained output").get("sha256") for item in outputs}) != 1
            or any(len(value) != len(reference_bytes) for value in output_bytes)
            or measured_item.get("correct") is not True
            or measured_item.get("deterministic") is not True
        ):
            raise ValueError("retained raw result drifts from the signed proposal")
    allowed_paths = {"proposal.json", "artifact-receipt.json", "usage.json"}
    for raw in raw_results:
        raw_item = _object(raw, "retained raw result")
        allowed_paths.add(str(_object(raw_item["reference"], "retained reference")["path"]))
        allowed_paths.update(str(_object(item, "retained output")["path"]) for item in raw_item["outputs"])
    actual_paths: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("retained allocation contains a symbolic link")
        if path.is_file():
            actual_paths.add(path.relative_to(root).as_posix())
    if actual_paths != allowed_paths:
        raise ValueError("retained allocation contains missing or unsupported files")
    return {"allocationId": allocation_id, "proposal": proposal, "artifactReceipt": receipt, "outputDirectory": str(root), "recovered": True}


def _fsync_tree(root: Path) -> None:
    for directory, _subdirs, _files in os.walk(root, topdown=False):
        descriptor = os.open(directory, os.O_RDONLY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)


def _publish_noreplace(stage: Path, target: Path) -> None:
    if sys.platform != "linux":
        raise RuntimeError("fixed worker requires Linux renameat2 no-replace publication")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("fixed worker host lacks renameat2")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(stage), -100, os.fsencode(target), 1) != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST: raise FileExistsError("worker output target appeared concurrently")
        raise OSError(error, os.strerror(error))
    descriptor = os.open(target.parent, os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def _policy_keys(path_value: str, label: str) -> set[str]:
    lines = _protected_file(path_value, label).decode("utf-8", errors="strict").splitlines()
    keys: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        key_index = next((index for index, field in enumerate(fields) if field.startswith("ssh-") or field.startswith("ecdsa-")), -1)
        if key_index < 0 or key_index + 1 >= len(fields):
            raise ValueError(f"{label} contains an invalid allowed-signers line")
        keys.add(f"{fields[key_index]} {fields[key_index + 1]}")
    if not keys:
        raise ValueError(f"{label} contains no signer keys")
    return keys


def _verify_fixed_trust_separation() -> None:
    if _policy_keys(CONTROLLER_POLICY, "controller signer policy") & _policy_keys(WORKER_POLICY, "worker signer policy"):
        raise ValueError("controller and worker signer policies must be key-disjoint")


def _write_exclusive(path: Path, value: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        os.write(descriptor, value)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_or_match(path: Path, value: bytes) -> None:
    try:
        _write_exclusive(path, value)
    except FileExistsError:
        retained = _protected_file(str(path), "retained raw artifact", root_owned=False, maximum=MAX_RAW_BYTES)
        if retained != value:
            raise ValueError("retained raw artifact conflicts with repeated measurement")


def _artifact(root: Path, relative: str, value: bytes) -> dict[str, object]:
    if not value or len(value) > MAX_RAW_BYTES or len(value) % 8:
        raise ValueError("raw float64 artifact bytes are empty, oversized, or misaligned")
    path = root / relative
    _write_or_match(path, value)
    return {"path": relative, "sha256": sha256(value), "bytes": len(value), "dtype": "float64-le", "elements": len(value) // 8}


def _json_artifact(root: Path, relative: str, value: Mapping[str, object]) -> dict[str, object]:
    encoded = f"{canonical_json(value)}\n".encode()
    _write_exclusive(root / relative, encoded)
    return {"path": relative, "sha256": sha256(encoded), "bytes": len(encoded)}


def repository_tree_digest(root: Path) -> str:
    result = subprocess.run([GIT, "ls-files", "-z"], cwd=root, capture_output=True, timeout=10, check=True, env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"})
    records = bytearray()
    for raw_name in sorted(result.stdout.split(b"\0")):
        if not raw_name:
            continue
        name = raw_name.decode("utf-8", errors="strict")
        path = root / name
        if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != path:
            raise ValueError("tracked repository tree contains an unsupported path")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_uid != os.getuid() or before.st_size > 32 * 1024 * 1024:
                raise ValueError("tracked repository file is unsafe or oversized")
            value = os.read(descriptor, before.st_size + 1); after = os.fstat(descriptor)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError("tracked repository file changed while read")
        finally: os.close(descriptor)
        records.extend(f"{name}\0{len(value)}\0{sha256(value)}\n".encode())
    return sha256(bytes(records))


class TorchTritonBackend:
    def __init__(self) -> None:
        if sys.byteorder != "little":
            raise RuntimeError("raw artifact contract requires a little-endian worker")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("the fixed Noeris worker requires CUDA")
        self.torch = torch
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
        self._inputs: dict[tuple[tuple[tuple[str, int], ...], int], tuple[object, object, object]] = {}

    @staticmethod
    def _sign_tensor(torch, elements: int, seed: int, name: str):
        bits = bytearray(elements)
        offset = 0; counter = 0
        while offset < elements:
            block = hashlib.sha256(f"pinned-float64-matmul-v1\0{seed}\0{name}\0{counter}".encode()).digest(); counter += 1
            for byte in block:
                for bit in range(8):
                    if offset >= elements: break
                    bits[offset] = (byte >> bit) & 1; offset += 1
        return torch.frombuffer(memoryview(bits), dtype=torch.uint8).to(torch.int8).mul_(2).sub_(1)

    def measure(self, config: Mapping[str, int], shape: Mapping[str, int], seed: int, warmups: int, samples: int) -> RawMeasurement:
        from .triton_kernels import matmul
        torch = self.torch
        normalized_shape = tuple(sorted((str(key), int(value)) for key, value in shape.items()))
        key = (normalized_shape, seed)
        if key not in self._inputs:
            m, n, k = int(shape["M"]), int(shape["N"]), int(shape["K"])
            scale_exponent = -math.ceil(math.log2(max(math.sqrt(k), 1)))
            scale = float(2.0**scale_exponent)
            a_signs = self._sign_tensor(torch, m * k, seed, "A").reshape(m, k)
            b_signs = self._sign_tensor(torch, k * n, seed, "B").reshape(k, n)
            a_cpu = a_signs.to(torch.float16).mul_(scale)
            b_cpu = b_signs.to(torch.float16).mul_(scale)
            reference = torch.matmul(a_signs.to(torch.int64), b_signs.to(torch.int64)).to(torch.float64).mul_(scale * scale).contiguous()
            self._inputs[key] = (a_cpu.cuda(), b_cpu.cuda(), reference)
        a, b, reference = self._inputs[key]
        matmul(a, b, dict(config)); torch.cuda.synchronize()  # JIT/setup, not a measured warmup.
        for _ in range(warmups):
            matmul(a, b, dict(config))
        torch.cuda.synchronize()
        timings_ns: list[int] = []
        for _ in range(samples):
            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            start.record(); matmul(a, b, dict(config)); end.record(); end.synchronize()
            nanoseconds = int(round(float(start.elapsed_time(end)) * 1_000_000))
            if nanoseconds < 1:
                raise RuntimeError("CUDA timing produced a non-positive sample")
            timings_ns.append(nanoseconds)
        outputs = tuple(matmul(a, b, dict(config)).detach().cpu().to(torch.float64).contiguous() for _ in range(2))
        output_bytes = tuple(value.numpy().astype("<f8", copy=False).tobytes(order="C") for value in outputs)
        reference_bytes = reference.numpy().astype("<f8", copy=False).tobytes(order="C")
        error = (outputs[0] - reference).abs()
        relative = error / reference.abs().clamp_min(torch.finfo(torch.float64).eps)
        return RawMeasurement(reference_bytes, output_bytes, tuple(timings_ns), float(error.max().item()), float(relative.max().item()), warmups)


def _runtime_environment(plan: Mapping[str, object]) -> dict[str, object]:
    import platform
    import torch
    import triton
    image_digest = _protected_file(IMAGE_DIGEST_FILE, "software image digest", maximum=256).decode().strip()
    if not DIGEST.fullmatch(image_digest):
        raise ValueError("software image digest file is invalid")
    evaluator = _object(plan.get("evaluator"), "plan evaluator")
    if image_digest != evaluator.get("softwareImageDigest"):
        raise ValueError("worker software image drifts from the plan")
    query = subprocess.run(["/usr/bin/nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True, env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"}).stdout.strip().splitlines()
    if len(query) != 1:
        raise ValueError("worker requires exactly one visible GPU")
    device_uuid, driver = (item.strip() for item in query[0].split(",", 1))
    gpu_name = torch.cuda.get_device_name(0)
    if "t4" not in gpu_name.lower() or plan.get("hardware") != "t4":
        raise ValueError("fixed Kaggle worker requires the planned Tesla T4 hardware")
    return {"cudaVersion": str(torch.version.cuda), "deviceUuid": device_uuid, "driverVersion": driver, "gpuName": gpu_name, "imageDigest": image_digest, "pythonVersion": platform.python_version(), "torchVersion": str(torch.__version__), "tritonVersion": str(triton.__version__)}


def _worker_identity(plan: Mapping[str, object], repository_root: Path) -> dict[str, str]:
    repository = _object(plan.get("repository"), "plan repository")
    commit = subprocess.run([GIT, "rev-parse", "HEAD"], cwd=repository_root, capture_output=True, text=True, timeout=10, check=True, env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"}).stdout.strip()
    tree_digest = repository_tree_digest(repository_root)
    evaluator = _object(plan.get("evaluator"), "plan evaluator")
    if commit != repository.get("commitSha") or tree_digest != repository.get("treeDigest"):
        raise ValueError("worker repository commit or tree bytes drift from the plan")
    return {"repositoryCommitSha": commit, "repositoryTreeDigest": tree_digest, "evaluatorDigest": str(evaluator["digest"])}


def _validate_resource_ceiling(plan: Mapping[str, object]) -> None:
    if plan.get("operator") != "matmul" or plan.get("hardware") != "t4":
        raise ValueError("fixed worker accepts only planned matmul on t4")
    arms = _object(plan.get("arms"), "plan arms")
    _exact(arms, {"champion", "challenger"}, "plan arms")
    for arm_id in ("champion", "challenger"):
        arm = _object(arms.get(arm_id), f"{arm_id} arm")
        config = _object(arm.get("config"), f"{arm_id} config")
        if set(config) != CONFIG_KEYS or any(not isinstance(value, int) or isinstance(value, bool) or value not in CONFIG_VALUES[key] for key, value in config.items()):
            raise ValueError("fixed worker accepts only the exact six integer matmul knobs")
    manifest = _object(plan.get("manifest"), "plan manifest")
    cases, rounds = manifest.get("cases"), plan.get("rounds")
    if not isinstance(cases, list) or not isinstance(rounds, list) or not 3 <= len(rounds) <= 5:
        raise ValueError("fixed worker requires a bounded complete tournament plan")
    verifier_bytes = 0
    for value in cases:
        case = _object(value, "plan case"); shape = _object(case.get("shape"), "plan case shape")
        if set(shape) != {"M", "N", "K"} or any(not isinstance(item, int) or isinstance(item, bool) or item < 1 for item in shape.values()):
            raise ValueError("fixed worker case shape is invalid")
        m, n, k = int(shape["M"]), int(shape["N"]), int(shape["K"])
        output_bytes = m * n * 8
        if m * n > 2_000_000 or output_bytes > MAX_RAW_BYTES or 2 * (m * k + k * n) > 128 * 1024 * 1024 or m * n * k > 2**38:
            raise ValueError("fixed worker case exceeds shape, memory, or FLOP ceilings")
        verifier_bytes += len(rounds) * 2 * 3 * output_bytes
    if verifier_bytes > MAX_VERIFIER_SERIES_BYTES:
        raise ValueError("planned raw artifacts exceed the 0brain verifier series ceiling")


def _run_worker_in_stage(
    *, candidate: object, authorization: object, plan_value: object, allocation_id: str,
    output_directory: str, kernel_ref: str, environment: object, worker_identity: object,
    backend: MeasurementBackend, signer: SignatureSigner, verifier=verify_ssh_signature,
    controller_policy: str = CONTROLLER_POLICY, worker_policy: str = WORKER_POLICY,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, object]:
    if not KERNEL_REF.fullmatch(kernel_ref):
        raise ValueError("Kaggle kernel ref is invalid")
    root = _validate_stage(output_directory)
    plan = _object(plan_value, "tournament plan")
    _validate_resource_ceiling(plan)
    rounds = plan.get("rounds")
    manifest = _object(plan.get("manifest"), "tournament manifest")
    cases = manifest.get("cases")
    if not isinstance(rounds, list) or not isinstance(cases, list):
        raise ValueError("tournament plan rounds or cases are invalid")
    selected = [item for item in rounds if isinstance(item, dict) and item.get("allocationId") == allocation_id]
    if len(selected) != 1:
        raise ValueError("worker allocation must select exactly one planned round")
    selected_round = selected[0]
    expected: list[tuple[dict[str, object], str, int]] = []
    orders = selected_round.get("armOrders")
    if not isinstance(orders, list) or len(orders) != len(cases):
        raise ValueError("planned arm orders are invalid")
    for case, order in zip(cases, orders, strict=True):
        case_object, order_object = _object(case, "worker case"), _object(order, "worker arm order")
        if order_object.get("caseId") != case_object.get("id") or not isinstance(order_object.get("order"), list):
            raise ValueError("worker arm order drifts from the manifest")
        for order_index, arm in enumerate(order_object["order"]):
            expected.append((case_object, str(arm), order_index))

    raw_results: list[dict[str, object]] = []
    call_index = 0
    def runner(arm: str, config: Mapping[str, int], shape: Mapping[str, int], seed: int, warmups: int, samples: int, _absolute: float, _relative: float) -> ArmMeasurement:
        nonlocal call_index
        if call_index >= len(expected):
            raise ValueError("worker received more measurements than planned")
        case, expected_arm, order_index = expected[call_index]; call_index += 1
        if arm != expected_arm or dict(shape) != dict(_object(case.get("shape"), "worker case shape")):
            raise ValueError("worker measurement call drifts from planned arm or shape")
        measurement = backend.measure(config, shape, seed, warmups, samples)
        if len(measurement.output_bytes) != 2 or len(measurement.timings_ns) != samples or any(not isinstance(item, int) or isinstance(item, bool) or item < 1 for item in measurement.timings_ns):
            raise ValueError("measurement backend returned invalid raw outputs or timings")
        case_id = str(case["id"]); prefix = f"raw/{case_id}"
        reference = _artifact(root, f"{prefix}/reference.f64le", measurement.reference_bytes)
        outputs = [_artifact(root, f"{prefix}/{arm}-output-{index + 1}.f64le", value) for index, value in enumerate(measurement.output_bytes)]
        raw_results.append({"caseId": case_id, "armId": arm, "armOrderIndex": order_index, "caseSeed": seed, "warmupsCompleted": measurement.warmups_completed, "timingsNs": list(measurement.timings_ns), "reference": reference, "outputs": outputs})
        return ArmMeasurement(str(reference["sha256"]), tuple(str(item["sha256"]) for item in outputs), tuple(item / 1_000_000 for item in measurement.timings_ns), measurement.max_absolute_error, measurement.max_relative_error, measurement.warmups_completed)

    started = now()
    usage_ref: dict[str, object] | None = None
    def usage_provider(_elapsed_ms: int) -> dict[str, object]:
        nonlocal usage_ref
        completed = now()
        if completed <= started:
            raise ValueError("worker completion timestamp must follow its start timestamp")
        budget = _object(plan.get("budget"), "plan budget")
        if (completed - started).total_seconds() * 1000 > int(budget["maxWallClockMinutes"]) * 60_000:
            raise ValueError("worker usage interval exceeds the plan wall-clock budget")
        usage = {"schemaVersion": 1, "contract": "noeris-kaggle-usage-v1", "allocationId": allocation_id, "kernelRef": kernel_ref, "provider": "kaggle", "tier": "free", "costUsd": 0, "status": "complete", "accelerator": "gpu", "startedAt": _timestamp(started), "completedAt": _timestamp(completed)}
        usage_ref = _json_artifact(root, "usage.json", usage)
        return {"provider": "kaggle", "tier": "free", "costUsd": 0, "usageReceiptDigest": usage_ref["sha256"]}

    def attestor(material: bytes, attested_allocation: str, device_uuid: str) -> Mapping[str, str]:
        body = _object(json.loads(material), "proposal signed body")
        signature_value = signer(material, "0research-noeris-allocation-evidence-v1")
        return {"allocationId": attested_allocation, "deviceUuid": device_uuid, "evidenceDigest": str(body["evidenceDigest"]), "namespace": "0research-noeris-allocation-evidence-v1", "principal": WORKER_PRINCIPAL, "signatureSsh": signature_value}

    proposal = _build_untrusted_tournament_proposal(
        plan_value=plan, authorization_value=authorization, candidate_value=candidate,
        allocation_id=allocation_id, environment_value=environment,
        worker_identity_value=worker_identity, worker_usage_value=usage_provider,
        controller_policy=controller_policy, worker_policy=worker_policy,
        runner=runner, attestor=attestor, verifier=verifier,
    )
    if call_index != len(expected) or usage_ref is None:
        raise ValueError("worker did not complete the exact planned allocation")
    receipt_body = {"schemaVersion": 1, "contract": "noeris-kernel-allocation-artifacts-v1", "allocationId": allocation_id, "evidenceDigest": proposal["evidenceDigest"], "workerPrincipal": WORKER_PRINCIPAL, "results": raw_results, "usageArtifact": usage_ref}
    receipt_material = canonical_json(receipt_body).encode()
    receipt = {**receipt_body, "signatureSsh": signer(receipt_material, "0research-noeris-allocation-artifacts-v1")}
    verifier(receipt_material, str(receipt["signatureSsh"]), WORKER_PRINCIPAL, "0research-noeris-allocation-artifacts-v1", worker_policy)
    _write_exclusive(root / "proposal.json", f"{canonical_json(proposal)}\n".encode())
    _write_exclusive(root / "artifact-receipt.json", f"{canonical_json(receipt)}\n".encode())
    return {"allocationId": allocation_id, "proposal": proposal, "artifactReceipt": receipt, "outputDirectory": str(root)}


def _run_worker_core(*, output_directory: str, publisher: Callable[[Path, Path], None] = _publish_noreplace, **kwargs) -> dict[str, object]:
    target = Path(output_directory).absolute()
    if target.exists() and not target.is_symlink():
        return _recover_retained(
            target=target,
            candidate=kwargs["candidate"],
            authorization=kwargs["authorization"],
            plan_value=kwargs["plan_value"],
            allocation_id=kwargs["allocation_id"],
            kernel_ref=kwargs["kernel_ref"],
            verifier=kwargs.get("verifier", verify_ssh_signature),
            controller_policy=kwargs.get("controller_policy", CONTROLLER_POLICY),
            worker_policy=kwargs.get("worker_policy", WORKER_POLICY),
        )
    target, stage = _prepare_stage(output_directory)
    try:
        result = _run_worker_in_stage(output_directory=str(stage), **kwargs)
        _fsync_tree(stage)
        publisher(stage, target)
        result["outputDirectory"] = str(target)
        return result
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def run_fixed_kaggle_worker(candidate_path: str, authorization_path: str, plan_path: str, allocation_id: str, output_directory: str, kernel_ref: str) -> dict[str, object]:
    _verify_fixed_trust_separation()
    candidate = _stable_json(candidate_path, "candidate")
    authorization = _stable_json(authorization_path, "controller authorization")
    plan = _stable_json(plan_path, "tournament plan")
    target = Path(output_directory).absolute()
    if target.exists() and not target.is_symlink():
        return _recover_retained(
            target=target,
            candidate=candidate,
            authorization=authorization,
            plan_value=plan,
            allocation_id=allocation_id,
            kernel_ref=kernel_ref,
            verifier=verify_ssh_signature,
            controller_policy=CONTROLLER_POLICY,
            worker_policy=WORKER_POLICY,
        )
    repository_root = Path(subprocess.run([GIT, "rev-parse", "--show-toplevel"], capture_output=True, text=True, timeout=10, check=True, env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"}).stdout.strip()).resolve(strict=True)
    return _run_worker_core(candidate=candidate, authorization=authorization, plan_value=plan, allocation_id=allocation_id, output_directory=output_directory, kernel_ref=kernel_ref, environment=_runtime_environment(_object(plan, "tournament plan")), worker_identity=_worker_identity(_object(plan, "tournament plan"), repository_root), backend=TorchTritonBackend(), signer=_fixed_sign)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True); parser.add_argument("--authorization", required=True); parser.add_argument("--plan", required=True)
    parser.add_argument("--allocation-id", required=True); parser.add_argument("--output", required=True); parser.add_argument("--kernel-ref", required=True)
    args = parser.parse_args(argv)
    result = run_fixed_kaggle_worker(args.candidate, args.authorization, args.plan, args.allocation_id, args.output, args.kernel_ref)
    print(canonical_json({"allocationId": result["allocationId"], "outputDirectory": result["outputDirectory"], "proposalEvidenceDigest": _object(result["proposal"], "proposal")["evidenceDigest"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
