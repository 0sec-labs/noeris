"""Exact offline execution-template and post-plan capsule contracts for Kaggle."""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path, PurePosixPath

from .zero_research_tournament import canonical_json, sha256


DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
KERNEL_REF = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}/[a-z0-9][a-z0-9._-]{2,127}$")
SIGNATURE = re.compile(r"^-----BEGIN SSH SIGNATURE-----\n[\s\S]+\n-----END SSH SIGNATURE-----\n?$")
MAX_TEMPLATE_FILES = 32
MAX_TEMPLATE_FILE_BYTES = 16 * 1024 * 1024
MAX_TEMPLATE_BYTES = 32 * 1024 * 1024
CAPSULE_AUTHORITY = {
    "acceptedEvidenceAllowed": False,
    "providerDispatchAllowed": False,
    "signingAllowed": False,
    "learningAllowed": False,
    "trainingAllowed": False,
    "modelWriteAllowed": False,
    "promotionAllowed": False,
    "githubWriteAllowed": False,
    "autoMergeAllowed": False,
    "deploymentAllowed": False,
    "externalPublicationAllowed": False,
}


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _exact(value: dict[str, object], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} has unsupported or missing fields")


def _domain(name: str, value: object) -> str:
    return sha256(f"{name}\0{canonical_json(value)}\n")


def _relative_file(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} is not a canonical POSIX-relative path")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or parsed.as_posix() != value or any(part in {"", ".", ".."} for part in parsed.parts):
        raise ValueError(f"{label} is not a canonical POSIX-relative path")
    return value


def _stable_file(path: Path, maximum: int, label: str) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size < 1 or before.st_size > maximum:
            raise ValueError(f"{label} is not a bounded regular file")
        value = os.read(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError(f"{label} changed while read")
        return value
    finally:
        os.close(descriptor)


def load_execution_capsule(capsule_path: str, package_root: str) -> dict[str, object]:
    """Read the one fixed capsule file and require canonical JSON plus newline."""

    root = Path(package_root).absolute()
    supplied = Path(capsule_path).absolute()
    expected = root / "execution-capsule.json"
    if supplied != expected:
        raise ValueError("execution capsule path is not the fixed package path")
    encoded = _stable_file(supplied, 8 * 1024 * 1024, "execution capsule")
    try:
        value = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError("execution capsule is not JSON") from error
    if encoded != f"{canonical_json(value)}\n".encode():
        raise ValueError("execution capsule is not canonical JSON")
    return validate_execution_capsule(value, package_root)


def validate_execution_template(value: object, package_root: str) -> dict[str, object]:
    """Recompute a canonical template manifest and its exact package file tree."""

    manifest = _object(value, "execution template")
    _exact(manifest, {"schemaVersion", "contract", "entrypoint", "files", "templateDigest"}, "execution template")
    if manifest.get("schemaVersion") != 1 or manifest.get("contract") != "noeris-kaggle-execution-template-v1" or manifest.get("entrypoint") != "run.py":
        raise ValueError("execution template identity is invalid")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not 1 <= len(raw_files) <= MAX_TEMPLATE_FILES:
        raise ValueError("execution template file count is outside bounds")
    files: list[dict[str, object]] = []
    for index, raw in enumerate(raw_files):
        item = _object(raw, f"execution template file {index}")
        _exact(item, {"path", "bytes", "sha256"}, f"execution template file {index}")
        relative = _relative_file(item.get("path"), f"execution template file {index}.path")
        size = item.get("bytes")
        digest = item.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= MAX_TEMPLATE_FILE_BYTES or not isinstance(digest, str) or not DIGEST.fullmatch(digest):
            raise ValueError("execution template file bounds or digest are invalid")
        files.append({"path": relative, "bytes": size, "sha256": digest})
    paths = [str(item["path"]) for item in files]
    if paths != sorted(paths) or len(set(paths)) != len(paths) or "run.py" not in paths or {"execution-capsule.json", "kernel-metadata.json"} & set(paths) or sum(int(item["bytes"]) for item in files) > MAX_TEMPLATE_BYTES:
        raise ValueError("execution template files must be unique, sorted, bounded, and include run.py")
    body = {"schemaVersion": 1, "contract": "noeris-kaggle-execution-template-v1", "entrypoint": "run.py", "files": files}
    template_digest = _domain("noeris-kaggle-execution-template-v1", body)
    if manifest.get("templateDigest") != template_digest:
        raise ValueError("execution template digest is not recomputable")

    root = Path(package_root).absolute()
    if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
        raise ValueError("execution package root is not a canonical directory")
    expected = {"execution-capsule.json", "kernel-metadata.json", *paths}
    actual: set[str] = set()
    actual_directories: set[str] = set()
    for entry in root.rglob("*"):
        if entry.is_symlink() or (not entry.is_file() and not entry.is_dir()):
            raise ValueError("execution package contains a symlink or unsupported entry")
        if entry.is_file():
            actual.add(entry.relative_to(root).as_posix())
        else:
            actual_directories.add(entry.relative_to(root).as_posix())
    if actual != expected:
        raise ValueError("execution package file tree drifts from the capsule template")
    expected_directories = {parent.as_posix() for relative in expected for parent in PurePosixPath(relative).parents if parent.as_posix() != "."}
    if actual_directories != expected_directories:
        raise ValueError("execution package directory tree drifts from the capsule template")
    for item in files:
        content = _stable_file(root / str(item["path"]), MAX_TEMPLATE_FILE_BYTES, f"execution template {item['path']}")
        if len(content) != item["bytes"] or sha256(content) != item["sha256"]:
            raise ValueError("execution template file bytes drift from the capsule manifest")
    return {**body, "templateDigest": template_digest}


def validate_execution_capsule(value: object, package_root: str) -> dict[str, object]:
    """Validate one inert, controller-signed, post-plan capsule and template tree."""

    capsule = _object(value, "execution capsule")
    _exact(capsule, {"schemaVersion", "contract", "allocationId", "kernelRef", "candidate", "controllerEnvelope", "plan", "executionTemplate", "controllerPrincipal", "authority", "capsuleDigest", "signatureSsh"}, "execution capsule")
    if capsule.get("schemaVersion") != 1 or capsule.get("contract") != "0research-noeris-kaggle-execution-capsule-v1":
        raise ValueError("execution capsule identity is invalid")
    allocation_id, kernel_ref, principal = capsule.get("allocationId"), capsule.get("kernelRef"), capsule.get("controllerPrincipal")
    if not isinstance(allocation_id, str) or not SAFE_ID.fullmatch(allocation_id) or not isinstance(kernel_ref, str) or not KERNEL_REF.fullmatch(kernel_ref) or not isinstance(principal, str) or not re.fullmatch(r"[A-Za-z0-9@._-]{3,128}", principal):
        raise ValueError("execution capsule allocation, kernel, or controller identity is invalid")
    signature = capsule.get("signatureSsh")
    if capsule.get("authority") != CAPSULE_AUTHORITY or not isinstance(signature, str) or not SIGNATURE.fullmatch(signature):
        raise ValueError("execution capsule authority or signature metadata is invalid")
    for key in ("candidate", "controllerEnvelope", "plan"):
        _object(capsule.get(key), f"execution capsule {key}")
    template = validate_execution_template(capsule.get("executionTemplate"), package_root)
    plan = _object(capsule.get("plan"), "execution capsule plan")
    envelope = _object(capsule.get("controllerEnvelope"), "execution capsule controller envelope")
    controller = _object(plan.get("controllerAuthorization"), "execution capsule plan controller authorization")
    rounds = plan.get("rounds")
    selected = [item for item in rounds if isinstance(item, dict) and item.get("allocationId") == allocation_id] if isinstance(rounds, list) else []
    if plan.get("schemaVersion") != 2 or plan.get("contract") != "noeris-kernel-tournament-plan-v2" or envelope.get("schemaVersion") != 2 or principal != envelope.get("controllerPrincipal") or principal != controller.get("principal") or len(selected) != 1 or selected[0].get("executionTemplateDigest") != template["templateDigest"]:
        raise ValueError("execution capsule does not bind the exact v2 plan round and template")
    metadata_bytes = _stable_file(Path(package_root).absolute() / "kernel-metadata.json", 64 * 1024, "Kaggle kernel metadata")
    try:
        metadata = _object(json.loads(metadata_bytes), "Kaggle kernel metadata")
    except json.JSONDecodeError as error:
        raise ValueError("Kaggle kernel metadata is not JSON") from error
    if metadata_bytes != f"{canonical_json(metadata)}\n".encode():
        raise ValueError("Kaggle kernel metadata is not canonical JSON")
    _exact(metadata, {"id", "title", "code_file", "language", "kernel_type", "is_private", "enable_gpu", "enable_internet", "machine_shape", "dataset_sources", "competition_sources", "kernel_sources", "model_sources"}, "Kaggle kernel metadata")
    if metadata != {"id": kernel_ref, "title": kernel_ref.split("/", 1)[1], "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": False, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}:
        raise ValueError("Kaggle kernel metadata expands authority or drifts from the capsule")
    body = {key: capsule[key] for key in capsule if key not in {"capsuleDigest", "signatureSsh"}}
    capsule_digest = _domain("0research-noeris-kaggle-execution-capsule-v1", body)
    if capsule.get("capsuleDigest") != capsule_digest:
        raise ValueError("execution capsule digest is not recomputable")
    return {**body, "capsuleDigest": capsule_digest, "signatureSsh": capsule["signatureSsh"], "executionTemplate": template}
