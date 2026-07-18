"""Fixed no-argument bootstrap for one offline Noeris Kaggle execution capsule."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path, PurePosixPath


MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024
FORBIDDEN_AUTHORITY_PATHS = (
    "/run/secrets/0research-noeris-worker-key",
    "/run/secrets/0research-noeris-controller-key",
    "/etc/0research/noeris-worker.allowed_signers",
    "/etc/0research/noeris-controller.allowed_signers",
)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(value: str | bytes) -> str:
    encoded = value.encode() if isinstance(value, str) else value
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _stable_file(path: Path, maximum: int, label: str) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size < 1 or before.st_size > maximum:
            raise ValueError(f"{label} is not a bounded single-link regular file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError(f"{label} ended before its stated size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError(f"{label} exceeds its stated size")
        value = b"".join(chunks)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError(f"{label} changed while read")
        return value
    finally:
        os.close(descriptor)


def preflight(package_root: str) -> dict[str, object]:
    """Verify the full inert package before importing any worker-owned module."""

    root = Path(package_root).absolute()
    if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
        raise ValueError("execution package root is not canonical")
    if any(Path(value).exists() or Path(value).is_symlink() for value in FORBIDDEN_AUTHORITY_PATHS):
        raise ValueError("provider template refuses signing or controller authority material")
    capsule_bytes = _stable_file(root / "execution-capsule.json", 8 * 1024 * 1024, "execution capsule")
    try:
        capsule = json.loads(capsule_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("execution capsule is not JSON") from error
    if not isinstance(capsule, dict) or capsule_bytes != f"{_canonical(capsule)}\n".encode() or set(capsule) != {"schemaVersion", "contract", "allocationId", "kernelRef", "candidate", "controllerEnvelope", "plan", "executionTemplate", "controllerPrincipal", "authority", "capsuleDigest", "signatureSsh"}:
        raise ValueError("execution capsule is not canonical or exact")
    template = capsule.get("executionTemplate")
    if not isinstance(template, dict) or set(template) != {"schemaVersion", "contract", "entrypoint", "files", "templateDigest"} or template.get("schemaVersion") != 1 or template.get("contract") != "noeris-kaggle-execution-template-v1" or template.get("entrypoint") != "run.py":
        raise ValueError("execution template identity is invalid")
    raw_files = template.get("files")
    if not isinstance(raw_files, list) or not 1 <= len(raw_files) <= 32:
        raise ValueError("execution template file count is outside bounds")
    files: list[dict[str, object]] = []
    for raw in raw_files:
        if not isinstance(raw, dict) or set(raw) != {"path", "bytes", "sha256"}:
            raise ValueError("execution template file entry is not exact")
        relative, size, digest = raw.get("path"), raw.get("bytes"), raw.get("sha256")
        parsed = PurePosixPath(str(relative))
        if not isinstance(relative, str) or not relative or "\\" in relative or parsed.is_absolute() or parsed.as_posix() != relative or any(part in {"", ".", ".."} for part in parsed.parts) or isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= MAX_FILE_BYTES or not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ValueError("execution template file entry is unsafe")
        files.append({"path": relative, "bytes": size, "sha256": digest})
    paths = [str(item["path"]) for item in files]
    if paths != sorted(paths) or len(set(paths)) != len(paths) or "run.py" not in paths or {"execution-capsule.json", "kernel-metadata.json"} & set(paths) or sum(int(item["bytes"]) for item in files) > MAX_TOTAL_BYTES:
        raise ValueError("execution template inventory is noncanonical")
    template_body = {"schemaVersion": 1, "contract": "noeris-kaggle-execution-template-v1", "entrypoint": "run.py", "files": files}
    template_digest = _sha256(f"noeris-kaggle-execution-template-v1\0{_canonical(template_body)}\n")
    if template.get("templateDigest") != template_digest:
        raise ValueError("execution template digest is not recomputable")
    plan = capsule.get("plan"); allocation_id = capsule.get("allocationId"); kernel_ref = capsule.get("kernelRef")
    rounds = plan.get("rounds") if isinstance(plan, dict) else None
    selected = [item for item in rounds if isinstance(item, dict) and item.get("allocationId") == allocation_id] if isinstance(rounds, list) else []
    if len(selected) != 1 or selected[0].get("executionTemplateDigest") != template_digest:
        raise ValueError("execution template drifts from the selected capsule plan round")
    capsule_body = {key: capsule[key] for key in capsule if key not in {"capsuleDigest", "signatureSsh"}}
    capsule_digest = _sha256(f"0research-noeris-kaggle-execution-capsule-v1\0{_canonical(capsule_body)}\n")
    if capsule.get("schemaVersion") != 1 or capsule.get("contract") != "0research-noeris-kaggle-execution-capsule-v1" or capsule.get("capsuleDigest") != capsule_digest:
        raise ValueError("execution capsule digest is not recomputable")
    metadata_bytes = _stable_file(root / "kernel-metadata.json", 64 * 1024, "Kaggle kernel metadata")
    try:
        metadata = json.loads(metadata_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("Kaggle kernel metadata is not JSON") from error
    expected_metadata = {"id": kernel_ref, "title": str(kernel_ref).split("/", 1)[1] if isinstance(kernel_ref, str) and "/" in kernel_ref else None, "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": False, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    if metadata_bytes != f"{_canonical(metadata)}\n".encode() or metadata != expected_metadata:
        raise ValueError("Kaggle kernel metadata is noncanonical or expands authority")
    expected = {"execution-capsule.json", "kernel-metadata.json", *paths}
    actual_files: set[str] = set(); actual_directories: set[str] = set()
    for entry in root.rglob("*"):
        if entry.is_symlink() or (not entry.is_file() and not entry.is_dir()):
            raise ValueError("execution package contains a symlink or unsupported entry")
        relative = entry.relative_to(root).as_posix()
        if entry.is_file(): actual_files.add(relative)
        else: actual_directories.add(relative)
    expected_directories = {parent.as_posix() for relative in expected for parent in PurePosixPath(relative).parents if parent.as_posix() != "."}
    if actual_files != expected or actual_directories != expected_directories:
        raise ValueError("execution package tree drifts from the capsule")
    for item in files:
        content = _stable_file(root / str(item["path"]), MAX_FILE_BYTES, f"execution template {item['path']}")
        if len(content) != item["bytes"] or _sha256(content) != item["sha256"]:
            raise ValueError("execution template bytes drift from the capsule")
    return capsule


def _fixed_output_directory(working_root: str = "/kaggle/working") -> str:
    outer = Path(working_root).absolute()
    outer_info = outer.lstat()
    if not outer.is_dir() or outer.is_symlink() or outer.resolve(strict=True) != outer or outer_info.st_uid != os.getuid():
        raise ValueError("Kaggle working root is not a canonical owned directory")
    private = outer / "0research"
    try:
        private.mkdir(mode=0o700)
    except FileExistsError:
        pass
    private_info = private.lstat()
    if not private.is_dir() or private.is_symlink() or private.resolve(strict=True) != private or private_info.st_uid != os.getuid() or private_info.st_mode & 0o077:
        raise ValueError("Kaggle private output root is not owner-only")
    return str(private / "capture")


def main() -> int:
    package_root = Path(__file__).absolute().parent
    capsule = preflight(str(package_root))
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(package_root))
    from research_engine.zero_research_kaggle_capture import run_fixed_kaggle_capsule
    from research_engine.zero_research_tournament import canonical_json

    result = run_fixed_kaggle_capsule(str(package_root / "execution-capsule.json"), str(package_root), _fixed_output_directory())
    capture = result.get("capture")
    if not isinstance(capture, dict):
        raise ValueError("fixed capsule execution returned no capture")
    print(canonical_json({"allocationId": result["allocationId"], "captureDigest": capture["captureDigest"], "outputDirectory": result["outputDirectory"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
