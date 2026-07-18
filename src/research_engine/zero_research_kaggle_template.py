"""Deterministically assemble the reviewed Noeris Kaggle execution template."""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

from .zero_research_tournament import canonical_json, sha256


TEMPLATE_SOURCES = {
    "run.py": "src/research_engine/zero_research_kaggle_run.py",
    "research_engine/zero_research_kaggle_capsule.py": "src/research_engine/zero_research_kaggle_capsule.py",
    "research_engine/zero_research_kaggle_capture.py": "src/research_engine/zero_research_kaggle_capture.py",
    "research_engine/zero_research_kaggle_worker.py": "src/research_engine/zero_research_kaggle_worker.py",
    "research_engine/zero_research_tournament.py": "src/research_engine/zero_research_tournament.py",
    "research_engine/zero_research_kaggle_matmul.py": "src/research_engine/zero_research_kaggle_matmul.py",
}
MINIMAL_INIT = b'"""Minimal offline 0research Kaggle template package."""\n'
MAX_SOURCE_BYTES = 16 * 1024 * 1024


def _stable_source(root: Path, relative: str) -> bytes:
    path = root / relative
    if path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError("template source path is not canonical")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size < 1 or before.st_size > MAX_SOURCE_BYTES:
            raise ValueError("template source is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError("template source ended before its stated size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("template source exceeds its stated size")
        value = b"".join(chunks)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("template source changed while read")
        return value
    finally:
        os.close(descriptor)


def _write_exact(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        offset = 0
        while offset < len(value):
            written = os.write(descriptor, value[offset:])
            if written < 1:
                raise OSError("template file write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_execution_template(repository_root: str, output_directory: str) -> dict[str, object]:
    """Create one exact source tree and return its canonical embedded manifest."""

    root = Path(repository_root).absolute()
    if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
        raise ValueError("template repository root is not canonical")
    target = Path(output_directory).absolute()
    parent = target.parent.resolve(strict=True); parent_info = parent.lstat()
    if parent.is_symlink() or parent_info.st_uid != os.getuid() or parent_info.st_mode & 0o077 or target.exists() or target.is_symlink():
        raise ValueError("template output must be absent under an owner-only canonical parent")
    files = {destination: _stable_source(root, source) for destination, source in TEMPLATE_SOURCES.items()}
    files["research_engine/__init__.py"] = MINIMAL_INIT
    target.mkdir(mode=0o700)
    try:
        for relative in sorted(files):
            _write_exact(target / relative, files[relative])
        for directory, _subdirectories, _filenames in os.walk(target, topdown=False):
            descriptor = os.open(directory, os.O_RDONLY)
            try: os.fsync(descriptor)
            finally: os.close(descriptor)
        entries = [{"path": relative, "bytes": len(files[relative]), "sha256": sha256(files[relative])} for relative in sorted(files)]
        body = {"schemaVersion": 1, "contract": "noeris-kaggle-execution-template-v1", "entrypoint": "run.py", "files": entries}
        return {**body, "templateDigest": sha256(f"noeris-kaggle-execution-template-v1\0{canonical_json(body)}\n")}
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
