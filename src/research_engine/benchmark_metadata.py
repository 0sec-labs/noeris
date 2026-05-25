"""Shared metadata helper for benchmark artifacts.

Captures Python, PyTorch, Triton, CUDA, GPU, driver/runtime versions,
git commit, benchmark command, and timestamp.  The resulting dict is
suitable for embedding as an ``"environment"`` block in JSON benchmark
artifacts to make results auditable and reproducible.

Usage in a benchmark script::

    from research_engine.benchmark_metadata import collect_environment

    env = collect_environment(command="python scripts/my_benchmark.py --gpu H100")
    payload = {"environment": env, "results": [...]}

All probes are best-effort: if a library is not importable or a system
call fails, the corresponding field is set to ``None`` rather than
raising.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


def _try_import_attr(module: str, attr: str) -> str | None:
    """Import *module* and return ``getattr(module, attr)`` or ``None``."""
    try:
        mod = __import__(module)
        return str(getattr(mod, attr, None))
    except Exception:
        return None


def _torch_version() -> str | None:
    try:
        import torch
        return torch.__version__
    except Exception:
        return None


def _triton_version() -> str | None:
    try:
        import triton
        return triton.__version__
    except Exception:
        return None


def _cuda_runtime_version() -> str | None:
    """Return the CUDA runtime version string from PyTorch."""
    try:
        import torch
        return torch.version.cuda or None
    except Exception:
        return None


def _cuda_driver_version() -> str | None:
    """Return the CUDA driver version via nvidia-smi, if available."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0].strip() or None
    except Exception:
        pass
    return None


def _gpu_name() -> str | None:
    """Return the GPU device name via PyTorch CUDA, if available."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def _gpu_count() -> int | None:
    """Return the number of visible CUDA devices, if available."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:
        pass
    return None


def _git_commit() -> str | None:
    """Return the short git commit hash of the working tree, if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except Exception:
        pass
    return None


def _git_dirty() -> bool | None:
    """Return whether the working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet", "HEAD"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode != 0
    except Exception:
        return None


def collect_environment(
    *,
    command: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect runtime and hardware metadata for a benchmark artifact.

    Parameters
    ----------
    command:
        The shell command used to launch this benchmark, if known.
    extra:
        Additional key-value pairs to merge into the environment block.

    Returns
    -------
    dict
        A dictionary with the following stable keys:

        - ``timestamp_utc``: ISO-8601 UTC timestamp.
        - ``python_version``: e.g. ``"3.11.12"``.
        - ``platform``: e.g. ``"Linux-6.1.0-x86_64-with-glibc2.36"``.
        - ``torch_version``: PyTorch version string, or ``None``.
        - ``triton_version``: Triton version string, or ``None``.
        - ``cuda_runtime_version``: CUDA runtime version, or ``None``.
        - ``cuda_driver_version``: NVIDIA driver version, or ``None``.
        - ``gpu_name``: GPU device name, or ``None``.
        - ``gpu_count``: Number of visible GPUs, or ``None``.
        - ``git_commit``: Short git commit hash, or ``None``.
        - ``git_dirty``: Whether working tree is dirty, or ``None``.
        - ``command``: The benchmark command, or ``None``.
    """
    env: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": _torch_version(),
        "triton_version": _triton_version(),
        "cuda_runtime_version": _cuda_runtime_version(),
        "cuda_driver_version": _cuda_driver_version(),
        "gpu_name": _gpu_name(),
        "gpu_count": _gpu_count(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "command": command,
    }
    if extra:
        env.update(extra)
    return env


# Stable schema keys — tests can import this to validate shape without
# depending on runtime values.
ENVIRONMENT_SCHEMA_KEYS = frozenset({
    "timestamp_utc",
    "python_version",
    "platform",
    "torch_version",
    "triton_version",
    "cuda_runtime_version",
    "cuda_driver_version",
    "gpu_name",
    "gpu_count",
    "git_commit",
    "git_dirty",
    "command",
})
