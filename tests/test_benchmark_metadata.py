"""Tests for research_engine.benchmark_metadata.

All tests run without CUDA — they validate the schema shape and
best-effort fallback behaviour of the metadata helper.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock


# Import helper directly from file to avoid pulling in torch via
# research_engine.__init__.
_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "research_engine"
    / "benchmark_metadata.py"
)
_spec = importlib.util.spec_from_file_location("benchmark_metadata", _MOD_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

collect_environment = _mod.collect_environment
ENVIRONMENT_SCHEMA_KEYS = _mod.ENVIRONMENT_SCHEMA_KEYS


class TestCollectEnvironment(unittest.TestCase):
    """Schema shape and type contract tests."""

    def test_returns_dict_with_stable_keys(self) -> None:
        env = collect_environment()
        self.assertIsInstance(env, dict)
        missing = ENVIRONMENT_SCHEMA_KEYS - set(env.keys())
        self.assertEqual(missing, set(), f"Missing keys: {missing}")

    def test_timestamp_is_valid_iso8601(self) -> None:
        env = collect_environment()
        ts = env["timestamp_utc"]
        self.assertIsInstance(ts, str)
        # Must parse without error
        dt = datetime.fromisoformat(ts)
        self.assertIsNotNone(dt.tzinfo)

    def test_python_version_present(self) -> None:
        env = collect_environment()
        self.assertIsInstance(env["python_version"], str)
        parts = env["python_version"].split(".")
        self.assertGreaterEqual(len(parts), 2)

    def test_platform_present(self) -> None:
        env = collect_environment()
        self.assertIsInstance(env["platform"], str)
        self.assertTrue(len(env["platform"]) > 0)

    def test_command_passthrough(self) -> None:
        cmd = "python scripts/my_bench.py --gpu H100"
        env = collect_environment(command=cmd)
        self.assertEqual(env["command"], cmd)

    def test_command_defaults_to_none(self) -> None:
        env = collect_environment()
        self.assertIsNone(env["command"])

    def test_extra_fields_merged(self) -> None:
        env = collect_environment(extra={"custom_key": 42, "another": "value"})
        self.assertEqual(env["custom_key"], 42)
        self.assertEqual(env["another"], "value")
        # Original keys still present
        self.assertIn("python_version", env)

    def test_json_serializable(self) -> None:
        env = collect_environment(command="test")
        # Must not raise
        text = json.dumps(env, indent=2)
        roundtrip = json.loads(text)
        self.assertEqual(set(roundtrip.keys()), set(env.keys()))

    def test_gpu_fields_none_without_cuda(self) -> None:
        """When torch.cuda is unavailable, GPU fields gracefully return None."""
        # Patch torch.cuda.is_available to return False
        with mock.patch.dict(sys.modules, {"torch": mock.MagicMock()}):
            torch_mock = sys.modules["torch"]
            torch_mock.cuda.is_available.return_value = False
            torch_mock.__version__ = "2.5.0"
            torch_mock.version.cuda = None

            env = collect_environment()
            # GPU-specific fields should be None when CUDA is unavailable
            self.assertIsNone(env["gpu_name"])
            self.assertIsNone(env["gpu_count"])

    def test_schema_keys_frozen(self) -> None:
        """ENVIRONMENT_SCHEMA_KEYS is a frozenset and matches collect output."""
        self.assertIsInstance(ENVIRONMENT_SCHEMA_KEYS, frozenset)
        env = collect_environment()
        # All schema keys must be present in the output (extra may add more)
        self.assertTrue(ENVIRONMENT_SCHEMA_KEYS.issubset(env.keys()))

    def test_environment_block_in_artifact(self) -> None:
        """Simulate embedding the environment block in a benchmark artifact."""
        env = collect_environment(command="python scripts/gemma4_layer_benchmark_pack.py")
        artifact = {
            "environment": env,
            "results": [{"speedup": 1.23}],
        }
        text = json.dumps(artifact, indent=2)
        parsed = json.loads(text)
        self.assertIn("environment", parsed)
        self.assertIn("timestamp_utc", parsed["environment"])
        self.assertIn("python_version", parsed["environment"])


class TestGracefulDegradation(unittest.TestCase):
    """Ensure collect_environment never raises, even without optional deps."""

    def test_no_torch_import(self) -> None:
        """When torch is not importable, version fields are None, no crash."""
        original = sys.modules.get("torch")
        sys.modules["torch"] = None  # type: ignore[assignment]
        try:
            env = collect_environment()
            self.assertIsNone(env["torch_version"])
            self.assertIsNone(env["cuda_runtime_version"])
        finally:
            if original is not None:
                sys.modules["torch"] = original
            else:
                sys.modules.pop("torch", None)

    def test_no_triton_import(self) -> None:
        """When triton is not importable, triton_version is None."""
        original = sys.modules.get("triton")
        sys.modules["triton"] = None  # type: ignore[assignment]
        try:
            env = collect_environment()
            self.assertIsNone(env["triton_version"])
        finally:
            if original is not None:
                sys.modules["triton"] = original
            else:
                sys.modules.pop("triton", None)

    def test_no_git_binary(self) -> None:
        """When git is not on PATH, git fields are None."""
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            env = collect_environment()
            self.assertIsNone(env["git_commit"])
            self.assertIsNone(env["git_dirty"])

    def test_no_nvidia_smi(self) -> None:
        """When nvidia-smi is not on PATH, driver version is None."""
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            env = collect_environment()
            self.assertIsNone(env["cuda_driver_version"])


if __name__ == "__main__":
    unittest.main()
