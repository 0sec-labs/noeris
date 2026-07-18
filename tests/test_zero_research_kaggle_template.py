from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from research_engine.zero_research_kaggle_run import _fixed_output_directory, preflight
from research_engine.zero_research_kaggle_template import build_execution_template
from research_engine.zero_research_tournament import canonical_json, sha256
from tests.test_zero_research_kaggle_capsule import package, write_capsule
from tests.test_zero_research_tournament import rehash_plan


def sealed_template(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    template_parent = tmp_path / "built"; template_parent.mkdir(mode=0o700, parents=True)
    root = template_parent / "template"
    manifest = build_execution_template(str(repository_root), str(root))
    _inert_root, _inert_path, capsule = package(tmp_path / "inert")
    authorization = capsule["controllerEnvelope"]; plan = capsule["plan"]
    for item in authorization["rounds"]:
        item["executionTemplateDigest"] = manifest["templateDigest"]
    authorization_digest = sha256(canonical_json(authorization))
    plan["controllerAuthorization"]["sha256"] = authorization_digest
    plan["controllerAuthorization"]["ref"] = f"0research-noeris-tournament-controller-envelope-v2:{authorization_digest}"
    for item in plan["rounds"]:
        item["executionTemplateDigest"] = manifest["templateDigest"]
    rehash_plan(plan)
    capsule["executionTemplate"] = manifest
    metadata = {"id": capsule["kernelRef"], "title": str(capsule["kernelRef"]).split("/", 1)[1], "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": False, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (root / "kernel-metadata.json").write_text(f"{canonical_json(metadata)}\n")
    capsule_path = root / "execution-capsule.json"; write_capsule(capsule_path, capsule)
    return repository_root, root, manifest, capsule_path, capsule


def test_builds_a_byte_identical_closed_template_and_preflights_without_worker_imports(tmp_path, monkeypatch):
    repository_root, root, manifest, _capsule_path, capsule = sealed_template(tmp_path)
    second_parent = tmp_path / "second"; second_parent.mkdir(mode=0o700)
    second_root = second_parent / "template"
    second = build_execution_template(str(repository_root), str(second_root))
    assert second == manifest
    for item in manifest["files"]:
        assert (root / item["path"]).read_bytes() == (second_root / item["path"]).read_bytes()

    imported = []; original_import = __import__
    monkeypatch.setattr("builtins.__import__", lambda name, *args, **kwargs: imported.append(name) or original_import(name, *args, **kwargs))
    verified = preflight(str(root))
    assert verified["capsuleDigest"] == capsule["capsuleDigest"]
    assert not any(name.startswith("research_engine") for name in imported)


def test_template_contains_every_static_internal_import_and_fixed_bootstrap_preflights_first(tmp_path):
    _repository_root, root, manifest, _capsule_path, _capsule = sealed_template(tmp_path)
    paths = {item["path"] for item in manifest["files"]}
    for item in manifest["files"]:
        if not str(item["path"]).endswith(".py"):
            continue
        tree = ast.parse((root / item["path"]).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                assert f"research_engine/{node.module.replace('.', '/')}.py" in paths
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and node.module.startswith("research_engine."):
                assert f"{node.module.replace('.', '/')}.py" in paths
    bootstrap = (root / "run.py").read_text()
    assert bootstrap.index("capsule = preflight") < bootstrap.index("from research_engine.zero_research_kaggle_capture")
    assert "argparse" not in bootstrap

    probe = "import sys;sys.dont_write_bytecode=True;sys.path.insert(0," + repr(str(root)) + ");from research_engine.zero_research_kaggle_capture import run_fixed_kaggle_capsule;assert callable(run_fixed_kaggle_capsule)"
    imported = subprocess.run([sys.executable, "-I", "-c", probe], cwd="/", capture_output=True, text=True, timeout=10)
    assert imported.returncode == 0, imported.stderr
    assert not (root / ".git").exists()


def test_preflight_rejects_source_metadata_and_tree_drift(tmp_path):
    _repository_root, root, _manifest, _capsule_path, _capsule = sealed_template(tmp_path / "source")
    source = root / "research_engine/zero_research_kaggle_capture.py"; source.write_text(f"{source.read_text()}\n# drift\n")
    with pytest.raises(ValueError, match="template bytes drift"):
        preflight(str(root))

    _repository_root, root, _manifest, _capsule_path, _capsule = sealed_template(tmp_path / "metadata")
    metadata = json.loads((root / "kernel-metadata.json").read_text()); metadata["enable_internet"] = True
    (root / "kernel-metadata.json").write_text(f"{canonical_json(metadata)}\n")
    with pytest.raises(ValueError, match="expands authority"):
        preflight(str(root))

    _repository_root, root, _manifest, _capsule_path, _capsule = sealed_template(tmp_path / "tree")
    (root / "research_engine/unreviewed.py").write_text("pass\n")
    with pytest.raises(ValueError, match="package tree drifts"):
        preflight(str(root))


def test_builder_rejects_unsafe_output_and_source_substitution(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    parent = tmp_path / "parent"; parent.mkdir(mode=0o700)
    existing = parent / "existing"; existing.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="output must be absent"):
        build_execution_template(str(repository_root), str(existing))
    source = repository_root / "src/research_engine/zero_research_kaggle_run.py"
    link = tmp_path / "source-link"; link.symlink_to(source)
    with pytest.raises(ValueError, match="repository root is not canonical"):
        build_execution_template(str(link), str(parent / "unused"))


def test_fixed_output_staging_creates_a_private_child_under_a_permissive_working_root(tmp_path):
    working = tmp_path / "working"; working.mkdir(mode=0o755); working.chmod(0o755)
    output = Path(_fixed_output_directory(str(working)))
    assert output == working / "0research" / "capture"
    assert not output.exists()
    assert (working / "0research").stat().st_mode & 0o077 == 0
    (working / "0research").chmod(0o755)
    with pytest.raises(ValueError, match="owner-only"):
        _fixed_output_directory(str(working))
