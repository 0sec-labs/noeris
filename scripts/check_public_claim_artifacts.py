#!/usr/bin/env python3
"""Validate that public benchmark artifact references exist and parse.

Checks public docs for docs/results artifact paths. For every referenced .json
artifact, validates JSON parsing as a basic guardrail against broken claim
links. Also checks a small set of high-value README headline claims against the
stable JSON fields that back them.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCS = [
    ROOT / "README.md",
    ROOT / "docs/paper/noeris.md",
    ROOT / "docs/results/README.md",
]

MARKDOWN_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
RAW_RESULTS_RE = re.compile(r"(?P<path>(?:\.\./)?docs/results/[A-Za-z0-9_./\-]+\.(?:json|md))")

README = ROOT / "README.md"
CLAIM_ARTIFACTS = [
    ROOT / "docs/results/a100-sliding-window-showdown.json",
    ROOT / "docs/results/a100-end-to-end-26layer.json",
    ROOT / "docs/results/a100-19model-generalization.json",
    ROOT / "docs/results/hardware-cross-learning-a100-to-h100.json",
    ROOT / "docs/results/qk-norm-rope-a100-full.json",
    ROOT / "docs/results/qk-norm-rope-h100-full.json",
    ROOT / "docs/results/gemma4-layer-bench-deeper-fusion-a100-after-geglu-retune.json",
    ROOT / "docs/results/gemma4-layer-bench-deeper-fusion-h100-after-geglu-retune.json",
]


def _normalize_path(doc_path: Path, raw: str) -> Path | None:
    raw = raw.strip()
    if not raw:
        return None
    if raw.startswith(("http://", "https://", "mailto:")):
        return None
    if "#" in raw:
        raw = raw.split("#", 1)[0]
    if not raw:
        return None
    if raw.startswith("../results/"):
        # paper-local links like ../results/foo.json
        return (doc_path.parent / raw).resolve()
    if raw.startswith("docs/results/"):
        return (ROOT / raw).resolve()
    if raw.startswith("./docs/results/"):
        return (ROOT / raw[2:]).resolve()
    return None


def _extract_paths(doc_path: Path) -> set[Path]:
    text = doc_path.read_text(encoding="utf-8")
    found: set[Path] = set()

    for m in MARKDOWN_LINK_RE.finditer(text):
        p = _normalize_path(doc_path, m.group(1))
        if p is not None:
            found.add(p)

    for m in RAW_RESULTS_RE.finditer(text):
        p = _normalize_path(doc_path, m.group("path"))
        if p is not None:
            found.add(p)

    return found


def _read_json(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def _find_readme_table_line(readme_text: str, label: str) -> str:
    prefix = f"| {label} |"
    for line in readme_text.splitlines():
        if line.startswith(prefix):
            return line
    raise AssertionError(f"README is missing table row: {label}")


def _expect_number(
    *,
    errors: list[str],
    label: str,
    actual: float,
    expected: float,
    tolerance: float,
) -> None:
    if abs(actual - expected) > tolerance:
        errors.append(
            f"{label}: expected {expected:.4f} +/- {tolerance}, found {actual:.4f}"
        )


def _extract_one_float(pattern: str, text: str, label: str) -> float:
    match = re.search(pattern, text)
    if not match:
        raise AssertionError(f"could not parse {label}: {text}")
    return float(match.group(1))


def _layer_speedups_by_name(payload: dict) -> dict[str, float]:
    rows = payload.get("layer_results")
    if not isinstance(rows, list):
        raise AssertionError("layer artifact is missing layer_results")
    speedups: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", ""))
        if not name:
            continue
        if row.get("correct") is not True:
            raise AssertionError(f"layer result is not correct: {name}")
        speedups[name] = float(row["layer_speedup"])
    return speedups


def _best_qk_speedups_by_shape(payload: dict) -> list[float]:
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise AssertionError("qk artifact is missing results")
    best_by_shape: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("correct") is not True:
            continue
        shape = str(row.get("shape_name", ""))
        if not shape:
            continue
        speedup = float(row["fusion_speedup"])
        best_by_shape[shape] = max(best_by_shape.get(shape, 0.0), speedup)
    if not best_by_shape:
        raise AssertionError("qk artifact has no correct shape results")
    return list(best_by_shape.values())


def _validate_claim_artifact_references(all_paths: set[Path]) -> list[str]:
    errors: list[str] = []
    for path in CLAIM_ARTIFACTS:
        if path not in all_paths:
            errors.append(
                "headline claim artifact is not referenced by a public doc: "
                f"{path.relative_to(ROOT)}"
            )
    return errors


def _validate_readme_headline_claims() -> list[str]:
    errors: list[str] = []
    readme_text = README.read_text(encoding="utf-8")

    sliding = _read_json("docs/results/a100-sliding-window-showdown.json")
    sliding_results = sliding["results"]
    max_sliding_speedup = max(float(row["speedup_vs_sdpa"]) for row in sliding_results)
    sliding_wins = sum(1 for row in sliding_results if row.get("noeris_wins") is True)
    sliding_total = len(sliding_results)
    line = _find_readme_table_line(
        readme_text,
        "Sliding-window attention vs cuDNN FlashAttention (A100)",
    )
    _expect_number(
        errors=errors,
        label="README sliding-window A100 speedup",
        actual=_extract_one_float(r"\*\*(\d+(?:\.\d+)?)x faster\*\*", line, "sliding speedup"),
        expected=max_sliding_speedup,
        tolerance=0.01,
    )
    if f"({sliding_wins}/{sliding_total} shapes)" not in line:
        errors.append(
            "README sliding-window A100 shape count does not match "
            f"artifact: expected ({sliding_wins}/{sliding_total} shapes)"
        )

    e2e = _read_json("docs/results/a100-end-to-end-26layer.json")
    line = _find_readme_table_line(readme_text, "End-to-end 26-layer Gemma 4 (A100)")
    match = re.search(
        r"\*\*(?P<speedup>\d+(?:\.\d+)?)x\*\* "
        r"\((?P<baseline>\d+(?:\.\d+)?) ms &rarr; (?P<noeris>\d+(?:\.\d+)?) ms\)",
        line,
    )
    if not match:
        errors.append(f"README end-to-end A100 row is not parseable: {line}")
    else:
        _expect_number(
            errors=errors,
            label="README end-to-end A100 speedup",
            actual=float(match.group("speedup")),
            expected=float(e2e["speedup"]),
            tolerance=0.01,
        )
        _expect_number(
            errors=errors,
            label="README end-to-end A100 baseline ms",
            actual=float(match.group("baseline")),
            expected=float(e2e["baseline_ms"]),
            tolerance=0.05,
        )
        _expect_number(
            errors=errors,
            label="README end-to-end A100 Noeris ms",
            actual=float(match.group("noeris")),
            expected=float(e2e["noeris_ms"]),
            tolerance=0.05,
        )

    _validate_layer_claim(
        errors=errors,
        readme_text=readme_text,
        label="Gemma 4 decoder layer deeper fusion (A100)",
        artifact="docs/results/gemma4-layer-bench-deeper-fusion-a100-after-geglu-retune.json",
    )
    _validate_layer_claim(
        errors=errors,
        readme_text=readme_text,
        label="Gemma 4 decoder layer deeper fusion (H100)",
        artifact="docs/results/gemma4-layer-bench-deeper-fusion-h100-after-geglu-retune.json",
    )

    transfer = _read_json("docs/results/a100-19model-generalization.json")
    fused_ms_rho = float(transfer["cross_hardware_comparison"]["spearman_rho_fused_ms"])
    fused_rho_count = sum(
        1
        for value in re.findall(r"&rho;=(\d+(?:\.\d+)?)", readme_text)
        if abs(float(value) - fused_ms_rho) <= 0.001
    )
    if fused_rho_count < 2:
        errors.append(
            "README should contain cross-hardware fused-ms rho twice "
            f"(expected approximately {fused_ms_rho:.4f})"
        )

    hardware = _read_json("docs/results/hardware-cross-learning-a100-to-h100.json")
    cost_model_rho = float(hardware["summary"]["mean_spearman_source_hw"])
    if not any(
        abs(float(value) - cost_model_rho) <= 0.001
        for value in re.findall(r"&rho;=(\d+(?:\.\d+)?)", readme_text)
    ):
        errors.append(
            "README should contain cost-model transfer rho "
            f"(expected approximately {cost_model_rho:.4f})"
        )

    _validate_qk_claim(
        errors=errors,
        readme_text=readme_text,
        label="Fused QK-RMSNorm+RoPE prologue (A100)",
        artifact="docs/results/qk-norm-rope-a100-full.json",
        expect_peak=False,
    )
    _validate_qk_claim(
        errors=errors,
        readme_text=readme_text,
        label="Fused QK-RMSNorm+RoPE prologue (H100)",
        artifact="docs/results/qk-norm-rope-h100-full.json",
        expect_peak=True,
    )

    return errors


def _validate_layer_claim(
    *,
    errors: list[str],
    readme_text: str,
    label: str,
    artifact: str,
) -> None:
    line = _find_readme_table_line(readme_text, label)
    actual = [float(value) for value in re.findall(r"\*\*(\d+(?:\.\d+)?)x\*\*", line)]
    expected_by_name = _layer_speedups_by_name(_read_json(artifact))
    expected = [
        expected_by_name["gemma4_31b_local"],
        expected_by_name["gemma4_31b_global"],
        expected_by_name["gemma4_e2b_local"],
    ]
    if len(actual) != len(expected):
        errors.append(f"README {label} row has {len(actual)} speedups, expected {len(expected)}")
        return
    values = zip(actual, expected, strict=True)
    for idx, (actual_value, expected_value) in enumerate(values, start=1):
        _expect_number(
            errors=errors,
            label=f"README {label} speedup {idx}",
            actual=actual_value,
            expected=expected_value,
            tolerance=0.01,
        )


def _validate_qk_claim(
    *,
    errors: list[str],
    readme_text: str,
    label: str,
    artifact: str,
    expect_peak: bool,
) -> None:
    line = _find_readme_table_line(readme_text, label)
    match = re.search(r"\*\*(?P<low>\d+(?:\.\d+)?)--(?P<high>\d+(?:\.\d+)?)x\*\*", line)
    if not match:
        errors.append(f"README {label} row is not parseable: {line}")
        return

    payload = _read_json(artifact)
    best_speedups = _best_qk_speedups_by_shape(payload)
    _expect_number(
        errors=errors,
        label=f"README {label} lower range",
        actual=float(match.group("low")),
        expected=min(best_speedups),
        tolerance=0.1,
    )
    _expect_number(
        errors=errors,
        label=f"README {label} upper range",
        actual=float(match.group("high")),
        expected=max(best_speedups),
        tolerance=0.1,
    )

    if expect_peak:
        peak_match = re.search(r"peak (?P<peak>\d+(?:\.\d+)?) GB/s", line)
        if not peak_match:
            errors.append(f"README {label} row is missing peak GB/s: {line}")
            return
        peak = max(
            float(row["gb_per_s"])
            for row in payload["results"]
            if row.get("correct") is True
        )
        _expect_number(
            errors=errors,
            label=f"README {label} peak GB/s",
            actual=float(peak_match.group("peak")),
            expected=peak,
            tolerance=1.0,
        )


def main() -> int:
    missing: list[Path] = []
    bad_json: list[tuple[Path, str]] = []
    all_paths: set[Path] = set()

    for doc in PUBLIC_DOCS:
        if not doc.exists():
            print(f"ERROR: missing document {doc}")
            return 2
        all_paths.update(_extract_paths(doc))

    for path in sorted(all_paths):
        if not path.exists():
            missing.append(path)
            continue
        if path.suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                bad_json.append((path, str(exc)))

    claim_errors = _validate_claim_artifact_references(all_paths)
    try:
        claim_errors.extend(_validate_readme_headline_claims())
    except Exception as exc:  # noqa: BLE001
        claim_errors.append(f"public claim validation crashed: {type(exc).__name__}: {exc}")

    print(f"Checked {len(all_paths)} docs/results references across {len(PUBLIC_DOCS)} docs.")
    if missing:
        print("\nMissing referenced artifacts:")
        for p in missing:
            print(f"- {p.relative_to(ROOT)}")
    if bad_json:
        print("\nInvalid referenced JSON artifacts:")
        for p, err in bad_json:
            print(f"- {p.relative_to(ROOT)}: {err}")
    if claim_errors:
        print("\nPublic claim mismatches:")
        for err in claim_errors:
            print(f"- {err}")

    if missing or bad_json or claim_errors:
        return 1

    print(
        "All referenced artifacts exist, referenced JSON files parse, "
        "and headline claims match artifacts."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
