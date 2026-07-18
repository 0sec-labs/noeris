"""Unsigned, bounded Kaggle capture for one planned Noeris allocation.

Kaggle cannot provide independently protected worker signing-key or policy
mounts. This executor therefore retains raw measurements and observable runtime
facts only. A separately isolated controller collector must reverify the signed
plan, provider status, runtime release, code identity, references, and raw bytes
before it may create worker-attested proposal and artifact receipts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

from .zero_research_kaggle_capsule import load_execution_capsule
from .zero_research_kaggle_worker import (
    ALLOCATION_ID,
    GIT,
    KERNEL_REF,
    MeasurementBackend,
    TorchTritonBackend,
    _artifact,
    _fsync_tree,
    _json_artifact,
    _object,
    _prepare_stage,
    _publish_noreplace,
    _retained_artifact,
    _retained_json,
    _stable_json,
    _validate_resource_ceiling,
    _validate_stage,
    _worker_identity,
)
from .zero_research_tournament import _case_seed, canonical_json, sha256
from .zero_research_tournament import (
    AUTHORITY as TOURNAMENT_AUTHORITY,
    _cross_check_authorization,
    _exact,
    _validate_authorization,
    _validate_candidate,
    _validate_plan_components,
)


BUILD_DATE = re.compile(r"^[0-9]{8}-[0-9]{6}$")
GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
CAPTURE_AUTHORITY = {
    "acceptedEvidenceAllowed": False,
    "learningAllowed": False,
    "trainingAllowed": False,
    "modelWriteAllowed": False,
    "promotionAllowed": False,
    "githubWriteAllowed": False,
    "autoMergeAllowed": False,
    "deploymentAllowed": False,
    "externalPublicationAllowed": False,
}


@dataclass(frozen=True, slots=True)
class CaptureInputs:
    candidate: dict[str, object]
    authorization: dict[str, object]
    plan: dict[str, object]
    round: dict[str, object]
    cases: list[dict[str, object]]


def _timestamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _validate_inputs(candidate_value: object, authorization_value: object, plan_value: object, allocation_id: str) -> CaptureInputs:
    candidate = _object(candidate_value, "capture candidate")
    authorization = _object(authorization_value, "capture controller authorization")
    plan = _object(plan_value, "capture tournament plan")
    if not ALLOCATION_ID.fullmatch(allocation_id):
        raise ValueError("capture allocation id is unsafe")
    _validate_candidate(candidate)
    _validate_authorization(authorization)
    _exact(plan, ["arms", "authority", "budget", "candidateDigest", "candidateId", "contract", "controllerAuthorization", "evaluator", "generator", "hardware", "manifest", "novelty", "operator", "planDigest", "planId", "randomizationDigest", "repository", "rounds", "schemaVersion"], "capture tournament plan")
    _validate_resource_ceiling(plan)
    plan_version = plan.get("schemaVersion")
    if plan_version not in {1, 2} or authorization.get("schemaVersion") != plan_version:
        raise ValueError("capture plan and controller authorization versions do not match")
    plan_body = {key: value for key, value in plan.items() if key not in {"planDigest", "planId"}}
    plan_digest = sha256(canonical_json(plan_body))
    if plan.get("contract") != f"noeris-kernel-tournament-plan-v{plan_version}" or plan.get("planDigest") != plan_digest or plan.get("planId") != f"noeris-{plan_digest[7:]}":
        raise ValueError("capture plan identity or digest is invalid")
    candidate_digest = sha256(canonical_json(candidate))
    if plan.get("candidateDigest") != candidate_digest or plan.get("candidateId") != candidate.get("id"):
        raise ValueError("capture plan does not bind the exact candidate")
    authorization_digest = sha256(canonical_json(authorization))
    controller = _object(plan.get("controllerAuthorization"), "capture controller authorization ref")
    if controller.get("sha256") != authorization_digest or controller.get("ref") != f"0research-noeris-tournament-controller-envelope-v{plan_version}:{authorization_digest}":
        raise ValueError("capture plan does not bind the exact controller authorization")
    change = _object(candidate.get("change"), "capture candidate change")
    if candidate.get("project") != "noeris" or change.get("kind") != "kernel_config":
        raise ValueError("capture candidate is not a Noeris kernel configuration")
    controller = _object(plan.get("controllerAuthorization"), "capture controller authorization ref")
    _exact(controller, ["namespace", "principal", "ref", "sha256"], "capture controller authorization ref")
    if authorization.get("candidateDigest") != candidate_digest or authorization.get("controllerPrincipal") != controller.get("principal") or controller.get("namespace") != f"0research-noeris-tournament-plan-v{plan_version}":
        raise ValueError("capture controller authorization identity or candidate binding is invalid")
    budget = _object(plan.get("budget"), "capture plan budget")
    if budget.get("maxUsd") != 0 or plan.get("authority") != TOURNAMENT_AUTHORITY:
        raise ValueError("capture plan must remain zero-dollar and authority-free")
    # These checks are deliberately independent of signature verification. The
    # provider-side capture is untrusted, but it must still refuse malformed or
    # resource-exhausting self-consistent inputs before any GPU work begins.
    _validate_plan_components(plan, authorization, candidate)
    _cross_check_authorization(plan, authorization, candidate)
    rounds = plan.get("rounds")
    selected = [item for item in rounds if isinstance(item, dict) and item.get("allocationId") == allocation_id] if isinstance(rounds, list) else []
    if len(selected) != 1:
        raise ValueError("capture allocation must select exactly one planned round")
    manifest = _object(plan.get("manifest"), "capture manifest")
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("capture manifest cases are invalid")
    cases = [_object(item, "capture case") for item in raw_cases]
    orders = selected[0].get("armOrders")
    if not isinstance(orders, list) or len(orders) != len(cases):
        raise ValueError("capture round does not cover every case")
    return CaptureInputs(candidate, authorization, plan, selected[0], cases)


def _private_directory(root: Path, relative: str) -> Path:
    """Create each staged directory component with owner-only permissions."""

    requested = Path(relative)
    if requested.is_absolute() or not requested.parts or any(part in {"", ".", ".."} for part in requested.parts):
        raise ValueError("capture directory path is unsafe")
    current = root
    for part in requested.parts:
        current = current / part
        try:
            current.mkdir(mode=0o700)
        except FileExistsError:
            pass
        info = current.lstat()
        if not current.is_dir() or current.is_symlink() or info.st_uid != os.getuid() or info.st_mode & 0o077 or current.resolve(strict=True) != current:
            raise ValueError("capture directory is not owner-only")
    return current


def _capture_environment() -> dict[str, object]:
    import torch
    import triton

    build_date = os.environ.get("BUILD_DATE", "")
    image_commit = os.environ.get("GIT_COMMIT", "")
    if not BUILD_DATE.fullmatch(build_date) or not GIT_COMMIT.fullmatch(image_commit):
        raise ValueError("Kaggle runtime BUILD_DATE or GIT_COMMIT is unavailable or invalid")
    query = subprocess.run(
        ["/usr/bin/nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=True,
        env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    ).stdout.strip().splitlines()
    if len(query) != 1:
        raise ValueError("capture requires exactly one visible GPU")
    device_uuid, driver = (item.strip() for item in query[0].split(",", 1))
    gpu_name = torch.cuda.get_device_name(0)
    if "t4" not in gpu_name.lower():
        raise ValueError("capture requires a Tesla T4")
    release = {
        "imageBuildDate": build_date,
        "imageGitCommit": image_commit,
        "cudaVersion": str(torch.version.cuda),
        "pythonVersion": platform.python_version(),
        "torchVersion": str(torch.__version__),
        "tritonVersion": str(triton.__version__),
    }
    return {
        **release,
        "runtimeFingerprintDigest": sha256(f"noeris-kaggle-runtime-v1\0{canonical_json(release)}"),
        "deviceUuid": device_uuid,
        "driverVersion": driver,
        "gpuName": gpu_name,
    }


def _validate_environment(value: object) -> dict[str, object]:
    environment = _object(value, "capture environment")
    expected = {"cudaVersion", "deviceUuid", "driverVersion", "gpuName", "imageBuildDate", "imageGitCommit", "pythonVersion", "runtimeFingerprintDigest", "torchVersion", "tritonVersion"}
    if set(environment) != expected or any(not isinstance(environment[key], str) or not str(environment[key]).strip() for key in expected):
        raise ValueError("capture environment fields are invalid")
    if not BUILD_DATE.fullmatch(str(environment["imageBuildDate"])) or not GIT_COMMIT.fullmatch(str(environment["imageGitCommit"])) or "t4" not in str(environment["gpuName"]).lower():
        raise ValueError("capture environment release or GPU identity is invalid")
    release = {key: environment[key] for key in ("imageBuildDate", "imageGitCommit", "cudaVersion", "pythonVersion", "torchVersion", "tritonVersion")}
    if environment["runtimeFingerprintDigest"] != sha256(f"noeris-kaggle-runtime-v1\0{canonical_json(release)}"):
        raise ValueError("capture runtime fingerprint is not recomputable")
    return environment


def _validate_worker_identity(value: object, plan: Mapping[str, object], execution_template_digest: str | None = None) -> dict[str, object]:
    identity = _object(value, "capture worker identity")
    expected_keys = {"repositoryCommitSha", "repositoryTreeDigest", "evaluatorDigest"} | ({"executionTemplateDigest"} if plan.get("schemaVersion") == 2 else set())
    if set(identity) != expected_keys:
        raise ValueError("capture worker identity fields are invalid")
    repository = _object(plan.get("repository"), "capture plan repository")
    evaluator = _object(plan.get("evaluator"), "capture plan evaluator")
    expected = {"repositoryCommitSha": repository.get("commitSha"), "repositoryTreeDigest": repository.get("treeDigest"), "evaluatorDigest": evaluator.get("digest")}
    if plan.get("schemaVersion") == 2:
        expected["executionTemplateDigest"] = execution_template_digest
    if identity != expected:
        raise ValueError("capture worker identity drifts from the plan")
    return identity


def _expected_results(inputs: CaptureInputs) -> list[tuple[dict[str, object], str, int, int]]:
    orders = inputs.round.get("armOrders")
    if not isinstance(orders, list):
        raise ValueError("capture arm orders are invalid")
    expected: list[tuple[dict[str, object], str, int, int]] = []
    for case, order_value in zip(inputs.cases, orders, strict=True):
        order = _object(order_value, "capture arm order")
        arm_order = order.get("order")
        if order.get("caseId") != case.get("id") or not isinstance(arm_order, list) or sorted(arm_order) != ["challenger", "champion"]:
            raise ValueError("capture arm order drifts from the plan")
        tensor_seed, round_seed = case.get("tensorSeed"), inputs.round.get("seed")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (tensor_seed, round_seed)):
            raise ValueError("capture case or round seed is invalid")
        case_seed = _case_seed(round_seed, tensor_seed, str(case.get("id")))
        expected.extend((case, str(arm_id), index, case_seed) for index, arm_id in enumerate(arm_order))
    return expected


def _execute_capture_in_stage(
    *, candidate: object, authorization: object, plan_value: object,
    allocation_id: str, output_directory: str, kernel_ref: str,
    environment: object, worker_identity: object, backend: MeasurementBackend,
    execution_capsule_digest: str | None = None, execution_template_digest: str | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, object]:
    if not KERNEL_REF.fullmatch(kernel_ref):
        raise ValueError("capture Kaggle kernel ref is invalid")
    inputs = _validate_inputs(candidate, authorization, plan_value, allocation_id)
    root = _validate_stage(output_directory)
    plan, round_value = inputs.plan, inputs.round
    if plan.get("schemaVersion") == 2 and (not isinstance(execution_capsule_digest, str) or not DIGEST.fullmatch(execution_capsule_digest) or not isinstance(execution_template_digest, str) or not DIGEST.fullmatch(execution_template_digest) or round_value.get("executionTemplateDigest") != execution_template_digest):
        raise ValueError("v2 capture requires exact execution capsule and template digests")
    _private_directory(root, allocation_id)
    capture_environment = _validate_environment(environment)
    capture_identity = _validate_worker_identity(worker_identity, plan, execution_template_digest)
    evaluator = _object(plan.get("evaluator"), "capture evaluator")
    samples, warmups = evaluator.get("samples"), evaluator.get("warmups")
    absolute_tolerance, relative_tolerance = evaluator.get("absoluteTolerance"), evaluator.get("relativeTolerance")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (samples, warmups)) or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in (absolute_tolerance, relative_tolerance)):
        raise ValueError("capture evaluator bounds are invalid")
    arms = _object(plan.get("arms"), "capture arms")
    started = now()
    results: list[dict[str, object]] = []
    references: dict[str, str] = {}
    for case, arm_id, order_index, case_seed in _expected_results(inputs):
        case_id, lane = case.get("id"), case.get("lane")
        _private_directory(root, f"{allocation_id}/raw/{case_id}")
        shape = _object(case.get("shape"), "capture case shape")
        arm = _object(arms.get(arm_id), "capture arm")
        config = _object(arm.get("config"), "capture arm config")
        measurement = backend.measure(config, shape, case_seed, warmups, samples)
        if (
            len(measurement.output_bytes) != 2
            or measurement.output_bytes[0] != measurement.output_bytes[1]
            or any(len(value) != len(measurement.reference_bytes) for value in measurement.output_bytes)
            or len(measurement.timings_ns) != samples
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 or item > 3_600_000_000_000 for item in measurement.timings_ns)
            or measurement.warmups_completed != warmups
            or not math.isfinite(measurement.max_absolute_error)
            or not math.isfinite(measurement.max_relative_error)
            or measurement.max_absolute_error < 0
            or measurement.max_relative_error < 0
            or measurement.max_absolute_error > absolute_tolerance
            or measurement.max_relative_error > relative_tolerance
        ):
            raise ValueError("capture measurement is incomplete, nondeterministic, or incorrect")
        prefix = f"{allocation_id}/raw/{case_id}"
        reference = _artifact(root, f"{prefix}/reference.f64le", measurement.reference_bytes)
        outputs = [_artifact(root, f"{prefix}/{arm_id}-output-{index + 1}.f64le", value) for index, value in enumerate(measurement.output_bytes)]
        prior = references.get(str(case_id))
        if prior is not None and prior != reference["sha256"]:
            raise ValueError("capture arms did not use the same reference")
        references[str(case_id)] = str(reference["sha256"])
        results.append({
            "caseId": case_id, "lane": lane, "armId": arm_id,
            "armOrderIndex": order_index, "caseSeed": case_seed,
            "warmupsCompleted": warmups, "timingsNs": list(measurement.timings_ns),
            "maxAbsoluteError": measurement.max_absolute_error,
            "maxRelativeError": measurement.max_relative_error,
            "reference": reference, "outputs": outputs,
        })
    completed = now()
    budget = _object(plan.get("budget"), "capture budget")
    if completed <= started or (completed - started).total_seconds() * 1000 > int(budget["maxWallClockMinutes"]) * 60_000:
        raise ValueError("capture usage interval is invalid or over budget")
    usage = {
        "schemaVersion": 1, "contract": "noeris-kaggle-self-report-v1",
        "allocationId": allocation_id, "kernelRef": kernel_ref,
        "providerClaim": "kaggle", "tierClaim": "free", "costUsdClaim": 0,
        "acceleratorClaim": "gpu", "startedAt": _timestamp(started),
        "completedAt": _timestamp(completed),
        "independentlyObserved": False,
    }
    usage_ref = _json_artifact(root, f"{allocation_id}/usage-self-report.json", usage)
    body = {
        "schemaVersion": plan.get("schemaVersion"), "contract": f"noeris-kaggle-allocation-capture-v{plan.get('schemaVersion')}",
        "acceptedBy0brain": False, "allocationId": allocation_id,
        "kernelRef": kernel_ref, "candidateDigest": sha256(canonical_json(inputs.candidate)),
        "controllerAuthorizationDigest": sha256(canonical_json(inputs.authorization)),
        "planId": plan["planId"], "planDigest": plan["planDigest"],
        "environment": capture_environment,
        "workerIdentity": capture_identity,
        **({"executionCapsuleDigest": execution_capsule_digest, "executionTemplateDigest": execution_template_digest} if plan.get("schemaVersion") == 2 else {}),
        "results": results, "usageSelfReport": usage_ref,
        "authority": CAPTURE_AUTHORITY,
    }
    capture_digest = sha256(canonical_json(body))
    capture = {**body, "captureDigest": capture_digest}
    _json_artifact(root, f"{allocation_id}/capture.json", capture)
    return {"allocationId": allocation_id, "capture": capture, "outputDirectory": str(root)}


def _recover_capture(target: Path, candidate: object, authorization: object, plan_value: object, allocation_id: str, kernel_ref: str, execution_capsule_digest: str | None = None, execution_template_digest: str | None = None) -> dict[str, object]:
    inputs = _validate_inputs(candidate, authorization, plan_value, allocation_id)
    if inputs.plan.get("schemaVersion") == 2 and (not isinstance(execution_capsule_digest, str) or not DIGEST.fullmatch(execution_capsule_digest) or not isinstance(execution_template_digest, str) or not DIGEST.fullmatch(execution_template_digest) or inputs.round.get("executionTemplateDigest") != execution_template_digest):
        raise ValueError("v2 recovery requires exact planned execution capsule and template digests")
    root = _validate_stage(str(target))
    capture, _capture_bytes = _retained_json(root, f"{allocation_id}/capture.json", "retained Kaggle capture")
    expected_keys = {"acceptedBy0brain", "allocationId", "authority", "candidateDigest", "captureDigest", "contract", "controllerAuthorizationDigest", "environment", "kernelRef", "planDigest", "planId", "results", "schemaVersion", "usageSelfReport", "workerIdentity"} | ({"executionCapsuleDigest", "executionTemplateDigest"} if inputs.plan.get("schemaVersion") == 2 else set())
    if set(capture) != expected_keys:
        raise ValueError("retained Kaggle capture has unsupported or missing fields")
    body = {key: value for key, value in capture.items() if key != "captureDigest"}
    if (
        capture.get("schemaVersion") != inputs.plan.get("schemaVersion")
        or capture.get("contract") != f"noeris-kaggle-allocation-capture-v{inputs.plan.get('schemaVersion')}"
        or capture.get("acceptedBy0brain") is not False
        or capture.get("allocationId") != allocation_id
        or capture.get("kernelRef") != kernel_ref
        or capture.get("candidateDigest") != sha256(canonical_json(inputs.candidate))
        or capture.get("controllerAuthorizationDigest") != sha256(canonical_json(inputs.authorization))
        or capture.get("planId") != inputs.plan.get("planId")
        or capture.get("planDigest") != inputs.plan.get("planDigest")
        or capture.get("captureDigest") != sha256(canonical_json(body))
        or capture.get("authority") != CAPTURE_AUTHORITY
        or (inputs.plan.get("schemaVersion") == 2 and (capture.get("executionCapsuleDigest") != execution_capsule_digest or capture.get("executionTemplateDigest") != execution_template_digest))
    ):
        raise ValueError("retained Kaggle capture binding or digest is invalid")
    _validate_environment(capture.get("environment"))
    _validate_worker_identity(capture.get("workerIdentity"), inputs.plan, execution_template_digest)
    usage_ref = _object(capture.get("usageSelfReport"), "retained capture usage ref")
    if usage_ref.get("path") != f"{allocation_id}/usage-self-report.json":
        raise ValueError("retained capture usage path is noncanonical")
    usage_bytes = _retained_artifact(root, usage_ref, "retained capture usage", float64=False)
    try:
        usage = _object(json.loads(usage_bytes), "retained capture usage")
    except json.JSONDecodeError as error:
        raise ValueError("retained capture usage is not JSON") from error
    expected_usage_keys = {"acceleratorClaim", "allocationId", "completedAt", "contract", "costUsdClaim", "independentlyObserved", "kernelRef", "providerClaim", "schemaVersion", "startedAt", "tierClaim"}
    if set(usage) != expected_usage_keys or usage.get("schemaVersion") != 1 or usage.get("contract") != "noeris-kaggle-self-report-v1" or usage.get("allocationId") != allocation_id or usage.get("kernelRef") != kernel_ref or usage.get("providerClaim") != "kaggle" or usage.get("tierClaim") != "free" or usage.get("costUsdClaim") != 0 or usage.get("acceleratorClaim") != "gpu" or usage.get("independentlyObserved") is not False:
        raise ValueError("retained capture usage claims are invalid")
    if usage_bytes != f"{canonical_json(usage)}\n".encode():
        raise ValueError("retained capture usage is not canonical JSON")
    try:
        started = datetime.fromisoformat(str(usage["startedAt"]).replace("Z", "+00:00"))
        completed = datetime.fromisoformat(str(usage["completedAt"]).replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("retained capture usage timestamps are invalid") from error
    budget = _object(inputs.plan.get("budget"), "capture budget")
    if _timestamp(started) != usage["startedAt"] or _timestamp(completed) != usage["completedAt"] or completed <= started or (completed - started).total_seconds() * 1000 > int(budget["maxWallClockMinutes"]) * 60_000:
        raise ValueError("retained capture usage interval is invalid or over budget")
    results = capture.get("results")
    expected_results = _expected_results(inputs)
    if not isinstance(results, list) or len(results) != len(expected_results):
        raise ValueError("retained capture results are invalid or incomplete")
    evaluator = _object(inputs.plan.get("evaluator"), "capture evaluator")
    samples, warmups = evaluator.get("samples"), evaluator.get("warmups")
    absolute_tolerance, relative_tolerance = evaluator.get("absoluteTolerance"), evaluator.get("relativeTolerance")
    allowed = {f"{allocation_id}/capture.json", f"{allocation_id}/usage-self-report.json"}
    references: dict[str, object] = {}
    for value, expected in zip(results, expected_results, strict=True):
        result = _object(value, "retained capture result")
        if set(result) != {"armId", "armOrderIndex", "caseId", "caseSeed", "lane", "maxAbsoluteError", "maxRelativeError", "outputs", "reference", "timingsNs", "warmupsCompleted"}:
            raise ValueError("retained capture result fields are invalid")
        case_id, arm_id = result.get("caseId"), result.get("armId")
        case, expected_arm, expected_order, expected_seed = expected
        timings = result.get("timingsNs")
        if (
            case_id != case.get("id") or result.get("lane") != case.get("lane")
            or arm_id != expected_arm or result.get("armOrderIndex") != expected_order
            or result.get("caseSeed") != expected_seed or result.get("warmupsCompleted") != warmups
            or not isinstance(timings, list) or len(timings) != samples
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 or item > 3_600_000_000_000 for item in timings)
            or isinstance(result.get("maxAbsoluteError"), bool) or not isinstance(result.get("maxAbsoluteError"), (int, float))
            or isinstance(result.get("maxRelativeError"), bool) or not isinstance(result.get("maxRelativeError"), (int, float))
            or not 0 <= float(result["maxAbsoluteError"]) <= float(absolute_tolerance)
            or not 0 <= float(result["maxRelativeError"]) <= float(relative_tolerance)
        ):
            raise ValueError("retained capture result drifts from the exact plan")
        reference = _object(result.get("reference"), "retained capture reference")
        outputs = result.get("outputs")
        if reference.get("path") != f"{allocation_id}/raw/{case_id}/reference.f64le" or not isinstance(outputs, list) or len(outputs) != 2:
            raise ValueError("retained capture raw paths are invalid")
        reference_bytes = _retained_artifact(root, reference, "retained capture reference", float64=True)
        allowed.add(str(reference["path"]))
        prior_reference = references.get(str(case_id))
        if prior_reference is not None and prior_reference != reference.get("sha256"):
            raise ValueError("retained capture arms use different references")
        references[str(case_id)] = reference.get("sha256")
        digests: set[object] = set()
        for index, output_value in enumerate(outputs):
            output = _object(output_value, "retained capture output")
            if output.get("path") != f"{allocation_id}/raw/{case_id}/{arm_id}-output-{index + 1}.f64le":
                raise ValueError("retained capture output path is invalid")
            output_bytes = _retained_artifact(root, output, "retained capture output", float64=True)
            if len(output_bytes) != len(reference_bytes) or output.get("elements") != reference.get("elements"):
                raise ValueError("retained capture output shape drifts from its reference")
            allowed.add(str(output["path"])); digests.add(output.get("sha256"))
        if len(digests) != 1:
            raise ValueError("retained capture outputs are nondeterministic")
    entries = list(root.rglob("*"))
    if any(path.is_symlink() or (not path.is_file() and not path.is_dir()) for path in entries):
        raise ValueError("retained capture tree contains unsupported entries")
    for directory in (path for path in entries if path.is_dir()):
        info = directory.lstat()
        if info.st_uid != os.getuid() or info.st_mode & 0o077 or directory.resolve(strict=True) != directory:
            raise ValueError("retained capture tree contains an unprotected directory")
    actual = {path.relative_to(root).as_posix() for path in entries if path.is_file()}
    expected_directories = {parent.as_posix() for relative in allowed for parent in Path(relative).parents if parent.as_posix() != "."}
    actual_directories = {path.relative_to(root).as_posix() for path in entries if path.is_dir()}
    if actual != allowed or actual_directories != expected_directories:
        raise ValueError("retained capture tree contains unsupported entries")
    return {"allocationId": allocation_id, "capture": capture, "outputDirectory": str(root), "recovered": True}


def _run_capture_core(*, output_directory: str, publisher: Callable[[Path, Path], None] = _publish_noreplace, **kwargs) -> dict[str, object]:
    target = Path(output_directory).absolute()
    if target.exists() and not target.is_symlink():
        return _recover_capture(target, kwargs["candidate"], kwargs["authorization"], kwargs["plan_value"], kwargs["allocation_id"], kwargs["kernel_ref"], kwargs.get("execution_capsule_digest"), kwargs.get("execution_template_digest"))
    target, stage = _prepare_stage(output_directory)
    try:
        result = _execute_capture_in_stage(output_directory=str(stage), **kwargs)
        _fsync_tree(stage); publisher(stage, target); result["outputDirectory"] = str(target); return result
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def run_fixed_kaggle_capture(candidate_path: str, authorization_path: str, plan_path: str, allocation_id: str, output_directory: str, kernel_ref: str) -> dict[str, object]:
    candidate = _stable_json(candidate_path, "capture candidate")
    authorization = _stable_json(authorization_path, "capture controller authorization")
    plan = _stable_json(plan_path, "capture tournament plan")
    plan_object = _object(plan, "capture tournament plan")
    if plan_object.get("schemaVersion") != 1:
        raise ValueError("legacy Kaggle capture accepts v1 plans only; v2 requires the fixed capsule wrapper")
    target = Path(output_directory).absolute()
    if target.exists() and not target.is_symlink():
        return _recover_capture(target, candidate, authorization, plan, allocation_id, kernel_ref)
    repository_root = Path(subprocess.run([GIT, "rev-parse", "--show-toplevel"], capture_output=True, text=True, timeout=10, check=True, env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"}).stdout.strip()).resolve(strict=True)
    return _run_capture_core(candidate=candidate, authorization=authorization, plan_value=plan, allocation_id=allocation_id, output_directory=output_directory, kernel_ref=kernel_ref, environment=_capture_environment(), worker_identity=_worker_identity(plan_object, repository_root), backend=TorchTritonBackend())


def run_fixed_kaggle_capsule(capsule_path: str, package_root: str, output_directory: str) -> dict[str, object]:
    """Execute one verified v2 capsule without requiring a provider-side git checkout."""

    capsule = load_execution_capsule(capsule_path, package_root)
    candidate = _object(capsule["candidate"], "capsule candidate")
    authorization = _object(capsule["controllerEnvelope"], "capsule controller envelope")
    plan = _object(capsule["plan"], "capsule plan")
    allocation_id, kernel_ref = str(capsule["allocationId"]), str(capsule["kernelRef"])
    inputs = _validate_inputs(candidate, authorization, plan, allocation_id)
    template_digest = str(_object(capsule["executionTemplate"], "capsule execution template")["templateDigest"])
    if inputs.round.get("executionTemplateDigest") != template_digest:
        raise ValueError("capsule execution template drifts from the selected signed round")
    target = Path(output_directory).absolute()
    if target.exists() and not target.is_symlink():
        return _recover_capture(target, candidate, authorization, plan, allocation_id, kernel_ref, str(capsule["capsuleDigest"]), template_digest)
    repository = _object(plan.get("repository"), "capsule plan repository")
    evaluator = _object(plan.get("evaluator"), "capsule plan evaluator")
    worker_identity = {"repositoryCommitSha": repository["commitSha"], "repositoryTreeDigest": repository["treeDigest"], "evaluatorDigest": evaluator["digest"], "executionTemplateDigest": template_digest}
    result = _run_capture_core(candidate=candidate, authorization=authorization, plan_value=plan, allocation_id=allocation_id, output_directory=output_directory, kernel_ref=kernel_ref, environment=_capture_environment(), worker_identity=worker_identity, backend=TorchTritonBackend(), execution_capsule_digest=str(capsule["capsuleDigest"]), execution_template_digest=template_digest)
    after = load_execution_capsule(capsule_path, package_root)
    if after["capsuleDigest"] != capsule["capsuleDigest"] or _object(after["executionTemplate"], "post-execution template")["templateDigest"] != template_digest:
        raise ValueError("execution capsule or template changed during GPU work")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True); parser.add_argument("--authorization", required=True); parser.add_argument("--plan", required=True)
    parser.add_argument("--allocation-id", required=True); parser.add_argument("--output", required=True); parser.add_argument("--kernel-ref", required=True)
    args = parser.parse_args(argv)
    result = run_fixed_kaggle_capture(args.candidate, args.authorization, args.plan, args.allocation_id, args.output, args.kernel_ref)
    print(canonical_json({"allocationId": result["allocationId"], "captureDigest": _object(result["capture"], "capture")["captureDigest"], "outputDirectory": result["outputDirectory"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
