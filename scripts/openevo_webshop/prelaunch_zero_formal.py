"""Zero-formal selected-GPU qualification and Formal Start receipt helpers.

This module is intentionally authority-free.  It builds/validates engineering
receipts around an already-authorized launch path; it never selects GPUs,
changes science, or grants retry/reuse authority.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

PRELAUNCH_SCHEMA = "openevo.prelaunch-zero-formal.202609281010"
PAYLOAD_SCHEMA = "openevo.prelaunch-zero-formal-payload.202609281010"
FORMAL_START_SCHEMA = "openevo.formal-start.202609281010"
QUALIFICATION_SCHEMA = "openevo.zero-formal-qualification.202609282338"
_ATTEMPT_RE = re.compile(r"^prelaunch-zero-formal-attempt-(\d{4})\.json$")

PAYLOAD_PROGRAM = r"""
import json
import math
import os
import sys
from pathlib import Path

PAYLOAD_SCHEMA = "openevo.prelaunch-zero-formal-payload.202609281010"
marker = Path(sys.argv[1])
expected_gpu_count = int(sys.argv[2])
def emit(status, stage, **extra):
    payload = {
        "schema": PAYLOAD_SCHEMA,
        "status": status,
        "stage": stage,
        "expected_gpu_count": expected_gpu_count,
        "formal_task_consumed": False,
        "model_calls": 0,
        "training_calls": 0,
        "final_panel_access": 0,
        "scientific_state_mutated": False,
        **extra,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)

try:
    payload = b"openevo-prelaunch-zero-formal\n"
    with marker.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if marker.read_bytes() != payload:
        raise RuntimeError("output marker readback mismatch")
    marker.unlink()
except BaseException as exc:
    emit("FAIL", "OUTPUT_WRITE", error_type=type(exc).__name__, error=str(exc)[:2000])
    raise SystemExit(42)

try:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    observed_gpu_count = int(torch.cuda.device_count())
    if observed_gpu_count != expected_gpu_count:
        raise RuntimeError(
            f"visible CUDA device count {observed_gpu_count} != expected {expected_gpu_count}"
        )

    devices = []
    for index in range(observed_gpu_count):
        capability = tuple(int(v) for v in torch.cuda.get_device_capability(index))
        tensor = torch.arange(16, dtype=torch.float32, device=f"cuda:{index}")
        checksum = float(((tensor * tensor) + 1.0).sum().item())
        torch.cuda.synchronize(index)
        if not math.isfinite(checksum):
            raise RuntimeError(f"non-finite CUDA kernel result on device {index}")
        devices.append(
            {
                "visible_index": index,
                "name": str(torch.cuda.get_device_name(index)),
                "capability": list(capability),
                "kernel_checksum": checksum,
            }
        )
    emit(
        "PASS",
        "COMPLETE",
        observed_gpu_count=observed_gpu_count,
        torch_version=str(torch.__version__),
        torch_cuda_version=str(torch.version.cuda),
        devices=devices,
    )
except BaseException as exc:
    emit("FAIL", "CUDA_KERNEL", error_type=type(exc).__name__, error=str(exc)[:2000])
    raise SystemExit(43)
"""


class PrelaunchError(RuntimeError):
    """The zero-formal qualification receipt is missing, ambiguous, or invalid."""


def build_qualification_receipt(
    *,
    campaign_id: str,
    execution_sha: str,
    current_campaign_sha256: str,
    design_sha256: str,
    activation_sha256: str,
    profile_sha256: str | None,
    host_router_path: str,
    host_control_head: str,
    registry_pointer_path: str,
    registry_pointer_sha256: str,
    selected_gpu_uuids: Sequence[str],
    prelaunch_receipt: Mapping[str, Any] | None,
    prelaunch_receipt_file_sha256: str | None = None,
    status: str = "ZERO_FORMAL_QUALIFIED",
    failure_reason: str | None = None,
) -> dict[str, Any]:
    """Bind an engineering-only qualification without granting launch authority."""
    projection = {
        "schema": QUALIFICATION_SCHEMA,
        "status": status,
        "campaign_id": campaign_id,
        "execution_sha": execution_sha,
        "candidate_hashes": {
            "current_campaign": current_campaign_sha256,
            "design": design_sha256,
            "activation": activation_sha256,
            "runtime_profile": profile_sha256,
        },
        "host_router_path": host_router_path,
        "host_control_head": host_control_head,
        "registry_pointer": {
            "path": registry_pointer_path,
            "sha256": registry_pointer_sha256,
        },
        "selected_gpu_uuids": list(selected_gpu_uuids),
        "prelaunch_receipt_sha256": (
            prelaunch_receipt.get("receipt_sha256") if prelaunch_receipt is not None else None
        ),
        "prelaunch_receipt_file_sha256": prelaunch_receipt_file_sha256,
        "failure_reason": failure_reason,
        "formal_task_consumed": False,
        "formal_task_budget": 0,
        "task_bearing_calls": 0,
        "model_calls": 0,
        "training_calls": 0,
        "final_panel_access": 0,
        "scientific_state_mutated": False,
        "scientific_authority_granted_by_receipt": False,
        "observed_at_epoch": time.time(),
    }
    return {**projection, "receipt_sha256": canonical_sha256(projection)}


def canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def next_attempt(control_dir: Path) -> int:
    maximum = 0
    if control_dir.exists():
        for path in control_dir.iterdir():
            match = _ATTEMPT_RE.fullmatch(path.name)
            if match:
                maximum = max(maximum, int(match.group(1)))
    return maximum + 1


def payload_command(runtime_python: str, marker_path: str, expected_gpu_count: int) -> list[str]:
    if not runtime_python:
        raise PrelaunchError("runtime_python must be non-empty")
    if expected_gpu_count < 1:
        raise PrelaunchError("expected_gpu_count must be >= 1")
    return [
        runtime_python,
        "-c",
        PAYLOAD_PROGRAM,
        marker_path,
        str(expected_gpu_count),
    ]


def parse_payload_output(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise PrelaunchError("zero-formal payload emitted no structured result")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise PrelaunchError("zero-formal payload final line is not JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema") != PAYLOAD_SCHEMA:
        raise PrelaunchError("zero-formal payload schema drift")
    for key in ("formal_task_consumed", "scientific_state_mutated"):
        if payload.get(key) is not False:
            raise PrelaunchError(f"zero-formal payload violated {key}=false")
    for key in ("model_calls", "training_calls", "final_panel_access"):
        if payload.get(key) != 0:
            raise PrelaunchError(f"zero-formal payload violated {key}=0")
    return payload


def build_prelaunch_receipt(
    *,
    run_id: str,
    attempt: int,
    selected_gpu_uuids: Sequence[str],
    image_id: str,
    command_identity: str,
    return_code: int,
    wall_seconds: float,
    payload: Mapping[str, Any] | None,
    payload_error: str | None,
    stderr_tail: str,
    runtime_assets: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    passed = (
        return_code == 0
        and payload is not None
        and payload.get("status") == "PASS"
        and int(payload.get("observed_gpu_count", -1)) == len(selected_gpu_uuids)
    )
    projection: dict[str, Any] = {
        "schema": PRELAUNCH_SCHEMA,
        "run_id": run_id,
        "attempt": attempt,
        "status": "PASS" if passed else "FAIL",
        "failure_class": None if passed else "PRELAUNCH_ENGINEERING",
        "failure_stage": None if passed else (
            payload.get("stage") if payload is not None else "DOCKER_OR_PAYLOAD_START"
        ),
        "selected_gpu_uuids": list(selected_gpu_uuids),
        "image_id": image_id,
        "command_identity": command_identity,
        "runtime_assets": [dict(item) for item in runtime_assets],
        "return_code": int(return_code),
        "wall_seconds": float(wall_seconds),
        "gpu_hours": float(wall_seconds) * len(selected_gpu_uuids) / 3600.0,
        "payload": dict(payload) if payload is not None else None,
        "payload_error": payload_error,
        "stderr_tail": stderr_tail[-4000:],
        "formal_task_consumed": False,
        "scientific_budget_consumed": 0,
        "model_calls": 0,
        "training_calls": 0,
        "final_panel_access": 0,
        "scientific_state_mutated": False,
        "scientific_authority_granted_by_receipt": False,
        "observed_at_epoch": time.time(),
    }
    return {**projection, "receipt_sha256": canonical_sha256(projection)}


def validate_prelaunch_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_gpu_uuids: Sequence[str] | None = None,
) -> None:
    if receipt.get("schema") != PRELAUNCH_SCHEMA:
        raise PrelaunchError("prelaunch receipt schema drift")
    projection = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != canonical_sha256(projection):
        raise PrelaunchError("prelaunch receipt identity mismatch")
    if receipt.get("status") != "PASS":
        raise PrelaunchError(
            f"selected-GPU zero-formal preflight failed: "
            f"{receipt.get('failure_stage') or 'unknown-stage'}"
        )
    if expected_gpu_uuids is not None and list(receipt.get("selected_gpu_uuids") or []) != list(expected_gpu_uuids):
        raise PrelaunchError("prelaunch selected GPU identity drift")
    if receipt.get("formal_task_consumed") is not False or receipt.get("scientific_budget_consumed") != 0:
        raise PrelaunchError("prelaunch receipt is not zero-formal")
    for key in ("model_calls", "training_calls", "final_panel_access"):
        if receipt.get(key) != 0:
            raise PrelaunchError(f"prelaunch receipt violated {key}=0")
    if receipt.get("scientific_state_mutated") is not False:
        raise PrelaunchError("prelaunch receipt mutated scientific state")


def build_formal_start_receipt(
    *,
    run_id: str,
    campaign_id: str,
    selected_gpu_uuids: Sequence[str],
    image_id: str,
    command_identity: str,
    prelaunch_receipt_path: Path,
    readiness: Mapping[str, Any],
    output_root: str,
    runtime_assets: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    prelaunch = json.loads(prelaunch_receipt_path.read_text(encoding="utf-8"))
    validate_prelaunch_receipt(prelaunch, expected_gpu_uuids=selected_gpu_uuids)
    expected_assets = [dict(item) for item in runtime_assets]
    if prelaunch.get("runtime_assets", []) != expected_assets:
        raise PrelaunchError("Formal Start runtime asset identity differs from zero-formal preflight")
    projection: dict[str, Any] = {
        "schema": FORMAL_START_SCHEMA,
        "status": "FORMAL_START_COMMITTED",
        "run_id": run_id,
        "campaign_id": campaign_id,
        "selected_gpu_uuids": list(selected_gpu_uuids),
        "image_id": image_id,
        "command_identity": command_identity,
        "runtime_assets": expected_assets,
        "prelaunch_receipt": {
            "path": str(prelaunch_receipt_path),
            "sha256": sha256_file(prelaunch_receipt_path),
            "receipt_sha256": prelaunch["receipt_sha256"],
        },
        "readiness": dict(readiness),
        "output_root": output_root,
        "formal_task_consumption_before": False,
        "formal_task_consumption_may_begin": True,
        "formal_start_committed": True,
        "scientific_authority_granted_by_receipt": False,
        "committed_at_epoch": time.time(),
    }
    return {**projection, "receipt_sha256": canonical_sha256(projection)}
