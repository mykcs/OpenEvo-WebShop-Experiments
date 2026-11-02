"""Runtime-neutral reliability records for OpenEVO experiment execution.

These records are observational/coordination artifacts.  They never grant
scientific authority, resource authority, retry authority, or reuse authority.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

LEASE_SCHEMA = "openevo.prelaunch-coordination-lease.202609281950"
LEASE_RELEASE_SCHEMA = "openevo.prelaunch-coordination-release.202609281950"
LEASE_HANDOFF_SCHEMA = "openevo.prelaunch-coordination-handoff.202609281950"
INCIDENT_SCHEMA = "openevo.engineering-incident.202609281950"
TIMELINE_SCHEMA = "openevo.experiment-lifecycle-telemetry.202609281950"

FAILURE_CATEGORIES = {
    "PRELAUNCH_ENGINEERING",
    "RESOURCE_ORCHESTRATION",
    "RUNTIME_INFRA",
    "INTERFACE_CONTRACT",
    "SCIENTIFIC",
}
EVENT_ORDER = ("T0", "T1", "T2", "T3", "T4", "T5")


class ReliabilityRecordError(RuntimeError):
    pass


def canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_projection(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in receipt.items() if key != "receipt_sha256"}
def build_prelaunch_lease(
    *,
    run_id: str,
    campaign_id: str,
    selected_gpu_uuids: Sequence[str],
    runtime_candidate_uuids: Sequence[str],
    readiness_receipt_sha256: str | None,
    ttl_seconds: int = 900,
    acquired_at_epoch: float | None = None,
) -> dict[str, Any]:
    if ttl_seconds < 1:
        raise ReliabilityRecordError("prelaunch coordination lease TTL must be positive")
    selected = list(selected_gpu_uuids)
    candidates = sorted(set(runtime_candidate_uuids))
    if not selected or len(selected) != len(set(selected)):
        raise ReliabilityRecordError("prelaunch coordination lease requires unique selected UUIDs")
    if not set(selected).issubset(set(candidates)):
        raise ReliabilityRecordError("selected GPU escaped runtime candidate set")
    acquired = float(acquired_at_epoch if acquired_at_epoch is not None else time.time())
    projection: dict[str, Any] = {
        "schema": LEASE_SCHEMA,
        "status": "ACTIVE",
        "mechanism": "KERNEL_FLOCK_COORDINATION_ONLY",
        "reservation_claimed": False,
        "run_id": run_id,
        "campaign_id": campaign_id,
        "selected_gpu_uuids": selected,
        "runtime_candidate_uuids": candidates,
        "controller_pid": os.getpid(),
        "readiness_receipt_sha256": readiness_receipt_sha256,
        "ttl_seconds": int(ttl_seconds),
        "acquired_at_epoch": acquired,
        "expires_at_epoch": acquired + int(ttl_seconds),
        "formal_task_consumed": False,
        "scientific_budget_consumed": 0,
        "scientific_authority_granted": False,
        "resource_authority_granted": False,
    }
    return {**projection, "receipt_sha256": canonical_sha256(projection)}


def validate_lease(
    receipt: Mapping[str, Any],
    *,
    expected_gpu_uuids: Sequence[str] | None = None,
) -> None:
    if receipt.get("schema") != LEASE_SCHEMA or receipt.get("status") != "ACTIVE":
        raise ReliabilityRecordError("prelaunch coordination lease is not ACTIVE")
    if receipt.get("receipt_sha256") != canonical_sha256(_identity_projection(receipt)):
        raise ReliabilityRecordError("prelaunch coordination lease identity mismatch")
    if receipt.get("reservation_claimed") is not False:
        raise ReliabilityRecordError("coordination lease must not claim GPU reservation")
    if expected_gpu_uuids is not None and list(receipt.get("selected_gpu_uuids") or []) != list(expected_gpu_uuids):
        raise ReliabilityRecordError("prelaunch coordination lease GPU identity drift")


def require_live_lease(
    receipt: Mapping[str, Any],
    *,
    now_epoch: float | None = None,
    expected_gpu_uuids: Sequence[str] | None = None,
) -> None:
    validate_lease(receipt, expected_gpu_uuids=expected_gpu_uuids)
    now = float(now_epoch if now_epoch is not None else time.time())
    if now >= float(receipt.get("expires_at_epoch", 0)):
        raise ReliabilityRecordError("prelaunch coordination lease expired before Formal Start")


def build_lease_release(
    lease: Mapping[str, Any],
    *,
    reason: str,
    released_at_epoch: float | None = None,
) -> dict[str, Any]:
    # Release remains legal after TTL expiry; expiry blocks Formal Start, not
    # cleanup of the project-owned coordination record.
    validate_lease(lease)
    projection = {
        "schema": LEASE_RELEASE_SCHEMA,
        "status": "RELEASED",
        "lease_receipt_sha256": lease["receipt_sha256"],
        "selected_gpu_uuids": list(lease["selected_gpu_uuids"]),
        "reason": str(reason),
        "released_at_epoch": float(
            released_at_epoch if released_at_epoch is not None else time.time()
        ),
        "formal_task_consumed": False,
        "scientific_authority_granted": False,
        "resource_authority_granted": False,
    }
    return {**projection, "receipt_sha256": canonical_sha256(projection)}


def build_lease_handoff(
    lease: Mapping[str, Any],
    *,
    formal_start_receipt_sha256: str,
    handed_off_at_epoch: float | None = None,
) -> dict[str, Any]:
    # Formal Start is the monotonic commit point. Once it has been written,
    # expiry must not turn that committed transition into a prelaunch release.
    # The caller supplies the Formal Start commit epoch so delayed bookkeeping
    # remains bound to that transition instead of checking the wall clock again.
    validate_lease(lease, expected_gpu_uuids=list(lease["selected_gpu_uuids"]))
    now = float(handed_off_at_epoch if handed_off_at_epoch is not None else time.time())
    if now < float(lease["acquired_at_epoch"]):
        raise ReliabilityRecordError("lease handoff predates lease acquisition")
    projection = {
        "schema": LEASE_HANDOFF_SCHEMA,
        "status": "FORMAL_HANDOFF",
        "lease_receipt_sha256": lease["receipt_sha256"],
        "formal_start_receipt_sha256": formal_start_receipt_sha256,
        "selected_gpu_uuids": list(lease["selected_gpu_uuids"]),
        "handed_off_at_epoch": now,
        "reservation_claimed": False,
        "scientific_authority_granted": False,
        "resource_authority_granted": False,
    }
    return {**projection, "receipt_sha256": canonical_sha256(projection)}
def failure_fingerprint(
    *,
    category: str,
    layer: str,
    code: str,
    validator_id: str,
    repair_class: str,
) -> str:
    if category not in FAILURE_CATEGORIES:
        raise ReliabilityRecordError(f"unknown engineering failure category: {category}")
    stable = {
        "category": category,
        "layer": str(layer),
        "code": str(code),
        "validator_id": str(validator_id),
        "repair_class": str(repair_class),
    }
    return canonical_sha256(stable)


def record_incident(
    root: Path,
    *,
    category: str,
    layer: str,
    code: str,
    validator_id: str,
    repair_class: str,
    detail: str,
    run_id: str,
    selected_gpu_uuids: Sequence[str] = (),
    observed_at_epoch: float | None = None,
) -> dict[str, Any]:
    fingerprint = failure_fingerprint(
        category=category,
        layer=layer,
        code=code,
        validator_id=validator_id,
        repair_class=repair_class,
    )
    bucket = root.resolve() / fingerprint.removeprefix("sha256:")
    bucket.mkdir(mode=0o700, parents=True, exist_ok=True)
    prior = []
    for path in sorted(bucket.glob("incident-*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            isinstance(value, dict)
            and value.get("schema") == INCIDENT_SCHEMA
            and value.get("fingerprint") == fingerprint
        ):
            prior.append(value)
    observed = float(observed_at_epoch if observed_at_epoch is not None else time.time())
    occurrence_index = len(prior) + 1
    projection: dict[str, Any] = {
        "schema": INCIDENT_SCHEMA,
        "fingerprint": fingerprint,
        "category": category,
        "layer": str(layer),
        "code": str(code),
        "validator_id": str(validator_id),
        "repair_class": str(repair_class),
        "detail": str(detail)[:4000],
        "run_id": run_id,
        "selected_gpu_uuids": list(selected_gpu_uuids),
        "occurrence_index": occurrence_index,
        "promotion_required": occurrence_index >= 2,
        "auto_repair_allowed": False,
        "observed_at_epoch": observed,
        "formal_task_consumed": False,
        "scientific_authority_granted": False,
        "retry_authority_granted": False,
    }
    receipt = {**projection, "receipt_sha256": canonical_sha256(projection)}
    path = bucket / f"incident-{time.time_ns()}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return {**receipt, "path": str(path)}
def _timeline_path(control_dir: Path) -> Path:
    return control_dir.resolve() / "lifecycle-telemetry.json"


def _empty_timeline(run_id: str, campaign_id: str) -> dict[str, Any]:
    return {
        "schema": TIMELINE_SCHEMA,
        "run_id": run_id,
        "campaign_id": campaign_id,
        "events": [],
        "summary": {key: None for key in EVENT_ORDER},
        "metrics": {},
        "observability_only": True,
        "scientific_authority_granted": False,
        "resource_authority_granted": False,
        "retry_authority_granted": False,
    }


def _derive_metrics(summary: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    def epoch(key: str) -> float | None:
        row = summary.get(key)
        return float(row["epoch"]) if isinstance(row, Mapping) and "epoch" in row else None

    t0, t3, t4, t5 = epoch("T0"), epoch("T3"), epoch("T4"), epoch("T5")
    if t0 is not None and t4 is not None and t4 >= t0:
        metrics["time_to_formal_start_seconds"] = t4 - t0
    if t0 is not None and t5 is not None and t5 >= t0:
        metrics["time_to_first_valid_step_seconds"] = t5 - t0
    if t3 is not None and t4 is not None and t4 >= t3:
        metrics["preflight_to_formal_start_seconds"] = t4 - t3
    return metrics


def record_lifecycle_event(
    control_dir: Path,
    *,
    run_id: str,
    campaign_id: str,
    key: str,
    epoch: float,
    meaning: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    if key not in EVENT_ORDER:
        raise ReliabilityRecordError(f"unknown lifecycle event key: {key}")
    path = _timeline_path(control_dir)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("schema") != TIMELINE_SCHEMA:
                raise ReliabilityRecordError("lifecycle telemetry schema drift")
            if value.get("run_id") != run_id or value.get("campaign_id") != campaign_id:
                raise ReliabilityRecordError("lifecycle telemetry identity drift")
        else:
            value = _empty_timeline(run_id, campaign_id)
        event_projection = {
            "key": key,
            "epoch": float(epoch),
            "meaning": str(meaning),
            "source": dict(source),
        }
        event = {**event_projection, "event_sha256": canonical_sha256(event_projection)}
        if not any(row.get("event_sha256") == event["event_sha256"] for row in value["events"]):
            value["events"].append(event)

        same_key = [row for row in value["events"] if row.get("key") == key]
        if same_key:
            chosen = min(same_key, key=lambda row: float(row["epoch"]))
            value["summary"][key] = chosen
        value["metrics"] = _derive_metrics(value["summary"])
        value["updated_at_epoch"] = time.time()
        projection = {key_: val for key_, val in value.items() if key_ != "receipt_sha256"}
        value["receipt_sha256"] = canonical_sha256(projection)

        serialized = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        fd, temporary_name = tempfile.mkstemp(
            prefix=path.name + ".", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        return {**value, "path": str(path)}
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def lifecycle_summary(control_dir: Path) -> dict[str, Any]:
    path = _timeline_path(control_dir)
    if not path.is_file():
        raise ReliabilityRecordError(f"lifecycle telemetry is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != TIMELINE_SCHEMA:
        raise ReliabilityRecordError("lifecycle telemetry schema drift")
    return {**value, "path": str(path)}

def incident_summary(root: Path) -> dict[str, Any]:
    resolved = root.resolve()
    rows: list[dict[str, Any]] = []
    if resolved.exists():
        for bucket in sorted(path for path in resolved.iterdir() if path.is_dir()):
            incidents: list[dict[str, Any]] = []
            for path in sorted(bucket.glob("incident-*.json")):
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(value, dict) and value.get("schema") == INCIDENT_SCHEMA:
                    incidents.append({**value, "path": str(path)})
            if not incidents:
                continue
            latest = max(incidents, key=lambda row: float(row.get("observed_at_epoch", 0)))
            rows.append(
                {
                    "fingerprint": latest.get("fingerprint"),
                    "category": latest.get("category"),
                    "layer": latest.get("layer"),
                    "code": latest.get("code"),
                    "validator_id": latest.get("validator_id"),
                    "repair_class": latest.get("repair_class"),
                    "occurrence_count": len(incidents),
                    "promotion_required": len(incidents) >= 2,
                    "latest_observed_at_epoch": latest.get("observed_at_epoch"),
                    "latest_path": latest.get("path"),
                }
            )
    return {
        "schema": "openevo.engineering-incident-summary.202609281950",
        "root": str(resolved),
        "fingerprint_count": len(rows),
        "repeated_fingerprint_count": sum(1 for row in rows if row["occurrence_count"] >= 2),
        "incidents": rows,
        "observability_only": True,
        "scientific_authority_granted": False,
    }

def classify_operation_result(
    payload: Mapping[str, Any],
    *,
    lifecycle_phase: str,
) -> str | None:
    """Project the existing typed OperationResult into the Dev failure taxonomy."""
    from scripts.openevo_runtime.result_codes import (
        OperationResult,
        RetryClass,
    )

    typed = OperationResult.from_mapping(payload)
    retry = typed.retry_class
    phase = str(lifecycle_phase).upper()
    if retry is RetryClass.WAIT_RESOURCE:
        return "RESOURCE_ORCHESTRATION"
    if retry is RetryClass.EXACT_RESUME:
        return "PRELAUNCH_ENGINEERING" if phase == "PRELAUNCH" else "RUNTIME_INFRA"
    if retry is RetryClass.CONTENT_REJECTION_NOOP:
        return "INTERFACE_CONTRACT"
    if retry in {RetryClass.INTEGRITY_UNKNOWN, RetryClass.AUTHORITY_DRIFT}:
        return "RUNTIME_INFRA"
    if retry is RetryClass.SCIENCE_DECISION_REQUIRED:
        return "SCIENTIFIC"
    if retry in {RetryClass.NONE, RetryClass.SEALED}:
        return None
    raise ReliabilityRecordError(f"unmapped retry class: {retry.value}")
