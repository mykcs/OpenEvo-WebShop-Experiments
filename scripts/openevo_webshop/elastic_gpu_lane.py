#!/usr/bin/env python3
"""Prospective elastic GPU binding contract for OpenEvo experiment runtimes.

This module separates *protocol identity* from *execution placement*.
Existing configurations remain exact-device by default.  A future runtime may
opt into QUALIFIED_RESOURCE_CLASS to select any live-idle subset of a frozen,
pre-qualified candidate pool while recording exact indices/UUIDs in every
selection and launch receipt.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

EXACT_DEVICE_SET = "EXACT_DEVICE_SET"
QUALIFIED_RESOURCE_CLASS = "QUALIFIED_RESOURCE_CLASS"
MODES = {EXACT_DEVICE_SET, QUALIFIED_RESOURCE_CLASS}
RAY_CAPACITY_SHARED_POOL = "RAY_CAPACITY_SHARED_POOL_" + "V" + "1"
IDLE_GATE_SHARED_POOL = "IDLE_GATE_SHARED_POOL_" + "V" + "1"


def queue_gate_freshness(
    resource_mode: str,
    observed_at_epoch: float,
    now_epoch: float,
    *,
    max_idle_snapshot_age_seconds: float = 60.0,
) -> dict:
    """Separate queue age from launch freshness.

    Ray-capacity queues revalidate physical identity/headroom during selection and
    again immediately before launch, so an old queue timestamp is audit metadata
    rather than launch authority. A legacy idle-snapshot gate has no later
    reservation/launch revalidation and therefore keeps the bounded age check.
    """
    observed = float(observed_at_epoch)
    now = float(now_epoch)
    if observed <= 0:
        raise RuntimeError("queue gate observed_at_epoch must be positive")
    age = max(0.0, now - observed)
    if resource_mode == RAY_CAPACITY_SHARED_POOL:
        return {
            "schema": "openevo.elastic-gpu-queue-freshness.202609181020",
            "status": "DEFER_TO_SELECTION_AND_LAUNCH_LIVE_GATE",
            "resource_mode": resource_mode,
            "queue_age_seconds": age,
            "queue_age_blocks_launch": False,
            "launch_freshness_authority": "RAY_SELECTION_AND_LAUNCH_LIVE_GATE",
        }
    if resource_mode == IDLE_GATE_SHARED_POOL:
        limit = float(max_idle_snapshot_age_seconds)
        if limit <= 0:
            raise RuntimeError("idle-snapshot max age must be positive")
        if age > limit:
            raise RuntimeError(
                f"idle-snapshot GPU gate is stale: age={age:.3f}s limit={limit:.3f}s"
            )
        return {
            "schema": "openevo.elastic-gpu-queue-freshness.202609181020",
            "status": "PASS",
            "resource_mode": resource_mode,
            "queue_age_seconds": age,
            "queue_age_blocks_launch": True,
            "max_idle_snapshot_age_seconds": limit,
            "launch_freshness_authority": "QUEUE_IDLE_SNAPSHOT",
        }
    raise RuntimeError(f"unknown shared-pool resource mode: {resource_mode}")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def binding_contract(cfg: dict) -> dict:
    """Return and validate the GPU binding contract.

    Backward compatibility is deliberate: configs without gpu_binding_contract
    retain the legacy exact-device meaning of managed_gpus + managed_gpu_uuids.
    """
    managed = [int(x) for x in cfg.get("managed_gpus") or []]
    uuids = {str(k): str(v) for k, v in (cfg.get("managed_gpu_uuids") or {}).items()}
    raw = cfg.get("gpu_binding_contract")
    if raw is None:
        return {
            "schema": "openevo.elastic-gpu-binding.202609181020",
            "mode": EXACT_DEVICE_SET,
            "candidate_gpu_indices": managed,
            "candidate_gpu_uuids": {str(i): uuids.get(str(i)) for i in managed},
            "resource_class": None,
            "paper_final_requires_exact_device_set": True,
            "selection_receipt_role": "scientific_identity_and_execution_evidence",
            "legacy_default": True,
        }

    c = dict(raw)
    if c.get("schema") != "openevo.elastic-gpu-binding.202609181020":
        raise RuntimeError("elastic GPU binding schema drift")
    mode = c.get("mode")
    if mode not in MODES:
        raise RuntimeError(f"unknown elastic GPU binding mode: {mode}")
    candidates = [int(x) for x in c.get("candidate_gpu_indices") or []]
    if len(candidates) != len(set(candidates)) or not candidates:
        raise RuntimeError("elastic GPU candidate set must be non-empty and unique")
    if candidates != managed:
        raise RuntimeError("elastic GPU candidate set must equal Ray managed_gpus")
    candidate_uuids = {str(k): str(v) for k, v in (c.get("candidate_gpu_uuids") or {}).items()}
    if set(candidate_uuids) != {str(i) for i in candidates}:
        raise RuntimeError("elastic GPU candidate UUID map must cover every candidate exactly")
    if candidate_uuids != {str(i): uuids.get(str(i)) for i in candidates}:
        raise RuntimeError("elastic GPU candidate UUID map disagrees with managed_gpu_uuids")
    if not all(candidate_uuids.values()):
        raise RuntimeError("elastic GPU candidate UUID is missing")

    resource_class = c.get("resource_class")
    if mode == QUALIFIED_RESOURCE_CLASS:
        if not isinstance(resource_class, dict):
            raise RuntimeError("qualified resource-class mode requires resource_class")
        if not str(resource_class.get("gpu_name") or "").strip():
            raise RuntimeError("qualified resource-class mode requires gpu_name")
        if int(resource_class.get("min_total_mib") or 0) <= 0:
            raise RuntimeError("qualified resource-class mode requires min_total_mib > 0")
        if c.get("selection_receipt_role") != "execution_evidence_not_protocol_identity":
            raise RuntimeError("qualified resource-class UUIDs must be execution evidence, not protocol identity")
        if c.get("paper_final_requires_exact_device_set") is not True:
            raise RuntimeError("qualified resource-class mode must preserve exact-device paper-final boundary")
    else:
        if c.get("selection_receipt_role") not in (None, "scientific_identity_and_execution_evidence"):
            raise RuntimeError("exact-device mode receipt role drift")
        c.setdefault("selection_receipt_role", "scientific_identity_and_execution_evidence")
        c.setdefault("paper_final_requires_exact_device_set", True)

    c["candidate_gpu_indices"] = candidates
    c["candidate_gpu_uuids"] = candidate_uuids
    c["legacy_default"] = False
    return c


def row_qualifies(contract: dict, index: int, row: dict) -> tuple[bool, str]:
    expected_uuid = contract["candidate_gpu_uuids"].get(str(index))
    if not expected_uuid or row.get("uuid") != expected_uuid:
        return False, "uuid_mismatch"
    if contract["mode"] == QUALIFIED_RESOURCE_CLASS:
        rc = contract["resource_class"]
        if row.get("name") != rc["gpu_name"]:
            return False, "gpu_class_mismatch"
        if int(row.get("total_mib", -1)) < int(rc["min_total_mib"]):
            return False, "gpu_memory_class_mismatch"
    return True, "qualified"


def binding_evidence(cfg: dict, selected: list[int], rows: dict[int, dict], checked_at_epoch: float) -> dict:
    contract = binding_contract(cfg)
    selected = [int(x) for x in selected]
    bad = [i for i in selected if i not in contract["candidate_gpu_indices"]]
    if bad:
        raise RuntimeError(f"selected GPU outside elastic candidate pool: {bad}")
    evidence_rows = []
    for i in selected:
        row = rows.get(i)
        if row is None:
            raise RuntimeError(f"selected GPU{i} missing from live rows")
        ok, reason = row_qualifies(contract, i, row)
        if not ok:
            raise RuntimeError(f"GPU{i} does not satisfy binding contract: {reason}")
        evidence_rows.append({
            "index": i,
            "uuid": row["uuid"],
            "name": row.get("name"),
            "total_mib": row.get("total_mib"),
        })
    public_contract = {k: v for k, v in contract.items() if k != "legacy_default"}
    return {
        "schema": "openevo.elastic-gpu-binding-evidence.202609181020",
        "mode": contract["mode"],
        "binding_contract_sha256": canonical_sha256(public_contract),
        "candidate_gpu_indices": list(contract["candidate_gpu_indices"]),
        "selected_devices": evidence_rows,
        "resource_class": contract.get("resource_class"),
        "selection_receipt_role": contract["selection_receipt_role"],
        "paper_final_requires_exact_device_set": bool(contract["paper_final_requires_exact_device_set"]),
        "checked_at_epoch": float(checked_at_epoch),
    }


def hardware_exposure_audit(receipts: list[dict], arm_field: str = "arm") -> dict:
    """Describe per-arm device exposure without turning it into a progression gate."""
    counts: dict[str, Counter] = {}
    rows = 0
    for receipt in receipts:
        selection = receipt.get("selection") if isinstance(receipt.get("selection"), dict) else receipt
        arm = str(receipt.get(arm_field) or selection.get(arm_field) or "")
        uuids = selection.get("selected_gpu_uuids") or [
            d.get("uuid") for d in (selection.get("gpu_binding") or {}).get("selected_devices", []) if d.get("uuid")
        ]
        if not arm or not uuids:
            continue
        rows += 1
        bucket = counts.setdefault(arm, Counter())
        bucket.update(map(str, uuids))
    arms = sorted(counts)
    all_uuids = sorted({u for c in counts.values() for u in c})
    max_delta = 0
    if len(arms) >= 2:
        for u in all_uuids:
            vals = [counts[a].get(u, 0) for a in arms]
            max_delta = max(max_delta, max(vals) - min(vals))
    return {
        "schema": "openevo.elastic-gpu-hardware-exposure-audit.202609181020",
        "status": "DESCRIPTIVE_ONLY",
        "receipt_count": rows,
        "arms": {a: dict(sorted(counts[a].items())) for a in arms},
        "max_per_uuid_count_delta_across_arms": max_delta,
        "blocks_progression": False,
    }


def build_qualification_cache(cfg: dict, rows: dict[int, dict], runtime_fingerprint: str, qualified_at_epoch: float) -> dict:
    contract = binding_contract(cfg)
    public_contract = {k: v for k, v in contract.items() if k != 'legacy_default'}
    devices = []
    failed = []
    for i in contract['candidate_gpu_indices']:
        row = rows.get(i)
        if row is None:
            devices.append({'index': i, 'status': 'FAIL', 'reason': 'missing_live_gpu'})
            failed.append(i)
            continue
        ok, reason = row_qualifies(contract, i, row)
        item = {
            'index': i,
            'uuid': row.get('uuid'),
            'name': row.get('name'),
            'total_mib': row.get('total_mib'),
            'status': 'PASS' if ok else 'FAIL',
            'reason': reason,
        }
        devices.append(item)
        if not ok:
            failed.append(i)
    return {
        'schema': 'openevo.elastic-gpu-qualification-cache.202609181020',
        'status': 'PASS' if not failed else 'FAIL',
        'binding_contract_sha256': canonical_sha256(public_contract),
        'runtime_fingerprint': str(runtime_fingerprint),
        'qualified_at_epoch': float(qualified_at_epoch),
        'devices': devices,
        'formal_task_consumption': 0,
        'model_calls': 0,
        'teacher_calls': 0,
        'training_calls': 0,
    }


def validate_qualification_cache(cfg: dict, cache: dict, runtime_fingerprint: str) -> None:
    contract = binding_contract(cfg)
    public_contract = {k: v for k, v in contract.items() if k != 'legacy_default'}
    if cache.get('schema') != 'openevo.elastic-gpu-qualification-cache.202609181020':
        raise RuntimeError('elastic GPU qualification-cache schema drift')
    if cache.get('status') != 'PASS':
        raise RuntimeError('elastic GPU qualification cache is not PASS')
    if cache.get('binding_contract_sha256') != canonical_sha256(public_contract):
        raise RuntimeError('elastic GPU qualification-cache binding drift')
    if cache.get('runtime_fingerprint') != str(runtime_fingerprint):
        raise RuntimeError('elastic GPU qualification-cache runtime drift')
    for field in ('formal_task_consumption', 'model_calls', 'teacher_calls', 'training_calls'):
        if int(cache.get(field, -1)) != 0:
            raise RuntimeError(f'elastic GPU qualification cache is not zero-formal: {field}')
    passed = {
        int(row['index']): row
        for row in cache.get('devices') or []
        if row.get('status') == 'PASS'
    }
    if set(passed) != set(contract['candidate_gpu_indices']):
        raise RuntimeError('elastic GPU qualification cache does not cover candidate pool')
