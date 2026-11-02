"""Typed, framework-neutral resource records for Execution Runtime modernization.

These records describe resource demand and lease lifecycle only. They do not decide
scientific readiness, qualify a topology, or authorize retry/reuse/fallback.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.openevo_webshop.experiment_passport import (
    PassportError,
    assert_no_credentials,
    canonical_bytes,
    validate_schema,
)

ROOT = Path(__file__).resolve().parents[2]
REV = "202609170600"
ENVELOPE_SCHEMA_ID = f"openevo.resource-envelope.{REV}"
REQUEST_SCHEMA_ID = f"openevo.resource-request.{REV}"
LEASE_SCHEMA_ID = f"openevo.resource-lease.{REV}"
RELEASE_SCHEMA_ID = f"openevo.lease-release.{REV}"
BLOCKED_SCHEMA_ID = f"openevo.resource-blocked.{REV}"
SCHEMA_PATHS = {
    ENVELOPE_SCHEMA_ID: ROOT / f"schemas/resource-envelope-{REV}.schema.json",
    REQUEST_SCHEMA_ID: ROOT / f"schemas/resource-request-{REV}.schema.json",
    LEASE_SCHEMA_ID: ROOT / f"schemas/resource-lease-{REV}.schema.json",
    RELEASE_SCHEMA_ID: ROOT / f"schemas/lease-release-{REV}.schema.json",
    BLOCKED_SCHEMA_ID: ROOT / f"schemas/resource-blocked-{REV}.schema.json",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
GPU_UUID = re.compile(r"^GPU-[A-Za-z0-9-]+$")
PLACEMENT_MODES = frozenset({"GANG_ALL_OR_NOTHING", "QUALIFIED_SUBSET"})
BLOCK_CLASSES = frozenset({
    "CAPACITY_UNAVAILABLE",
    "HOST_CAPACITY_UNAVAILABLE",
    "UUID_DRIFT",
    "UNQUALIFIED_TOPOLOGY",
    "PLACEMENT_UNAVAILABLE",
    "LEASE_STATE_AMBIGUOUS",
})


class ResourceRecordError(ValueError):
    """A resource record violates the engineering-only runtime contract."""


def _schema(schema_id: str) -> dict[str, Any]:
    path = SCHEMA_PATHS[schema_id]
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResourceRecordError(f"resource schema is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise ResourceRecordError(f"resource schema must be an object: {path.name}")
    return value


def _validate_schema(payload: Mapping[str, Any], schema_id: str) -> None:
    try:
        validate_schema(dict(payload), _schema(schema_id))
    except PassportError as exc:
        raise ResourceRecordError(str(exc)) from exc


def _digest(payload: Mapping[str, Any]) -> str:
    assert_no_credentials(payload)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ResourceRecordError(f"{label} must be a SHA-256 identity")
    return value


def _nonnegative_int(value: Any, label: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        cmp = ">= 1" if positive else ">= 0"
        raise ResourceRecordError(f"{label} must be an integer {cmp}")
    return value


def resource_envelope_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = ("schema", "scope_model", "gpu_scope", "host_scope", "accounting")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ResourceRecordError(f"missing resource envelope field: {missing[0]}")
    if payload["schema"] != ENVELOPE_SCHEMA_ID:
        raise ResourceRecordError(f"schema must equal {ENVELOPE_SCHEMA_ID}")
    if payload["scope_model"] != "PER_GPU_PLUS_HOST_GLOBAL":
        raise ResourceRecordError("resource envelope scope_model drift")
    gpu = payload["gpu_scope"]
    host = payload["host_scope"]
    accounting = payload["accounting"]
    if not isinstance(gpu, Mapping) or not isinstance(host, Mapping) or not isinstance(accounting, Mapping):
        raise ResourceRecordError("resource envelope scopes/accounting must be objects")
    _nonnegative_int(gpu.get("gpu_count"), "gpu_count", positive=True)
    _nonnegative_int(gpu.get("vram_mib_per_gpu"), "vram_mib_per_gpu", positive=True)
    _nonnegative_int(gpu.get("process_slots_per_gpu"), "process_slots_per_gpu", positive=True)
    share = gpu.get("compute_share_advisory")
    if share is not None and (type(share) not in (int, float) or not 0 <= share <= 1):
        raise ResourceRecordError("compute_share_advisory must be null or between 0 and 1")
    _nonnegative_int(host.get("ram_mib"), "host_ram_mib")
    _nonnegative_int(host.get("cpu_slots"), "host_cpu_slots")
    if not isinstance(host.get("disk_io_class"), str) or not host["disk_io_class"]:
        raise ResourceRecordError("disk_io_class must be non-empty")
    if accounting.get("gpu_capacity_is_per_physical_gpu") is not True:
        raise ResourceRecordError("GPU capacity must be declared per physical GPU")
    if accounting.get("host_capacity_counted_once_per_host") is not True:
        raise ResourceRecordError("host-global capacity must be counted once per host")
    projection = {key: payload[key] for key in required}
    canonical_bytes(projection)
    return projection


def build_resource_envelope(
    *, gpu_count: int, vram_mib_per_gpu: int, process_slots_per_gpu: int,
    compute_share_advisory: float | None, host_ram_mib: int, cpu_slots: int,
    disk_io_class: str,
) -> dict[str, Any]:
    projection = {
        "schema": ENVELOPE_SCHEMA_ID,
        "scope_model": "PER_GPU_PLUS_HOST_GLOBAL",
        "gpu_scope": {
            "gpu_count": gpu_count,
            "vram_mib_per_gpu": vram_mib_per_gpu,
            "process_slots_per_gpu": process_slots_per_gpu,
            "compute_share_advisory": compute_share_advisory,
        },
        "host_scope": {
            "ram_mib": host_ram_mib,
            "cpu_slots": cpu_slots,
            "disk_io_class": disk_io_class,
        },
        "accounting": {
            "gpu_capacity_is_per_physical_gpu": True,
            "host_capacity_counted_once_per_host": True,
        },
    }
    projection = resource_envelope_projection(projection)
    result = {**projection, "envelope_id": _digest(projection)}
    validate_resource_envelope(result)
    return result


def validate_resource_envelope(payload: Mapping[str, Any]) -> None:
    _validate_schema(payload, ENVELOPE_SCHEMA_ID)
    projection = resource_envelope_projection(payload)
    if payload.get("envelope_id") != _digest(projection):
        raise ResourceRecordError("envelope_id does not match canonical projection")


def resource_request_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "schema", "operation_id", "execution_generation_id", "resource_policy_identity",
        "qualified_topology_identity", "envelope", "placement_mode", "preferred_gpu_indices",
        "fallback_topology_identity", "scientific_authority_granted",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ResourceRecordError(f"missing resource request field: {missing[0]}")
    if payload["schema"] != REQUEST_SCHEMA_ID:
        raise ResourceRecordError(f"schema must equal {REQUEST_SCHEMA_ID}")
    for key in ("operation_id", "execution_generation_id", "resource_policy_identity", "qualified_topology_identity"):
        _sha(payload[key], key)
    fallback = payload["fallback_topology_identity"]
    if fallback is not None:
        _sha(fallback, "fallback_topology_identity")
    if payload["scientific_authority_granted"] is not False:
        raise ResourceRecordError("resource request cannot grant scientific authority")
    validate_resource_envelope(payload["envelope"])
    if payload["placement_mode"] not in PLACEMENT_MODES:
        raise ResourceRecordError("unsupported placement_mode")
    preferred = payload["preferred_gpu_indices"]
    if not isinstance(preferred, list) or any(type(v) is not int or v < 0 for v in preferred):
        raise ResourceRecordError("preferred_gpu_indices must be non-negative integers")
    if len(preferred) != len(set(preferred)):
        raise ResourceRecordError("preferred_gpu_indices must be unique")
    if len(preferred) < payload["envelope"]["gpu_scope"]["gpu_count"]:
        raise ResourceRecordError("preferred_gpu_indices cannot satisfy requested gpu_count")
    projection = {key: payload[key] for key in required}
    canonical_bytes(projection)
    return projection


def build_resource_request(
    *, operation_id: str, execution_generation_id: str, resource_policy_identity: str,
    qualified_topology_identity: str, envelope: Mapping[str, Any], placement_mode: str,
    preferred_gpu_indices: Sequence[int], fallback_topology_identity: str | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": REQUEST_SCHEMA_ID,
        "operation_id": operation_id,
        "execution_generation_id": execution_generation_id,
        "resource_policy_identity": resource_policy_identity,
        "qualified_topology_identity": qualified_topology_identity,
        "envelope": dict(envelope),
        "placement_mode": placement_mode,
        "preferred_gpu_indices": list(preferred_gpu_indices),
        "fallback_topology_identity": fallback_topology_identity,
        "scientific_authority_granted": False,
    }
    projection = resource_request_projection(payload)
    result = {**payload, "request_id": _digest(projection)}
    validate_resource_request(result)
    return result


def validate_resource_request(payload: Mapping[str, Any]) -> None:
    _validate_schema(payload, REQUEST_SCHEMA_ID)
    projection = resource_request_projection(payload)
    if payload.get("request_id") != _digest(projection):
        raise ResourceRecordError("request_id does not match canonical projection")


def _selected_gpu_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ResourceRecordError("selected_gpus must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ResourceRecordError("selected_gpus rows must be objects")
        index = row.get("index")
        uuid = row.get("uuid")
        if type(index) is not int or index < 0:
            raise ResourceRecordError("selected GPU index must be non-negative integer")
        if not isinstance(uuid, str) or not GPU_UUID.fullmatch(uuid):
            raise ResourceRecordError("selected GPU UUID must use GPU-* identity")
        normalized.append({"index": index, "uuid": uuid})
    if len({r["index"] for r in normalized}) != len(normalized):
        raise ResourceRecordError("selected GPU indices must be unique")
    if len({r["uuid"] for r in normalized}) != len(normalized):
        raise ResourceRecordError("selected GPU UUIDs must be unique")
    return normalized


def resource_lease_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "schema", "request_id", "operation_id", "execution_generation_id",
        "qualified_topology_identity", "selected_gpus", "placement_group_name",
        "lease_receipt_identity", "scientific_authority_granted", "topology_authority_granted",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ResourceRecordError(f"missing resource lease field: {missing[0]}")
    if payload["schema"] != LEASE_SCHEMA_ID:
        raise ResourceRecordError(f"schema must equal {LEASE_SCHEMA_ID}")
    for key in ("request_id", "operation_id", "execution_generation_id", "qualified_topology_identity", "lease_receipt_identity"):
        _sha(payload[key], key)
    rows = _selected_gpu_rows(payload["selected_gpus"])
    name = payload["placement_group_name"]
    if name is not None and (not isinstance(name, str) or not name):
        raise ResourceRecordError("placement_group_name must be null or non-empty")
    if payload["scientific_authority_granted"] is not False or payload["topology_authority_granted"] is not False:
        raise ResourceRecordError("lease records cannot grant science/topology authority")
    projection = {key: payload[key] for key in required}
    projection["selected_gpus"] = rows
    canonical_bytes(projection)
    return projection


def build_resource_lease(
    request: Mapping[str, Any], *, selected_gpus: Sequence[Mapping[str, Any]],
    lease_receipt_identity: str, placement_group_name: str | None,
) -> dict[str, Any]:
    validate_resource_request(request)
    rows = _selected_gpu_rows(list(selected_gpus))
    expected_count = request["envelope"]["gpu_scope"]["gpu_count"]
    if len(rows) != expected_count:
        raise ResourceRecordError("selected GPU count must equal the frozen request envelope")
    if not {row["index"] for row in rows}.issubset(set(request["preferred_gpu_indices"])):
        raise ResourceRecordError("lease selected GPU outside frozen preferred set")
    payload = {
        "schema": LEASE_SCHEMA_ID,
        "request_id": request["request_id"],
        "operation_id": request["operation_id"],
        "execution_generation_id": request["execution_generation_id"],
        "qualified_topology_identity": request["qualified_topology_identity"],
        "selected_gpus": rows,
        "placement_group_name": placement_group_name,
        "lease_receipt_identity": lease_receipt_identity,
        "scientific_authority_granted": False,
        "topology_authority_granted": False,
    }
    projection = resource_lease_projection(payload)
    result = {**payload, "lease_id": _digest(projection)}
    validate_resource_lease(result)
    return result


def validate_resource_lease(payload: Mapping[str, Any]) -> None:
    _validate_schema(payload, LEASE_SCHEMA_ID)
    projection = resource_lease_projection(payload)
    if payload.get("lease_id") != _digest(projection):
        raise ResourceRecordError("lease_id does not match canonical projection")


def lease_release_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "schema", "lease_id", "request_id", "release_receipt_identity", "released_gpu_indices",
        "placement_group_released", "all_resources_released", "scientific_authority_granted",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ResourceRecordError(f"missing lease release field: {missing[0]}")
    if payload["schema"] != RELEASE_SCHEMA_ID:
        raise ResourceRecordError(f"schema must equal {RELEASE_SCHEMA_ID}")
    for key in ("lease_id", "request_id", "release_receipt_identity"):
        _sha(payload[key], key)
    indices = payload["released_gpu_indices"]
    if not isinstance(indices, list) or any(type(v) is not int or v < 0 for v in indices):
        raise ResourceRecordError("released_gpu_indices must be non-negative integers")
    if len(indices) != len(set(indices)):
        raise ResourceRecordError("released_gpu_indices must be unique")
    for key in ("placement_group_released", "all_resources_released"):
        if type(payload[key]) is not bool:
            raise ResourceRecordError(f"{key} must be boolean")
    if payload["scientific_authority_granted"] is not False:
        raise ResourceRecordError("release record cannot grant scientific authority")
    projection = {key: payload[key] for key in required}
    canonical_bytes(projection)
    return projection


def build_lease_release(
    lease: Mapping[str, Any], *, release_receipt_identity: str,
    released_gpu_indices: Sequence[int], placement_group_released: bool,
    all_resources_released: bool,
) -> dict[str, Any]:
    validate_resource_lease(lease)
    payload = {
        "schema": RELEASE_SCHEMA_ID,
        "lease_id": lease["lease_id"],
        "request_id": lease["request_id"],
        "release_receipt_identity": release_receipt_identity,
        "released_gpu_indices": list(released_gpu_indices),
        "placement_group_released": placement_group_released,
        "all_resources_released": all_resources_released,
        "scientific_authority_granted": False,
    }
    projection = lease_release_projection(payload)
    result = {**payload, "release_id": _digest(projection)}
    validate_lease_release(result)
    return result


def validate_lease_release(payload: Mapping[str, Any]) -> None:
    _validate_schema(payload, RELEASE_SCHEMA_ID)
    projection = lease_release_projection(payload)
    if payload.get("release_id") != _digest(projection):
        raise ResourceRecordError("release_id does not match canonical projection")


def resource_blocked_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "schema", "request_id", "operation_id", "block_class", "retryable_resource_wait",
        "scientific_state_mutated", "side_effect_materialized", "scientific_retry_authorized",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ResourceRecordError(f"missing resource blocked field: {missing[0]}")
    if payload["schema"] != BLOCKED_SCHEMA_ID:
        raise ResourceRecordError(f"schema must equal {BLOCKED_SCHEMA_ID}")
    _sha(payload["request_id"], "request_id")
    _sha(payload["operation_id"], "operation_id")
    if payload["block_class"] not in BLOCK_CLASSES:
        raise ResourceRecordError("unsupported resource block_class")
    retryable_expected = payload["block_class"] in {
        "CAPACITY_UNAVAILABLE", "HOST_CAPACITY_UNAVAILABLE", "PLACEMENT_UNAVAILABLE"
    }
    if payload["retryable_resource_wait"] is not retryable_expected:
        raise ResourceRecordError("retryable_resource_wait disagrees with block_class")
    if payload["scientific_state_mutated"] is not False or payload["side_effect_materialized"] is not False:
        raise ResourceRecordError("resource block requires proof of zero scientific side effect")
    if payload["scientific_retry_authorized"] is not False:
        raise ResourceRecordError("resource blocked record cannot authorize scientific retry")
    projection = {key: payload[key] for key in required}
    canonical_bytes(projection)
    return projection


def build_resource_blocked(request: Mapping[str, Any], *, block_class: str) -> dict[str, Any]:
    validate_resource_request(request)
    payload = {
        "schema": BLOCKED_SCHEMA_ID,
        "request_id": request["request_id"],
        "operation_id": request["operation_id"],
        "block_class": block_class,
        "retryable_resource_wait": block_class in {
            "CAPACITY_UNAVAILABLE", "HOST_CAPACITY_UNAVAILABLE", "PLACEMENT_UNAVAILABLE"
        },
        "scientific_state_mutated": False,
        "side_effect_materialized": False,
        "scientific_retry_authorized": False,
    }
    projection = resource_blocked_projection(payload)
    result = {**payload, "block_id": _digest(projection)}
    validate_resource_blocked(result)
    return result


def validate_resource_blocked(payload: Mapping[str, Any]) -> None:
    _validate_schema(payload, BLOCKED_SCHEMA_ID)
    projection = resource_blocked_projection(payload)
    if payload.get("block_id") != _digest(projection):
        raise ResourceRecordError("block_id does not match canonical projection")
