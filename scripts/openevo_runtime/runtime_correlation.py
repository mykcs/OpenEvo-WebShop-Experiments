"""OpenTelemetry-compatible correlation metadata for OpenEVO runtime events.

Correlation is observational only. It can connect logs/spans/events to exact OpenEVO
identities, but it cannot become operation identity, scientific authority, or retry
permission.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from scripts.openevo_webshop.experiment_passport import (
    PassportError,
    assert_no_credentials,
    canonical_bytes,
    validate_schema,
)

ROOT = Path(__file__).resolve().parents[2]
REV = "202609171101"
SCHEMA_ID = f"openevo.runtime-correlation.{REV}"
SCHEMA_PATH = ROOT / f"schemas/runtime-correlation-{REV}.schema.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ARM = re.compile(r"^[A-Za-z0-9_.-]+$")


class RuntimeCorrelationError(ValueError):
    """Correlation metadata drifted from its bound OpenEVO identities."""


def _digest(value: Mapping[str, Any]) -> str:
    assert_no_credentials(value)
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _schema() -> dict[str, Any]:
    try:
        value = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeCorrelationError("runtime-correlation schema is unreadable") from exc
    if not isinstance(value, dict):
        raise RuntimeCorrelationError("runtime-correlation schema must be an object")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise RuntimeCorrelationError(f"{label} must be a SHA-256 identity")
    return value


def to_otel_attributes(payload: Mapping[str, Any]) -> dict[str, str | int]:
    attrs: dict[str, str | int] = {
        "openevo.generation_id": str(payload["generation_id"]),
        "openevo.operation_id": str(payload["operation_id"]),
        "openevo.arm": str(payload["arm"]),
        "openevo.round": int(payload["round_index"]),
        "openevo.attempt": int(payload["attempt_index"]),
        "openevo.event_id": str(payload["event_id"]),
    }
    lease_id = payload.get("lease_id")
    if lease_id is not None:
        attrs["openevo.lease_id"] = str(lease_id)
    return attrs


def correlation_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "schema",
        "event_id",
        "generation_id",
        "operation_id",
        "arm",
        "round_index",
        "attempt_index",
        "lease_id",
        "otel_attributes",
        "telemetry_authority",
        "operation_identity_authority",
        "scientific_authority",
        "scientific_retry_authorized",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeCorrelationError(f"missing runtime-correlation field: {missing[0]}")
    if payload["schema"] != SCHEMA_ID:
        raise RuntimeCorrelationError(f"schema must equal {SCHEMA_ID}")
    for key in ("event_id", "generation_id", "operation_id"):
        _sha(payload[key], key)
    lease = payload["lease_id"]
    if lease is not None:
        _sha(lease, "lease_id")
    arm = payload["arm"]
    if not isinstance(arm, str) or not ARM.fullmatch(arm):
        raise RuntimeCorrelationError("arm must be a non-empty token")
    for key in ("round_index", "attempt_index"):
        value = payload[key]
        if type(value) is not int or value < 0:
            raise RuntimeCorrelationError(f"{key} must be a non-negative integer")
    expected_attrs = to_otel_attributes(payload)
    if payload["otel_attributes"] != expected_attrs:
        raise RuntimeCorrelationError("otel_attributes drift from bound correlation fields")
    for key in (
        "telemetry_authority",
        "operation_identity_authority",
        "scientific_authority",
        "scientific_retry_authorized",
    ):
        if payload[key] is not False:
            raise RuntimeCorrelationError(f"correlation metadata cannot grant {key}")
    projection = {key: payload[key] for key in required}
    assert_no_credentials(projection)
    canonical_bytes(projection)
    return projection


def build_runtime_correlation(
    *,
    event_id: str,
    generation_id: str,
    operation_id: str,
    arm: str,
    round_index: int,
    attempt_index: int,
    lease_id: str | None,
) -> dict[str, Any]:
    payload = {
        "schema": SCHEMA_ID,
        "event_id": event_id,
        "generation_id": generation_id,
        "operation_id": operation_id,
        "arm": arm,
        "round_index": round_index,
        "attempt_index": attempt_index,
        "lease_id": lease_id,
        "otel_attributes": {},
        "telemetry_authority": False,
        "operation_identity_authority": False,
        "scientific_authority": False,
        "scientific_retry_authorized": False,
    }
    payload["otel_attributes"] = to_otel_attributes(payload)
    projection = correlation_projection(payload)
    result = {**payload, "correlation_id": _digest(projection)}
    validate_runtime_correlation(result)
    return result


def validate_runtime_correlation(payload: Mapping[str, Any]) -> None:
    try:
        validate_schema(dict(payload), _schema())
    except PassportError as exc:
        raise RuntimeCorrelationError(str(exc)) from exc
    projection = correlation_projection(payload)
    if payload.get("correlation_id") != _digest(projection):
        raise RuntimeCorrelationError("correlation_id does not match canonical projection")
