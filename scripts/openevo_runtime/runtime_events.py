"""Framework-neutral durable event records for Execution Runtime modernization.

Events are projections of already-authoritative OpenEVO receipts. They are transport
records only: event delivery, duplication, or framework history never grants science,
retry, reuse, or resource authority.
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
REV = "202609171058"
SCHEMA_ID = f"openevo.runtime-event.{REV}"
SCHEMA_PATH = ROOT / f"schemas/runtime-event-{REV}.schema.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
EVENT_SUBJECTS = {
    "OPERATION_READY": "OPERATION_INTENT",
    "RESOURCE_LEASED": "RESOURCE_LEASE",
    "SIDE_EFFECT_MATERIALIZED": "OPERATION_TERMINAL",
    "ROUND_SEALED": "ROUND_RECEIPT",
    "INTEGRITY_BLOCKED": "OPERATION_RESULT",
    "PAIR_AUDIT": "PAIR_RECEIPT",
}


class RuntimeEventError(ValueError):
    """A runtime event violates the transport-only event contract."""


def _digest(value: Mapping[str, Any]) -> str:
    assert_no_credentials(value)
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _schema() -> dict[str, Any]:
    try:
        value = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeEventError("runtime-event schema is unreadable") from exc
    if not isinstance(value, dict):
        raise RuntimeEventError("runtime-event schema must be an object")
    return value


def event_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "schema",
        "event_type",
        "subject_kind",
        "subject_id",
        "source_receipt_identity",
        "payload_identity",
        "immutable_receipt_projection",
        "scientific_authority",
        "scientific_retry_authorized",
        "resource_authority",
        "reuse_authority",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeEventError(f"missing runtime-event field: {missing[0]}")
    if payload["schema"] != SCHEMA_ID:
        raise RuntimeEventError(f"schema must equal {SCHEMA_ID}")
    event_type = payload["event_type"]
    expected_subject = EVENT_SUBJECTS.get(event_type)
    if expected_subject is None:
        raise RuntimeEventError("unsupported event_type")
    if payload["subject_kind"] != expected_subject:
        raise RuntimeEventError("subject_kind disagrees with event_type")
    for key in ("subject_id", "source_receipt_identity", "payload_identity"):
        value = payload[key]
        if not isinstance(value, str) or not HEX64.fullmatch(value):
            raise RuntimeEventError(f"{key} must be a SHA-256 identity")
    if payload["immutable_receipt_projection"] is not True:
        raise RuntimeEventError("runtime event must project an immutable receipt")
    for key in (
        "scientific_authority",
        "scientific_retry_authorized",
        "resource_authority",
        "reuse_authority",
    ):
        if payload[key] is not False:
            raise RuntimeEventError(f"runtime event cannot grant {key}")
    projection = {key: payload[key] for key in required}
    assert_no_credentials(projection)
    canonical_bytes(projection)
    return projection


def build_runtime_event(
    *,
    event_type: str,
    subject_id: str,
    source_receipt_identity: str,
    payload_identity: str,
) -> dict[str, Any]:
    subject_kind = EVENT_SUBJECTS.get(event_type)
    if subject_kind is None:
        raise RuntimeEventError("unsupported event_type")
    payload = {
        "schema": SCHEMA_ID,
        "event_type": event_type,
        "subject_kind": subject_kind,
        "subject_id": subject_id,
        "source_receipt_identity": source_receipt_identity,
        "payload_identity": payload_identity,
        "immutable_receipt_projection": True,
        "scientific_authority": False,
        "scientific_retry_authorized": False,
        "resource_authority": False,
        "reuse_authority": False,
    }
    projection = event_projection(payload)
    result = {**payload, "event_id": _digest(projection)}
    validate_runtime_event(result)
    return result


def validate_runtime_event(payload: Mapping[str, Any]) -> None:
    try:
        validate_schema(dict(payload), _schema())
    except PassportError as exc:
        raise RuntimeEventError(str(exc)) from exc
    projection = event_projection(payload)
    if payload.get("event_id") != _digest(projection):
        raise RuntimeEventError("event_id does not match canonical projection")
