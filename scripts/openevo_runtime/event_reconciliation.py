"""Reconcile lossy/duplicate workflow delivery from immutable OpenEVO receipts.

The receipt projection is authoritative.  Framework event history is only a delivery
cache: missing events can be rebuilt and byte-identical duplicates can be ignored.
Conflicting deliveries fail closed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from scripts.openevo_runtime.runtime_events import (
    build_runtime_event,
    validate_runtime_event,
)


class EventReconciliationError(ValueError):
    """Delivered events cannot be reconciled with immutable receipt truth."""


@dataclass(frozen=True)
class EventReconciliation:
    events: tuple[Mapping[str, Any], ...]
    reconstructed_missing: int
    ignored_duplicates: int
    authority_source: str = "IMMUTABLE_OPEN_EVO_RECEIPTS"
    framework_history_authority: bool = False


def _logical_key(event: Mapping[str, Any]) -> tuple[str, str]:
    return str(event["event_type"]), str(event["subject_id"])


def _event_from_receipt_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "event_type",
        "subject_id",
        "source_receipt_identity",
        "payload_identity",
    )
    missing = [key for key in required if key not in row]
    if missing:
        raise EventReconciliationError(f"receipt projection missing {missing[0]}")
    return build_runtime_event(
        event_type=str(row["event_type"]),
        subject_id=str(row["subject_id"]),
        source_receipt_identity=str(row["source_receipt_identity"]),
        payload_identity=str(row["payload_identity"]),
    )


def reconcile_runtime_events(
    receipt_projections: Sequence[Mapping[str, Any]],
    delivered_events: Sequence[Mapping[str, Any]],
) -> EventReconciliation:
    """Return exactly one canonical event per immutable receipt projection."""
    expected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in receipt_projections:
        event = _event_from_receipt_projection(row)
        key = _logical_key(event)
        if key in expected and expected[key]["event_id"] != event["event_id"]:
            raise EventReconciliationError("receipt truth contains conflicting logical events")
        expected[key] = event
    seen: dict[tuple[str, str], Mapping[str, Any]] = {}
    duplicates = 0
    for raw in delivered_events:
        validate_runtime_event(raw)
        key = _logical_key(raw)
        if key not in expected:
            raise EventReconciliationError("framework delivered an event with no immutable receipt")
        if raw["event_id"] != expected[key]["event_id"]:
            raise EventReconciliationError("framework event conflicts with immutable receipt projection")
        prior = seen.get(key)
        if prior is not None:
            if prior["event_id"] != raw["event_id"]:
                raise EventReconciliationError("duplicate logical event has conflicting identity")
            duplicates += 1
            continue
        seen[key] = raw

    reconstructed = 0
    canonical: list[Mapping[str, Any]] = []
    for key in sorted(expected):
        event = seen.get(key)
        if event is None:
            event = expected[key]
            reconstructed += 1
        canonical.append(event)
    return EventReconciliation(
        events=tuple(canonical),
        reconstructed_missing=reconstructed,
        ignored_duplicates=duplicates,
    )
