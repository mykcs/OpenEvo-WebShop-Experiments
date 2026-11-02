import pytest

from scripts.openevo_runtime.runtime_events import (
    EVENT_SUBJECTS,
    RuntimeEventError,
    build_runtime_event,
    validate_runtime_event,
)


def _event(event_type: str):
    return build_runtime_event(
        event_type=event_type,
        subject_id="1" * 64,
        source_receipt_identity="2" * 64,
        payload_identity="3" * 64,
    )


def test_all_required_transition_families_have_framework_neutral_events() -> None:
    assert set(EVENT_SUBJECTS) == {
        "OPERATION_READY",
        "RESOURCE_LEASED",
        "SIDE_EFFECT_MATERIALIZED",
        "ROUND_SEALED",
        "INTEGRITY_BLOCKED",
        "PAIR_AUDIT",
    }
    for event_type, subject_kind in EVENT_SUBJECTS.items():
        event = _event(event_type)
        validate_runtime_event(event)
        assert event["subject_kind"] == subject_kind


def test_event_identity_is_content_addressed_and_deterministic() -> None:
    first = _event("OPERATION_READY")
    second = _event("OPERATION_READY")
    assert first == second
    changed = build_runtime_event(
        event_type="OPERATION_READY",
        subject_id="1" * 64,
        source_receipt_identity="2" * 64,
        payload_identity="4" * 64,
    )
    assert first["event_id"] != changed["event_id"]


def test_event_subject_type_mismatch_fails_closed() -> None:
    event = _event("RESOURCE_LEASED")
    event["subject_kind"] = "OPERATION_INTENT"
    with pytest.raises(RuntimeEventError, match="subject_kind"):
        validate_runtime_event(event)


def test_event_cannot_grant_scientific_or_retry_authority() -> None:
    event = _event("INTEGRITY_BLOCKED")
    event["scientific_retry_authorized"] = True
    with pytest.raises(RuntimeEventError):
        validate_runtime_event(event)


def test_unknown_event_type_fails_closed() -> None:
    with pytest.raises(RuntimeEventError, match="unsupported event_type"):
        build_runtime_event(
            event_type="FRAMEWORK_RESUME",
            subject_id="1" * 64,
            source_receipt_identity="2" * 64,
            payload_identity="3" * 64,
        )
