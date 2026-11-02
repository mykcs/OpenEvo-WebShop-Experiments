import pytest

from scripts.openevo_runtime.runtime_correlation import (
    RuntimeCorrelationError,
    build_runtime_correlation,
    to_otel_attributes,
    validate_runtime_correlation,
)


def _correlation(*, attempt: int = 0, lease: str | None = "4" * 64):
    return build_runtime_correlation(
        event_id="1" * 64,
        generation_id="2" * 64,
        operation_id="3" * 64,
        arm="BOUNDED_OFF",
        round_index=41,
        attempt_index=attempt,
        lease_id=lease,
    )


def test_otel_attributes_cover_required_execution_correlation_fields() -> None:
    correlation = _correlation()
    attrs = to_otel_attributes(correlation)
    assert attrs["openevo.generation_id"] == "2" * 64
    assert attrs["openevo.operation_id"] == "3" * 64
    assert attrs["openevo.arm"] == "BOUNDED_OFF"
    assert attrs["openevo.round"] == 41
    assert attrs["openevo.attempt"] == 0
    assert attrs["openevo.lease_id"] == "4" * 64
    assert attrs["openevo.event_id"] == "1" * 64


def test_correlation_identity_changes_with_attempt() -> None:
    first = _correlation(attempt=0)
    second = _correlation(attempt=1)
    assert first["correlation_id"] != second["correlation_id"]


def test_no_lease_omits_only_otel_lease_attribute() -> None:
    correlation = _correlation(lease=None)
    assert correlation["lease_id"] is None
    assert "openevo.lease_id" not in correlation["otel_attributes"]
    validate_runtime_correlation(correlation)


def test_otel_attribute_drift_fails_closed() -> None:
    correlation = _correlation()
    correlation["otel_attributes"]["openevo.round"] = 42
    with pytest.raises(RuntimeCorrelationError, match="otel_attributes"):
        validate_runtime_correlation(correlation)


def test_telemetry_cannot_be_promoted_to_authority() -> None:
    correlation = _correlation()
    correlation["telemetry_authority"] = True
    with pytest.raises(RuntimeCorrelationError):
        validate_runtime_correlation(correlation)
