from __future__ import annotations

import pytest

from scripts.openevo_runtime.shadow_parity import (
    ShadowClass,
    compare_observed_status,
    compare_status_inventory,
    modernization_class,
    reference_class,
    typed_result_for_reference_status,
)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("ROUND_ADVANCED", ShadowClass.READY),
        ("WAITING_FOR_SHARED_GPU_POOL", ShadowClass.WAIT_RESOURCE),
        ("RECOVERED_COMPLETED_ROUND", ShadowClass.EXACT_RESUME),
        ("PASS_WITH_CONTENT_REJECTION", ShadowClass.CONTENT_NOOP),
        ("BLOCKED_INTEGRITY_OR_UNRESOLVED_HEALTH", ShadowClass.BLOCKED_INTEGRITY),
        ("ARM_SEALED", ShadowClass.SEALED),
    ],
)
def test_full_shadow_decision_vocabulary_round_trips_through_typed_control(status: str, expected: ShadowClass) -> None:
    typed = typed_result_for_reference_status(operation_id=f"test:{status}", status=status)
    assert typed is not None
    assert reference_class(status) is expected
    assert modernization_class(typed) is expected


def test_unknown_reference_status_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown predecessor shadow status"):
        reference_class("MYSTERY_STATUS")


def test_pair_audit_history_never_becomes_operation_authority() -> None:
    assert reference_class("OBSERVE_ONLY") is ShadowClass.AUDIT_ONLY
    assert typed_result_for_reference_status(operation_id="pair-audit", status="OBSERVE_ONLY") is None
    assert compare_observed_status(source="watcher", status="OBSERVE_ONLY", count=3) is None


def test_inventory_compares_every_observed_operation_transition() -> None:
    inventory = {
        "off": {"ROUND_ADVANCED": 2, "WAITING_FOR_SHARED_GPU_POOL": 5},
        "scheduler": {"ACTIVE": 3, "DISPATCHED": 1},
        "watcher": {"OBSERVE_ONLY": 4},
    }
    result = compare_status_inventory(inventory)
    assert result["observed_transition_count"] == 15
    assert result["compared_operation_transition_count"] == 11
    assert result["audit_only_transition_count"] == 4
    assert result["unexplained_divergence_count"] == 0
    assert result["parity"] is True
    assert result["live_mutation"] is False
    assert result["scientific_retry_authorized"] is False
    assert result["scientific_authority_granted"] is False
    assert result["framework_history_is_scientific_identity"] is False


def test_inventory_rejects_nonpositive_counts() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        compare_status_inventory({"off": {"ROUND_ADVANCED": 0}})
