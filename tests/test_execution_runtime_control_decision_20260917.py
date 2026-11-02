from __future__ import annotations

import pytest

from scripts.openevo_runtime.control_decision import (
    ControlAction,
    scheduler_decision,
    supervisor_decision,
)
from scripts.openevo_runtime.result_codes import (
    OperationResult,
    OperationResultError,
    OperationStatus,
    ReasonCode,
)


def make_result(*, status: OperationStatus, reason: ReasonCode, message: str = "") -> OperationResult:
    return OperationResult(
        status=status,
        reason_code=reason,
        operation_id="prospective-op-1",
        scientific_state_mutated=False if reason is ReasonCode.GPU_RESOURCE_UNAVAILABLE else None,
        side_effect_materialized=False if reason is ReasonCode.GPU_RESOURCE_UNAVAILABLE else None,
        message=message,
    )

def test_typed_resource_wait_drives_control_without_message_marker() -> None:
    result = make_result(
        status=OperationStatus.WAITING,
        reason=ReasonCode.GPU_RESOURCE_UNAVAILABLE,
        message="capacity temporarily unavailable",
    )
    for decide in (supervisor_decision, scheduler_decision):
        decision = decide(result)
        assert decision.action is ControlAction.WAIT_RESOURCE
        assert decision.resource_wait is True
        assert decision.scientific_retry_authorized is False


def test_human_message_cannot_turn_authority_drift_into_resource_wait() -> None:
    result = OperationResult(
        status=OperationStatus.BLOCKED,
        reason_code=ReasonCode.AUTHORITY_IDENTITY_DRIFT,
        operation_id="prospective-op-2",
        scientific_state_mutated=None,
        side_effect_materialized=None,
        message="no permitted idle post-rollout GPU; target GPU has a compute process",
    )
    assert supervisor_decision(result).action is ControlAction.BLOCK
    assert scheduler_decision(result).action is ControlAction.BLOCK

def test_exact_resume_is_readback_not_retry_authority() -> None:
    result = OperationResult(
        status=OperationStatus.BLOCKED,
        reason_code=ReasonCode.ACKNOWLEDGEMENT_MISSING_AFTER_MATERIALIZATION,
        operation_id="prospective-op-3",
        scientific_state_mutated=None,
        side_effect_materialized=True,
    )
    decision = supervisor_decision(result)
    assert decision.action is ControlAction.READBACK_REQUIRED
    assert decision.blocks_progress is True
    assert decision.scientific_retry_authorized is False


def test_content_rejection_noop_and_success_are_typed() -> None:
    noop = OperationResult(
        status=OperationStatus.NOOP,
        reason_code=ReasonCode.CONTENT_REJECTED_KEEP_PRIOR,
        operation_id="prospective-op-4",
        scientific_state_mutated=False,
        side_effect_materialized=None,
    )
    success = OperationResult(
        status=OperationStatus.SUCCEEDED,
        reason_code=ReasonCode.NONE,
        operation_id="prospective-op-5",
        scientific_state_mutated=None,
        side_effect_materialized=None,
    )
    assert supervisor_decision(noop).action is ControlAction.ACCEPT_NOOP
    assert supervisor_decision(success).action is ControlAction.CONTINUE
    assert scheduler_decision(success).action is ControlAction.CONTINUE

def test_mapping_input_is_validated_before_control_flow() -> None:
    invalid = {
        "schema": "openevo.operation-result.202609162150",
        "status": "WAITING",
        "reason_code": "GPU_RESOURCE_UNAVAILABLE",
        "retry_class": "WAIT_RESOURCE",
        "operation_id": "prospective-op-6",
        "scientific_state_mutated": None,
        "side_effect_materialized": None,
        "message": "resource wait",
        "details": {},
    }
    with pytest.raises(OperationResultError, match="WAIT_RESOURCE"):
        scheduler_decision(invalid)
