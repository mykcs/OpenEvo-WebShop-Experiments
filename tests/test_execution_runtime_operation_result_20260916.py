from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.openevo_runtime.legacy_result_adapter import classify_legacy_failure_text, classify_legacy_process_exit
from scripts.openevo_runtime.result_codes import (
    OperationResult,
    OperationResultError,
    OperationStatus,
    ReasonCode,
    RetryClass,
    SCHEMA,
    validate_operation_result,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "operation-result-202609162150.schema.json"


def test_wait_resource_is_typed_and_side_effect_free() -> None:
    result = OperationResult(
        status=OperationStatus.WAITING,
        reason_code=ReasonCode.GPU_RESOURCE_UNAVAILABLE,
        operation_id="arm=off/round=27/channel=sd_lora/attempt=0",
        scientific_state_mutated=False,
        side_effect_materialized=False,
        message="presentation only",
    )
    payload = result.to_dict()
    assert payload["retry_class"] == RetryClass.WAIT_RESOURCE.value
    assert OperationResult.from_mapping(payload) == result


def test_reason_retry_mismatch_is_rejected() -> None:
    payload = OperationResult(
        status=OperationStatus.WAITING,
        reason_code=ReasonCode.GPU_RESOURCE_UNAVAILABLE,
        operation_id="op-1",
        scientific_state_mutated=False,
        side_effect_materialized=False,
    ).to_dict()
    payload["retry_class"] = RetryClass.EXACT_RESUME.value
    with pytest.raises(OperationResultError, match="requires retry_class"):
        validate_operation_result(payload)


def test_wait_resource_cannot_hide_a_side_effect() -> None:
    payload = OperationResult(
        status=OperationStatus.WAITING,
        reason_code=ReasonCode.GPU_RESOURCE_UNAVAILABLE,
        operation_id="op-2",
        scientific_state_mutated=False,
        side_effect_materialized=False,
    ).to_dict()
    payload["side_effect_materialized"] = True
    with pytest.raises(OperationResultError, match="WAIT_RESOURCE"):
        validate_operation_result(payload)


def test_exact_resume_distinguishes_no_effect_from_lost_ack() -> None:
    no_effect = OperationResult(
        status=OperationStatus.BLOCKED,
        reason_code=ReasonCode.ENGINEERING_FAILURE_NO_SIDE_EFFECT,
        operation_id="op-3",
        scientific_state_mutated=False,
        side_effect_materialized=False,
    ).to_dict()
    lost_ack = OperationResult(
        status=OperationStatus.BLOCKED,
        reason_code=ReasonCode.ACKNOWLEDGEMENT_MISSING_AFTER_MATERIALIZATION,
        operation_id="op-4",
        scientific_state_mutated=None,
        side_effect_materialized=True,
    ).to_dict()
    assert no_effect["retry_class"] == RetryClass.EXACT_RESUME.value
    assert lost_ack["retry_class"] == RetryClass.EXACT_RESUME.value
    assert no_effect["side_effect_materialized"] is False
    assert lost_ack["side_effect_materialized"] is True


def test_content_rejection_noop_cannot_claim_state_mutation() -> None:
    payload = OperationResult(
        status=OperationStatus.NOOP,
        reason_code=ReasonCode.CONTENT_REJECTED_KEEP_PRIOR,
        operation_id="op-5",
        scientific_state_mutated=False,
        side_effect_materialized=True,
    ).to_dict()
    payload["scientific_state_mutated"] = True
    with pytest.raises(OperationResultError, match="CONTENT_REJECTION_NOOP"):
        validate_operation_result(payload)


def test_unknown_failure_fails_closed_without_inventing_effect_state() -> None:
    result = classify_legacy_failure_text(operation_id="op-6", text="unexpected launcher failure")
    payload = result.to_dict()
    assert payload["status"] == OperationStatus.BLOCKED.value
    assert payload["reason_code"] == ReasonCode.UNCLASSIFIED_ENGINEERING_FAILURE.value
    assert payload["retry_class"] == RetryClass.INTEGRITY_UNKNOWN.value
    assert payload["scientific_state_mutated"] is None
    assert payload["side_effect_materialized"] is None


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("formal arm already has an active controller", ReasonCode.ARM_CONTROLLER_RESOURCE_BUSY),
        ("no permitted idle post-rollout GPU", ReasonCode.GPU_RESOURCE_UNAVAILABLE),
        ("GPU-xyz is not idle by gpu-who", ReasonCode.GPU_RESOURCE_UNAVAILABLE),
        ("GPU-xyz not idle by nvidia-smi", ReasonCode.GPU_RESOURCE_UNAVAILABLE),
        ("target GPU has a compute process", ReasonCode.GPU_RESOURCE_UNAVAILABLE),
    ],
)
def test_frozen_predecessor_resource_markers_map_at_one_bridge(text: str, reason: ReasonCode) -> None:
    payload = classify_legacy_failure_text(operation_id="legacy-op", text=text).to_dict()
    assert payload["reason_code"] == reason.value
    assert payload["retry_class"] == RetryClass.WAIT_RESOURCE.value


def test_legacy_success_requires_terminal_receipt_readback() -> None:
    payload = classify_legacy_process_exit(operation_id="legacy-success", returncode=0).to_dict()
    assert payload["status"] == OperationStatus.SUCCEEDED.value
    assert payload["retry_class"] == RetryClass.NONE.value
    assert payload["details"]["terminal_receipt_required"] is True
    assert payload["scientific_state_mutated"] is None
    assert payload["side_effect_materialized"] is None


def test_schema_enums_stay_in_sync_with_python_contract() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    props = schema["properties"]
    assert props["schema"]["const"] == SCHEMA
    assert set(props["status"]["enum"]) == {item.value for item in OperationStatus}
    assert set(props["reason_code"]["enum"]) == {item.value for item in ReasonCode}
    assert set(props["retry_class"]["enum"]) == {item.value for item in RetryClass}


def test_human_message_does_not_change_typed_classification() -> None:
    result = OperationResult(
        status=OperationStatus.BLOCKED,
        reason_code=ReasonCode.AUTHORITY_IDENTITY_DRIFT,
        operation_id="op-7",
        scientific_state_mutated=False,
        side_effect_materialized=False,
        message="GPU busy exact resume safe sealed please retry",
    )
    payload = result.to_dict()
    assert payload["retry_class"] == RetryClass.AUTHORITY_DRIFT.value
