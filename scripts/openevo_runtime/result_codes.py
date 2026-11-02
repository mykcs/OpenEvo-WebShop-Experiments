"""Typed operation outcomes for the OpenEVO execution control plane.

Human-readable text is presentation only. Control flow consumes these enums and the
validated OperationResult contract. This module owns no scientific admissibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

SCHEMA = "openevo.operation-result.202609162150"


class OperationResultError(ValueError):
    """Raised when an operation result violates the shared machine contract."""


class OperationStatus(str, Enum):
    READY = "READY"
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    BLOCKED = "BLOCKED"
    SUCCEEDED = "SUCCEEDED"
    NOOP = "NOOP"
    SEALED = "SEALED"
    FAILED = "FAILED"


class RetryClass(str, Enum):
    NONE = "NONE"
    WAIT_RESOURCE = "WAIT_RESOURCE"
    EXACT_RESUME = "EXACT_RESUME"
    CONTENT_REJECTION_NOOP = "CONTENT_REJECTION_NOOP"
    INTEGRITY_UNKNOWN = "INTEGRITY_UNKNOWN"
    AUTHORITY_DRIFT = "AUTHORITY_DRIFT"
    SCIENCE_DECISION_REQUIRED = "SCIENCE_DECISION_REQUIRED"
    SEALED = "SEALED"


class ReasonCode(str, Enum):
    NONE = "NONE"
    GPU_RESOURCE_UNAVAILABLE = "GPU_RESOURCE_UNAVAILABLE"
    ARM_CONTROLLER_RESOURCE_BUSY = "ARM_CONTROLLER_RESOURCE_BUSY"
    ENGINEERING_FAILURE_NO_SIDE_EFFECT = "ENGINEERING_FAILURE_NO_SIDE_EFFECT"
    ACKNOWLEDGEMENT_MISSING_AFTER_MATERIALIZATION = "ACKNOWLEDGEMENT_MISSING_AFTER_MATERIALIZATION"
    CONTENT_REJECTED_KEEP_PRIOR = "CONTENT_REJECTED_KEEP_PRIOR"
    INTEGRITY_MISMATCH = "INTEGRITY_MISMATCH"
    SIDE_EFFECT_STATUS_UNKNOWN = "SIDE_EFFECT_STATUS_UNKNOWN"
    AUTHORITY_IDENTITY_DRIFT = "AUTHORITY_IDENTITY_DRIFT"
    SCIENCE_SEMANTICS_CHANGE_REQUIRED = "SCIENCE_SEMANTICS_CHANGE_REQUIRED"
    OPERATION_ALREADY_SEALED = "OPERATION_ALREADY_SEALED"
    UNCLASSIFIED_ENGINEERING_FAILURE = "UNCLASSIFIED_ENGINEERING_FAILURE"


REASON_RETRY_CLASS: dict[ReasonCode, RetryClass] = {
    ReasonCode.NONE: RetryClass.NONE,
    ReasonCode.GPU_RESOURCE_UNAVAILABLE: RetryClass.WAIT_RESOURCE,
    ReasonCode.ARM_CONTROLLER_RESOURCE_BUSY: RetryClass.WAIT_RESOURCE,
    ReasonCode.ENGINEERING_FAILURE_NO_SIDE_EFFECT: RetryClass.EXACT_RESUME,
    ReasonCode.ACKNOWLEDGEMENT_MISSING_AFTER_MATERIALIZATION: RetryClass.EXACT_RESUME,
    ReasonCode.CONTENT_REJECTED_KEEP_PRIOR: RetryClass.CONTENT_REJECTION_NOOP,
    ReasonCode.INTEGRITY_MISMATCH: RetryClass.INTEGRITY_UNKNOWN,
    ReasonCode.SIDE_EFFECT_STATUS_UNKNOWN: RetryClass.INTEGRITY_UNKNOWN,
    ReasonCode.AUTHORITY_IDENTITY_DRIFT: RetryClass.AUTHORITY_DRIFT,
    ReasonCode.SCIENCE_SEMANTICS_CHANGE_REQUIRED: RetryClass.SCIENCE_DECISION_REQUIRED,
    ReasonCode.OPERATION_ALREADY_SEALED: RetryClass.SEALED,
    ReasonCode.UNCLASSIFIED_ENGINEERING_FAILURE: RetryClass.INTEGRITY_UNKNOWN,
}

ALLOWED_STATUS: dict[RetryClass, frozenset[OperationStatus]] = {
    RetryClass.NONE: frozenset({OperationStatus.READY, OperationStatus.RUNNING, OperationStatus.SUCCEEDED, OperationStatus.FAILED}),
    RetryClass.WAIT_RESOURCE: frozenset({OperationStatus.WAITING, OperationStatus.BLOCKED}),
    RetryClass.EXACT_RESUME: frozenset({OperationStatus.BLOCKED}),
    RetryClass.CONTENT_REJECTION_NOOP: frozenset({OperationStatus.NOOP}),
    RetryClass.INTEGRITY_UNKNOWN: frozenset({OperationStatus.BLOCKED}),
    RetryClass.AUTHORITY_DRIFT: frozenset({OperationStatus.BLOCKED}),
    RetryClass.SCIENCE_DECISION_REQUIRED: frozenset({OperationStatus.BLOCKED}),
    RetryClass.SEALED: frozenset({OperationStatus.SEALED}),
}


def _enum_value(enum_type: type[Enum], value: Any, field_name: str) -> Enum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = [item.value for item in enum_type]
        raise OperationResultError(f"{field_name} must be one of {allowed!r}; observed={value!r}") from exc


def validate_operation_result(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != SCHEMA:
        raise OperationResultError(f"schema must equal {SCHEMA!r}")
    operation_id = payload.get("operation_id")
    if not isinstance(operation_id, str) or not operation_id.strip():
        raise OperationResultError("operation_id must be a non-empty string")
    status = _enum_value(OperationStatus, payload.get("status"), "status")
    reason = _enum_value(ReasonCode, payload.get("reason_code"), "reason_code")
    retry = _enum_value(RetryClass, payload.get("retry_class"), "retry_class")
    expected_retry = REASON_RETRY_CLASS[reason]
    if retry is not expected_retry:
        raise OperationResultError(
            f"reason_code={reason.value} requires retry_class={expected_retry.value}; observed={retry.value}"
        )
    if status not in ALLOWED_STATUS[retry]:
        allowed = sorted(item.value for item in ALLOWED_STATUS[retry])
        raise OperationResultError(f"retry_class={retry.value} does not allow status={status.value}; allowed={allowed}")
    for field_name in ("scientific_state_mutated", "side_effect_materialized"):
        value = payload.get(field_name)
        if value is not None and type(value) is not bool:
            raise OperationResultError(f"{field_name} must be boolean or null")
    if retry is RetryClass.WAIT_RESOURCE:
        if payload.get("scientific_state_mutated") is not False or payload.get("side_effect_materialized") is not False:
            raise OperationResultError("WAIT_RESOURCE must prove no scientific mutation and no materialized side effect")
    if reason is ReasonCode.ENGINEERING_FAILURE_NO_SIDE_EFFECT and payload.get("side_effect_materialized") is not False:
        raise OperationResultError("ENGINEERING_FAILURE_NO_SIDE_EFFECT requires side_effect_materialized=false")
    if reason is ReasonCode.ACKNOWLEDGEMENT_MISSING_AFTER_MATERIALIZATION and payload.get("side_effect_materialized") is not True:
        raise OperationResultError("ACKNOWLEDGEMENT_MISSING_AFTER_MATERIALIZATION requires side_effect_materialized=true")
    if retry is RetryClass.CONTENT_REJECTION_NOOP and payload.get("scientific_state_mutated") is not False:
        raise OperationResultError("CONTENT_REJECTION_NOOP must prove scientific_state_mutated=false")
    message = payload.get("message", "")
    if not isinstance(message, str):
        raise OperationResultError("message must be a string")
    details = payload.get("details", {})
    if not isinstance(details, Mapping):
        raise OperationResultError("details must be an object")


@dataclass(frozen=True)
class OperationResult:
    status: OperationStatus
    reason_code: ReasonCode
    operation_id: str
    scientific_state_mutated: bool | None
    side_effect_materialized: bool | None
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def retry_class(self) -> RetryClass:
        return REASON_RETRY_CLASS[self.reason_code]

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "status": self.status.value,
            "reason_code": self.reason_code.value,
            "retry_class": self.retry_class.value,
            "operation_id": self.operation_id,
            "scientific_state_mutated": self.scientific_state_mutated,
            "side_effect_materialized": self.side_effect_materialized,
            "message": self.message,
            "details": dict(self.details),
        }
        validate_operation_result(payload)
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OperationResult":
        validate_operation_result(payload)
        return cls(
            status=OperationStatus(payload["status"]),
            reason_code=ReasonCode(payload["reason_code"]),
            operation_id=str(payload["operation_id"]),
            scientific_state_mutated=payload.get("scientific_state_mutated"),
            side_effect_materialized=payload.get("side_effect_materialized"),
            message=str(payload.get("message", "")),
            details=dict(payload.get("details", {})),
        )
