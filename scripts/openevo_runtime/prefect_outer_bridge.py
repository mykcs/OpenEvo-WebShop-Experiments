"""Prefect-facing outer-orchestration policy for zero-formal/advisory work.

This module contains no Prefect dependency.  It is the OpenEVO-owned policy surface
that a Prefect wrapper may call.  Framework cache/retry/history never becomes
scientific, reuse, resource, or operation-identity authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from scripts.openevo_runtime.operation_records import (
    validate_exact_reuse_candidate,
    validate_operation_intent,
)
from scripts.openevo_runtime.result_codes import (
    OperationResult,
    OperationStatus,
    ReasonCode,
    RetryClass,
)
from scripts.openevo_runtime.runtime_events import validate_runtime_event


class PrefectBridgeError(ValueError):
    """A framework observation disagrees with OpenEVO-owned identity/state."""


class PrefectAction(str, Enum):
    DISPATCH_ENGINEERING = "DISPATCH_ENGINEERING"
    WAIT_RESOURCE = "WAIT_RESOURCE"
    TRANSIENT_INFRA_RETRY = "TRANSIENT_INFRA_RETRY"
    READBACK_REQUIRED = "READBACK_REQUIRED"
    OBSERVE_EVENT = "OBSERVE_EVENT"
    ACCEPT_ENGINEERING_CACHE = "ACCEPT_ENGINEERING_CACHE"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class PrefectBridgeDecision:
    action: PrefectAction
    operation_id: str
    reason: str
    framework_retry_allowed: bool = False
    scientific_retry_authorized: bool = False
    reuse_authorized_by_framework: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "operation_id": self.operation_id,
            "reason": self.reason,
            "framework_retry_allowed": self.framework_retry_allowed,
            "scientific_retry_authorized": self.scientific_retry_authorized,
            "reuse_authorized_by_framework": self.reuse_authorized_by_framework,
        }


def _coerce_result(value: OperationResult | Mapping[str, Any]) -> OperationResult:
    return value if isinstance(value, OperationResult) else OperationResult.from_mapping(value)


def transient_infrastructure_retry_allowed(
    result: OperationResult | Mapping[str, Any],
) -> bool:
    """Only a proved no-side-effect engineering failure is framework-retryable."""
    typed = _coerce_result(result)
    return (
        typed.reason_code is ReasonCode.ENGINEERING_FAILURE_NO_SIDE_EFFECT
        and typed.retry_class is RetryClass.EXACT_RESUME
        and typed.scientific_state_mutated is False
        and typed.side_effect_materialized is False
    )


def cached_result_decision(
    intent: Mapping[str, Any],
    *,
    cached_operation_digest: str,
    reuse_candidate: Mapping[str, Any] | None,
) -> PrefectBridgeDecision:
    """Gate a Prefect cache observation through exact Passport/OpenEVO policy."""
    validate_operation_intent(intent)
    operation_id = str(intent["idempotency_key"])
    if cached_operation_digest != operation_id:
        return PrefectBridgeDecision(
            PrefectAction.BLOCK,
            operation_id,
            "framework cache identity does not equal OpenEVO operation identity",
        )
    if reuse_candidate is None:
        return PrefectBridgeDecision(
            PrefectAction.BLOCK,
            operation_id,
            "exact Passport reuse candidate is required before cache acceptance",
        )
    validate_exact_reuse_candidate(reuse_candidate)
    if reuse_candidate["requested_operation_digest"] != operation_id:
        return PrefectBridgeDecision(
            PrefectAction.BLOCK,
            operation_id,
            "reuse candidate belongs to another operation",
        )
    if reuse_candidate["execution_generation_id"] != intent["execution_generation_id"]:
        return PrefectBridgeDecision(
            PrefectAction.BLOCK,
            operation_id,
            "reuse candidate belongs to another execution generation",
        )
    if reuse_candidate["side_effect_class"] != intent["side_effect_class"]:
        return PrefectBridgeDecision(
            PrefectAction.BLOCK,
            operation_id,
            "reuse candidate side-effect class disagrees with the frozen intent",
        )
    automatic = reuse_candidate["existing_policy_automatic_reuse_permitted"] is True
    review = reuse_candidate["policy_review_required"] is True
    pure_engineering = intent["side_effect_class"] == "PURE_ENGINEERING"
    if automatic and not review and pure_engineering:
        return PrefectBridgeDecision(
            PrefectAction.ACCEPT_ENGINEERING_CACHE,
            operation_id,
            "existing Passport policy permits this exact pure-engineering reuse",
        )
    return PrefectBridgeDecision(
        PrefectAction.BLOCK,
        operation_id,
        "OpenEVO reuse policy requires authority/review outside the workflow engine",
    )


def result_decision(
    intent: Mapping[str, Any],
    result: OperationResult | Mapping[str, Any],
) -> PrefectBridgeDecision:
    """Translate one typed OpenEVO result without granting new authority."""
    validate_operation_intent(intent)
    typed = _coerce_result(result)
    operation_id = str(intent["idempotency_key"])
    if typed.operation_id != operation_id:
        raise PrefectBridgeError("OperationResult belongs to another operation")
    if typed.retry_class is RetryClass.WAIT_RESOURCE:
        return PrefectBridgeDecision(
            PrefectAction.WAIT_RESOURCE,
            operation_id,
            "resource wait is workflow progress, not workflow failure",
        )
    if transient_infrastructure_retry_allowed(typed):
        return PrefectBridgeDecision(
            PrefectAction.TRANSIENT_INFRA_RETRY,
            operation_id,
            "machine evidence proves a no-side-effect engineering failure",
            framework_retry_allowed=True,
        )
    if typed.retry_class is RetryClass.EXACT_RESUME:
        return PrefectBridgeDecision(
            PrefectAction.READBACK_REQUIRED,
            operation_id,
            "materialized/ambiguous state requires OpenEVO readback before progress",
        )
    if typed.status in {OperationStatus.READY, OperationStatus.RUNNING}:
        return PrefectBridgeDecision(
            PrefectAction.DISPATCH_ENGINEERING,
            operation_id,
            "OpenEVO says the engineering operation may progress",
        )
    return PrefectBridgeDecision(
        PrefectAction.BLOCK,
        operation_id,
        "terminal/scientific/integrity state is not retry authority for Prefect",
    )


def event_visibility_decision(
    intent: Mapping[str, Any],
    event: Mapping[str, Any],
) -> PrefectBridgeDecision:
    """Expose a validated runtime event for operator visibility only."""
    validate_operation_intent(intent)
    validate_runtime_event(event)
    operation_id = str(intent["idempotency_key"])
    if event["event_type"] == "OPERATION_READY" and event["subject_id"] != intent["intent_id"]:
        raise PrefectBridgeError("OPERATION_READY event belongs to another intent")
    return PrefectBridgeDecision(
        PrefectAction.OBSERVE_EVENT,
        operation_id,
        "validated immutable-receipt event is visible to operators but grants no authority",
    )
