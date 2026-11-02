"""Read-only shadow parity for predecessor controller observations.

This module is migration evidence only. It consumes already-emitted reference statuses,
normalizes them through the typed Execution Runtime contract, and never writes a live
run, launches work, grants retry/reuse authority, or treats framework history as
scientific identity.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from scripts.openevo_runtime.control_decision import ControlAction, supervisor_decision
from scripts.openevo_runtime.result_codes import OperationResult, OperationStatus, ReasonCode

SCHEMA = "openevo.execution-runtime.shadow-parity.202609171600"


class ShadowClass(str, Enum):
    READY = "ready"
    WAIT_RESOURCE = "wait_resource"
    EXACT_RESUME = "exact_resume"
    CONTENT_NOOP = "content_noop"
    BLOCKED_INTEGRITY = "blocked_integrity"
    SEALED = "sealed"
    AUDIT_ONLY = "audit_only"


# Frozen predecessor machine statuses observed in the exact 4f5db462 controller/log
# surface. This table is a compatibility edge only; new producers emit typed results.
_REFERENCE_STATUS_CLASS: dict[str, ShadowClass] = {
    "DATASET_READY": ShadowClass.READY,
    "ROUND_ADVANCED": ShadowClass.READY,
    "ROUND_PASS": ShadowClass.READY,
    "DISPATCHED": ShadowClass.READY,
    "ACTIVE": ShadowClass.READY,
    "IDLE": ShadowClass.READY,
    "ROLLOUT_PHASE_PASS": ShadowClass.READY,
    "ROLLOUT_BATCH_COMPLETE": ShadowClass.READY,
    "WAITING_FOR_SHARED_GPU_POOL": ShadowClass.WAIT_RESOURCE,
    "WAITING_FOR_GPU_RESOURCE": ShadowClass.WAIT_RESOURCE,
    "WAITING_FOR_ARM_CONTROLLER_RESOURCE": ShadowClass.WAIT_RESOURCE,
    "RECOVERED_COMPLETED_ROUND": ShadowClass.EXACT_RESUME,
    "PASS_WITH_CONTENT_REJECTION": ShadowClass.CONTENT_NOOP,
    "BLOCKED_INTEGRITY_OR_UNRESOLVED_HEALTH": ShadowClass.BLOCKED_INTEGRITY,
    "ARM_SEALED": ShadowClass.SEALED,
    "OBSERVE_ONLY": ShadowClass.AUDIT_ONLY,
}


_ACTION_CLASS: dict[ControlAction, ShadowClass] = {
    ControlAction.CONTINUE: ShadowClass.READY,
    ControlAction.WAIT_RESOURCE: ShadowClass.WAIT_RESOURCE,
    ControlAction.READBACK_REQUIRED: ShadowClass.EXACT_RESUME,
    ControlAction.ACCEPT_NOOP: ShadowClass.CONTENT_NOOP,
    ControlAction.ACCEPT_SEAL: ShadowClass.SEALED,
    ControlAction.BLOCK: ShadowClass.BLOCKED_INTEGRITY,
}


def reference_class(status: str) -> ShadowClass:
    try:
        return _REFERENCE_STATUS_CLASS[status]
    except KeyError as exc:
        raise ValueError(f"unknown predecessor shadow status: {status!r}") from exc


def typed_result_for_reference_status(*, operation_id: str, status: str) -> OperationResult | None:
    """Project a frozen predecessor status into the typed compatibility boundary."""
    klass = reference_class(status)
    if klass is ShadowClass.AUDIT_ONLY:
        return None
    if klass is ShadowClass.WAIT_RESOURCE:
        return OperationResult(
            status=OperationStatus.WAITING,
            reason_code=ReasonCode.GPU_RESOURCE_UNAVAILABLE,
            operation_id=operation_id,
            scientific_state_mutated=False,
            side_effect_materialized=False,
            details={"shadow_source_status": status},
        )
    if klass is ShadowClass.EXACT_RESUME:
        return OperationResult(
            status=OperationStatus.BLOCKED,
            reason_code=ReasonCode.ACKNOWLEDGEMENT_MISSING_AFTER_MATERIALIZATION,
            operation_id=operation_id,
            scientific_state_mutated=None,
            side_effect_materialized=True,
            details={"shadow_source_status": status},
        )
    if klass is ShadowClass.CONTENT_NOOP:
        return OperationResult(
            status=OperationStatus.NOOP,
            reason_code=ReasonCode.CONTENT_REJECTED_KEEP_PRIOR,
            operation_id=operation_id,
            scientific_state_mutated=False,
            side_effect_materialized=True,
            details={"shadow_source_status": status},
        )
    if klass is ShadowClass.BLOCKED_INTEGRITY:
        return OperationResult(
            status=OperationStatus.BLOCKED,
            reason_code=ReasonCode.INTEGRITY_MISMATCH,
            operation_id=operation_id,
            scientific_state_mutated=None,
            side_effect_materialized=None,
            details={"shadow_source_status": status},
        )
    if klass is ShadowClass.SEALED:
        return OperationResult(
            status=OperationStatus.SEALED,
            reason_code=ReasonCode.OPERATION_ALREADY_SEALED,
            operation_id=operation_id,
            scientific_state_mutated=False,
            side_effect_materialized=True,
            details={"shadow_source_status": status},
        )
    return OperationResult(
        status=OperationStatus.SUCCEEDED,
        reason_code=ReasonCode.NONE,
        operation_id=operation_id,
        scientific_state_mutated=None,
        side_effect_materialized=None,
        details={"shadow_source_status": status},
    )


def modernization_class(result: OperationResult | Mapping[str, Any]) -> ShadowClass:
    decision = supervisor_decision(result)
    return _ACTION_CLASS[decision.action]


@dataclass(frozen=True)
class ShadowComparison:
    source: str
    reference_status: str
    reference_class: ShadowClass
    modernization_class: ShadowClass
    observation_count: int

    @property
    def parity(self) -> bool:
        return self.reference_class is self.modernization_class

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "reference_status": self.reference_status,
            "reference_class": self.reference_class.value,
            "modernization_class": self.modernization_class.value,
            "observation_count": self.observation_count,
            "parity": self.parity,
        }


def compare_observed_status(*, source: str, status: str, count: int) -> ShadowComparison | None:
    if type(count) is not int or count <= 0:
        raise ValueError("shadow observation count must be a positive integer")
    ref = reference_class(status)
    typed = typed_result_for_reference_status(operation_id=f"shadow:{source}:{status}", status=status)
    if ref is ShadowClass.AUDIT_ONLY:
        if typed is not None:
            raise AssertionError("audit-only reference status unexpectedly produced an operation result")
        return None
    assert typed is not None
    modern = modernization_class(typed)
    return ShadowComparison(source, status, ref, modern, count)


def compare_status_inventory(inventory: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    """Compare every observed top-level predecessor transition in a bounded snapshot."""
    comparisons: list[dict[str, Any]] = []
    audit_only: list[dict[str, Any]] = []
    observed = 0
    compared = 0
    for source in sorted(inventory):
        statuses = inventory[source]
        if not isinstance(statuses, Mapping):
            raise ValueError(f"inventory[{source!r}] must be an object")
        for status in sorted(statuses):
            count = statuses[status]
            observed += count
            row = compare_observed_status(source=source, status=status, count=count)
            if row is None:
                audit_only.append({"source": source, "reference_status": status, "observation_count": count})
                continue
            compared += count
            comparisons.append(row.to_dict())
    divergences = [row for row in comparisons if not row["parity"]]
    return {
        "schema": SCHEMA,
        "observed_transition_count": observed,
        "compared_operation_transition_count": compared,
        "audit_only_transition_count": observed - compared,
        "comparisons": comparisons,
        "audit_only": audit_only,
        "unexplained_divergence_count": len(divergences),
        "parity": not divergences,
        "live_mutation": False,
        "scientific_retry_authorized": False,
        "scientific_authority_granted": False,
        "framework_history_is_scientific_identity": False,
    }
