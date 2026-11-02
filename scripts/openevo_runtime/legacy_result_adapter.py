"""Narrow bridge from predecessor text errors to typed OperationResult.

Only this compatibility boundary may recognize the frozen predecessor messages below.
New producers must emit OperationResult directly; consumers must not branch on text.
The bridge is prospective/shadow code and does not mutate the live #523 execution.
"""
from __future__ import annotations

from dataclasses import dataclass

from scripts.openevo_runtime.result_codes import OperationResult, OperationStatus, ReasonCode


@dataclass(frozen=True)
class LegacyMarker:
    text: str
    reason_code: ReasonCode


LEGACY_RESOURCE_MARKERS: tuple[LegacyMarker, ...] = (
    # Canonical Ray backend still emits these frozen text sentinels.  Keep
    # translation here, at the compatibility edge; broker consumers receive
    # typed OperationResult values and never branch on stderr themselves.
    LegacyMarker("RAY_NO_FEASIBLE_PLACEMENT", ReasonCode.GPU_RESOURCE_UNAVAILABLE),
    LegacyMarker("RAY_ADMISSION_PAUSED", ReasonCode.GPU_RESOURCE_UNAVAILABLE),
    LegacyMarker("formal arm already has an active controller", ReasonCode.ARM_CONTROLLER_RESOURCE_BUSY),
    LegacyMarker("no permitted idle post-rollout GPU", ReasonCode.GPU_RESOURCE_UNAVAILABLE),
    LegacyMarker("is not idle by gpu-who", ReasonCode.GPU_RESOURCE_UNAVAILABLE),
    LegacyMarker("not idle by nvidia-smi", ReasonCode.GPU_RESOURCE_UNAVAILABLE),
    LegacyMarker("target GPU has a compute process", ReasonCode.GPU_RESOURCE_UNAVAILABLE),
)


def classify_legacy_failure_text(*, operation_id: str, text: str) -> OperationResult:
    for marker in LEGACY_RESOURCE_MARKERS:
        if marker.text in text:
            return OperationResult(
                status=OperationStatus.WAITING,
                reason_code=marker.reason_code,
                operation_id=operation_id,
                scientific_state_mutated=False,
                side_effect_materialized=False,
                message=text,
                details={"compatibility_source": "async-successor-20260916", "matched_marker": marker.text},
            )
    return OperationResult(
        status=OperationStatus.BLOCKED,
        reason_code=ReasonCode.UNCLASSIFIED_ENGINEERING_FAILURE,
        operation_id=operation_id,
        scientific_state_mutated=None,
        side_effect_materialized=None,
        message=text,
        details={"compatibility_source": "async-successor-20260916", "classification": "FAIL_CLOSED"},
    )


def classify_legacy_process_exit(*, operation_id: str, returncode: int, stdout: str = "", stderr: str = "") -> OperationResult:
    if returncode == 0:
        return OperationResult(
            status=OperationStatus.SUCCEEDED,
            reason_code=ReasonCode.NONE,
            operation_id=operation_id,
            scientific_state_mutated=None,
            side_effect_materialized=None,
            message="legacy process exited successfully; caller must validate canonical terminal receipt",
            details={"returncode": returncode, "terminal_receipt_required": True},
        )
    combined = "\n".join(part for part in (stdout, stderr) if part)
    return classify_legacy_failure_text(operation_id=operation_id, text=combined)
