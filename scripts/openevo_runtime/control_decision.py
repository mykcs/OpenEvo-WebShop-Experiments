"""Framework-neutral typed control decisions for prospective supervisors/schedulers.

This module is a non-live Execution Runtime consumer. It never parses human text and
never grants scientific retry authority. The active #523 execution is intentionally
not imported or patched here.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Any

from scripts.openevo_runtime.result_codes import (
    OperationResult,
    OperationStatus,
    RetryClass,
)


class ControlAction(str, Enum):
    CONTINUE = "CONTINUE"
    WAIT_RESOURCE = "WAIT_RESOURCE"
    READBACK_REQUIRED = "READBACK_REQUIRED"
    ACCEPT_NOOP = "ACCEPT_NOOP"
    ACCEPT_SEAL = "ACCEPT_SEAL"
    BLOCK = "BLOCK"

@dataclass(frozen=True)
class ControlDecision:
    action: ControlAction
    operation_id: str
    blocks_progress: bool
    resource_wait: bool
    scientific_retry_authorized: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "operation_id": self.operation_id,
            "blocks_progress": self.blocks_progress,
            "resource_wait": self.resource_wait,
            "scientific_retry_authorized": self.scientific_retry_authorized,
        }


def _coerce(result: OperationResult | Mapping[str, Any]) -> OperationResult:
    return result if isinstance(result, OperationResult) else OperationResult.from_mapping(result)


def _decision(result: OperationResult, action: ControlAction, *, blocks: bool, resource_wait: bool = False) -> ControlDecision:
    return ControlDecision(
        action=action,
        operation_id=result.operation_id,
        blocks_progress=blocks,
        resource_wait=resource_wait,
    )

def supervisor_decision(result: OperationResult | Mapping[str, Any]) -> ControlDecision:
    """Consume a validated typed outcome for prospective supervisor control flow."""
    typed = _coerce(result)
    if typed.retry_class is RetryClass.WAIT_RESOURCE:
        return _decision(typed, ControlAction.WAIT_RESOURCE, blocks=False, resource_wait=True)
    if typed.retry_class is RetryClass.EXACT_RESUME:
        return _decision(typed, ControlAction.READBACK_REQUIRED, blocks=True)
    if typed.retry_class is RetryClass.CONTENT_REJECTION_NOOP:
        return _decision(typed, ControlAction.ACCEPT_NOOP, blocks=False)
    if typed.retry_class is RetryClass.SEALED:
        return _decision(typed, ControlAction.ACCEPT_SEAL, blocks=False)
    if typed.retry_class is not RetryClass.NONE:
        return _decision(typed, ControlAction.BLOCK, blocks=True)
    if typed.status in {OperationStatus.SUCCEEDED, OperationStatus.READY, OperationStatus.RUNNING}:
        return _decision(typed, ControlAction.CONTINUE, blocks=False)
    return _decision(typed, ControlAction.BLOCK, blocks=True)


def scheduler_decision(result: OperationResult | Mapping[str, Any]) -> ControlDecision:
    """Consume typed resource/operation state without inspecting message/stderr text."""
    typed = _coerce(result)
    if typed.retry_class is RetryClass.WAIT_RESOURCE:
        return _decision(typed, ControlAction.WAIT_RESOURCE, blocks=False, resource_wait=True)
    if typed.status is OperationStatus.SUCCEEDED and typed.retry_class is RetryClass.NONE:
        return _decision(typed, ControlAction.CONTINUE, blocks=False)
    if typed.retry_class is RetryClass.SEALED:
        return _decision(typed, ControlAction.ACCEPT_SEAL, blocks=False)
    return _decision(typed, ControlAction.BLOCK, blocks=True)
