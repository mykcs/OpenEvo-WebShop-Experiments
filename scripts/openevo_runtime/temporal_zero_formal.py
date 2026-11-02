"""Temporal zero-formal prototype for Execution Runtime modernization.

Workflow code is deterministic and side-effect free. Every external action is an
Activity, and every Activity is bound to an existing OpenEVO OperationIntent whose
stable operation digest is also the Temporal Activity ID. This module is prototype
only: it consumes zero formal rows, zero final-panel budget, and grants no science,
retry, reuse, or resource authority.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any, Mapping

from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

ACTIVITY_KINDS = (
    "rollout_like",
    "transition_like",
    "reflector_like",
    "artifact_readback",
    "seal_like",
)
ACTIVITY_NAMES = {kind: f"openevo_zero_formal_{kind}" for kind in ACTIVITY_KINDS}
NO_RETRY = RetryPolicy(maximum_attempts=1)
ACTIVITY_TIMEOUT = timedelta(seconds=30)
def _zero_formal_activity(request: Mapping[str, Any], expected_kind: str) -> dict[str, Any]:
    """Validate one synthetic Activity request and echo its typed terminal result."""
    from scripts.openevo_runtime.operation_records import (
        build_operation_terminal,
        validate_operation_intent,
    )
    from scripts.openevo_runtime.result_codes import OperationResult

    if request.get("activity_kind") != expected_kind:
        raise ValueError("activity kind disagrees with registered Activity")
    if request.get("synthetic_zero_formal") is not True:
        raise ValueError("Temporal prototype Activities must be synthetic/zero-formal")
    if request.get("formal_rows_consumed") != 0 or request.get("final_panel_access") != 0:
        raise ValueError("Temporal prototype cannot consume formal rows/final-panel budget")
    intent = request.get("intent")
    if not isinstance(intent, Mapping):
        raise ValueError("Activity request requires one OperationIntent")
    validate_operation_intent(intent)
    result = request.get("result")
    if not isinstance(result, Mapping):
        raise ValueError("Activity request requires one typed result")
    typed = OperationResult.from_mapping(result)
    if typed.operation_id != intent["idempotency_key"]:
        raise ValueError("Activity result must bind the OpenEVO operation idempotency key")
    receipt = request.get("terminal_receipt_identity")
    terminal = build_operation_terminal(intent, typed, terminal_receipt_identity=receipt)
    return {
        "activity_kind": expected_kind,
        "activity_id": intent["idempotency_key"],
        "terminal": terminal,
        "formal_rows_consumed": 0,
        "final_panel_access": 0,
        "scientific_authority_granted": False,
        "scientific_retry_authorized": False,
    }
@activity.defn(name=ACTIVITY_NAMES["rollout_like"])
async def rollout_like_activity(request: dict[str, Any]) -> dict[str, Any]:
    return _zero_formal_activity(request, "rollout_like")


@activity.defn(name=ACTIVITY_NAMES["transition_like"])
async def transition_like_activity(request: dict[str, Any]) -> dict[str, Any]:
    return _zero_formal_activity(request, "transition_like")


@activity.defn(name=ACTIVITY_NAMES["reflector_like"])
async def reflector_like_activity(request: dict[str, Any]) -> dict[str, Any]:
    return _zero_formal_activity(request, "reflector_like")


@activity.defn(name=ACTIVITY_NAMES["artifact_readback"])
async def artifact_readback_activity(request: dict[str, Any]) -> dict[str, Any]:
    return _zero_formal_activity(request, "artifact_readback")


@activity.defn(name=ACTIVITY_NAMES["seal_like"])
async def seal_like_activity(request: dict[str, Any]) -> dict[str, Any]:
    return _zero_formal_activity(request, "seal_like")


ACTIVITY_FUNCTIONS = {
    "rollout_like": rollout_like_activity,
    "transition_like": transition_like_activity,
    "reflector_like": reflector_like_activity,
    "artifact_readback": artifact_readback_activity,
    "seal_like": seal_like_activity,
}
@workflow.defn
class ArmWorkflow:
    """Deterministic zero-formal arm workflow; all side effects live in Activities."""

    def __init__(self) -> None:
        self._resource_releases: set[str] = set()
        self._terminal_readbacks: dict[str, dict[str, Any]] = {}
        self._integrity_ambiguities: set[str] = set()
        self._state: dict[str, Any] = {"status": "STARTING", "operation_id": None}

    @workflow.signal
    def resource_available(self, operation_id: str) -> None:
        self._resource_releases.add(operation_id)

    @workflow.signal
    def accept_terminal_readback(self, response: dict[str, Any]) -> None:
        operation_id = response.get("activity_id")
        terminal = response.get("terminal")
        if not isinstance(operation_id, str) or not isinstance(terminal, dict):
            raise ValueError("terminal readback requires activity_id and terminal")
        if terminal.get("operation_digest") != operation_id:
            raise ValueError("terminal readback operation identity drift")
        existing = self._terminal_readbacks.get(operation_id)
        if existing is not None and existing != response:
            self._integrity_ambiguities.add(operation_id)
            return
        self._terminal_readbacks[operation_id] = dict(response)

    @workflow.signal
    def terminal_readback_ambiguous(self, operation_id: str) -> None:
        self._integrity_ambiguities.add(operation_id)

    @workflow.query
    def runtime_state(self) -> dict[str, Any]:
        return dict(self._state)

    @workflow.run
    async def run(self, plan: dict[str, Any]) -> dict[str, Any]:
        if plan.get("synthetic_zero_formal") is not True:
            raise ValueError("ArmWorkflow prototype accepts synthetic/zero-formal plans only")
        if plan.get("formal_rows_consumed") != 0 or plan.get("final_panel_access") != 0:
            raise ValueError("ArmWorkflow prototype cannot consume formal science budget")
        operations = plan.get("operations")
        if not isinstance(operations, list):
            raise ValueError("ArmWorkflow operations must be a list")
        terminals: list[dict[str, Any]] = []
        for request in operations:
            if not isinstance(request, dict):
                raise ValueError("ArmWorkflow operation must be an object")
            kind = request.get("activity_kind")
            fn = ACTIVITY_FUNCTIONS.get(kind)
            if fn is None:
                raise ValueError(f"unsupported zero-formal activity kind: {kind!r}")
            intent = request.get("intent")
            if not isinstance(intent, dict) or not isinstance(intent.get("idempotency_key"), str):
                raise ValueError("Activity requires a frozen OpenEVO idempotency key")
            operation_id = intent["idempotency_key"]
            if request.get("synthetic_wait_resource") is True:
                self._state = {"status": "WAIT_RESOURCE", "operation_id": operation_id}
                await workflow.wait_condition(lambda: operation_id in self._resource_releases)
            self._state = {"status": "RUNNING", "operation_id": operation_id}
            try:
                terminal = await workflow.execute_activity(
                    fn,
                    request,
                    activity_id=operation_id,
                    start_to_close_timeout=ACTIVITY_TIMEOUT,
                    retry_policy=NO_RETRY,
                )
            except ActivityError:
                if request.get("synthetic_ack_loss_after_materialization") is not True:
                    raise
                self._state = {"status": "READBACK_REQUIRED", "operation_id": operation_id}
                await workflow.wait_condition(
                    lambda: operation_id in self._terminal_readbacks
                    or operation_id in self._integrity_ambiguities
                )
                if operation_id in self._integrity_ambiguities:
                    self._state = {"status": "BLOCKED_INTEGRITY", "operation_id": operation_id}
                    return {
                        "workflow": "ArmWorkflow",
                        "arm": plan.get("arm"),
                        "workflow_status": "BLOCKED_INTEGRITY",
                        "blocked_operation_id": operation_id,
                        "terminals": terminals,
                        "formal_rows_consumed": 0,
                        "final_panel_access": 0,
                        "scientific_authority_granted": False,
                        "scientific_retry_authorized": False,
                    }
                terminal = dict(self._terminal_readbacks[operation_id])
                terminal["recovered_from_ack_loss"] = True
            terminals.append(terminal)
        self._state = {"status": "SEALED", "operation_id": None}
        return {
            "workflow": "ArmWorkflow",
            "arm": plan.get("arm"),
            "terminals": terminals,
            "formal_rows_consumed": 0,
            "final_panel_access": 0,
            "scientific_authority_granted": False,
        }
@workflow.defn
class PairAuditWorkflow:
    """Asynchronous transport-only consumer for immutable pair-audit events."""

    def __init__(self) -> None:
        self._receipts: dict[str, dict[str, Any]] = {}

    @workflow.signal
    def accept_pair_receipt(self, receipt: dict[str, Any]) -> None:
        identity = receipt.get("pair_receipt_identity")
        if not isinstance(identity, str) or not identity:
            raise ValueError("pair receipt requires immutable identity")
        existing = self._receipts.get(identity)
        if existing is not None and existing != receipt:
            raise ValueError("conflicting duplicate pair receipt")
        self._receipts[identity] = dict(receipt)

    @workflow.query
    def receipt_count(self) -> int:
        return len(self._receipts)

    @workflow.run
    async def run(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("synthetic_zero_formal") is not True:
            raise ValueError("PairAuditWorkflow prototype is zero-formal only")
        expected = request.get("expected_receipts")
        if type(expected) is not int or expected < 0:
            raise ValueError("expected_receipts must be a non-negative integer")
        await workflow.wait_condition(lambda: len(self._receipts) >= expected)
        return {
            "workflow": "PairAuditWorkflow",
            "pair_receipts": [self._receipts[key] for key in sorted(self._receipts)],
            "admission_authority": False,
            "blocks_arm_progression": False,
            "formal_rows_consumed": 0,
            "final_panel_access": 0,
            "scientific_authority_granted": False,
        }


REGISTERED_ACTIVITIES = list(ACTIVITY_FUNCTIONS.values())
REGISTERED_WORKFLOWS = [ArmWorkflow, PairAuditWorkflow]
