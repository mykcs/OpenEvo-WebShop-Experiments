"""Fail-closed guard for opportunistic zero-formal GPU canary preparation.

This module does not acquire resources or grant execution authority.  It only answers
whether a fresh read-only witness shows an observed gap in the already-authorized
formal resource lane.  A sampled idle GPU is never sufficient evidence by itself.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Mapping, Sequence

WITNESS_SCHEMA = "openevo.execution-runtime-formal-priority-witness.202609172225"
DECISION_SCHEMA = "openevo.execution-runtime-opportunistic-canary-decision.202609172225"
HANDOFF_SCHEMA = "openevo.execution-runtime-formal-priority-handoff.202609172225"


class OpportunisticCanaryGuardError(RuntimeError):
    """The formal-priority witness or handoff cannot be trusted exactly."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _exact_int_list(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise OpportunisticCanaryGuardError(f"{label} must be an integer list")
    if len(value) != len(set(value)):
        raise OpportunisticCanaryGuardError(f"{label} contains duplicates")
    return list(value)


def _require_nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise OpportunisticCanaryGuardError(f"{label} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class OpportunisticCanaryDecision:
    report: Mapping[str, Any]

    @property
    def observed_gap(self) -> bool:
        return bool(self.report["observed_formal_gap"])

    @property
    def gate_pass(self) -> bool:
        return bool(self.report["opportunistic_gate_pass"])


def evaluate_opportunistic_canary_gap(
    witness: Mapping[str, Any],
    *,
    expected_execution_git_sha: str,
    expected_topology_id: str,
    candidate_gpu_indices: Sequence[int],
    forbidden_gpu_indices: Sequence[int] = (),
    handoff: Mapping[str, Any] | None = None,
    max_witness_age_seconds: float = 1.0,
    now_epoch: float | None = None,
) -> OpportunisticCanaryDecision:
    """Evaluate a formal-priority witness without acquiring or reserving a GPU.

    ``opportunistic_gate_pass`` means only that the resource-only canary may proceed to
    its separate execution-authority check.  It is never scientific authority, topology
    qualification, or permission to make the formal experiment wait.
    """
    if witness.get("schema") != WITNESS_SCHEMA:
        raise OpportunisticCanaryGuardError("formal-priority witness schema drift")
    if witness.get("execution_git_sha") != expected_execution_git_sha:
        raise OpportunisticCanaryGuardError("formal execution identity drift")
    if witness.get("topology_id") != expected_topology_id:
        raise OpportunisticCanaryGuardError("qualified topology identity drift")

    candidates = list(candidate_gpu_indices)
    if not candidates or any(type(index) is not int for index in candidates):
        raise OpportunisticCanaryGuardError("candidate GPU set must be a non-empty integer sequence")
    if len(candidates) != len(set(candidates)):
        raise OpportunisticCanaryGuardError("candidate GPU set contains duplicates")
    forbidden = set(forbidden_gpu_indices)
    if set(candidates) & forbidden:
        raise OpportunisticCanaryGuardError("candidate GPU set intersects explicitly forbidden GPUs")
    witnessed_candidates = _exact_int_list(witness.get("candidate_gpu_indices"), "witness candidate GPUs")
    if witnessed_candidates != candidates:
        raise OpportunisticCanaryGuardError("witness candidate GPU set drift")

    pending = _require_nonnegative_int(witness.get("pending_formal_gpu_requests"), "pending formal GPU requests")
    reservations = _require_nonnegative_int(witness.get("active_formal_reservations"), "active formal reservations")
    leases = _require_nonnegative_int(witness.get("active_formal_gpu_leases"), "active formal GPU leases")
    final_access = _require_nonnegative_int(witness.get("final_panel_access_count"), "final-panel access count")
    observed_at = witness.get("observed_at_epoch")
    if not isinstance(observed_at, (int, float)):
        raise OpportunisticCanaryGuardError("witness observed_at_epoch missing")
    now = time.time() if now_epoch is None else float(now_epoch)
    age = now - float(observed_at)
    if age < 0:
        raise OpportunisticCanaryGuardError("formal-priority witness is from the future")

    blockers: list[str] = []
    if age > float(max_witness_age_seconds):
        blockers.append("FORMAL_PRIORITY_WITNESS_STALE")
    if witness.get("scheduler_status") != "IDLE":
        blockers.append("FORMAL_SCHEDULER_NOT_IDLE")
    if pending:
        blockers.append("FORMAL_GPU_REQUEST_PENDING")
    if reservations:
        blockers.append("FORMAL_RAY_RESERVATION_ACTIVE")
    if leases:
        blockers.append("FORMAL_GPU_LEASE_ACTIVE")
    if final_access:
        blockers.append("FINAL_PANEL_BOUNDARY_NOT_ZERO")
    if witness.get("formal_work_may_wait_for_canary") is not False:
        blockers.append("FORMAL_PRIORITY_CONTRACT_MISSING")

    observed_gap = not blockers
    handoff_bound = False
    if handoff is not None:
        if handoff.get("schema") != HANDOFF_SCHEMA:
            raise OpportunisticCanaryGuardError("formal-priority handoff schema drift")
        if handoff.get("execution_git_sha") != expected_execution_git_sha:
            raise OpportunisticCanaryGuardError("formal-priority handoff execution drift")
        if handoff.get("topology_id") != expected_topology_id:
            raise OpportunisticCanaryGuardError("formal-priority handoff topology drift")
        if _exact_int_list(handoff.get("candidate_gpu_indices"), "handoff candidate GPUs") != candidates:
            raise OpportunisticCanaryGuardError("formal-priority handoff candidate GPU drift")
        valid_until = handoff.get("valid_until_epoch")
        if not isinstance(valid_until, (int, float)) or float(valid_until) < now:
            blockers.append("FORMAL_PRIORITY_HANDOFF_EXPIRED")
        elif handoff.get("formal_work_may_wait_for_canary") is not False:
            blockers.append("FORMAL_PRIORITY_HANDOFF_ALLOWS_INTERFERENCE")
        elif handoff.get("canary_must_yield_to_formal") is not True:
            blockers.append("FORMAL_PRIORITY_HANDOFF_NOT_YIELDING")
        else:
            handoff_bound = True
    else:
        blockers.append("NONINTERFERENCE_HANDOFF_MISSING")

    report: dict[str, Any] = {
        "schema": DECISION_SCHEMA,
        "status": "OPPORTUNISTIC_GATE_PASS" if observed_gap and handoff_bound and not blockers else "WAIT_FORMAL_PRIORITY",
        "execution_git_sha": expected_execution_git_sha,
        "topology_id": expected_topology_id,
        "candidate_gpu_indices": candidates,
        "forbidden_gpu_indices": sorted(forbidden),
        "witness_age_seconds": age,
        "max_witness_age_seconds": float(max_witness_age_seconds),
        "observed_formal_gap": observed_gap,
        "formal_priority_handoff_bound": handoff_bound,
        "opportunistic_gate_pass": observed_gap and handoff_bound and not blockers,
        "blockers": blockers,
        "gpu_requests": 0,
        "formal_rows_consumed": 0,
        "model_calls": 0,
        "final_panel_access_count": 0,
        "scientific_authority": False,
        "resource_topology_authority": False,
        "canary_execution_authority": False,
        "formal_work_may_wait_for_canary": False,
    }
    report["decision_identity"] = hashlib.sha256(_canonical_bytes(report)).hexdigest()
    return OpportunisticCanaryDecision(report=report)
