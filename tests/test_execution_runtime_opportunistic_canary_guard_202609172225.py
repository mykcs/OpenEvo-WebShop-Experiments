import pytest

from scripts.openevo_webshop.opportunistic_canary_guard import (
    HANDOFF_SCHEMA,
    WITNESS_SCHEMA,
    OpportunisticCanaryGuardError,
    evaluate_opportunistic_canary_gap,
)

EXECUTION = "4" * 40
TOPOLOGY = "5" * 64
NOW = 1_000.0


def witness(**overrides):
    doc = {
        "schema": WITNESS_SCHEMA,
        "execution_git_sha": EXECUTION,
        "topology_id": TOPOLOGY,
        "candidate_gpu_indices": [4],
        "observed_at_epoch": NOW - 0.1,
        "scheduler_status": "IDLE",
        "pending_formal_gpu_requests": 0,
        "active_formal_reservations": 0,
        "active_formal_gpu_leases": 0,
        "final_panel_access_count": 0,
        "formal_work_may_wait_for_canary": False,
    }
    doc.update(overrides)
    return doc


def handoff(**overrides):
    doc = {
        "schema": HANDOFF_SCHEMA,
        "execution_git_sha": EXECUTION,
        "topology_id": TOPOLOGY,
        "candidate_gpu_indices": [4],
        "valid_until_epoch": NOW + 1.0,
        "formal_work_may_wait_for_canary": False,
        "canary_must_yield_to_formal": True,
    }
    doc.update(overrides)
    return doc


def decide(doc=None, token=None):
    return evaluate_opportunistic_canary_gap(
        witness() if doc is None else doc,
        expected_execution_git_sha=EXECUTION,
        expected_topology_id=TOPOLOGY,
        candidate_gpu_indices=[4],
        forbidden_gpu_indices=[0, 1, 2, 3],
        handoff=token,
        max_witness_age_seconds=1.0,
        now_epoch=NOW,
    )


def test_idle_sample_alone_never_grants_canary_execution() -> None:
    result = decide()
    assert result.observed_gap is True
    assert result.gate_pass is False
    assert result.report["status"] == "WAIT_FORMAL_PRIORITY"
    assert result.report["blockers"] == ["NONINTERFERENCE_HANDOFF_MISSING"]
    assert result.report["gpu_requests"] == 0
    assert result.report["canary_execution_authority"] is False


def test_exact_formal_priority_handoff_only_passes_opportunistic_gate() -> None:
    result = decide(token=handoff())
    assert result.observed_gap is True
    assert result.gate_pass is True
    assert result.report["status"] == "OPPORTUNISTIC_GATE_PASS"
    assert result.report["blockers"] == []
    assert result.report["scientific_authority"] is False
    assert result.report["resource_topology_authority"] is False
    assert result.report["canary_execution_authority"] is False


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"scheduler_status": "ACTIVE"}, "FORMAL_SCHEDULER_NOT_IDLE"),
        ({"pending_formal_gpu_requests": 1}, "FORMAL_GPU_REQUEST_PENDING"),
        ({"active_formal_reservations": 1}, "FORMAL_RAY_RESERVATION_ACTIVE"),
        ({"active_formal_gpu_leases": 1}, "FORMAL_GPU_LEASE_ACTIVE"),
        ({"observed_at_epoch": NOW - 2.0}, "FORMAL_PRIORITY_WITNESS_STALE"),
    ],
)
def test_any_formal_demand_or_stale_witness_closes_window(change, reason) -> None:
    result = decide(witness(**change), token=handoff())
    assert result.gate_pass is False
    assert reason in result.report["blockers"]


def test_forbidden_student_gpu_cannot_enter_candidate_set() -> None:
    with pytest.raises(OpportunisticCanaryGuardError, match="forbidden GPUs"):
        evaluate_opportunistic_canary_gap(
            witness(candidate_gpu_indices=[0]),
            expected_execution_git_sha=EXECUTION,
            expected_topology_id=TOPOLOGY,
            candidate_gpu_indices=[0],
            forbidden_gpu_indices=[0, 1, 2, 3],
            now_epoch=NOW,
        )


def test_execution_or_topology_drift_fails_closed() -> None:
    with pytest.raises(OpportunisticCanaryGuardError, match="execution identity drift"):
        evaluate_opportunistic_canary_gap(
            witness(execution_git_sha="6" * 40),
            expected_execution_git_sha=EXECUTION,
            expected_topology_id=TOPOLOGY,
            candidate_gpu_indices=[4],
            now_epoch=NOW,
        )
    with pytest.raises(OpportunisticCanaryGuardError, match="topology identity drift"):
        evaluate_opportunistic_canary_gap(
            witness(topology_id="7" * 64),
            expected_execution_git_sha=EXECUTION,
            expected_topology_id=TOPOLOGY,
            candidate_gpu_indices=[4],
            now_epoch=NOW,
        )
