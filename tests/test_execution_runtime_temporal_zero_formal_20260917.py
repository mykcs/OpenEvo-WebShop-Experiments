from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

temporalio = pytest.importorskip("temporalio")
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.exceptions import ApplicationError
from temporalio.worker import Worker

from scripts.openevo_runtime.operation_records import build_operation_intent
from scripts.openevo_runtime.result_codes import OperationResult, OperationStatus, ReasonCode
from scripts.openevo_runtime.temporal_zero_formal import (
    ACTIVITY_KINDS,
    ArmWorkflow,
    PairAuditWorkflow,
    REGISTERED_ACTIVITIES,
    REGISTERED_WORKFLOWS,
)

ROOT = Path(__file__).resolve().parents[1]


def _ref(seed: str) -> dict:
    return {
        "kind": "unresolved", "repository": None, "revision": None, "path": None,
        "sha256": seed * 64, "evidence_path": None, "reason": "zero-formal Temporal test",
    }

def _spec(kind: str) -> dict:
    return {
        "operation_type": f"runtime.temporal-zero-formal.{kind}",
        "operation_schema_version": "1",
        "side_effect_class": "PURE_ENGINEERING",
        "identity_payload": {
            "execution_generation_id": "1" * 64,
            "code_identity": _ref("2"),
            "config_identity": _ref("3"),
            "runtime_identity_policy": "NOT_IDENTITY_BEARING",
            "runtime_identity": None,
            "dataset_schedule_identity": None,
            "contract_identities": [],
            "input_artifacts": [],
            "stochastic_identity": {},
        },
    }


def _request(kind: str, receipt_seed: str) -> dict:
    intent = build_operation_intent(_spec(kind), admission_binding_identity="a" * 64)
    result = OperationResult(
        status=OperationStatus.SUCCEEDED,
        reason_code=ReasonCode.NONE,
        operation_id=intent["idempotency_key"],
        scientific_state_mutated=False,
        side_effect_materialized=False,
        message="synthetic zero-formal activity",
    )
    return {
        "activity_kind": kind,
        "intent": intent,
        "result": result.to_dict(),
        "terminal_receipt_identity": receipt_seed * 64,
        "synthetic_zero_formal": True,
        "formal_rows_consumed": 0,
        "final_panel_access": 0,
    }

def _plan() -> dict:
    seeds = dict(zip(ACTIVITY_KINDS, "cdefb", strict=True))
    return {
        "arm": "SYNTHETIC_OFF",
        "synthetic_zero_formal": True,
        "formal_rows_consumed": 0,
        "final_panel_access": 0,
        "operations": [_request(kind, seeds[kind]) for kind in ACTIVITY_KINDS],
    }


async def _run_arm_workflow() -> None:
    plan = _plan()
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="openevo-zero-formal-arm",
            workflows=REGISTERED_WORKFLOWS,
            activities=REGISTERED_ACTIVITIES,
        ):
            result = await env.client.execute_workflow(
                ArmWorkflow.run,
                plan,
                id="openevo-zero-formal-arm-1",
                task_queue="openevo-zero-formal-arm",
            )
    assert result["formal_rows_consumed"] == 0
    assert result["final_panel_access"] == 0
    assert result["scientific_authority_granted"] is False
    assert len(result["terminals"]) == len(ACTIVITY_KINDS)
    expected_ids = [row["intent"]["idempotency_key"] for row in plan["operations"]]
    assert [row["activity_id"] for row in result["terminals"]] == expected_ids
    assert [row["terminal"]["operation_digest"] for row in result["terminals"]] == expected_ids


def test_arm_workflow_executes_all_zero_formal_activity_shapes_with_stable_ids() -> None:
    asyncio.run(_run_arm_workflow())


async def _run_pair_audit_workflow() -> None:
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="openevo-zero-formal-pair",
            workflows=REGISTERED_WORKFLOWS,
            activities=REGISTERED_ACTIVITIES,
        ):
            handle = await env.client.start_workflow(
                PairAuditWorkflow.run,
                {"synthetic_zero_formal": True, "expected_receipts": 2},
                id="openevo-zero-formal-pair-1",
                task_queue="openevo-zero-formal-pair",
            )
            assert await handle.query(PairAuditWorkflow.receipt_count) == 0
            first = {"pair_receipt_identity": "1" * 64, "pair_round": 7}
            second = {"pair_receipt_identity": "2" * 64, "pair_round": 9}
            await handle.signal(PairAuditWorkflow.accept_pair_receipt, first)
            await handle.signal(PairAuditWorkflow.accept_pair_receipt, first)
            assert await handle.query(PairAuditWorkflow.receipt_count) == 1
            await handle.signal(PairAuditWorkflow.accept_pair_receipt, second)
            result = await handle.result()
    assert result["admission_authority"] is False
    assert result["blocks_arm_progression"] is False
    assert result["formal_rows_consumed"] == 0
    assert result["final_panel_access"] == 0
    assert [row["pair_round"] for row in result["pair_receipts"]] == [7, 9]


def test_pair_audit_workflow_is_async_audit_only_consumer() -> None:
    asyncio.run(_run_pair_audit_workflow())


def test_temporal_prototype_has_no_project_runtime_dependency() -> None:
    manifests = [ROOT / "requirements-ci.txt", ROOT / "requirements-smoke.txt"]
    for path in manifests:
        assert "temporalio" not in path.read_text(encoding="utf-8").lower()


async def _run_independent_arm_lag() -> None:
    fast = _plan()
    slow = _plan()
    slow["arm"] = "SYNTHETIC_ON"
    async with await WorkflowEnvironment.start_time_skipping() as env:
        slow_handle = await env.client.start_workflow(
            ArmWorkflow.run, slow, id="openevo-zero-formal-slow-arm",
            task_queue="openevo-zero-formal-slow-arm-queue",
        )
        # No worker serves the slow arm yet. The fast arm must still complete.
        async with Worker(
            env.client,
            task_queue="openevo-zero-formal-fast-arm-queue",
            workflows=REGISTERED_WORKFLOWS,
            activities=REGISTERED_ACTIVITIES,
        ):
            fast_result = await env.client.execute_workflow(
                ArmWorkflow.run, fast, id="openevo-zero-formal-fast-arm",
                task_queue="openevo-zero-formal-fast-arm-queue",
            )
            assert fast_result["arm"] == "SYNTHETIC_OFF"
            assert fast_result["terminals"]
        # Only after fast is sealed do we bring up the lagging arm's worker.
        async with Worker(
            env.client,
            task_queue="openevo-zero-formal-slow-arm-queue",
            workflows=REGISTERED_WORKFLOWS,
            activities=REGISTERED_ACTIVITIES,
        ):
            slow_result = await slow_handle.result()
            assert slow_result["arm"] == "SYNTHETIC_ON"
            assert slow_result["terminals"]


def test_independent_arm_lag_does_not_gate_peer_progress() -> None:
    asyncio.run(_run_independent_arm_lag())


async def _run_wait_resource_without_failure() -> None:
    plan = _plan()
    plan["operations"] = [plan["operations"][0]]
    op_id = plan["operations"][0]["intent"]["idempotency_key"]
    plan["operations"][0]["synthetic_wait_resource"] = True
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="openevo-zero-formal-resource-wait",
            workflows=REGISTERED_WORKFLOWS,
            activities=REGISTERED_ACTIVITIES,
        ):
            handle = await env.client.start_workflow(
                ArmWorkflow.run, plan, id="openevo-zero-formal-resource-wait-1",
                task_queue="openevo-zero-formal-resource-wait",
            )
            state = await handle.query(ArmWorkflow.runtime_state)
            assert state == {"status": "WAIT_RESOURCE", "operation_id": op_id}
            await handle.signal(ArmWorkflow.resource_available, op_id)
            result = await handle.result()
            assert result["formal_rows_consumed"] == 0
            assert result["final_panel_access"] == 0
            assert len(result["terminals"]) == 1
            assert result["terminals"][0]["activity_id"] == op_id


def test_wait_resource_is_durable_wait_not_workflow_failure() -> None:
    asyncio.run(_run_wait_resource_without_failure())


WORKER_INTERRUPTION_SCRIPT = r"""
import asyncio
import json
import sys
from datetime import timedelta
from pathlib import Path
from temporalio import activity
from temporalio.client import Client
from temporalio.worker import Worker
from scripts.openevo_runtime.temporal_zero_formal import ArmWorkflow, _zero_formal_activity

log = Path(sys.argv[3])
queue = sys.argv[4]

def record(request):
    op_id = request["intent"]["idempotency_key"]
    with log.open("a", encoding="utf-8") as handle:
        handle.write(op_id + "\n")

@activity.defn(name="openevo_zero_formal_rollout_like")
async def rollout(request):
    record(request)
    return _zero_formal_activity(request, "rollout_like")

@activity.defn(name="openevo_zero_formal_transition_like")
async def transition(request):
    record(request)
    return _zero_formal_activity(request, "transition_like")

async def main():
    client = await Client.connect(sys.argv[1], namespace=sys.argv[2])
    worker = Worker(
        client,
        task_queue=queue,
        workflows=[ArmWorkflow],
        activities=[rollout, transition],
        max_cached_workflows=0,
        sticky_queue_schedule_to_start_timeout=timedelta(seconds=1),
    )
    await worker.run()

asyncio.run(main())
"""


async def _run_worker_interruption_recovery(tmp_path: Path) -> None:
    plan = _plan()
    plan["operations"] = plan["operations"][:2]
    first_id = plan["operations"][0]["intent"]["idempotency_key"]
    second_id = plan["operations"][1]["intent"]["idempotency_key"]
    plan["operations"][1]["synthetic_wait_resource"] = True
    log_path = tmp_path / "activity-calls.log"
    queue = "openevo-zero-formal-worker-recovery"

    async with await WorkflowEnvironment.start_time_skipping() as env:
        target = env.client.service_client.config.target_host
        namespace = env.client.namespace

        async def start_worker():
            return await asyncio.create_subprocess_exec(
                sys.executable, "-c", WORKER_INTERRUPTION_SCRIPT,
                target, namespace, str(log_path), queue,
                cwd=str(ROOT), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )

        worker1 = await start_worker()
        await asyncio.sleep(0.25)
        handle = await env.client.start_workflow(
            ArmWorkflow.run, plan, id="openevo-zero-formal-worker-recovery-1",
            task_queue=queue,
        )
        state = None
        for _ in range(100):
            try:
                state = await asyncio.wait_for(handle.query(ArmWorkflow.runtime_state), timeout=1)
            except TimeoutError:
                state = None
            if state == {"status": "WAIT_RESOURCE", "operation_id": second_id}:
                break
            await asyncio.sleep(0.02)
        assert state == {"status": "WAIT_RESOURCE", "operation_id": second_id}
        assert log_path.read_text(encoding="utf-8").splitlines() == [first_id]

        worker1.kill()
        await worker1.wait()
        worker2 = await start_worker()
        try:
            await asyncio.sleep(0.25)
            await handle.signal(ArmWorkflow.resource_available, second_id)
            result = await asyncio.wait_for(handle.result(), timeout=10)
        finally:
            worker2.kill()
            await worker2.wait()

    assert result["formal_rows_consumed"] == 0
    assert result["final_panel_access"] == 0
    assert [row["activity_id"] for row in result["terminals"]] == [first_id, second_id]
    assert log_path.read_text(encoding="utf-8").splitlines() == [first_id, second_id]


def test_worker_interruption_resumes_without_duplicate_openevo_operations(tmp_path: Path) -> None:
    asyncio.run(_run_worker_interruption_recovery(tmp_path))


async def _run_ack_loss_readback(*, ambiguous: bool, tmp_path: Path) -> tuple[dict, int]:
    from scripts.openevo_runtime.temporal_zero_formal import _zero_formal_activity

    plan = _plan()
    plan["operations"] = [plan["operations"][0]]
    request = plan["operations"][0]
    request["synthetic_ack_loss_after_materialization"] = True
    request["result"]["side_effect_materialized"] = True
    op_id = request["intent"]["idempotency_key"]
    receipt_path = tmp_path / f"materialized-terminal-{int(ambiguous)}.json"
    calls = 0

    @activity.defn(name="openevo_zero_formal_rollout_like")
    async def lost_ack_activity(payload: dict) -> dict:
        nonlocal calls
        calls += 1
        response = _zero_formal_activity(payload, "rollout_like")
        receipt_path.write_text(json.dumps(response, sort_keys=True) + "\n", encoding="utf-8")
        raise ApplicationError("synthetic lost acknowledgement", non_retryable=True)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="openevo-zero-formal-ack-loss",
            workflows=[ArmWorkflow],
            activities=[lost_ack_activity],
        ):
            handle = await env.client.start_workflow(
                ArmWorkflow.run,
                plan,
                id=f"openevo-zero-formal-ack-loss-{int(ambiguous)}",
                task_queue="openevo-zero-formal-ack-loss",
            )
            state = None
            for _ in range(100):
                state = await handle.query(ArmWorkflow.runtime_state)
                if state == {"status": "READBACK_REQUIRED", "operation_id": op_id}:
                    break
                await asyncio.sleep(0.01)
            assert state == {"status": "READBACK_REQUIRED", "operation_id": op_id}
            assert calls == 1
            materialized = json.loads(receipt_path.read_text(encoding="utf-8"))
            assert materialized["terminal"]["operation_digest"] == op_id

            if ambiguous:
                await handle.signal(ArmWorkflow.terminal_readback_ambiguous, op_id)
            else:
                await handle.signal(
                    ArmWorkflow.accept_terminal_readback,
                    materialized,
                )
            result = await handle.result()
    return result, calls


def test_ack_loss_adopts_exactly_one_materialized_terminal(tmp_path: Path) -> None:
    result, calls = asyncio.run(_run_ack_loss_readback(ambiguous=False, tmp_path=tmp_path))
    assert calls == 1
    assert result["formal_rows_consumed"] == 0
    assert result["final_panel_access"] == 0
    assert len(result["terminals"]) == 1
    assert result["terminals"][0]["recovered_from_ack_loss"] is True


def test_integrity_ambiguity_blocks_instead_of_retrying(tmp_path: Path) -> None:
    result, calls = asyncio.run(_run_ack_loss_readback(ambiguous=True, tmp_path=tmp_path))
    assert calls == 1
    assert result["workflow_status"] == "BLOCKED_INTEGRITY"
    assert result["terminals"] == []
    assert result["scientific_retry_authorized"] is False
