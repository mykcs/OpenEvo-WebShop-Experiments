from __future__ import annotations

import copy
import unittest

from scripts.openevo_runtime.event_reconciliation import (
    EventReconciliationError,
    reconcile_runtime_events,
)
from scripts.openevo_runtime.operation_records import (
    build_exact_reuse_candidate_records,
    build_operation_intent,
)
from scripts.openevo_runtime.prefect_outer_bridge import (
    PrefectAction,
    cached_result_decision,
    event_visibility_decision,
    result_decision,
)
from scripts.openevo_runtime.result_codes import OperationResult, OperationStatus, ReasonCode
from scripts.openevo_runtime.runtime_events import build_runtime_event
from scripts.openevo_webshop import experiment_artifact_dag as dag


def _ref(seed: str) -> dict:
    return {
        "kind": "unresolved",
        "repository": None,
        "revision": None,
        "path": None,
        "sha256": seed * 64,
        "evidence_path": None,
        "reason": "unit-test exact identity",
    }


def _spec(side_effect: str = "PURE_ENGINEERING") -> dict:
    scientific = side_effect in dag.SCIENTIFIC_NO_AUTO_REUSE
    return {
        "operation_type": "runtime.prefect-bridge-test",
        "operation_schema_version": "1",
        "side_effect_class": side_effect,
        "identity_payload": {
            "execution_generation_id": "1" * 64,
            "code_identity": _ref("2"),
            "config_identity": _ref("3"),
            "runtime_identity_policy": "NOT_IDENTITY_BEARING",
            "runtime_identity": None,
            "dataset_schedule_identity": _ref("4") if scientific else None,
            "contract_identities": (
                [{"name": "science", "identity": _ref("5")}] if scientific else []
            ),
            "input_artifacts": [],
            "stochastic_identity": ({"seed": 7} if scientific else {}),
        },
    }


def _intent(side_effect: str = "PURE_ENGINEERING") -> dict:
    return build_operation_intent(_spec(side_effect), admission_binding_identity="a" * 64)


def _candidate(intent: dict, *, automatic: bool, review: bool, side_effect: str) -> dict:
    query = {
        "status": "REUSE_CANDIDATE",
        "classification": "EXACT_CACHE_HIT",
        "operation_digest": intent["operation_digest"],
        "candidate_count": 1,
        "node_path": (
            "configs/experiment/passport-artifacts/operation-nodes/"
            + intent["operation_digest"] + "/" + "e" * 64 + ".json"
        ),
        "node_sha256": "e" * 64,
        "side_effect_class": side_effect,
        "execution_generation_id": intent["execution_generation_id"],
        "outputs": [{
            "artifact_type": "test",
            "digest_algorithm": "sha256",
            "digest": "f" * 64,
            "size_bytes": 12,
            "logical_role": "output",
        }],
        "reuse": {
            "default_policy": dag.DEFAULT_REUSE_POLICIES[side_effect],
            "automatic_reuse_permitted": automatic,
            "operation_allowlisted": automatic,
            "requires_experiment_policy_review": review,
        },
        "producer_attempt": {"attempt_id": "attempt-1", "source": "PILOT"},
    }
    return build_exact_reuse_candidate_records(query)[0]


class PrefectBridgeTests(unittest.TestCase):
    def test_wait_resource_is_progress_not_framework_failure(self) -> None:
        intent = _intent()
        result = OperationResult(
            status=OperationStatus.WAITING,
            reason_code=ReasonCode.GPU_RESOURCE_UNAVAILABLE,
            operation_id=intent["idempotency_key"],
            scientific_state_mutated=False,
            side_effect_materialized=False,
        )
        decision = result_decision(intent, result)
        self.assertEqual(decision.action, PrefectAction.WAIT_RESOURCE)
        self.assertFalse(decision.framework_retry_allowed)
        self.assertFalse(decision.scientific_retry_authorized)

    def test_only_proved_no_side_effect_engineering_failure_is_framework_retryable(self) -> None:
        intent = _intent()
        result = OperationResult(
            status=OperationStatus.BLOCKED,
            reason_code=ReasonCode.ENGINEERING_FAILURE_NO_SIDE_EFFECT,
            operation_id=intent["idempotency_key"],
            scientific_state_mutated=False,
            side_effect_materialized=False,
        )
        decision = result_decision(intent, result)
        self.assertEqual(decision.action, PrefectAction.TRANSIENT_INFRA_RETRY)
        self.assertTrue(decision.framework_retry_allowed)
        self.assertFalse(decision.scientific_retry_authorized)

    def test_materialized_ack_loss_requires_readback_not_prefect_retry(self) -> None:
        intent = _intent()
        result = OperationResult(
            status=OperationStatus.BLOCKED,
            reason_code=ReasonCode.ACKNOWLEDGEMENT_MISSING_AFTER_MATERIALIZATION,
            operation_id=intent["idempotency_key"],
            scientific_state_mutated=False,
            side_effect_materialized=True,
        )
        decision = result_decision(intent, result)
        self.assertEqual(decision.action, PrefectAction.READBACK_REQUIRED)
        self.assertFalse(decision.framework_retry_allowed)

    def test_prefect_cache_cannot_cross_operation_identity(self) -> None:
        intent = _intent()
        candidate = _candidate(intent, automatic=True, review=False, side_effect="PURE_ENGINEERING")
        decision = cached_result_decision(
            intent,
            cached_operation_digest="b" * 64,
            reuse_candidate=candidate,
        )
        self.assertEqual(decision.action, PrefectAction.BLOCK)

    def test_exact_pure_engineering_cache_still_requires_passport_candidate(self) -> None:
        intent = _intent()
        decision = cached_result_decision(
            intent,
            cached_operation_digest=intent["operation_digest"],
            reuse_candidate=None,
        )
        self.assertEqual(decision.action, PrefectAction.BLOCK)

    def test_exact_allowlisted_pure_engineering_cache_can_be_accepted(self) -> None:
        intent = _intent()
        candidate = _candidate(intent, automatic=True, review=False, side_effect="PURE_ENGINEERING")
        decision = cached_result_decision(
            intent,
            cached_operation_digest=intent["operation_digest"],
            reuse_candidate=candidate,
        )
        self.assertEqual(decision.action, PrefectAction.ACCEPT_ENGINEERING_CACHE)
        self.assertFalse(decision.reuse_authorized_by_framework)

    def test_scientific_cache_never_becomes_prefect_reuse_authority(self) -> None:
        intent = _intent("FORMAL_ROLLOUT")
        candidate = _candidate(intent, automatic=False, review=True, side_effect="FORMAL_ROLLOUT")
        decision = cached_result_decision(
            intent,
            cached_operation_digest=intent["operation_digest"],
            reuse_candidate=candidate,
        )
        self.assertEqual(decision.action, PrefectAction.BLOCK)
        self.assertFalse(decision.reuse_authorized_by_framework)

    def test_valid_event_is_operator_visibility_only(self) -> None:
        intent = _intent()
        event = build_runtime_event(
            event_type="OPERATION_READY",
            subject_id=intent["intent_id"],
            source_receipt_identity="c" * 64,
            payload_identity="d" * 64,
        )
        decision = event_visibility_decision(intent, event)
        self.assertEqual(decision.action, PrefectAction.OBSERVE_EVENT)
        self.assertFalse(decision.scientific_retry_authorized)


class EventReconciliationTests(unittest.TestCase):
    def _receipts(self) -> list[dict]:
        return [
            {
                "event_type": "ROUND_SEALED",
                "subject_id": "1" * 64,
                "source_receipt_identity": "2" * 64,
                "payload_identity": "3" * 64,
            },
            {
                "event_type": "PAIR_AUDIT",
                "subject_id": "4" * 64,
                "source_receipt_identity": "5" * 64,
                "payload_identity": "6" * 64,
            },
        ]

    def test_missing_event_is_reconstructed_from_receipt_truth(self) -> None:
        reconciled = reconcile_runtime_events(self._receipts(), [])
        self.assertEqual(reconciled.reconstructed_missing, 2)
        self.assertEqual(len(reconciled.events), 2)
        self.assertEqual(reconciled.authority_source, "IMMUTABLE_OPEN_EVO_RECEIPTS")
        self.assertFalse(reconciled.framework_history_authority)

    def test_identical_duplicate_delivery_is_ignored(self) -> None:
        receipt = self._receipts()[0]
        event = build_runtime_event(**receipt)
        reconciled = reconcile_runtime_events([receipt], [event, copy.deepcopy(event)])
        self.assertEqual(reconciled.ignored_duplicates, 1)
        self.assertEqual(reconciled.reconstructed_missing, 0)

    def test_conflicting_delivery_fails_closed(self) -> None:
        receipt = self._receipts()[0]
        event = build_runtime_event(**receipt)
        bad = build_runtime_event(
            event_type=receipt["event_type"],
            subject_id=receipt["subject_id"],
            source_receipt_identity="9" * 64,
            payload_identity=receipt["payload_identity"],
        )
        with self.assertRaises(EventReconciliationError):
            reconcile_runtime_events([receipt], [event, bad])


if __name__ == "__main__":
    unittest.main()
