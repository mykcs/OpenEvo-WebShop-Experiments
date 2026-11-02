from __future__ import annotations

import copy

import pytest

from scripts.openevo_runtime.operation_records import (
    OperationRecordError,
    build_exact_reuse_candidate_records,
    build_operation_intent,
    build_operation_terminal,
    validate_exact_reuse_candidate,
    validate_operation_intent,
    validate_operation_terminal,
)
from scripts.openevo_runtime.result_codes import OperationResult, OperationStatus, ReasonCode
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
        "operation_type": "runtime.contract-test",
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


def test_operation_intent_reuses_passport_digest_as_idempotency_key() -> None:
    spec = _spec()
    intent = build_operation_intent(spec, admission_binding_identity="a" * 64)
    assert intent["operation_digest"] == dag.operation_digest(spec)
    assert intent["idempotency_key"] == intent["operation_digest"]
    assert intent["execution_generation_id"] == "1" * 64
    validate_operation_intent(intent)


def test_operation_intent_changes_when_admission_binding_changes() -> None:
    spec = _spec()
    first = build_operation_intent(spec, admission_binding_identity="a" * 64)
    second = build_operation_intent(spec, admission_binding_identity="b" * 64)
    assert first["operation_digest"] == second["operation_digest"]
    assert first["intent_id"] != second["intent_id"]


def test_operation_intent_rejects_legacy_spec_without_generation() -> None:
    spec = _spec()
    spec["schema"] = dag.LEGACY_NODE_SCHEMA_ID
    spec["identity_payload"].pop("execution_generation_id")
    with pytest.raises(OperationRecordError, match="execution_generation_id"):
        build_operation_intent(spec, admission_binding_identity="a" * 64)


def test_terminal_binds_typed_result_and_receipt_but_ignores_human_message_in_identity() -> None:
    intent = build_operation_intent(_spec(), admission_binding_identity="a" * 64)
    result = OperationResult(
        status=OperationStatus.SUCCEEDED,
        reason_code=ReasonCode.NONE,
        operation_id=intent["idempotency_key"],
        scientific_state_mutated=False,
        side_effect_materialized=False,
        message="first human explanation",
    )
    first = build_operation_terminal(
        intent, result, terminal_receipt_identity="c" * 64
    )
    changed = result.to_dict()
    changed["message"] = "presentation changed only"
    changed["details"] = {"presentation": "changed"}
    second = build_operation_terminal(
        intent, changed, terminal_receipt_identity="c" * 64
    )
    assert first["terminal_id"] == second["terminal_id"]
    validate_operation_terminal(first)


def test_nonterminal_wait_resource_cannot_be_mislabeled_terminal() -> None:
    intent = build_operation_intent(_spec(), admission_binding_identity="a" * 64)
    result = OperationResult(
        status=OperationStatus.WAITING,
        reason_code=ReasonCode.GPU_RESOURCE_UNAVAILABLE,
        operation_id=intent["idempotency_key"],
        scientific_state_mutated=False,
        side_effect_materialized=False,
    )
    with pytest.raises(OperationRecordError, match="not terminal"):
        build_operation_terminal(intent, result, terminal_receipt_identity="c" * 64)


def test_terminal_requires_result_to_bind_exact_intent_idempotency_key() -> None:
    intent = build_operation_intent(_spec(), admission_binding_identity="a" * 64)
    result = OperationResult(
        status=OperationStatus.BLOCKED,
        reason_code=ReasonCode.AUTHORITY_IDENTITY_DRIFT,
        operation_id="wrong-operation-id",
        scientific_state_mutated=False,
        side_effect_materialized=False,
    )
    with pytest.raises(OperationRecordError, match="frozen idempotency_key"):
        build_operation_terminal(intent, result, terminal_receipt_identity="c" * 64)


def _query_result(*, scientific: bool = False) -> dict:
    side_effect = "FORMAL_ROLLOUT" if scientific else "PURE_ENGINEERING"
    return {
        "status": "REUSE_CANDIDATE",
        "classification": "EXACT_CACHE_HIT",
        "operation_digest": "d" * 64,
        "candidate_count": 1,
        "node_path": (
            "configs/experiment/passport-artifacts/operation-nodes/"
            + "d" * 64
            + "/"
            + "e" * 64
            + ".json"
        ),
        "node_sha256": "e" * 64,
        "side_effect_class": side_effect,
        "execution_generation_id": "1" * 64,
        "outputs": [
            {
                "artifact_type": "test",
                "digest_algorithm": "sha256",
                "digest": "f" * 64,
                "size_bytes": 12,
                "logical_role": "output",
            }
        ],
        "reuse": {
            "default_policy": dag.DEFAULT_REUSE_POLICIES[side_effect],
            "automatic_reuse_permitted": not scientific,
            "operation_allowlisted": not scientific,
            "requires_experiment_policy_review": scientific,
        },
        "producer_attempt": {
            "attempt_id": "attempt-1",
            "source": "LIVE" if scientific else "PILOT",
            "execution_sha": "9" * 40 if scientific else None,
            "created_at": "2026-09-17T03:52:00+08:00",
        },
    }


def test_exact_reuse_candidate_is_advisory_even_when_existing_engineering_policy_allows_reuse() -> None:
    record = build_exact_reuse_candidate_records(_query_result())[0]
    assert record["existing_policy_automatic_reuse_permitted"] is True
    assert record["reuse_authorized_by_candidate_record"] is False
    validate_exact_reuse_candidate(record)


def test_scientific_exact_candidate_preserves_policy_review_requirement_and_zero_authority() -> None:
    record = build_exact_reuse_candidate_records(_query_result(scientific=True))[0]
    assert record["existing_policy_automatic_reuse_permitted"] is False
    assert record["policy_review_required"] is True
    assert record["reuse_authorized_by_candidate_record"] is False


def test_compatible_candidate_cannot_be_promoted_to_exact_reuse_record() -> None:
    result = _query_result()
    result["status"] = "COMPATIBLE_HISTORICAL_CANDIDATE"
    result.pop("classification")
    with pytest.raises(OperationRecordError, match="only exact Passport"):
        build_exact_reuse_candidate_records(result)


def test_reuse_candidate_tampering_fails_closed() -> None:
    record = build_exact_reuse_candidate_records(_query_result())[0]
    tampered = copy.deepcopy(record)
    tampered["reuse_authorized_by_candidate_record"] = True
    with pytest.raises(OperationRecordError, match="advisory"):
        validate_exact_reuse_candidate(tampered)
