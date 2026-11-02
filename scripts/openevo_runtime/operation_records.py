"""Framework-neutral operation intent/terminal/reuse-candidate records.

These records bind existing OpenEVO/Passport identities. They are orchestration
contracts, not scientific, launch, retry, topology, or reuse authority.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from scripts.openevo_runtime.result_codes import (
    OperationResult,
    OperationStatus,
    validate_operation_result,
)
from scripts.openevo_webshop import experiment_artifact_dag as dag
from scripts.openevo_webshop.experiment_passport import (
    PassportError,
    assert_no_credentials,
    canonical_bytes,
    validate_schema,
)

ROOT = Path(__file__).resolve().parents[2]
INTENT_SCHEMA_ID = "openevo.operation-intent.202609170352"
TERMINAL_SCHEMA_ID = "openevo.operation-terminal.202609170352"
REUSE_CANDIDATE_SCHEMA_ID = "openevo.exact-reuse-candidate.202609170352"
INTENT_SCHEMA_PATH = ROOT / "schemas/operation-intent-202609170352.schema.json"
TERMINAL_SCHEMA_PATH = ROOT / "schemas/operation-terminal-202609170352.schema.json"
REUSE_CANDIDATE_SCHEMA_PATH = ROOT / "schemas/exact-reuse-candidate-202609170352.schema.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
TERMINAL_STATUSES = frozenset({
    OperationStatus.BLOCKED,
    OperationStatus.SUCCEEDED,
    OperationStatus.NOOP,
    OperationStatus.SEALED,
    OperationStatus.FAILED,
})


class OperationRecordError(ValueError):
    """Operation record violates the framework-neutral runtime contract."""


def _schema(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OperationRecordError(f"schema is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise OperationRecordError(f"schema must be an object: {path.name}")
    return value


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise OperationRecordError(f"{label} must be a SHA-256 identity")
    return value


def _validate_schema(payload: Mapping[str, Any], path: Path) -> None:
    try:
        validate_schema(dict(payload), _schema(path))
    except PassportError as exc:
        raise OperationRecordError(str(exc)) from exc


def intent_identity_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "schema",
        "operation_digest",
        "idempotency_key",
        "execution_generation_id",
        "operation_type",
        "side_effect_class",
        "admission_binding_identity",
    )
    missing = [field for field in fields if field not in payload]
    if missing:
        raise OperationRecordError(f"missing intent identity field: {missing[0]}")
    projection = {field: payload[field] for field in fields}
    if projection["schema"] != INTENT_SCHEMA_ID:
        raise OperationRecordError(f"schema must equal {INTENT_SCHEMA_ID}")
    operation_digest = _require_sha(projection["operation_digest"], "operation_digest")
    if projection["idempotency_key"] != operation_digest:
        raise OperationRecordError("idempotency_key must equal the Passport operation_digest")
    _require_sha(projection["execution_generation_id"], "execution_generation_id")
    _require_sha(projection["admission_binding_identity"], "admission_binding_identity")
    if not isinstance(projection["operation_type"], str) or not projection["operation_type"]:
        raise OperationRecordError("operation_type must be non-empty")
    if projection["side_effect_class"] not in dag.SIDE_EFFECT_CLASSES:
        raise OperationRecordError("side_effect_class is not recognized")
    assert_no_credentials(projection)
    canonical_bytes(projection)
    return projection


def build_operation_intent(
    operation_spec: Mapping[str, Any], *, admission_binding_identity: str,
) -> dict[str, Any]:
    """Freeze one already-admissible operation identity for orchestration.

    ``admission_binding_identity`` references the external OpenEVO/Passport/release
    authority that admitted the operation. Its presence is not evaluated here and
    does not grant permission by itself.
    """
    identity = dag.operation_identity_payload(operation_spec)
    generation_id = identity.get("execution_generation_id")
    _require_sha(generation_id, "execution_generation_id")
    operation_digest = dag.operation_digest(operation_spec)
    projection = {
        "schema": INTENT_SCHEMA_ID,
        "operation_digest": operation_digest,
        "idempotency_key": operation_digest,
        "execution_generation_id": generation_id,
        "operation_type": operation_spec.get("operation_type"),
        "side_effect_class": operation_spec.get("side_effect_class"),
        "admission_binding_identity": admission_binding_identity,
    }
    result = {**projection, "intent_id": _digest(projection)}
    validate_operation_intent(result)
    return result


def validate_operation_intent(payload: Mapping[str, Any]) -> None:
    _validate_schema(payload, INTENT_SCHEMA_PATH)
    projection = intent_identity_projection(payload)
    if payload.get("intent_id") != _digest(projection):
        raise OperationRecordError("intent_id does not match canonical identity projection")


def _terminal_result_projection(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": result["status"],
        "reason_code": result["reason_code"],
        "retry_class": result["retry_class"],
        "scientific_state_mutated": result["scientific_state_mutated"],
        "side_effect_materialized": result["side_effect_materialized"],
    }


def terminal_identity_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "schema",
        "intent_id",
        "operation_digest",
        "idempotency_key",
        "execution_generation_id",
        "terminal_receipt_identity",
        "result",
    )
    missing = [field for field in fields if field not in payload]
    if missing:
        raise OperationRecordError(f"missing terminal field: {missing[0]}")
    result = payload["result"]
    if not isinstance(result, Mapping):
        raise OperationRecordError("result must be an object")
    validate_operation_result(result)
    status = OperationStatus(result["status"])
    if status not in TERMINAL_STATUSES:
        raise OperationRecordError(f"status={status.value} is not terminal")
    operation_digest = _require_sha(payload["operation_digest"], "operation_digest")
    if payload["idempotency_key"] != operation_digest:
        raise OperationRecordError("terminal idempotency_key must equal operation_digest")
    if result["operation_id"] != payload["idempotency_key"]:
        raise OperationRecordError("OperationResult.operation_id must equal the frozen idempotency_key")
    projection = {
        "schema": payload["schema"],
        "intent_id": _require_sha(payload["intent_id"], "intent_id"),
        "operation_digest": operation_digest,
        "idempotency_key": payload["idempotency_key"],
        "execution_generation_id": _require_sha(
            payload["execution_generation_id"], "execution_generation_id"
        ),
        "terminal_receipt_identity": _require_sha(
            payload["terminal_receipt_identity"], "terminal_receipt_identity"
        ),
        "result": _terminal_result_projection(result),
    }
    if projection["schema"] != TERMINAL_SCHEMA_ID:
        raise OperationRecordError(f"schema must equal {TERMINAL_SCHEMA_ID}")
    assert_no_credentials(projection)
    canonical_bytes(projection)
    return projection


def build_operation_terminal(
    intent: Mapping[str, Any],
    result: OperationResult | Mapping[str, Any],
    *,
    terminal_receipt_identity: str,
) -> dict[str, Any]:
    """Bind one durable machine terminal to one frozen intent.

    The terminal receipt identity is external durable truth. A terminal record never
    turns process death, framework retry, or a workflow status into scientific retry.
    """
    validate_operation_intent(intent)
    result_payload = result.to_dict() if isinstance(result, OperationResult) else dict(result)
    validate_operation_result(result_payload)
    payload = {
        "schema": TERMINAL_SCHEMA_ID,
        "intent_id": intent["intent_id"],
        "operation_digest": intent["operation_digest"],
        "idempotency_key": intent["idempotency_key"],
        "execution_generation_id": intent["execution_generation_id"],
        "terminal_receipt_identity": terminal_receipt_identity,
        "result": result_payload,
    }
    projection = terminal_identity_projection(payload)
    terminal = {**payload, "terminal_id": _digest(projection)}
    validate_operation_terminal(terminal)
    return terminal


def validate_operation_terminal(payload: Mapping[str, Any]) -> None:
    _validate_schema(payload, TERMINAL_SCHEMA_PATH)
    projection = terminal_identity_projection(payload)
    if payload.get("terminal_id") != _digest(projection):
        raise OperationRecordError("terminal_id does not match canonical identity projection")


def _validate_node_path(path: Any) -> str:
    if not isinstance(path, str) or not path or "\\" in path:
        raise OperationRecordError("node_path must be one repository-relative POSIX path")
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts:
        raise OperationRecordError("node_path must not escape the repository")
    return path


def reuse_candidate_identity_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "schema",
        "requested_operation_digest",
        "candidate_operation_digest",
        "execution_generation_id",
        "node_path",
        "node_sha256",
        "side_effect_class",
        "output_artifacts",
        "producer_attempt",
        "reuse_policy",
        "existing_policy_automatic_reuse_permitted",
        "policy_review_required",
        "reuse_authorized_by_candidate_record",
    )
    missing = [field for field in fields if field not in payload]
    if missing:
        raise OperationRecordError(f"missing reuse-candidate field: {missing[0]}")
    requested = _require_sha(
        payload["requested_operation_digest"], "requested_operation_digest"
    )
    candidate = _require_sha(
        payload["candidate_operation_digest"], "candidate_operation_digest"
    )
    if candidate != requested:
        raise OperationRecordError(
            "exact reuse candidate digest must equal requested operation digest"
        )
    if payload["reuse_authorized_by_candidate_record"] is not False:
        raise OperationRecordError(
            "reuse candidate records are advisory and cannot authorize reuse"
        )
    if type(payload["existing_policy_automatic_reuse_permitted"]) is not bool:
        raise OperationRecordError(
            "existing_policy_automatic_reuse_permitted must be boolean"
        )
    if type(payload["policy_review_required"]) is not bool:
        raise OperationRecordError("policy_review_required must be boolean")
    projection = {field: payload[field] for field in fields}
    projection["execution_generation_id"] = _require_sha(
        payload["execution_generation_id"], "execution_generation_id"
    )
    projection["node_path"] = _validate_node_path(payload["node_path"])
    projection["node_sha256"] = _require_sha(payload["node_sha256"], "node_sha256")
    if projection["side_effect_class"] not in dag.SIDE_EFFECT_CLASSES:
        raise OperationRecordError("side_effect_class is not recognized")
    if not isinstance(projection["output_artifacts"], list) or not projection["output_artifacts"]:
        raise OperationRecordError("exact reuse candidate requires output_artifacts")
    for index, artifact in enumerate(projection["output_artifacts"]):
        if not isinstance(artifact, Mapping):
            raise OperationRecordError(f"output_artifacts[{index}] must be an object")
        dag._validate_artifact(artifact, label=f"output_artifacts[{index}]")
    reuse = projection["reuse_policy"]
    if not isinstance(reuse, Mapping):
        raise OperationRecordError("reuse_policy must be an object")
    if reuse.get("automatic_reuse_permitted") is not projection[
        "existing_policy_automatic_reuse_permitted"
    ]:
        raise OperationRecordError(
            "reuse policy automatic permission disagrees with candidate projection"
        )
    if reuse.get("requires_experiment_policy_review") is not projection[
        "policy_review_required"
    ]:
        raise OperationRecordError(
            "reuse policy review flag disagrees with candidate projection"
        )
    assert_no_credentials(projection)
    canonical_bytes(projection)
    return projection


def _candidate_rows(query_result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if (
        query_result.get("status") != "REUSE_CANDIDATE"
        or query_result.get("classification") != "EXACT_CACHE_HIT"
    ):
        raise OperationRecordError(
            "only exact Passport REUSE_CANDIDATE results may be normalized"
        )
    count = query_result.get("candidate_count")
    if type(count) is not int or count < 1:
        raise OperationRecordError("exact reuse query must report candidate_count >= 1")
    if count == 1:
        return [query_result]
    rows = query_result.get("candidates")
    if not isinstance(rows, list) or len(rows) != count:
        raise OperationRecordError(
            "multiple exact candidates require one row per candidate"
        )
    return rows


def build_exact_reuse_candidate_records(
    query_result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Normalize exact #488 DAG hits into advisory, content-addressed records.

    Compatibility/similarity candidates are deliberately rejected here. Existing
    Passport reuse policy is preserved verbatim; this record itself always grants
    zero reuse authority.
    """
    requested_digest = _require_sha(
        query_result.get("operation_digest"), "operation_digest"
    )
    records: list[dict[str, Any]] = []
    for row in _candidate_rows(query_result):
        reuse = row.get("reuse")
        if not isinstance(reuse, Mapping):
            raise OperationRecordError("candidate reuse projection is missing")
        payload = {
            "schema": REUSE_CANDIDATE_SCHEMA_ID,
            "requested_operation_digest": requested_digest,
            "candidate_operation_digest": requested_digest,
            "execution_generation_id": row.get("execution_generation_id"),
            "node_path": row.get("node_path"),
            "node_sha256": row.get("node_sha256"),
            "side_effect_class": row.get("side_effect_class"),
            "output_artifacts": [dict(item) for item in (row.get("outputs") or [])],
            "producer_attempt": dict(row.get("producer_attempt") or {}),
            "reuse_policy": dict(reuse),
            "existing_policy_automatic_reuse_permitted": reuse.get(
                "automatic_reuse_permitted"
            ),
            "policy_review_required": reuse.get("requires_experiment_policy_review"),
            "reuse_authorized_by_candidate_record": False,
        }
        projection = reuse_candidate_identity_projection(payload)
        record = {**payload, "candidate_id": _digest(projection)}
        validate_exact_reuse_candidate(record)
        records.append(record)
    return records


def validate_exact_reuse_candidate(payload: Mapping[str, Any]) -> None:
    projection = reuse_candidate_identity_projection(payload)
    _validate_schema(payload, REUSE_CANDIDATE_SCHEMA_PATH)
    if payload.get("candidate_id") != _digest(projection):
        raise OperationRecordError(
            "candidate_id does not match canonical identity projection"
        )
