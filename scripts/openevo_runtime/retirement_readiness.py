"""Fail-closed Phase-H retirement readiness for Execution Runtime modernization.

This module prepares retirement only. It never stops a process, removes a file,
changes GPU admission, or grants scientific authority.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

HEX64 = re.compile(r"^[0-9a-f]{64}$")
READINESS_SCHEMA = "openevo.execution-runtime-retirement-readiness.202609201900"
PROOF_SCHEMA = "openevo.execution-runtime-production-campaign-proof.202609201900"


class RetirementReadinessError(ValueError):
    """Retirement evidence is malformed or internally inconsistent."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RetirementReadinessError(f"expected JSON object: {path}")
    return value


def _identity(value: Any, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise RetirementReadinessError(f"{label} must be a SHA-256 identity")
    return value


def _validate_production_proof(
    proof: Mapping[str, Any],
    *,
    execution_generation_id: str,
    resource_canary_receipt_identity: str,
    exact_consumer_validation_receipt_identity: str,
) -> None:
    if proof.get("schema") != PROOF_SCHEMA:
        raise RetirementReadinessError("production proof schema drift")
    if proof.get("runtime_modernization_proved") is not True:
        raise RetirementReadinessError("production proof must affirm runtime modernization")
    if proof.get("closure_gates_passed") is not True:
        raise RetirementReadinessError("production proof must affirm closure gates")
    if proof.get("final_panel_boundary_respected") is not True:
        raise RetirementReadinessError("production proof must affirm final-panel boundary")
    if proof.get("proof_mutates_scientific_semantics") is not False:
        raise RetirementReadinessError("retirement proof cannot mutate scientific semantics")
    if proof.get("execution_generation_id") != execution_generation_id:
        raise RetirementReadinessError("production proof generation mismatch")
    if proof.get("resource_canary_receipt_identity") != resource_canary_receipt_identity:
        raise RetirementReadinessError("production proof resource-canary mismatch")
    if proof.get("exact_consumer_validation_receipt_identity") != exact_consumer_validation_receipt_identity:
        raise RetirementReadinessError("production proof consumer-validation mismatch")
    rows = proof.get("formal_rows_consumed")
    if type(rows) is not int or rows <= 0:
        raise RetirementReadinessError("production proof requires positive formal_rows_consumed")


def evaluate_retirement_readiness(
    release: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    production_proof: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if release.get("schema") != "openevo.execution-runtime-prospective-adoption.202609171752":
        raise RetirementReadinessError("unexpected adoption release schema")
    generation = _identity(release.get("execution_generation_id"), "execution_generation_id")
    blockers: list[str] = []

    resource_canary = release.get("resource_canary_receipt_identity")
    consumer_validation = release.get("exact_consumer_validation_receipt_identity")
    if not isinstance(resource_canary, str) or not HEX64.fullmatch(resource_canary):
        blockers.append("ZERO_FORMAL_RESOURCE_CANARY_PENDING")
    if not isinstance(consumer_validation, str) or not HEX64.fullmatch(consumer_validation):
        blockers.append("EXACT_CONSUMER_VALIDATION_PENDING")
    if release.get("target_campaign_is_new") is not True:
        blockers.append("NEW_CAMPAIGN_BOUNDARY_NOT_CONFIRMED")
    if release.get("hot_migration_allowed") is not False:
        blockers.append("HOT_MIGRATION_BOUNDARY_DRIFT")
    if release.get("existing_formal_lineage_migration_allowed") is not False:
        blockers.append("EXISTING_LINEAGE_MIGRATION_BOUNDARY_DRIFT")

    if production_proof is None:
        blockers.append("PRODUCTION_CAMPAIGN_PROOF_MISSING")
    elif not blockers:
        _validate_production_proof(
            production_proof,
            execution_generation_id=generation,
            resource_canary_receipt_identity=resource_canary,
            exact_consumer_validation_receipt_identity=consumer_validation,
        )

    ready = not blockers
    normalized_candidates = []
    for row in candidates:
        path = row.get("path")
        if not isinstance(path, str) or not path:
            raise RetirementReadinessError("retirement candidate path must be non-empty")
        action = row.get("post_gate_action")
        if action not in {"RETIRE_CONTROL_PATH", "KEEP_HISTORY_ONLY"}:
            raise RetirementReadinessError(f"unsupported post_gate_action for {path}")
        normalized_candidates.append(
            {
                **dict(row),
                "retirement_authorized": ready and action == "RETIRE_CONTROL_PATH",
                "history_only_mark_authorized": ready and action == "KEEP_HISTORY_ONLY",
            }
        )

    return {
        "schema": READINESS_SCHEMA,
        "status": "READY_FOR_PHASE_H_RETIREMENT" if ready else "BLOCKED_PHASE_H_RETIREMENT",
        "execution_generation_id": generation,
        "blockers": blockers,
        "phase_h_mutation_authorized": ready,
        "scientific_authority_granted": False,
        "gpu_authority_granted": False,
        "current_formal_hot_migration_authorized": False,
        "candidates": normalized_candidates,
    }
