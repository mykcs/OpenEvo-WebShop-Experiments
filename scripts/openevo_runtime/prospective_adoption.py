"""Prospective next-campaign adoption guard for Execution Runtime modernization.

This module is engineering governance only. It never grants scientific, formal,
resource-topology, repair, reuse, or final-panel authority. It exists to make
"new campaign only" and pre-first-task consumer validation mechanical.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from scripts.openevo_webshop.experiment_passport import canonical_bytes

REV = "202609171752"
SCHEMA = f"openevo.execution-runtime-prospective-adoption.{REV}"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
ALLOWED_STATUS = {"BLOCKED_ENGINEERING", "READY_FOR_ADOPTION_VALIDATION"}


class ProspectiveAdoptionError(RuntimeError):
    """Prospective adoption contract is malformed or crosses an authority boundary."""


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _require_hex(value: Any, *, width: int, name: str) -> str:
    pattern = HEX40 if width == 40 else HEX64
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ProspectiveAdoptionError(f"{name} must be {width} lowercase hex")
    return value


def adoption_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "schema", "target_campaign_id", "campaign_class", "status", "blocker_reason",
        "implementation_contract_sha256", "production_runtime_package_identity",
        "execution_generation_id", "resource_canary_receipt_identity",
        "exact_consumer_validation_receipt_identity", "reference_controller_git_sha",
        "predecessor_controller_retained", "predecessor_retention_until_closure",
        "exact_consumer_validation_required_before_first_formal_task",
        "target_campaign_is_new", "target_campaign_formal_rows_consumed",
        "existing_formal_lineage_migration_allowed", "hot_migration_allowed",
        "live_adoption", "formal_launch_authority", "scientific_authority",
        "resource_topology_authority", "repair_semantics_authority", "reuse_authority",
        "final_panel_authority", "closure_gates",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ProspectiveAdoptionError(f"missing adoption field: {missing[0]}")
    return {key: payload[key] for key in required}


def validate_prospective_adoption_release(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != SCHEMA:
        raise ProspectiveAdoptionError("unexpected adoption schema")
    projection = adoption_projection(payload)
    if payload.get("release_id") != _digest(projection):
        raise ProspectiveAdoptionError("release_id does not match canonical projection")
    if payload["status"] not in ALLOWED_STATUS:
        raise ProspectiveAdoptionError("unsupported adoption status")
    if not isinstance(payload["target_campaign_id"], str) or not payload["target_campaign_id"]:
        raise ProspectiveAdoptionError("target_campaign_id must be non-empty")
    if payload["target_campaign_is_new"] is not True or payload["target_campaign_formal_rows_consumed"] != 0:
        raise ProspectiveAdoptionError("adoption release is new-campaign/pre-first-formal-task only")
    if payload["existing_formal_lineage_migration_allowed"] is not False or payload["hot_migration_allowed"] is not False:
        raise ProspectiveAdoptionError("existing formal lineage migration is forbidden")
    for field in (
        "live_adoption", "formal_launch_authority", "scientific_authority",
        "resource_topology_authority", "repair_semantics_authority", "reuse_authority",
        "final_panel_authority",
    ):
        if payload[field] is not False:
            raise ProspectiveAdoptionError(f"{field} must remain false")
    if payload["predecessor_controller_retained"] is not True or payload["predecessor_retention_until_closure"] is not True:
        raise ProspectiveAdoptionError("predecessor controller must remain available through closure")
    if payload["exact_consumer_validation_required_before_first_formal_task"] is not True:
        raise ProspectiveAdoptionError("exact consumer validation must gate the first formal task")
    _require_hex(payload["implementation_contract_sha256"], width=64, name="implementation_contract_sha256")
    _require_hex(payload["reference_controller_git_sha"], width=40, name="reference_controller_git_sha")
    for field in ("production_runtime_package_identity", "execution_generation_id", "resource_canary_receipt_identity", "exact_consumer_validation_receipt_identity"):
        value = payload[field]
        if value is not None:
            _require_hex(value, width=64, name=field)

    gates = payload["closure_gates"]
    required_gates = {
        "ZERO_FORMAL_RESOURCE_CANARY_PASS",
        "EXACT_CONSUMER_VALIDATION_PASS",
        "NEW_CAMPAIGN_BOUNDARY_CONFIRMED",
        "PREDECESSOR_REFERENCE_RETAINED",
    }
    if not isinstance(gates, list) or not required_gates.issubset(set(gates)):
        raise ProspectiveAdoptionError("closure_gates missing required adoption guards")
    if payload["status"] == "READY_FOR_ADOPTION_VALIDATION":
        for field in (
            "production_runtime_package_identity", "execution_generation_id",
            "resource_canary_receipt_identity",
        ):
            if payload[field] is None:
                raise ProspectiveAdoptionError(f"{field} required before adoption validation")
        if payload["blocker_reason"] is not None:
            raise ProspectiveAdoptionError("ready release cannot retain a blocker_reason")
    elif not isinstance(payload["blocker_reason"], str) or not payload["blocker_reason"]:
        raise ProspectiveAdoptionError("blocked release requires blocker_reason")


def build_prospective_adoption_release(**values: Any) -> dict[str, Any]:
    payload = {"schema": SCHEMA, **values}
    projection = adoption_projection(payload)
    result = {**payload, "release_id": _digest(projection)}
    validate_prospective_adoption_release(result)
    return result


def point_of_use_adoption_decision(
    release: Mapping[str, Any], *, observed_campaign_id: str,
    formal_rows_consumed: int, production_runtime_package_identity: str | None,
    execution_generation_id: str | None, resource_canary_receipt_identity: str | None,
    consumer_validation_receipt_identity: str | None,
    predecessor_controller_available: bool,
) -> dict[str, Any]:
    validate_prospective_adoption_release(release)
    blockers: list[str] = []

    if observed_campaign_id != release["target_campaign_id"]:
        blockers.append("CAMPAIGN_IDENTITY_DRIFT")
    if formal_rows_consumed != 0:
        blockers.append("FORMAL_TASK_ALREADY_CONSUMED")
    if not predecessor_controller_available:
        blockers.append("PREDECESSOR_REFERENCE_UNAVAILABLE")
    if release["status"] != "READY_FOR_ADOPTION_VALIDATION":
        blockers.append("ADOPTION_RELEASE_NOT_READY")
    checks = (
        ("production_runtime_package_identity", production_runtime_package_identity, "PRODUCTION_RUNTIME_PACKAGE_MISSING_OR_DRIFT"),
        ("execution_generation_id", execution_generation_id, "EXECUTION_GENERATION_MISSING_OR_DRIFT"),
        ("resource_canary_receipt_identity", resource_canary_receipt_identity, "RESOURCE_CANARY_MISSING_OR_DRIFT"),
    )
    for field, observed, blocker in checks:
        if release[field] is None or observed != release[field]:
            blockers.append(blocker)
    required_consumer = release["exact_consumer_validation_receipt_identity"]
    if required_consumer is None or consumer_validation_receipt_identity != required_consumer:
        blockers.append("EXACT_CONSUMER_VALIDATION_MISSING_OR_DRIFT")
    return {
        "decision": "BLOCKED" if blockers else "ADOPTION_PRECONDITIONS_MET",
        "blockers": blockers,
        "campaign_id": release["target_campaign_id"],
        "formal_rows_consumed": formal_rows_consumed,
        "formal_launch_authority": False,
        "scientific_authority": False,
        "existing_formal_lineage_migration_allowed": False,
        "external_campaign_release_still_required": True,
    }
