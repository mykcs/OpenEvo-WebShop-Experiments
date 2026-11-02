#!/usr/bin/env python3
"""Migration-safe FROZEN Passport gate for formal campaign execution.

The gate is read-only. Existing pre-Passport campaigns are grandfathered;
new stable-named formal campaigns must bind a registered FROZEN Passport.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from scripts.openevo_webshop.experiment_registry import (
    RegistryError,
    load_bound_passport,
    load_current_registry,
    resolve_experiment,
    validate_registry,
)

PASSPORT_GATE_CUTOFF = "20260912-0000"
_STABLE_PREFIX = re.compile(r"^(\d{8}-\d{4})")


class PassportGateError(ValueError):
    pass

def gate_requirement(campaign_id: str | None, formal_allowed: bool) -> str:
    """Return MANDATORY, GRANDFATHERED, or NOT_APPLICABLE."""
    if not formal_allowed:
        return "NOT_APPLICABLE"
    text = str(campaign_id or "")
    match = _STABLE_PREFIX.match(text)
    if match is None:
        return "GRANDFATHERED_LEGACY_ID"
    return "MANDATORY" if match.group(1) >= PASSPORT_GATE_CUTOFF else "GRANDFATHERED"


def validate_binding(repo_root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    experiment_id = binding.get("experiment_id")
    passport_sha = binding.get("passport_sha256")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise PassportGateError("experiment_identity.experiment_id is required")
    if not isinstance(passport_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", passport_sha):
        raise PassportGateError("experiment_identity.passport_sha256 must be a bare SHA-256")
    registry, router = load_current_registry(repo_root)
    errors = validate_registry(registry, repo_root)
    if errors:
        raise PassportGateError("Experiment Registry invalid: " + "; ".join(errors))
    try:
        entry = resolve_experiment(registry, experiment_id)
    except RegistryError as exc:
        raise PassportGateError(str(exc)) from exc
    if entry["experiment_id"] != experiment_id:
        raise PassportGateError("formal launch binding must use canonical experiment_id, not an alias")
    if entry["passport_sha256"] != passport_sha:
        raise PassportGateError("formal launch Passport SHA does not match Registry")
    try:
        passport = load_bound_passport(repo_root, entry)
    except RegistryError as exc:
        raise PassportGateError(str(exc)) from exc
    if passport.get("status") != "FROZEN":
        raise PassportGateError("formal launch requires status=FROZEN Passport")
    approved_roles = list((passport.get("tracking_policy") or {}).get("approved_artifact_roles") or [])
    if "run-manifest" not in approved_roles:
        raise PassportGateError("FROZEN Passport does not permit run-manifest cross-links")
    return {
        "status": "PASS",
        "requirement": "BOUND",
        "experiment_id": experiment_id,
        "passport_sha256": passport_sha,
        "passport_status": "FROZEN",
        "approved_artifact_roles": approved_roles,
        "registry_path": router["path"],
        "registry_sha256": router["sha256"],
    }


def evaluate_release_gate(release: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    formal = release.get("formal_task_consumption_allowed") is True
    campaign_id = str(release.get("campaign_id") or "")
    requirement = gate_requirement(campaign_id, formal)
    binding = release.get("experiment_identity")
    if binding is None:
        if requirement == "MANDATORY":
            return {
                "status": "FAIL",
                "requirement": requirement,
                "experiment_id": None,
                "passport_sha256": None,
                "passport_status": "MISSING",
                "reason": (
                    "new formal campaign requires experiment_identity with canonical "
                    "experiment_id + passport_sha256 before execution readiness"
                ),
            }
        return {
            "status": "GRANDFATHERED" if requirement.startswith("GRANDFATHERED") else "NOT_APPLICABLE",
            "requirement": requirement,
            "experiment_id": None,
            "passport_sha256": None,
            "passport_status": "NOT_BOUND",
            "reason": "pre-Passport campaign retained for compatibility" if formal else "formal execution is not enabled",
        }
    if not isinstance(binding, Mapping):
        return {
            "status": "FAIL", "requirement": requirement,
            "experiment_id": None, "passport_sha256": None,
            "passport_status": "INVALID", "reason": "experiment_identity must be an object",
        }
    try:
        result = validate_binding(repo_root, binding)
    except PassportGateError as exc:
        return {
            "status": "FAIL", "requirement": requirement,
            "experiment_id": binding.get("experiment_id"),
            "passport_sha256": binding.get("passport_sha256"),
            "passport_status": "INVALID", "reason": str(exc),
        }
    result["requirement"] = requirement
    result["reason"] = "registered FROZEN Passport binding verified"
    return result
