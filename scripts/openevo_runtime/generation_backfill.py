"""Read-only historical ExecutionGeneration backfill and handoff audit.

This module never writes historical artifacts and never grants launch, retry, or reuse
permission. It projects already-frozen identities into the shared ExecutionGeneration
contract so older runs can be replayed by the modern runtime without rewriting history.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.openevo_runtime.execution_generation import (
    IDENTITY_FIELDS,
    build_execution_generation_from_legacy_bindings,
    validate_execution_generation,
)
from scripts.openevo_webshop.experiment_passport import assert_no_credentials, canonical_bytes

BACKFILL_SCHEMA = "openevo.execution-generation-historical-backfill.202609170458"
HANDOFF_SCHEMA = "openevo.execution-generation-handoff-audit.202609170458"
SCIENCE_SENSITIVE_FIELDS = frozenset({"science_contract_identity"})
POLICY_SENSITIVE_FIELDS = frozenset(
    {"carrier_policy_identity", "artifact_reuse_manifest_identity"}
)
EXECUTION_PARITY_FIELDS = frozenset({"execution_git_sha", "runtime_package_identity"})
CONTROL_FIELDS = frozenset(
    {
        "ready_identity",
        "owner_release_identity",
        "recovery_policy_identity",
        "resource_policy_identity",
    }
)


class HistoricalBackfillError(ValueError):
    pass


def sha256_file(path: str | Path) -> str:
    """Hash one existing source without mutating it."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_witnesses(witnesses: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Verify repository/local witnesses by exact SHA-256, read-only."""
    verified: list[dict[str, Any]] = []
    for witness in witnesses:
        path = witness.get("path")
        expected = witness.get("sha256")
        if not isinstance(path, str) or not path:
            raise HistoricalBackfillError("source witness path must be non-empty")
        if not isinstance(expected, str) or len(expected) != 64:
            raise HistoricalBackfillError("source witness sha256 must be 64 hex characters")
        observed = sha256_file(path)
        if observed != expected:
            raise HistoricalBackfillError(f"source witness changed: {path}")
        verified.append({"path": path, "sha256": observed, "verified": True})
    return verified


def build_historical_generation_backfill(
    snapshot: Mapping[str, Any],
    *,
    source_witnesses: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Project one historical snapshot into ExecutionGeneration without side effects."""
    required = {
        "label",
        "execution_git_sha",
        "science_contract_identity",
        "runtime_package_identity",
        "legacy_bindings",
    }
    missing = sorted(required - set(snapshot))
    if missing:
        raise HistoricalBackfillError(f"missing historical snapshot field: {missing[0]}")
    if not isinstance(snapshot["label"], str) or not snapshot["label"]:
        raise HistoricalBackfillError("historical snapshot label must be non-empty")
    if not isinstance(snapshot["legacy_bindings"], Mapping):
        raise HistoricalBackfillError("legacy_bindings must be a mapping")

    generation, mapping_audit = build_execution_generation_from_legacy_bindings(
        snapshot["legacy_bindings"],
        execution_git_sha=snapshot["execution_git_sha"],
        science_contract_identity=snapshot["science_contract_identity"],
        runtime_package_identity=snapshot["runtime_package_identity"],
    )
    validate_execution_generation(generation)
    verified_sources = verify_source_witnesses(source_witnesses)
    record = {
        "schema": BACKFILL_SCHEMA,
        "label": snapshot["label"],
        "generation": generation,
        "mapping_audit": mapping_audit,
        "source_witnesses": verified_sources,
        "read_only": True,
        "historical_bytes_rewritten": False,
        "scientific_permission_granted": False,
        "scientific_retry_authorized": False,
        "automatic_reuse_authorized": False,
    }
    assert_no_credentials(record)
    canonical_bytes(record)
    return record


def audit_generation_handoff(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    declared_changed_fields: Sequence[str],
    execution_parity_proof_identity: str | None = None,
    policy_equivalence_proof_identity: str | None = None,
    scientific_amendment_identity: str | None = None,
) -> dict[str, Any]:
    """Prove a generation change is declared control/runtime drift or fail closed.

    Science-contract changes require an explicit scientific amendment. Carrier/reuse
    binding changes require exact policy-equivalence proof unless an amendment owns the
    semantic change. Execution/runtime byte changes similarly require exact parity proof.
    """
    validate_execution_generation(before)
    validate_execution_generation(after)
    changed = sorted(
        field for field in IDENTITY_FIELDS if field != "schema" and before[field] != after[field]
    )
    declared = sorted(set(declared_changed_fields))
    unknown_declared = sorted(set(declared) - (set(IDENTITY_FIELDS) - {"schema"}))
    if unknown_declared:
        raise HistoricalBackfillError(f"undeclared identity field: {unknown_declared[0]}")
    if changed != declared:
        raise HistoricalBackfillError(
            f"changed fields do not match declaration: observed={changed} declared={declared}"
        )

    science_sensitive = sorted(set(changed) & SCIENCE_SENSITIVE_FIELDS)
    policy_sensitive = sorted(set(changed) & POLICY_SENSITIVE_FIELDS)
    parity_sensitive = sorted(set(changed) & EXECUTION_PARITY_FIELDS)
    if science_sensitive and scientific_amendment_identity is None:
        raise HistoricalBackfillError(
            "science-sensitive generation change requires scientific amendment"
        )
    if (
        policy_sensitive
        and scientific_amendment_identity is None
        and policy_equivalence_proof_identity is None
    ):
        raise HistoricalBackfillError(
            "carrier/reuse binding change requires policy-equivalence proof or scientific amendment"
        )
    if parity_sensitive and scientific_amendment_identity is None and execution_parity_proof_identity is None:
        raise HistoricalBackfillError(
            "execution/runtime generation change requires exact parity proof"
        )

    audit = {
        "schema": HANDOFF_SCHEMA,
        "before_generation_id": before["generation_id"],
        "after_generation_id": after["generation_id"],
        "changed_fields": changed,
        "declared_changed_fields": declared,
        "science_sensitive_changed_fields": science_sensitive,
        "policy_sensitive_changed_fields": policy_sensitive,
        "execution_parity_changed_fields": parity_sensitive,
        "control_changed_fields": sorted(set(changed) & CONTROL_FIELDS),
        "execution_parity_proof_identity": execution_parity_proof_identity,
        "policy_equivalence_proof_identity": policy_equivalence_proof_identity,
        "scientific_amendment_identity": scientific_amendment_identity,
        "control_only": not science_sensitive and scientific_amendment_identity is None,
        "scientific_state_mutated_by_audit": False,
        "launch_authority": False,
        "scientific_retry_authorized": False,
        "automatic_reuse_authorized": False,
    }
    assert_no_credentials(audit)
    canonical_bytes(audit)
    return audit
