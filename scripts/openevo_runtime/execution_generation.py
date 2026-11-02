"""Content-addressed ExecutionGeneration for the OpenEVO control plane.

The generation groups exact control/runtime bindings. Its digest is identity, not
scientific permission: Passport and the bound authorities still decide admissibility.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from scripts.openevo_webshop.experiment_passport import (
    PassportError,
    assert_no_credentials,
    canonical_bytes,
    validate_schema,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ID = "openevo.execution-generation.202609170115"
SCHEMA_PATH = ROOT / "schemas/execution-generation-202609170115.schema.json"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
IDENTITY_FIELDS = (
    "schema",
    "execution_git_sha",
    "science_contract_identity",
    "runtime_package_identity",
    "ready_identity",
    "owner_release_identity",
    "recovery_policy_identity",
    "carrier_policy_identity",
    "resource_policy_identity",
    "artifact_reuse_manifest_identity",
)
OPTIONAL_SHA256_FIELDS = frozenset({
    "ready_identity",
    "owner_release_identity",
    "recovery_policy_identity",
    "carrier_policy_identity",
    "resource_policy_identity",
    "artifact_reuse_manifest_identity",
})
LEGACY_BINDING_ALIASES: dict[str, tuple[str, ...]] = {
    "ready_identity": ("ready_identity", "ready_sha256", "consumer_ready_sha256"),
    "owner_release_identity": (
        "owner_release_identity",
        "owner_release_sha256",
        "launch_release_sha256",
        "owner_launch_release_sha256",
    ),
    "recovery_policy_identity": (
        "recovery_policy_identity",
        "sealed_rollout_recovery_sha256",
        "recovery_sha256",
        "recovery_manifest_sha256",
    ),
    "carrier_policy_identity": (
        "carrier_policy_identity",
        "carrier_adoption_identity",
        "carrier_adoption_identity_sha256",
        "completion_first_policy_identity_sha256",
    ),
    "resource_policy_identity": (
        "resource_policy_identity",
        "async_resource_successor_sha256",
        "resource_successor_sha256",
        "resource_migration_sha256",
    ),
    "artifact_reuse_manifest_identity": (
        "artifact_reuse_manifest_identity",
        "artifact_reuse_manifest_sha256",
        "sealed_rollout_reuse_manifest_sha256",
    ),
}
LEGACY_MAPPING_SCHEMA = "openevo.execution-generation-legacy-mapping.202609170147"


class ExecutionGenerationError(ValueError):
    pass


def _schema() -> dict[str, Any]:
    try:
        value = json.loads(SCHEMA_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ExecutionGenerationError("ExecutionGeneration schema is unreadable") from exc
    if not isinstance(value, dict):
        raise ExecutionGenerationError("ExecutionGeneration schema must be an object")
    return value


def identity_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical, versioned fields that define one generation."""
    allowed = set(IDENTITY_FIELDS) | {"generation_id"}
    extra = set(payload) - allowed
    if extra:
        raise ExecutionGenerationError(f"undeclared generation field: {sorted(extra)[0]}")
    missing = [field for field in IDENTITY_FIELDS if field not in payload]
    if missing:
        raise ExecutionGenerationError(f"missing generation identity field: {missing[0]}")
    projection = {field: payload[field] for field in IDENTITY_FIELDS}
    if projection["schema"] != SCHEMA_ID:
        raise ExecutionGenerationError(f"schema must equal {SCHEMA_ID}")
    if not isinstance(projection["execution_git_sha"], str) or not HEX40.fullmatch(
        projection["execution_git_sha"]
    ):
        raise ExecutionGenerationError("execution_git_sha must be an immutable 40-hex Git SHA")
    for field in IDENTITY_FIELDS:
        if field in {"schema", "execution_git_sha"}:
            continue
        value = projection[field]
        if field in OPTIONAL_SHA256_FIELDS and value is None:
            continue
        if not isinstance(value, str) or not HEX64.fullmatch(value):
            raise ExecutionGenerationError(
                f"{field} must be a SHA-256 identity or explicit null when optional"
            )
    assert_no_credentials(projection)
    canonical_bytes(projection)
    return projection


def _map_legacy_optional_identity(
    legacy_bindings: Mapping[str, Any],
    *,
    field: str,
) -> tuple[str | None, list[str]]:
    aliases = LEGACY_BINDING_ALIASES[field]
    seen: list[tuple[str, str]] = []
    for alias in aliases:
        if alias not in legacy_bindings or legacy_bindings[alias] is None:
            continue
        value = legacy_bindings[alias]
        if not isinstance(value, str) or not HEX64.fullmatch(value):
            raise ExecutionGenerationError(
                f"legacy binding {alias} for {field} must be a SHA-256 identity or null"
            )
        seen.append((alias, value))
    distinct = {value for _, value in seen}
    if len(distinct) > 1:
        raise ExecutionGenerationError(f"conflicting legacy aliases for {field}")
    return (seen[0][1] if seen else None), [alias for alias, _ in seen]


def build_execution_generation_from_legacy_bindings(
    legacy_bindings: Mapping[str, Any],
    *,
    execution_git_sha: str,
    science_contract_identity: str,
    runtime_package_identity: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Map existing control-plane bindings into one ExecutionGeneration.

    This adapter is deliberately narrow: science/runtime identities are explicit
    inputs, optional legacy roles map only through declared hash aliases, and
    conflicting aliases fail closed. It grants no execution, retry, reuse, or
    scientific permission.
    """
    if not isinstance(legacy_bindings, Mapping):
        raise ExecutionGenerationError("legacy_bindings must be a mapping")
    if "execution_git_sha" in legacy_bindings:
        observed = legacy_bindings["execution_git_sha"]
        if observed != execution_git_sha:
            raise ExecutionGenerationError(
                "legacy execution_git_sha does not match requested generation"
            )

    projection: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "execution_git_sha": execution_git_sha,
        "science_contract_identity": science_contract_identity,
        "runtime_package_identity": runtime_package_identity,
    }
    mapped_aliases: dict[str, list[str]] = {}
    for field in OPTIONAL_SHA256_FIELDS:
        value, aliases = _map_legacy_optional_identity(legacy_bindings, field=field)
        projection[field] = value
        mapped_aliases[field] = aliases

    generation = build_execution_generation(projection)
    recognized = {"execution_git_sha"}
    for aliases in LEGACY_BINDING_ALIASES.values():
        recognized.update(aliases)
    audit = {
        "schema": LEGACY_MAPPING_SCHEMA,
        "generation_id": generation["generation_id"],
        "source_execution_git_sha": execution_git_sha,
        "mapped_aliases": {
            key: mapped_aliases[key] for key in sorted(mapped_aliases)
        },
        "ignored_legacy_keys": sorted(set(legacy_bindings) - recognized),
        "scientific_permission_granted": False,
        "scientific_retry_authorized": False,
        "automatic_reuse_authorized": False,
        "live_execution_mutated": False,
    }
    assert_no_credentials(audit)
    canonical_bytes(audit)
    return generation, audit


def generation_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(identity_projection(payload))).hexdigest()


def build_execution_generation(payload: Mapping[str, Any]) -> dict[str, Any]:
    projection = identity_projection(payload)
    result = {**projection, "generation_id": generation_digest(projection)}
    try:
        validate_schema(result, _schema())
    except PassportError as exc:
        raise ExecutionGenerationError(str(exc)) from exc
    return result


def validate_execution_generation(payload: Mapping[str, Any]) -> None:
    try:
        validate_schema(dict(payload), _schema())
    except PassportError as exc:
        raise ExecutionGenerationError(str(exc)) from exc
    expected = generation_digest(payload)
    if payload.get("generation_id") != expected:
        raise ExecutionGenerationError(
            "generation_id does not match canonical identity projection"
        )
