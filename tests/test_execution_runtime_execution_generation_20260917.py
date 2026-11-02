from __future__ import annotations

import copy

import pytest

from scripts.openevo_runtime.execution_generation import (
    ExecutionGenerationError,
    SCHEMA_ID,
    build_execution_generation,
    build_execution_generation_from_legacy_bindings,
    generation_digest,
    validate_execution_generation,
)


def base_payload() -> dict[str, object]:
    return {
        "schema": SCHEMA_ID,
        "execution_git_sha": "1" * 40,
        "science_contract_identity": "2" * 64,
        "runtime_package_identity": "3" * 64,
        "ready_identity": "4" * 64,
        "owner_release_identity": "5" * 64,
        "recovery_policy_identity": "6" * 64,
        "carrier_policy_identity": "7" * 64,
        "resource_policy_identity": "8" * 64,
        "artifact_reuse_manifest_identity": None,
    }


def test_generation_is_content_addressed_and_valid() -> None:
    generation = build_execution_generation(base_payload())
    assert generation["generation_id"] == generation_digest(generation)
    validate_execution_generation(generation)


def test_key_order_does_not_change_generation_identity() -> None:
    payload = base_payload()
    reversed_payload = dict(reversed(list(payload.items())))
    assert generation_digest(payload) == generation_digest(reversed_payload)


def test_subordinate_binding_change_changes_generation_identity() -> None:
    first = base_payload()
    second = copy.deepcopy(first)
    second["resource_policy_identity"] = "9" * 64
    assert generation_digest(first) != generation_digest(second)


def test_explicit_absence_is_identity_bearing() -> None:
    absent = base_payload()
    present = copy.deepcopy(absent)
    present["artifact_reuse_manifest_identity"] = "a" * 64
    assert generation_digest(absent) != generation_digest(present)


def test_generation_id_tampering_fails_closed() -> None:
    generation = build_execution_generation(base_payload())
    generation["generation_id"] = "f" * 64
    with pytest.raises(ExecutionGenerationError, match="generation_id"):
        validate_execution_generation(generation)


def test_unknown_field_cannot_escape_identity_projection() -> None:
    payload = base_payload()
    payload["presentation_note"] = "not identity"
    with pytest.raises(ExecutionGenerationError, match="undeclared generation field"):
        build_execution_generation(payload)


def test_required_science_and_runtime_identities_cannot_be_null() -> None:
    payload = base_payload()
    payload["science_contract_identity"] = None
    with pytest.raises(ExecutionGenerationError, match="science_contract_identity"):
        build_execution_generation(payload)


def legacy_payload() -> dict[str, object]:
    return {
        "execution_git_sha": "fbdb21b2739ffeeca90e17eac83c14adb6087be2",
        "ready_sha256": "3493da01babaf762acc7b6337d685ec86111df352ad7e180d7dfbc9f74576432",
        "owner_release_sha256": "7c6484f3629b149c6b4afe21f9dfa8d5c02bc789784386402d9fd04f2017aecb",
        "sealed_rollout_recovery_sha256": "6e0369b3880e6ae36223f9f5eed826582e73f02a1a72e7ec02d9ae78804c569b",
        "carrier_adoption_identity": "bdeacdd126f89dffa9ec8c31a12e20b7bbb9ae8367f2ff80b92e9c8e848ad0e4",
        "async_resource_successor_sha256": "8" * 64,
        "legacy_note": "presentation-only",
    }


def build_from_legacy(payload: dict[str, object] | None = None):
    return build_execution_generation_from_legacy_bindings(
        legacy_payload() if payload is None else payload,
        execution_git_sha="fbdb21b2739ffeeca90e17eac83c14adb6087be2",
        science_contract_identity="2" * 64,
        runtime_package_identity="3" * 64,
    )


def test_legacy_bindings_map_to_generation_without_granting_authority() -> None:
    generation, audit = build_from_legacy()
    validate_execution_generation(generation)
    assert generation["ready_identity"] == legacy_payload()["ready_sha256"]
    assert generation["owner_release_identity"] == legacy_payload()["owner_release_sha256"]
    assert generation["recovery_policy_identity"] == legacy_payload()["sealed_rollout_recovery_sha256"]
    assert generation["carrier_policy_identity"] == legacy_payload()["carrier_adoption_identity"]
    assert generation["resource_policy_identity"] == "8" * 64
    assert generation["artifact_reuse_manifest_identity"] is None
    assert audit["mapped_aliases"]["ready_identity"] == ["ready_sha256"]
    assert audit["ignored_legacy_keys"] == ["legacy_note"]
    assert audit["scientific_permission_granted"] is False
    assert audit["scientific_retry_authorized"] is False
    assert audit["automatic_reuse_authorized"] is False
    assert audit["live_execution_mutated"] is False


def test_legacy_duplicate_aliases_are_allowed_only_when_hashes_match() -> None:
    payload = legacy_payload()
    payload["consumer_ready_sha256"] = payload["ready_sha256"]
    generation, audit = build_from_legacy(payload)
    assert generation["ready_identity"] == payload["ready_sha256"]
    assert audit["mapped_aliases"]["ready_identity"] == [
        "ready_sha256",
        "consumer_ready_sha256",
    ]


def test_conflicting_legacy_aliases_fail_closed() -> None:
    payload = legacy_payload()
    payload["consumer_ready_sha256"] = "f" * 64
    with pytest.raises(ExecutionGenerationError, match="conflicting legacy aliases"):
        build_from_legacy(payload)


def test_legacy_execution_sha_drift_fails_closed() -> None:
    payload = legacy_payload()
    payload["execution_git_sha"] = "0" * 40
    with pytest.raises(ExecutionGenerationError, match="execution_git_sha"):
        build_from_legacy(payload)


def test_malformed_legacy_digest_fails_closed() -> None:
    payload = legacy_payload()
    payload["owner_release_sha256"] = "not-a-sha"
    with pytest.raises(ExecutionGenerationError, match="owner_release_sha256"):
        build_from_legacy(payload)
