from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.openevo_runtime.execution_generation import build_execution_generation
from scripts.openevo_runtime.generation_backfill import (
    HistoricalBackfillError,
    audit_generation_handoff,
    build_historical_generation_backfill,
    sha256_file,
)


def _snapshot() -> dict[str, object]:
    return {
        "label": "historical-async-523",
        "execution_git_sha": "f" * 40,
        "science_contract_identity": "1" * 64,
        "runtime_package_identity": "2" * 64,
        "legacy_bindings": {
            "execution_git_sha": "f" * 40,
            "ready_sha256": "3" * 64,
            "owner_launch_release_sha256": "4" * 64,
            "sealed_rollout_recovery_sha256": "5" * 64,
            "carrier_adoption_identity": "6" * 64,
        },
    }


def test_historical_backfill_is_read_only_and_grants_no_authority(tmp_path: Path) -> None:
    source = tmp_path / "receipt.json"
    source.write_bytes(b"sealed")
    snapshot = _snapshot()
    original = copy.deepcopy(snapshot)
    record = build_historical_generation_backfill(
        snapshot,
        source_witnesses=[{"path": str(source), "sha256": sha256_file(source)}],
    )
    assert snapshot == original
    assert record["read_only"] is True
    assert record["historical_bytes_rewritten"] is False
    assert record["scientific_permission_granted"] is False
    assert record["scientific_retry_authorized"] is False
    assert record["automatic_reuse_authorized"] is False
    assert record["source_witnesses"][0]["verified"] is True


def test_historical_source_drift_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "receipt.json"
    source.write_text("before")
    expected = sha256_file(source)
    source.write_text("after")
    with pytest.raises(HistoricalBackfillError, match="source witness changed"):
        build_historical_generation_backfill(
            _snapshot(),
            source_witnesses=[{"path": str(source), "sha256": expected}],
        )


def _generation(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "openevo.execution-generation.202609170115",
        "execution_git_sha": "a" * 40,
        "science_contract_identity": "1" * 64,
        "runtime_package_identity": "2" * 64,
        "ready_identity": "3" * 64,
        "owner_release_identity": "4" * 64,
        "recovery_policy_identity": "5" * 64,
        "carrier_policy_identity": "6" * 64,
        "resource_policy_identity": "7" * 64,
        "artifact_reuse_manifest_identity": None,
    }
    payload.update(updates)
    return build_execution_generation(payload)


def test_control_runtime_handoff_requires_declared_change_and_parity_proof() -> None:
    before = _generation()
    after = _generation(execution_git_sha="b" * 40, ready_identity="8" * 64)
    audit = audit_generation_handoff(
        before,
        after,
        declared_changed_fields=["execution_git_sha", "ready_identity"],
        execution_parity_proof_identity="9" * 64,
    )
    assert audit["control_only"] is True
    assert audit["science_sensitive_changed_fields"] == []
    assert audit["execution_parity_changed_fields"] == ["execution_git_sha"]


def test_handoff_rejects_undeclared_generation_drift() -> None:
    before = _generation()
    after = _generation(ready_identity="8" * 64, resource_policy_identity="9" * 64)
    with pytest.raises(HistoricalBackfillError, match="changed fields do not match"):
        audit_generation_handoff(
            before,
            after,
            declared_changed_fields=["ready_identity"],
        )


def test_carrier_binding_change_requires_equivalence_proof_or_amendment() -> None:
    before = _generation()
    after = _generation(carrier_policy_identity="8" * 64)
    with pytest.raises(HistoricalBackfillError, match="policy-equivalence"):
        audit_generation_handoff(
            before,
            after,
            declared_changed_fields=["carrier_policy_identity"],
        )
    control_audit = audit_generation_handoff(
        before,
        after,
        declared_changed_fields=["carrier_policy_identity"],
        policy_equivalence_proof_identity="a" * 64,
    )
    assert control_audit["control_only"] is True
    assert control_audit["policy_sensitive_changed_fields"] == ["carrier_policy_identity"]
    audit = audit_generation_handoff(
        before,
        after,
        declared_changed_fields=["carrier_policy_identity"],
        scientific_amendment_identity="9" * 64,
    )
    assert audit["control_only"] is False
    assert audit["policy_sensitive_changed_fields"] == ["carrier_policy_identity"]


def test_reuse_admissibility_change_is_science_sensitive() -> None:
    before = _generation()
    after = _generation(artifact_reuse_manifest_identity="8" * 64)
    with pytest.raises(HistoricalBackfillError, match="policy-equivalence"):
        audit_generation_handoff(
            before,
            after,
            declared_changed_fields=["artifact_reuse_manifest_identity"],
        )


def test_science_contract_change_requires_scientific_amendment() -> None:
    before = _generation()
    after = _generation(science_contract_identity="8" * 64)
    with pytest.raises(HistoricalBackfillError, match="scientific amendment"):
        audit_generation_handoff(
            before,
            after,
            declared_changed_fields=["science_contract_identity"],
        )
    audit = audit_generation_handoff(
        before,
        after,
        declared_changed_fields=["science_contract_identity"],
        scientific_amendment_identity="9" * 64,
    )
    assert audit["control_only"] is False
    assert audit["science_sensitive_changed_fields"] == ["science_contract_identity"]
