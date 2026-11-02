#!/usr/bin/env python3
"""Register the Gated-Delta private remote backup without claiming full recovery."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.openevo_webshop import experiment_artifact_dag as dag

ROOT = Path(__file__).resolve().parents[2]
STAMP = "202609290943"
CREATED_AT = "2026-09-29T09:43:00+08:00"
CODE_REL = "scripts/openevo_webshop/gated_delta_private_remote_backup_dag_backfill.py"
OUT_DIR = f"docs/evidence/experiment-passports/{STAMP}-gated-delta-private-remote-backup"
CONFIG_REL = f"{OUT_DIR}/REMOTE_BACKUP_CONFIG.json"
BINDING_REL = f"{OUT_DIR}/GATED_DELTA_PRIVATE_REMOTE_BACKUP_BINDING.json"
RECEIPT_REL = f"{OUT_DIR}/PRODUCTION_DAG_BINDING_RECEIPT.json"
INDEX_REL = "configs/experiment/passport-artifacts/index/experiment-passport-artifact-index-20260929-gated-delta-private-remote-backup.json"
OPERATION_TYPE = "gated-delta-weight.register.private-remote-backup-pending-full-fresh-restore"

SOURCE_REL = "docs/evidence/recovery-disclosure-audit-20260926/GATED_DELTA_WEIGHT_SOURCE_MEMBER_TABLE_20260929.json"
JOIN_REL = "docs/evidence/recovery-disclosure-audit-20260926/GATED_DELTA_WEIGHT_REGISTRY_LFS_JOIN_20260929.json"
PROVIDER_REL = "docs/evidence/recovery-disclosure-audit-20260926/GATED_DELTA_WEIGHT_PROVIDER_READBACK_20260929.json"
FRESH_REL = "docs/evidence/recovery-disclosure-audit-20260926/GATED_DELTA_WEIGHT_FRESH_RESTORE_STATUS_20260929.json"
DUP_REL = "docs/evidence/recovery-disclosure-audit-20260926/GATED_DELTA_WEIGHT_PRIVATE_DUPLICATE_OBSERVATION_20260929.json"
INCIDENT_REL = "docs/evidence/recovery-disclosure-audit-20260926/GATED_DELTA_HF_CREDENTIAL_HANDLING_INCIDENT_20260929.json"
CAPACITY_REL = "docs/evidence/recovery-disclosure-audit-20260926/GATED_DELTA_WEIGHT_PRIVATE_CAPACITY_EXCEPTION_20260928.json"
REPO_ID = "openevo-recovery-2/openevo-gated-delta-d1-paired-private-recovery-20260928"
REVISION = "60234e69b1cb082fa57352a5b49b0d356cfc3e40"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(rel: str, value: Any) -> Path:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return path


def _repo_ref(rel: str) -> dict[str, Any]:
    p = ROOT / rel
    if not p.is_file():
        raise RuntimeError(f"missing evidence: {rel}")
    return {
        "kind": "repository_file", "repository": None, "revision": None,
        "path": rel, "sha256": _sha(p), "evidence_path": None, "reason": None,
    }


def _artifact(rel: str, role: str, derived: list[str] | None = None) -> dict[str, Any]:
    p = ROOT / rel
    return {
        "artifact_type": "json-evidence",
        "digest_algorithm": "sha256",
        "digest": _sha(p),
        "size_bytes": p.stat().st_size,
        "logical_role": role,
        "locations": [_repo_ref(rel)],
        "derived_from_digests": sorted(set(derived or [])),
    }


def main() -> int:
    provider = json.loads((ROOT / PROVIDER_REL).read_text())
    fresh = json.loads((ROOT / FRESH_REL).read_text())
    duplicate = json.loads((ROOT / DUP_REL).read_text())
    incident = json.loads((ROOT / INCIDENT_REL).read_text())

    if provider.get("repo_id") != REPO_ID or provider.get("immutable_revision") != REVISION:
        raise RuntimeError("authoritative provider identity drifted")
    if provider.get("private") is not True or provider.get("canonical_payload_count") != 13:
        raise RuntimeError("authoritative provider readback is not private 13/13")
    if provider.get("canonical_payload_bytes") != 26984360168:
        raise RuntimeError("authoritative provider byte count drifted")
    if fresh.get("status") != "REMOTE_BACKED_UP_FULL_FRESH_RESTORE_PENDING":
        raise RuntimeError("fresh-restore status must remain pending")
    if fresh.get("fresh_restore", {}).get("verified_count") != 2:
        raise RuntimeError("this snapshot expects exactly 2/13 fresh restores")
    if fresh.get("claim_boundary", {}).get("recovery_verified") is not False:
        raise RuntimeError("partial fresh restore cannot claim RECOVERY_VERIFIED")
    if duplicate.get("provider_delete_authorized") is not False:
        raise RuntimeError("duplicate provider deletion must remain unauthorized")
    if incident.get("status") != "CONTAINED_LOCALLY_GLOBAL_REVOKE_PENDING_HUMAN_REAUTH":
        raise RuntimeError("credential incident boundary drifted")

    config = {
        "schema": f"openevo.gated-delta-private-remote-backup-dag-config.{STAMP}",
        "operation_type": OPERATION_TYPE,
        "authoritative_provider": {"repo_id": REPO_ID, "revision": REVISION},
        "remote_backup_state": "REMOTE_BACKED_UP",
        "full_fresh_restore_state": "PENDING_11_OF_13",
        "hard_boundaries": {
            "recovery_verified": False, "scientific_reuse_authorized": False,
            "public_release_authorized": False, "server_delete_authorized": False,
            "provider_delete_authorized": False, "visibility_change_authorized": False,
        },
    }
    _write_json(CONFIG_REL, config)

    binding = {
        "schema": f"openevo.gated-delta-private-remote-backup-dag-binding.{STAMP}",
        "status": "PASS_REMOTE_BACKUP_REGISTERED_FULL_FRESH_RESTORE_PENDING",
        "authoritative_provider": {
            "repo_id": REPO_ID, "immutable_revision": REVISION,
            "provider_visibility": "private", "payload_count": 13,
            "payload_bytes": 26984360168, "provider_sha_size_readback": "PASS_13_OF_13",
        },
        "fresh_restore": {
            "status": "PARTIAL_2_OF_13", "verified_count": 2,
            "verified_bytes": 4136170640, "remaining_count": 11,
            "remaining_bytes": 22848189528, "evidence": FRESH_REL,
        },
        "duplicate_provider": {
            "repo_id": duplicate["repo_id"], "immutable_revision": duplicate["immutable_revision"],
            "retention": duplicate["retention"], "delete_authorized": False,
        },
        "credential_boundary": {
            "status": incident["status"], "evidence": INCIDENT_REL,
        },
        "source_server_state": "ANALYSIS_HOLD_KEEP_LOCAL",
        "automatic_reuse_authorized": False,
        "scientific_semantics_changed": False,
        "public_release_authorized": False,
        "server_delete_authorized": False,
        "provider_delete_authorized": False,
    }
    _write_json(BINDING_REL, binding)

    inputs = [
        _artifact(SOURCE_REL, "gated-delta-live-source-member-table"),
        _artifact(JOIN_REL, "gated-delta-current-registry-lfs-join"),
        _artifact(PROVIDER_REL, "gated-delta-private-provider-readback"),
        _artifact(FRESH_REL, "gated-delta-partial-fresh-restore-status"),
        _artifact(DUP_REL, "gated-delta-private-duplicate-observation"),
        _artifact(INCIDENT_REL, "gated-delta-credential-boundary"),
    ]

    node = {
        "schema": dag.NODE_SCHEMA_ID,
        "node_kind": "operation",
        "operation_type": OPERATION_TYPE,
        "operation_schema_version": STAMP,
        "operation_digest": "0" * 64,
        "identity_payload": {
            "code_identity": _repo_ref(CODE_REL),
            "config_identity": _repo_ref(CONFIG_REL),
            "runtime_identity_policy": "NOT_IDENTITY_BEARING",
            "runtime_identity": None,
            "dataset_schedule_identity": None,
            "contract_identities": [],
            "input_artifacts": inputs,
            "stochastic_identity": {
                "mode": "deterministic-provider-observation-registration",
                "scientific_execution_or_replay": "not_authorized",
            },
        },
        "side_effect_class": "PURE_ENGINEERING",
        "status": "PASS",
        "outputs": [_artifact(BINDING_REL, "private-remote-backup-binding", [x["digest"] for x in inputs])],
        "provenance": {
            "producer_scope": "ENGINEERING_GLOBAL",
            "experiment_id": None,
            "passport_sha256": None,
            "evidence_paths": [SOURCE_REL, JOIN_REL, PROVIDER_REL, FRESH_REL, DUP_REL, INCIDENT_REL, CAPACITY_REL],
            "external_views": [{
                "kind": "hf", "repository": REPO_ID, "revision": REVISION,
                "path": None, "sha256": _sha(ROOT / PROVIDER_REL),
                "evidence_path": PROVIDER_REL,
                "reason": "Private immutable provider readback is remote-backup evidence only; full fresh restore remains pending.",
            }],
        },

        "producer_attempt": {
            "attempt_id": f"{OPERATION_TYPE}-{STAMP}",
            "source": "BACKFILL", "execution_sha": None, "created_at": CREATED_AT,
        },
        "reuse": {
            "mechanical_eligibility": "ELIGIBLE",
            "default_policy": dag.DEFAULT_REUSE_POLICIES["PURE_ENGINEERING"],
            "operation_allowlisted": False,
            "automatic_reuse_permitted": False,
            "reason": "Remote backup registration only; full recovery, scientific reuse, deletion, and public release remain unauthorized.",
        },
        "created_at": CREATED_AT,
        "completed_at": CREATED_AT,
        "supersedes": [],
    }
    node["operation_digest"] = dag.operation_digest(node)

    prior, _ = dag.load_current_index(ROOT)
    entries = [e for e in prior["entries"] if e.get("operation_type") != OPERATION_TYPE]
    previous_count = len(entries)
    entries.append(dag.write_operation_node(ROOT, node))
    index = dag.build_index(entries, generated_at=CREATED_AT)
    router = dag.write_index_snapshot(ROOT, index, snapshot_rel=INDEX_REL)

    lookup = dag.query_exact(node, repo_root=ROOT)
    if lookup.get("status") != "REUSE_CANDIDATE":
        raise RuntimeError(f"unexpected DAG lookup: {lookup}")
    if lookup.get("reuse", {}).get("automatic_reuse_permitted") is not False:
        raise RuntimeError("remote-backup node must never auto-reuse")

    receipt = {
        "schema": f"openevo.gated-delta-private-remote-backup-dag-receipt.{STAMP}",
        "status": "PASS_REMOTE_BACKUP_DAG_REGISTRATION_FULL_FRESH_RESTORE_PENDING",
        "operation_type": OPERATION_TYPE,
        "operation_digest": node["operation_digest"],
        "lookup_status": lookup["status"],
        "automatic_reuse_permitted": False,
        "prior_entry_count": previous_count,
        "current_entry_count": len(index["entries"]),
        "dag_router": router,
        "authoritative_provider": {"repo_id": REPO_ID, "revision": REVISION},
        "fresh_restore_verified_count": 2,
        "fresh_restore_remaining_count": 11,
        "recovery_verified": False,
        "scientific_reuse_authorized": False,
        "provider_mutation_performed": False,
        "server_delete_performed": False,
        "provider_delete_performed": False,
    }
    _write_json(RECEIPT_REL, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
