from __future__ import annotations

import json
from pathlib import Path

from scripts.openevo_webshop import experiment_artifact_dag as dag

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/recovery-disclosure-audit-20260926"
OPERATION_TYPE = "gated-delta-weight.register.private-remote-backup-pending-full-fresh-restore"
VERIFIED_OPERATION_TYPE = "gated-delta-weight.register.private-recovery-verified-after-fresh-restore"
AUTHORITATIVE_REPO = "openevo-recovery-2/openevo-gated-delta-d1-paired-private-recovery-20260928"
AUTHORITATIVE_REV = "60234e69b1cb082fa57352a5b49b0d356cfc3e40"
DUPLICATE_REPO = "openevo-recovery-2/gated-delta-d1-private-recovery-20260928"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_remote_backup_does_not_claim_full_recovery() -> None:
    fresh = load(EVIDENCE / "GATED_DELTA_WEIGHT_FRESH_RESTORE_STATUS_20260929.json")
    assert fresh["status"] == "REMOTE_BACKED_UP_FULL_FRESH_RESTORE_PENDING"
    assert fresh["repo_id"] == AUTHORITATIVE_REPO
    assert fresh["immutable_revision"] == AUTHORITATIVE_REV

    assert fresh["provider_sha_size_readback"]["status"] == "PASS_13_OF_13"
    assert fresh["fresh_restore"]["verified_count"] == 2
    assert fresh["fresh_restore"]["remaining_count"] == 11
    assert fresh["claim_boundary"]["remote_backed_up"] is True
    assert fresh["claim_boundary"]["recovery_verified"] is False
    assert fresh["claim_boundary"]["server_delete_authorized"] is False
    assert fresh["claim_boundary"]["provider_delete_authorized"] is False

    registry = load(ROOT / "docs/operations/publication/HUGGINGFACE_ASSET_REGISTRY.json")
    assets = {item["repo_id"]: item for item in registry["assets"]}
    authoritative = assets[AUTHORITATIVE_REPO]
    duplicate = assets[DUPLICATE_REPO]
    assert authoritative["canonical"] is True
    assert authoritative["recovery_state"] == "RECOVERY_VERIFIED"
    assert authoritative["fresh_restore_verified_count"] == 13
    assert authoritative["provider_delete_authorized"] is False
    assert duplicate["canonical"] is False
    assert duplicate["provider_visibility"] == "private"
    assert duplicate["provider_delete_authorized"] is False

    overflow = registry["namespace_placement_policy"]["namespaces"]["openevo-recovery-2"]
    assert overflow["status"] == "PRIVATE_RECOVERY_VERIFIED_OVERFLOW_ACTIVE"
    assert overflow["upload_state"] == "RECOVERY_VERIFIED"
    assert overflow["fresh_restore_verified_count"] == 13

    capacity = load(EVIDENCE / "GATED_DELTA_WEIGHT_PRIVATE_CAPACITY_EXCEPTION_20260928.json")
    assert capacity["transfer_state"]["recovery_state"] == "REMOTE_BACKED_UP_PENDING_FULL_FRESH_RESTORE"
    assert capacity["transfer_state"]["fresh_download_sha_size_readback"].startswith("PARTIAL_2_OF_13")
    assert capacity["owner_authorization"]["server_deletion_authorized"] is False
    assert capacity["owner_authorization"]["public_release_authorized"] is False


def test_passport_node_registers_remote_backup_without_reuse_authority() -> None:
    index, _ = dag.load_current_index(ROOT)
    entries = [e for e in index["entries"] if e["operation_type"] == OPERATION_TYPE]
    assert len(entries) == 1
    entry = entries[0]
    node = json.loads((ROOT / entry["path"]).read_text())
    assert node["status"] == "PASS"
    assert node["side_effect_class"] == "PURE_ENGINEERING"
    assert node["reuse"]["automatic_reuse_permitted"] is False
    assert node["reuse"]["operation_allowlisted"] is False

    result = dag.query_exact(node, repo_root=ROOT)
    assert result["status"] == "REUSE_CANDIDATE"
    assert result["reuse"]["automatic_reuse_permitted"] is False

    binding_rel = node["outputs"][0]["locations"][0]["path"]
    binding = load(ROOT / binding_rel)
    assert binding["status"] == "PASS_REMOTE_BACKUP_REGISTERED_FULL_FRESH_RESTORE_PENDING"
    assert binding["authoritative_provider"]["repo_id"] == AUTHORITATIVE_REPO
    assert binding["authoritative_provider"]["immutable_revision"] == AUTHORITATIVE_REV
    assert binding["fresh_restore"]["verified_count"] == 2
    assert binding["fresh_restore"]["remaining_count"] == 11
    assert binding["automatic_reuse_authorized"] is False
    assert binding["public_release_authorized"] is False
    assert binding["server_delete_authorized"] is False
    assert binding["provider_delete_authorized"] is False


def test_full_fresh_restore_successor_promotes_only_current_state() -> None:
    fresh = load(EVIDENCE / "GATED_DELTA_WEIGHT_FRESH_RESTORE_VERIFY_20260929.json")
    assert fresh["status"] == "RECOVERY_VERIFIED"
    assert fresh["fresh_restore"]["verified_count"] == 13
    assert fresh["fresh_restore"]["verified_bytes"] == 26984360168
    assert fresh["fresh_restore"]["remaining_count"] == 0
    assert fresh["fresh_restore"]["sha256_size_header_status"] == "PASS_13_OF_13"
    assert all(item["header_status"] == "PASS" for item in fresh["fresh_restore"]["files"])
    assert fresh["claim_boundary"]["recovery_verified"] is True
    assert fresh["claim_boundary"]["scientific_reuse_authorized"] is False
    assert fresh["claim_boundary"]["server_delete_authorized"] is False
    assert fresh["claim_boundary"]["provider_delete_authorized"] is False

    registry = load(ROOT / "docs/operations/publication/HUGGINGFACE_ASSET_REGISTRY.json")
    assets = {item["repo_id"]: item for item in registry["assets"]}
    authoritative = assets[AUTHORITATIVE_REPO]
    assert authoritative["recovery_state"] == "RECOVERY_VERIFIED"
    assert authoritative["fresh_restore_verified_count"] == 13
    assert authoritative["recovery_evidence"].endswith("GATED_DELTA_WEIGHT_FRESH_RESTORE_VERIFY_20260929.json")
    duplicate = assets[DUPLICATE_REPO]
    assert duplicate["canonical"] is False
    assert duplicate["provider_visibility"] == "private"
    assert duplicate["provider_delete_authorized"] is False

    index, _ = dag.load_current_index(ROOT)
    entries = [entry for entry in index["entries"] if entry["operation_type"] == VERIFIED_OPERATION_TYPE]
    assert len(entries) == 1
    node = load(ROOT / entries[0]["path"])
    assert node["supersedes"] == ["68704422695faae75a3bb0f0f728a91a351ac3f37df82dac6237268066147efc"]
    assert node["reuse"]["automatic_reuse_permitted"] is False
    result = dag.query_exact(node, repo_root=ROOT)
    assert result["reuse"]["automatic_reuse_permitted"] is False
    binding_rel = node["outputs"][0]["locations"][0]["path"]
    binding = load(ROOT / binding_rel)
    assert binding["status"] == "PASS_PRIVATE_FULL_FRESH_RESTORE_VERIFIED"
    assert binding["fresh_restore"]["verified_count"] == 13
    assert binding["public_release_authorized"] is False
