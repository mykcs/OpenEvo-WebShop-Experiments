import importlib.util
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/governance/audit_recovery_disclosure_policy.py"


def load_audit():
    spec = importlib.util.spec_from_file_location("recovery_disclosure_audit", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_repository_recovery_disclosure_policy_audit_passes():
    mod = load_audit()
    report = mod.audit(ROOT)
    assert report["status"] == "PASS", report["errors"]
    assert report["errors"] == []
    assert report["live_checks_required"]


def test_hf_registry_detects_private_public_mismatch():
    mod = load_audit()
    registry = {
        "state_model_policy": mod.POLICY,
        "scientific_preservation_policy": {
            "state_model_authority": mod.POLICY,
            "default_recovery_state_for_new_server_assets": "SERVER_ONLY",
            "default_disclosure_state": "PRIVATE",
        },
        "assets": [
            {
                "repo_id": "example/private-method",
                "recovery_state": "RECOVERY_VERIFIED",
                "recoverability_class": "REMOTE_VERIFIED_EXACT",
                "disclosure_state": "PRIVATE",
                "provider_visibility": "public",
            }
        ],
    }
    errors = mod.audit_hf_registry(registry)
    assert any("conflicts with public provider visibility" in e for e in errors)


def test_hf_registry_requires_embargo_release_trigger():
    mod = load_audit()
    registry = {
        "state_model_policy": mod.POLICY,
        "scientific_preservation_policy": {
            "state_model_authority": mod.POLICY,
            "default_recovery_state_for_new_server_assets": "SERVER_ONLY",
            "default_disclosure_state": "PRIVATE",
        },
        "assets": [],
        "proposed_public_mirror": {
            "disclosure_state": "EMBARGOED",
            "provider_visibility": "unknown",
        },
    }
    errors = mod.audit_hf_registry(registry)
    assert any("EMBARGOED requires release_trigger" in e for e in errors)


def test_public_catalog_rejects_private_projection():
    mod = load_audit()
    catalog = {
        "state_model_policy": mod.POLICY,
        "status": "GENERATED_PUBLIC_SAFE",
        "nodes": [
            {
                "canonical_id": "private-node",
                "disclosure_state": "PRIVATE",
                "provider_bindings": [],
            }
        ],
    }
    errors = mod.audit_public_catalog(catalog)
    assert any("not disclosure_state=PUBLIC" in e for e in errors)


def test_namespace_audit_detects_stale_planned_recovery_org_policy():
    mod = load_audit()
    registry = {
        "namespace_placement_policy": {
            "public_openevo_namespace": "openevo-recovery",
            "all_openevo_canonical_target_namespace": "openevo-recovery",
            "private_openevo_target_namespace": "openevo-recovery",
            "namespaces": {
                "openevo-recovery": {
                    "role": "OPEN_EVO_PROJECT_NAMESPACE",
                    "status": "ACTIVE_VERIFIED_20260921",
                }
            }
        },
        "namespace_live_snapshot_20260921": {
            "planned_recovery_org": {"state": "NOT_CREATED_OR_NOT_BOUND"},
            "snapshot_role": "HISTORICAL_PRE_TRANSFER",
            "superseded_by": "namespace_post_transfer_snapshot_202609211651",
        },
        "namespace_post_transfer_snapshot_202609211651": {
            "snapshot_role": "CURRENT_POST_TRANSFER_BASELINE",
            "openevo-recovery": {"placement_state": "HEALTHY"},
        },
    }
    stale_policy = "planned Recovery organization is unresolved"
    plan = "Project organization: **`openevo-recovery`**"
    errors = mod.audit_namespace_placement(registry, stale_policy, plan)
    assert any("canonical policy does not bind" in e for e in errors)


def test_namespace_audit_requires_historical_label_on_pretransfer_snapshot():
    mod = load_audit()
    registry = {
        "namespace_placement_policy": {
            "public_openevo_namespace": "openevo-recovery",
            "all_openevo_canonical_target_namespace": "openevo-recovery",
            "private_openevo_target_namespace": "openevo-recovery",
            "namespaces": {
                "openevo-recovery": {
                    "role": "OPEN_EVO_PROJECT_NAMESPACE",
                    "status": "ACTIVE_VERIFIED_20260921",
                }
            }
        },
        "namespace_live_snapshot_20260921": {
            "planned_recovery_org": {"state": "NOT_CREATED_OR_NOT_BOUND"},
        },
        "namespace_post_transfer_snapshot_202609211651": {
            "snapshot_role": "CURRENT_POST_TRANSFER_BASELINE",
            "openevo-recovery": {"placement_state": "HEALTHY"},
        },
    }
    policy = "`openevo-recovery` OPEN_EVO_PROJECT_NAMESPACE / ACTIVE_VERIFIED"
    plan = "Project organization: **`openevo-recovery`**"
    errors = mod.audit_namespace_placement(registry, policy, plan)
    assert any("not marked historical" in e for e in errors)


def test_namespace_audit_rejects_public_openevo_asset_in_personal_namespace():
    mod = load_audit()
    registry = {
        "namespace_placement_policy": {
            "public_openevo_namespace": "openevo-recovery",
            "all_openevo_canonical_target_namespace": "openevo-recovery",
            "private_openevo_target_namespace": "openevo-recovery",
            "namespaces": {
                "openevo-recovery": {
                    "role": "OPEN_EVO_PROJECT_NAMESPACE",
                    "status": "ACTIVE_VERIFIED_20260921",
                }
            },
        },
        "assets": [
            {
                "repo_id": "example-account/openevo-example-public",
                "disclosure_state": "PUBLIC",
                "provider_visibility": "public",
            }
        ],
        "namespace_post_transfer_snapshot_202609211651": {
            "snapshot_role": "CURRENT_POST_TRANSFER_BASELINE",
            "openevo-recovery": {"placement_state": "HEALTHY"},
        },
    }
    policy = "`openevo-recovery` OPEN_EVO_PROJECT_NAMESPACE / ACTIVE_VERIFIED"
    plan = "Project organization: **`openevo-recovery`**"
    errors = mod.audit_namespace_placement(registry, policy, plan)
    assert any("outside project namespace" in e for e in errors)


def test_public_catalog_rejects_personal_namespace_for_openevo_hf_binding():
    mod = load_audit()
    catalog = {
        "state_model_policy": mod.POLICY,
        "status": "GENERATED_PUBLIC_SAFE",
        "nodes": [
            {
                "canonical_id": "public-openevo-model",
                "disclosure_state": "PUBLIC",
                "provider_bindings": [
                    {
                        "provider": "huggingface",
                        "provider_id": "example-account/openevo-example-public",
                        "disclosure_state": "PUBLIC",
                        "provider_visibility": "public",
                    }
                ],
            }
        ],
    }
    errors = mod.audit_public_catalog(catalog)
    assert any("outside openevo-recovery" in e for e in errors)


def test_namespace_audit_rejects_private_openevo_current_canonical_in_personal_namespace():
    mod = load_audit()
    registry = {
        "namespace_placement_policy": {
            "public_openevo_namespace": "openevo-recovery",
            "all_openevo_canonical_target_namespace": "openevo-recovery",
            "private_openevo_target_namespace": "openevo-recovery",
            "namespaces": {"openevo-recovery": {"role": "OPEN_EVO_PROJECT_NAMESPACE", "status": "ACTIVE_VERIFIED_20260921"}},
        },
        "assets": [{"repo_id":"example-account/openevo-private-example","disclosure_state":"PRIVATE","provider_visibility":"private"}],
        "namespace_post_transfer_snapshot_202609211651": {"snapshot_role":"CURRENT_POST_TRANSFER_BASELINE","openevo-recovery":{"placement_state":"HEALTHY"}},
    }
    errors = mod.audit_namespace_placement(registry, "`openevo-recovery` OPEN_EVO_PROJECT_NAMESPACE / ACTIVE_VERIFIED", "Project organization: **`openevo-recovery`**")
    assert any("remains in personal namespace" in e for e in errors)


def test_namespace_audit_allows_kyo_private_only_with_capacity_blocker_and_org_target():
    mod = load_audit()
    base = {
        "namespace_placement_policy": {
            "public_openevo_namespace": "openevo-recovery",
            "all_openevo_canonical_target_namespace": "openevo-recovery",
            "private_openevo_target_namespace": "openevo-recovery",
            "namespaces": {"openevo-recovery": {"role": "OPEN_EVO_PROJECT_NAMESPACE", "status": "ACTIVE_VERIFIED_20260921"}},
        },
        "assets": [{
            "repo_id":"kyo-myk/openevo-capacity-blocked",
            "disclosure_state":"PRIVATE",
            "provider_visibility":"private",
            "target_namespace":"openevo-recovery",
            "namespace_migration_status":"CAPACITY_BLOCKED_FREE_ORG_100GB",
        }],
        "namespace_post_transfer_snapshot_202609211651": {"snapshot_role":"CURRENT_POST_TRANSFER_BASELINE","openevo-recovery":{"placement_state":"HEALTHY"}},
    }
    errors = mod.audit_namespace_placement(base, "`openevo-recovery` OPEN_EVO_PROJECT_NAMESPACE / ACTIVE_VERIFIED", "Project organization: **`openevo-recovery`**")
    assert not any("kyo-myk OpenEVO private overflow" in e for e in errors)
    base["assets"][0].pop("namespace_migration_status")
    errors = mod.audit_namespace_placement(base, "`openevo-recovery` OPEN_EVO_PROJECT_NAMESPACE / ACTIVE_VERIFIED", "Project organization: **`openevo-recovery`**")
    assert any("kyo-myk OpenEVO private overflow" in e for e in errors)


def test_namespace_audit_allows_only_the_owner_approved_gated_delta_private_overflow():
    mod = load_audit()
    record = {
        "role": "TEMPORARY_PRIVATE_OVERFLOW_CAPACITY_EXCEPTION",
        "status": "OWNER_APPROVED_PRIVATE_RECOVERY_PENDING_UPLOAD",
        "target_namespace": "openevo-recovery",
        "owner_authorization_evidence": "docs/evidence/GATED_DELTA_WEIGHT_PRIVATE_CAPACITY_EXCEPTION_20260928.json",
        "authorized_asset_scope": {
            "scientific_line": "Gated-Delta D1 paired and Route-S final state",
            "unique_sha_count": 13,
            "unique_bytes": 26984360168,
            "disclosure_state": "PRIVATE",
            "other_assets_allowed": False,
        },
    }
    registry = {
        "namespace_placement_policy": {
            "public_openevo_namespace": "openevo-recovery",
            "all_openevo_canonical_target_namespace": "openevo-recovery",
            "private_openevo_target_namespace": "openevo-recovery",
            "namespaces": {
                "openevo-recovery": {"role": "OPEN_EVO_PROJECT_NAMESPACE", "status": "ACTIVE_VERIFIED_20260921"},
                "openevo-recovery-2": record,
            },
        },
        "namespace_post_transfer_snapshot_202609211651": {"snapshot_role": "CURRENT_POST_TRANSFER_BASELINE", "openevo-recovery": {"placement_state": "HEALTHY"}},
    }
    policy = "`openevo-recovery` OPEN_EVO_PROJECT_NAMESPACE / ACTIVE_VERIFIED; openevo-recovery-2 approved for exactly the 13 unmatched Gated-Delta D1 paired assets"
    plan = "Project organization: **`openevo-recovery`**; openevo-recovery-2 for exactly the 13 unmatched Gated-Delta D1 paired weight payloads"
    assert mod.audit_namespace_placement(registry, policy, plan) == []

    record["authorized_asset_scope"]["unique_sha_count"] = 14
    errors = mod.audit_namespace_placement(registry, policy, plan)
    assert any("exact unique_sha_count=13" in error for error in errors)


def test_namespace_audit_accepts_remote_backed_gated_delta_only_while_fresh_restore_is_partial():
    mod = load_audit()
    repo = "openevo-recovery-2/openevo-gated-delta-d1-paired-private-recovery-20260928"
    record = {
        "role": "TEMPORARY_PRIVATE_OVERFLOW_CAPACITY_EXCEPTION",
        "status": "PRIVATE_REMOTE_BACKUP_PENDING_FULL_FRESH_RESTORE",
        "target_namespace": "openevo-recovery",
        "owner_authorization_evidence": "docs/evidence/recovery-disclosure-audit-20260926/GATED_DELTA_WEIGHT_PRIVATE_CAPACITY_EXCEPTION_20260928.json",
        "authorized_asset_scope": {
            "scientific_line": "Gated-Delta D1 paired and Route-S final state",
            "unique_sha_count": 13,
            "unique_bytes": 26984360168,
            "disclosure_state": "PRIVATE",
            "other_assets_allowed": False,
        },
        "upload_state": "REMOTE_BACKED_UP_PENDING_FULL_FRESH_RESTORE",
        "verified_repo_id": repo,
        "immutable_revision": "60234e69b1cb082fa57352a5b49b0d356cfc3e40",
        "provider_visibility": "private",
        "provider_sha_size_readback": "PASS_13_OF_13",
        "fresh_restore_verified_count": 2,
        "recovery_evidence": "docs/evidence/recovery-disclosure-audit-20260926/GATED_DELTA_WEIGHT_FRESH_RESTORE_STATUS_20260929.json",
    }
    registry = {
        "namespace_placement_policy": {
            "public_openevo_namespace": "openevo-recovery",
            "all_openevo_canonical_target_namespace": "openevo-recovery",
            "private_openevo_target_namespace": "openevo-recovery",
            "namespaces": {
                "openevo-recovery": {"role": "OPEN_EVO_PROJECT_NAMESPACE", "status": "ACTIVE_VERIFIED_20260921"},
                "openevo-recovery-2": record,
            },
        },
        "namespace_post_transfer_snapshot_202609211651": {
            "snapshot_role": "CURRENT_POST_TRANSFER_BASELINE",
            "openevo-recovery": {"placement_state": "HEALTHY"},
        },
        "assets": [{
            "repo_id": repo,
            "canonical": True,
            "recovery_state": "REMOTE_BACKED_UP",
            "fresh_restore_verified_count": 2,
            "disclosure_state": "PRIVATE",
            "provider_visibility": "private",
            "target_namespace": "openevo-recovery",
            "namespace_migration_status": "PRIVATE_OVERFLOW_REMOTE_BACKED_PENDING_FULL_FRESH_RESTORE",
        }],
    }
    bt = chr(96)
    policy = bt + "openevo-recovery" + bt + " OPEN_EVO_PROJECT_NAMESPACE / ACTIVE_VERIFIED; openevo-recovery-2 approved for Gated-Delta D1 paired"
    plan = "Project organization: **" + bt + "openevo-recovery" + bt + "**; openevo-recovery-2 for exactly the 13 unmatched Gated-Delta D1 paired weight payloads"
    assert mod.audit_namespace_placement(registry, policy, plan) == []

    record["fresh_restore_verified_count"] = 13
    errors = mod.audit_namespace_placement(registry, policy, plan)
    assert any("remain below 13/13 fresh restore" in error for error in errors)


def test_namespace_audit_accepts_duplicate_gated_delta_overflow_only_as_noncanonical_hold():
    mod = load_audit()
    registry = {
        "namespace_placement_policy": {
            "public_openevo_namespace": "openevo-recovery",
            "all_openevo_canonical_target_namespace": "openevo-recovery",
            "private_openevo_target_namespace": "openevo-recovery",
            "namespaces": {"openevo-recovery": {"role": "OPEN_EVO_PROJECT_NAMESPACE", "status": "ACTIVE_VERIFIED_20260921"}},
        },
        "assets": [{
            "repo_id": "openevo-recovery-2/gated-delta-d1-private-recovery-20260928",
            "canonical": False,
            "provider_delete_authorized": False,
            "disclosure_state": "PRIVATE",
            "provider_visibility": "private",
            "target_namespace": "openevo-recovery",
            "namespace_migration_status": "PRIVATE_OVERFLOW_DUPLICATE_HOLD_PENDING_OWNER_DECISION",
        }],
        "namespace_post_transfer_snapshot_202609211651": {
            "snapshot_role": "CURRENT_POST_TRANSFER_BASELINE",
            "openevo-recovery": {"placement_state": "HEALTHY"},
        },
    }
    bt = chr(96)
    policy = bt + "openevo-recovery" + bt + " OPEN_EVO_PROJECT_NAMESPACE / ACTIVE_VERIFIED"
    plan = "Project organization: **" + bt + "openevo-recovery" + bt + "**"
    assert mod.audit_namespace_placement(registry, policy, plan) == []

    registry["assets"][0]["provider_delete_authorized"] = True
    errors = mod.audit_namespace_placement(registry, policy, plan)
    assert any("non-canonical and non-deletable" in error for error in errors)


def test_namespace_audit_accepts_explicitly_registered_q17_additional_overflow():
    q17_repo = "openevo-recovery-2/openevo-q17-directapply-round32-65-archive-recovery-20260930"
    extra = {
        "repo_id": q17_repo,
        "repo_type": "dataset",
        "scientific_line": "Q17 DirectApply rounds 32-65 recovery archive",
        "provider_visibility": "private",
        "immutable_revision": "20850446df41dcd7b244561a4defd204fa096efc",
        "file_count": 36,
        "tree_bytes": 25270742978,
        "target_namespace": "openevo-recovery",
        "capacity_blocker": "openevo-recovery private capacity above heavy-write threshold",
        "status": "TEMPORARY_PRIVATE_OVERFLOW_REGISTERED_20261003",
        "visibility_change_authorized": False,
        "deletion_authorized": False,
    }
    overflow = {
        "role": "TEMPORARY_PRIVATE_OVERFLOW_CAPACITY_EXCEPTION",
        "status": "OWNER_APPROVED_PRIVATE_RECOVERY_PENDING_UPLOAD",
        "target_namespace": "openevo-recovery",
        "owner_authorization_evidence": "docs/evidence/GATED_DELTA_WEIGHT_PRIVATE_CAPACITY_EXCEPTION_20260928.json",
        "authorized_asset_scope": {
            "scientific_line": "Gated-Delta D1 paired and Route-S final state",
            "unique_sha_count": 13,
            "unique_bytes": 26984360168,
            "disclosure_state": "PRIVATE",
            "other_assets_allowed": False,
        },
        "additional_authorized_overflow_assets": [extra],
    }
    registry = {
        "namespace_placement_policy": {
            "public_openevo_namespace": "openevo-recovery",
            "all_openevo_canonical_target_namespace": "openevo-recovery",
            "private_openevo_target_namespace": "openevo-recovery",
            "namespaces": {
                "openevo-recovery": {"role": "OPEN_EVO_PROJECT_NAMESPACE", "status": "ACTIVE_VERIFIED_20260921"},
                "openevo-recovery-2": overflow,
            },
        },
        "assets": [{
            "repo_id": q17_repo,
            "canonical": True,
            "revision": extra["immutable_revision"],
            "recovery_state": "REMOTE_BACKED_UP",
            "disclosure_state": "PRIVATE",
            "provider_visibility": "private",
            "target_namespace": "openevo-recovery",
            "namespace_migration_status": "TEMPORARY_OVERFLOW_CAPACITY_BLOCKED_20261003",
            "publication_intent": "INTERNAL_PRIVATE",
            "retention_intent": "KEEP_EXACT",
        }],
        "namespace_post_transfer_snapshot_202609211651": {
            "snapshot_role": "CURRENT_POST_TRANSFER_BASELINE",
            "openevo-recovery": {"placement_state": "HEALTHY"},
        },
    }
    bt = chr(96)
    policy = bt + "openevo-recovery" + bt + " OPEN_EVO_PROJECT_NAMESPACE / ACTIVE_VERIFIED; openevo-recovery-2 approved for Gated-Delta D1 paired"
    plan = "Project organization: **" + bt + "openevo-recovery" + bt + "**; openevo-recovery-2 for exactly the 13 unmatched Gated-Delta D1 paired weight payloads"
    mod = load_audit()
    assert mod.audit_namespace_placement(registry, policy, plan) == []

    extra["deletion_authorized"] = True
    errors = mod.audit_namespace_placement(registry, policy, plan)
    assert any("additional overflow authorization must remain private and non-deletable" in error for error in errors)


def test_namespace_audit_accepts_verified_gated_delta_private_overflow_only_with_receipt_binding():
    mod = load_audit()
    verified_repo = "openevo-recovery-2/openevo-gated-delta-d1-paired-private-recovery-20260928"
    record = {
        "role": "TEMPORARY_PRIVATE_OVERFLOW_CAPACITY_EXCEPTION",
        "status": "PRIVATE_RECOVERY_VERIFIED_OVERFLOW_ACTIVE",
        "target_namespace": "openevo-recovery",
        "owner_authorization_evidence": "docs/evidence/recovery-disclosure-audit-20260926/GATED_DELTA_WEIGHT_PRIVATE_CAPACITY_EXCEPTION_20260928.json",
        "authorized_asset_scope": {
            "scientific_line": "Gated-Delta D1 paired and Route-S final state",
            "unique_sha_count": 13,
            "unique_bytes": 26984360168,
            "disclosure_state": "PRIVATE",
            "other_assets_allowed": False,
        },
        "upload_state": "RECOVERY_VERIFIED",
        "verified_repo_id": verified_repo,
        "immutable_revision": "60234e69b1cb082fa57352a5b49b0d356cfc3e40",
        "provider_visibility": "private",
        "fresh_restore_verified_count": 13,
        "fresh_restore_verified_bytes": 26984360168,
        "recovery_evidence": "docs/evidence/recovery-disclosure-audit-20260926/GATED_DELTA_WEIGHT_PRIVATE_RECOVERY_BINDING_20260929.json",
    }
    registry = {
        "namespace_placement_policy": {
            "public_openevo_namespace": "openevo-recovery",
            "all_openevo_canonical_target_namespace": "openevo-recovery",
            "private_openevo_target_namespace": "openevo-recovery",
            "namespaces": {
                "openevo-recovery": {"role": "OPEN_EVO_PROJECT_NAMESPACE", "status": "ACTIVE_VERIFIED_20260921"},
                "openevo-recovery-2": record,
            },
        },
        "namespace_post_transfer_snapshot_202609211651": {
            "snapshot_role": "CURRENT_POST_TRANSFER_BASELINE",
            "openevo-recovery": {"placement_state": "HEALTHY"},
        },
        "assets": [{
            "repo_id": verified_repo,
            "disclosure_state": "PRIVATE",
            "provider_visibility": "private",
            "target_namespace": "openevo-recovery",
            "namespace_migration_status": "PRIVATE_OVERFLOW_VERIFIED_PENDING_CANONICAL_CAPACITY",
            "canonical": True,
            "recovery_state": "RECOVERY_VERIFIED",
            "fresh_restore_verified_count": 13,
            "fresh_restore_verified_bytes": 26984360168,
            "recovery_evidence": "docs/evidence/recovery-disclosure-audit-20260926/GATED_DELTA_WEIGHT_FRESH_RESTORE_VERIFY_20260929.json",
        }],
    }
    bt = chr(96)
    policy = bt + "openevo-recovery" + bt + " OPEN_EVO_PROJECT_NAMESPACE / ACTIVE_VERIFIED; openevo-recovery-2 approved for Gated-Delta D1 paired"
    plan = "Project organization: **" + bt + "openevo-recovery" + bt + "**; openevo-recovery-2 for exactly the 13 unmatched Gated-Delta D1 paired weight payloads"
    assert mod.audit_namespace_placement(registry, policy, plan) == []

    record["provider_visibility"] = "public"
    errors = mod.audit_namespace_placement(registry, policy, plan)
    assert any("retain private provider visibility" in error for error in errors)

    record["provider_visibility"] = "private"
    registry["assets"][0]["fresh_restore_verified_bytes"] = 1
    errors = mod.audit_namespace_placement(registry, policy, plan)
    assert any("exact 26,984,360,168-byte restore" in error for error in errors)


def test_verified_gated_delta_audit_binds_each_checkpoint_to_live_readback_manifests():
    mod = load_audit()
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "docs/operations/publication/HUGGINGFACE_ASSET_REGISTRY.json").read_text())
    assert mod.audit_gated_delta_verified_evidence(root, registry) == []
    with tempfile.TemporaryDirectory() as tmp:
        fixture_root = Path(tmp)
        relatives = [
            registry["namespace_placement_policy"]["namespaces"]["openevo-recovery-2"]["recovery_evidence"],
            "docs/evidence/recovery-disclosure-audit-20260926/GATED_DELTA_WEIGHT_PROVIDER_MANIFEST_20260929.json",
            "docs/evidence/recovery-disclosure-audit-20260926/GATED_DELTA_WEIGHT_PROVIDER_READBACK_20260929.json",
            "docs/evidence/recovery-disclosure-audit-20260926/GATED_DELTA_WEIGHT_SAFETENSORS_HEADER_AUDIT_20260929.json",
        ]
        for relative in relatives:
            destination = fixture_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, destination)
        proof_path = fixture_root / relatives[0]
        proof = json.loads(proof_path.read_text())
        proof["fresh_restore"]["files"][0]["header_sha256"] = "0" * 64
        proof_path.write_text(json.dumps(proof))
        errors = mod.audit_gated_delta_verified_evidence(fixture_root, registry)
        assert any("safetensors header mismatch" in error for error in errors)


def test_namespace_audit_rejects_public_asset_in_private_overflow_namespace():
    mod = load_audit()
    registry = {
        "namespace_placement_policy": {
            "public_openevo_namespace": "openevo-recovery",
            "all_openevo_canonical_target_namespace": "openevo-recovery",
            "private_openevo_target_namespace": "openevo-recovery",
            "namespaces": {"openevo-recovery": {"role": "OPEN_EVO_PROJECT_NAMESPACE", "status": "ACTIVE_VERIFIED_20260921"}},
        },
        "assets": [{"repo_id": "openevo-recovery-2/example", "disclosure_state": "PUBLIC", "provider_visibility": "public"}],
        "namespace_post_transfer_snapshot_202609211651": {"snapshot_role": "CURRENT_POST_TRANSFER_BASELINE", "openevo-recovery": {"placement_state": "HEALTHY"}},
    }
    errors = mod.audit_namespace_placement(registry, "`openevo-recovery`", "Project organization: **`openevo-recovery`**")
    assert any("public OpenEVO asset canonical repo_id is outside project namespace" in error for error in errors)
    assert any("overflow asset must remain private" in error for error in errors)


def test_namespace_audit_enforces_public_collection_language_order_and_scope_snapshot():
    mod = load_audit()
    registry = {
        "namespace_placement_policy": {
            "public_openevo_namespace": "openevo-recovery",
            "all_openevo_canonical_target_namespace": "openevo-recovery",
            "private_openevo_target_namespace": "openevo-recovery",
            "namespaces": {"openevo-recovery": {"role": "OPEN_EVO_PROJECT_NAMESPACE", "status": "ACTIVE_VERIFIED_20260921"}},
        },
        "openevo_public_collection_current": {"private":False,"item_count":13,"required_item":"openevo-recovery/openevo-standards","language_order":"Chinese first, English second"},
        "namespace_post_transfer_snapshot_202609211651": {"snapshot_role":"CURRENT_POST_TRANSFER_BASELINE","openevo-recovery":{"placement_state":"HEALTHY"}},
    }
    errors = mod.audit_namespace_placement(registry, "`openevo-recovery` OPEN_EVO_PROJECT_NAMESPACE / ACTIVE_VERIFIED", "Project organization: **`openevo-recovery`**")
    assert not any("Collection" in e for e in errors)


def test_hf_registry_current_requires_publication_and_retention_intents_for_private_assets():
    mod = load_audit()
    registry = {
        "state_model_policy": mod.POLICY,
        "scientific_preservation_policy": {
            "state_model_authority": mod.POLICY,
            "default_recovery_state_for_new_server_assets": "SERVER_ONLY",
            "default_disclosure_state": "PRIVATE",
        },
        "research_asset_standard_202610231400": {
            "status": "CURRENT",
            "human_entry": "docs/operations/publication/README.md",
            "publication_intent": sorted(mod.PUBLICATION_INTENTS),
            "retention_intent": sorted(mod.RETENTION_INTENTS),
        },
        "assets": [{
            "repo_id": "openevo-recovery/private-example",
            "recovery_state": "RECOVERY_VERIFIED",
            "recoverability_class": "REMOTE_VERIFIED_EXACT",
            "disclosure_state": "PRIVATE",
            "provider_visibility": "private",
        }],
    }
    errors = mod.audit_hf_registry(registry)
    assert any("publication_intent" in e for e in errors)
    assert any("retention_intent" in e for e in errors)
    registry["assets"][0]["publication_intent"] = "INTERNAL_PRIVATE"
    registry["assets"][0]["retention_intent"] = "KEEP_EXACT"
    errors = mod.audit_hf_registry(registry)
    assert not any("publication_intent" in e or "retention_intent" in e for e in errors)


def test_hf_registry_current_requires_completed_payload_ledger_for_decompose_required():
    mod = load_audit()
    registry = {
        "state_model_policy": mod.POLICY,
        "scientific_preservation_policy": {
            "state_model_authority": mod.POLICY,
            "default_recovery_state_for_new_server_assets": "SERVER_ONLY",
            "default_disclosure_state": "PRIVATE",
        },
        "research_asset_standard_202610231400": {
            "status": "CURRENT",
            "human_entry": "docs/operations/publication/README.md",
            "publication_intent": sorted(mod.PUBLICATION_INTENTS),
            "retention_intent": sorted(mod.RETENTION_INTENTS),
        },
        "assets": [{
            "repo_id": "openevo-recovery/mixed-private-example",
            "recovery_state": "RECOVERY_VERIFIED",
            "recoverability_class": "REMOTE_VERIFIED_EXACT",
            "disclosure_state": "PRIVATE",
            "provider_visibility": "private",
            "publication_intent": "INTERNAL_PRIVATE",
            "retention_intent": "DECOMPOSE_REQUIRED",
        }],
    }
    errors = mod.audit_hf_registry(registry)
    assert any("payload_level_classification_required" in e for e in errors)
    assert any("payload_decomposition_status" in e for e in errors)
    assert any("payload_decomposition_evidence" in e for e in errors)
    asset = registry["assets"][0]
    asset["payload_level_classification_required"] = True
    asset["payload_decomposition_status"] = "COMPLETE_202609220158"
    asset["payload_decomposition_evidence"] = "docs/evidence/example.json"
    errors = mod.audit_hf_registry(registry)
    assert not any("DECOMPOSE_REQUIRED" in e for e in errors)


def test_local_deletion_receipt_links_bind_repository_bytes_not_server_originals():
    registry = json.loads((ROOT / "docs/operations/publication/HUGGINGFACE_ASSET_REGISTRY.json").read_text())
    for asset in registry["assets"]:
        if "deletion_receipt" not in asset:
            continue
        path = (ROOT / asset["deletion_receipt"]).resolve(strict=True)
        assert path.is_relative_to(ROOT.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == asset["deletion_receipt_sha256"]
        receipt = json.loads(path.read_text())
        assert receipt["server_receipt_sha256"] == asset["server_deletion_receipt_sha256"]
        passport = json.loads((ROOT / asset["recovery_evidence"]).read_text())
        for key in ("deletion_receipt", "deletion_receipt_sha256", "server_deletion_receipt_sha256"):
            assert passport["reclaim"][key] == asset[key]
        assert receipt["recovery"]["immutable_revision"] == asset["revision"]
        assert receipt["authorization"]["manifest_sha256"] == passport["reclaim"]["manifest_sha256"]
