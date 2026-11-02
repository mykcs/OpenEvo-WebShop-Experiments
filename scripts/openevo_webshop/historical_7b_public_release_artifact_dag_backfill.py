#!/usr/bin/env python3
"""Backfill the verified historical 7B public release into the current Passport Artifact DAG.

This is engineering/provenance only. The historical 7B lineage has no current FROZEN
Experiment Passport binding, so this script deliberately keeps experiment_id and
passport_sha256 null rather than inventing scientific identity.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.openevo_webshop import experiment_artifact_dag as dag

ROOT = Path(__file__).resolve().parents[2]
STAMP = "202609212302"
CREATED_AT = "2026-09-21T23:02:39+08:00"
CODE_REL = "scripts/openevo_webshop/historical_7b_public_release_artifact_dag_backfill.py"
OUT_DIR = ROOT / f"docs/evidence/experiment-passports/{STAMP}-historical-7b-public-release-dag"
CONFIG_REL = f"docs/evidence/experiment-passports/{STAMP}-historical-7b-public-release-dag/BACKFILL_CONFIG_{STAMP}.json"
INDEX_REL = f"configs/experiment/passport-artifacts/index/experiment-passport-artifact-index-{STAMP}.json"
RELEASE_RECEIPT = "docs/evidence/huggingface-stage2-checkpoint-decomposition-20260921/QWEN25_7B_PUBLIC_RELEASE_EXECUTION_202609212302.json"
LINEAGE_INVENTORY = "docs/evidence/experiment-passports/202609111853/LINEAGE_INVENTORY_202609111853.json"
BINDING_REL = f"docs/evidence/experiment-passports/{STAMP}-historical-7b-public-release-dag/HISTORICAL_7B_PUBLIC_RELEASE_DAG_BINDING.json"
DAG_RECEIPT_REL = f"docs/evidence/experiment-passports/{STAMP}-historical-7b-public-release-dag/PRODUCTION_DAG_BACKFILL_RECEIPT.json"
OPERATION_TYPE = "historical-7b.backfill.public-final-adapter-release"
PUBLIC_REPO = "example-account/openevo-ceiling1-qwen25-7b-historical-final-adapters"
PUBLIC_REVISION = "e18e0c7b02c247717f76c0b9fe2273c61b338fc4"
LEGACY_LINEAGE_ID = "qwen25-7b-long-run"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(rel: str, value: Any) -> Path:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path.write_bytes(raw)
    return path


def _repo_ref(rel: str) -> dict[str, Any]:
    p = ROOT / rel
    if not p.is_file():
        raise RuntimeError(f"missing repository evidence: {rel}")
    return {
        "kind": "repository_file",
        "repository": None,
        "revision": None,
        "path": rel,
        "sha256": _sha(p),
        "evidence_path": None,
        "reason": None,
    }


def _artifact(rel: str, role: str, derived: list[str] | None = None) -> dict[str, Any]:
    p = ROOT / rel
    digest = _sha(p)
    return {
        "artifact_type": "json-evidence",
        "digest_algorithm": "sha256",
        "digest": digest,
        "size_bytes": p.stat().st_size,
        "logical_role": role,
        "locations": [_repo_ref(rel)],
        "derived_from_digests": sorted(set(derived or [])),
    }


def main() -> int:
    release = json.loads((ROOT / RELEASE_RECEIPT).read_text())
    if release.get("status") != "PASS_PUBLIC_RELEASE_EXACT_AND_RECONSTRUCTIBLE":
        raise RuntimeError("historical 7B release receipt is not closed")
    provider = release.get("public_provider", {})
    if provider.get("repo_id") != PUBLIC_REPO or provider.get("immutable_revision") != PUBLIC_REVISION:
        raise RuntimeError("public provider identity mismatch")
    if provider.get("provider_visibility") != "public":
        raise RuntimeError("public provider visibility not verified")
    lineage_text = (ROOT / LINEAGE_INVENTORY).read_text()
    if LEGACY_LINEAGE_ID not in lineage_text or '"identity_trust": "INDEX_ONLY"' not in lineage_text:
        raise RuntimeError("legacy lineage inventory does not preserve INDEX_ONLY identity boundary")

    config = {
        "schema": f"openevo.historical-7b-public-release-dag-backfill-config.{STAMP}",
        "operation_type": OPERATION_TYPE,
        "legacy_lineage_id": LEGACY_LINEAGE_ID,
        "experiment_id": None,
        "passport_sha256": None,
        "release_receipt": RELEASE_RECEIPT,
        "public_provider": {"repo_id": PUBLIC_REPO, "revision": PUBLIC_REVISION},
        "hard_boundaries": {
            "new_experiment_identity_invented": False,
            "scientific_replay_authorized": False,
            "model_update_reuse_authorized": False,
            "final_evaluation_reuse_authorized": False,
            "provider_mutation_authorized_by_this_backfill": False,
            "private_duplicate_retirement_authorized": False,
        },
    }
    _write_json(CONFIG_REL, config)

    binding = {
        "schema": f"openevo.historical-7b-public-release-dag-binding.{STAMP}",
        "status": "PASS_BACKFILLED_FROM_VERIFIED_PUBLIC_RELEASE",
        "legacy_lineage_id": LEGACY_LINEAGE_ID,
        "identity_status": "LEGACY_INDEX_ONLY_NO_NEW_PASSPORT_IDENTITY",
        "experiment_id": None,
        "passport_sha256": None,
        "source_evidence": [
            {"path": RELEASE_RECEIPT, "sha256": _sha(ROOT / RELEASE_RECEIPT), "size_bytes": (ROOT / RELEASE_RECEIPT).stat().st_size},
            {"path": LINEAGE_INVENTORY, "sha256": _sha(ROOT / LINEAGE_INVENTORY), "size_bytes": (ROOT / LINEAGE_INVENTORY).stat().st_size},
        ],
        "public_provider": {
            "repo_id": PUBLIC_REPO,
            "revision": PUBLIC_REVISION,
            "provider_visibility": "public",
            "disclosure_state": "PUBLIC",
            "recovery_state": "RECOVERY_VERIFIED",
        },
        "interpretation": "The verified public adapter package is bound to the existing historical 7B legacy lineage as provider/provenance evidence only. No FROZEN Passport, canonical_run_id, training event, or scientific result is synthesized by this backfill.",
        "automatic_scientific_reuse_authorized": False,
        "scientific_semantics_changed": False,
    }
    _write_json(BINDING_REL, binding)

    inputs = [_artifact(RELEASE_RECEIPT, "public-release-receipt"), _artifact(LINEAGE_INVENTORY, "legacy-lineage-inventory")]
    derived = [item["digest"] for item in inputs]
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
            "stochastic_identity": {"mode": "deterministic-backfill", "randomness": "none"},
        },
        "side_effect_class": "PURE_ENGINEERING",
        "status": "PASS",
        "outputs": [_artifact(BINDING_REL, "backfill-binding", derived)],
        "provenance": {
            "producer_scope": "ENGINEERING_GLOBAL",
            "experiment_id": None,
            "passport_sha256": None,
            "evidence_paths": [LINEAGE_INVENTORY, RELEASE_RECEIPT],
            "external_views": [{
                "kind": "hf",
                "repository": PUBLIC_REPO,
                "revision": PUBLIC_REVISION,
                "path": None,
                "sha256": _sha(ROOT / RELEASE_RECEIPT),
                "evidence_path": RELEASE_RECEIPT,
                "reason": "public immutable revision bound by exact anonymous fresh-readback release evidence; network view is not scientific reuse authority",
            }],
        },
        "producer_attempt": {
            "attempt_id": f"{OPERATION_TYPE}-{STAMP}",
            "source": "BACKFILL",
            "execution_sha": None,
            "created_at": CREATED_AT,
        },
        "reuse": {
            "mechanical_eligibility": "ELIGIBLE",
            "default_policy": dag.DEFAULT_REUSE_POLICIES["PURE_ENGINEERING"],
            "operation_allowlisted": False,
            "automatic_reuse_permitted": False,
            "reason": "This is a deterministic provider/provenance backfill only. Historical model-state consumption remains subject to the original scientific authority.",
        },
        "created_at": CREATED_AT,
        "completed_at": CREATED_AT,
        "supersedes": [],
    }
    node["operation_digest"] = dag.operation_digest(node)

    prior, _ = dag.load_current_index(ROOT)
    entries = [entry for entry in prior["entries"] if entry.get("operation_type") != OPERATION_TYPE]
    predecessor_entry_count = len(entries)
    entries.append(dag.write_operation_node(ROOT, node))
    index = dag.build_index(entries, generated_at=CREATED_AT)
    router = dag.write_index_snapshot(ROOT, index, snapshot_rel=INDEX_REL)

    lookup = dag.query_exact(node, repo_root=ROOT)
    if lookup.get("status") != "REUSE_CANDIDATE" or lookup.get("reuse", {}).get("automatic_reuse_permitted") is not False:
        raise RuntimeError(f"unexpected DAG lookup result: {lookup}")

    receipt = {
        "schema": f"openevo.historical-7b-public-release-production-dag-backfill-receipt.{STAMP}",
        "status": "PASS_CURRENT_DAG_PROVIDER_BACKFILL",
        "legacy_lineage_id": LEGACY_LINEAGE_ID,
        "experiment_id": None,
        "passport_sha256": None,
        "operation_type": OPERATION_TYPE,
        "operation_digest": node["operation_digest"],
        "lookup_status": lookup["status"],
        "automatic_reuse_permitted": False,
        "prior_entry_count": predecessor_entry_count,
        "current_entry_count": len(index["entries"]),
        "dag_router": router,
        "public_provider": {"repo_id": PUBLIC_REPO, "revision": PUBLIC_REVISION},
        "new_scientific_identity_invented": False,
        "provider_mutation_performed_by_backfill": False,
        "private_delete_performed": False,
    }
    _write_json(DAG_RECEIPT_REL, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
