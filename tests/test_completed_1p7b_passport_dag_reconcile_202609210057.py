"""Current completed-1.7B Passport / Artifact-DAG reconciliation regressions."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.openevo_webshop import experiment_artifact_dag as dag
from scripts.openevo_webshop import experiment_passport as passport
from scripts.openevo_webshop import experiment_registry as registry

ROOT = Path(__file__).resolve().parents[1]
STAMP = "202609210057"
EVID = ROOT / f"docs/evidence/experiment-passports/{STAMP}-completed-1p7b-production-dag"
Q17 = "202609082022-q17-directapply-no-gdr"
BOUNDED = "202609160255-bounded-effective-state-gdr"
BOUNDED_SHA = "5d9adea0312ea93f4a12fa561fae5fa5fa8bd8f0f5cbff7d86c7db51371b5e42"

PRODUCTION_TYPES = {
    "completed-1p7b.backfill.q17-publication",
    "completed-1p7b.backfill.q17-fresh-restore",
    "completed-1p7b.backfill.bounded-publication",
    "completed-1p7b.backfill.bounded-fresh-restore",
    "completed-1p7b.backfill.run-passport-control-plane",
    "completed-1p7b.backfill.asset-ledger",
}

def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))

def test_current_registry_routes_q17_and_bounded_as_complete_sealed() -> None:
    reg, router = registry.load_current_registry(ROOT)
    assert registry.validate_registry(reg, ROOT) == []
    assert router["path"].startswith("configs/experiment/experiment-registry-")
    entries = {row["experiment_id"]: row for row in reg["entries"]}
    assert {Q17, BOUNDED} <= set(entries)
    assert {entries[Q17]["lifecycle_class"], entries[BOUNDED]["lifecycle_class"]} == {"COMPLETE_SEALED"}
    assert entries[BOUNDED]["passport_sha256"] == BOUNDED_SHA
    assert "docs/evidence/experiment-passports/202609121011/Q17_RECOVERY_OBSERVATION_202609121011.json" in entries[Q17]["latest_authority_paths"]

def test_original_bounded_frozen_passport_validates_unchanged() -> None:
    path = ROOT / "docs/evidence/experiment-passports/202609160255-bounded-effective-state-gdr/FROZEN_PASSPORT_ORIGINAL.json"
    payload = passport.load_passport(path, root=ROOT)
    assert payload["status"] == "FROZEN"
    assert payload["experiment_id"] == BOUNDED
    assert payload["passport_payload_sha256"] == BOUNDED_SHA

def test_current_artifact_index_preserves_pilot_plus_six_production_backfills() -> None:
    index, router = dag.load_current_index(ROOT)
    assert dag.validate_index(index, repo_root=ROOT) == []
    assert router["path"].startswith(
        "configs/experiment/passport-artifacts/index/experiment-passport-artifact-index-"
    )
    assert len(index["entries"]) >= 7
    operation_types = {row["operation_type"] for row in index["entries"]}
    assert "passport-artifact-dag.pilot.deterministic-file-summary" in operation_types
    production = {
        row["operation_type"]
        for row in index["entries"]
        if row["operation_type"].startswith("completed-1p7b.backfill.")
    }
    assert production == PRODUCTION_TYPES

def test_all_production_backfills_are_verified_candidates_not_automatic_reuse() -> None:
    index, _ = dag.load_current_index(ROOT)
    rows = [row for row in index["entries"] if row["operation_type"] in PRODUCTION_TYPES]
    assert len(rows) == 6
    for entry in rows:
        node = dag._load_indexed_node(ROOT, entry)
        assert node["producer_attempt"]["source"] == "BACKFILL"
        assert node["reuse"]["automatic_reuse_permitted"] is False
        result = dag.query_exact(node, repo_root=ROOT, index=index)
        assert result["status"] == "REUSE_CANDIDATE"
        assert result["classification"] == "EXACT_CACHE_HIT"
        assert result["reuse"]["automatic_reuse_permitted"] is False

def test_run_passport_control_plane_is_current_135_of_135_and_keeps_unresolved_ids() -> None:
    rec = load(f"docs/evidence/experiment-passports/{STAMP}-completed-1p7b-production-dag/RUN_PASSPORT_135_REMOTE_VERIFY.json")
    assert rec["status"] == "PASS_REMOTE_VERIFIED_135_OF_135"
    assert rec["coverage"]["live_run_roots"] == 135
    assert rec["coverage"]["passport_index_count"] == 135
    assert rec["coverage"]["missing"] == []
    assert rec["coverage"]["stale"] == []
    assert rec["coverage"]["old_remote_passports_preserved_exact"] == 124
    assert rec["fresh_download_verify"]["status"] == "PASS"
    assert rec["fresh_download_verify"]["old_124_changed"] == 0
    assert rec["bounded_pair_root"]["canonical_run_id"] is None
    assert rec["bounded_final_closeout_root"]["canonical_run_id"] is None

def test_bounded_current_reconciliation_preserves_unresolved_run_id() -> None:
    rec = load(f"docs/evidence/experiment-passports/{STAMP}-completed-1p7b-production-dag/BOUNDED_CURRENT_MAIN_RECONCILIATION.json")
    assert rec["status"] == "PASS_CURRENT_MAIN_REGISTERED_RUN_ID_UNRESOLVED"
    assert rec["experiment_passport"]["payload_sha256"] == BOUNDED_SHA
    assert rec["experiment_passport"]["current_validator"] == "PASS"
    assert rec["experiment_registry"]["lifecycle_class"] == "COMPLETE_SEALED"
    assert rec["canonical_run_id_resolution"]["value"] is None
    assert rec["canonical_run_id_resolution"]["status"] == "UNRESOLVED_PRESERVED"
    assert rec["scientific_semantics_changed"] is False

def test_historical_publication_shadow_remains_historical_not_directly_promoted() -> None:
    current = load(f"docs/evidence/experiment-passports/{STAMP}-completed-1p7b-production-dag/PRODUCTION_DAG_BACKFILL_RECEIPT.json")
    assert current["source_shadow_status"] == "ADVISORY_SHADOW_ONLY"
    assert current["source_workline_pr"] == 555
    assert current["status"] == "PASS_CURRENT_DAG_PRODUCTION_BACKFILL"
    assert current["historical_shadow_promoted_directly"] is False
    assert current["production_node_count"] == 6
    assert current["current_entry_count"] == 7
    assert all(row["status"] == "REUSE_CANDIDATE" for row in current["lookups"])
    assert all(row["automatic_reuse_permitted"] is False for row in current["lookups"])

def test_backfill_module_has_no_provider_or_scientific_execution_surface() -> None:
    source = (ROOT / "scripts/openevo_webshop/completed_1p7b_artifact_dag_backfill.py").read_text(encoding="utf-8")
    forbidden = (
        "import subprocess", "import requests", "huggingface_hub", "subprocess.",
        "import torch", "torch.", "hf upload", "docker ", "nvidia-smi",
    )
    assert all(token not in source for token in forbidden)
