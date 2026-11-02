#!/usr/bin/env python3
"""Backfill verified completed-1.7B publication/recovery evidence into the current Passport Artifact DAG.

This records engineering provenance only. It never authorizes scientific replay, model update,
final evaluation, provider mutation, or server deletion.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.openevo_webshop import experiment_artifact_dag as dag
from scripts.openevo_webshop import experiment_passport as passport
from scripts.openevo_webshop import experiment_registry as registry

ROOT = Path(__file__).resolve().parents[2]
STAMP = "202609210057"
OUT_DIR = ROOT / f"docs/evidence/experiment-passports/{STAMP}-completed-1p7b-production-dag"
CONFIG_REL = f"docs/evidence/experiment-passports/{STAMP}-completed-1p7b-production-dag/BACKFILL_CONFIG_{STAMP}.json"
INDEX_REL = f"configs/experiment/passport-artifacts/index/experiment-passport-artifact-index-{STAMP}.json"
CODE_REL = "scripts/openevo_webshop/completed_1p7b_artifact_dag_backfill.py"

Q17_ID = "202609082022-q17-directapply-no-gdr"
Q17_PASSPORT = "configs/experiment/passports/202609082022-q17-directapply-no-gdr/passport.json"
Q17_PASSPORT_SHA = "155e8885d2af2d4ee8b281f75cfec5c8b3af18b2c42b63f440da2af3e00805b6"
BOUNDED_ID = "202609160255-bounded-effective-state-gdr"
BOUNDED_PASSPORT = "docs/evidence/experiment-passports/202609160255-bounded-effective-state-gdr/FROZEN_PASSPORT_ORIGINAL.json"
BOUNDED_PASSPORT_SHA = "5d9adea0312ea93f4a12fa561fae5fa5fa8bd8f0f5cbff7d86c7db51371b5e42"

EVID = "docs/evidence/completed-1p7b-publication-20260918"

def _historical_shadow_rel() -> str:
    root = ROOT / EVID
    matches = []
    for path in sorted(root.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if value.get("status") == "ADVISORY_SHADOW_ONLY" and value.get("generated_from", {}).get("workline_pr") == 555:
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"expected one historical publication shadow, found {len(matches)}")
    return matches[0].relative_to(ROOT).as_posix()

CUR = f"docs/evidence/experiment-passports/{STAMP}-completed-1p7b-production-dag"

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _write_json(rel: str, value: Any) -> Path:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if path.exists() and path.read_bytes() != raw:
        path.write_bytes(raw)
    elif not path.exists():
        path.write_bytes(raw)
    return path

def _repo_ref(rel: str) -> dict[str, Any]:
    p = ROOT / rel
    if not p.is_file():
        raise RuntimeError(f"missing repository evidence: {rel}")
    return {"kind": "repository_file", "repository": None, "revision": None, "path": rel,
            "sha256": _sha(p), "evidence_path": None, "reason": None}

def _hf_ref(repo: str, revision: str, evidence_rel: str) -> dict[str, Any]:
    # sha256 deliberately binds the exact Git evidence record proving the provider revision.
    # HF remains an external view; current DAG MVP does not treat network-only pointers as local reuse proof.
    return {"kind": "hf", "repository": repo, "revision": revision, "path": None,
            "sha256": _sha(ROOT / evidence_rel), "evidence_path": evidence_rel,
            "reason": "immutable HF revision bound by exact repository evidence; network view is not automatic reuse authority"}

def _artifact(rel: str, role: str, derived: list[str] | None = None) -> dict[str, Any]:
    p = ROOT / rel
    digest = _sha(p)
    return {"artifact_type": "json-evidence", "digest_algorithm": "sha256", "digest": digest,
            "size_bytes": p.stat().st_size, "logical_role": role, "locations": [_repo_ref(rel)],
            "derived_from_digests": sorted(set(derived or []))}

def _source_rows(paths: list[str]) -> list[dict[str, Any]]:
    return [{"path": rel, "sha256": _sha(ROOT / rel), "size_bytes": (ROOT / rel).stat().st_size} for rel in paths]
def _reuse(side_effect: str) -> dict[str, Any]:
    return {
        "mechanical_eligibility": "ELIGIBLE",
        "default_policy": dag.DEFAULT_REUSE_POLICIES[side_effect],
        "operation_allowlisted": False,
        "automatic_reuse_permitted": False,
        "reason": "Verified historical engineering evidence may be surfaced as REUSE_CANDIDATE only; consumer/scientific authority remains external to the DAG.",
    }

def _node(*, operation_type: str, side_effect: str, input_paths: list[tuple[str, str]],
          output_rel: str, experiment_id: str | None, passport_sha: str | None, passport_rel: str | None,
          external_views: list[dict[str, Any]], created_at: str) -> dict[str, Any]:
    inputs = [_artifact(rel, role) for rel, role in input_paths]
    derived = [item["digest"] for item in inputs]
    contracts = []
    if passport_rel is not None:
        contracts.append({"name": "experiment-passport", "identity": _repo_ref(passport_rel)})
    node = {
        "schema": dag.NODE_SCHEMA_ID,
        "node_kind": "operation",
        "operation_type": operation_type,
        "operation_schema_version": STAMP,
        "operation_digest": "0" * 64,
        "identity_payload": {
            "code_identity": _repo_ref(CODE_REL),
            "config_identity": _repo_ref(CONFIG_REL),
            "runtime_identity_policy": "NOT_IDENTITY_BEARING",
            "runtime_identity": None,
            "dataset_schedule_identity": None,
            "contract_identities": contracts,
            "input_artifacts": inputs,
            "stochastic_identity": {"mode": "deterministic-backfill", "randomness": "none"},
        },
        "side_effect_class": side_effect,
        "status": "PASS",
        "outputs": [_artifact(output_rel, "backfill-binding", derived)],
        "provenance": {
            "producer_scope": "EXPERIMENT" if experiment_id else "ENGINEERING_GLOBAL",
            "experiment_id": experiment_id,
            "passport_sha256": passport_sha,
            "evidence_paths": sorted({rel for rel, _ in input_paths}),
            "external_views": external_views,
        },
        "producer_attempt": {"attempt_id": f"{operation_type}-{STAMP}", "source": "BACKFILL",
                             "execution_sha": None, "created_at": created_at},
        "reuse": _reuse(side_effect),
        "created_at": created_at,
        "completed_at": created_at,
        "supersedes": [],
    }
    node["operation_digest"] = dag.operation_digest(node)
    return node

def _binding(schema: str, experiment_id: str | None, passport_sha: str | None,
             sources: list[str], remote_views: list[dict[str, str]], interpretation: str) -> dict[str, Any]:
    return {
        "schema": schema, "status": "PASS_BACKFILLED_FROM_VERIFIED_EVIDENCE",
        "experiment_id": experiment_id, "passport_sha256": passport_sha,
        "source_evidence": _source_rows(sources), "remote_views": remote_views,
        "interpretation": interpretation,
        "automatic_scientific_reuse_authorized": False,
        "scientific_semantics_changed": False,
    }
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Validate current Registry + both FROZEN Passports before admitting any production node.
    reg, _ = registry.load_current_registry(ROOT)
    q17_entry = registry.resolve_experiment(reg, Q17_ID)
    bounded_entry = registry.resolve_experiment(reg, BOUNDED_ID)
    q17 = passport.load_passport(ROOT / Q17_PASSPORT, root=ROOT)
    bounded = passport.load_passport(ROOT / BOUNDED_PASSPORT, root=ROOT)
    if q17_entry["passport_sha256"] != Q17_PASSPORT_SHA or q17["passport_payload_sha256"] != Q17_PASSPORT_SHA:
        raise RuntimeError("Q17 Passport/Registry binding mismatch")
    if bounded_entry["passport_sha256"] != BOUNDED_PASSPORT_SHA or bounded["passport_payload_sha256"] != BOUNDED_PASSPORT_SHA:
        raise RuntimeError("Bounded Passport/Registry binding mismatch")

    operations = [
        "completed-1p7b.backfill.q17-publication",
        "completed-1p7b.backfill.q17-fresh-restore",
        "completed-1p7b.backfill.bounded-publication",
        "completed-1p7b.backfill.bounded-fresh-restore",
        "completed-1p7b.backfill.run-passport-control-plane",
        "completed-1p7b.backfill.asset-ledger",
    ]
    config = {
        "schema": f"openevo.completed-1p7b-artifact-dag-backfill-config.{STAMP}",
        "source_workline_pr": 555,
        "source_workline_head": "9dc6674435e6f392dfeb86a41db4c672baeded7a",
        "current_dag_schema": dag.NODE_SCHEMA_ID,
        "operation_schema_version": STAMP,
        "operations": operations,
        "hard_boundaries": {
            "scientific_replay_authorized": False, "model_update_reuse_authorized": False,
            "final_evaluation_reuse_authorized": False, "provider_mutation_authorized": False,
            "server_delete_authorized": False,
        },
    }
    _write_json(CONFIG_REL, config)

    q17_pub_src = [
        f"{EVID}/Q17_PUBLIC_PRIVATE_RECOVERY_LEDGER.json",
        f"{EVID}/Q17_REMOTE_OBJECT_VERIFY.json",
    ]
    q17_pub_rel = f"{CUR}/Q17_PUBLICATION_DAG_BINDING.json"
    _write_json(q17_pub_rel, _binding(
        f"openevo.completed-1p7b.q17-publication-dag-binding.{STAMP}", Q17_ID, Q17_PASSPORT_SHA, q17_pub_src,
        [
            {"repo": "example-account/openevo-q17-directapply-1p7b-checkpoints", "revision": "b809626944e17e23f1e4c0d12cda0a474c36e08b", "role": "public-model-state"},
            {"repo": "example-account/q17-direct-apply-only-w2-202609082022-raw-evidence", "revision": "348856b31ccbe4c1cd67a02501bd098a1f73b0a8", "role": "private-raw-recovery"},
        ], "Q17 public model-state and private raw recovery surfaces are immutable and remotely exact."))

    q17_restore_src = [f"{EVID}/Q17_FRESH_RESTORE_VERIFY.json", q17_pub_rel]
    q17_restore_rel = f"{CUR}/Q17_RESTORE_DAG_BINDING.json"
    _write_json(q17_restore_rel, _binding(
        f"openevo.completed-1p7b.q17-restore-dag-binding.{STAMP}", Q17_ID, Q17_PASSPORT_SHA, q17_restore_src,
        [{"repo": "example-account/q17-direct-apply-only-w2-202609082022-raw-evidence", "revision": "348856b31ccbe4c1cd67a02501bd098a1f73b0a8", "role": "fresh-restore-source"}],
        "Q17 R0/R64/R159 fresh restore verification is preserved as deterministic recovery evidence."))

    bounded_pub_src = [
        f"{EVID}/OFF_HF_PUBLICATION_RECEIPT.json", f"{EVID}/ON_HF_PUBLICATION_RECEIPT.json",
        f"{EVID}/BOUNDED_HF_SPLIT_PUBLICATION_VERIFY.json", f"{EVID}/BOUNDED_PRIVATE_HF_FULL_EXACT_VERIFY.json",
        f"{EVID}/OWNER_HF_SPLIT_PLACEMENT_DECISION_20260919.json",
    ]
    bounded_pub_rel = f"{CUR}/BOUNDED_PUBLICATION_DAG_BINDING.json"
    _write_json(bounded_pub_rel, _binding(
        f"openevo.completed-1p7b.bounded-publication-dag-binding.{STAMP}", BOUNDED_ID, BOUNDED_PASSPORT_SHA, bounded_pub_src,
        [
            {"repo": "example-account/openevo-bounded-effective-state-gdr-1p7b-states", "revision": "75d66482136c14dd83723babb0f77289d4bdcb69", "role": "public-model-state"},
            {"repo": "example-account/openevo-bounded-effective-state-gdr-1p7b-raw-evidence", "revision": "9be9e97700ac125d891e53798832c77836b51758", "role": "private-raw-recovery"},
        ], "Bounded OFF/ON public state and private recovery surfaces are immutable and remotely exact."))

    bounded_restore_src = [
        f"{EVID}/BOUNDED_PUBLIC_HF_FRESH_RESTORE_VERIFY.json",
        f"{EVID}/BOUNDED_PRIVATE_HF_FRESH_RESTORE_VERIFY.json",
        f"{EVID}/OFF_REMOTE_RESTORE_MATRIX.json", f"{EVID}/ON_REMOTE_RESTORE_MATRIX.json", bounded_pub_rel,
    ]
    bounded_restore_rel = f"{CUR}/BOUNDED_RESTORE_DAG_BINDING.json"
    _write_json(bounded_restore_rel, _binding(
        f"openevo.completed-1p7b.bounded-restore-dag-binding.{STAMP}", BOUNDED_ID, BOUNDED_PASSPORT_SHA, bounded_restore_src,
        [
            {"repo": "example-account/openevo-bounded-effective-state-gdr-1p7b-states", "revision": "75d66482136c14dd83723babb0f77289d4bdcb69", "role": "public-restore-source"},
            {"repo": "example-account/openevo-bounded-effective-state-gdr-1p7b-raw-evidence", "revision": "9be9e97700ac125d891e53798832c77836b51758", "role": "private-restore-source"},
        ], "Bounded OFF/ON fresh restore samples and full round recovery matrices are deterministic recovery evidence."))

    runpass_src = [f"{CUR}/RUN_PASSPORT_135_REMOTE_VERIFY.json", f"{CUR}/BOUNDED_CURRENT_MAIN_RECONCILIATION.json",
                   "configs/experiment/experiment-registry-202609210057.json"]
    runpass_rel = f"{CUR}/RUN_PASSPORT_135_DAG_BINDING.json"
    _write_json(runpass_rel, _binding(
        f"openevo.completed-1p7b.run-passport-control-dag-binding.{STAMP}", None, None, runpass_src,
        [dict(json.loads((ROOT / runpass_src[0]).read_text())["provider"], role="private-run-passport-control-plane")],
        "All 135 current top-level run roots have a remotely verified Run Passport; unresolved canonical_run_id values remain explicitly unresolved."))
    ledger_src = [
        f"{EVID}/COMPLETED_1P7B_ASSET_LEDGER.json", q17_pub_rel, q17_restore_rel,
        bounded_pub_rel, bounded_restore_rel, runpass_rel,
    ]
    ledger_rel = f"{CUR}/COMPLETED_1P7B_PRODUCTION_DAG_LEDGER.json"
    _write_json(ledger_rel, _binding(
        f"openevo.completed-1p7b.production-dag-ledger.{STAMP}", None, None, ledger_src,
        [
            {"repo": "example-account/openevo-q17-directapply-1p7b-checkpoints", "revision": "b809626944e17e23f1e4c0d12cda0a474c36e08b", "role": "q17-public"},
            {"repo": "example-account/q17-direct-apply-only-w2-202609082022-raw-evidence", "revision": "348856b31ccbe4c1cd67a02501bd098a1f73b0a8", "role": "q17-private"},
            {"repo": "example-account/openevo-bounded-effective-state-gdr-1p7b-states", "revision": "75d66482136c14dd83723babb0f77289d4bdcb69", "role": "bounded-public"},
            {"repo": "example-account/openevo-bounded-effective-state-gdr-1p7b-raw-evidence", "revision": "9be9e97700ac125d891e53798832c77836b51758", "role": "bounded-private"},
            dict(json.loads((ROOT / runpass_src[0]).read_text())["provider"], role="run-passport-control"),
        ], "Completed Q17 and Bounded lineages now have current Passport/Registry/Run-Passport/provider DAG bindings."))

    # Build current production nodes. HF views are bound to exact Git evidence but remain external/unverified views.
    specs = [
        _node(operation_type=operations[0], side_effect="PURE_ENGINEERING",
              input_paths=[(p, f"q17-source-{i}") for i, p in enumerate(q17_pub_src)], output_rel=q17_pub_rel,
              experiment_id=Q17_ID, passport_sha=Q17_PASSPORT_SHA, passport_rel=Q17_PASSPORT,
              external_views=[_hf_ref("example-account/openevo-q17-directapply-1p7b-checkpoints", "b809626944e17e23f1e4c0d12cda0a474c36e08b", q17_pub_src[1]),
                              _hf_ref("example-account/q17-direct-apply-only-w2-202609082022-raw-evidence", "348856b31ccbe4c1cd67a02501bd098a1f73b0a8", q17_pub_src[0])],
              created_at="2026-09-21T00:57:10+08:00"),
        _node(operation_type=operations[1], side_effect="DETERMINISTIC_COMPUTATION",
              input_paths=[(p, f"q17-restore-source-{i}") for i, p in enumerate(q17_restore_src)], output_rel=q17_restore_rel,
              experiment_id=Q17_ID, passport_sha=Q17_PASSPORT_SHA, passport_rel=Q17_PASSPORT,
              external_views=[_hf_ref("example-account/q17-direct-apply-only-w2-202609082022-raw-evidence", "348856b31ccbe4c1cd67a02501bd098a1f73b0a8", q17_restore_src[0])],
              created_at="2026-09-21T00:57:20+08:00"),
        _node(operation_type=operations[2], side_effect="PURE_ENGINEERING",
              input_paths=[(p, f"bounded-source-{i}") for i, p in enumerate(bounded_pub_src)], output_rel=bounded_pub_rel,
              experiment_id=BOUNDED_ID, passport_sha=BOUNDED_PASSPORT_SHA, passport_rel=BOUNDED_PASSPORT,
              external_views=[_hf_ref("example-account/openevo-bounded-effective-state-gdr-1p7b-states", "75d66482136c14dd83723babb0f77289d4bdcb69", bounded_pub_src[2]),
                              _hf_ref("example-account/openevo-bounded-effective-state-gdr-1p7b-raw-evidence", "9be9e97700ac125d891e53798832c77836b51758", bounded_pub_src[3])],
              created_at="2026-09-21T00:57:30+08:00"),
        _node(operation_type=operations[3], side_effect="DETERMINISTIC_COMPUTATION",
              input_paths=[(p, f"bounded-restore-source-{i}") for i, p in enumerate(bounded_restore_src)], output_rel=bounded_restore_rel,
              experiment_id=BOUNDED_ID, passport_sha=BOUNDED_PASSPORT_SHA, passport_rel=BOUNDED_PASSPORT,
              external_views=[_hf_ref("example-account/openevo-bounded-effective-state-gdr-1p7b-states", "75d66482136c14dd83723babb0f77289d4bdcb69", bounded_restore_src[0]),
                              _hf_ref("example-account/openevo-bounded-effective-state-gdr-1p7b-raw-evidence", "9be9e97700ac125d891e53798832c77836b51758", bounded_restore_src[1])],
              created_at="2026-09-21T00:57:40+08:00"),
        _node(operation_type=operations[4], side_effect="PURE_ENGINEERING",
              input_paths=[(p, f"run-passport-source-{i}") for i, p in enumerate(runpass_src)], output_rel=runpass_rel,
              experiment_id=None, passport_sha=None, passport_rel=None,
              external_views=[],
              created_at="2026-09-21T00:57:50+08:00"),
        _node(operation_type=operations[5], side_effect="DETERMINISTIC_COMPUTATION",
              input_paths=[(p, f"ledger-source-{i}") for i, p in enumerate(ledger_src)], output_rel=ledger_rel,
              experiment_id=None, passport_sha=None, passport_rel=None, external_views=[],
              created_at="2026-09-21T00:58:00+08:00"),
    ]

    prior, prior_router = dag.load_current_index(ROOT)
    keep = [entry for entry in prior["entries"] if not str(entry.get("operation_type", "")).startswith("completed-1p7b.backfill.")]
    entries = list(keep)
    for node in specs:
        entries.append(dag.write_operation_node(ROOT, node))
    index = dag.build_index(entries, generated_at="2026-09-21T00:58:10+08:00")
    router = dag.write_index_snapshot(ROOT, index, snapshot_rel=INDEX_REL)

    lookups = []
    for node in specs:
        result = dag.query_exact(node, repo_root=ROOT)
        lookups.append({"operation_type": node["operation_type"], "operation_digest": node["operation_digest"],
                        "status": result["status"], "classification": result.get("classification"),
                        "automatic_reuse_permitted": result.get("reuse", {}).get("automatic_reuse_permitted")})
        if result["status"] != "REUSE_CANDIDATE":
            raise RuntimeError(f"production DAG lookup failed: {node['operation_type']} -> {result['status']}")

    receipt = {
        "schema": f"openevo.completed-1p7b-production-dag-backfill-receipt.{STAMP}",
        "status": "PASS_CURRENT_DAG_PRODUCTION_BACKFILL",
        "source_shadow": _historical_shadow_rel(),
        "source_shadow_status": "ADVISORY_SHADOW_ONLY",
        "source_workline_pr": 555, "source_workline_head": "9dc6674435e6f392dfeb86a41db4c672baeded7a",
        "current_registry": {"path": "configs/experiment/experiment-registry-202609210057.json",
                             "sha256": _sha(ROOT / "configs/experiment/experiment-registry-202609210057.json"),
                             "experiments": [Q17_ID, BOUNDED_ID]},
        "dag_router": router, "prior_entry_count": len(keep), "current_entry_count": len(index["entries"]),
        "production_node_count": len(specs), "lookups": lookups,
        "historical_shadow_promoted_directly": False,
        "scientific_reuse_authorized": False, "provider_mutation_performed": False, "server_delete_performed": False,
    }
    _write_json(f"{CUR}/PRODUCTION_DAG_BACKFILL_RECEIPT.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
