#!/usr/bin/env python3
"""Validate the clean H1.40-R2 replacement without opening formal work."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.openevo_webshop.campaign_router import is_external_successor_router

CAMPAIGN_ID = "20260820-0140r2-h140-g2-multigeneration"
SUCCESSOR_CAMPAIGN_ID = "20260821-0141-h141-magnitude-screen"
ORIGINAL_CAMPAIGN_ID = "20260820-0140-h140-g2-multigeneration"
CONFIG = Path("configs/experiment/h1.40-r2-g2-multigeneration-v1.json")
ORIGINAL_CONFIG = Path("configs/experiment/h1.40-g2-multigeneration-v1.json")
ORIGINAL_CONFIG_SHA256 = "5ba0e3e2d6fc1738bd0214eb1b05d3acb7d4ff853ebad19c32442d4e6c4f526f"
MANIFEST_DIR = Path("configs/experiment/manifests")
G2_RECONCILIATION = Path("docs/evidence/remote-runs/2026-08-20/h1.40-g2-r2/g2-training-reconciliation-v1.json")
G2_REPORT = Path("docs/evidence/remote-runs/2026-08-20/h1.40-g2-r2/G2_TRAINING_REPORT.md")
G2_RECONCILIATION_SHA256 = "6eecba8c710d6077b7134ff911b74b5d96fa01df7d41b0894775913c86fef878"
MANIFESTS = {
    "exclusion_ledger": "h1.40-g2-exclusion-ledger-v1.json",
    "e1_retention": "h1.40-g2-e1-retention-set-v1.json",
    "e2_reservoir": "h1.40-g2-e2-reservoir-v1.json",
    "h1_39_preservation": "h1.40-g2-h139-preservation-anchor-v1.json",
    "t2": "h1.40-g2-t2-panel-v1.json",
}
EXPECTED_FILE_HASHES = {
    "exclusion_ledger": "92cc0fe33f2d3a8db66759ac87ef6a4ceda5bd9c14bf99c1a2bf302fe0c10a04",
    "e1_retention": "62146a8913bc8d7ba0a0531853ca1b01550975beb875143f805575ad577ca635",
    "e2_reservoir": "589cbc18c30638b504cdcb1fb3d3aaf17ffb332479c8d64bb30787a38dfe12f6",
    "h1_39_preservation": "607ecbdc3243e078df1e2319a364b59997ac22a0a790ec79a734f199eec9027c",
    "t2": "b7499d094327fa294d4e561473e6853738e5ff00e5e6dd0fe183adacf6f63b9f",
}
EXPECTED_CONTENT_HASHES = {
    "exclusion_ledger": "3f9a17a6b140da7b300c5f6205aba832676adc9521c6cfda814cfe40088b72e8",
    "e1_retention": "0dac2d5aba03f5f738749ca0d8b1dc627cb417f6b91cd213eafebbebf7265ad9",
    "e2_reservoir": "c50d8fb05acdee87750a14de8f1134658a802d483bd7ee3370174bb741f38230",
    "h1_39_preservation": "65f41a037133982b3d1421768911384e3e0d82df5a0031ce8e4a632dc4093172",
    "t2": "28f9bc9830f9e26fffe518372c3e6f79df76a2c020748fe8263c7cc36aae194e",
}
EXPECTED_TASK_DIGESTS = {
    "e1_retention": "1d433152c2c1dce96a274a76f4ca442a8977311bd7bafaf7b072ae7ecc6dbc90",
    "e2_reservoir": "e4a88d99828e0c341e37a7da149b88307949076ba6c13cec69996ee2cf463bac",
    "h1_39_preservation": "a01b1d98861561910ff0fd5f55710963debcb80096939c663d6ba6aa6e113135",
    "t2": "2a3719d03381b636ae97dfb0c1f7eb0fceaedf7c9f34e3cf16545f2d5afb338b",
}


def read(root: Path, path: Path) -> dict[str, Any]:
    value = json.loads((root / path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha(root: Path, path: Path) -> str:
    return hashlib.sha256((root / path).read_bytes()).hexdigest()


def task_rows(manifest: dict[str, Any], expected_count: int, label: str) -> set[int]:
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != expected_count:
        raise ValueError(f"{label} task count mismatch")
    indices: set[int] = set()
    for position, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"{label} task row is not an object")
        idx = task.get("goal_idx")
        task_id = task.get("task_id")
        text = task.get("task_description")
        if not isinstance(idx, int) or idx in indices or task_id != f"webshop_{idx:06d}" or task.get("sample_index") != position:
            raise ValueError(f"{label} task identity/order mismatch")
        if not isinstance(text, str) or hashlib.sha256(text.encode()).hexdigest() != task.get("task_description_sha256"):
            raise ValueError(f"{label} task description hash mismatch")
        indices.add(idx)
    return indices


def validate(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = read(root, CONFIG)
    original = read(root, ORIGINAL_CONFIG)
    current = read(root, Path("configs/experiment/current-campaign.json"))
    if file_sha(root, ORIGINAL_CONFIG) != ORIGINAL_CONFIG_SHA256:
        raise ValueError("original frozen H1.40 config changed")
    if config.get("campaign_id") != CAMPAIGN_ID or config.get("status") != "preregistered-pre-execution" or config.get("formal_task_consumption_allowed") is not False:
        raise ValueError("R2 config is not closed at the preregistration boundary")
    provenance = config.get("provenance") or {}
    if provenance.get("parent_original_campaign_id") != ORIGINAL_CAMPAIGN_ID or provenance.get("parent_original_config_sha256") != ORIGINAL_CONFIG_SHA256:
        raise ValueError("R2 replacement provenance does not bind the original campaign/config")
    if provenance.get("attempt1_engineering_invalid_gpu_hours") != "unknown" or provenance.get("pooled_denominators_allowed") is not False:
        raise ValueError("Attempt 1 unknown was imported into R2 or pooling was enabled")
    if config.get("generations", {}).get("g2", {}).get("adapter_file_sha256") is not None:
        raise ValueError("R2 G2 adapter identity was pre-created")
    if config.get("formal_execution_guard") != {"tasks_consumed": 0, "gpu_hours_consumed": 0.0, "g2_adapter_created": False, "t2_panel_evaluated": False, "launcher_must_fail_closed_while_status_is_preregistered": True}:
        raise ValueError("R2 formal execution guard is not zero/closed")

    frozen = config.get("frozen_scientific_contract") or {}
    if config.get("scientific_question") != original.get("scientific_question"):
        raise ValueError("scientific question changed")
    for r2_key, original_key in (("g0", "g0"), ("g1", "g1"), ("g2", "g2"), ("method_identity", "method_identity"), ("success_qualification", "success_qualification"), ("training", "training"), ("matrices", "matrices"), ("analysis", "analysis"), ("retention", "retention"), ("classification", "classification")):
        source = original.get("generations", {}).get(original_key) if r2_key in {"g0", "g1", "g2"} else original.get(r2_key)
        if frozen.get(r2_key) != source:
            raise ValueError(f"frozen scientific contract changed: {r2_key}")

    selection = frozen.get("selection") or {}
    original_selection = original.get("selection") or {}
    expected_selection = {
        "harvest_task_count": original_selection.get("harvest_task_count"),
        "harvest_max_attempts": original_selection.get("harvest_max_attempts"),
        "selection_rule": original_selection.get("selection_rule"),
        "harvest_stop_rule": original_selection.get("harvest_stop_rule"),
        "e1_retention_task_count": original_selection.get("e1_retention_task_count"),
        "preservation_task_count": original_selection.get("preservation_task_count"),
        "t2_task_count": original_selection.get("t2_task_count"),
    }
    if any(selection.get(key) != value for key, value in expected_selection.items()):
        raise ValueError("selection counts/rules changed")
    if selection.get("seeds") != {"e2_selection": original_selection.get("e2_selection_seed"), "preservation_selection": original_selection.get("preservation_selection_seed"), "t2_selection": original_selection.get("t2_selection_seed"), "harvest": original_selection.get("harvest_seeds"), "evaluation": original_selection.get("evaluation_seeds"), "t2": original_selection.get("t2_seeds")}:
        raise ValueError("selection seeds changed")
    if selection.get("materialized_manifest_file_sha256") != EXPECTED_FILE_HASHES or selection.get("manifest_content_sha256") != EXPECTED_CONTENT_HASHES or selection.get("task_identity_digests") != EXPECTED_TASK_DIGESTS:
        raise ValueError("manifest or task identity hashes changed")

    loaded = {name: read(root, MANIFEST_DIR / path) for name, path in MANIFESTS.items()}
    if any(file_sha(root, MANIFEST_DIR / path) != EXPECTED_FILE_HASHES[name] for name, path in MANIFESTS.items()):
        raise ValueError("materialized manifest file hash mismatch")
    exclusion_payload = dict(loaded["exclusion_ledger"])
    exclusion_hash = exclusion_payload.pop("ledger_sha256", None)
    if exclusion_hash != digest(exclusion_payload) or exclusion_hash != EXPECTED_CONTENT_HASHES["exclusion_ledger"]:
        raise ValueError("exclusion ledger content hash mismatch")
    for name in ("e1_retention", "e2_reservoir", "h1_39_preservation", "t2"):
        manifest = loaded[name]
        field = "manifest_sha256"
        payload = dict(manifest)
        expected_self = payload.pop(field, None)
        if expected_self != digest(payload):
            raise ValueError(f"{name} self hash mismatch")
    partitions = {
        "e1": task_rows(loaded["e1_retention"], 8, "E1"),
        "e2": task_rows(loaded["e2_reservoir"], 64, "E2"),
        "anchor": task_rows(loaded["h1_39_preservation"], 32, "H1.39 preservation"),
        "t2": task_rows(loaded["t2"], 64, "T2"),
    }
    if len(set().union(*partitions.values())) != sum(len(v) for v in partitions.values()):
        raise ValueError("R2 task partitions overlap")
    successor_router = current.get("campaign_id") == SUCCESSOR_CAMPAIGN_ID and current.get("replacement_of") == CAMPAIGN_ID
    external_successor_router = is_external_successor_router(
        current,
        {CAMPAIGN_ID, SUCCESSOR_CAMPAIGN_ID},
        predecessor_routes={str(CONFIG)},
    )
    if external_successor_router:
        return {"status": "pass", "campaign_id": CAMPAIGN_ID, "partition_counts": {name: len(value) for name, value in partitions.items()}, "formal_task_consumption_allowed": False, "g2_adapter_created": True, "t2_panel_evaluated": False, "original_config_sha256": ORIGINAL_CONFIG_SHA256}
    if successor_router:
        if current.get("status") not in {"preregistered-pre-execution", "launch-authorized", "blocked-pre-execution", "closed-upstream-gates-failed", "closed-measurement-invalid", "closed-result"}:
            raise ValueError("successor router does not preserve an H1.40 closeout boundary")
        if current.get("status") == "launch-authorized" and current.get("formal_task_consumption_allowed") is not True:
            raise ValueError("launch-authorized successor router unexpectedly closes the H1.40 boundary")
        if current.get("status") != "launch-authorized" and current.get("formal_task_consumption_allowed") is not False:
            raise ValueError("closed successor router unexpectedly opens the H1.40 boundary")
        return {"status": "pass", "campaign_id": CAMPAIGN_ID, "partition_counts": {name: len(value) for name, value in partitions.items()}, "formal_task_consumption_allowed": False, "g2_adapter_created": True, "t2_panel_evaluated": False, "original_config_sha256": ORIGINAL_CONFIG_SHA256}
    if current.get("campaign_id") != CAMPAIGN_ID or current.get("status") not in {"preregistered-pre-execution", "launch-authorized", "closed-upstream-gates-failed"}:
        raise ValueError("current campaign is not routed to R2")
    expected_open = current.get("status") == "launch-authorized"
    if current.get("formal_task_consumption_allowed") is not expected_open:
        raise ValueError("current R2 router open/closed state mismatch")
    if current.get("preregistration") != str(CONFIG) or current.get("replacement_of") != ORIGINAL_CAMPAIGN_ID:
        raise ValueError("current campaign R2 routing/provenance mismatch")
    latest = current.get("latest_event") or {}
    if latest.get("classification") == "r2-upstream-gates-failed":
        if latest.get("formal_task_consumption") != 640 or latest.get("formal_gpu_hours_consumed") != 3.7551676636612837 or not latest.get("g2_adapter_created") or latest.get("t2_panel_evaluated"):
            raise ValueError("R2 upstream closeout event denominator/state mismatch")
        upstream = latest.get("upstream_evaluation") or {}
        if upstream.get("status") != "complete-upstream-gate-failed" or upstream.get("formal_task_consumption") != 384 or upstream.get("t2_evaluation_allowed") is not False:
            raise ValueError("R2 upstream closeout reconciliation state mismatch")
        reconciliation = root / str(upstream.get("summary", ""))
        if not reconciliation.is_file():
            raise ValueError("R2 upstream closeout summary is missing")
    elif latest.get("classification") == "r2-g2-training-passed":
        if latest.get("formal_task_consumption") != 256 or latest.get("formal_gpu_hours_consumed") != 2.219149268141514 or not latest.get("g2_adapter_created") or latest.get("t2_panel_evaluated"):
            raise ValueError("R2 G2 training event denominator/state mismatch")
        if latest.get("engineering_invalid_gpu_hours") != 0.011175462235551093:
            raise ValueError("R2 engineering-invalid resource accounting mismatch")
        if latest.get("g2_training_reconciliation") != str(G2_RECONCILIATION) or latest.get("g2_training_reconciliation_sha256") != G2_RECONCILIATION_SHA256:
            raise ValueError("R2 G2 reconciliation pointer/hash mismatch")
        if file_sha(root, G2_RECONCILIATION) != G2_RECONCILIATION_SHA256 or not (root / G2_REPORT).is_file():
            raise ValueError("R2 G2 reconciliation/report evidence is missing or changed")
        result = read(root, G2_RECONCILIATION)
        if result.get("status") != "pass" or result.get("increment_count") != 8 or result.get("training_record_count") != 16 or result.get("resource_unknown_count") != 0 or result.get("training_gpu_hours") != 0.03383210371056015:
            raise ValueError("R2 G2 training evidence is not a passing complete result")
        if result.get("final_adapter", {}).get("adapter_model_sha256") != latest.get("g2_final_adapter_model_sha256") or result.get("final_adapter", {}).get("state_weights_sha256") != latest.get("g2_final_state_weights_sha256"):
            raise ValueError("R2 final adapter identity mismatch")
    elif latest.get("classification") == "r2-e2-harvest-selection-passed":
        if latest.get("formal_task_consumption") != 256 or latest.get("formal_gpu_hours_consumed") != 2.185317164430954 or latest.get("g2_adapter_created") or latest.get("t2_panel_evaluated"):
            raise ValueError("R2 E2 harvest event denominator/state mismatch")
        evidence = root / str(latest.get("e2_harvest_reconciliation", ""))
        if not evidence.is_file() or file_sha(root, Path(latest["e2_harvest_reconciliation"])) != latest.get("e2_harvest_reconciliation_sha256"):
            raise ValueError("R2 E2 reconciliation evidence hash mismatch")
        result = read(root, Path(latest["e2_harvest_reconciliation"]))
        if result.get("status") != "pass" or result.get("selection", {}).get("record_count") != 16 or result.get("resource_accounting", {}).get("resource_unknown_count") != 0:
            raise ValueError("R2 E2 selection evidence is not a passing complete result")
    elif latest.get("classification") not in {"r2-preregistration-published", "r2-activation-published", "r2-zero-task-qualification-passed", "r2-formal-execution-release-published"} or latest.get("formal_task_consumption") != 0 or latest.get("formal_gpu_hours_consumed") != 0.0 or latest.get("g2_adapter_created") or latest.get("t2_panel_evaluated"):
        raise ValueError("R2 latest event is not zero-state")
    if latest.get("classification") == "r2-zero-task-qualification-passed":
        qualification = latest.get("qualification") or {}
        if qualification.get("status") != "pass" or qualification.get("formal_task_consumption") != 0 or qualification.get("g2_adapter_created") or qualification.get("t2_panel_evaluated"):
            raise ValueError("R2 zero-task qualification evidence is not a pass")
        evidence_dir = root / str(qualification.get("evidence_dir", ""))
        for arm in ("base", "sd-lora"):
            value = read(evidence_dir, Path(f"{arm}.json"))
            if value.get("status") != "pass" or value.get("task_consumption") != "none" or value.get("formal_panel_consumed") is not False:
                raise ValueError(f"R2 {arm} qualification evidence is not task-free")
        for name, expected in (qualification.get("file_sha256") or {}).items():
            if file_sha(root, evidence_dir / name) != expected:
                raise ValueError(f"R2 qualification evidence hash mismatch: {name}")
    return {"status": "pass", "campaign_id": CAMPAIGN_ID, "partition_counts": {name: len(value) for name, value in partitions.items()}, "formal_task_consumption_allowed": False, "g2_adapter_created": latest.get("g2_adapter_created", False), "t2_panel_evaluated": latest.get("t2_panel_evaluated", False), "original_config_sha256": ORIGINAL_CONFIG_SHA256}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps(validate(args.root), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
