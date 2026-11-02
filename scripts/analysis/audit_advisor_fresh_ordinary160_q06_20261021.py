#!/usr/bin/env python3
"""CPU-only provenance audit: Q06 Ordinary160 vs frozen advisor-fresh Ordinary160.

This audits *committed records*, not current server bytes, model load, GPU state,
Formal Start, WebShop completion or scientific results. It changes no inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ADAPTER_SHA = "18032c5cabd135b2464d5cbb2a3949aeb6130046dcc9f33acb33aee38c1baf46"
EXPECTED_CONFIG_SHA = "20740b6f9b558428b48225c2896fa8488a3e16e350fce54380fbbfb67d4d5e54"
EXPECTED_HF_REVISION = "b809626944e17e23f1e4c0d12cda0a474c36e08b"

Q06 = "docs/evidence/q06-remaining-science-202610032022/ordinary-r160-manifest.json"
QUALIFICATION = "docs/evidence/q06-remaining-science-202610032022/ordinary-r160-real-cache-qualification.json"
FRESH = "configs/experiment/manifests/advisor-fresh-checkpoints-20261020.json"
PREREG = "configs/experiment/preregistrations/advisor-fresh-checkpoints-20261020.json"
REMAINING = "docs/evidence/advisor-fresh-canonical-20261020/remaining-two-cells-202610201758.json"
PROFILE = "configs/experiment/profiles/advisor-fresh-202610201758-ordinary-gpu5.json"


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError("Ordinary160 identity audit FAILED: " + reason)


def _read(root: Path, relative: str) -> dict:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def audit(root: Path = ROOT) -> dict:
    q06, qualification = _read(root, Q06), _read(root, QUALIFICATION)
    fresh, prereg = _read(root, FRESH), _read(root, PREREG)
    remaining, profile = _read(root, REMAINING), _read(root, PROFILE)
    frozen_manifest_sha = hashlib.sha256((root / FRESH).read_bytes()).hexdigest()
    fresh_checkpoint = fresh["checkpoints"]["ordinary160"]
    q06_checkpoint = q06["checkpoints"]["after_160_updates"]
    q06_source = q06["endpoint"]["source_files"]
    recovery = fresh_checkpoint["cold_recovery"]

    require(q06["checkpoint_id"] == "after_160_updates", "Q06 checkpoint selector drift")
    require(q06["actual_parameter_update_events"] == 160 and q06["completed_stage2_rounds"] == 160,
            "Q06 160-update training lineage drift")
    require(q06_checkpoint["adapter_sha256"] == EXPECTED_ADAPTER_SHA
            == q06_source["adapter"]["sha256"]
            == fresh_checkpoint["files"]["adapter_model.safetensors"],
            "Q06 source weight SHA != frozen fresh weight SHA")
    require(q06_source["adapter_config"]["sha256"] == EXPECTED_CONFIG_SHA
            == fresh_checkpoint["files"]["adapter_config.json"],
            "Q06 adapter configuration SHA != frozen fresh configuration SHA")
    require(q06_source["adapter"]["bytes"] == recovery["bytes"] == 2042395200,
            "Ordinary160 adapter byte length differs")
    require(q06["endpoint"]["adapter_path"].endswith("/round-159/sd-lora/artifacts/workers/"
            "sd-lora-a5e38d152da991e4310a2f47accdbf94/adapter"),
            "Q06 historical Q17 round-159 source lineage changed")
    require(q06["model"]["revision"] == fresh["model"]["revision"]
            == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
            "Base Qwen3 revision mismatch")
    require(q06["execution_image"]["id"] == fresh["image_id"], "Docker image identity differs")
    require(q06["panel_sha256"] != fresh["panel_sha256"], "Q06 development and new panels conflated")

    require(prereg["manifest_sha256"] == frozen_manifest_sha
            == remaining["scientific_manifest_sha256"]
            == prereg["frozen_scientific_contract"]["manifest"]["sha256"],
            "Frozen prospective manifest not byte-bound to preregistration and successor")
    require(prereg["checkpoint_count"] == 7 and prereg["episodes"] == 448
            and prereg["episodes_per_checkpoint"] == len(fresh["tasks"]) == 64
            and len(prereg["contrasts"]) == 4,
            "448-task four-contrast preregistration has changed")
    require(prereg["final_panel_access"] == prereg["optimizer_updates"] == prereg["teacher_calls"] == 0,
            "Frozen zero-access/zero-training boundary changed")
    require(fresh_checkpoint["availability_at_freeze"] == "PENDING_EXACT_PUBLIC_RESTORATION",
            "Do not rewrite frozen availability using later evidence")
    require(recovery["repository"] == "openevo-recovery/openevo-q17-directapply-1p7b-checkpoints"
            and recovery["revision"] == EXPECTED_HF_REVISION
            and recovery["filename"] == "checkpoints/round-159/adapter_model.safetensors",
            "Pinned historical backup location or commit drift")
    restore_receipts = [x["receipt"] for x in remaining["proof_receipts"]
                        if x.get("receipt", {}).get("status") == "EXACT_PUBLIC_RESTORATION_PASS"]
    require(len(restore_receipts) == 1
            and restore_receipts[0]["sha256"] == EXPECTED_ADAPTER_SHA
            and restore_receipts[0]["revision"] == EXPECTED_HF_REVISION
            and restore_receipts[0]["bytes"] == recovery["bytes"],
            "Post-freeze exact public HTTP restoration receipt is absent or inconsistent")
    require(profile["run_id"] == "advisor-fresh-202610201758-ordinary-gpu5"
            and profile["command"][-2:] == ["--checkpoint-id", "ordinary160"],
            "Frozen successor profile no longer selects only Ordinary160")
    require(str(fresh_checkpoint["adapter_path"]) in [x["path"] for x in profile["required_paths"]],
            "Restored Ordinary160 path absent from canonical prelaunch gate")

    require(qualification["status"] == "PASS"
            and qualification["model_forward_calls"] == 16
            and qualification["environment_calls"] == 0
            and qualification["formal_task_consumed"] is False
            and qualification["scientific_complete"] is False
            and qualification["production_script_sha256"] == q06["producer_sha256"],
            "Q06 synthetic scorer qualification no longer has the recorded non-scientific scope")
    require(remaining["prior_completed_tasks"] == 320
            and len(remaining["preserved_completed_states"]) == 5
            and set(remaining["new_checkpoint_ids"]) == {"ordinary160", "reverse156"}
            and remaining["new_tasks_total"] == 128
            and remaining["automatic_replay"] is False,
            "Historical 320/128 successor snapshot changed; inspect before relying on it")

    return {
        "status": "PASS_GITHUB_RECORD_IDENTITY_ONLY",
        "q06_and_fresh_same_adapter_sha256": EXPECTED_ADAPTER_SHA,
        "same_base_revision_and_adapter_config": True,
        "backup_repo": recovery["repository"],
        "backup_revision": recovery["revision"],
        "post_freeze_restoration_receipt": "EXACT_PUBLIC_RESTORATION_PASS",
        "q06_synthetic_calls_not_scientific": True,
        "recorded_completed_snapshot": 320,
        "recorded_remaining_snapshot": 128,
        "live_restored_model_load_verified": False,
        "live_formal_start_and_all_tasks_verified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(audit(args.repo_root), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
