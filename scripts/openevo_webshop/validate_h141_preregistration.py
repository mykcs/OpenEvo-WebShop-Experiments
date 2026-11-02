#!/usr/bin/env python3
"""Validate the H1.41 scientific contract and its successor router."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.openevo_webshop.campaign_router import is_external_successor_router

CAMPAIGN_ID = "20260821-0141-h141-magnitude-screen"
PREDECESSOR_ID = "20260820-0140r2-h140-g2-multigeneration"
CONFIG = Path("configs/experiment/h1.41-magnitude-screen-v2.json")
PATH_CONTRACT = Path("configs/experiment/h1.41-path-contract-v1.json")
RELEASE = Path("configs/experiment/activations/h1.41-magnitude-screen-execution-release-v2.json")
REFERENCE = Path("docs/evidence/remote-runs/2026-08-21/h1.41/h1.41-g1-deterministic-reference-v1.json")
AUTHORIZED_GPU_UUIDS = (
    "GPU-00000000-0000-0000-0000-000000000000",
    "GPU-00000000-0000-0000-0000-000000000000",
    "GPU-00000000-0000-0000-0000-000000000000",
    "GPU-00000000-0000-0000-0000-000000000000",
    "GPU-00000000-0000-0000-0000-000000000000",
    "GPU-00000000-0000-0000-0000-000000000000",
)


def read(root: Path, relative: Path) -> dict[str, Any]:
    value = json.loads((root / relative).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {relative}")
    return value


def sha256(root: Path, relative: Path) -> str:
    return hashlib.sha256((root / relative).read_bytes()).hexdigest()


def validate_deterministic_reference(root: Path, config: dict[str, Any], reference: dict[str, Any]) -> None:
    if sha256(root, REFERENCE) != config.get("source_artifacts", {}).get("g1_deterministic_reference_sha256"):
        raise ValueError("H1.41 deterministic G1 reference hash mismatch")
    source = reference.get("source") or {}
    seeds = source.get("historical_rollout_seeds")
    if source.get("temperature") != 0.0 or seeds != [2026104100, 2026104101, 2026104102, 2026104103]:
        raise ValueError("H1.41 deterministic G1 reference decoding contract mismatch")
    panels = config.get("frozen_scientific_contract", {}).get("evaluation_panels", {})
    reference_panels = reference.get("panels") or {}
    stage_map = {"acquisition_e2": "acquisition", "retention_e1": "retention", "preservation_h139": "preservation"}
    for panel_name, reference_name in stage_map.items():
        expected_ids = panels.get(panel_name, {}).get("task_ids", [])
        rows = {row.get("task_id"): row for row in reference_panels.get(reference_name, [])}
        if set(rows) != set(expected_ids):
            raise ValueError(f"H1.41 deterministic G1 task set mismatch: {panel_name}")
        for task_id in expected_ids:
            row = rows[task_id]
            scores = row.get("scores_by_seed") or {}
            if row.get("valid_attempt_count") != 4 or set(scores) != {str(seed) for seed in seeds} or len(set(scores.values())) != 1:
                raise ValueError(f"H1.41 G1 reference is not four-attempt seed-invariant: {task_id}")


def validate(root: Path, *, require_router: bool = False) -> dict[str, Any]:
    root = root.resolve()
    config = read(root, CONFIG)
    path_contract = read(root, PATH_CONTRACT)
    release = read(root, RELEASE)
    reference = read(root, REFERENCE)
    current = read(root, Path("configs/experiment/current-campaign.json"))
    if config.get("campaign_id") != CAMPAIGN_ID or config.get("protocol_revision") != "v2-deterministic-g1-reference" or config.get("status") != "preregistered-pre-execution" or config.get("formal_task_consumption_allowed") is not False:
        raise ValueError("H1.41 config is not closed at preregistration")
    if config.get("parent_closeout", {}).get("classification") != "complete-upstream-gate-failed" or config.get("parent_closeout", {}).get("t2_panel_reuse_forbidden") is not True:
        raise ValueError("H1.40 closeout provenance/T2 boundary is missing")
    source = config.get("frozen_scientific_contract", {}).get("source_selection", {})
    if source.get("total_records") != 16 or source.get("records_per_task") != 2 or len(source.get("task_ids", [])) != 8:
        raise ValueError("H1.41 source selection is not the sealed 8x2 E2 set")
    arms = config.get("frozen_scientific_contract", {}).get("arms", {})
    if arms.get("C0_exact_h140", {}).get("reset_all_to_coefficient_init") is not False or arms.get("C1_reset_magnitude", {}).get("reset_all_to_coefficient_init") is not True:
        raise ValueError("H1.41 arms do not isolate magnitude initialization")
    expected_attempts = sum(len(panel.get("task_ids", [])) * len(panel.get("rollout_seeds", [])) for panel in (config.get("frozen_scientific_contract", {}).get("evaluation_panels") or {}).values()) * len(arms)
    expected_resource_contract = {"formal_task_attempts": expected_attempts, "new_harvest_attempts": 0, "t2_attempts": 0, "gpu_hours_cap": 6.0, "accounting": "exact per attempt; unknown intervals are not reported as zero", "authorized_pool_basis": "current parent standing pool intersected with immediate live-idle UUIDs", "kill_reset_preempt_allowed": False}
    if config.get("resource_contract") != expected_resource_contract:
        raise ValueError("H1.41 resource contract changed")
    validate_deterministic_reference(root, config, reference)
    if config.get("formal_execution_guard", {}).get("tasks_consumed") != 0 or config.get("formal_execution_guard", {}).get("t2_panel_evaluated") is not False:
        raise ValueError("H1.41 formal guard is not zero/T2-closed")
    if path_contract.get("campaign_id") != CAMPAIGN_ID or path_contract.get("requirements", {}).get("clean_core_must_remain_immutable") is not True:
        raise ValueError("H1.41 path contract is not bound to the campaign")
    if require_router or current.get("campaign_id") == CAMPAIGN_ID:
        if current.get("campaign_id") != CAMPAIGN_ID or current.get("preregistration") != str(CONFIG) or current.get("replacement_of") != PREDECESSOR_ID:
            raise ValueError("current router is not H1.41 successor-bound")
        if current.get("status") not in {"preregistered-pre-execution", "launch-authorized", "blocked-pre-execution", "closed-upstream-gates-failed", "closed-measurement-invalid", "closed-result"}:
            raise ValueError("H1.41 router has an unsupported status")
        expected_open = current.get("status") == "launch-authorized"
        if current.get("formal_task_consumption_allowed") is not expected_open:
            raise ValueError("H1.41 router/formal gate mismatch")
        if current.get("authorized_gpu_uuids") != list(AUTHORIZED_GPU_UUIDS):
            raise ValueError("H1.41 router GPU authorization set mismatch")
    elif not is_external_successor_router(
        current,
        {CAMPAIGN_ID},
        predecessor_routes={str(CONFIG)},
    ):
        raise ValueError("current router is incomplete or unexpectedly reopens H1.41")
    if release.get("campaign_id") != CAMPAIGN_ID or release.get("status") != "launch-authorized" or release.get("formal_task_consumption_allowed") is not True:
        raise ValueError("H1.41 execution release is not launch-authorized")
    if release.get("scientific_contract_changed") is not True or release.get("amends", {}).get("replacement") != str(CONFIG):
        raise ValueError("H1.41 v2 release does not record its preregistration amendment")
    if release.get("preregistration") != {"path": str(CONFIG), "sha256": sha256(root, CONFIG)}:
        raise ValueError("H1.41 release preregistration hash mismatch")
    if release.get("path_contract") != {"path": str(PATH_CONTRACT), "sha256": sha256(root, PATH_CONTRACT)}:
        raise ValueError("H1.41 release path-contract hash mismatch")
    patch = Path("scripts/openevo_webshop/patches/openevo-sd-lora-reset-magnitudes.patch")
    if release.get("upstream_patch") != {"path": str(patch), "sha256": sha256(root, patch)}:
        raise ValueError("H1.41 release patch hash mismatch")
    if release.get("parent_policy", {}).get("authorized_gpu_uuids") != list(AUTHORIZED_GPU_UUIDS) or release.get("parent_policy", {}).get("live_idle_required_immediately_before_launch") is not True:
        raise ValueError("H1.41 release parent GPU policy mismatch")
    if release.get("zero_state_at_release") != {"formal_task_consumption": 0, "formal_gpu_hours_consumed": 0.0, "g2_adapter_created": False, "t2_panel_evaluated": False, "qualification_status": "not-started"}:
        raise ValueError("H1.41 release did not preserve zero state")
    return {"status": "pass", "campaign_id": CAMPAIGN_ID, "router_campaign": current.get("campaign_id"), "formal_task_consumption_allowed": current.get("formal_task_consumption_allowed")}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--require-router", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(args.root, require_router=args.require_router), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
