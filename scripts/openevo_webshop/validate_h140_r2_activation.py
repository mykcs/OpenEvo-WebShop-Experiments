#!/usr/bin/env python3
"""Validate the R2 activation receipt and its zero-state boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.openevo_webshop.campaign_router import is_external_successor_router

CAMPAIGN_ID = "20260820-0140r2-h140-g2-multigeneration"
SUCCESSOR_CAMPAIGN_ID = "20260821-0141-h141-magnitude-screen"
ACTIVATION = Path("configs/experiment/activations/h1.40-r2-g2-execution-activation-v1.json")
RELEASE = Path("configs/experiment/activations/h1.40-r2-g2-formal-execution-release-v1.json")
E2_RECONCILIATION = Path("docs/evidence/remote-runs/2026-08-20/h1.40-g2-r2/e2-harvest-reconciliation-v1.json")
G2_RECONCILIATION = Path("docs/evidence/remote-runs/2026-08-20/h1.40-g2-r2/g2-training-reconciliation-v1.json")
G2_RECONCILIATION_SHA256 = "6eecba8c710d6077b7134ff911b74b5d96fa01df7d41b0894775913c86fef878"
CONFIG = Path("configs/experiment/h1.40-r2-g2-multigeneration-v1.json")
POOL = (
    "GPU-00000000-0000-0000-0000-000000000000",
    "GPU-00000000-0000-0000-0000-000000000000",
    "GPU-00000000-0000-0000-0000-000000000000",
    "GPU-00000000-0000-0000-0000-000000000000",
    "GPU-00000000-0000-0000-0000-000000000000",
    "GPU-00000000-0000-0000-0000-000000000000",
)
MANIFESTS = {
    "exclusion_ledger": "configs/experiment/manifests/h1.40-g2-exclusion-ledger-v1.json",
    "e1_retention": "configs/experiment/manifests/h1.40-g2-e1-retention-set-v1.json",
    "e2_reservoir": "configs/experiment/manifests/h1.40-g2-e2-reservoir-v1.json",
    "h1_39_preservation": "configs/experiment/manifests/h1.40-g2-h139-preservation-anchor-v1.json",
    "t2": "configs/experiment/manifests/h1.40-g2-t2-panel-v1.json",
}


def read(root: Path, path: Path) -> dict[str, Any]:
    value = json.loads((root / path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha(root: Path, path: Path) -> str:
    return hashlib.sha256((root / path).read_bytes()).hexdigest()


def validate(root: Path) -> dict[str, Any]:
    root = root.resolve()
    activation = read(root, ACTIVATION)
    config = read(root, CONFIG)
    current = read(root, Path("configs/experiment/current-campaign.json"))
    if activation.get("campaign_id") != CAMPAIGN_ID or activation.get("status") != "launch-authorized" or activation.get("formal_task_consumption_allowed") is not True:
        raise ValueError("R2 activation is not launch-authorized")
    prereg = activation.get("preregistration") or {}
    if prereg.get("path") != str(CONFIG) or prereg.get("sha256") != sha(root, CONFIG):
        raise ValueError("R2 preregistration hash mismatch")
    original = activation.get("original_frozen_contract") or {}
    original_path = Path("configs/experiment/h1.40-g2-multigeneration-v1.json")
    if original.get("sha256") != "5ba0e3e2d6fc1738bd0214eb1b05d3acb7d4ff853ebad19c32442d4e6c4f526f" or sha(root, original_path) != original.get("sha256"):
        raise ValueError("original frozen scientific config hash mismatch")
    if config.get("status") != "preregistered-pre-execution" or config.get("formal_task_consumption_allowed") is not False:
        raise ValueError("R2 scientific contract was opened or edited")
    for name, path in MANIFESTS.items():
        expected = (activation.get("manifest_sha256") or {}).get(name)
        if expected != sha(root, Path(path)):
            raise ValueError(f"manifest hash mismatch: {name}")
    if tuple((activation.get("parent_policy") or {}).get("authorized_gpu_uuids", ())) != POOL:
        raise ValueError("R2 activation GPU set does not match standing pool")
    if activation.get("zero_state_at_activation") != {"formal_task_consumption": 0, "formal_gpu_hours_consumed": 0.0, "g2_adapter_created": False, "t2_panel_evaluated": False, "qualification_status": "not-started"}:
        raise ValueError("R2 activation did not preserve zero state")
    replacement = activation.get("replacement") or {}
    if replacement.get("attempt1_engineering_invalid_gpu_hours") != "unknown" or replacement.get("attempt1_pooled_into_r2") is not False:
        raise ValueError("Attempt 1 unknown was pooled into R2")
    if is_external_successor_router(
        current,
        {CAMPAIGN_ID, SUCCESSOR_CAMPAIGN_ID},
        predecessor_routes={str(CONFIG)},
    ):
        return {
            "status": "pass",
            "campaign_id": CAMPAIGN_ID,
            "formal_task_consumption_allowed": False,
            "formal_task_consumption": 640,
            "formal_gpu_hours_consumed": 3.7551676636612837,
            "authorized_gpu_count": len(POOL),
        }
    successor_router = current.get("campaign_id") == SUCCESSOR_CAMPAIGN_ID and current.get("replacement_of") == CAMPAIGN_ID
    if successor_router:
        if current.get("status") == "launch-authorized" and current.get("formal_task_consumption_allowed") is not True:
            raise ValueError("launch-authorized successor router unexpectedly closes the H1.40 replacement boundary")
        if current.get("status") == "blocked-pre-execution" and current.get("formal_task_consumption_allowed") is not False:
            raise ValueError("blocked successor router unexpectedly opens the H1.40 replacement boundary")
        return {"status": "pass", "campaign_id": CAMPAIGN_ID, "formal_task_consumption_allowed": False, "formal_task_consumption": 640, "formal_gpu_hours_consumed": 3.7551676636612837, "authorized_gpu_count": len(POOL)}
    terminal_router = current.get("campaign_id") == CAMPAIGN_ID and current.get("status") == "closed-upstream-gates-failed"
    active_router = current.get("campaign_id") == CAMPAIGN_ID and current.get("status") == "launch-authorized"
    if not active_router and not terminal_router:
        raise ValueError("current campaign is not routed to active or terminal R2")
    if active_router and current.get("formal_task_consumption_allowed") is not True:
        raise ValueError("launch-authorized R2 router is unexpectedly closed")
    if terminal_router and current.get("formal_task_consumption_allowed") is not False:
        raise ValueError("terminal R2 router is unexpectedly open")
    latest = current.get("latest_event") or {}
    post_e2 = latest.get("classification") == "r2-e2-harvest-selection-passed"
    post_g2 = latest.get("classification") == "r2-g2-training-passed"
    post_upstream = latest.get("classification") == "r2-upstream-gates-failed"
    if latest.get("classification") not in {"r2-activation-published", "r2-zero-task-qualification-passed", "r2-formal-execution-release-published", "r2-e2-harvest-selection-passed", "r2-g2-training-passed", "r2-upstream-gates-failed"}:
        raise ValueError("current R2 activation event classification is unsupported")
    if post_upstream:
        if latest.get("formal_task_consumption") != 640 or latest.get("formal_gpu_hours_consumed") != 3.7551676636612837 or not latest.get("g2_adapter_created") or latest.get("t2_panel_evaluated"):
            raise ValueError("current R2 upstream closeout denominator/state mismatch")
        if (latest.get("upstream_evaluation") or {}).get("status") != "complete-upstream-gate-failed":
            raise ValueError("current R2 upstream closeout status mismatch")
    elif post_g2:
        if latest.get("formal_task_consumption") != 256 or latest.get("formal_gpu_hours_consumed") != 2.219149268141514 or not latest.get("g2_adapter_created") or latest.get("t2_panel_evaluated"):
            raise ValueError("current R2 G2 event denominator/state mismatch")
        if latest.get("g2_training_reconciliation") != str(G2_RECONCILIATION) or latest.get("g2_training_reconciliation_sha256") != G2_RECONCILIATION_SHA256 or sha(root, G2_RECONCILIATION) != G2_RECONCILIATION_SHA256:
            raise ValueError("current R2 G2 reconciliation pointer/hash mismatch")
        g2 = read(root, G2_RECONCILIATION)
        if g2.get("status") != "pass" or g2.get("increment_count") != 8 or g2.get("training_record_count") != 16 or g2.get("resource_unknown_count") != 0:
            raise ValueError("current R2 G2 reconciliation is not a complete passing training result")
    elif post_e2:
        if latest.get("formal_task_consumption") != 256 or latest.get("formal_gpu_hours_consumed") != 2.185317164430954:
            raise ValueError("current R2 E2 event denominator mismatch")
        if latest.get("g2_adapter_created") or latest.get("t2_panel_evaluated"):
            raise ValueError("current R2 E2 event opened a downstream stage")
        if latest.get("e2_harvest_reconciliation") != str(E2_RECONCILIATION) or latest.get("e2_harvest_reconciliation_sha256") != sha(root, E2_RECONCILIATION):
            raise ValueError("current R2 E2 reconciliation pointer/hash mismatch")
        e2 = read(root, E2_RECONCILIATION)
        if e2.get("status") != "pass" or e2.get("frozen_attempt_matrix", {}).get("attempt_count") != 256 or e2.get("resource_accounting", {}).get("resource_unknown_count") != 0 or e2.get("selection", {}).get("record_count") != 16:
            raise ValueError("current R2 E2 reconciliation is not a complete passing selection")
    elif latest.get("formal_task_consumption") != 0 or latest.get("formal_gpu_hours_consumed") != 0.0 or latest.get("g2_adapter_created") or latest.get("t2_panel_evaluated"):
        raise ValueError("current R2 pre-E2 event is not zero-state")
    if latest.get("classification") == "r2-zero-task-qualification-passed" and (latest.get("qualification") or {}).get("status") != "pass":
        raise ValueError("current R2 qualification state is not pass")
    if latest.get("classification") in {"r2-formal-execution-release-published", "r2-e2-harvest-selection-passed", "r2-g2-training-passed", "r2-upstream-gates-failed"}:
        if latest.get("release_receipt") != str(RELEASE):
            raise ValueError("current R2 event does not point to the formal release receipt")
        if latest.get("release_receipt_sha256") != sha(root, RELEASE):
            raise ValueError("current R2 release receipt hash mismatch")
        release = read(root, RELEASE)
        if release.get("status") != "pass" or release.get("formal_task_consumption_allowed") is not True:
            raise ValueError("formal release receipt does not authorize execution")
        if release.get("contract_sha256") != "sha256:" + sha(root, CONFIG):
            raise ValueError("formal release receipt contract hash mismatch")
        if release.get("qualification", {}).get("qualification_gpu_hours") != 0.007486578837253748:
            raise ValueError("formal release receipt qualification accounting mismatch")
        qualification = release.get("qualification") or {}
        evidence_dir = Path(str(qualification.get("evidence_dir", "")))
        for name, expected in (qualification.get("evidence_sha256") or {}).items():
            if sha(root, evidence_dir / name) != expected:
                raise ValueError(f"formal release qualification evidence hash mismatch: {name}")
    return {"status": "pass", "campaign_id": CAMPAIGN_ID, "formal_task_consumption_allowed": current.get("formal_task_consumption_allowed"), "formal_task_consumption": latest.get("formal_task_consumption"), "formal_gpu_hours_consumed": latest.get("formal_gpu_hours_consumed"), "authorized_gpu_count": len(POOL)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps(validate(args.root), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
