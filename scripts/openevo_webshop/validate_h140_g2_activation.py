#!/usr/bin/env python3
"""Fail-closed validator for the H1.40-G2 execution activation amendment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.openevo_webshop.campaign_router import is_external_successor_router

CAMPAIGN_ID = "20260820-0140-h140-g2-multigeneration"
SUCCESSOR_CAMPAIGN_ID = "20260820-0140r2-h140-g2-multigeneration"
POST_H140_SUCCESSOR_CAMPAIGN_ID = "20260821-0141-h141-magnitude-screen"
ACTIVATION = Path("configs/experiment/activations/h1.40-g2-execution-activation-v1.json")
CONFIG = Path("configs/experiment/h1.40-g2-multigeneration-v1.json")
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
        raise ValueError(f"expected object: {path}")
    return value


def file_sha(root: Path, path: Path) -> str:
    return hashlib.sha256((root / path).read_bytes()).hexdigest()


def validate(root: Path) -> dict[str, Any]:
    root = root.resolve()
    activation = read(root, ACTIVATION)
    config = read(root, CONFIG)
    current = read(root, Path("configs/experiment/current-campaign.json"))
    if activation.get("campaign_id") != CAMPAIGN_ID or activation.get("status") != "launch-authorized":
        raise ValueError("activation campaign/status mismatch")
    if activation.get("formal_task_consumption_allowed") is not True:
        raise ValueError("activation does not open formal consumption")
    if config.get("status") != "preregistered-pre-execution" or config.get("formal_task_consumption_allowed") is not False:
        raise ValueError("frozen H1.40 scientific contract was edited or opened")
    prereg = activation.get("preregistration") or {}
    if prereg.get("path") != str(CONFIG) or prereg.get("sha256") != file_sha(root, CONFIG):
        raise ValueError("frozen preregistration hash mismatch")
    for name, path in MANIFESTS.items():
        expected = (activation.get("manifest_sha256") or {}).get(name)
        if expected != file_sha(root, Path(path)):
            raise ValueError(f"manifest hash mismatch: {name}")
    policy = activation.get("parent_policy") or {}
    if tuple(policy.get("authorized_gpu_uuids", ())) != POOL:
        raise ValueError("activation GPU set does not match current parent standing pool")
    if activation.get("zero_state_at_activation") != {
        "formal_task_consumption": 0,
        "formal_gpu_hours_consumed": 0.0,
        "g2_adapter_created": False,
        "t2_panel_evaluated": False,
    }:
        raise ValueError("activation did not preserve zero formal state")
    if is_external_successor_router(
        current,
        {CAMPAIGN_ID, SUCCESSOR_CAMPAIGN_ID, POST_H140_SUCCESSOR_CAMPAIGN_ID},
        predecessor_routes={str(CONFIG)},
    ):
        return {
            "status": "pass",
            "campaign_id": CAMPAIGN_ID,
            "activation": str(ACTIVATION),
            "formal_task_consumption_allowed": activation.get("formal_task_consumption_allowed"),
            "authorized_gpu_count": len(POOL),
            "formal_task_consumption": 0,
            "formal_gpu_hours_consumed": 0.0,
            "runtime_requalification_required": True,
        }
    historical_router = current.get("campaign_id") == CAMPAIGN_ID and current.get("status") in {"launch-authorized", "blocked-pre-execution"}
    successor_router = current.get("campaign_id") in {SUCCESSOR_CAMPAIGN_ID, POST_H140_SUCCESSOR_CAMPAIGN_ID} and current.get("replacement_of") in {CAMPAIGN_ID, SUCCESSOR_CAMPAIGN_ID} and current.get("status") in {"preregistered-pre-execution", "launch-authorized", "blocked-pre-execution", "closed-upstream-gates-failed", "closed-measurement-invalid", "closed-result"}
    if not historical_router and not successor_router:
        raise ValueError("current campaign is not routed to active/blocked H1.40 or its closed successor")
    latest = current.get("latest_event") or {}
    if historical_router and (latest.get("classification") not in {"activation-published", "runtime-gate-blocked"} or latest.get("formal_task_consumption") != 0 or latest.get("formal_gpu_hours_consumed") != 0.0):
        raise ValueError("current campaign activation event is not zero-state")
    if successor_router:
        post_g2 = latest.get("classification") == "r2-g2-training-passed"
        post_e2 = latest.get("classification") == "r2-e2-harvest-selection-passed"
        post_upstream = latest.get("classification") == "r2-upstream-gates-failed"
        if post_upstream:
            if latest.get("formal_task_consumption") != 640 or latest.get("formal_gpu_hours_consumed") != 3.7551676636612837 or not latest.get("g2_adapter_created") or latest.get("t2_panel_evaluated"):
                raise ValueError("successor router upstream closeout denominator/state mismatch")
        elif post_g2:
            if latest.get("formal_task_consumption") != 256 or latest.get("formal_gpu_hours_consumed") != 2.219149268141514 or not latest.get("g2_adapter_created") or latest.get("t2_panel_evaluated"):
                raise ValueError("successor router G2 denominator/state mismatch")
        elif post_e2:
            if latest.get("formal_task_consumption") != 256 or latest.get("formal_gpu_hours_consumed") != 2.185317164430954:
                raise ValueError("successor router E2 denominator mismatch")
        elif current.get("campaign_id") == POST_H140_SUCCESSOR_CAMPAIGN_ID:
            if current.get("status") == "closed-measurement-invalid":
                if (
                    latest.get("classification") != "h1.41-v2-reconciled-measurement-invalid"
                    or latest.get("formal_task_consumption") != 384
                    or latest.get("formal_gpu_hours_consumed") != 1.5673375382128392
                    or not latest.get("g2_adapter_created")
                    or latest.get("t2_panel_evaluated")
                ):
                    raise ValueError("H1.41 closed measurement-invalid router state mismatch")
            elif (
                latest.get("classification") not in {"h1.41-launch-authorized", "h1.41-preregistration-blocked", "h1.41-v2-launch-authorized"}
                or latest.get("formal_task_consumption") != 0
                or latest.get("formal_gpu_hours_consumed") != 0.0
                or latest.get("g2_adapter_created")
                or latest.get("t2_panel_evaluated")
            ):
                raise ValueError("H1.41 successor router does not preserve its zero state")
        elif latest.get("classification") not in {"r2-preregistration-published", "r2-activation-published", "r2-zero-task-qualification-passed", "r2-formal-execution-release-published"} or latest.get("formal_task_consumption") != 0 or latest.get("formal_gpu_hours_consumed") != 0.0:
            raise ValueError("successor router does not preserve its current state")
    if historical_router and current.get("status") == "launch-authorized" and current.get("formal_task_consumption_allowed") is not True:
        raise ValueError("launch-authorized H1.40 router is unexpectedly closed")
    if historical_router and current.get("status") == "blocked-pre-execution" and current.get("formal_task_consumption_allowed") is not False:
        raise ValueError("blocked H1.40 router is unexpectedly open")
    if successor_router and current.get("status") == "preregistered-pre-execution" and current.get("formal_task_consumption_allowed") is not False:
        raise ValueError("closed R2 successor router is unexpectedly open")
    if successor_router and current.get("status") == "launch-authorized" and current.get("formal_task_consumption_allowed") is not True:
        raise ValueError("launch-authorized R2 successor router is unexpectedly closed")
    if current.get("authorized_gpu_uuids") != list(POOL):
        raise ValueError("current campaign GPU allowlist mismatch")
    return {
        "status": "pass",
        "campaign_id": CAMPAIGN_ID,
        "activation": str(ACTIVATION),
        "formal_task_consumption_allowed": activation.get("formal_task_consumption_allowed"),
        "authorized_gpu_count": len(POOL),
        "formal_task_consumption": 0,
        "formal_gpu_hours_consumed": 0.0,
        "runtime_requalification_required": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps(validate(args.root), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
