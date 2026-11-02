#!/usr/bin/env python3
"""Fail-closed H1.40-G2 preregistration and partition validator."""

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
CONFIG = Path("configs/experiment/h1.40-g2-multigeneration-v1.json")
MANIFEST_DIR = Path("configs/experiment/manifests")
MANIFESTS = {
    "ledger": "h1.40-g2-exclusion-ledger-v1.json",
    "e1": "h1.40-g2-e1-retention-set-v1.json",
    "e2": "h1.40-g2-e2-reservoir-v1.json",
    "anchor": "h1.40-g2-h139-preservation-anchor-v1.json",
    "t2": "h1.40-g2-t2-panel-v1.json",
}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def verify_self_hash(value: dict[str, Any], field: str, label: str) -> None:
    expected = value.get(field)
    payload = dict(value)
    payload.pop(field, None)
    if expected != digest(payload):
        raise ValueError(f"{label} self hash mismatch")


def verify_task_rows(manifest: dict[str, Any], expected_count: int, label: str) -> set[int]:
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != expected_count:
        raise ValueError(f"{label} task count mismatch")
    indices: set[int] = set()
    ids: set[str] = set()
    for position, task in enumerate(tasks):
        idx = task.get("goal_idx")
        task_name = task.get("task_id")
        text = task.get("task_description")
        if not isinstance(idx, int) or idx in indices or not isinstance(task_name, str) or task_name != f"webshop_{idx:06d}" or task.get("sample_index") != position:
            raise ValueError(f"{label} task identity/order mismatch")
        if task_name in ids or not isinstance(text, str) or hashlib.sha256(text.encode()).hexdigest() != task.get("task_description_sha256"):
            raise ValueError(f"{label} task description/hash mismatch")
        indices.add(idx)
        ids.add(task_name)
    return indices


def validate(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = read(root / CONFIG)
    if config.get("campaign_id") != CAMPAIGN_ID or config.get("status") != "preregistered-pre-execution" or config.get("formal_task_consumption_allowed") is not False:
        raise ValueError("H1.40 config is not closed at the preregistration-only boundary")
    if config.get("generations", {}).get("g2", {}).get("adapter_file_sha256") is not None or config.get("generations", {}).get("g2", {}).get("adapter_tensor_digest") is not None:
        raise ValueError("G2 adapter identity was pre-created")
    if config.get("formal_execution_guard") != {"tasks_consumed": 0, "gpu_hours_consumed": 0.0, "g2_adapter_created": False, "t2_panel_evaluated": False, "launcher_must_fail_closed_while_status_is_preregistered": True}:
        raise ValueError("formal execution guard is not zero/closed")
    loaded = {name: read(root / MANIFEST_DIR / path) for name, path in MANIFESTS.items()}
    verify_self_hash(loaded["ledger"], "ledger_sha256", "H1.40 exclusion ledger")
    for name in ("e1", "e2", "anchor", "t2"):
        verify_self_hash(loaded[name], "manifest_sha256", f"H1.40 {name} manifest")
    e1 = verify_task_rows(loaded["e1"], 8, "E1")
    e2 = verify_task_rows(loaded["e2"], 64, "E2")
    anchor = verify_task_rows(loaded["anchor"], 32, "H1.39 preservation")
    t2 = verify_task_rows(loaded["t2"], 64, "T2")
    partitions = {"e1": e1, "e2": e2, "anchor": anchor, "t2": t2}
    all_indices = set().union(*partitions.values())
    if sum(len(value) for value in partitions.values()) != len(all_indices):
        raise ValueError("H1.40 task partitions overlap")
    excluded = set(loaded["ledger"].get("excluded_goal_indices", []))
    reserved = set(loaded["ledger"].get("reserved_h140_goal_indices", []))
    # E1 and the H1.39 anchor intentionally revisit historical identities for
    # retention/preservation. Only the fresh E2 and T2 partitions must be
    # disjoint from the prior-identity exclusion ledger.
    if (e2 | t2) & excluded or not all_indices <= reserved:
        raise ValueError("H1.40 task partitions violate exclusion/reservation ledger")
    expected = {
        "e2": (2026104000, 2026104001, 2026104002, 2026104003),
        "e1": (2026104100, 2026104101, 2026104102, 2026104103),
        "anchor": (2026104100, 2026104101, 2026104102, 2026104103),
        "t2": (2026104200, 2026104201, 2026104202, 2026104203),
    }
    for name, seeds in expected.items():
        if tuple(loaded[name].get("rollout_seeds", [])) != seeds:
            raise ValueError(f"{name} seed contract mismatch")
    if loaded["e2"].get("attempts") != 256 or loaded["t2"].get("attempts") != 768 or loaded["anchor"].get("attempts") != 256 or loaded["e1"].get("attempts") != 64:
        raise ValueError("H1.40 matrix attempt count mismatch")
    # The scientific config remains permanently closed/frozen even after the
    # control-plane router is activated by a separate amendment.
    current = read(root / "configs/experiment/current-campaign.json")
    known_router = current.get("campaign_id") in {CAMPAIGN_ID, SUCCESSOR_CAMPAIGN_ID, POST_H140_SUCCESSOR_CAMPAIGN_ID}
    external_successor = is_external_successor_router(
        current,
        {CAMPAIGN_ID, SUCCESSOR_CAMPAIGN_ID, POST_H140_SUCCESSOR_CAMPAIGN_ID},
        predecessor_routes={str(CONFIG)},
    )
    if not known_router and not external_successor:
        raise ValueError("current-campaign is not routed to H1.40 or its sealed successor")
    if current.get("campaign_id") in {SUCCESSOR_CAMPAIGN_ID, POST_H140_SUCCESSOR_CAMPAIGN_ID} and current.get("replacement_of") not in {CAMPAIGN_ID, SUCCESSOR_CAMPAIGN_ID}:
        raise ValueError("current-campaign is not routed to H1.40 or its sealed successor")
    return {"status": "pass", "campaign_id": CAMPAIGN_ID, "partition_counts": {name: len(value) for name, value in partitions.items()}, "reserved_count": len(reserved), "formal_task_consumption_allowed": False, "g2_adapter_created": False, "t2_panel_evaluated": False, "router_status": current.get("status")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps(validate(args.root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
