#!/usr/bin/env python3
"""Select the first frozen 8x2 qualified E2 records from CPU-readable harvest rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


TARGET_IDENTITIES = 8
RECORDS_PER_IDENTITY = 2


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def select(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = manifest.get("tasks") or []
    task_order = [str(task["task_id"]) for task in tasks]
    seeds = [int(seed) for seed in manifest["rollout_seeds"]]
    expected = {(task_id, seed) for task_id in task_order for seed in seeds}
    observed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("task_id")), int(row.get("rollout_seed")))
        if key not in expected:
            raise ValueError(f"row outside frozen E2 harvest matrix: {key}")
        if key in observed:
            raise ValueError(f"duplicate E2 harvest row: {key}")
        observed[key] = row
    missing = sorted(expected - set(observed))
    if missing:
        raise ValueError(f"incomplete E2 harvest matrix: missing {len(missing)} rows")

    selected: list[dict[str, Any]] = []
    ready_tasks: list[str] = []
    for task_id in task_order:
        qualified = [observed[(task_id, seed)] for seed in seeds if observed[(task_id, seed)].get("qualified_positive") is True]
        if len(qualified) >= RECORDS_PER_IDENTITY:
            ready_tasks.append(task_id)
            selected.extend(qualified[:RECORDS_PER_IDENTITY])
        if len(ready_tasks) == TARGET_IDENTITIES:
            break
    if len(ready_tasks) < TARGET_IDENTITIES:
        return {
            "schema": "openevo.webshop-h1.40-g2-e2-selection.v1",
            "status": "MGH HARVEST_INSUFFICIENT",
            "qualified_identity_count": len(ready_tasks),
            "required_identity_count": TARGET_IDENTITIES,
            "selected_records": [],
            "selection_outcome_blind": False,
            "reason": "fewer than 8 reservoir identities reached two qualified independent successes within 256 frozen attempts",
        }
    records = [{"task_id": str(row["task_id"]), "rollout_seed": int(row["rollout_seed"]), "record_sha256": str(row.get("record_sha256") or digest(row))} for row in selected]
    result = {
        "schema": "openevo.webshop-h1.40-g2-e2-selection.v1",
        "status": "pass",
        "selection_rule": "first 8 reservoir identities in frozen manifest order with at least 2 qualified successes; first 2 qualified rows in frozen seed order; no reward magnitude ranking",
        "qualified_identity_count": len(ready_tasks),
        "selected_task_ids": ready_tasks,
        "selected_records": records,
        "record_count": len(records),
        "selection_outcome_blind": True,
        "selection_sha256": digest({"selected_task_ids": ready_tasks, "selected_records": records}),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--harvest-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.harvest_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = select(manifest, rows)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "record_count": result.get("record_count", 0), "selection_sha256": result.get("selection_sha256")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
