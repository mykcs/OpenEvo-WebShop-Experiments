#!/usr/bin/env python3
"""Fail-closed reconciliation for the train-only H1.38 feasibility reservoir."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_SEEDS = [2026090400, 2026090401, 2026090402, 2026090403]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    manifest_path = args.run_root / "manifests" / "h1.38-endogenous-reservoir-manifest.json"
    manifest = read_json(manifest_path)
    tasks = [int(row["goal_idx"]) for row in manifest.get("tasks", [])]
    expected = config["budget"]
    if len(tasks) != expected["task_count"] or len(set(tasks)) != len(tasks):
        raise SystemExit("manifest task set mismatch")
    if manifest.get("controls", {}).get("rollout_seeds") != EXPECTED_SEEDS:
        raise SystemExit("manifest seed contract mismatch")

    rows: list[dict[str, Any]] = []
    missing: list[int] = []
    malformed: list[dict[str, Any]] = []
    for task in tasks:
        path = args.run_root / "harvest" / f"task{task}" / "attempts.jsonl"
        if not path.is_file():
            missing.append(task)
            continue
        cell = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        seeds = sorted(int(row.get("rollout_seed", -1)) for row in cell)
        if len(cell) != 4 or seeds != EXPECTED_SEEDS or any(int(row.get("goal_idx", -1)) != task or row.get("arm") != "self-generated-harvest" for row in cell):
            malformed.append({"task": task, "path": str(path), "row_count": len(cell), "seeds": seeds})
            continue
        rows.extend(cell)
    if (missing or malformed) and not args.allow_incomplete:
        raise SystemExit(json.dumps({"missing_cells": missing, "malformed_cells": malformed}, sort_keys=True))
    if not args.allow_incomplete and len(rows) != expected["maximum_logical_attempts"]:
        raise SystemExit("attempt count mismatch")

    def valid(row: dict[str, Any]) -> bool:
        return bool(row.get("identity_ok")) and bool(row.get("trace_complete")) and not bool(row.get("parser_or_fallback_failure")) and row.get("error") is None

    qualified = [row for row in rows if valid(row) and float(row.get("final_task_score") or 0.0) > 0.0]
    qualified_ids = sorted({str(row["task_id"]) for row in qualified})
    parser_failures = sum(bool(row.get("parser_or_fallback_failure")) for row in rows)
    identity_failures = sum(not bool(row.get("identity_ok")) for row in rows)
    incomplete = sum(not bool(row.get("trace_complete")) for row in rows)
    runtime = sum(row.get("error") is not None for row in rows)
    def record_hash(row: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    identity_ledger: dict[str, dict[str, Any]] = {}
    for task in tasks:
        task_rows = [row for row in rows if int(row.get("goal_idx", -1)) == task]
        successes = [row for row in task_rows if row in qualified]
        trace_steps = [step for row in task_rows for step in ((row.get("episode") or {}).get("trace") or []) if isinstance(step, dict)]
        valid_actions = sum(bool(step.get("action_valid")) for step in trace_steps)
        task_id = f"webshop_{task:06d}"
        identity_ledger[task_id] = {
            "goal_idx": task,
            "task_description_sha256": next((str(row.get("task_description_sha256")) for row in manifest["tasks"] if int(row["goal_idx"]) == task), None),
            "logical_attempts": len(task_rows),
            "qualified_successes": len(successes),
            "qualified_success_rollout_seeds": sorted(int(row["rollout_seed"]) for row in successes),
            "qualified_success_record_sha256": [record_hash(row) for row in successes],
            "qualified_success_rewards": [float(row.get("final_task_score") or 0.0) for row in successes],
            "qualified_success_trajectory_lengths": [len(((row.get("episode") or {}).get("trace") or [])) for row in successes],
            "valid_action_rate": valid_actions / len(trace_steps) if trace_steps else None,
            "parser_or_fallback_failures": sum(bool(row.get("parser_or_fallback_failure")) for row in task_rows),
            "multi_rollout_ready": len(successes) >= 2,
        }
    multi_ready = sorted(task_id for task_id, item in identity_ledger.items() if item["multi_rollout_ready"])
    gpu_hours = 0.0
    gpu_uuid_by_task: dict[str, str] = {}
    for task in tasks:
        cell = args.run_root / "harvest" / f"task{task}"
        summary = cell / "summary.json"
        before = cell / "gpu-before.json"
        if summary.is_file():
            gpu_hours += float(read_json(summary).get("wall_time_seconds") or 0.0) / 3600
        if before.is_file():
            gpu_uuid_by_task[f"webshop_{task:06d}"] = str(read_json(before).get("expected_uuid") or "")
    sufficient = len(qualified) >= 32 and len(qualified_ids) >= 16 and len(multi_ready) >= 8
    feasibility = len(qualified_ids) >= 16 and len(qualified) >= 32
    outcome = {
        "schema": "openevo.webshop-h1.38-endogenous-reservoir-analysis.v2",
        "campaign_id": config["campaign_id"],
        "manifest_file_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "expected_attempts": expected["maximum_logical_attempts"],
        "observed_attempts": len(rows),
        "task_count": len(tasks),
        "rollout_seeds": EXPECTED_SEEDS,
        "valid_attempts": sum(valid(row) for row in rows),
        "qualified_success_episode_count": len(qualified),
        "qualified_success_task_count": len(qualified_ids),
        "qualified_success_task_ids": qualified_ids,
        "multi_rollout_ready_task_count": len(multi_ready),
        "multi_rollout_ready_task_ids": multi_ready,
        "identity_ledger": identity_ledger,
        "records_per_identity": {task_id: item["qualified_successes"] for task_id, item in identity_ledger.items()},
        "gpu_uuid_by_task": gpu_uuid_by_task,
        "measured_gpu_hours": gpu_hours,
        "qualification_rate": len(qualified) / len(rows),
        "failure_taxonomy": {
            "parser_or_fallback": parser_failures,
            "identity_failure": identity_failures,
            "incomplete_trace": incomplete,
            "runtime_error": runtime,
        },
        "sufficiency_reached": sufficient,
        "decision": "feasible-for-preregistration-only" if feasibility else "feasibility-negative",
        "threshold": {"minimum_qualified_task_identities": 16, "minimum_qualified_episodes": 32, "minimum_multi_rollout_ready_task_identities": 8},
        "claim_boundary": "Train-only success-supply measurement only; no adapter was trained and no held-out evaluation was accessed.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(outcome, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": outcome["decision"], "qualified_success_task_count": len(qualified_ids), "qualified_success_episode_count": len(qualified), "multi_rollout_ready_task_count": len(multi_ready), "sufficiency_reached": sufficient}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
