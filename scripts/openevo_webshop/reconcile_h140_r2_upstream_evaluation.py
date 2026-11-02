#!/usr/bin/env python3
"""Fail-closed reconciliation for the frozen R2 upstream evaluation gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CAMPAIGN_ID = "20260820-0140r2-h140-g2-multigeneration"
CONFIG = Path("configs/experiment/h1.40-r2-g2-multigeneration-v1.json")
SCHEDULER_NAME = "scheduler-receipt.json"
STAGES = ("acquisition", "retention", "preservation")
STAGE_COUNTS = {"acquisition": 8, "retention": 8, "preservation": 32}
RUNNER_EVALUATION_STAGES = {"acquisition": "acquisition", "retention": "retention", "preservation": "fresh"}
SEEDS = (2026104100, 2026104101, 2026104102, 2026104103)
BOOTSTRAP_ROUNDS = 10_000
BOOTSTRAP_SEED = 20260820404
RETENTION_MARGIN = 0.05


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile from an empty sample")
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def bootstrap_ci(differences: list[float], seed: int) -> dict[str, Any]:
    if not differences:
        return {"status": "insufficient", "task_cluster_count": 0, "rounds": BOOTSTRAP_ROUNDS, "seed": seed}
    rng = random.Random(seed)
    samples = [
        sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences)
        for _ in range(BOOTSTRAP_ROUNDS)
    ]
    return {
        "status": "pass",
        "task_cluster_count": len(differences),
        "rounds": BOOTSTRAP_ROUNDS,
        "seed": seed,
        "point_estimate": sum(differences) / len(differences),
        "ci95": [quantile(samples, 0.025), quantile(samples, 0.975)],
    }


def row_is_valid(row: dict[str, Any], expected_task_id: str, expected_goal_idx: int, expected_stage: str, identity: dict[str, str]) -> tuple[bool, str | None]:
    if row.get("task_id") != expected_task_id or row.get("goal_idx") != expected_goal_idx:
        return False, "task-or-seed-identity"
    if row.get("arm") != "sd-lora":
        return False, "runner-arm-label-drift"
    if row.get("evaluation_stage") != RUNNER_EVALUATION_STAGES[expected_stage]:
        return False, "evaluation-stage-identity"
    if row.get("rollout_seed") not in SEEDS:
        return False, "rollout-seed-identity"
    receipt = row.get("adapter_load_receipt")
    if not isinstance(receipt, dict) or receipt.get("verified") is not True or receipt.get("adapter_loaded") is not True:
        return False, "adapter-load-unverified"
    if receipt.get("adapter_file_sha256") != identity["adapter_file_sha256"] or receipt.get("loaded_tensor_digest") != identity["tensor_digest"]:
        return False, "adapter-identity-mismatch"
    if row.get("identity_ok") is not True or row.get("trace_complete") is not True:
        return False, "identity-or-trace-invalid"
    if row.get("parser_or_fallback_failure") is not False or row.get("error") not in (None, ""):
        return False, "parser-or-runtime-invalid"
    score = row.get("final_task_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        return False, "score-invalid"
    return True, None


def reconcile(root: Path, run_root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    run_root = run_root.resolve()
    scheduler_path = run_root / SCHEDULER_NAME
    scheduler = load(scheduler_path)
    if scheduler.get("schema") != "openevo.webshop-h1.40-r2-upstream-evaluation-scheduler.v1" or scheduler.get("campaign_id") != CAMPAIGN_ID or scheduler.get("status") != "pass":
        raise ValueError("upstream scheduler is not a complete passing result")
    expected_cells = scheduler.get("expected_cell_count")
    if expected_cells != 96 or scheduler.get("completed_cell_count") != expected_cells or len(scheduler.get("cells", [])) != expected_cells:
        raise ValueError("upstream scheduler cell denominator mismatch")
    identities = scheduler.get("adapter_identities")
    if not isinstance(identities, dict) or set(identities) != {"g1", "g2"}:
        raise ValueError("scheduler adapter identities are missing")

    by_stage_arm: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for cell in scheduler["cells"]:
        stage, arm = cell.get("stage"), cell.get("arm")
        if stage not in STAGES or arm not in {"g1", "g2"}:
            raise ValueError("scheduler contains an unknown stage or arm")
        by_stage_arm.setdefault((stage, arm), []).append(cell)

    stage_results: dict[str, Any] = {}
    total_gpu_hours = 0.0
    for stage_index, stage in enumerate(STAGES):
        expected_tasks = STAGE_COUNTS[stage]
        arm_results: dict[str, Any] = {}
        task_means: dict[str, dict[str, float | None]] = {}
        for arm in ("g1", "g2"):
            cells = by_stage_arm.get((stage, arm), [])
            if len(cells) != expected_tasks:
                raise ValueError(f"{stage}/{arm} cell count mismatch")
            seen_tasks: set[str] = set()
            invalid_reasons: list[str] = []
            valid_rows = 0
            qualified = 0
            cell_summaries: list[dict[str, Any]] = []
            for cell in cells:
                task_id = str(cell.get("task_id"))
                if task_id in seen_tasks:
                    raise ValueError(f"duplicate {stage}/{arm} task cell: {task_id}")
                seen_tasks.add(task_id)
                accounting = cell.get("resource_accounting")
                if not isinstance(accounting, dict) or accounting.get("status") != "exact":
                    raise ValueError(f"{stage}/{arm}/{task_id} resource receipt is not exact")
                total_gpu_hours += float(accounting["gpu_hours"])
                if cell.get("return_code") != 0:
                    invalid_reasons.append(f"{task_id}:payload-exit-{cell.get('return_code')}")
                attempts_path = run_root / "cells" / stage / arm / task_id / "attempts.jsonl"
                rows = [json.loads(line) for line in attempts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                if len(rows) != len(SEEDS):
                    invalid_reasons.append(f"{task_id}:attempt-count")
                scores: list[float] = []
                for row in rows:
                    expected_identity = identities[arm]
                    valid, reason = row_is_valid(row, task_id, int(cell["goal_idx"]), stage, expected_identity)
                    if valid:
                        valid_rows += 1
                        score = float(row["final_task_score"])
                        scores.append(score)
                        qualified += int(bool(row.get("qualified_positive")))
                    elif reason:
                        invalid_reasons.append(f"{task_id}:{reason}")
                task_means.setdefault(task_id, {})[arm] = sum(scores) / len(scores) if len(scores) == len(SEEDS) else None
                cell_summaries.append({"task_id": task_id, "valid_attempt_count": len(scores), "mean_task_score": task_means[task_id][arm]})
            arm_results[arm] = {"cell_count": len(cells), "valid_attempt_count": valid_rows, "qualified_positive_count": qualified, "invalid_reasons": invalid_reasons, "cells": sorted(cell_summaries, key=lambda value: value["task_id"])}
        differences = [float(values["g2"] - values["g1"]) for values in task_means.values() if values.get("g1") is not None and values.get("g2") is not None]
        ci = bootstrap_ci(differences, BOOTSTRAP_SEED + stage_index)
        structural_pass = all(not arm_results[arm]["invalid_reasons"] for arm in ("g1", "g2")) and ci.get("status") == "pass"
        if stage == "acquisition":
            gate_pass = structural_pass and float(ci.get("point_estimate", 0.0)) > 0.0
            rule = "G2-minus-G1 point estimate > 0"
        else:
            lower = ci.get("ci95", [float("-inf")])[0] if ci.get("status") == "pass" else float("-inf")
            gate_pass = structural_pass and lower > -RETENTION_MARGIN
            rule = f"G2-minus-G1 lower 95% CI > {-RETENTION_MARGIN}"
        stage_results[stage] = {"arms": arm_results, "task_cluster_count": len(differences), "g2_minus_g1": ci, "structural_pass": structural_pass, "gate_pass": gate_pass, "gate_rule": rule}

    upstream_pass = all(stage_results[stage]["gate_pass"] for stage in STAGES)
    result = {
        "schema": "openevo.webshop-h1.40-r2-upstream-evaluation-reconciliation.v1",
        "status": "pass" if upstream_pass else "complete-upstream-gate-failed",
        "campaign_id": CAMPAIGN_ID,
        "experiment_git_sha": scheduler.get("experiment_git_sha"),
        "scheduler_sha256": sha(scheduler_path),
        "formal_task_consumption": 384,
        "evaluation_gpu_hours": total_gpu_hours,
        "resource_unknown_count": 0,
        "stages": stage_results,
        "t2_evaluation_allowed": upstream_pass,
        "stopping_rule": "T2 remains unopened unless acquisition, retention, and H1.39 preservation gates all pass.",
        "claim_boundary": "Upstream G1-vs-G2 acquisition, retention, and preservation only; no fresh-task transfer or MG classification is supported by this reconciliation.",
    }
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reconcile(args.root, args.run_root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
