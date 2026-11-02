#!/usr/bin/env python3
"""Fail-closed reconciliation for the H1.41 v2 magnitude screen.

The historical G1 result is a deterministic task-level reference, not a
same-attempt paired arm. Any malformed or invalid C0/C1 attempt invalidates
the stage and cannot be rescued by scoring it as zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any


CAMPAIGN_ID = "20260821-0141-h141-magnitude-screen"
CONFIG = Path("configs/experiment/h1.41-magnitude-screen-v2.json")
REFERENCE = Path("docs/evidence/remote-runs/2026-08-21/h1.41/h1.41-g1-deterministic-reference-v1.json")
SCHEDULER_NAME = "scheduler-receipt.json"
STAGES = ("acquisition", "retention", "preservation")
STAGE_TO_PANEL = {"acquisition": "acquisition_e2", "retention": "retention_e1", "preservation": "preservation_h139"}
STAGE_TO_RUNNER = {"acquisition": "acquisition", "retention": "retention", "preservation": "fresh"}
ARMS = ("C0_exact_h140", "C1_reset_magnitude")
BOOTSTRAP_ROUNDS = 10_000
BOOTSTRAP_SEED = 2026082140
SIGN_FLIP_ROUNDS = 10_000
SIGN_FLIP_SEED = 2026082141
PRESERVATION_MARGIN = 0.05


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bootstrap_ci(values: list[float], seed: int) -> dict[str, Any]:
    if not values:
        return {"status": "insufficient", "task_cluster_count": 0, "rounds": BOOTSTRAP_ROUNDS, "seed": seed}
    rng = random.Random(seed)
    samples = [sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(BOOTSTRAP_ROUNDS)]
    return {
        "status": "pass",
        "task_cluster_count": len(values),
        "rounds": BOOTSTRAP_ROUNDS,
        "seed": seed,
        "point_estimate": sum(values) / len(values),
        "ci95": [quantile(samples, 0.025), quantile(samples, 0.975)],
    }


def sign_flip(values: list[float], seed: int) -> dict[str, Any]:
    if not values:
        return {"status": "insufficient", "task_cluster_count": 0, "rounds": SIGN_FLIP_ROUNDS, "seed": seed}
    rng = random.Random(seed)
    observed = abs(sum(values) / len(values))
    extreme = sum(abs(sum(value if rng.randrange(2) else -value for value in values) / len(values)) >= observed for _ in range(SIGN_FLIP_ROUNDS))
    return {"status": "pass", "task_cluster_count": len(values), "rounds": SIGN_FLIP_ROUNDS, "seed": seed, "p_value": (extreme + 1) / (SIGN_FLIP_ROUNDS + 1)}


def reference_scores(root: Path, config: dict[str, Any]) -> dict[tuple[str, str], float]:
    reference = load(root / REFERENCE)
    if sha(root / REFERENCE) != config["source_artifacts"]["g1_deterministic_reference_sha256"]:
        raise ValueError("deterministic G1 reference hash mismatch")
    source = reference.get("source") or {}
    seeds = source.get("historical_rollout_seeds")
    if source.get("temperature") != 0.0 or not isinstance(seeds, list) or len(seeds) != 4:
        raise ValueError("deterministic G1 reference source contract mismatch")
    output: dict[tuple[str, str], float] = {}
    for stage, panel_name in STAGE_TO_PANEL.items():
        expected_ids = config["frozen_scientific_contract"]["evaluation_panels"][panel_name]["task_ids"]
        rows = {row.get("task_id"): row for row in (reference.get("panels") or {}).get(stage, [])}
        if set(rows) != set(expected_ids):
            raise ValueError(f"deterministic G1 reference task set mismatch: {stage}")
        for task_id in expected_ids:
            row = rows[task_id]
            scores = row.get("scores_by_seed") or {}
            if row.get("valid_attempt_count") != 4 or set(scores) != {str(seed) for seed in seeds} or len(set(scores.values())) != 1:
                raise ValueError(f"deterministic G1 reference is not invariant: {stage}/{task_id}")
            output[(stage, task_id)] = float(next(iter(scores.values())))
    return output


def row_valid(row: dict[str, Any], task_id: str, goal_idx: int, stage: str, seeds: list[int], identity: dict[str, Any]) -> tuple[bool, str | None]:
    if row.get("task_id") != task_id or row.get("goal_idx") != goal_idx:
        return False, "task-or-goal-identity"
    if row.get("arm") != "sd-lora" or row.get("evaluation_stage") != STAGE_TO_RUNNER[stage] or row.get("rollout_seed") not in seeds:
        return False, "runner-identity"
    receipt = row.get("adapter_load_receipt")
    if not isinstance(receipt, dict) or receipt.get("verified") is not True or receipt.get("adapter_loaded") is not True:
        return False, "adapter-load-unverified"
    if receipt.get("adapter_file_sha256") != identity.get("adapter_file_sha256") or receipt.get("loaded_tensor_digest") != identity.get("tensor_digest"):
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
    config = load(root / CONFIG)
    refs = reference_scores(root, config)
    scheduler_path = run_root / SCHEDULER_NAME
    scheduler = load(scheduler_path)
    if scheduler.get("schema") != "openevo.webshop-h1.41-upstream-evaluation-scheduler.v1" or scheduler.get("campaign_id") != CAMPAIGN_ID or scheduler.get("status") != "pass":
        raise ValueError("H1.41 scheduler is not a complete passing result")
    if scheduler.get("expected_cell_count") != 96 or scheduler.get("completed_cell_count") != 96 or len(scheduler.get("cells", [])) != 96:
        raise ValueError("H1.41 scheduler cell denominator mismatch")
    identities = scheduler.get("adapter_identities")
    if not isinstance(identities, dict) or set(identities) != set(ARMS):
        raise ValueError("H1.41 adapter identities are incomplete")

    panels = config["frozen_scientific_contract"]["evaluation_panels"]
    stage_results: dict[str, Any] = {}
    total_gpu_hours = 0.0
    for stage_index, stage in enumerate(STAGES):
        panel = panels[STAGE_TO_PANEL[stage]]
        task_ids = panel["task_ids"]
        seeds = panel["rollout_seeds"]
        by_arm = {arm: {} for arm in ARMS}
        invalid: dict[str, list[str]] = {arm: [] for arm in ARMS}
        arm_cells: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
        for cell in scheduler["cells"]:
            if cell.get("stage") != stage or cell.get("arm") not in ARMS:
                continue
            arm = cell["arm"]
            task_id = str(cell.get("task_id"))
            arm_cells[arm].append(cell)
            accounting = cell.get("resource_accounting")
            if not isinstance(accounting, dict) or accounting.get("status") != "exact":
                invalid[arm].append(f"{task_id}:resource-accounting")
            else:
                total_gpu_hours += float(accounting.get("gpu_hours", 0.0))
            if cell.get("return_code") != 0:
                invalid[arm].append(f"{task_id}:payload-exit-{cell.get('return_code')}")
            attempts_path = run_root / "cells" / stage / arm / task_id / "attempts.jsonl"
            if not attempts_path.is_file():
                invalid[arm].append(f"{task_id}:missing-attempts")
                continue
            rows = [json.loads(line) for line in attempts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(rows) != len(seeds):
                invalid[arm].append(f"{task_id}:attempt-count")
            scores = []
            for row in rows:
                good, reason = row_valid(row, task_id, int(cell["goal_idx"]), stage, seeds, identities[arm])
                if good:
                    scores.append(float(row["final_task_score"]))
                elif reason:
                    invalid[arm].append(f"{task_id}:{reason}")
            if len(scores) == len(seeds):
                by_arm[arm][task_id] = sum(scores) / len(scores)
        for arm in ARMS:
            missing = sorted(set(task_ids) - set(by_arm[arm]))
            invalid[arm].extend(f"{task}:no-complete-valid-cluster" for task in missing)
            if len(arm_cells[arm]) != len(task_ids):
                invalid[arm].append(f"cell-count:{len(arm_cells[arm])}")
        stage_arm_valid = all(not invalid[arm] for arm in ARMS)
        stage_metrics: dict[str, Any] = {}
        contrasts: dict[str, Any] = {}
        for arm_index, arm in enumerate(ARMS):
            values = [by_arm[arm][task] - refs[(stage, task)] for task in task_ids if task in by_arm[arm]]
            contrasts[f"{arm}_minus_G1"] = {"ci": bootstrap_ci(values, BOOTSTRAP_SEED + stage_index * 10 + arm_index), "sign_flip": sign_flip(values, SIGN_FLIP_SEED + stage_index * 10 + arm_index)}
            stage_metrics[arm] = {"cell_count": len(arm_cells[arm]), "valid_task_clusters": len(values), "invalid_reasons": sorted(set(invalid[arm]))}
        paired = [by_arm[ARMS[1]][task] - by_arm[ARMS[0]][task] for task in task_ids if task in by_arm[ARMS[0]] and task in by_arm[ARMS[1]]]
        contrasts["C1_minus_C0"] = {"ci": bootstrap_ci(paired, BOOTSTRAP_SEED + stage_index * 10 + 2), "sign_flip": sign_flip(paired, SIGN_FLIP_SEED + stage_index * 10 + 2)}
        c0 = contrasts["C0_exact_h140_minus_G1"]["ci"]
        c1 = contrasts["C1_reset_magnitude_minus_G1"]["ci"]
        c0_pass = stage_arm_valid and c0.get("status") == "pass" and ((c0["point_estimate"] > 0.0) if stage == "acquisition" else c0["ci95"][0] > -PRESERVATION_MARGIN)
        c1_pass = stage_arm_valid and c1.get("status") == "pass" and ((c1["point_estimate"] > 0.0) if stage == "acquisition" else c1["ci95"][0] > -PRESERVATION_MARGIN)
        stage_results[stage] = {"arms": stage_metrics, "contrasts": contrasts, "stage_valid": stage_arm_valid, "C0_gate_pass": c0_pass, "C1_gate_pass": c1_pass, "gate_rule": "point estimate > 0" if stage == "acquisition" else f"lower 95% CI > {-PRESERVATION_MARGIN}"}

    all_valid = all(stage_results[stage]["stage_valid"] for stage in STAGES)
    c0_all = all(stage_results[stage]["C0_gate_pass"] for stage in STAGES)
    c1_all = all(stage_results[stage]["C1_gate_pass"] for stage in STAGES)
    favorable_c1_effect = all(
        stage_results[stage]["contrasts"]["C1_minus_C0"]["ci"].get("status") == "pass"
        and stage_results[stage]["contrasts"]["C1_minus_C0"]["ci"]["ci95"][0] > 0.0
        for stage in STAGES
    )
    if not all_valid:
        selector = "MEASUREMENT_INVALID"
    elif c1_all and favorable_c1_effect and stage_results["preservation"]["C1_gate_pass"]:
        selector = "C1_MECHANISM_SUPPORTED"
    elif c0_all and not c1_all:
        selector = "C0_ONLY_PASS"
    else:
        selector = "NULL_OR_INCONCLUSIVE"
    result = {
        "schema": "openevo.webshop-h1.41-magnitude-screen-reconciliation.v2",
        "status": "complete",
        "campaign_id": CAMPAIGN_ID,
        "scheduler_sha256": sha(scheduler_path),
        "reference_sha256": sha(root / REFERENCE),
        "formal_task_consumption": 384,
        "evaluation_gpu_hours": total_gpu_hours,
        "resource_unknown_count": 0,
        "stages": stage_results,
        "selector": selector,
        "t2_evaluation_allowed": False,
        "claim_boundary": "H1.41 updater mechanism screen only; no fresh-task transfer or H1.40 T2 claim is supported.",
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
