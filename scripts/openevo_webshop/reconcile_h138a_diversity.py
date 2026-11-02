#!/usr/bin/env python3
"""Fail-closed reconciliation and task-cluster inference for H1.38A."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


ARMS = ("base", "diversity-4", "diversity-8", "diversity-16")
RESAMPLES = 10_000
RESAMPLE_SEED = 20260818138


def valid(row: dict[str, Any]) -> bool:
    return bool(row.get("identity_ok") and row.get("trace_complete") and not row.get("parser_or_fallback_failure") and row.get("error") is None)


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ci(values: list[float], rng: random.Random) -> list[float] | None:
    if not values:
        return None
    samples = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(RESAMPLES))
    return [samples[int(0.025 * (RESAMPLES - 1))], samples[int(0.975 * (RESAMPLES - 1))]]


def sign_flip(values: list[float], rng: random.Random) -> float | None:
    if not values:
        return None
    observed = abs(sum(values) / len(values))
    extreme = 0
    for _ in range(RESAMPLES):
        estimate = abs(sum(value if rng.randrange(2) else -value for value in values) / len(values))
        extreme += estimate >= observed
    return (extreme + 1) / (RESAMPLES + 1)


def reconcile(manifest_path: Path, cell_specs: list[str], output: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = [int(value) for value in manifest.get("evaluation_tasks", [])]
    seeds = [int(value) for value in manifest.get("evaluation", {}).get("rollout_seeds", [])]
    errors: list[str] = []
    if not tasks or len(tasks) != len(set(tasks)) or not seeds or len(seeds) != len(set(seeds)):
        errors.append("invalid manifest task or seed list")
    if tuple(manifest.get("evaluation", {}).get("arms", [])) != ARMS:
        errors.append("manifest arm contract mismatch")
    cells: dict[tuple[str, int], Path] = {}
    for spec in cell_specs:
        try:
            arm, task, root = spec.split(":", 2)
            key = (arm, int(task))
        except ValueError:
            errors.append(f"invalid cell spec: {spec}")
            continue
        if arm not in ARMS or key in cells:
            errors.append(f"duplicate/unexpected cell: {spec}")
        cells[key] = Path(root)
    expected_cells = {(arm, task) for arm in ARMS for task in tasks}
    errors += [f"missing cell: {item}" for item in sorted(expected_cells - set(cells))]
    errors += [f"unexpected cell: {item}" for item in sorted(set(cells) - expected_cells)]
    indexed: dict[tuple[str, int, int], dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    for (arm, task), root in sorted(cells.items()):
        attempt_path = root / "attempts.jsonl"
        if not attempt_path.is_file():
            errors.append(f"missing attempts: {attempt_path}")
            continue
        for row in load_rows(attempt_path):
            key = (str(row.get("arm")), int(row.get("goal_idx", -1)), int(row.get("rollout_seed", -1)))
            if key[0] != arm or key[1] != task or key[2] not in seeds or key in indexed:
                errors.append(f"attempt identity mismatch: {key}")
            indexed[key] = row
            all_rows.append(row)
    expected_attempts = {(arm, task, seed) for arm in ARMS for task in tasks for seed in seeds}
    if set(indexed) != expected_attempts:
        errors.append(f"attempt coverage mismatch missing={len(expected_attempts - set(indexed))} extra={len(set(indexed) - expected_attempts)}")
    arm_metrics: dict[str, Any] = {}
    for arm in ARMS:
        rows = [row for row in all_rows if row.get("arm") == arm]
        good = [row for row in rows if valid(row)]
        arm_metrics[arm] = {"attempt_count": len(rows), "scientific_valid_count": len(good), "mean_score_valid": sum(float(row.get("final_task_score") or 0.0) for row in good) / len(good) if good else None, "parser_or_fallback_failures": sum(bool(row.get("parser_or_fallback_failure")) for row in rows), "runtime_errors": sum(row.get("error") is not None for row in rows)}
    comparisons: dict[str, Any] = {}
    rng = random.Random(RESAMPLE_SEED)
    for arm in ARMS[1:]:
        complete: list[dict[str, Any]] = []
        all_attempt_deltas: list[float] = []
        for task in tasks:
            pairs = []
            for seed in seeds:
                base, treatment = indexed.get(("base", task, seed)), indexed.get((arm, task, seed))
                if base is not None and treatment is not None:
                    all_attempt_deltas.append(float(treatment.get("final_task_score") or 0.0) - float(base.get("final_task_score") or 0.0))
                    if valid(base) and valid(treatment):
                        pairs.append(float(treatment.get("final_task_score") or 0.0) - float(base.get("final_task_score") or 0.0))
            if len(pairs) == len(seeds):
                complete.append({"goal_idx": task, "paired_seed_count": len(pairs), "mean_delta": sum(pairs) / len(pairs)})
        deltas = [row["mean_delta"] for row in complete]
        comparisons[arm] = {"complete_valid_task_count": len(complete), "complete_valid_task_deltas": complete, "task_weighted_delta": sum(deltas) / len(deltas) if deltas else None, "task_cluster_bootstrap_ci95": ci(deltas, rng), "task_level_sign_flip_p": sign_flip(deltas, rng), "attempt_weighted_itt_delta": sum(all_attempt_deltas) / len(all_attempt_deltas) if all_attempt_deltas else None, "positive_tie_negative_tasks": {"positive": sum(value > 0 for value in deltas), "tie": sum(value == 0 for value in deltas), "negative": sum(value < 0 for value in deltas)}}
    result = {"schema": "openevo.webshop-h1.38a-pooled-diversity-reconciliation.v1", "status": "invalid" if errors else "complete", "manifest_file_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "expected_attempt_count": len(expected_attempts), "observed_attempt_count": len(all_rows), "arm_metrics": arm_metrics, "comparisons_vs_base": comparisons, "resampling": {"task_cluster_bootstrap_rounds": RESAMPLES, "task_level_sign_flip_rounds": RESAMPLES, "seed": RESAMPLE_SEED}, "structural_errors": errors, "claim_boundary": "Pooled endogenous diversity only; not an upstream-v16 continual-learning result."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cell", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reconcile(args.manifest.resolve(), args.cell, args.output.resolve())
    print(json.dumps({"status": result["status"], "errors": result["structural_errors"]}, sort_keys=True))
    raise SystemExit(2 if result["status"] == "invalid" else 0)


if __name__ == "__main__":
    main()
