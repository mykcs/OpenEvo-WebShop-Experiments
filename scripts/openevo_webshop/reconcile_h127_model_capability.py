#!/usr/bin/env python3
"""Reconcile the frozen H1.27 model-capability matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid(row: dict[str, Any]) -> bool:
    return bool(
        row.get("identity_ok")
        and row.get("trace_complete")
        and not row.get("parser_or_fallback_failure")
        and not row.get("error")
    )


def _buy_now(row: dict[str, Any]) -> bool:
    trace = (row.get("episode") or {}).get("trace") or []
    return any("buy now" in str(step.get("action", "")).lower() for step in trace)


def _gpu_wall_seconds(output: Path) -> float | None:
    before_path = output / "gpu-before.json"
    after_path = output / "gpu-after.json"
    if not before_path.is_file() or not after_path.is_file():
        return None
    before = json.loads(before_path.read_text(encoding="utf-8"))["timestamp_utc"]
    after = json.loads(after_path.read_text(encoding="utf-8"))["timestamp_utc"]
    return max(0.0, (datetime.fromisoformat(after.replace("Z", "+00:00")) - datetime.fromisoformat(before.replace("Z", "+00:00"))).total_seconds())


def reconcile(run_root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arms = {item["model_arm"]: item for item in manifest["model_arms"]}
    tasks = {int(item["goal_idx"]): item for item in manifest["tasks"]}
    seeds = tuple(int(seed) for seed in manifest["controls"]["rollout_seeds"])
    cells_root = run_root / "formal" / "cells"
    cells: dict[str, Any] = {}
    indexed: dict[tuple[str, int, int], dict[str, Any]] = {}

    for arm, model in arms.items():
        for task, task_doc in tasks.items():
            name = f"{arm}-{task}"
            output = cells_root / name / "output"
            attempts_path = output / "attempts.jsonl"
            summary_path = output / "summary.json"
            if not attempts_path.is_file():
                cells[name] = {"status": "missing", "attempt_count": 0, "valid_count": 0}
                continue
            rows = [json.loads(line) for line in attempts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            seen: set[int] = set()
            duplicates: list[int] = []
            identity_failures: list[dict[str, Any]] = []
            for row in rows:
                seed = int(row.get("rollout_seed", -1))
                if seed in seen:
                    duplicates.append(seed)
                seen.add(seed)
                if (
                    int(row.get("goal_idx", -1)) != task
                    or row.get("task_id") != task_doc["task_id"]
                    or row.get("arm") != "base-30"
                    or seed not in seeds
                ):
                    identity_failures.append({"seed": seed, "task": row.get("goal_idx"), "arm": row.get("arm")})
                else:
                    indexed.setdefault((arm, task, seed), row)
            missing_seeds = sorted(set(seeds) - seen)
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
            summary_identity_ok = bool(
                summary
                and int(summary.get("goal_idx", -1)) == task
                and summary.get("model") == model["model_path"]
            )
            complete = bool(
                len(rows) == len(seeds)
                and not duplicates
                and not identity_failures
                and not missing_seeds
                and summary_identity_ok
            )
            valid = [row for row in rows if _valid(row)]
            cells[name] = {
                "status": "complete" if complete else "incomplete-or-invalid",
                "attempt_count": len(rows),
                "valid_count": len(valid),
                "runtime_error_count": sum(bool(row.get("error")) for row in rows),
                "parser_or_fallback_invalid_count": sum(bool(row.get("parser_or_fallback_failure")) for row in rows),
                "qualified_positive_count": sum(bool(row.get("qualified_positive")) for row in valid),
                "full_win_count": sum(bool(row.get("won")) for row in valid),
                "buy_now_count": sum(_buy_now(row) for row in valid),
                "mean_task_score_valid": sum(float(row.get("final_task_score") or 0.0) for row in valid) / len(valid) if valid else 0.0,
                "missing_seeds": missing_seeds,
                "duplicate_seeds": duplicates,
                "identity_failures": identity_failures,
                "summary_identity_ok": summary_identity_ok,
                "attempts_sha256": _sha256(attempts_path),
                "wandb_run_id": (summary or {}).get("wandb_run_id"),
                "wandb_url": (summary or {}).get("wandb_url"),
                "gpu_wall_seconds": _gpu_wall_seconds(output),
            }

    arm_metrics: dict[str, Any] = {}
    for arm in arms:
        rows = [row for (row_arm, _, _), row in indexed.items() if row_arm == arm]
        valid = [row for row in rows if _valid(row)]
        task_metrics = {}
        for task in tasks:
            task_rows = [row for (row_arm, row_task, _), row in indexed.items() if row_arm == arm and row_task == task]
            task_valid = [row for row in task_rows if _valid(row)]
            task_metrics[str(task)] = {
                "attempt_count": len(task_rows),
                "valid_count": len(task_valid),
                "qualified_positive_count": sum(bool(row.get("qualified_positive")) for row in task_valid),
                "mean_task_score_valid": sum(float(row.get("final_task_score") or 0.0) for row in task_valid) / len(task_valid) if task_valid else 0.0,
            }
        arm_metrics[arm] = {
            "attempt_count": len(rows),
            "valid_count": len(valid),
            "invalid_count": len(rows) - len(valid),
            "qualified_positive_count": sum(bool(row.get("qualified_positive")) for row in valid),
            "full_win_count": sum(bool(row.get("won")) for row in valid),
            "buy_now_count": sum(_buy_now(row) for row in valid),
            "mean_task_score_valid": sum(float(row.get("final_task_score") or 0.0) for row in valid) / len(valid) if valid else 0.0,
            "task_metrics": task_metrics,
        }

    baseline = "QWEN25-3B"
    comparisons: dict[str, Any] = {}
    for arm in arms:
        if arm == baseline:
            continue
        task_rows = []
        for task in tasks:
            paired = []
            for seed in seeds:
                base_row = indexed.get((baseline, task, seed))
                arm_row = indexed.get((arm, task, seed))
                if base_row is None or arm_row is None or not _valid(base_row) or not _valid(arm_row):
                    continue
                paired.append(float(arm_row.get("final_task_score") or 0.0) - float(base_row.get("final_task_score") or 0.0))
            task_rows.append({
                "goal_idx": task,
                "paired_valid_count": len(paired),
                "mean_score_delta": sum(paired) / len(paired) if paired else None,
                "positive_pair_count": sum(delta > 0 for delta in paired),
                "full_seed_coverage": len(paired) == len(seeds),
            })
        positive_tasks = [row["goal_idx"] for row in task_rows if row["full_seed_coverage"] and (row["mean_score_delta"] or 0.0) > 0.0]
        comparisons[arm] = {"tasks": task_rows, "positive_task_ids": positive_tasks, "positive_task_count": len(positive_tasks)}

    execution_complete = len(cells) == len(arms) * len(tasks) and all(cell["status"] == "complete" for cell in cells.values())
    candidates = [arm for arm, item in comparisons.items() if item["positive_task_count"] >= 2]
    if not execution_complete:
        classification = "incomplete-or-invalid-execution"
    elif candidates:
        classification = "cross-task-capability-signal-needs-independent-confirmation"
    else:
        classification = "scale-only-not-supported-or-task-local"
    measured_gpu_seconds = [cell["gpu_wall_seconds"] for cell in cells.values() if cell.get("gpu_wall_seconds") is not None]
    return {
        "schema": "seed3090.webshop-h1.27-model-capability-reconciliation.v1",
        "status": "complete" if execution_complete else "incomplete-or-invalid",
        "classification": classification,
        "run_root": str(run_root),
        "manifest_internal_sha256": manifest.get("manifest_sha256"),
        "manifest_file_sha256": _sha256(manifest_path),
        "expected_attempt_count": len(arms) * len(tasks) * len(seeds),
        "observed_attempt_count": sum(cell["attempt_count"] for cell in cells.values()),
        "scientific_valid_attempt_count": sum(cell["valid_count"] for cell in cells.values()),
        "measured_gpu_hours": sum(measured_gpu_seconds) / 3600.0,
        "gpu_duration_coverage_cells": len(measured_gpu_seconds),
        "cells": cells,
        "arms": arm_metrics,
        "paired_against_qwen25_3b": comparisons,
        "confirmation_candidates": candidates,
        "denominator_rule": "Runtime, identity, trace, parser/fallback, duplicate, and missing-key failures are reported and excluded from scientific-valid and paired denominators.",
        "claim_boundary": "H1.27 diagnoses base-model capability only. Even a positive signal requires fresh-task/seed confirmation and does not establish Q3.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = reconcile(args.run_root, args.task_manifest)
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.chmod(0o600)
    print(json.dumps({key: payload[key] for key in ("status", "classification", "observed_attempt_count", "scientific_valid_attempt_count")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
