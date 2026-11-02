#!/usr/bin/env python3
"""Reconcile the frozen H1.18 three-arm, four-partition evaluation panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ARMS = ("base", "success-state", "student-recovery")
CELL_RE = re.compile(r"^(base|success-state|student-recovery):(P[1-4]):(.+)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        rows.append(row)
    return rows


def _parse_cell(value: str) -> tuple[str, str, Path]:
    match = CELL_RE.match(value)
    if not match:
        raise ValueError(f"invalid --cell {value!r}; expected ARM:P1:PATH")
    return match.group(1), match.group(2), Path(match.group(3)).expanduser().resolve()


def _selected_tasks(manifest: dict[str, Any]) -> dict[str, list[int]]:
    partitions = manifest.get("task_selection", {}).get("partitions")
    if not isinstance(partitions, dict):
        raise ValueError("manifest.task_selection.partitions is missing")
    result: dict[str, list[int]] = {}
    for partition in ("P1", "P2", "P3", "P4"):
        tasks = partitions.get(partition)
        if not isinstance(tasks, list) or len(tasks) != 2 or any(isinstance(task, bool) or not isinstance(task, int) for task in tasks):
            raise ValueError(f"partition {partition} must contain exactly two integer goal indices")
        result[partition] = [int(task) for task in tasks]
    return result


def reconcile(
    manifest_path: Path,
    cell_specs: list[str],
    output: Path,
    *,
    train_task_ids: set[str],
    expected_git_sha: str,
    expected_upstream_sha: str,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    partitions = _selected_tasks(manifest)
    seeds = manifest.get("rollout", {}).get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 4 or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("manifest.rollout.seeds must contain four integer seeds")
    selected_tasks = {task for tasks in partitions.values() for task in tasks}
    selected_task_ids = {f"webshop_{task:06d}" for task in selected_tasks}
    overlap = selected_task_ids & train_task_ids
    if overlap:
        raise ValueError(f"training/evaluation task overlap: {sorted(overlap)}")
    if expected_upstream_sha != "796e6248a61112a61c880885844d3b0420c0b6ae":
        raise ValueError("H1.18 is frozen to the pinned upstream OpenEvo stable SHA")

    cells: dict[tuple[str, str], Path] = {}
    structural_errors: list[str] = []
    for spec in cell_specs:
        try:
            arm, partition, path = _parse_cell(spec)
        except ValueError as exc:
            structural_errors.append(str(exc))
            continue
        key = (arm, partition)
        if key in cells:
            structural_errors.append(f"duplicate cell {arm}/{partition}")
        cells[key] = path
    expected_keys = {(arm, partition) for arm in ARMS for partition in partitions}
    missing = sorted(expected_keys - set(cells))
    unexpected = sorted(set(cells) - expected_keys)
    structural_errors.extend(f"missing cell {arm}/{partition}" for arm, partition in missing)
    structural_errors.extend(f"unexpected cell {arm}/{partition}" for arm, partition in unexpected)

    rows: list[dict[str, Any]] = []
    cell_summaries: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int, str]] = set()
    runtime_error_count = 0
    parser_failure_count = 0
    # The tracked evaluation protocol includes the UUID-bound GPU namespace.
    # Therefore fingerprints are expected to be stable within a cell, but
    # may legitimately differ across cells that run on different GPUs.
    protocol_fingerprints: dict[str, set[str]] = {arm: set() for arm in ARMS}
    protocol_fingerprints_by_cell: dict[str, set[str]] = {}
    for (arm, partition), cell_path in sorted(cells.items()):
        attempts_path = cell_path / "attempts.jsonl"
        if not attempts_path.is_file():
            structural_errors.append(f"missing attempts file: {attempts_path}")
            continue
        cell_rows = _load_jsonl(attempts_path)
        cell_key = f"{arm}:{partition}"
        cell_fingerprints = protocol_fingerprints_by_cell.setdefault(cell_key, set())
        expected_goals = set(partitions.get(partition, []))
        for row in cell_rows:
            goal = int(row.get("goal_idx", -1))
            seed = int(row.get("rollout_seed", -1))
            row_arm = str(row.get("arm"))
            key = (row_arm, goal, seed)
            if row_arm != arm:
                structural_errors.append(f"arm mismatch in {attempts_path}: {row_arm!r} != {arm!r}")
            if goal not in expected_goals or seed not in seeds:
                structural_errors.append(f"unexpected task/seed in {attempts_path}: goal={goal} seed={seed}")
            if key in seen_keys:
                structural_errors.append(f"duplicate task/seed/arm: {key}")
            seen_keys.add(key)
            if row.get("task_id") != f"webshop_{goal:06d}":
                structural_errors.append(f"task identity mismatch in {attempts_path}: goal={goal}")
            fingerprint = row.get("protocol_fingerprint")
            if not isinstance(fingerprint, str) or len(fingerprint) != 64:
                structural_errors.append(f"missing protocol fingerprint in {attempts_path}: {key}")
            elif row_arm in protocol_fingerprints:
                protocol_fingerprints[row_arm].add(fingerprint)
                cell_fingerprints.add(fingerprint)
            if row.get("error"):
                runtime_error_count += 1
            if row.get("parser_or_fallback_failure"):
                parser_failure_count += 1
            rows.append({**row, "cell": f"{arm}:{partition}", "cell_path": str(cell_path)})
        valid = [row for row in cell_rows if row.get("identity_ok") and row.get("trace_complete") and not row.get("parser_or_fallback_failure") and not row.get("error")]
        cell_summaries.append({
            "arm": arm,
            "partition": partition,
            "path": str(cell_path),
            "attempt_count": len(cell_rows),
            "valid_count": len(valid),
            "qualified_positive_count": sum(bool(row.get("qualified_positive")) for row in valid),
            "runtime_error_count": sum(bool(row.get("error")) for row in cell_rows),
            "parser_failure_count": sum(bool(row.get("parser_or_fallback_failure")) for row in cell_rows),
            "goal_indices": sorted({int(row.get("goal_idx", -1)) for row in cell_rows}),
            "rollout_seeds": sorted({int(row.get("rollout_seed", -1)) for row in cell_rows}),
            "protocol_fingerprints": sorted(cell_fingerprints),
        })

    expected_attempts = len(expected_keys) * len(seeds) * 2
    if len(rows) != expected_attempts:
        structural_errors.append(f"expected {expected_attempts} attempts, found {len(rows)}")
    expected_row_keys = {(arm, goal, seed) for arm in ARMS for goal in selected_tasks for seed in seeds}
    if seen_keys != expected_row_keys:
        structural_errors.append(f"complete task/seed coverage mismatch: missing={sorted(expected_row_keys - seen_keys)} extra={sorted(seen_keys - expected_row_keys)}")
    for cell_key, fingerprints in protocol_fingerprints_by_cell.items():
        if len(fingerprints) != 1:
            structural_errors.append(f"protocol fingerprint is not stable for {cell_key}: {sorted(fingerprints)}")

    by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for row in rows:
        if row.get("arm") in by_arm:
            by_arm[row["arm"]].append(row)
    arm_metrics = {}
    for arm, arm_rows in by_arm.items():
        valid = [row for row in arm_rows if row.get("identity_ok") and row.get("trace_complete") and not row.get("parser_or_fallback_failure") and not row.get("error")]
        arm_metrics[arm] = {
            "attempt_count": len(arm_rows),
            "valid_count": len(valid),
            "qualified_positive_count": sum(bool(row.get("qualified_positive")) for row in valid),
            "mean_task_score_all_attempts": sum(float(row.get("final_task_score", 0.0) or 0.0) for row in arm_rows) / max(1, len(arm_rows)),
            "mean_task_score_valid": sum(float(row.get("final_task_score", 0.0) or 0.0) for row in valid) / max(1, len(valid)),
        }

    indexed = {(row.get("arm"), int(row.get("goal_idx", -1)), int(row.get("rollout_seed", -1))): row for row in rows}
    paired: list[dict[str, Any]] = []
    for goal in sorted(selected_tasks):
        partition = next(name for name, tasks in partitions.items() if goal in tasks)
        for seed in seeds:
            base = indexed.get(("base", goal, seed))
            success = indexed.get(("success-state", goal, seed))
            recovery = indexed.get(("student-recovery", goal, seed))
            if not base or not success or not recovery:
                continue
            paired.append({
                "partition": partition,
                "goal_idx": goal,
                "rollout_seed": seed,
                "base_score": float(base.get("final_task_score", 0.0) or 0.0),
                "success_state_score": float(success.get("final_task_score", 0.0) or 0.0),
                "student_recovery_score": float(recovery.get("final_task_score", 0.0) or 0.0),
                "success_state_minus_base": float(success.get("final_task_score", 0.0) or 0.0) - float(base.get("final_task_score", 0.0) or 0.0),
                "student_recovery_minus_base": float(recovery.get("final_task_score", 0.0) or 0.0) - float(base.get("final_task_score", 0.0) or 0.0),
                "student_recovery_minus_success_state": float(recovery.get("final_task_score", 0.0) or 0.0) - float(success.get("final_task_score", 0.0) or 0.0),
            })
    delta_sums = {
        name: sum(float(row[name]) for row in paired)
        for name in ("success_state_minus_base", "student_recovery_minus_base", "student_recovery_minus_success_state")
    }
    result = {
        "schema": "seed3090.webshop-h1.18-reconciliation.v1",
        "status": "invalid" if structural_errors else ("complete-with-invalid-attempts" if runtime_error_count or parser_failure_count else "complete"),
        "classification": "structural-invalid" if structural_errors else ("runtime-or-parser-invalid-attempts" if runtime_error_count or parser_failure_count else "scientifically-valid-complete-panel"),
        "experiment": "H1.18-reachability-filtered-recovery-diagnostic-v1",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_file_sha256": _sha256(manifest_path),
        "experiment_git_sha": expected_git_sha,
        "upstream_openevo_sha": expected_upstream_sha,
        "train_task_ids": sorted(train_task_ids),
        "evaluation_task_ids": sorted(selected_task_ids),
        "evaluation_seeds": [int(seed) for seed in seeds],
        "attempt_count": len(rows),
        "valid_count": sum(row.get("identity_ok") and row.get("trace_complete") and not row.get("parser_or_fallback_failure") and not row.get("error") for row in rows),
        "runtime_error_count": runtime_error_count,
        "parser_failure_count": parser_failure_count,
        "qualified_positive_count": sum(bool(row.get("qualified_positive")) for row in rows),
        "arm_metrics": arm_metrics,
        "paired_count": len(paired),
        "paired_delta_sums": delta_sums,
        "paired_rows": paired,
        "protocol_fingerprints_by_arm": {arm: sorted(values) for arm, values in protocol_fingerprints.items()},
        "protocol_fingerprints_by_cell": {cell: sorted(values) for cell, values in sorted(protocol_fingerprints_by_cell.items())},
        "cell_summaries": cell_summaries,
        "structural_errors": structural_errors,
        "source_cell_specs": cell_specs,
    }
    _atomic_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cell", action="append", required=True, help="ARM:PARTITION:/absolute/cell/root")
    parser.add_argument("--train-task-id", action="append", required=True)
    parser.add_argument("--experiment-git-sha", required=True)
    parser.add_argument("--upstream-openevo-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reconcile(args.manifest, args.cell, args.output, train_task_ids=set(args.train_task_id), expected_git_sha=args.experiment_git_sha, expected_upstream_sha=args.upstream_openevo_sha)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] != "invalid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
