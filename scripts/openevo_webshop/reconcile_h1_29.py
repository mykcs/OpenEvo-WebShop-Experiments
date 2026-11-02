#!/usr/bin/env python3
"""Reconcile the H1.29 BASE versus SELF-EVOLUTION panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any


ARMS = ("base", "self-evolution")
CELL_RE = re.compile(r"^(base|self-evolution):(\d+):(.+)$")


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


def _valid(row: dict[str, Any]) -> bool:
    return bool(row.get("identity_ok") and row.get("trace_complete") and not row.get("parser_or_fallback_failure") and not row.get("error"))


def _episode_diagnostics(row: dict[str, Any]) -> dict[str, float]:
    episode = row.get("episode") if isinstance(row.get("episode"), dict) else {}
    trace = episode.get("trace") if isinstance(episode.get("trace"), list) else []
    steps = [step for step in trace if isinstance(step, dict)]
    valid_actions = sum(bool(step.get("action_valid")) for step in steps)
    buy_now = sum("buy now" in str(step.get("action") or "").lower() for step in steps)
    next_page = sum("next >" in str(step.get("action") or "").lower() for step in steps)
    count = len(steps)
    return {
        "valid_action_rate": valid_actions / count if count else 0.0,
        "buy_now_transition_rate": 1.0 if buy_now else 0.0,
        "premature_next_page_rate": next_page / count if count else 0.0,
        "wall_time_seconds": float(row.get("attempt_wall_time_seconds") or 0.0),
        "seconds_per_step": float(row.get("attempt_wall_time_seconds") or 0.0) / count if count else 0.0,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def reconcile(
    manifest_path: Path,
    cell_specs: list[str],
    output: Path,
    experiment: str = "H1.29-self-evolution-v1",
    schema: str = "openevo.webshop-h1.29-reconciliation.v1",
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = [int(value) for value in manifest.get("evaluation_tasks", [])]
    seeds = [int(value) for value in manifest.get("evaluation", {}).get("rollout_seeds", [])]
    structural_errors: list[str] = []
    if len(tasks) != 8 or len(set(tasks)) != 8:
        structural_errors.append("H1.29 manifest must contain eight unique evaluation tasks")
    if len(seeds) != 4 or len(set(seeds)) != 4:
        structural_errors.append("H1.29 manifest must contain four unique evaluation seeds")
    if manifest.get("evaluation", {}).get("arms") != list(ARMS):
        structural_errors.append("H1.29 manifest arm contract mismatch")

    cells: dict[tuple[str, int], Path] = {}
    for spec in cell_specs:
        match = CELL_RE.fullmatch(spec)
        if not match:
            structural_errors.append(f"invalid cell spec: {spec}")
            continue
        key = (match.group(1), int(match.group(2)))
        if key in cells:
            structural_errors.append(f"duplicate cell: {key}")
        cells[key] = Path(match.group(3)).expanduser().resolve()
    expected_cells = {(arm, task) for arm in ARMS for task in tasks}
    structural_errors.extend(f"missing cell: {key}" for key in sorted(expected_cells - set(cells)))
    structural_errors.extend(f"unexpected cell: {key}" for key in sorted(set(cells) - expected_cells))

    rows: list[dict[str, Any]] = []
    cell_summaries: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for (arm, task), root in sorted(cells.items()):
        attempts_path = root / "attempts.jsonl"
        if not attempts_path.is_file():
            structural_errors.append(f"missing attempts file: {attempts_path}")
            continue
        cell_rows = _load_jsonl(attempts_path)
        fingerprints: set[str] = set()
        for row in cell_rows:
            key = (str(row.get("arm")), int(row.get("goal_idx", -1)), int(row.get("rollout_seed", -1)))
            if key[0] != arm or key[1] != task or key[2] not in seeds:
                structural_errors.append(f"cell identity mismatch in {attempts_path}: {key}")
            if key in seen:
                structural_errors.append(f"duplicate attempt: {key}")
            seen.add(key)
            if row.get("task_id") != f"webshop_{task:06d}":
                structural_errors.append(f"task_id mismatch in {attempts_path}: {key}")
            fingerprint = row.get("protocol_fingerprint")
            if not isinstance(fingerprint, str) or len(fingerprint) != 64:
                structural_errors.append(f"missing protocol fingerprint: {key}")
            else:
                fingerprints.add(fingerprint)
            rows.append({**row, "cell_path": str(root)})
        if len(fingerprints) != 1:
            structural_errors.append(f"protocol fingerprint unstable for {arm}/{task}: {sorted(fingerprints)}")
        cell_summaries.append({
            "arm": arm,
            "goal_idx": task,
            "path": str(root),
            "attempt_count": len(cell_rows),
            "scientific_valid_count": sum(_valid(row) for row in cell_rows),
            "qualified_positive_count": sum(bool(row.get("qualified_positive")) for row in cell_rows if _valid(row)),
            "runtime_error_count": sum(bool(row.get("error")) for row in cell_rows),
            "parser_failure_count": sum(bool(row.get("parser_or_fallback_failure")) for row in cell_rows),
            "protocol_fingerprints": sorted(fingerprints),
        })

    expected_attempts = {(arm, task, seed) for arm in ARMS for task in tasks for seed in seeds}
    if seen != expected_attempts:
        structural_errors.append(f"attempt coverage mismatch: missing={sorted(expected_attempts - seen)} extra={sorted(seen - expected_attempts)}")
    by_arm = {arm: [row for row in rows if row.get("arm") == arm] for arm in ARMS}
    arm_metrics: dict[str, Any] = {}
    for arm, arm_rows in by_arm.items():
        valid = [row for row in arm_rows if _valid(row)]
        diagnostics = [_episode_diagnostics(row) for row in valid]
        arm_metrics[arm] = {
            "attempt_count": len(arm_rows),
            "scientific_valid_count": len(valid),
            "qualified_positive_count": sum(bool(row.get("qualified_positive")) for row in valid),
            "mean_score_valid": sum(float(row.get("final_task_score") or 0.0) for row in valid) / len(valid) if valid else 0.0,
            "win_rate_valid": sum(bool(row.get("won")) for row in valid) / len(valid) if valid else 0.0,
            "median_wall_time_seconds_valid": statistics.median(item["wall_time_seconds"] for item in diagnostics) if diagnostics else 0.0,
            "median_seconds_per_step_valid": statistics.median(item["seconds_per_step"] for item in diagnostics) if diagnostics else 0.0,
            "mean_valid_action_rate": sum(item["valid_action_rate"] for item in diagnostics) / len(diagnostics) if diagnostics else 0.0,
            "mean_buy_now_transition_rate": sum(item["buy_now_transition_rate"] for item in diagnostics) / len(diagnostics) if diagnostics else 0.0,
            "mean_premature_next_page_rate": sum(item["premature_next_page_rate"] for item in diagnostics) / len(diagnostics) if diagnostics else 0.0,
        }

    indexed = {(str(row.get("arm")), int(row.get("goal_idx", -1)), int(row.get("rollout_seed", -1))): row for row in rows}
    paired_rows: list[dict[str, Any]] = []
    complete_task_metrics: list[dict[str, Any]] = []
    for task in tasks:
        task_pairs: list[dict[str, Any]] = []
        for seed in seeds:
            pair_rows = [indexed.get((arm, task, seed)) for arm in ARMS]
            if all(row is not None and _valid(row) for row in pair_rows):
                scores = {arm: float(indexed[(arm, task, seed)].get("final_task_score") or 0.0) for arm in ARMS}
                paired_by_arm = {arm: indexed[(arm, task, seed)] for arm in ARMS}
                times = {arm: _episode_diagnostics(paired_by_arm[arm])["wall_time_seconds"] for arm in ARMS}
                pair = {"goal_idx": task, "rollout_seed": seed, **{f"{arm}_score": score for arm, score in scores.items()}, "self_evolution_minus_base": scores["self-evolution"] - scores["base"], "base_wall_time_seconds": times["base"], "self_evolution_wall_time_seconds": times["self-evolution"], "self_evolution_minus_base_wall_time_seconds": times["self-evolution"] - times["base"]}
                paired_rows.append(pair)
                task_pairs.append(pair)
        if len(task_pairs) == len(seeds):
            means = {arm: sum(row[f"{arm}_score"] for row in task_pairs) / len(task_pairs) for arm in ARMS}
            complete_task_metrics.append({
                "goal_idx": task,
                "paired_seed_count": len(task_pairs),
                "mean_scores": means,
                "self_evolution_exceeds_base": means["self-evolution"] > means["base"],
            })

    positive_tasks = [row["goal_idx"] for row in complete_task_metrics if row["self_evolution_exceeds_base"]]
    paired_delta = sum(row["self_evolution_minus_base"] for row in paired_rows)
    positive_candidate = bool(not structural_errors and len(positive_tasks) >= 2 and paired_delta > 0)
    runtime_errors = sum(bool(row.get("error")) for row in rows)
    parser_failures = sum(bool(row.get("parser_or_fallback_failure")) for row in rows)
    if structural_errors:
        classification = "structural-invalid"
    elif positive_candidate:
        classification = "self-evolution-positive-candidate"
    elif len(complete_task_metrics) < 2:
        classification = "insufficient-complete-valid-pairing"
    else:
        classification = "self-evolution-not-supported"
    result = {
        "schema": schema,
        "experiment": experiment,
        "status": "invalid" if structural_errors else "complete",
        "classification": classification,
        "manifest_file_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "attempt_count": len(rows),
        "scientific_valid_count": sum(_valid(row) for row in rows),
        "runtime_error_count": runtime_errors,
        "parser_failure_count": parser_failures,
        "arm_metrics": arm_metrics,
        "complete_valid_paired_attempt_count": len(paired_rows),
        "complete_valid_task_count": len(complete_task_metrics),
        "complete_task_metrics": complete_task_metrics,
        "positive_task_ids": positive_tasks,
        "paired_delta_sums": {"self_evolution_minus_base": paired_delta},
        "paired_rows": paired_rows,
        "cell_summaries": cell_summaries,
        "structural_errors": structural_errors,
    }
    _atomic_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cell", action="append", required=True, help="ARM:GOAL:/absolute/cell/root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", default="H1.29-self-evolution-v1")
    parser.add_argument("--schema", default="openevo.webshop-h1.29-reconciliation.v1")
    args = parser.parse_args()
    result = reconcile(args.manifest.resolve(), args.cell, args.output.resolve(), experiment=args.experiment, schema=args.schema)
    print(json.dumps(result, sort_keys=True))
    return 2 if result["status"] == "invalid" else 0


if __name__ == "__main__":
    raise SystemExit(main())
