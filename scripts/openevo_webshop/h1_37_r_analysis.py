#!/usr/bin/env python3
"""Reconcile the preregistered H1.37-R same-period bridge.

The bridge has no new held-out tasks and no SELF rerun.  It compares exact
task/seed rows from the original H1.37 BASE, resumed H1.37 SELF-EVOLUTION, and
new BASE-R.  Scientific complete-case estimates require both rows valid; the
separate all-attempt robustness assigns invalid rows score zero without
reclassifying them as valid evidence.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

SEEDS = (2026090200, 2026090201, 2026090202, 2026090203)
BOOTSTRAP_ROUNDS = 10000
PERMUTATION_ROUNDS = 10000
RESAMPLE_SEED = 20260818


def load_rows(root: Path, arm: str, tasks: list[int], label: str) -> dict[tuple[int, int], dict[str, Any]]:
    indexed: dict[tuple[int, int], dict[str, Any]] = {}
    for task in tasks:
        path = root / "evaluation" / f"{arm}-task{task}" / "attempts.jsonl"
        if not path.is_file():
            raise ValueError(f"{label}: missing attempts file: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row.get("goal_idx", -1)), int(row.get("rollout_seed", -1)))
            if key[0] != task or key[1] not in SEEDS or key in indexed or row.get("arm") != arm:
                raise ValueError(f"{label}: invalid or duplicate attempt in {path}: {key}")
            if row.get("task_id") != f"webshop_{task:06d}" or not isinstance(row.get("protocol_fingerprint"), str) or len(row["protocol_fingerprint"]) != 64:
                raise ValueError(f"{label}: malformed identity or protocol fingerprint in {path}: {key}")
            indexed[key] = row
    expected = {(task, seed) for task in tasks for seed in SEEDS}
    if set(indexed) != expected:
        raise ValueError(f"{label}: exact 48x4 coverage failure; missing={sorted(expected - set(indexed))} extra={sorted(set(indexed) - expected)}")
    return indexed


def valid(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("identity_ok") and row.get("trace_complete") and not row.get("parser_or_fallback_failure") and not row.get("error"))


def failure_taxonomy(rows: dict[tuple[int, int], dict[str, Any]], expected: set[tuple[int, int]]) -> dict[str, int]:
    values = list(rows.values())
    return {
        "expected_attempts": len(expected), "observed_attempts": len(values), "missing_attempts": len(expected - set(rows)),
        "scientific_valid": sum(valid(row) for row in values),
        "parser_or_fallback": sum(bool(row.get("parser_or_fallback_failure")) for row in values),
        "runtime_error": sum(bool(row.get("error")) for row in values),
        "identity_failure": sum(not bool(row.get("identity_ok")) for row in values),
        "incomplete_trace": sum(not bool(row.get("trace_complete")) for row in values),
    }


def bootstrap_task_ci(task_deltas: dict[int, list[float]]) -> dict[str, float]:
    task_ids = sorted(task_deltas)
    if not task_ids:
        return {"lower": 0.0, "upper": 0.0, "half_width": 0.0}
    rng = random.Random(RESAMPLE_SEED)
    stats = []
    for _ in range(BOOTSTRAP_ROUNDS):
        sample = [rng.choice(task_ids) for _ in task_ids]
        values = [value for task in sample for value in task_deltas[task]]
        stats.append(sum(values) / len(values))
    stats.sort()
    lower = stats[int(BOOTSTRAP_ROUNDS * 0.025)]
    upper = stats[min(int(BOOTSTRAP_ROUNDS * 0.975), BOOTSTRAP_ROUNDS - 1)]
    return {"lower": lower, "upper": upper, "half_width": (upper - lower) / 2}


def sign_flip(task_deltas: dict[int, list[float]]) -> dict[str, float]:
    task_means = [sum(v) / len(v) for _, v in sorted(task_deltas.items()) if v]
    if not task_means:
        return {"task_weighted_delta": 0.0, "p_value_two_sided": 1.0}
    observed = sum(task_means) / len(task_means)
    rng = random.Random(RESAMPLE_SEED + 1)
    extreme = 0
    for _ in range(PERMUTATION_ROUNDS):
        statistic = sum(value if rng.random() < 0.5 else -value for value in task_means) / len(task_means)
        if abs(statistic) >= abs(observed):
            extreme += 1
    return {"task_weighted_delta": observed, "p_value_two_sided": (extreme + 1) / (PERMUTATION_ROUNDS + 1)}


def comparison(name: str, left: dict[tuple[int, int], dict[str, Any]], right: dict[tuple[int, int], dict[str, Any]], tasks: list[int]) -> dict[str, Any]:
    expected = {(task, seed) for task in tasks for seed in SEEDS}
    valid_pairs: dict[int, list[float]] = {}
    itt_pairs: dict[int, list[float]] = {}
    missing: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for key in sorted(expected):
        a, b = left.get(key), right.get(key)
        if a is None or b is None:
            missing.append({"task": key[0], "seed": key[1], "left_present": a is not None, "right_present": b is not None})
            continue
        # Right minus left. The preregistered ITT-style robustness fixes the full
        # 48 x 4 denominator and deterministically maps *every invalid row* to 0.
        itt_pairs.setdefault(key[0], []).append((float(b.get("final_task_score") or 0.0) if valid(b) else 0.0) - (float(a.get("final_task_score") or 0.0) if valid(a) else 0.0))
        if valid(a) and valid(b):
            valid_pairs.setdefault(key[0], []).append(float(b.get("final_task_score") or 0.0) - float(a.get("final_task_score") or 0.0))
        else:
            invalid.append({"task": key[0], "seed": key[1], "left_valid": valid(a), "right_valid": valid(b)})
    valid_values = [value for values in valid_pairs.values() for value in values]
    itt_values = [value for values in itt_pairs.values() for value in values]
    task_stat = sign_flip(valid_pairs)
    return {
        "comparison": name,
        "direction": "right minus left",
        "expected_pairs": len(expected),
        "complete_valid_pairs": len(valid_values),
        "complete_valid_tasks": len(valid_pairs),
        "missing_pair_count": len(missing),
        "invalid_pair_count": len(invalid),
        "missing_pairs": missing,
        "invalid_pairs": invalid,
        "attempt_weighted_delta_complete_valid": sum(valid_values) / len(valid_values) if valid_values else 0.0,
        "task_weighted_delta_complete_valid": task_stat["task_weighted_delta"],
        "task_cluster_bootstrap_ci95": bootstrap_task_ci(valid_pairs),
        "task_level_sign_flip_p_value_two_sided": task_stat["p_value_two_sided"],
        "all_attempt_itt_style": {"paired_attempts": len(itt_values), "attempt_weighted_delta": sum(itt_values) / len(itt_values) if itt_values else 0.0, "invalid_score_policy": "every invalid row contributes 0; it remains invalid outside the scientific denominator"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--bridge-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    tasks = [int(value) for value in config["planned_matrix"]["tasks"]]
    if len(tasks) != 48 or len(set(tasks)) != 48:
        raise SystemExit("H1.37-R config must contain exactly 48 unique tasks")
    expected = {(task, seed) for task in tasks for seed in SEEDS}
    original_manifest = args.original_root / "manifests" / "h1.37-task-manifest.json"
    bridge_manifest = args.bridge_root / "manifests" / "h1.37-r-period-bridge-manifest.json"
    import hashlib
    expected_original_sha = config["original_evidence"]["manifest_file_sha256"]
    if not original_manifest.is_file() or hashlib.sha256(original_manifest.read_bytes()).hexdigest() != expected_original_sha:
        raise SystemExit("original H1.37 manifest pin mismatch")
    bridge = json.loads(bridge_manifest.read_text(encoding="utf-8")) if bridge_manifest.is_file() else None
    if not isinstance(bridge, dict) or bridge.get("source_h1_37_manifest_sha256") != expected_original_sha or bridge.get("evaluation_tasks") != tasks or bridge.get("evaluation", {}).get("rollout_seeds") != list(SEEDS):
        raise SystemExit("bridge manifest identity mismatch")
    base_original = load_rows(args.original_root, "base", tasks, "base-original")
    self_resumed = load_rows(args.original_root, "self-evolution", tasks, "self-resumed")
    base_r = load_rows(args.bridge_root, "base-r", tasks, "base-r")
    result = {
        "schema": "openevo.webshop-h1.37-r-period-bridge-analysis.v1",
        "campaign_id": config["campaign_id"], "task_count": len(tasks), "seeds": list(SEEDS), "planned_attempts": len(expected),
        "original_h1_37_context": {"all_64_task_complete_valid_pairs": 244, "all_64_task_complete_valid_tasks": 61, "note": "The H1.37 61/64 task result is retained; bridge estimates use only the preregistered 48 original-period tasks."},
        "failure_taxonomy_by_arm": {"base_original": failure_taxonomy(base_original, expected), "self_resumed": failure_taxonomy(self_resumed, expected), "base_r": failure_taxonomy(base_r, expected)},
        "comparisons": [comparison("period_reproducibility_base_r_minus_base_original", base_original, base_r, tasks), comparison("same_period_transfer_self_resumed_minus_base_r", base_r, self_resumed, tasks)],
        "inference": {"unit": "task cluster", "bootstrap_rounds": BOOTSTRAP_ROUNDS, "permutation_rounds": PERMUTATION_ROUNDS, "seed": RESAMPLE_SEED},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
