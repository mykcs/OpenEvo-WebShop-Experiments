#!/usr/bin/env python3
"""Fail-closed H1.39-MR reconciliation with explicit invalidity taxonomy."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ARMS = ("base", "ordinary-lora", "sd-lora")
CELL_RE = re.compile(r"^(base|ordinary-lora|sd-lora):(\d+):(.+)$")
INVALID_KEYS = ("parser_or_fallback", "runtime", "identity", "incomplete_trace", "adapter_load_unverified", "base_fallback")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: attempt is not an object")
        rows.append(value)
    return rows


def _adapter_issue(row: dict[str, Any], arm: str, expected: dict[str, Any]) -> tuple[bool, bool]:
    receipt = row.get("adapter_load_receipt")
    if not isinstance(receipt, dict):
        return True, arm != "base"
    expected_loaded = arm != "base"
    loaded = bool(receipt.get("adapter_loaded"))
    base_fallback = expected_loaded and not loaded
    mismatch = (
        receipt.get("verified") is not True
        or loaded != expected_loaded
        or receipt.get("adapter_file_sha256") != expected.get("adapter_file_sha256")
        or receipt.get("loaded_tensor_digest") != expected.get("adapter_tensor_digest")
    )
    if not expected_loaded:
        mismatch = receipt.get("verified") is not True or loaded or receipt.get("adapter_file_sha256") is not None or receipt.get("loaded_tensor_digest") is not None
    return mismatch, base_fallback


def _invalid_reasons(row: dict[str, Any], arm: str, task: int, seed: int, expected: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if row.get("task_id") != f"webshop_{task:06d}" or int(row.get("goal_idx", -1)) != task or int(row.get("rollout_seed", -1)) != seed:
        reasons.append("identity")
    if not row.get("trace_complete"):
        reasons.append("incomplete_trace")
    if row.get("parser_or_fallback_failure"):
        reasons.append("parser_or_fallback")
    if row.get("error"):
        reasons.append("runtime")
    adapter_issue, base_fallback = _adapter_issue(row, arm, expected)
    if adapter_issue:
        reasons.append("adapter_load_unverified")
    if base_fallback:
        reasons.append("base_fallback")
    return sorted(set(reasons))


def reconcile(manifest_path: Path, cell_specs: list[str], output: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = [int(value) for value in manifest.get("evaluation_tasks", [])]
    seeds = [int(value) for value in manifest.get("evaluation", {}).get("rollout_seeds", [])]
    expected_identities = manifest.get("arm_identities") if isinstance(manifest.get("arm_identities"), dict) else {}
    structural_errors: list[str] = []
    if manifest.get("schema") != "openevo.webshop-h1.39-mr-independent-panel.v1":
        structural_errors.append("manifest schema mismatch")
    if len(tasks) != 64 or len(set(tasks)) != 64:
        structural_errors.append("manifest must contain exactly 64 unique tasks")
    if len(seeds) != 4 or len(set(seeds)) != 4:
        structural_errors.append("manifest must contain exactly 4 unique seeds")
    if manifest.get("evaluation", {}).get("arms") != list(ARMS):
        structural_errors.append("manifest arm contract mismatch")
    if set(expected_identities) != set(ARMS):
        structural_errors.append("manifest arm identities are incomplete")

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
    invalid_counts = {arm: {key: 0 for key in INVALID_KEYS} for arm in ARMS}
    seen: set[tuple[str, int, int]] = set()
    cell_summaries: list[dict[str, Any]] = []
    for (arm, task), root in sorted(cells.items()):
        attempts_path = root / "attempts.jsonl"
        if not attempts_path.is_file():
            structural_errors.append(f"missing attempts file: {attempts_path}")
            continue
        try:
            cell_rows = _load_jsonl(attempts_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            structural_errors.append(f"unreadable attempts file {attempts_path}: {type(exc).__name__}")
            continue
        fingerprints: set[str] = set()
        for row in cell_rows:
            key = (str(row.get("arm")), int(row.get("goal_idx", -1)), int(row.get("rollout_seed", -1)))
            if key in seen:
                structural_errors.append(f"duplicate attempt: {key}")
            seen.add(key)
            fingerprint = row.get("protocol_fingerprint")
            if not isinstance(fingerprint, str) or len(fingerprint) != 64:
                structural_errors.append(f"missing protocol fingerprint: {key}")
            else:
                fingerprints.add(fingerprint)
            reasons = _invalid_reasons(row, arm, task, key[2], expected_identities.get(arm, {}))
            for reason in reasons:
                invalid_counts[arm][reason] += 1
            rows.append({**row, "cell_arm": arm, "cell_task": task, "invalid_reasons": reasons, "cell_path": str(root)})
        if len(cell_rows) != len(seeds):
            structural_errors.append(f"attempt count mismatch for {arm}/{task}: {len(cell_rows)}")
        if len(fingerprints) != 1:
            structural_errors.append(f"protocol fingerprint unstable for {arm}/{task}: {sorted(fingerprints)}")
        cell_summaries.append({"arm": arm, "goal_idx": task, "attempt_count": len(cell_rows), "protocol_fingerprints": sorted(fingerprints)})

    expected_attempts = {(arm, task, seed) for arm in ARMS for task in tasks for seed in seeds}
    if seen != expected_attempts:
        structural_errors.append(f"attempt coverage mismatch: missing={sorted(expected_attempts - seen)} extra={sorted(seen - expected_attempts)}")

    def valid(row: dict[str, Any]) -> bool:
        return not row["invalid_reasons"]

    by_arm = {arm: [row for row in rows if row["cell_arm"] == arm] for arm in ARMS}
    arm_metrics: dict[str, Any] = {}
    for arm in ARMS:
        arm_rows = by_arm[arm]
        valid_rows = [row for row in arm_rows if valid(row)]
        arm_metrics[arm] = {
            "attempt_count": len(arm_rows),
            "valid_attempts": len(valid_rows),
            "invalid_attempts": len(arm_rows) - len(valid_rows),
            "valid_mean": sum(float(row.get("final_task_score") or 0.0) for row in valid_rows) / len(valid_rows) if valid_rows else 0.0,
            "itt_mean": sum(float(row.get("final_task_score") or 0.0) if valid(row) else 0.0 for row in arm_rows) / len(arm_rows) if arm_rows else 0.0,
            "invalid_counts": invalid_counts[arm],
        }

    indexed = {(row["cell_arm"], int(row.get("goal_idx", -1)), int(row.get("rollout_seed", -1))): row for row in rows}
    complete_tasks: list[dict[str, Any]] = []
    for task in tasks:
        task_rows = [indexed.get((arm, task, seed)) for seed in seeds for arm in ARMS]
        if len(task_rows) == len(seeds) * len(ARMS) and all(row is not None and valid(row) for row in task_rows):
            means = {arm: sum(float(indexed[(arm, task, seed)].get("final_task_score") or 0.0) for seed in seeds) / len(seeds) for arm in ARMS}
            complete_tasks.append({"goal_idx": task, "means": means})

    result = {
        "schema": "openevo.webshop-h1.39-mr-reconciliation.v1",
        "campaign_id": manifest.get("campaign_id"),
        "status": "invalid" if structural_errors else "complete",
        "classification": "R0 STRUCTURALLY_INVALID" if structural_errors else "pending-analysis",
        "manifest_file_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "attempts": {"expected": len(expected_attempts), "observed": len(rows), "missing": len(expected_attempts - seen), "valid": sum(valid(row) for row in rows)},
        "arm_metrics": arm_metrics,
        "invalid_taxonomy": INVALID_KEYS,
        "complete_valid_task_count": len(complete_tasks),
        "complete_valid_tasks": complete_tasks,
        "cell_summaries": cell_summaries,
        "structural_errors": structural_errors,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cell", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reconcile(args.manifest.resolve(), args.cell, args.output.resolve())
    print(json.dumps({"status": result["status"], "attempts": result["attempts"], "complete_valid_task_count": result["complete_valid_task_count"], "structural_errors": result["structural_errors"]}, sort_keys=True))
    return 2 if result["status"] == "invalid" else 0


if __name__ == "__main__":
    raise SystemExit(main())
