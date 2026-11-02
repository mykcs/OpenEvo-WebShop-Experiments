#!/usr/bin/env python3
"""Reconcile H1.17 BASE-15 versus BASE-30 without hiding failures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CONDITIONS = {"base-15": "base", "base-30": "base-30"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid(row: dict[str, Any]) -> bool:
    return bool(row.get("identity_ok") and row.get("trace_complete") and not row.get("parser_or_fallback_failure") and not row.get("error"))


def reconcile(run_root: Path, manifest_path: Path, cell_paths: dict[str, Path]) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seeds = tuple(int(seed) for seed in manifest["rollout"]["seeds"])
    partitions = {str(name): {int(goal) for goal in goals} for name, goals in manifest["task_selection"]["partitions"].items()}
    expected_cells = tuple(f"{condition}-{partition}" for condition in CONDITIONS for partition in partitions)
    missing = [cell for cell in expected_cells if cell not in cell_paths]
    if missing:
        raise ValueError(f"missing cell paths: {missing}")
    indexed: dict[tuple[int, int, str], dict[str, Any]] = {}
    cells: dict[str, Any] = {}
    for cell in expected_cells:
        condition, partition = cell.rsplit("-", 1)
        expected_tasks = partitions[partition]
        path = cell_paths[cell]
        attempts_path = path / "attempts.jsonl"
        if not attempts_path.is_file():
            cells[cell] = {"status": "missing", "path": str(path), "attempt_count": 0, "valid_count": 0, "runtime_failures": []}
            continue
        rows = [json.loads(line) for line in attempts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        failures: list[dict[str, Any]] = []
        seen: set[tuple[int, int]] = set()
        duplicates: list[tuple[int, int]] = []
        for row in rows:
            key = (int(row.get("goal_idx", -1)), int(row.get("rollout_seed", -1)))
            if key in seen:
                duplicates.append(key)
            seen.add(key)
            if row.get("arm") != CONDITIONS[condition] or key[0] not in expected_tasks or key[1] not in seeds:
                failures.append({"kind": "identity", "task": key, "arm": row.get("arm")})
            if not _valid(row):
                failures.append({"kind": "runtime_or_contract", "task": key, "error": row.get("error")})
            if row.get("arm") == CONDITIONS[condition] and key[0] in expected_tasks and key[1] in seeds:
                indexed.setdefault((key[0], key[1], condition), row)
        expected_keys = {(task, seed) for task in expected_tasks for seed in seeds}
        missing_keys = sorted(expected_keys - seen)
        cells[cell] = {
            "status": "pass" if len(rows) == len(expected_keys) and not failures and not missing_keys and not duplicates else "invalid",
            "path": str(path),
            "attempt_count": len(rows),
            "valid_count": sum(_valid(row) for row in rows),
            "qualified_positive_count": sum(bool(row.get("qualified_positive")) for row in rows),
            "mean_task_score": sum(float(row.get("final_task_score") or 0.0) for row in rows) / len(rows) if rows else 0.0,
            "runtime_failures": failures,
            "duplicate_keys": duplicates,
            "missing_keys": missing_keys,
            "attempts_sha256": _sha256(attempts_path),
        }
    paired: list[dict[str, Any]] = []
    excluded = 0
    for partition, tasks in partitions.items():
        for task in tasks:
            for seed in seeds:
                left = indexed.get((task, seed, "base-30")); right = indexed.get((task, seed, "base-15"))
                if left is None or right is None or not _valid(left) or not _valid(right):
                    excluded += 1
                    continue
                paired.append({"partition": partition, "goal_idx": task, "rollout_seed": seed, "delta_base30_minus_base15": float(left["final_task_score"]) - float(right["final_task_score"])})
    complete = all(cells[cell].get("status") == "pass" for cell in expected_cells)
    return {
        "schema": "seed3090.webshop-h1.17-horizon-reconciliation.v1",
        "run_root": str(run_root),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "status": "pass" if complete else "incomplete-or-invalid",
        "expected_attempt_count": sum(len(tasks) for tasks in partitions.values()) * len(seeds) * len(CONDITIONS),
        "observed_attempt_count": sum(int(cells[cell].get("attempt_count", 0)) for cell in expected_cells),
        "scientific_valid_attempt_count": sum(int(cells[cell].get("valid_count", 0)) for cell in expected_cells),
        "scientific_qualified_positive_count": sum(int(cells[cell].get("qualified_positive_count", 0)) for cell in expected_cells),
        "cells": cells,
        "paired": paired,
        "paired_excluded_count": excluded,
        "denominator_rule": "runtime, parser, identity, duplicate, and missing-key failures are separately reported and excluded from valid and paired denominators",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--cell", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {}
    for item in args.cell:
        name, separator, raw = item.partition("=")
        if not separator or not name or not raw:
            raise ValueError(f"invalid --cell value: {item!r}")
        paths[name] = Path(raw)
    payload = reconcile(args.run_root, args.task_manifest, paths)
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.chmod(0o600)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
