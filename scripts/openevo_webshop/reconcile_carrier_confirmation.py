#!/usr/bin/env python3
"""Reconcile a balanced BASE/TEXT-MEMORY/PARAMETRIC confirmation panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ARMS = ("base", "text-memory", "parametric")


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
    tasks = manifest["tasks"]
    seeds = tuple(int(seed) for seed in manifest["rollout"]["seeds"])
    partitions = {str(name): {int(goal) for goal in goals} for name, goals in manifest["task_selection"]["partitions"].items()}
    cells_expected = tuple(f"{arm}-{partition}" for arm in ARMS for partition in partitions)
    missing_cells = [cell for cell in cells_expected if cell not in cell_paths]
    if missing_cells:
        raise ValueError(f"missing cell paths: {missing_cells}")
    indexed: dict[tuple[int, int, str], dict[str, Any]] = {}
    cells: dict[str, Any] = {}
    for cell in cells_expected:
        if cell.startswith("text-memory-"):
            arm, partition = "text-memory", cell.removeprefix("text-memory-")
        else:
            arm, partition = cell.rsplit("-", 1)
        expected_tasks = partitions[partition]
        path = cell_paths[cell]
        attempts_path = path / "attempts.jsonl"
        if not attempts_path.is_file():
            cells[cell] = {"status": "missing", "path": str(path), "attempt_count": 0, "valid_count": 0, "runtime_failures": []}
            continue
        rows = [json.loads(line) for line in attempts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        failures: list[dict[str, Any]] = []
        duplicates: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for row in rows:
            key = (int(row.get("goal_idx", -1)), int(row.get("rollout_seed", -1)))
            if key in seen:
                duplicates.append(key)
            seen.add(key)
            if row.get("arm") != arm or key[0] not in expected_tasks or key[1] not in seeds:
                failures.append({"kind": "identity", "task": key, "arm": row.get("arm")})
            if not _valid(row):
                failures.append({"kind": "runtime_or_contract", "task": key, "error": row.get("error")})
            if row.get("arm") == arm and key[0] in expected_tasks and key[1] in seeds and key not in indexed:
                indexed[(key[0], key[1], arm)] = row
        expected_keys = {(task["goal_idx"], seed) for task in tasks if task["partition"] == partition for seed in seeds}
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
    paired: dict[str, list[dict[str, Any]]] = {"text_memory_minus_base": [], "parametric_minus_base": [], "parametric_minus_text_memory": []}
    paired_excluded = 0
    for task in tasks:
        partition = task["partition"]
        goal_idx = int(task["goal_idx"])
        for seed in seeds:
            rows = {arm: indexed.get((goal_idx, seed, arm)) for arm in ARMS}
            if not all(rows.values()) or not all(_valid(row) for row in rows.values()):
                paired_excluded += 1
                continue
            score = {arm: float(rows[arm]["final_task_score"]) for arm in ARMS}
            for name, left, right in (("text_memory_minus_base", "text-memory", "base"), ("parametric_minus_base", "parametric", "base"), ("parametric_minus_text_memory", "parametric", "text-memory")):
                paired[name].append({"partition": partition, "goal_idx": goal_idx, "rollout_seed": seed, "delta": score[left] - score[right]})
    all_complete = all(cells[cell].get("status") == "pass" for cell in cells_expected)
    return {
        "schema": "seed3090.webshop-h1.16-carrier-confirmation-reconciliation.v1",
        "run_root": str(run_root),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "status": "pass" if all_complete else "incomplete-or-invalid",
        "expected_attempt_count": len(tasks) * len(seeds) * len(ARMS),
        "observed_attempt_count": sum(int(cells[cell].get("attempt_count", 0)) for cell in cells_expected),
        "scientific_valid_attempt_count": sum(int(cells[cell].get("valid_count", 0)) for cell in cells_expected),
        "scientific_qualified_positive_count": sum(int(cells[cell].get("qualified_positive_count", 0)) for cell in cells_expected),
        "cells": cells,
        "paired": paired,
        "paired_excluded_count": paired_excluded,
        "denominator_rule": "runtime, parser, identity, duplicate, and missing-key failures are reported separately and never enter valid or paired scientific denominators",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--cell", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths: dict[str, Path] = {}
    for item in args.cell:
        cell, separator, raw_path = item.partition("=")
        if not separator or not cell or not raw_path:
            raise ValueError(f"invalid --cell value: {item!r}")
        paths[cell] = Path(raw_path)
    payload = reconcile(args.run_root, args.task_manifest, paths)
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.chmod(0o600)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
