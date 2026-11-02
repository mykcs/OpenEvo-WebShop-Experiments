#!/usr/bin/env python3
"""Materialize the fresh, contamination-audited H1.28 evaluation panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

try:
    from scripts.openevo_webshop.parametric_rnd import (
        ENV_SEED,
        MAX_NEW,
        all_goals,
        atomic_json,
        digest_json,
    )
except ModuleNotFoundError:  # Direct script execution from this directory.
    from parametric_rnd import ENV_SEED, MAX_NEW, all_goals, atomic_json, digest_json


SELECTION_SEED = 20260852
TASK_COUNT = 8
MIN_GOAL_INDEX = 500
ROLLOUT_SEEDS = [2026085200, 2026085201, 2026085202, 2026085203]
TASK_KEYS = {
    "goal_idx",
    "goal_index",
    "goal_indices",
    "selected_goal_indices",
    "task_id",
    "task_ids",
    "train_task_ids",
    "evaluation_task_ids",
    "excluded_goal_indices",
    "excluded_unreachable_task_ids",
}
TASK_ID = re.compile(r"^webshop_(\d+)$")


def _task_indices(value: Any) -> set[int]:
    if isinstance(value, bool):
        return set()
    if isinstance(value, int):
        return {value}
    if isinstance(value, str):
        match = TASK_ID.fullmatch(value)
        return {int(match.group(1))} if match else set()
    if isinstance(value, list):
        result: set[int] = set()
        for item in value:
            result.update(_task_indices(item))
        return result
    return set()


def collect_prior_goal_indices(paths: list[Path]) -> tuple[set[int], dict[int, list[str]]]:
    used: set[int] = set()
    sources: dict[int, set[str]] = {}

    def walk(value: Any, source: Path) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in TASK_KEYS:
                    for index in _task_indices(item):
                        used.add(index)
                        sources.setdefault(index, set()).add(str(source))
                walk(item, source)
        elif isinstance(value, list):
            for item in value:
                walk(item, source)

    for root in paths:
        candidates = [root] if root.is_file() else sorted(root.rglob("*.json"))
        for path in candidates:
            if "h1.28" in str(path).lower():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            walk(payload, path)
    return used, {index: sorted(values) for index, values in sorted(sources.items())}


def select_indices(goal_count: int, excluded: set[int], *, selection_seed: int = SELECTION_SEED) -> list[int]:
    pool = [index for index in range(MIN_GOAL_INDEX, goal_count) if index not in excluded]
    random.Random(selection_seed).shuffle(pool)
    selected = sorted(pool[:TASK_COUNT])
    if len(selected) != TASK_COUNT:
        raise ValueError("H1.28 selection did not produce eight fresh tasks")
    return selected


def build_manifest(
    seed_root: Path,
    prior_paths: list[Path],
    *,
    experiment_sha: str,
    selection_seed: int = SELECTION_SEED,
) -> dict[str, Any]:
    goals = all_goals(seed_root)
    excluded, sources = collect_prior_goal_indices(prior_paths)
    selected = select_indices(len(goals), excluded, selection_seed=selection_seed)
    tasks = []
    for sample_index, goal_idx in enumerate(selected):
        description = str(goals[goal_idx].get("instruction_text") or "").strip()
        if not description:
            raise ValueError(f"WebShop goal {goal_idx} has no instruction_text")
        tasks.append({
            "sample_index": sample_index,
            "task_id": f"webshop_{goal_idx:06d}",
            "goal_idx": goal_idx,
            "partition": f"P{sample_index // 2 + 1}",
            "split": "fresh-unused-task-label-fidelity-diagnostic",
            "task_description": description,
            "task_description_sha256": hashlib.sha256(description.encode()).hexdigest(),
        })
    payload: dict[str, Any] = {
        "schema": "openevo.webshop-h1.28-label-fidelity-manifest.v1",
        "run_kind": "phase_h1.28_matched_action_label_fidelity",
        "source": {
            "experiment_git_sha": experiment_sha,
            "official_open_evo_stable": "796e6248a61112a61c880885844d3b0420c0b6ae",
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "model_revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        },
        "benchmark": {
            "name": "webshop",
            "catalog": "indexes_1k",
            "num_products": 1000,
            "environment_seed": ENV_SEED,
            "max_steps": 30,
            "max_new_tokens": MAX_NEW,
            "temperature": 0.0,
            "prompt_contract": "seed3090.webshop-compact-action-prompt.v2",
            "action_contract": "seed3090.webshop-action-parser.v1",
        },
        "task_selection": {
            "selection_seed": selection_seed,
            "goal_count": len(goals),
            "min_goal_index": MIN_GOAL_INDEX,
            "prior_json_roots": [str(path.resolve()) for path in prior_paths],
            "prior_used_goal_count": len(excluded),
            "prior_used_goal_indices_sha256": hashlib.sha256(
                json.dumps(sorted(excluded), separators=(",", ":")).encode()
            ).hexdigest(),
            "selected_goal_indices": selected,
            "selected_prior_source_matches": {str(index): sources.get(index, []) for index in selected},
            "partitions": {f"P{i}": selected[(i - 1) * 2 : i * 2] for i in range(1, 5)},
            "split": "fresh-unused-task-label-fidelity-diagnostic",
        },
        "rollout": {
            "attempts": 96,
            "cells": 24,
            "tasks_per_cell": 1,
            "seeds": ROLLOUT_SEEDS,
            "independent_seed_only": True,
        },
        "arms": ["base", "teacher-action", "oracle-action"],
        "tasks": tasks,
        "stop_rule": "one matched action-label-fidelity diagnostic; no prompt, rank, LR, epoch, reasoning-format, or terminal-state change",
    }
    if any(payload["task_selection"]["selected_prior_source_matches"].values()):
        raise ValueError("selected H1.28 task overlaps prior versioned evidence")
    payload["manifest_sha256"] = digest_json(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--prior-json-root", type=Path, action="append", required=True)
    parser.add_argument("--experiment-sha", required=True)
    parser.add_argument("--selection-seed", type=int, default=SELECTION_SEED)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_manifest(
        args.seed_root.expanduser().resolve(),
        [path.expanduser().resolve() for path in args.prior_json_root],
        experiment_sha=args.experiment_sha,
        selection_seed=args.selection_seed,
    )
    atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps({"manifest": str(args.output.resolve()), "manifest_sha256": payload["manifest_sha256"], "goal_indices": payload["task_selection"]["selected_goal_indices"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
