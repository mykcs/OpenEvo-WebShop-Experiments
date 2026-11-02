#!/usr/bin/env python3
"""Materialize the H1.29 self-evolution train/evaluation split."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

try:
    from scripts.openevo_webshop.parametric_rnd import ENV_SEED, all_goals, atomic_json, digest_json
except ModuleNotFoundError:
    from parametric_rnd import ENV_SEED, all_goals, atomic_json, digest_json


SELECTION_SEED = 20260901
KNOWN_TRAIN_TASKS = [2822, 5135, 5294, 6769]
ADDITIONAL_TRAIN_COUNT = 4
EVALUATION_COUNT = 8
MIN_GOAL_INDEX = 500
HARVEST_SEEDS = [2026090100, 2026090101, 2026090102, 2026090103]
EVALUATION_SEEDS = [2026090200, 2026090201, 2026090202, 2026090203]
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
}


def _indices(value: Any) -> set[int]:
    if isinstance(value, bool):
        return set()
    if isinstance(value, int):
        return {value}
    if isinstance(value, str) and value.startswith("webshop_"):
        try:
            return {int(value.split("_", 1)[1])}
        except ValueError:
            return set()
    if isinstance(value, list):
        result: set[int] = set()
        for item in value:
            result.update(_indices(item))
        return result
    return set()


def collect_prior_indices(paths: list[Path]) -> set[int]:
    used: set[int] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in TASK_KEYS:
                    used.update(_indices(item))
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for root in paths:
        candidates = [root] if root.is_file() else sorted(root.rglob("*.json"))
        for path in candidates:
            if "h1.29" in str(path).lower():
                continue
            try:
                walk(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
    return used


def _task_row(goal: dict[str, Any], goal_idx: int, sample_index: int, split: str, partition: str) -> dict[str, Any]:
    description = str(goal.get("instruction_text") or "").strip()
    if not description:
        raise ValueError(f"WebShop goal {goal_idx} has no instruction_text")
    return {
        "sample_index": sample_index,
        "task_id": f"webshop_{goal_idx:06d}",
        "goal_idx": goal_idx,
        "partition": partition,
        "split": split,
        "task_description": description,
        "task_description_sha256": hashlib.sha256(description.encode()).hexdigest(),
    }


def build_manifest(seed_root: Path, prior_paths: list[Path], *, experiment_sha: str, selection_seed: int) -> dict[str, Any]:
    goals = all_goals(seed_root)
    prior = collect_prior_indices(prior_paths)
    if len(set(KNOWN_TRAIN_TASKS)) != len(KNOWN_TRAIN_TASKS):
        raise ValueError("known train tasks are duplicated")
    pool = [
        idx for idx in range(MIN_GOAL_INDEX, len(goals))
        if idx not in prior and idx not in KNOWN_TRAIN_TASKS
    ]
    random.Random(selection_seed).shuffle(pool)
    additional_train = sorted(pool[:ADDITIONAL_TRAIN_COUNT])
    evaluation = sorted(pool[ADDITIONAL_TRAIN_COUNT:ADDITIONAL_TRAIN_COUNT + EVALUATION_COUNT])
    if len(additional_train) != ADDITIONAL_TRAIN_COUNT or len(evaluation) != EVALUATION_COUNT:
        raise ValueError("H1.29 selection did not produce the requested fresh tasks")
    train_indices = KNOWN_TRAIN_TASKS + additional_train
    tasks = []
    for sample_index, goal_idx in enumerate(train_indices):
        tasks.append(_task_row(goals[goal_idx], goal_idx, sample_index, "train-only-self-evolution", f"T{sample_index + 1}"))
    for offset, goal_idx in enumerate(evaluation):
        tasks.append(_task_row(goals[goal_idx], goal_idx, len(tasks), "fresh-unused-self-evolution-evaluation", f"P{offset // 2 + 1}"))
    payload: dict[str, Any] = {
        "schema": "openevo.webshop-h1.29-self-evolution-manifest.v1",
        "run_kind": "phase_h1.29_self_evolution",
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
            "max_new_tokens": 512,
            "temperature": 0.0,
            "prompt_contract": "seed3090.webshop-compact-action-prompt.v2",
            "action_contract": "seed3090.webshop-action-parser.v1",
        },
        "train_tasks": train_indices,
        "evaluation_tasks": evaluation,
        "tasks": tasks,
        "task_selection": {
            "selection_seed": selection_seed,
            "goal_count": len(goals),
            "min_goal_index": MIN_GOAL_INDEX,
            "prior_used_goal_count": len(prior),
            "prior_used_goal_indices_sha256": hashlib.sha256(json.dumps(sorted(prior), separators=(",", ":")).encode()).hexdigest(),
            "known_train_tasks": KNOWN_TRAIN_TASKS,
            "additional_train_tasks": additional_train,
            "evaluation_tasks": evaluation,
            "evaluation_is_disjoint_from_train": not set(train_indices) & set(evaluation),
            "split": "train-only-self-evolution versus fresh-unused-self-evolution-evaluation",
        },
        "harvest": {
            "tasks": train_indices,
            "rollout_seeds": HARVEST_SEEDS,
            "attempts": len(train_indices) * len(HARVEST_SEEDS),
            "qualification": "identity_ok and trace_complete and no parser/fallback/error and final_task_score > 0",
            "minimum_success_task_count": 2,
            "minimum_success_episode_count": 8,
        },
        "evaluation": {
            "tasks": evaluation,
            "rollout_seeds": EVALUATION_SEEDS,
            "arms": ["base", "self-evolution"],
            "cells": len(evaluation) * 2,
            "attempts": len(evaluation) * len(EVALUATION_SEEDS) * 2,
            "task_role": "fresh held-out evaluation only",
        },
        "contamination": {
            "prior_versioned_goal_count": len(prior),
            "train_evaluation_overlap": False,
            "evaluation_outcomes_used_for_training": False,
            "teacher_oracle_gpt_used": False,
        },
        "manifest_sha256": "",
    }
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
    output = args.output.expanduser().resolve()
    atomic_json(output, payload)
    print(json.dumps({"manifest": str(output), "manifest_sha256": payload["manifest_sha256"], "manifest_file_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "train_tasks": payload["train_tasks"], "evaluation_tasks": payload["evaluation_tasks"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
