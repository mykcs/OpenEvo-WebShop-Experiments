#!/usr/bin/env python3
"""Materialize the H1.33 H0.1-positive replay and fresh-task manifest."""

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
    from parametric_rnd import ENV_SEED, all_goals, atomic_json, digest_json  # type: ignore


SCHEMA = "openevo.webshop-h1.33-positive-replay-manifest.v1"
SELECTION_SEED = 2026081801
MIN_GOAL_INDEX = 500
EVALUATION_COUNT = 8
EVALUATION_SEEDS = [2026081800, 2026081801, 2026081802, 2026081803]
PROMPT_CONTRACT = "seed3090.webshop-compact-action-prompt.v2"
ACTION_CONTRACT = "seed3090.webshop-action-parser.v1"


def _task_row(goal: dict[str, Any], goal_idx: int, sample_index: int, split: str, partition: str) -> dict[str, Any]:
    description = str(goal.get("instruction_text") or "").strip()
    if not description:
        raise ValueError(f"goal {goal_idx} has no instruction_text")
    return {
        "sample_index": sample_index,
        "task_id": f"webshop_{goal_idx:06d}",
        "goal_idx": goal_idx,
        "partition": partition,
        "split": split,
        "task_description": description,
        "task_description_sha256": hashlib.sha256(description.encode()).hexdigest(),
    }


def build_manifest(
    seed_root: Path,
    train_tasks: list[int],
    evaluation_tasks: list[int] | None,
    *,
    prior_json_roots: list[Path],
    experiment_sha: str,
    h0_reconciliation_sha256: str,
    selection_seed: int = SELECTION_SEED,
) -> dict[str, Any]:
    goals = all_goals(seed_root)
    train_tasks = list(dict.fromkeys(train_tasks))
    if len(train_tasks) < 8 or any(task < MIN_GOAL_INDEX for task in train_tasks):
        raise ValueError("H1.33 requires at least eight valid H0.1 source task identities")
    prior_indices: set[int] = set(train_tasks)
    for root in prior_json_roots:
        candidates = [root] if root.is_file() else sorted(root.rglob("*.json"))
        for path in candidates:
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            values = document.get("evaluation_tasks", []) if isinstance(document, dict) else []
            if isinstance(values, list):
                prior_indices.update(int(value) for value in values if isinstance(value, int) and not isinstance(value, bool))
    if evaluation_tasks is None:
        pool = [idx for idx in range(MIN_GOAL_INDEX, len(goals)) if idx not in prior_indices]
        random.Random(selection_seed).shuffle(pool)
        evaluation_tasks = pool[:EVALUATION_COUNT]
    evaluation_tasks = list(dict.fromkeys(evaluation_tasks))
    if len(evaluation_tasks) != EVALUATION_COUNT or set(train_tasks) & set(evaluation_tasks):
        raise ValueError("H1.33 evaluation tasks must be eight fresh identities disjoint from training")
    tasks = [_task_row(goals[idx], idx, offset, "h0.1-positive-train", f"T{offset + 1}") for offset, idx in enumerate(train_tasks)]
    tasks.extend(_task_row(goals[idx], idx, len(tasks), "h1.33-fresh-evaluation", f"P{offset + 1}") for offset, idx in enumerate(evaluation_tasks))
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "run_kind": "phase_h1.33_positive_replay_transfer",
        "status": "preregistered-not-authorized",
        "formal_task_consumption_allowed": False,
        "scientific_result": False,
        "source": {
            "experiment_git_sha": experiment_sha,
            "h0_reconciliation_sha256": h0_reconciliation_sha256,
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
            "prompt_contract": PROMPT_CONTRACT,
            "action_contract": ACTION_CONTRACT,
        },
        "train_tasks": train_tasks,
        "evaluation_tasks": evaluation_tasks,
        "tasks": tasks,
        "training_source": {
            "experiment": "H0.1-qwen-success-discovery-v2",
            "teacher": "disabled",
            "qualification": "identity_ok and trace_complete and no parser_or_fallback_failure and no error and final_task_score > 0",
            "task_count": len(train_tasks),
            "minimum_success_task_count": 8,
            "minimum_success_episode_count": 16,
        },
        "evaluation": {
            "tasks": evaluation_tasks,
            "rollout_seeds": EVALUATION_SEEDS,
            "arms": ["base", "self-evolution"],
            "cells": len(evaluation_tasks) * 2,
            "attempts": len(evaluation_tasks) * len(EVALUATION_SEEDS) * 2,
            "task_role": "fresh held-out evaluation only",
        },
        "task_selection": {
            "selection_seed": selection_seed,
            "goal_count": len(goals),
            "min_goal_index": MIN_GOAL_INDEX,
            "evaluation_is_disjoint_from_train": not set(train_tasks) & set(evaluation_tasks),
        },
        "contamination": {
            "evaluation_outcomes_used_for_training": False,
            "teacher_oracle_gpt_used": False,
            "h1_30_formal_tasks_consumed": False,
        },
        "manifest_sha256": "",
    }
    payload["manifest_sha256"] = digest_json(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--train-task-id", action="append", type=int, required=True)
    parser.add_argument("--evaluation-task-id", action="append", type=int)
    parser.add_argument("--prior-json-root", action="append", type=Path, default=[])
    parser.add_argument("--experiment-sha", required=True)
    parser.add_argument("--h0-reconciliation-sha256", required=True)
    parser.add_argument("--selection-seed", type=int, default=SELECTION_SEED)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_manifest(
        args.seed_root.expanduser().resolve(),
        args.train_task_id,
        args.evaluation_task_id,
        prior_json_roots=[path.expanduser().resolve() for path in args.prior_json_root],
        experiment_sha=args.experiment_sha,
        h0_reconciliation_sha256=args.h0_reconciliation_sha256,
        selection_seed=args.selection_seed,
    )
    output = args.output.expanduser().resolve()
    atomic_json(output, payload)
    print(json.dumps({"manifest": str(output), "manifest_sha256": payload["manifest_sha256"], "evaluation_tasks": payload["evaluation_tasks"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
