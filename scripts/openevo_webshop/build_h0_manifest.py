#!/usr/bin/env python3
"""Materialize the pre-registered Phase H0 WebShop task manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from parametric_rnd import (
    ENV_SEED,
    MAX_STEPS,
    TEMPERATURE,
    all_goals,
    atomic_json,
    digest_json,
)


H0_SELECTION_SEED = 20260814
H0_TASK_COUNT = 4
H0_MIN_GOAL_INDEX = 500
H0_MAX_NEW_TOKENS = 256
EXCLUDED_GOAL_INDICES = frozenset(
    {
        612,
        566,
        6680,
        2590,
        3420,
        6624,
        5216,
        2655,
        5457,
        3768,
        4513,
        1657,
    }
)


def select_h0_indices(
    goal_count: int, *, selection_seed: int = H0_SELECTION_SEED
) -> list[int]:
    if goal_count <= H0_MIN_GOAL_INDEX:
        raise ValueError("WebShop goal catalog is too small for H0 selection")
    pool = [
        index
        for index in range(H0_MIN_GOAL_INDEX, goal_count)
        if index not in EXCLUDED_GOAL_INDICES
    ]
    random.Random(selection_seed).shuffle(pool)
    selected = sorted(pool[:H0_TASK_COUNT])
    if len(selected) != H0_TASK_COUNT:
        raise ValueError("H0 task selection did not produce four tasks")
    return selected


def _task_row(
    goal: dict[str, Any], *, goal_idx: int, sample_index: int
) -> dict[str, Any]:
    description = str(goal.get("instruction_text") or "").strip()
    if not description:
        raise ValueError(f"WebShop goal {goal_idx} has no instruction_text")
    return {
        "sample_index": sample_index,
        "task_id": f"webshop_{goal_idx:06d}",
        "goal_idx": goal_idx,
        "split": "train-only-h0",
        "task_description": description,
        "task_description_sha256": hashlib.sha256(
            description.encode("utf-8")
        ).hexdigest(),
    }


def build_manifest(seed_root: Path, *, experiment_sha: str) -> dict[str, Any]:
    goals = all_goals(seed_root)
    selected = select_h0_indices(len(goals))
    tasks = [
        _task_row(goals[index], goal_idx=index, sample_index=position)
        for position, index in enumerate(selected)
    ]
    payload: dict[str, Any] = {
        "schema": "seed3090.webshop-h0-natural-success-manifest.v1",
        "run_kind": "phase_h0_natural_success_search",
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
            "max_steps": MAX_STEPS,
            "max_new_tokens": H0_MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "prompt_contract": "seed3090.webshop-shared-prompt.v1",
            "action_contract": "seed3090.webshop-action-parser.v1",
            "runner": "openevo_webshop.runner.run_episode",
            "parser_retry": {
                "enabled": True,
                "seed_offset": 900000,
                "max_retries_per_step": 1,
            },
        },
        "task_selection": {
            "selection_seed": H0_SELECTION_SEED,
            "goal_count": len(goals),
            "min_goal_index": H0_MIN_GOAL_INDEX,
            "excluded_goal_indices": sorted(EXCLUDED_GOAL_INDICES),
            "selected_goal_indices": selected,
            "split": "train-only-h0",
            "phase_g_tasks_consumed": True,
        },
        "rollout": {
            "initial_rollouts": 128,
            "tasks": selected,
            "rollouts_per_task": 32,
            "generation_seeds": list(range(2026081400, 2026081432)),
            "independent_seed_only": True,
            "extension_limit_if_zero_success": 256,
        },
        "formal_held_out": "unseen-and-withheld",
        "tasks": tasks,
    }
    payload["manifest_sha256"] = digest_json(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--experiment-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_manifest(
        args.seed_root.expanduser().resolve(), experiment_sha=args.experiment_sha
    )
    atomic_json(args.output.expanduser().resolve(), payload)
    print(
        json.dumps(
            {
                "manifest": str(args.output.resolve()),
                "manifest_sha256": payload["manifest_sha256"],
                "task_indices": payload["task_selection"]["selected_goal_indices"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
