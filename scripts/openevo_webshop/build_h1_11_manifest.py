#!/usr/bin/env python3
"""Materialize the pre-registered H1.11 fresh-task evaluation panel."""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path
from typing import Any

from parametric_rnd import ENV_SEED, MAX_STEPS, all_goals, atomic_json, digest_json


SELECTION_SEED = 20260820
TASK_COUNT = 8
MIN_GOAL_INDEX = 500
EXCLUDED_GOAL_INDICES = frozenset({
    612, 566, 6680, 2590, 3420, 6624, 5216, 2655, 5457, 3768, 4513, 1657,
    2822, 5135, 5294, 6769, 694, 2747, 3098, 3396, 4666, 4676, 4724, 6199,
    863, 1985, 2062, 3759, 5169, 5170, 6017, 6306,
})


def select_indices(goal_count: int, *, selection_seed: int = SELECTION_SEED) -> list[int]:
    pool = [idx for idx in range(MIN_GOAL_INDEX, goal_count) if idx not in EXCLUDED_GOAL_INDICES]
    random.Random(selection_seed).shuffle(pool)
    selected = sorted(pool[:TASK_COUNT])
    if len(selected) != TASK_COUNT:
        raise ValueError("H1.11 selection did not produce eight tasks")
    return selected


def build_manifest(seed_root: Path, *, experiment_sha: str, selection_seed: int = SELECTION_SEED) -> dict[str, Any]:
    goals = all_goals(seed_root)
    selected = select_indices(len(goals), selection_seed=selection_seed)
    tasks = []
    for sample_index, goal_idx in enumerate(selected):
        description = str(goals[goal_idx].get("instruction_text") or "").strip()
        if not description:
            raise ValueError(f"WebShop goal {goal_idx} has no instruction_text")
        tasks.append({
            "sample_index": sample_index,
            "task_id": f"webshop_{goal_idx:06d}",
            "goal_idx": goal_idx,
            "partition": "A" if sample_index < 4 else "B",
            "split": "held-out-h1.11-new-supervision",
            "task_description": description,
            "task_description_sha256": hashlib.sha256(description.encode()).hexdigest(),
        })
    payload: dict[str, Any] = {
        "schema": "seed3090.webshop-h1.11-new-supervision-manifest.v1",
        "run_kind": "phase_h1.11_new_positive_supervision_fit",
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
            "max_new_tokens": 512,
            "temperature": 0.0,
            "prompt_contract": "seed3090.webshop-compact-action-prompt.v2",
            "action_contract": "seed3090.webshop-action-parser.v1",
            "runner": "openevo_webshop.runner.run_episode",
        },
        "task_selection": {
            "selection_seed": selection_seed,
            "goal_count": len(goals),
            "min_goal_index": MIN_GOAL_INDEX,
            "excluded_goal_indices": sorted(EXCLUDED_GOAL_INDICES),
            "selected_goal_indices": selected,
            "partitions": {"A": selected[:4], "B": selected[4:]},
            "split": "held-out-h1.11-new-supervision",
        },
        "rollout": {"attempts": 96, "tasks_per_partition": 4, "seeds": list(range(2026082000, 2026082004)), "independent_seed_only": True},
        "tasks": tasks,
    }
    payload["manifest_sha256"] = digest_json(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--experiment-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, default=SELECTION_SEED)
    args = parser.parse_args()
    payload = build_manifest(args.seed_root.expanduser().resolve(), experiment_sha=args.experiment_sha, selection_seed=args.selection_seed)
    atomic_json(args.output.expanduser().resolve(), payload)
    print({"manifest": str(args.output.resolve()), "manifest_sha256": payload["manifest_sha256"], "goal_indices": payload["task_selection"]["selected_goal_indices"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
