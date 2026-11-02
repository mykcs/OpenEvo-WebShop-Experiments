#!/usr/bin/env python3
"""Build the H1.37 frozen-adapter fresh-task evaluation manifest.

H1.37 changes NOTHING about the model, the trained adapter, the parser, or the
inference contract. Its only independent variable is the evaluation sample: a
brand-new set of 64 WebShop task identities that no prior campaign has ever
selected, observed, or consumed. The frozen H1.36 self-evolution adapter and its
frozen Qwen2.5-7B base are evaluated paired on this untouched sample.

The exclusion set below is frozen at preregistration time. It is the union of
every goal_idx that appears in any committed prior campaign manifest or config
under docs/evidence/remote-runs or configs/experiment (a strict superset of the
H1.29/H1.36 train+evaluation tasks and the H1.36 self-generated-training task
identities). Over-exclusion is conservative: it can only shrink the eligible
pool, never admit a previously touched task. Selection uses a fixed preregistered
seed so the 64-task sample is reproducible and sealed before any evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

try:
    from scripts.openevo_webshop.parametric_rnd import ENV_SEED, all_goals, atomic_json, digest_json
except ModuleNotFoundError:  # pragma: no cover - script-style invocation
    from parametric_rnd import ENV_SEED, all_goals, atomic_json, digest_json


UPSTREAM_OPENEVO_STABLE = "796e6248a61112a61c880885844d3b0420c0b6ae"

# Frozen inference objects (carried over byte-identical from H1.36; H1.37 does not
# retrain or modify them). The adapter content sha256 is pinned fail-closed by the
# launcher at run time and recorded in the run evidence, not in this task manifest.
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_LOCAL_DIR = "/workspace/models/qwen2.5-7b-instruct"
MODEL_CONTENT_PIN = "0c32faf81d8f2f83cf79860d2a48111f166348d6dd38cd348576d8b3a0141c1e"
FROZEN_ADAPTER_WORKER = "sd-lora-b7e36404ca6320620e3f86e5225699e3"
H136_RUN_NAME = "phase-h1.36-strong-base-self-evolution-v1-20260818"

# Frozen inference contract (identical to the H1.36 v3 evaluation re-run).
EVALUATION_SEEDS = [2026090200, 2026090201, 2026090202, 2026090203]
ACTION_CONTRACT = "seed3090.webshop-action-parser.v3-action-command"
PROMPT_CONTRACT = "seed3090.webshop-compact-action-prompt.v2"

# Sample size (preregistered): 64 fresh tasks x 4 seeds x 2 arms = 512 episodes.
EVALUATION_TASK_COUNT = 64
TRAIN_START = 500

# Preregistered selection seed, fixed before any evaluation. Distinct from the
# H1.36 dates so it cannot be confused with a prior campaign's sampling.
SELECTION_SEED = 20260818

# Frozen exclusion set: union of every goal_idx observed/consumed in any prior
# committed campaign (229 identities). Computed 2026-08-18 by scanning
# docs/evidence/remote-runs and configs/experiment; sha256 of the JSON-serialized
# sorted list is 0d73fb07a1c06403cdcbbd042f2807757126e064dfdc8c05e771fbab98ebe9c4.
EXCLUDED_GOAL_INDICES = [500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 519, 529, 550, 551, 566, 577, 594, 612, 622, 643, 680, 688, 714, 744, 863, 904, 920, 958, 980, 1006, 1010, 1082, 1098, 1101, 1155, 1166, 1193, 1263, 1270, 1321, 1354, 1393, 1416, 1437, 1507, 1568, 1573, 1584, 1610, 1614, 1623, 1647, 1657, 1663, 1711, 1714, 1719, 1720, 1740, 1763, 1795, 1880, 1888, 1910, 1920, 1981, 1983, 1985, 1990, 2062, 2081, 2087, 2112, 2114, 2163, 2164, 2193, 2198, 2244, 2251, 2279, 2316, 2331, 2334, 2350, 2408, 2431, 2530, 2532, 2550, 2562, 2590, 2655, 2687, 2709, 2725, 2807, 2813, 2822, 2858, 2887, 2900, 2901, 2927, 2936, 2964, 3020, 3028, 3071, 3094, 3113, 3166, 3178, 3222, 3227, 3246, 3293, 3301, 3325, 3353, 3409, 3420, 3449, 3512, 3589, 3697, 3735, 3759, 3768, 3809, 3822, 3848, 3881, 3918, 3924, 3939, 4047, 4086, 4111, 4146, 4169, 4170, 4313, 4317, 4334, 4415, 4427, 4457, 4486, 4513, 4559, 4597, 4600, 4648, 4709, 4819, 4849, 4863, 4913, 4924, 4928, 4949, 4980, 5004, 5013, 5065, 5069, 5135, 5169, 5170, 5216, 5238, 5265, 5294, 5376, 5377, 5420, 5434, 5442, 5443, 5457, 5469, 5493, 5561, 5567, 5571, 5593, 5634, 5661, 5674, 5678, 5711, 5790, 5823, 5841, 5894, 5914, 5951, 6017, 6026, 6091, 6174, 6228, 6261, 6262, 6306, 6331, 6384, 6411, 6428, 6459, 6469, 6484, 6558, 6583, 6602, 6603, 6617, 6624, 6680, 6709, 6731, 6737, 6756, 6757, 6769, 6809, 6826, 6898]
EXCLUSION_SHA256 = "0d73fb07a1c06403cdcbbd042f2807757126e064dfdc8c05e771fbab98ebe9c4"


def _description(goals: list[dict[str, Any]], goal_idx: int) -> str:
    if goal_idx >= len(goals):
        raise ValueError(f"goal index {goal_idx} outside seed-root goal list of {len(goals)}")
    description = str(goals[goal_idx].get("instruction_text") or "").strip()
    if not description:
        raise ValueError(f"WebShop goal {goal_idx} has no instruction_text")
    return description


def select_evaluation_tasks(goal_count: int) -> list[int]:
    """Reproducibly pick EVALUATION_TASK_COUNT fresh goal indices from the eligible pool."""
    excluded = set(EXCLUDED_GOAL_INDICES)
    eligible = [idx for idx in range(TRAIN_START, goal_count) if idx not in excluded]
    if len(eligible) < EVALUATION_TASK_COUNT:
        raise ValueError(f"eligible pool {len(eligible)} smaller than requested {EVALUATION_TASK_COUNT}")
    pool = list(eligible)
    random.Random(SELECTION_SEED).shuffle(pool)
    return sorted(pool[:EVALUATION_TASK_COUNT])


def build_manifest(seed_root: Path, *, experiment_sha: str, generated_at_utc: str) -> dict[str, Any]:
    goals = all_goals(seed_root.expanduser().resolve())
    goal_count = len(goals)
    evaluation_indices = select_evaluation_tasks(goal_count)
    if len(set(evaluation_indices)) != EVALUATION_TASK_COUNT:
        raise ValueError("evaluation task selection contains duplicates")
    if set(evaluation_indices) & set(EXCLUDED_GOAL_INDICES):
        raise ValueError("evaluation selection leaked a previously consumed task")

    tasks: list[dict[str, Any]] = []
    for sample_index, goal_idx in enumerate(evaluation_indices):
        description = _description(goals, goal_idx)
        tasks.append({
            "sample_index": sample_index,
            "task_id": f"webshop_{goal_idx:06d}",
            "goal_idx": goal_idx,
            "split": "fresh-heldout-evaluation",
            "task_description": description,
            "task_description_sha256": hashlib.sha256(description.encode()).hexdigest(),
        })

    payload: dict[str, Any] = {
        "schema": "openevo.webshop-h1.37-frozen-adapter-fresh-task-eval-manifest.v1",
        "run_kind": "phase_h1.37_frozen_adapter_fresh_task_eval",
        "generated_at_utc": generated_at_utc,
        "source": {
            "experiment_git_sha": experiment_sha,
            "official_open_evo_stable": UPSTREAM_OPENEVO_STABLE,
            "model": MODEL_ID,
            "model_local_dir": MODEL_LOCAL_DIR,
            "model_revision": MODEL_CONTENT_PIN,
            "model_revision_kind": "local-content-pin",
            "frozen_adapter": {
                "worker_id": FROZEN_ADAPTER_WORKER,
                "source_run": H136_RUN_NAME,
                "note": "H1.36-trained SD-LoRA adapter reused frozen; content sha256 pinned fail-closed by the launcher at run time.",
            },
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
        "train_tasks": [],
        "evaluation_tasks": evaluation_indices,
        "tasks": tasks,
        "task_selection": {
            "provenance": "fresh reproducible sample from the never-consumed eligible pool; NOT reused from any prior campaign",
            "sampling_universe": f"goal indices [{TRAIN_START}, {goal_count}) under environment_seed {ENV_SEED} ({goal_count - TRAIN_START} candidates)",
            "goal_count": goal_count,
            "selection_seed": SELECTION_SEED,
            "requested_count": EVALUATION_TASK_COUNT,
            "eligible_pool_count": goal_count - TRAIN_START - len([i for i in EXCLUDED_GOAL_INDICES if i >= TRAIN_START]),
            "exclusion": {
                "policy": "exclude every goal_idx selected/observed/consumed in any prior committed campaign (superset of H1.29/H1.36 train+eval and H1.36 self-generated-training identities)",
                "excluded_count": len(EXCLUDED_GOAL_INDICES),
                "excluded_sha256": EXCLUSION_SHA256,
            },
        },
        "evaluation": {
            "tasks": evaluation_indices,
            "rollout_seeds": EVALUATION_SEEDS,
            "arms": ["base", "self-evolution"],
            "cells": len(evaluation_indices) * 2,
            "attempts": len(evaluation_indices) * len(EVALUATION_SEEDS) * 2,
            "task_role": "previously untouched held-out evaluation for frozen-adapter fresh-task transfer",
        },
        "contamination": {
            "train_evaluation_overlap": False,
            "evaluation_outcomes_used_for_training": False,
            "teacher_oracle_gpt_used": False,
            "no_retraining": True,
            "frozen_adapter_reused": True,
            "evaluation_tasks_disjoint_from_all_prior": True,
        },
        "manifest_sha256": "",
    }
    payload["manifest_sha256"] = digest_json(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--experiment-sha", required=True)
    parser.add_argument("--generated-at-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_manifest(
        args.seed_root,
        experiment_sha=args.experiment_sha,
        generated_at_utc=args.generated_at_utc,
    )
    output = args.output.expanduser().resolve()
    atomic_json(output, payload)
    print(json.dumps({
        "manifest": str(output),
        "manifest_sha256": payload["manifest_sha256"],
        "manifest_file_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "evaluation_task_count": len(payload["evaluation_tasks"]),
        "evaluation_tasks": payload["evaluation_tasks"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
