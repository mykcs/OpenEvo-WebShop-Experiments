#!/usr/bin/env python3
"""Freeze the independent H1.39-MR WebShop panel.

The selector is deliberately independent of H1.38B outcomes.  It reads only
the pinned WebShop seed root and a committed, machine-readable exclusion ledger.
No reward, attempt result, or task family label is consulted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

try:
    from scripts.openevo_webshop.parametric_rnd import ENV_SEED, all_goals, atomic_json, digest_json
except ModuleNotFoundError:
    from parametric_rnd import ENV_SEED, all_goals, atomic_json, digest_json  # type: ignore


CAMPAIGN_ID = "20260820-0139-h139-mr-independent-replication"
SCHEMA = "openevo.webshop-h1.39-mr-independent-panel.v1"
MIN_GOAL_INDEX = 500
TASK_COUNT = 64
SELECTION_SEED = 2026082039
ROLLOUT_SEEDS = [2026103900, 2026103901, 2026103902, 2026103903]
ARMS = ["base", "ordinary-lora", "sd-lora"]
ARM_IDENTITIES = {
    "base": {"adapter": "none", "adapter_file_sha256": None, "adapter_tensor_digest": None},
    "ordinary-lora": {
        "adapter": "frozen-h1.38b-final-ordinary",
        "adapter_file_sha256": "22a5adb1a0bdda5b56372a2926bc2724e45b416493403ece86755330add8cd7d",
        "adapter_tensor_digest": "e3be874b71fde1c989e26531255cad063607b73b4727bda59ba824a0f4a7c262",
    },
    "sd-lora": {
        "adapter": "frozen-h1.38b-final-sd",
        "adapter_file_sha256": "47e63d417f85cb2defed5ff1ee934128b96e6a4f78a8f410fa052957d919190c",
        "adapter_tensor_digest": "eedc36232f04a8a7d6e8fc6a38b935758db65206f4945a6cfa35c4ab666a48f9",
    },
}
TASK_KEY_NAMES = {
    "goal_idx", "goal_index", "goal_indices", "selected_goal_indices",
    "task_id", "task_ids", "train_task_ids", "evaluation_task_ids", "evaluation_tasks", "train_tasks",
    "excluded_goal_indices", "completed_task_ids", "completed_goal_indices",
}
TASK_ID_RE = re.compile(r"(?:webshop[_-])?(\d{3,})$")


def _indices(value: Any) -> set[int]:
    if isinstance(value, bool):
        return set()
    if isinstance(value, int):
        return {value}
    if isinstance(value, str):
        match = TASK_ID_RE.fullmatch(value.strip())
        return {int(match.group(1))} if match else set()
    if isinstance(value, list):
        found: set[int] = set()
        for item in value:
            found.update(_indices(item))
        return found
    return set()


def collect_exclusions(repo_root: Path) -> tuple[set[int], dict[str, list[int]]]:
    """Collect explicit task identities from committed configs/evidence only."""
    roots = [repo_root / "configs" / "experiment", repo_root / "docs" / "evidence"]
    used: set[int] = set()
    sources: dict[str, list[int]] = {}

    def walk(value: Any) -> set[int]:
        found: set[int] = set()
        if isinstance(value, dict):
            for key, item in value.items():
                if key in TASK_KEY_NAMES:
                    found.update(_indices(item))
                found.update(walk(item))
        elif isinstance(value, list):
            for item in value:
                found.update(walk(item))
        return found

    for root in roots:
        for path in sorted(root.rglob("*.json")):
            try:
                found = walk(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            found = {idx for idx in found if idx >= MIN_GOAL_INDEX}
            if found:
                rel = path.relative_to(repo_root).as_posix()
                sources[rel] = sorted(found)
                used.update(found)
    return used, sources


def build_exclusion_ledger(repo_root: Path, *, experiment_sha: str) -> dict[str, Any]:
    exclusions, sources = collect_exclusions(repo_root)
    payload: dict[str, Any] = {
        "schema": "openevo.webshop-h1.39-mr-exclusion-ledger.v1",
        "campaign_id": CAMPAIGN_ID,
        "experiment_git_sha": experiment_sha,
        "policy": "exclude every explicitly committed WebShop task identity from prior formal, reserved, aborted, training, evaluation, reservoir, H1.38A-C, and H1.38B evidence; never select by reward or task family",
        "minimum_goal_index": MIN_GOAL_INDEX,
        "excluded_goal_indices": sorted(exclusions),
        "excluded_task_ids": [f"webshop_{idx:06d}" for idx in sorted(exclusions)],
        "source_files": sources,
        "excluded_count": len(exclusions),
    }
    payload["excluded_indices_sha256"] = hashlib.sha256(
        json.dumps(payload["excluded_goal_indices"], separators=(",", ":")).encode()
    ).hexdigest()
    payload["ledger_sha256"] = digest_json(payload)
    return payload


def _task_row(goal: dict[str, Any], goal_idx: int, sample_index: int) -> dict[str, Any]:
    description = str(goal.get("instruction_text") or "").strip()
    if not description:
        raise ValueError(f"goal {goal_idx} has no instruction_text")
    return {
        "sample_index": sample_index,
        "task_id": f"webshop_{goal_idx:06d}",
        "goal_idx": goal_idx,
        "split": "h1.39-mr-independent-fresh",
        "task_description": description,
        "task_description_sha256": hashlib.sha256(description.encode()).hexdigest(),
    }


def build_manifest(seed_root: Path, *, repo_root: Path, experiment_sha: str) -> dict[str, Any]:
    goals = all_goals(seed_root.expanduser().resolve())
    ledger = build_exclusion_ledger(repo_root.resolve(), experiment_sha=experiment_sha)
    excluded = set(ledger["excluded_goal_indices"])
    eligible = [idx for idx in range(MIN_GOAL_INDEX, len(goals)) if idx not in excluded]
    eligible_digest = digest_json({"goal_count": len(goals), "eligible_goal_indices": eligible})
    if len(eligible) < TASK_COUNT:
        raise ValueError(f"eligible pool {len(eligible)} smaller than requested {TASK_COUNT}")
    shuffled = list(eligible)
    random.Random(SELECTION_SEED).shuffle(shuffled)
    selected = sorted(shuffled[:TASK_COUNT])
    tasks = [_task_row(goals[idx], idx, pos) for pos, idx in enumerate(selected)]
    selected_digest = digest_json(tasks)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "experiment_git_sha": experiment_sha,
        "benchmark": {
            "name": "webshop",
            "catalog": "indexes_1k",
            "environment_seed": ENV_SEED,
            "max_steps": 30,
            "max_new_tokens": 512,
            "temperature": 0.0,
            "prompt_contract": "seed3090.webshop-compact-action-prompt.v2",
            "parser_contract": "seed3090.webshop-action-parser.v3-action-command",
            "evaluator": "WebShop task_score/won",
        },
        "task_selection": {
            "selection_seed": SELECTION_SEED,
            "goal_count": len(goals),
            "minimum_goal_index": MIN_GOAL_INDEX,
            "eligible_pool_count": len(eligible),
            "eligible_pool_digest": eligible_digest,
            "exclusion_ledger_sha256": ledger["ledger_sha256"],
            "excluded_indices_sha256": ledger["excluded_indices_sha256"],
            "selected_panel_digest": selected_digest,
            "manual_replacement_allowed": False,
        },
        "evaluation_tasks": selected,
        "tasks": tasks,
        "evaluation": {
            "arms": ARMS,
            "rollout_seeds": ROLLOUT_SEEDS,
            "cells": TASK_COUNT * len(ARMS),
            "attempts": TASK_COUNT * len(ARMS) * len(ROLLOUT_SEEDS),
            "task_role": "untouched task-ID-disjoint independent replication panel",
        },
        "arm_identities": ARM_IDENTITIES,
        "contamination": {
            "task_id_disjoint_from_committed_exclusions": not set(selected) & excluded,
            "evaluation_outcomes_used_for_selection": False,
            "manual_browsing_or_replacement": False,
            "adapter_retraining": False,
        },
        "manifest_sha256": "",
    }
    payload["manifest_sha256"] = digest_json(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--seed-root", type=Path)
    parser.add_argument("--experiment-sha", required=True)
    parser.add_argument("--exclusion-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    ledger = build_exclusion_ledger(args.repo_root.resolve(), experiment_sha=args.experiment_sha)
    atomic_json(args.exclusion_ledger.resolve(), ledger)
    if args.seed_root is None:
        print(json.dumps({"exclusion_ledger": str(args.exclusion_ledger.resolve()), "ledger_sha256": ledger["ledger_sha256"], "excluded_count": ledger["excluded_count"]}, sort_keys=True))
        return 0
    if args.output is None:
        parser.error("--output is required with --seed-root")
    manifest = build_manifest(args.seed_root, repo_root=args.repo_root, experiment_sha=args.experiment_sha)
    atomic_json(args.output.resolve(), manifest)
    print(json.dumps({"manifest": str(args.output.resolve()), "manifest_sha256": manifest["manifest_sha256"], "selected_panel_digest": manifest["task_selection"]["selected_panel_digest"], "tasks": manifest["evaluation_tasks"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
