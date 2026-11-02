#!/usr/bin/env python3
"""Build the frozen, outcome-blind H1.40-G2 task partitions.

This builder is CPU-only.  It never opens WebShop, reads reward/attempt output,
or creates a formal execution record.  A seed root is only used as a pinned
catalog of task identities and descriptions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


CAMPAIGN_ID = "20260820-0140-h140-g2-multigeneration"
MIN_GOAL_INDEX = 500
TASK_COUNT = 64
ANCHOR_COUNT = 32
E1_TASK_IDS = [607, 2192, 4238, 4477, 4533, 5753, 5819, 6751]
E2_SELECTION_SEED = 20260820401
PRESERVATION_SELECTION_SEED = 20260820402
T2_SELECTION_SEED = 20260820403
HARVEST_SEEDS = [2026104000, 2026104001, 2026104002, 2026104003]
EVALUATION_SEEDS = [2026104100, 2026104101, 2026104102, 2026104103]
T2_SEEDS = [2026104200, 2026104201, 2026104202, 2026104203]
TASK_KEYS = {
    "goal_idx", "goal_index", "goal_indices", "selected_goal_indices", "task_id", "task_ids",
    "train_task_ids", "evaluation_task_ids", "evaluation_tasks", "train_tasks",
    "excluded_goal_indices", "completed_task_ids", "completed_goal_indices", "identity_order",
}
TASK_ID_RE = re.compile(r"(?:webshop[_-])?(\d{3,})$")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def task_id(index: int) -> str:
    return f"webshop_{index:06d}"


def indices(value: Any) -> set[int]:
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
            found.update(indices(item))
        return found
    return set()


def collect_committed_ids(repo_root: Path) -> tuple[set[int], dict[str, list[int]]]:
    roots = [repo_root / "configs" / "experiment", repo_root / "docs" / "evidence"]
    used: set[int] = set()
    sources: dict[str, list[int]] = {}

    def walk(value: Any) -> set[int]:
        found: set[int] = set()
        if isinstance(value, dict):
            for key, item in value.items():
                if key in TASK_KEYS:
                    found.update(indices(item))
                found.update(walk(item))
        elif isinstance(value, list):
            for item in value:
                found.update(walk(item))
        return found

    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.json")):
            rel = path.relative_to(repo_root).as_posix()
            if "/h1.40" in rel or rel.startswith("configs/experiment/h1.40"):
                continue
            try:
                found = {idx for idx in walk(json.loads(path.read_text(encoding="utf-8"))) if idx >= MIN_GOAL_INDEX}
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if found:
                sources[rel] = sorted(found)
                used.update(found)
    return used, sources


def load_goals(*, seed_root: Path | None, goals_json: Path | None) -> list[dict[str, Any]]:
    if goals_json is not None:
        value = json.loads(goals_json.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError("goals JSON must be an array")
        return value
    if seed_root is None:
        raise ValueError("one of --seed-root or --goals-json is required")
    try:
        from scripts.openevo_webshop.parametric_rnd import all_goals
    except ModuleNotFoundError:
        from parametric_rnd import all_goals  # type: ignore
    return all_goals(seed_root.resolve())


def row(goals: list[dict[str, Any]], index: int, position: int, split: str) -> dict[str, Any]:
    text = str(goals[index].get("instruction_text") or "").strip()
    if not text:
        raise ValueError(f"goal {index} has empty instruction_text")
    return {
        "sample_index": position,
        "task_id": task_id(index),
        "goal_idx": index,
        "split": split,
        "task_description": text,
        "task_description_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def select_shuffled(pool: set[int], seed: int, count: int) -> list[int]:
    values = sorted(pool)
    random.Random(seed).shuffle(values)
    if len(values) < count:
        raise ValueError(f"eligible pool {len(values)} smaller than requested {count}")
    return values[:count]


def build(repo_root: Path, goals: list[dict[str, Any]], experiment_sha: str, output_dir: Path) -> dict[str, Any]:
    if len(goals) <= MIN_GOAL_INDEX:
        raise ValueError("goal catalog does not contain the required train-only range")
    prior, sources = collect_committed_ids(repo_root)
    excluded = set(prior) | set(E1_TASK_IDS)

    h139_path = repo_root / "configs/experiment/manifests/h1.39-mr-independent-panel-v1.json"
    h139 = json.loads(h139_path.read_text(encoding="utf-8"))
    h139_tasks = [int(value) for value in h139["evaluation_tasks"]]
    excluded.update(h139_tasks)
    sources["configs/experiment/h1.39-mr-independent-panel-v1.json"] = sorted(h139_tasks)
    eligible = {idx for idx in range(MIN_GOAL_INDEX, len(goals)) if idx not in excluded}
    e2_indices = select_shuffled(eligible, E2_SELECTION_SEED, TASK_COUNT)
    remaining = eligible - set(e2_indices)
    t2_indices = select_shuffled(remaining, T2_SELECTION_SEED, TASK_COUNT)
    anchor_indices = sorted(h139_tasks, key=lambda idx: hashlib.sha256(f"{PRESERVATION_SELECTION_SEED}:{task_id(idx)}".encode()).hexdigest())[:ANCHOR_COUNT]
    reserved = sorted(set(e2_indices) | set(t2_indices) | set(anchor_indices) | set(E1_TASK_IDS))

    ledger = {
        "schema": "openevo.webshop-h1.40-g2-exclusion-ledger.v1",
        "campaign_id": CAMPAIGN_ID,
        "experiment_git_sha": experiment_sha,
        "policy": "Exclude every committed prior identity, the exact H1.39 preservation anchor, E1 retention identities, E2 reservoir identities, and T2 identities; selection never reads reward or task outcomes.",
        "minimum_goal_index": MIN_GOAL_INDEX,
        "excluded_goal_indices": sorted(excluded),
        "excluded_task_ids": [task_id(idx) for idx in sorted(excluded)],
        "reserved_h140_goal_indices": reserved,
        "reserved_h140_task_ids": [task_id(idx) for idx in reserved],
        "source_files": sources,
        "excluded_count": len(excluded),
        "eligible_pool_count_before_h140": len(eligible),
        "selection_seeds": {"e2_reservoir": E2_SELECTION_SEED, "preservation_anchor": PRESERVATION_SELECTION_SEED, "t2": T2_SELECTION_SEED},
    }
    ledger["excluded_indices_sha256"] = digest(ledger["excluded_goal_indices"])
    ledger["ledger_sha256"] = digest(ledger)

    e1_tasks = [row(goals, idx, pos, "h1.40-e1-retention") for pos, idx in enumerate(E1_TASK_IDS)]
    e2_tasks = [row(goals, idx, pos, "h1.40-e2-harvest-reservoir") for pos, idx in enumerate(e2_indices)]
    anchor_tasks = [row(goals, idx, pos, "h1.40-h139-preservation-anchor") for pos, idx in enumerate(anchor_indices)]
    t2_tasks = [row(goals, idx, pos, "h1.40-t2-fresh") for pos, idx in enumerate(t2_indices)]
    common = {
        "campaign_id": CAMPAIGN_ID,
        "experiment_git_sha": experiment_sha,
        "benchmark": {
            "name": "webshop", "catalog": "indexes_1k", "environment_seed": 2026,
            "max_steps": 30, "max_new_tokens": 512, "temperature": 0.0,
            "prompt_contract": "seed3090.webshop-compact-action-prompt.v2",
            "parser_contract": "seed3090.webshop-action-parser.v3-action-command",
            "evaluator": "WebShop task_score/won",
        },
        "exclusion_ledger_sha256": ledger["ledger_sha256"],
        "task_selection_outcome_blind": True,
        "manual_replacement_allowed": False,
    }

    def manifest(schema: str, role: str, tasks: list[dict[str, Any]], selection_seed: int, seeds: list[int], arms: list[str], attempts: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        value = {**common, "schema": schema, "role": role, "selection_seed": selection_seed, "rollout_seeds": seeds, "arms": arms, "cells": len(tasks) * len(arms), "attempts": attempts, "tasks": tasks}
        if extra:
            value.update(extra)
        value["manifest_sha256"] = digest(value)
        return value

    documents = {
        "h1.40-g2-exclusion-ledger-v1.json": ledger,
        "h1.40-g2-e1-retention-set-v1.json": manifest("openevo.webshop-h1.40-g2-e1-retention.v1", "prior-generation retention", e1_tasks, 0, EVALUATION_SEEDS, ["g1", "g2"], 64, {"source": "H1.38B sealed D8 identity order", "training_identity_count": 8}),
        "h1.40-g2-e2-reservoir-v1.json": manifest("openevo.webshop-h1.40-g2-e2-reservoir.v1", "G1 endogenous harvest reservoir", e2_tasks, E2_SELECTION_SEED, HARVEST_SEEDS, ["g1"], 256, {"maximum_harvest_attempts": 256, "qualification_contract": "see preregistration success_qualification", "stop_rule": "MGH HARVEST_INSUFFICIENT after 256 attempts if fewer than 8 identities have two qualified successes"}),
        "h1.40-g2-h139-preservation-anchor-v1.json": manifest("openevo.webshop-h1.40-g2-h139-preservation.v1", "H1.39 replicated-transfer preservation anchor", anchor_tasks, PRESERVATION_SELECTION_SEED, EVALUATION_SEEDS, ["g1", "g2"], 256, {"selection_rule": "sort H1.39 task IDs by sha256(seed:task_id), take first 32; no reward/outcome fields consulted"}),
        "h1.40-g2-t2-panel-v1.json": manifest("openevo.webshop-h1.40-g2-t2-panel.v1", "T2 untouched fresh panel", t2_tasks, T2_SELECTION_SEED, T2_SEEDS, ["g0", "g1", "g2"], 768, {"primary_contrast": "g2_minus_g1", "secondary_contrasts": ["g1_minus_g0", "g2_minus_g0"]}),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in documents.items():
        (output_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"campaign_id": CAMPAIGN_ID, "ledger_sha256": ledger["ledger_sha256"], "eligible_pool_count": len(eligible), "e2_tasks": e2_indices, "t2_tasks": t2_indices, "anchor_tasks": anchor_indices, "e1_tasks": E1_TASK_IDS, "manifest_sha256": {name: value["manifest_sha256"] for name, value in documents.items() if "manifest_sha256" in value}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-sha", required=True)
    parser.add_argument("--seed-root", type=Path)
    parser.add_argument("--goals-json", type=Path)
    args = parser.parse_args()
    result = build(args.repo_root.resolve(), load_goals(seed_root=args.seed_root, goals_json=args.goals_json), args.experiment_sha, args.output_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
