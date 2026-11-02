#!/usr/bin/env python3
"""Build and verify the method-independent H1.30 matched WebShop manifest.

The existing qualification manifest intentionally uses a different contract
(stochastic 15-step promotion). H1.30 has its own deterministic 30-step,
four-rollout-seed contract and must never reuse that manifest by accident.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.openevo_webshop.webshop_task_manifest import (
        ACTION_CONTRACT,
        DATA_FILES,
        DATA_SOURCE,
        EVALUATOR,
        NUM_PRODUCTS,
        canonical_bytes,
        index_tree_identity,
        live_goals,
    )
except ModuleNotFoundError:
    from webshop_task_manifest import (  # type: ignore
        ACTION_CONTRACT,
        DATA_FILES,
        DATA_SOURCE,
        EVALUATOR,
        NUM_PRODUCTS,
        canonical_bytes,
        index_tree_identity,
        live_goals,
    )


SCHEMA = "openevo.webshop-h1.30-matched-manifest.v1"
ENVIRONMENT_SEED = 2026
SELECTION_SEED = 20260910
ROLLOUT_SEEDS = [2026091100, 2026091101, 2026091102, 2026091103]
TASK_COUNT = 8
MIN_GOAL_INDEX = 500
MAX_STEPS = 30
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0
HISTORY_LENGTH = 2
PROMPT_CONTRACT = "seed3090.webshop-compact-action-prompt.v2"
ARMS = ["OPEN_EVO", "SEED"]

_INDEX_KEYS = {
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


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _goal_indices(value: Any) -> set[int]:
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
            result.update(_goal_indices(item))
        return result
    return set()


def _collect(value: Any) -> set[int]:
    used: set[int] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _INDEX_KEYS:
                used.update(_goal_indices(item))
            used.update(_collect(item))
    elif isinstance(value, list):
        for item in value:
            used.update(_collect(item))
    return used


def collect_prior_indices(paths: Iterable[Path]) -> set[int]:
    """Collect task identities from prior JSON evidence without guessing."""

    used: set[int] = set()
    for root in paths:
        root = root.expanduser().resolve()
        candidates = [root] if root.is_file() else sorted(root.rglob("*.json"))
        for path in candidates:
            try:
                used.update(_collect(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
    return used


def _task_row(goal: dict[str, Any], goal_idx: int, sample_index: int) -> dict[str, Any]:
    description = str(goal.get("instruction_text") or "").strip()
    if not description:
        raise ValueError(f"WebShop goal {goal_idx} has no instruction_text")
    return {
        "sample_index": sample_index,
        "task_id": f"webshop_{goal_idx:06d}",
        "goal_idx": goal_idx,
        "split": "h1.30-matched-evaluation",
        "task_description": description,
        "task_description_sha256": sha256_text(description),
    }


def build_manifest(
    goals: list[dict[str, Any]],
    *,
    index_identity: dict[str, Any],
    prior_indices: set[int],
    experiment_git_sha: str,
    seed_source_commit: str,
    selection_seed: int = SELECTION_SEED,
) -> dict[str, Any]:
    if len(goals) <= MIN_GOAL_INDEX:
        raise ValueError("pinned WebShop goal list is too small")
    pool = [idx for idx in range(MIN_GOAL_INDEX, len(goals)) if idx not in prior_indices]
    random.Random(selection_seed).shuffle(pool)
    selected = pool[:TASK_COUNT]
    if len(selected) != TASK_COUNT or len(set(selected)) != TASK_COUNT:
        raise ValueError("could not select eight unused H1.30 task identities")
    tasks = [_task_row(goals[idx], idx, sample_index) for sample_index, idx in enumerate(selected)]
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": "H1.30-open-evo-versus-seed",
        "benchmark": "webshop",
        "arms": ARMS,
        "source": {
            "experiment_git_sha": experiment_git_sha,
            "seed_source_commit": seed_source_commit,
            "catalog": {
                "source": DATA_SOURCE,
                "files": [
                    {"name": name, "bytes": size, "sha256": digest}
                    for name, (size, digest) in sorted(DATA_FILES.items())
                ],
            },
        },
        "controls": {
            "environment_seed": ENVIRONMENT_SEED,
            "selection_seed": int(selection_seed),
            "num_products": NUM_PRODUCTS,
            "max_steps": MAX_STEPS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "history_length": HISTORY_LENGTH,
            "prompt_contract": PROMPT_CONTRACT,
            "action_contract": ACTION_CONTRACT,
            "rollout_seeds": ROLLOUT_SEEDS,
        },
        "index": {"name": "indexes_1k", "materialized_tree": index_identity},
        "evaluator": EVALUATOR,
        "task_count": TASK_COUNT,
        "tasks": tasks,
        "access_policy": {
            "selection_pool_start": MIN_GOAL_INDEX,
            "prior_indices_excluded": sorted(prior_indices),
            "evaluation_outcomes_feed_training": False,
            "teacher_access_during_evaluation": False,
        },
    }
    document["manifest_sha256"] = digest_json(document)
    validate_manifest(document)
    return document


def _without_digest(document: dict[str, Any]) -> dict[str, Any]:
    payload = dict(document)
    payload.pop("manifest_sha256", None)
    return payload


def validate_manifest(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError("H1.30 manifest schema mismatch")
    if document.get("arms") != ARMS or document.get("benchmark") != "webshop":
        raise ValueError("H1.30 arm or benchmark mismatch")
    controls = document.get("controls")
    expected = {
        "environment_seed": ENVIRONMENT_SEED,
        "selection_seed": controls.get("selection_seed") if isinstance(controls, dict) else None,
        "num_products": NUM_PRODUCTS,
        "max_steps": MAX_STEPS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "history_length": HISTORY_LENGTH,
        "prompt_contract": PROMPT_CONTRACT,
        "action_contract": ACTION_CONTRACT,
        "rollout_seeds": ROLLOUT_SEEDS,
    }
    if not isinstance(controls, dict) or any(controls.get(key) != value for key, value in expected.items()):
        raise ValueError("H1.30 controls mismatch")
    tasks = document.get("tasks")
    if document.get("task_count") != TASK_COUNT or not isinstance(tasks, list) or len(tasks) != TASK_COUNT:
        raise ValueError("H1.30 task count mismatch")
    ids: set[str] = set()
    indices: set[int] = set()
    for sample_index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError("H1.30 task row is not an object")
        goal_idx = task.get("goal_idx")
        task_id = task.get("task_id")
        if (
            not isinstance(goal_idx, int)
            or goal_idx < MIN_GOAL_INDEX
            or task_id != f"webshop_{goal_idx:06d}"
            or task.get("sample_index") != sample_index
            or task.get("split") != "h1.30-matched-evaluation"
        ):
            raise ValueError("H1.30 task identity mismatch")
        if task_id in ids or goal_idx in indices:
            raise ValueError("H1.30 task identities are duplicated")
        description = str(task.get("task_description") or "").strip()
        if task.get("task_description_sha256") != sha256_text(description):
            raise ValueError(f"H1.30 task description hash mismatch: {task_id}")
        ids.add(task_id)
        indices.add(goal_idx)
    if not isinstance(document.get("manifest_sha256"), str) or document["manifest_sha256"] != digest_json(_without_digest(document)):
        raise ValueError("H1.30 manifest digest mismatch")
    return document


def write_json(path: Path, document: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--seed-root", type=Path, required=True)
    capture.add_argument("--data-dir", type=Path)
    capture.add_argument("--index-dir", type=Path)
    capture.add_argument("--prior-root", type=Path, action="append", default=[])
    capture.add_argument("--out", type=Path, required=True)
    capture.add_argument("--experiment-git-sha", required=True)
    capture.add_argument("--seed-source-commit", required=True)
    capture.add_argument("--selection-seed", type=int, default=SELECTION_SEED)
    verify = subparsers.add_parser("verify")
    verify.add_argument("manifest", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        document = json.loads(args.manifest.expanduser().resolve().read_text(encoding="utf-8"))
        validate_manifest(document)
        print(f"H1.30_MANIFEST_VERIFY_PASS={document['manifest_sha256']}")
        return 0
    goals, index_identity = live_goals(
        args.seed_root.expanduser().resolve(),
        ENVIRONMENT_SEED,
        data_dir=args.data_dir,
        index_dir=args.index_dir,
    )
    prior_indices = collect_prior_indices(args.prior_root)
    document = build_manifest(
        goals,
        index_identity=index_identity,
        prior_indices=prior_indices,
        experiment_git_sha=args.experiment_git_sha,
        seed_source_commit=args.seed_source_commit,
        selection_seed=args.selection_seed,
    )
    write_json(args.out, document)
    print(f"H1.30_MANIFEST={args.out.expanduser().resolve()}")
    print(f"H1.30_MANIFEST_SHA256={document['manifest_sha256']}")
    print(f"H1.30_TASKS={[task['task_id'] for task in document['tasks']]}")
    print("GPU_WORKLOAD_LAUNCHED=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
