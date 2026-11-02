#!/usr/bin/env python3
"""Build a draft, method-independent H1.30 train/dev WebShop manifest.

The H1.30 evaluation manifest is frozen separately.  This manifest selects
disjoint train and validation identities from the remaining goal pool so the
OpenEvo harvest and SEED online route can share an auditable task panel.
Generation is CPU/control-plane work and never launches a GPU task.
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
    from scripts.openevo_webshop.build_h1_30_manifest import (
        MIN_GOAL_INDEX,
        PROMPT_CONTRACT,
        validate_manifest as validate_evaluation_manifest,
    )
    from scripts.openevo_webshop.build_h1_30_manifest import live_goals
    from scripts.openevo_webshop.webshop_task_manifest import (
        ACTION_CONTRACT,
        HISTORY_LENGTH,
        canonical_bytes,
    )
except ModuleNotFoundError:
    from build_h1_30_manifest import (  # type: ignore
        MIN_GOAL_INDEX,
        PROMPT_CONTRACT,
        validate_manifest as validate_evaluation_manifest,
    )
    from build_h1_30_manifest import live_goals  # type: ignore
    from webshop_task_manifest import (  # type: ignore
        ACTION_CONTRACT,
        HISTORY_LENGTH,
        canonical_bytes,
    )


SCHEMA = "openevo.webshop-h1.30-training-manifest.v1"
ENVIRONMENT_SEED = 2026
SELECTION_SEED = 20260911
TRAIN_ROLLOUT_SEEDS = [2026091200, 2026091201, 2026091202, 2026091203]
TRAIN_COUNT = 16
DEV_COUNT = 8
MAX_STEPS = 30
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.4


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def task_row(goal: dict[str, Any], goal_idx: int, split: str, sample_index: int) -> dict[str, Any]:
    description = str(goal.get("instruction_text") or "").strip()
    if not description:
        raise ValueError(f"WebShop goal {goal_idx} has no instruction_text")
    return {
        "sample_index": sample_index,
        "task_id": f"webshop_{goal_idx:06d}",
        "goal_idx": goal_idx,
        "split": split,
        "task_description": description,
        "task_description_sha256": hashlib.sha256(description.encode("utf-8")).hexdigest(),
    }


def _goal_indices(rows: Iterable[dict[str, Any]]) -> set[int]:
    result: set[int] = set()
    for row in rows:
        value = row.get("goal_idx")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("training manifest contains a non-integer goal_idx")
        result.add(value)
    return result


def build_manifest(
    goals: list[dict[str, Any]],
    *,
    index_identity: dict[str, Any],
    prior_indices: set[int],
    evaluation_manifest: dict[str, Any],
    experiment_git_sha: str,
    seed_source_commit: str,
    selection_seed: int = SELECTION_SEED,
) -> dict[str, Any]:
    validate_evaluation_manifest(evaluation_manifest)
    evaluation_indices = _goal_indices(evaluation_manifest["tasks"])
    excluded = set(prior_indices) | evaluation_indices
    pool = [idx for idx in range(MIN_GOAL_INDEX, len(goals)) if idx not in excluded]
    random.Random(selection_seed).shuffle(pool)
    selected = pool[: TRAIN_COUNT + DEV_COUNT]
    if len(selected) != TRAIN_COUNT + DEV_COUNT or len(set(selected)) != len(selected):
        raise ValueError("could not select disjoint H1.30 train/dev identities")
    train_indices = selected[:TRAIN_COUNT]
    dev_indices = selected[TRAIN_COUNT:]
    train_tasks = [task_row(goals[idx], idx, "h1.30-training", i) for i, idx in enumerate(train_indices)]
    dev_tasks = [task_row(goals[idx], idx, "h1.30-validation", i) for i, idx in enumerate(dev_indices)]
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "draft-not-authorized",
        "formal_task_consumption_allowed": False,
        "experiment": "H1.30-open-evo-versus-seed",
        "benchmark": "webshop",
        "arms": ["OPEN_EVO", "SEED"],
        "source": {
            "experiment_git_sha": experiment_git_sha,
            "seed_source_commit": seed_source_commit,
            "evaluation_manifest_sha256": evaluation_manifest["manifest_sha256"],
            "index": {"name": "indexes_1k", "materialized_tree": index_identity},
        },
        "controls": {
            "environment_seed": ENVIRONMENT_SEED,
            "selection_seed": int(selection_seed),
            "max_steps": MAX_STEPS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "history_length": HISTORY_LENGTH,
            "prompt_contract": PROMPT_CONTRACT,
            "action_contract": ACTION_CONTRACT,
            "rollout_seeds": TRAIN_ROLLOUT_SEEDS,
        },
        "training_contract": {
            "status": "draft-requires-explicit-scientific-freeze",
            "train_task_count": TRAIN_COUNT,
            "validation_task_count": DEV_COUNT,
            "open_evo": {
                "harvest": "same train task identities and rollout seeds; qualified self-generated traces only",
                "adapter_training": "one bounded run after capability gate; no evaluation outcomes",
            },
            "seed": {
                "route": "manifest-bound on-policy WebShop training",
                "train_batch_size": TRAIN_COUNT,
                "group_size": 8,
                "gpu_count": 8,
                "update_steps": 1,
            },
            "shared_exclusions": [
                "evaluation task identities",
                "prior campaign identities collected by the builder",
                "evaluation outcomes and traces",
                "teacher data unless a separate teacher condition is registered",
            ],
        },
        "train_tasks": train_tasks,
        "validation_tasks": dev_tasks,
        "evaluation_manifest_sha256": evaluation_manifest["manifest_sha256"],
        "selection_policy": {
            "minimum_goal_index": MIN_GOAL_INDEX,
            "excluded_goal_indices": sorted(excluded),
            "train_and_validation_disjoint": True,
            "evaluation_disjoint": True,
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
        raise ValueError("H1.30 training manifest schema mismatch")
    if document.get("status") != "draft-not-authorized":
        raise ValueError("training manifest must remain draft-not-authorized")
    if document.get("formal_task_consumption_allowed") is not False:
        raise ValueError("training manifest must fail closed")
    train = document.get("train_tasks")
    validation = document.get("validation_tasks")
    if not isinstance(train, list) or len(train) != TRAIN_COUNT:
        raise ValueError("H1.30 training task count mismatch")
    if not isinstance(validation, list) or len(validation) != DEV_COUNT:
        raise ValueError("H1.30 validation task count mismatch")
    all_ids: set[str] = set()
    all_indices: set[int] = set()
    for expected_split, rows in (("h1.30-training", train), ("h1.30-validation", validation)):
        for sample_index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError("training manifest task row is not an object")
            goal_idx = row.get("goal_idx")
            task_id = row.get("task_id")
            description = str(row.get("task_description") or "").strip()
            if (
                isinstance(goal_idx, bool)
                or not isinstance(goal_idx, int)
                or goal_idx < MIN_GOAL_INDEX
                or task_id != f"webshop_{goal_idx:06d}"
                or row.get("sample_index") != sample_index
                or row.get("split") != expected_split
                or row.get("task_description_sha256") != hashlib.sha256(description.encode("utf-8")).hexdigest()
            ):
                raise ValueError("H1.30 training task identity or description mismatch")
            if task_id in all_ids or goal_idx in all_indices:
                raise ValueError("H1.30 train/dev task identities are duplicated")
            all_ids.add(task_id)
            all_indices.add(goal_idx)
    if document.get("manifest_sha256") != digest_json(_without_digest(document)):
        raise ValueError("H1.30 training manifest digest mismatch")
    return document


def write_json(path: Path, document: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--seed-root", type=Path, required=True)
    capture.add_argument("--data-dir", type=Path)
    capture.add_argument("--index-dir", type=Path)
    capture.add_argument("--evaluation-manifest", type=Path, required=True)
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
        print(f"H1.30_TRAINING_MANIFEST_VERIFY_PASS={document['manifest_sha256']}")
        return 0

    from scripts.openevo_webshop.build_h1_30_manifest import collect_prior_indices

    seed_root = args.seed_root.expanduser().resolve()
    evaluation_manifest = json.loads(args.evaluation_manifest.expanduser().resolve().read_text(encoding="utf-8"))
    goals, index_identity = live_goals(
        seed_root,
        ENVIRONMENT_SEED,
        data_dir=args.data_dir,
        index_dir=args.index_dir,
    )
    prior_indices = collect_prior_indices(args.prior_root)
    document = build_manifest(
        goals,
        index_identity=index_identity,
        prior_indices=prior_indices,
        evaluation_manifest=evaluation_manifest,
        experiment_git_sha=args.experiment_git_sha,
        seed_source_commit=args.seed_source_commit,
        selection_seed=args.selection_seed,
    )
    write_json(args.out, document)
    print(f"H1.30_TRAINING_MANIFEST={args.out.expanduser().resolve()}")
    print(f"H1.30_TRAINING_MANIFEST_SHA256={document['manifest_sha256']}")
    print(f"H1.30_TRAIN_TASKS={[task['task_id'] for task in document['train_tasks']]}")
    print(f"H1.30_VALIDATION_TASKS={[task['task_id'] for task in document['validation_tasks']]}")
    print("GPU_WORKLOAD_LAUNCHED=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
