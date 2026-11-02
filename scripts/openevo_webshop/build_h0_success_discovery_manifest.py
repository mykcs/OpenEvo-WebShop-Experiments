#!/usr/bin/env python3
"""Build and verify the preregistered H0.1 Qwen success-discovery manifest.

This manifest is train-only diagnostic material.  It never authorizes an
OpenEvo-vs-SEED evaluation and selects task identities disjoint from the
repository's recorded prior evidence.
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
        collect_prior_indices,
        live_goals,
    )
    from scripts.openevo_webshop.webshop_task_manifest import (
        ACTION_CONTRACT,
        DATA_FILES,
        DATA_SOURCE,
        EVALUATOR,
        HISTORY_LENGTH,
        NUM_PRODUCTS,
        PROMPT_CONTRACT,
        canonical_bytes,
        sha256_text,
    )
except ModuleNotFoundError:
    from build_h1_30_manifest import collect_prior_indices, live_goals  # type: ignore
    from webshop_task_manifest import (  # type: ignore
        ACTION_CONTRACT,
        DATA_FILES,
        DATA_SOURCE,
        EVALUATOR,
        HISTORY_LENGTH,
        NUM_PRODUCTS,
        PROMPT_CONTRACT,
        canonical_bytes,
        sha256_text,
    )


SCHEMA = "openevo.webshop-h0.1-success-discovery.v2"
EXPERIMENT = "H0.1-qwen-success-discovery-v2"
ENVIRONMENT_SEED = 2026
SELECTION_SEED = 2026081701
MIN_GOAL_INDEX = 500
TASK_COUNT = 16
ROLLOUT_SEEDS = [2026081700 + offset for offset in range(8)]
MAX_STEPS = 30
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.4
PROMPT_CONTRACT = "seed3090.webshop-compact-action-prompt.v2"


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def task_row(goal: dict[str, Any], goal_idx: int, sample_index: int) -> dict[str, Any]:
    description = str(goal.get("instruction_text") or "").strip()
    if not description:
        raise ValueError(f"WebShop goal {goal_idx} has no instruction_text")
    return {
        "sample_index": sample_index,
        "task_id": f"webshop_{goal_idx:06d}",
        "goal_idx": goal_idx,
        "split": "h0.1-train-only-success-discovery",
        "task_description": description,
        "task_description_sha256": sha256_text(description),
    }


def _without_digest(document: dict[str, Any]) -> dict[str, Any]:
    payload = dict(document)
    payload.pop("manifest_sha256", None)
    return payload


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
        raise ValueError("could not select disjoint H0.1 task identities")
    tasks = [task_row(goals[idx], idx, sample_index) for sample_index, idx in enumerate(selected)]
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "preregistered-not-authorized",
        "formal_task_consumption_allowed": False,
        "scientific_result": False,
        "experiment": EXPERIMENT,
        "benchmark": "webshop",
        "split": "train-only",
        "arm": "QWEN_BASE_NATURAL_ROLLOUT",
        "scientific_question": "Under the frozen Qwen2.5-3B WebShop contract, how often does native rollout produce a qualified positive trajectory?",
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
            "index": {"name": "indexes_1k", "materialized_tree": index_identity},
        },
        "model": {
            "id": "Qwen/Qwen2.5-3B-Instruct",
            "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
            "teacher": "disabled",
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
            "evaluator": EVALUATOR,
            "rollout_seeds": ROLLOUT_SEEDS,
            "parser_retry": {"enabled": True, "seed_offset": 900000, "max_retries_per_step": 1},
        },
        "budget": {
            "task_count": TASK_COUNT,
            "rollout_seed_count": len(ROLLOUT_SEEDS),
            "attempt_count": TASK_COUNT * len(ROLLOUT_SEEDS),
            "gpu_count": 8,
            "parallelism": "one UUID-bound child per GPU; two waves of eight tasks",
            "extension_allowed": False,
        },
        "qualification": {
            "qualified_positive": "identity_ok and complete_trace and no parser_or_fallback_failure and final_task_score > 0",
            "primary_metrics": ["qualified_positive_count", "qualified_positive_task_count", "qualified_positive_rate"],
            "teacher_trigger": "zero qualified positives after all 128 attempts makes teacher bootstrap eligible; it does not authorize it",
            "pure_followup_trigger": "any qualified positive is preserved for a separately preregistered pure self-evolution condition",
        },
        "task_count": TASK_COUNT,
        "tasks": tasks,
        "access_policy": {
            "selection_pool_start": MIN_GOAL_INDEX,
            "prior_indices_excluded": sorted(prior_indices),
            "evaluation_outcomes_feed_training": False,
            "teacher_access_during_rollout": False,
        },
    }
    document["manifest_sha256"] = digest_json(document)
    validate_manifest(document)
    return document


def validate_manifest(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError("H0.1 success-discovery manifest schema mismatch")
    if document.get("status") != "preregistered-not-authorized":
        raise ValueError("H0.1 manifest must remain preregistered-not-authorized")
    if document.get("formal_task_consumption_allowed") is not False:
        raise ValueError("H0.1 manifest must fail closed")
    controls = document.get("controls")
    if not isinstance(controls, dict) or controls.get("rollout_seeds") != ROLLOUT_SEEDS:
        raise ValueError("H0.1 rollout seed contract mismatch")
    tasks = document.get("tasks")
    if document.get("task_count") != TASK_COUNT or not isinstance(tasks, list) or len(tasks) != TASK_COUNT:
        raise ValueError("H0.1 task count mismatch")
    seen: set[int] = set()
    for sample_index, row in enumerate(tasks):
        if not isinstance(row, dict):
            raise ValueError("H0.1 task row is not an object")
        goal_idx = row.get("goal_idx")
        description = str(row.get("task_description") or "").strip()
        if (
            isinstance(goal_idx, bool)
            or not isinstance(goal_idx, int)
            or goal_idx < MIN_GOAL_INDEX
            or row.get("sample_index") != sample_index
            or row.get("task_id") != f"webshop_{goal_idx:06d}"
            or row.get("split") != "h0.1-train-only-success-discovery"
            or row.get("task_description_sha256") != sha256_text(description)
            or goal_idx in seen
        ):
            raise ValueError("H0.1 task identity or description mismatch")
        seen.add(goal_idx)
    if document.get("manifest_sha256") != digest_json(_without_digest(document)):
        raise ValueError("H0.1 manifest digest mismatch")
    return document


def write_json(path: Path, document: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--seed-root", type=Path, required=True)
    capture.add_argument("--data-dir", type=Path, required=True)
    capture.add_argument("--index-dir", type=Path, required=True)
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
        print(f"H0.1_MANIFEST_VERIFY_PASS={document['manifest_sha256']}")
        return 0
    goals, index_identity = live_goals(
        args.seed_root.expanduser().resolve(),
        ENVIRONMENT_SEED,
        data_dir=args.data_dir.expanduser().resolve(),
        index_dir=args.index_dir.expanduser().resolve(),
    )
    document = build_manifest(
        goals,
        index_identity=index_identity,
        prior_indices=collect_prior_indices(args.prior_root),
        experiment_git_sha=args.experiment_git_sha,
        seed_source_commit=args.seed_source_commit,
        selection_seed=args.selection_seed,
    )
    write_json(args.out, document)
    print(f"H0.1_MANIFEST={args.out.expanduser().resolve()}")
    print(f"H0.1_MANIFEST_SHA256={document['manifest_sha256']}")
    print(f"H0.1_TASKS={[task['task_id'] for task in document['tasks']]}")
    print("GPU_WORKLOAD_LAUNCHED=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
