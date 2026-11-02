#!/usr/bin/env python3
"""Materialize the fixed train-only H1.38 diversity-feasibility reservoir.

The builder reads the pinned WebShop catalog only after deriving an exact set of
previously consumed task identities from the committed evidence root.  The
expected count and digest make that discovery set immutable for this campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

try:
    from scripts.openevo_webshop.build_h1_30_manifest import collect_prior_indices
    from scripts.openevo_webshop.webshop_comparison_contract import ACTION_COMMAND_CONTRACT
    from scripts.openevo_webshop.webshop_task_manifest import (
        DATA_FILES,
        DATA_SOURCE,
        EVALUATOR,
        HISTORY_LENGTH,
        NUM_PRODUCTS,
        canonical_bytes,
        live_goals,
        sha256_text,
    )
except ModuleNotFoundError:  # pragma: no cover - script invocation in runtime
    from build_h1_30_manifest import collect_prior_indices  # type: ignore
    from webshop_comparison_contract import ACTION_COMMAND_CONTRACT  # type: ignore
    from webshop_task_manifest import (  # type: ignore
        DATA_FILES,
        DATA_SOURCE,
        EVALUATOR,
        HISTORY_LENGTH,
        NUM_PRODUCTS,
        canonical_bytes,
        live_goals,
        sha256_text,
    )


SCHEMA = "openevo.webshop-h1.38-endogenous-reservoir-manifest.v2"
TASK_COUNT = 64
MIN_GOAL_INDEX = 500
SELECTION_SEED = 2026081804
ROLLOUT_SEEDS = [2026090400, 2026090401, 2026090402, 2026090403]
EXPECTED_PRIOR_COUNT = 381
EXPECTED_PRIOR_DIGEST = "3371a49f4a0a53bddac33fd77f8e5b49265a4add0686e6aa648f476d073e3829"


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _prior_digest(indices: set[int]) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in sorted(indices)).encode()).hexdigest()


def _task_row(goal: dict[str, Any], goal_idx: int, sample_index: int) -> dict[str, Any]:
    description = str(goal.get("instruction_text") or "").strip()
    if not description:
        raise ValueError(f"goal {goal_idx} lacks instruction text")
    return {
        "sample_index": sample_index,
        "task_id": f"webshop_{goal_idx:06d}",
        "goal_idx": goal_idx,
        "split": "h1.38-feasibility-train-only",
        "task_description": description,
        "task_description_sha256": sha256_text(description),
    }


def build_manifest(
    goals: list[dict[str, Any]], *, index_identity: dict[str, Any], prior_indices: set[int], experiment_git_sha: str, seed_source_commit: str, aborted_ledger_sha256: str
) -> dict[str, Any]:
    if len(goals) <= MIN_GOAL_INDEX:
        raise ValueError("WebShop goal catalog is too small")
    if len(prior_indices) != EXPECTED_PRIOR_COUNT or _prior_digest(prior_indices) != EXPECTED_PRIOR_DIGEST:
        raise ValueError("committed prior-manifest task identity set drifted")
    pool = [idx for idx in range(MIN_GOAL_INDEX, len(goals)) if idx not in prior_indices]
    random.Random(SELECTION_SEED).shuffle(pool)
    selected = pool[:TASK_COUNT]
    if len(selected) != TASK_COUNT or len(set(selected)) != TASK_COUNT:
        raise ValueError("could not select 64 unconsumed train-only identities")
    tasks = [_task_row(goals[idx], idx, offset) for offset, idx in enumerate(selected)]
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "preregistered-not-authorized",
        "formal_task_consumption_allowed": False,
        "campaign_id": "20260818-2345-h138-endogenous-reservoir-v2",
        "benchmark": "webshop",
        "source": {
            "experiment_git_sha": experiment_git_sha,
            "seed_source_commit": seed_source_commit,
            "catalog": {"source": DATA_SOURCE, "files": [{"name": name, "bytes": size, "sha256": digest} for name, (size, digest) in sorted(DATA_FILES.items())]},
            "index": {"name": "indexes_1k", "materialized_tree": index_identity},
        },
        "controls": {
            "environment_seed": 2026,
            "selection_seed": SELECTION_SEED,
            "num_products": NUM_PRODUCTS,
            "max_steps": 30,
            "max_new_tokens": 512,
            "temperature": 0.4,
            "history_length": HISTORY_LENGTH,
            "prompt_contract": "seed3090.webshop-compact-action-prompt.v2",
            "action_contract": ACTION_COMMAND_CONTRACT,
            "evaluator": EVALUATOR,
            "rollout_seeds": ROLLOUT_SEEDS,
        },
        "budget": {"task_count": TASK_COUNT, "rollout_seed_count": len(ROLLOUT_SEEDS), "attempt_count": TASK_COUNT * len(ROLLOUT_SEEDS), "extension_allowed": False},
        "access_policy": {
            "selection_pool_start": MIN_GOAL_INDEX,
            "prior_manifest_task_count": len(prior_indices),
            "prior_manifest_task_ids_sha256": _prior_digest(prior_indices),
            "aborted_preprotocol_ledger_sha256": aborted_ledger_sha256,
            "train_only": True,
            "adapter_training": False,
            "heldout_evaluation_access": False,
            "teacher_oracle_seed_access": False,
        },
        "tasks": tasks,
        "manifest_sha256": "",
    }
    document["manifest_sha256"] = _canonical_digest({key: value for key, value in document.items() if key != "manifest_sha256"})
    validate_manifest(document)
    return document


def validate_manifest(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError("unexpected H1.38 feasibility manifest schema")
    if document.get("status") != "preregistered-not-authorized" or document.get("formal_task_consumption_allowed") is not False:
        raise ValueError("feasibility manifest must fail closed before launch")
    controls = document.get("controls")
    tasks = document.get("tasks")
    if not isinstance(controls, dict) or controls.get("rollout_seeds") != ROLLOUT_SEEDS:
        raise ValueError("rollout seed contract mismatch")
    if not isinstance(tasks, list) or len(tasks) != TASK_COUNT:
        raise ValueError("task count mismatch")
    seen: set[int] = set()
    for offset, row in enumerate(tasks):
        if not isinstance(row, dict):
            raise ValueError("malformed task row")
        idx = row.get("goal_idx")
        text = str(row.get("task_description") or "")
        if not isinstance(idx, int) or idx < MIN_GOAL_INDEX or idx in seen or row.get("sample_index") != offset or row.get("task_id") != f"webshop_{idx:06d}" or row.get("split") != "h1.38-feasibility-train-only" or row.get("task_description_sha256") != sha256_text(text):
            raise ValueError("task identity validation failed")
        seen.add(idx)
    expected = _canonical_digest({key: value for key, value in document.items() if key != "manifest_sha256"})
    if document.get("manifest_sha256") != expected:
        raise ValueError("manifest digest mismatch")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--aborted-exclusion-ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--experiment-git-sha", required=True)
    parser.add_argument("--seed-source-commit", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        validate_manifest(json.loads(args.out.read_text(encoding="utf-8")))
        return 0
    ledger_raw = args.aborted_exclusion_ledger.read_bytes()
    ledger = json.loads(ledger_raw.decode("utf-8"))
    extra = ledger.get("task_ids") if isinstance(ledger, dict) else None
    if not isinstance(extra, list) or len(extra) != 64 or len({value for value in extra if isinstance(value, int)}) != 64:
        raise SystemExit("aborted exclusion ledger is malformed")
    goals, index_identity = live_goals(args.seed_root.resolve(), 2026, data_dir=args.data_dir.resolve(), index_dir=args.index_dir.resolve())
    document = build_manifest(goals, index_identity=index_identity, prior_indices=collect_prior_indices([args.prior_root.resolve()]) | set(extra), experiment_git_sha=args.experiment_git_sha, seed_source_commit=args.seed_source_commit, aborted_ledger_sha256=hashlib.sha256(ledger_raw).hexdigest())
    args.out.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest_sha256": document["manifest_sha256"], "tasks": [row["goal_idx"] for row in document["tasks"]]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
