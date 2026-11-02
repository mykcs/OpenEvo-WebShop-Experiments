#!/usr/bin/env python3
"""Build a deterministic, identity-only Q17 dev task schedule for E5.

This adapter is a preparation artifact. It never loads a model, starts WebShop,
consumes a task, or reads the final-panel manifest. Its JSON output contains no
task instructions or answers. A future execution wrapper must consume this
schedule and bind its own source/runtime SHA before it can be activated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_REL = Path(
    "docs/evidence/q17-validation-curve-20260923/"
    "coarse-manifest-q17-directapply-validation-curve-202609231600.json"
)
DESIGN_REL = Path("configs/experiment/designs/post-advisor-unanswered-202609282338.json")
EXPECTED_MANIFEST_SHA256 = "2d5d1620065a15840ae20fed324038312aec74b0895e13c9d283177928bf1a3b"
EXPECTED_PANEL_SHA256 = "4033523c01697576b7f9ce8cddc3a63cf6c1514ef5554e01a8c82b59ee62d478"
EXPECTED_E5_BUDGET = {
    "train_data_size": 16,
    "train_batch_size": 16,
    "group_size": 8,
    "updates": 160,
    "rollout_slots": 20480,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def build_schedule(repo_root: Path = ROOT) -> dict[str, Any]:
    manifest_path = repo_root / MANIFEST_REL
    design_path = repo_root / DESIGN_REL
    actual_sha = _sha256(manifest_path)
    if actual_sha != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"Q17 manifest SHA mismatch: {actual_sha}")

    manifest = _load_json(manifest_path)
    design = _load_json(design_path)
    q17 = design["global_boundaries"]["development_panel"]
    if q17 != {"size": 64, "sha256": EXPECTED_PANEL_SHA256,
               "role": "development/held-out only; not final panel"}:
        raise ValueError("frozen campaign dev-panel identity or role changed")
    if design["global_boundaries"]["final_panel_access"] != 0:
        raise ValueError("frozen campaign no longer locks final-panel access")

    e5 = next(x for x in design["experiments"] if x["id"] == "E5_FULL_SEED_STYLE_STAGE2")
    budget = e5["stage2_budget"]
    budget_tokens = ("TRAIN_DATA_SIZE=16", "data.train_batch_size=16", "GROUP_SIZE=8",
                     "trainer.total_epochs=160", "20480 rollout slots")
    if not all(token in budget for token in budget_tokens):
        raise ValueError("frozen E5 training budget changed")
    if "final panel remains locked" not in e5["evaluation"]:
        raise ValueError("frozen E5 evaluation no longer locks final panel")
    if manifest.get("panel_size") != 64 or manifest.get("panel_sha256") != EXPECTED_PANEL_SHA256:
        raise ValueError("Q17 manifest panel identity mismatch")
    if manifest.get("final_panel_access_allowed") is not False:
        raise ValueError("Q17 manifest does not prohibit final-panel access")

    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 64:
        raise ValueError("Q17 manifest must contain exactly 64 tasks")
    ordinals = [row.get("probe_ordinal") for row in tasks]
    goal_idxs = [row.get("goal_idx") for row in tasks]
    task_ids = [row.get("task_identity") for row in tasks]
    sessions = [row.get("source_session_index") for row in tasks]
    instruction_hashes = [row.get("instruction_sha256") for row in tasks]
    generation_seeds = [row.get("paired_generation_seed") for row in tasks]
    if ordinals != list(range(64)):
        raise ValueError("Q17 probe order is not exactly 0..63")
    for name, values in (("goal_idx", goal_idxs), ("task_identity", task_ids),
                         ("source_session_index", sessions),
                         ("instruction_sha256", instruction_hashes),
                         ("paired_generation_seed", generation_seeds)):
        if any(value is None for value in values) or len(set(values)) != 64:
            raise ValueError(f"Q17 {name} must contain 64 unique values")

    benchmark = manifest.get("benchmark", {})
    heldout = benchmark.get("held_out_pool")
    train_pool_start = benchmark.get("train_pool_start")
    if heldout != [0, 499] or train_pool_start != 500:
        raise ValueError("Q17 held-out/train pool boundary changed")
    if any(not isinstance(index, int) or not heldout[0] <= index <= heldout[1]
           for index in goal_idxs):
        raise ValueError("Q17 dev goal index is outside the frozen held-out pool")
    if any(index >= train_pool_start for index in goal_idxs):
        raise ValueError("Q17 dev task overlaps the frozen training goal pool")
    if goal_idxs != sessions:
        raise ValueError("Q17 session indices must exactly match the frozen goal indices")
    if set(row["worker_seed"] for row in tasks) != {0}:
        raise ValueError("pinned SEED adapter only accepts the frozen Q17 worker_seed=0 panel")

    exclusions = manifest.get("exclusions", {})
    required_zero = ("training_goal_overlap", "training_instruction_overlap",
                     "previous_final_goal_overlap", "current_final_goal_overlap",
                     "prior_diagnostic_instruction_overlap")
    if any(exclusions.get(key) != 0 for key in required_zero):
        raise ValueError("Q17 manifest overlap exclusion is not zero")

    # Omit instruction, answer-like content, ASIN, and category; the environment
    # resolves the frozen goal_idx against the pinned WebShop data at execution.
    schedule = [
        {
            "probe_ordinal": row["probe_ordinal"],
            "goal_idx": row["goal_idx"],
            "task_identity": row["task_identity"],
            "source_session_index": row["source_session_index"],
            "instruction_sha256": row["instruction_sha256"],
            "worker_seed": row["worker_seed"],
            "paired_generation_seed": row["paired_generation_seed"],
        }
        for row in tasks
    ]
    return {
        "schema": "openevo.post-advisor-e5-q17-dev-schedule.v1",
        "status": "PREPARATION_ONLY_NOT_EXECUTABLE",
        "source_manifest": str(MANIFEST_REL),
        "source_manifest_sha256": actual_sha,
        "panel_sha256": EXPECTED_PANEL_SHA256,
        "panel_size": 64,
        "role": "development/held-out only",
        "task_order_policy": "frozen probe_ordinal 0..63; no shuffle or replacement",
        "goal_pool": {"dev_inclusive": [0, 499], "training_starts_at": 500},
        "training_budget": EXPECTED_E5_BUDGET,
        "external_teacher_calls": 0,
        "seed_local_policy_analysis": "preserve pinned SEED_ENABLE_ANALYSIS=True, policy_vllm",
        "final_panel_access": 0,
        "runtime_consumer_bound": False,
        "tasks": schedule,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        help="write the identity-only schedule JSON (no model/task execution)")
    args = parser.parse_args()
    schedule = build_schedule()
    encoded = json.dumps(schedule, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
        print(json.dumps({"status": schedule["status"], "output": str(args.output),
                          "task_count": len(schedule["tasks"]),
                          "runtime_consumer_bound": False}, sort_keys=True))
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
