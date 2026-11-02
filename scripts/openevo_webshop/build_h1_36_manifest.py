#!/usr/bin/env python3
"""Materialize the H1.36 strong-base self-evolution manifest from the frozen H1.29 split.

H1.36 changes ONLY the base model (Qwen2.5-3B -> Qwen2.5-7B). To keep the
comparison task-matched, it reuses the frozen H1.29 train/evaluation task
identities verbatim instead of selecting fresh tasks. Task description text is
re-derived from the seed root and cross-checked against the frozen H1.29 public
record's per-task sha256, which proves the reused tasks are content-identical
to the ones H1.29 ran.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from scripts.openevo_webshop.parametric_rnd import ENV_SEED, all_goals, atomic_json, digest_json
except ModuleNotFoundError:  # pragma: no cover - script-style invocation
    from parametric_rnd import ENV_SEED, all_goals, atomic_json, digest_json


# Frozen H1.29 provenance (public record committed under docs/evidence/.../h1.29/).
H129_FROZEN_SCHEMA = "openevo.webshop-h1.29-public-manifest-record.v1"
# sha256 of the committed public-record file itself -- the artifact this builder loads
# and verifies. Distinct from the original H1.29 server-side run manifest below; the two
# were previously conflated into one constant, which made the file check fail-closed.
H129_FROZEN_FILE_SHA256 = "d048893f0555855cd94d9e38573f8d67adab5c57021abf3937766b0e878db90d"
# sha256 of the original H1.29 server-side run manifest (schema
# openevo.webshop-h1.29-self-evolution-manifest.v1). The public record anchors it via its
# server_file_sha256 field; verifying it proves the record points at the real H1.29
# execution manifest rather than a substituted one.
H129_SERVER_FILE_SHA256 = "a7c3fc7a398a5513c38f5420e57b39f80e44fd11386869bcf5401406b7e86249"
H129_CANONICAL_SHA256 = "e42188b12a01d7cc7034fa1cc501fe8238b034fd5f7267c526ee3ce5bde85362"
H129_EXECUTION_GIT_SHA = "290385835f1232fa6853971fe6a5f00ceca43f6c"

UPSTREAM_OPENEVO_STABLE = "796e6248a61112a61c880885844d3b0420c0b6ae"

# Qwen2.5-7B is a pinned read-only local directory, not an HF hub snapshot, so a
# content pin replaces the HF commit hash used for the 3B run.
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_LOCAL_DIR = "/workspace/models/qwen2.5-7b-instruct"
MODEL_CONTENT_PIN = "0c32faf81d8f2f83cf79860d2a48111f166348d6dd38cd348576d8b3a0141c1e"
BASE_MODEL_SCALE_CHANGE = "Qwen/Qwen2.5-3B-Instruct -> Qwen/Qwen2.5-7B-Instruct"

HARVEST_SEEDS = [2026090100, 2026090101, 2026090102, 2026090103]
EVALUATION_SEEDS = [2026090200, 2026090201, 2026090202, 2026090203]
MINIMUM_SUCCESS_TASK_COUNT = 2
MINIMUM_SUCCESS_EPISODE_COUNT = 8

# Errata for the frozen H1.29 public record (file sha H129_FROZEN_FILE_SHA256). That record
# is a derived, committed summary of the H1.29 run. Forensics on 2026-08-18 showed its stored
# per-task hash for goal 1082 is a corrupt 63-character copy ("25ab31a4faf..."), while the
# authoritative execution-time server manifest (sha256 H129_SERVER_FILE_SHA256) records
# "25ab4caf4faf...". The live seed-root description independently hashes to "25ab4caf4faf...",
# confirming the public record's value -- not the seed root and not the executed task -- is the
# defective artifact. The record's own canonical hash still verifies, so the corruption was
# baked in at record-generation time, not a later file edit. The frozen record file is left
# byte-identical (its integrity anchors still prove provenance); only the expected hash used
# for the per-task identity check is corrected to the execution-manifest ground truth. Every
# other task is still verified strictly against the frozen record, so the gate stays
# fail-closed for all 16 tasks.
H129_RECORD_ERRATA = {
    1082: "25ab4caf4faf3707b47f1e3d5024f841b1b1608cccc6cc5997e1fa833c33476f",
}


def _load_frozen(frozen_manifest_path: Path) -> dict[str, Any]:
    raw = frozen_manifest_path.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != H129_FROZEN_FILE_SHA256:
        raise ValueError(
            f"frozen H1.29 record sha256 mismatch: expected {H129_FROZEN_FILE_SHA256}, got {actual_sha}"
        )
    record = json.loads(raw.decode("utf-8"))
    if record.get("schema") != H129_FROZEN_SCHEMA:
        raise ValueError(f"unexpected frozen record schema {record.get('schema')!r}")
    if record.get("canonical_sha256") != H129_CANONICAL_SHA256:
        raise ValueError("frozen H1.29 record canonical sha256 mismatch")
    if record.get("execution_git_sha") != H129_EXECUTION_GIT_SHA:
        raise ValueError("frozen H1.29 record execution git sha mismatch")
    if record.get("server_file_sha256") != H129_SERVER_FILE_SHA256:
        raise ValueError("frozen H1.29 record does not anchor the expected H1.29 server manifest")
    return record


def _verified_description(goals: list[dict[str, Any]], goal_idx: int, expected_sha: str) -> str:
    if goal_idx >= len(goals):
        raise ValueError(f"goal index {goal_idx} is outside the seed-root goal list of {len(goals)}")
    description = str(goals[goal_idx].get("instruction_text") or "").strip()
    if not description:
        raise ValueError(f"WebShop goal {goal_idx} has no instruction_text")
    actual_sha = hashlib.sha256(description.encode()).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(
            f"task description identity mismatch for goal {goal_idx}: seed root sha {actual_sha} != frozen {expected_sha}"
        )
    return description


def build_manifest(seed_root: Path, frozen_manifest_path: Path, *, experiment_sha: str) -> dict[str, Any]:
    record = _load_frozen(frozen_manifest_path.expanduser().resolve())
    goals = all_goals(seed_root.expanduser().resolve())

    train_indices = [int(idx) for idx in record["train_tasks"]]
    evaluation_indices = [int(idx) for idx in record["evaluation_tasks"]]
    if set(train_indices) & set(evaluation_indices):
        raise ValueError("frozen H1.29 train and evaluation tasks overlap")
    expected_sha = {int(row["goal_idx"]): str(row["task_description_sha256"]) for row in record["task_rows"]}
    expected_split = {int(row["goal_idx"]): str(row["split"]) for row in record["task_rows"]}
    # Apply the documented record errata: correct the defective per-task hashes to the
    # authoritative execution-manifest values (see H129_RECORD_ERRATA). The frozen record file
    # is untouched; this only fixes the expected value used for the live identity check.
    for goal_idx, authoritative_sha in H129_RECORD_ERRATA.items():
        expected_sha[goal_idx] = authoritative_sha

    tasks: list[dict[str, Any]] = []
    for sample_index, goal_idx in enumerate(train_indices):
        description = _verified_description(goals, goal_idx, expected_sha[goal_idx])
        tasks.append({
            "sample_index": sample_index,
            "task_id": f"webshop_{goal_idx:06d}",
            "goal_idx": goal_idx,
            "partition": f"T{sample_index + 1}",
            "split": expected_split[goal_idx],
            "task_description": description,
            "task_description_sha256": expected_sha[goal_idx],
        })
    for offset, goal_idx in enumerate(evaluation_indices):
        description = _verified_description(goals, goal_idx, expected_sha[goal_idx])
        tasks.append({
            "sample_index": len(tasks),
            "task_id": f"webshop_{goal_idx:06d}",
            "goal_idx": goal_idx,
            "partition": f"P{offset // 2 + 1}",
            "split": expected_split[goal_idx],
            "task_description": description,
            "task_description_sha256": expected_sha[goal_idx],
        })

    payload: dict[str, Any] = {
        "schema": "openevo.webshop-h1.36-strong-base-self-evolution-manifest.v1",
        "run_kind": "phase_h1.36_strong_base_self_evolution",
        "source": {
            "experiment_git_sha": experiment_sha,
            "official_open_evo_stable": UPSTREAM_OPENEVO_STABLE,
            "model": MODEL_ID,
            "model_local_dir": MODEL_LOCAL_DIR,
            "model_revision": MODEL_CONTENT_PIN,
            "model_revision_kind": "local-content-pin",
            "base_model_scale_change": BASE_MODEL_SCALE_CHANGE,
            "reused_from_h1_29": {
                "execution_git_sha": H129_EXECUTION_GIT_SHA,
                "frozen_file_sha256": H129_FROZEN_FILE_SHA256,
                "server_file_sha256": H129_SERVER_FILE_SHA256,
                "canonical_sha256": H129_CANONICAL_SHA256,
                "record_errata": {
                    "corrected_goal_indices": sorted(H129_RECORD_ERRATA),
                    "applied": {str(idx): sha for idx, sha in H129_RECORD_ERRATA.items()},
                    "reason": "the frozen H1.29 public record stores a corrupt 63-char hash for these goals; corrected to the authoritative execution-time server-manifest value (server_file_sha256), which the live seed root reproduces. The frozen record file is left byte-identical.",
                },
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
            "prompt_contract": "seed3090.webshop-compact-action-prompt.v2",
            "action_contract": "seed3090.webshop-action-parser.v1",
        },
        "train_tasks": train_indices,
        "evaluation_tasks": evaluation_indices,
        "tasks": tasks,
        "task_selection": {
            "provenance": "reused-frozen-h1.29-manifest (not a fresh selection) for a task-matched 7B-versus-3B capability comparison",
            "frozen_source_sha256": H129_FROZEN_FILE_SHA256,
            "goal_count": len(goals),
            "evaluation_is_disjoint_from_train": not set(train_indices) & set(evaluation_indices),
            "description_identity": "every task_description re-derived from the seed root and verified against the frozen record sha256",
            "split": "train-only-self-evolution versus reused-self-evolution-evaluation",
        },
        "harvest": {
            "tasks": train_indices,
            "rollout_seeds": HARVEST_SEEDS,
            "attempts": len(train_indices) * len(HARVEST_SEEDS),
            "qualification": "identity_ok and trace_complete and no parser/fallback/error and final_task_score > 0",
            "minimum_success_task_count": MINIMUM_SUCCESS_TASK_COUNT,
            "minimum_success_episode_count": MINIMUM_SUCCESS_EPISODE_COUNT,
        },
        "evaluation": {
            "tasks": evaluation_indices,
            "rollout_seeds": EVALUATION_SEEDS,
            "arms": ["base", "self-evolution"],
            "cells": len(evaluation_indices) * 2,
            "attempts": len(evaluation_indices) * len(EVALUATION_SEEDS) * 2,
            "task_role": "reused held-out evaluation for task-matched 7B-versus-3B comparison",
        },
        "contamination": {
            "train_evaluation_overlap": False,
            "evaluation_outcomes_used_for_training": False,
            "teacher_oracle_gpt_used": False,
            "h1_29_3b_trajectories_used": False,
        },
        "manifest_sha256": "",
    }
    payload["manifest_sha256"] = digest_json(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--experiment-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_manifest(
        args.seed_root,
        args.frozen_manifest,
        experiment_sha=args.experiment_sha,
    )
    output = args.output.expanduser().resolve()
    atomic_json(output, payload)
    print(json.dumps({
        "manifest": str(output),
        "manifest_sha256": payload["manifest_sha256"],
        "manifest_file_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "train_tasks": payload["train_tasks"],
        "evaluation_tasks": payload["evaluation_tasks"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
