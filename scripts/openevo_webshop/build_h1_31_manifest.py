#!/usr/bin/env python3
"""Build and verify the H1.31 teacher-bootstrap task/teacher contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

try:
    from scripts.openevo_webshop.webshop_task_manifest import canonical_bytes
except ModuleNotFoundError:
    from webshop_task_manifest import canonical_bytes  # type: ignore


SCHEMA = "openevo.webshop-h1.31-teacher-bootstrap-manifest.v1"
TRAIN_COUNT = 16
EVALUATION_COUNT = 8
MAX_STEPS = 30
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0
TEACHER_ROLLOUT_SEEDS = [2026110000, 2026110001, 2026110002, 2026110003]
EVALUATION_ROLLOUT_SEEDS = [2026110100, 2026110101, 2026110102, 2026110103]
PROMPT_CONTRACT = "seed3090.webshop-compact-action-prompt.v2"
ACTION_CONTRACT = "seed3090.webshop-action-parser.v1"
EVALUATOR = "seed3090.webshop-task-score-won-evaluator.v1"


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _without_digest(document: dict[str, Any]) -> dict[str, Any]:
    payload = dict(document)
    payload.pop("manifest_sha256", None)
    return payload


def _load(path: Path, schema: str) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != schema:
        raise ValueError(f"expected {schema}, got {document.get('schema')}")
    return document


def _copy_tasks(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for sample_index, row in enumerate(rows):
        task = dict(row)
        task["sample_index"] = sample_index
        task["split"] = split
        copied.append(task)
    return copied


def _validate_tasks(rows: Any, expected_count: int, split: str) -> None:
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(f"{split} task count mismatch")
    indices: set[int] = set()
    task_ids: set[str] = set()
    for sample_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{split} task is not an object")
        goal_idx = row.get("goal_idx")
        task_id = row.get("task_id")
        description = str(row.get("task_description") or "").strip()
        if (
            isinstance(goal_idx, bool)
            or not isinstance(goal_idx, int)
            or task_id != f"webshop_{goal_idx:06d}"
            or row.get("sample_index") != sample_index
            or row.get("split") != split
            or row.get("task_description_sha256") != hashlib.sha256(description.encode()).hexdigest()
        ):
            raise ValueError(f"{split} task identity mismatch at {sample_index}")
        if goal_idx in indices or task_id in task_ids:
            raise ValueError("train/evaluation task identities are duplicated")
        indices.add(goal_idx)
        task_ids.add(task_id)


def build_manifest(
    training_manifest: dict[str, Any],
    evaluation_manifest: dict[str, Any],
    *,
    experiment_git_sha: str,
    seed_source_commit: str,
) -> dict[str, Any]:
    if training_manifest.get("formal_task_consumption_allowed") is not False:
        raise ValueError("source training manifest is not fail-closed")
    if evaluation_manifest.get("formal_task_consumption_allowed", False) is True:
        raise ValueError("evaluation manifest must not already authorize this bootstrap")
    train_tasks = _copy_tasks(training_manifest["train_tasks"], "h1.31-teacher-train")
    evaluation_tasks = _copy_tasks(evaluation_manifest["tasks"], "h1.31-evaluation")
    train_indices = {row["goal_idx"] for row in train_tasks}
    evaluation_indices = {row["goal_idx"] for row in evaluation_tasks}
    if train_indices & evaluation_indices:
        raise ValueError("teacher train and evaluation task identities overlap")
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "draft-not-authorized",
        "formal_task_consumption_allowed": False,
        "experiment": "H1.31-teacher-assisted-openevo-bootstrap",
        "benchmark": "webshop",
        "source": {
            "experiment_git_sha": experiment_git_sha,
            "seed_source_commit": seed_source_commit,
            "training_manifest_sha256": training_manifest["manifest_sha256"],
            "evaluation_manifest_sha256": evaluation_manifest["manifest_sha256"],
            "catalog": "indexes_1k",
        },
        "teacher_contract": {
            "model": "Qwen2.5-14B-Instruct",
            "artifact_payload_sha256": "c8088f27057871992a2bf885fce0f5b3f5db56e1549cd60de9add5f434a89c9f",
            "artifact_config_sha256": "0f2085dbbe2ee251bd6a6a0797d84a6ce34436044d629aa3cba793b43d311a9e",
            "artifact_tokenizer_sha256": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
            "prompt_contract": PROMPT_CONTRACT,
            "action_contract": ACTION_CONTRACT,
            "evaluator": EVALUATOR,
            "rollout_seeds": TEACHER_ROLLOUT_SEEDS,
            "max_steps": MAX_STEPS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "train_only": True,
            "evaluation_access": False,
            "accepted_success": "positive task_score, parser-valid, fallback-free, complete trace, independent audit pending",
        },
        "student_contract": {
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "model_revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
            "prompt_contract": PROMPT_CONTRACT,
            "action_contract": ACTION_CONTRACT,
            "evaluator": EVALUATOR,
            "evaluation_rollout_seeds": EVALUATION_ROLLOUT_SEEDS,
            "max_steps": MAX_STEPS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "teacher_access_during_evaluation": False,
        },
        "train_tasks": train_tasks,
        "evaluation_tasks": evaluation_tasks,
        "counts": {"teacher_train": TRAIN_COUNT, "evaluation": EVALUATION_COUNT},
        "split_policy": {
            "teacher_train_evaluation_disjoint": True,
            "teacher_outputs_training_only": True,
            "h1.29_outputs_excluded": True,
            "evaluation_outcomes_feed_training": False,
        },
    }
    document["manifest_sha256"] = digest_json(document)
    validate_manifest(document)
    return document


def validate_manifest(document: dict[str, Any]) -> dict[str, Any]:
    if document.get("schema") != SCHEMA or document.get("status") != "draft-not-authorized":
        raise ValueError("H1.31 manifest schema/status mismatch")
    if document.get("formal_task_consumption_allowed") is not False:
        raise ValueError("H1.31 manifest is not fail-closed")
    _validate_tasks(document.get("train_tasks"), TRAIN_COUNT, "h1.31-teacher-train")
    _validate_tasks(document.get("evaluation_tasks"), EVALUATION_COUNT, "h1.31-evaluation")
    train = {row["goal_idx"] for row in document["train_tasks"]}
    evaluation = {row["goal_idx"] for row in document["evaluation_tasks"]}
    if train & evaluation:
        raise ValueError("H1.31 train/evaluation overlap")
    teacher = document.get("teacher_contract", {})
    student = document.get("student_contract", {})
    if teacher.get("evaluation_access") is not False or student.get("teacher_access_during_evaluation") is not False:
        raise ValueError("teacher boundary is not fail-closed")
    if document.get("manifest_sha256") != digest_json(_without_digest(document)):
        raise ValueError("H1.31 manifest digest mismatch")
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
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--experiment-git-sha", required=True)
    parser.add_argument("--seed-source-commit", required=True)
    args = parser.parse_args()
    document = build_manifest(
        _load(args.training_manifest, "openevo.webshop-h1.30-training-manifest.v1"),
        _load(args.evaluation_manifest, "openevo.webshop-h1.30-matched-manifest.v1"),
        experiment_git_sha=args.experiment_git_sha,
        seed_source_commit=args.seed_source_commit,
    )
    write_json(args.out, document)
    print(json.dumps({"manifest": str(args.out.resolve()), "manifest_sha256": document["manifest_sha256"], "gpu_workload_launched": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
