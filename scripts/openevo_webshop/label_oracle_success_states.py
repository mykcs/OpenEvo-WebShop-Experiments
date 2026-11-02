#!/usr/bin/env python3
"""Label deterministic oracle-path states with the frozen training-only teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.openevo_webshop.label_student_recovery_states import ORACLE_INSTRUCTION
from scripts.openevo_webshop.webshop_comparison_contract import parse_webshop_action


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _load_states(path: Path, *, train_task_ids: set[str], evaluation_task_ids: set[str]) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: state must be an object")
        for key in ("task_id", "rollout_seed", "source_step", "prompt", "oracle_action"):
            if key not in row:
                raise ValueError(f"{path}:{line_number}: missing {key}")
        task_id = str(row["task_id"])
        seed = row["rollout_seed"]
        step = row["source_step"]
        if task_id not in train_task_ids or task_id in evaluation_task_ids:
            raise ValueError(f"{path}:{line_number}: task is outside the training-only set: {task_id}")
        if isinstance(seed, bool) or not isinstance(seed, int) or isinstance(step, bool) or not isinstance(step, int):
            raise ValueError(f"{path}:{line_number}: rollout_seed/source_step must be integers")
        if row.get("source_state") != "deterministic-oracle-success-path":
            raise ValueError(f"{path}:{line_number}: unexpected source_state")
        if not isinstance(row["prompt"], str) or not row["prompt"].strip():
            raise ValueError(f"{path}:{line_number}: prompt is empty")
        key = (task_id, seed, step)
        if key in seen:
            raise ValueError(f"{path}:{line_number}: duplicate state {key}")
        seen.add(key)
        states.append(row)
    states.sort(key=lambda row: (row["task_id"], row["rollout_seed"], row["source_step"]))
    return states


def _require_gpu(expected_uuid: str) -> dict[str, Any]:
    from scripts.openevo_webshop.parametric_rnd import gpu_snapshot

    import os

    visible = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
    if visible != [expected_uuid]:
        raise ValueError(f"CUDA_VISIBLE_DEVICES must contain exactly {expected_uuid}; observed {visible!r}")
    snapshot = gpu_snapshot(expected_uuid)
    if snapshot["compute_apps"]:
        raise ValueError(f"compute app exists before teacher load: {snapshot['compute_apps']}")
    return snapshot


def label_states(
    state_path: Path,
    output_labels: Path,
    output_summary: Path,
    *,
    model_path: Path,
    teacher_identity: str,
    teacher_payload_sha256: str,
    teacher_evidence_sha256: str,
    gpu_uuid: str,
    train_task_ids: set[str],
    evaluation_task_ids: set[str],
    expected_count: int,
    generation_seed_base: int,
    max_new: int,
) -> dict[str, Any]:
    if train_task_ids & evaluation_task_ids:
        raise ValueError("training/evaluation task IDs overlap")
    states = _load_states(state_path, train_task_ids=train_task_ids, evaluation_task_ids=evaluation_task_ids)
    if len(states) != expected_count:
        raise ValueError(f"expected {expected_count} oracle states, found {len(states)}")

    from openevo_webshop.inference import make_backend

    pre_gpu = _require_gpu(gpu_uuid)
    backend = make_backend("transformers", str(model_path), device="cuda", revision=None, adapter_path=None, local_files_only=True)
    labels: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        for index, state in enumerate(states):
            oracle_prompt = str(state["prompt"]) + ORACLE_INSTRUCTION
            generation_seed = generation_seed_base + index
            completion = backend.generate(oracle_prompt, max_new=max_new, temperature=0.0, generation_seed=generation_seed)
            action, valid = parse_webshop_action(completion, require_think=True)
            if not valid:
                failures.append({
                    "task_id": state["task_id"],
                    "rollout_seed": state["rollout_seed"],
                    "source_step": state["source_step"],
                    "reason": "teacher-parser-failure",
                    "completion_sha256": _sha256_bytes(completion.encode()),
                })
                continue
            labels.append({
                "task_id": state["task_id"],
                "rollout_seed": state["rollout_seed"],
                "source_step": state["source_step"],
                "source_state": "deterministic-oracle-success-path",
                "oracle_action": state["oracle_action"],
                "teacher_action": action,
                "teacher_identity": teacher_identity,
                "teacher_payload_sha256": teacher_payload_sha256,
                "teacher_evidence_sha256": teacher_evidence_sha256,
                "teacher_generation_seed": generation_seed,
                "teacher_prompt_sha256": _sha256_bytes(oracle_prompt.encode()),
                "teacher_completion_sha256": _sha256_bytes(completion.encode()),
                "source_states": state_path.name,
                "source_states_sha256": _sha256_file(state_path),
            })
    finally:
        backend.close()

    if failures or len(labels) != expected_count:
        raise ValueError(f"strict success-state label gate failed: labels={len(labels)} failures={len(failures)}")
    labels.sort(key=lambda row: (row["task_id"], row["rollout_seed"], row["source_step"]))
    _atomic_text(output_labels, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in labels))
    summary = {
        "schema": "seed3090.webshop-h1.18-success-state-labels.v1",
        "teacher_identity": teacher_identity,
        "teacher_payload_sha256": teacher_payload_sha256,
        "teacher_evidence_sha256": teacher_evidence_sha256,
        "model_path": str(model_path),
        "gpu_uuid": gpu_uuid,
        "pre_gpu_snapshot": pre_gpu,
        "state_file": state_path.name,
        "state_file_sha256": _sha256_file(state_path),
        "state_count": len(states),
        "valid_label_count": len(labels),
        "teacher_parser_failure_count": len(failures),
        "train_task_ids": sorted(train_task_ids),
        "evaluation_task_ids": sorted(evaluation_task_ids),
        "labels_sha256": _sha256_file(output_labels),
        "source_state": "deterministic-oracle-success-path",
        "evaluation_allowed": False,
    }
    _atomic_text(output_summary, json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output-labels", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--teacher-identity", required=True)
    parser.add_argument("--teacher-payload-sha256", required=True)
    parser.add_argument("--teacher-evidence-sha256", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--train-task-id", action="append", required=True)
    parser.add_argument("--evaluation-task-id", action="append", required=True)
    parser.add_argument("--expected-count", type=int, default=96)
    parser.add_argument("--generation-seed-base", type=int, default=2026082800)
    parser.add_argument("--max-new", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(label_states(args.states, args.output_labels, args.output_summary, model_path=args.model_path, teacher_identity=args.teacher_identity, teacher_payload_sha256=args.teacher_payload_sha256, teacher_evidence_sha256=args.teacher_evidence_sha256, gpu_uuid=args.gpu_uuid, train_task_ids=set(args.train_task_id), evaluation_task_ids=set(args.evaluation_task_id), expected_count=args.expected_count, generation_seed_base=args.generation_seed_base, max_new=args.max_new), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
