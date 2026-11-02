#!/usr/bin/env python3
"""Label deterministic student-visited recovery states with a train-only oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from scripts.openevo_webshop.webshop_comparison_contract import parse_webshop_action


ORACLE_INSTRUCTION = (
    "\n\nTraining-only oracle instruction: choose one admissible action that best "
    "recovers progress from the current student-visited state. Do not invent "
    "an action outside Allowed actions. Output exactly the two tagged lines "
    "required above; do not add commentary."
)


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


def _load_episode(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = json.loads(path.read_text(encoding="utf-8"))
    raw = source.get("episode") if isinstance(source.get("episode"), dict) else source
    if not isinstance(raw, dict) or not isinstance(raw.get("trace"), list):
        raise ValueError(f"{path}: missing episode.trace")
    return source, raw


def _action_kind(action: str) -> str:
    return action.split("[", 1)[0].lower() if "[" in action else action.lower()


def candidate_states(raw: dict[str, Any], *, max_per_episode: int) -> list[dict[str, Any]]:
    """Select states before deterministic over-navigation/deviation actions."""

    trace = raw["trace"]
    candidates: list[dict[str, Any]] = []
    actions = [str(row.get("action") or "") for row in trace]
    for index, row in enumerate(trace):
        if not isinstance(row, dict) or index == 0:
            continue
        if row.get("used_fallback") or not row.get("action_valid") or row.get("parser_failure"):
            continue
        action = actions[index].lower()
        prior = [value.lower() for value in actions[max(0, index - 2):index]]
        next_repeat = action.startswith("click[next >]") and any(value.startswith("click[next >]") for value in prior)
        repeated = bool(prior) and action == prior[-1]
        backtrack = action.startswith("click[back to search]")
        if not (next_repeat or repeated or backtrack):
            continue
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            continue
        reasons = []
        if next_repeat:
            reasons.append("repeated-next")
        if repeated:
            reasons.append("repeated-action")
        if backtrack:
            reasons.append("backtrack")
        candidates.append({"source_step": index, "reason": "+".join(reasons)})
        if len(candidates) >= max_per_episode:
            break
    return candidates


def _require_gpu(expected_uuid: str) -> dict[str, Any]:
    from scripts.openevo_webshop.parametric_rnd import gpu_snapshot

    visible = [value.strip() for value in __import__("os").environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
    if visible != [expected_uuid]:
        raise ValueError(f"CUDA_VISIBLE_DEVICES must contain exactly {expected_uuid}; observed {visible!r}")
    snapshot = gpu_snapshot(expected_uuid)
    if snapshot["compute_apps"]:
        raise ValueError(f"compute app exists before teacher load: {snapshot['compute_apps']}")
    return snapshot


def label_states(
    episode_paths: list[Path],
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
    max_per_episode: int,
    generation_seed_base: int,
    max_new: int,
) -> dict[str, Any]:
    if train_task_ids & evaluation_task_ids:
        raise ValueError("training/evaluation task IDs overlap")
    from openevo_webshop.inference import make_backend

    pre_gpu = _require_gpu(gpu_uuid)
    backend = make_backend("transformers", str(model_path), device="cuda", revision=None, adapter_path=None, local_files_only=True)
    labels: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    episode_count = 0
    candidate_count = 0
    try:
        for episode_path in sorted(episode_paths):
            source, raw = _load_episode(episode_path)
            task_id = str(source.get("task_id") or raw.get("task_id") or f"webshop-{raw.get('episode')}")
            rollout_seed = source.get("rollout_seed", raw.get("rollout_seed"))
            if not isinstance(rollout_seed, int) or isinstance(rollout_seed, bool):
                raise ValueError(f"{episode_path}: missing integer rollout_seed")
            if task_id in evaluation_task_ids:
                raise ValueError(f"{episode_path}: evaluation task is forbidden: {task_id}")
            if task_id not in train_task_ids:
                raise ValueError(f"{episode_path}: task is not in train_task_ids: {task_id}")
            episode_count += 1
            candidates = candidate_states(raw, max_per_episode=max_per_episode)
            candidate_count += len(candidates)
            for ordinal, candidate in enumerate(candidates):
                step_index = int(candidate["source_step"])
                step = raw["trace"][step_index]
                oracle_prompt = str(step["prompt"]) + ORACLE_INSTRUCTION
                seed = generation_seed_base + len(labels) + len(failures)
                completion = backend.generate(oracle_prompt, max_new=max_new, temperature=0.0, generation_seed=seed)
                action, valid = parse_webshop_action(completion, require_think=True)
                if not valid:
                    failures.append({"task_id": task_id, "rollout_seed": rollout_seed, "source_step": step_index, "reason": "teacher-parser-failure", "completion_sha256": _sha256_bytes(completion.encode())})
                    continue
                labels.append({
                    "task_id": task_id,
                    "rollout_seed": rollout_seed,
                    "source_step": step_index,
                    "candidate_ordinal": ordinal,
                    "candidate_reason": candidate["reason"],
                    "recovery_action": action,
                    "teacher_identity": teacher_identity,
                    "teacher_payload_sha256": teacher_payload_sha256,
                    "teacher_evidence_sha256": teacher_evidence_sha256,
                    "teacher_generation_seed": seed,
                    "teacher_prompt_sha256": _sha256_bytes(oracle_prompt.encode()),
                    "teacher_completion_sha256": _sha256_bytes(completion.encode()),
                    "source_episode": episode_path.name,
                    "source_episode_sha256": _sha256_file(episode_path),
                })
    finally:
        backend.close()
    if not labels:
        raise ValueError(f"no valid teacher labels produced; candidate_count={candidate_count}, failures={len(failures)}")
    labels.sort(key=lambda row: (row["task_id"], row["rollout_seed"], row["source_step"]))
    _atomic_text(output_labels, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in labels))
    summary = {
        "schema": "seed3090.webshop-h1.14-recovery-labels.v1",
        "teacher_identity": teacher_identity,
        "teacher_payload_sha256": teacher_payload_sha256,
        "teacher_evidence_sha256": teacher_evidence_sha256,
        "model_path": str(model_path),
        "gpu_uuid": gpu_uuid,
        "pre_gpu_snapshot": pre_gpu,
        "episode_count": episode_count,
        "candidate_count": candidate_count,
        "valid_label_count": len(labels),
        "teacher_parser_failure_count": len(failures),
        "failures": failures,
        "train_task_ids": sorted(train_task_ids),
        "evaluation_task_ids": sorted(evaluation_task_ids),
        "labels_sha256": _sha256_file(output_labels),
        "state_rule": "repeated-next, repeated-action, or backtrack before the student's action; parser-valid states only",
    }
    _atomic_text(output_summary, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", action="append", type=Path, required=True)
    parser.add_argument("--output-labels", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--teacher-identity", required=True)
    parser.add_argument("--teacher-payload-sha256", required=True)
    parser.add_argument("--teacher-evidence-sha256", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--train-task-id", action="append", required=True)
    parser.add_argument("--evaluation-task-id", action="append", required=True)
    parser.add_argument("--max-per-episode", type=int, default=4)
    parser.add_argument("--generation-seed-base", type=int, default=2026082500)
    parser.add_argument("--max-new", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(label_states(args.episode, args.output_labels, args.output_summary, model_path=args.model_path, teacher_identity=args.teacher_identity, teacher_payload_sha256=args.teacher_payload_sha256, teacher_evidence_sha256=args.teacher_evidence_sha256, gpu_uuid=args.gpu_uuid, train_task_ids=set(args.train_task_id), evaluation_task_ids=set(args.evaluation_task_id), max_per_episode=args.max_per_episode, generation_seed_base=args.generation_seed_base, max_new=args.max_new), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
