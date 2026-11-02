#!/usr/bin/env python3
"""Build a provenance-bound student-state recovery dataset.

The builder accepts student rollout episodes and training-only teacher labels.
It keeps the original state prompt and uses the canonical parsed action as the
target, so H1.14 changes source-state distribution without changing the action
target representation used by its matched successful-state control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.openevo_webshop.webshop_comparison_contract import parse_webshop_action


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _record_json(value: Any) -> bytes:
    # Match OpenEvo's strict worker snapshot representation exactly.
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_payload(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = json.loads(path.read_text(encoding="utf-8"))
    raw = source.get("episode") if isinstance(source.get("episode"), dict) else source
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: episode must be an object")
    return source, raw


def _load_labels(paths: list[Path]) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: label must be an object")
            for key in ("task_id", "rollout_seed", "source_step", "recovery_action", "teacher_identity"):
                if key not in value:
                    raise ValueError(f"{path}:{line_number}: missing {key}")
            action, valid = parse_webshop_action(
                f"<think>training oracle label</think><action>{value['recovery_action']}</action>"
            )
            if not valid or action != value["recovery_action"]:
                raise ValueError(f"{path}:{line_number}: recovery_action is not canonical: {value['recovery_action']!r}")
            value["label_file"] = str(path)
            value["label_file_sha256"] = _sha256(path)
            labels.append(value)
    return labels


def build_dataset(
    episode_paths: list[Path],
    label_paths: list[Path],
    output_dir: Path,
    *,
    train_task_ids: set[str],
    evaluation_task_ids: set[str],
) -> dict[str, Any]:
    if not episode_paths or not label_paths:
        raise ValueError("student episodes and teacher labels are both required")
    overlap = train_task_ids & evaluation_task_ids
    if overlap:
        raise ValueError(f"training/evaluation task overlap: {sorted(overlap)}")

    episodes: dict[tuple[str, int], tuple[Path, dict[str, Any], dict[str, Any]]] = {}
    for path in episode_paths:
        source, raw = _episode_payload(path)
        task_id = str(source.get("task_id") or raw.get("task_id") or f"webshop-{raw.get('episode')}")
        seed = source.get("rollout_seed", raw.get("rollout_seed"))
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"{path}: missing integer rollout_seed")
        if task_id in evaluation_task_ids:
            raise ValueError(f"{path}: student rollout uses evaluation task {task_id}")
        if task_id not in train_task_ids:
            raise ValueError(f"{path}: student rollout task {task_id} is not in train_task_ids")
        key = (task_id, seed)
        if key in episodes:
            raise ValueError(f"duplicate student rollout: {key}")
        trace = raw.get("trace")
        if not isinstance(trace, list) or not trace:
            raise ValueError(f"{path}: trace must be a non-empty list")
        episodes[key] = (path, source, raw)

    labels = _load_labels(label_paths)
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    teacher_identities: set[str] = set()
    for label in labels:
        task_id = str(label["task_id"])
        seed = int(label["rollout_seed"])
        step_index = int(label["source_step"])
        key = (task_id, seed)
        if task_id in evaluation_task_ids:
            raise ValueError(f"label uses evaluation task: {task_id}")
        if key not in episodes:
            raise ValueError(f"label has no matching student rollout: {key}")
        if (task_id, seed, step_index) in seen:
            raise ValueError(f"duplicate recovery label: {(task_id, seed, step_index)}")
        seen.add((task_id, seed, step_index))
        episode_path, source, raw = episodes[key]
        trace = raw["trace"]
        if step_index < 0 or step_index >= len(trace):
            raise ValueError(f"{episode_path}: source_step out of range: {step_index}")
        step = trace[step_index]
        if not isinstance(step, dict) or not isinstance(step.get("prompt"), str) or not step["prompt"].strip():
            raise ValueError(f"{episode_path}: source_step {step_index} has no prompt")
        if step.get("used_fallback") or not step.get("action_valid") or step.get("parser_failure"):
            raise ValueError(f"{episode_path}: source_step {step_index} is not parser-valid")
        teacher_identity = str(label["teacher_identity"])
        teacher_identities.add(teacher_identity)
        event_id = f"webshop-recovery:{task_id}:{seed}:{step_index}:{_sha256(episode_path)[:16]}"
        records.append(
            {
                "event_id": event_id,
                "task_id": task_id,
                "session_id": f"{task_id}:{seed}",
                "status": "completed",
                "reward": 0.0,
                "traces": [
                    {
                        "prompt_messages": [{"role": "user", "content": step["prompt"]}],
                        "response_messages": [{"role": "assistant", "content": label["recovery_action"]}],
                        "metadata": {
                            "source_step": step.get("step", step_index),
                            "state_source": "student-visited-recovery",
                            "teacher_identity": teacher_identity,
                            "teacher_label_sha256": label["label_file_sha256"],
                        },
                    }
                ],
                "metadata": {
                    "source_episode": episode_path.name,
                    "source_episode_sha256": _sha256(episode_path),
                    "rollout_seed": seed,
                    "source_task_id": task_id,
                    "state_source": "student-visited-recovery",
                    "teacher_identity": teacher_identity,
                },
            }
        )

    records.sort(key=lambda row: row["event_id"])
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    records_bytes = b"".join(_record_json(record) + b"\n" for record in records)
    records_path = output_dir / "records.jsonl"
    records_path.write_bytes(records_bytes)
    records_path.chmod(0o600)
    manifest = {
        "name": "seed3090-webshop-student-state-recovery",
        "schema": "openevo.dataset.v1",
        "source": "seed3090-webshop-student-visited-states",
        "state_source": "student-visited-recovery",
        "train_task_ids": sorted(train_task_ids),
        "evaluation_task_ids": sorted(evaluation_task_ids),
        "teacher_identities": sorted(teacher_identities),
        "record_count": len(records),
        "trace_count": len(records),
        "records_path": "records.jsonl",
        "records_uri": records_path.resolve().as_uri(),
        "records_byte_size": len(records_bytes),
        "records_sha256": hashlib.sha256(records_bytes).hexdigest(),
        "source_episode_sha256": sorted({_sha256(path) for path in episode_paths}),
        "label_file_sha256": sorted({_sha256(path) for path in label_paths}),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode()
    manifest_path.write_bytes(manifest_bytes)
    manifest_path.chmod(0o600)
    return {"manifest": str(manifest_path), "manifest_sha256": hashlib.sha256(_canonical(manifest)).hexdigest(), **manifest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", action="append", type=Path, required=True)
    parser.add_argument("--label", action="append", type=Path, required=True)
    parser.add_argument("--train-task-id", action="append", required=True)
    parser.add_argument("--evaluation-task-id", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.episode, args.label, args.output_dir, train_task_ids=set(args.train_task_id), evaluation_task_ids=set(args.evaluation_task_id)), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
