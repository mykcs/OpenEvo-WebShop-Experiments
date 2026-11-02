#!/usr/bin/env python3
"""Audit train-only WebShop supervision content before H1.28 preregistration."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DATASET_ARGUMENTS = {
    "h1.11-positive": "h111_positive",
    "h1.11-negative": "h111_negative",
    "h1.13-positive-action-only": "h113_positive_action_only",
    "h1.18-student-recovery": "h118_student_recovery",
    "h1.18-teacher-action": "h118_teacher_action",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: dataset is empty")
    return rows


def _response_text(trace: dict[str, Any]) -> str:
    messages = trace.get("response_messages")
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError("each trace must contain exactly one response message")
    content = messages[0].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("response message content must be non-empty text")
    return content.strip()


def canonical_action(text: str) -> str:
    lowered = text.strip().lower()
    start = lowered.find("<action>")
    end = lowered.find("</action>", start + len("<action>"))
    if start >= 0 and end >= 0:
        lowered = lowered[start + len("<action>") : end].strip()
    return lowered


def action_kind(action: str) -> str:
    if action.startswith("search["):
        return "search"
    if action == "click[buy now]":
        return "buy_now"
    if action == "click[next >]":
        return "next"
    if action in {"click[< prev]", "click[prev]"}:
        return "previous"
    if action == "click[back to search]":
        return "back_to_search"
    if action.startswith("click["):
        return "other_click"
    return "other"


def summarize_dataset(path: Path) -> dict[str, Any]:
    rows = _load_jsonl(path)
    tasks: Counter[str] = Counter()
    rewards: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    think_count = 0
    trace_count = 0
    positive_records = 0
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id:
            raise ValueError("record is missing task_id")
        tasks[task_id] += 1
        reward = float(row.get("reward", 0.0) or 0.0)
        rewards[format(reward, ".12g")] += 1
        positive_records += int(reward > 0.0)
        traces = row.get("traces")
        if not isinstance(traces, list) or not traces:
            raise ValueError(f"{task_id}: record must have at least one trace")
        for trace in traces:
            response = _response_text(trace)
            action = canonical_action(response)
            actions[action] += 1
            kinds[action_kind(action)] += 1
            think_count += int("<think>" in response and "</think>" in response)
            trace_count += 1
    return {
        "path_basename": path.name,
        "sha256": _sha256(path),
        "record_count": len(rows),
        "trace_count": trace_count,
        "task_count": len(tasks),
        "task_record_counts": dict(sorted(tasks.items())),
        "positive_record_count": positive_records,
        "reward_record_counts": dict(sorted(rewards.items())),
        "think_trace_count": think_count,
        "unique_action_count": len(actions),
        "action_kind_counts": dict(sorted(kinds.items())),
        "top_action_counts": dict(actions.most_common(12)),
    }


def summarize_teacher_oracle_agreement(path: Path) -> dict[str, Any]:
    rows = _load_jsonl(path)
    per_task: dict[str, Counter[str]] = defaultdict(Counter)
    mismatches: Counter[str] = Counter()
    exact = 0
    compared = 0
    for row in rows:
        task_id = str(row.get("task_id") or "")
        traces = row.get("traces")
        if not isinstance(traces, list) or len(traces) != 1:
            raise ValueError("H1.18 teacher-action control must have one trace per record")
        trace = traces[0]
        metadata = trace.get("metadata") or {}
        oracle_raw = metadata.get("oracle_action")
        if not isinstance(oracle_raw, str) or not oracle_raw.strip():
            raise ValueError("H1.18 teacher-action trace is missing oracle_action")
        teacher = canonical_action(_response_text(trace))
        oracle = canonical_action(oracle_raw)
        is_exact = teacher == oracle
        compared += 1
        exact += int(is_exact)
        per_task[task_id]["compared"] += 1
        per_task[task_id]["exact"] += int(is_exact)
        if not is_exact:
            mismatches[f"{action_kind(teacher)}->{action_kind(oracle)}"] += 1
    return {
        "compared": compared,
        "exact": exact,
        "mismatch": compared - exact,
        "exact_rate": exact / compared if compared else None,
        "per_task": {
            task: {
                "compared": counts["compared"],
                "exact": counts["exact"],
                "exact_rate": counts["exact"] / counts["compared"],
            }
            for task, counts in sorted(per_task.items())
        },
        "mismatch_action_kind_counts": dict(sorted(mismatches.items())),
    }


def audit(paths: dict[str, Path]) -> dict[str, Any]:
    missing = set(DATASET_ARGUMENTS) - set(paths)
    if missing:
        raise ValueError(f"missing datasets: {sorted(missing)}")
    datasets = {name: summarize_dataset(paths[name]) for name in DATASET_ARGUMENTS}
    agreement = summarize_teacher_oracle_agreement(paths["h1.18-teacher-action"])
    return {
        "schema": "openevo.webshop-h1.28-experience-content-audit.v1",
        "scope": "train-only historical artifacts; no evaluation outcome was used",
        "datasets": datasets,
        "teacher_oracle_action_agreement": agreement,
        "frozen_candidate_factor": "next-action label fidelity on identical deterministic oracle-success states",
        "matched_h1_28_contrast": {
            "control": "existing H1.18 teacher-action SD-LoRA adapter",
            "treatment": "same 96 prompts/states and optimizer budget with exact metadata.oracle_action targets",
            "unchanged": [
                "training prompts and state identities",
                "training tasks and acquisition seeds",
                "record count and SD-LoRA optimizer budget",
                "base model, evaluation prompt/parser, tasks, seeds, and horizon",
            ],
            "excluded_changes": ["reasoning format", "terminal buy-now state coverage", "rank", "learning rate", "epochs"],
        },
        "claim_boundary": "This audit identifies a supervision-fidelity confound; it does not establish causal transfer or Q3.",
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, destination in DATASET_ARGUMENTS.items():
        parser.add_argument(f"--{flag}", dest=destination, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: getattr(args, destination).expanduser().resolve() for name, destination in DATASET_ARGUMENTS.items()}
    payload = audit(paths)
    _atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps({"output": str(args.output.resolve()), "exact_rate": payload["teacher_oracle_action_agreement"]["exact_rate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
