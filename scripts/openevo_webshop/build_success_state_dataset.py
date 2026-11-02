#!/usr/bin/env python3
"""Build a provenance-bound success-state control dataset from oracle states."""

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
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


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
    return rows


def build_dataset(
    state_path: Path,
    label_path: Path,
    output_dir: Path,
    *,
    train_task_ids: set[str],
    evaluation_task_ids: set[str],
    expected_count: int,
) -> dict[str, Any]:
    if train_task_ids & evaluation_task_ids:
        raise ValueError("training/evaluation task IDs overlap")
    states = _load_jsonl(state_path)
    labels = _load_jsonl(label_path)
    if len(states) != expected_count or len(labels) != expected_count:
        raise ValueError(f"expected {expected_count} states and labels, found {len(states)} and {len(labels)}")
    state_map: dict[tuple[str, int, int], dict[str, Any]] = {}
    for state in states:
        key = (str(state["task_id"]), int(state["rollout_seed"]), int(state["source_step"]))
        if key in state_map:
            raise ValueError(f"duplicate oracle state {key}")
        if state.get("source_state") != "deterministic-oracle-success-path":
            raise ValueError(f"unexpected source state for {key}")
        if key[0] not in train_task_ids or key[0] in evaluation_task_ids:
            raise ValueError(f"oracle state outside training-only set: {key[0]}")
        state_map[key] = state

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    teacher_identities: set[str] = set()
    for label in labels:
        key = (str(label["task_id"]), int(label["rollout_seed"]), int(label["source_step"]))
        if key in seen or key not in state_map:
            raise ValueError(f"label does not match exactly one state: {key}")
        seen.add(key)
        if label.get("source_state") != "deterministic-oracle-success-path":
            raise ValueError(f"unexpected label source state for {key}")
        action, valid = parse_webshop_action(f"<think>training oracle label</think><action>{label['teacher_action']}</action>")
        if not valid or action != label["teacher_action"]:
            raise ValueError(f"teacher action is not canonical: {key}")
        state = state_map[key]
        teacher_identity = str(label["teacher_identity"])
        teacher_identities.add(teacher_identity)
        event_id = f"webshop-success-state:{key[0]}:{key[1]}:{key[2]}:{_sha256(state_path)[:16]}"
        records.append({
            "event_id": event_id,
            "task_id": key[0],
            "session_id": f"{key[0]}:{key[1]}",
            "status": "completed",
            "reward": 0.0,
            "traces": [{
                "prompt_messages": [{"role": "user", "content": state["prompt"]}],
                "response_messages": [{"role": "assistant", "content": label["teacher_action"]}],
                "metadata": {
                    "source_step": key[2],
                    "state_source": "deterministic-oracle-success-path",
                    "oracle_action": state["oracle_action"],
                    "teacher_identity": teacher_identity,
                    "teacher_label_sha256": _sha256(label_path),
                },
            }],
            "metadata": {
                "source_states": state_path.name,
                "source_states_sha256": _sha256(state_path),
                "rollout_seed": key[1],
                "source_task_id": key[0],
                "state_source": "deterministic-oracle-success-path",
                "teacher_identity": teacher_identity,
            },
        })
    if len(seen) != expected_count:
        raise ValueError(f"label/state key mismatch: matched {len(seen)} of {expected_count}")
    records.sort(key=lambda row: row["event_id"])
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    records_bytes = b"".join(_record_json(record) + b"\n" for record in records)
    records_path = output_dir / "records.jsonl"
    records_path.write_bytes(records_bytes)
    records_path.chmod(0o600)
    manifest = {
        "name": "seed3090-webshop-oracle-success-state-control",
        "schema": "openevo.dataset.v1",
        "source": "seed3090-webshop-deterministic-oracle-success-states",
        "state_source": "deterministic-oracle-success-path",
        "train_task_ids": sorted(train_task_ids),
        "evaluation_task_ids": sorted(evaluation_task_ids),
        "teacher_identities": sorted(teacher_identities),
        "record_count": len(records),
        "trace_count": len(records),
        "records_path": "records.jsonl",
        "records_uri": records_path.resolve().as_uri(),
        "records_byte_size": len(records_bytes),
        "records_sha256": hashlib.sha256(records_bytes).hexdigest(),
        "state_file_sha256": _sha256(state_path),
        "label_file_sha256": _sha256(label_path),
        "teacher_evaluation_allowed": False,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode()
    manifest_path.write_bytes(manifest_bytes)
    manifest_path.chmod(0o600)
    return {"manifest": str(manifest_path), "manifest_sha256": hashlib.sha256(_canonical(manifest)).hexdigest(), **manifest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--label", type=Path, required=True)
    parser.add_argument("--train-task-id", action="append", required=True)
    parser.add_argument("--evaluation-task-id", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=96)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.states, args.label, args.output_dir, train_task_ids=set(args.train_task_id), evaluation_task_ids=set(args.evaluation_task_id), expected_count=args.expected_count), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
