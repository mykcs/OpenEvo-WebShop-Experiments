#!/usr/bin/env python3
"""Convert qualified Qwen2.5-3B train-only successes into an OpenEvo dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def _load_rows(path: Path) -> list[dict[str, Any]]:
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
    cell_roots: list[Path],
    output_dir: Path,
    *,
    train_task_ids: set[str],
    evaluation_task_ids: set[str],
    minimum_success_task_count: int,
    minimum_success_episode_count: int,
) -> dict[str, Any]:
    if train_task_ids & evaluation_task_ids:
        raise ValueError("training/evaluation task IDs overlap")
    episodes: list[dict[str, Any]] = []
    seen_attempts: set[tuple[str, int]] = set()
    source_files: list[str] = []
    for root in cell_roots:
        attempts_path = root / "attempts.jsonl"
        if not attempts_path.is_file():
            raise ValueError(f"missing harvest attempts file: {attempts_path}")
        source_files.append(str(attempts_path))
        for row in _load_rows(attempts_path):
            task_id = str(row.get("task_id") or "")
            seed = row.get("rollout_seed")
            if task_id not in train_task_ids or task_id in evaluation_task_ids:
                raise ValueError(f"harvest row outside train split: {task_id}")
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError(f"missing integer rollout seed for {task_id}")
            key = (task_id, seed)
            if key in seen_attempts:
                raise ValueError(f"duplicate harvest attempt: {key}")
            seen_attempts.add(key)
            if not (
                bool(row.get("identity_ok"))
                and bool(row.get("trace_complete"))
                and not bool(row.get("parser_or_fallback_failure"))
                and not bool(row.get("error"))
                and float(row.get("final_task_score") or 0.0) > 0.0
            ):
                continue
            raw = row.get("episode")
            trace = raw.get("trace") if isinstance(raw, dict) else None
            if not isinstance(trace, list) or not trace:
                raise ValueError(f"qualified harvest row has no trace: {key}")
            traces: list[dict[str, Any]] = []
            for index, step in enumerate(trace):
                if not isinstance(step, dict):
                    raise ValueError(f"trace step is not an object: {key}/{index}")
                prompt = step.get("prompt")
                completion = step.get("raw_completion")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(f"missing prompt in qualified trace: {key}/{index}")
                if not isinstance(completion, str) or not completion.strip():
                    raise ValueError(f"missing completion in qualified trace: {key}/{index}")
                if not bool(step.get("action_valid")) or bool(step.get("used_fallback")) or bool(step.get("parser_failure")):
                    raise ValueError(f"qualified trace contains invalid action: {key}/{index}")
                traces.append({
                    "prompt_messages": [{"role": "user", "content": prompt}],
                    "response_messages": [{"role": "assistant", "content": completion}],
                    "metadata": {
                        "source_step": step.get("step", index),
                        "generation_seed": step.get("generation_seed"),
                        "environment_seed": step.get("environment_seed"),
                        "source_route": "qwen2.5-3b-self-generated-success",
                        "action": step.get("action"),
                    },
                })
            episodes.append({
                "event_id": f"webshop-self-evolution:{task_id}:{seed}:{_sha256(attempts_path)[:16]}",
                "task_id": task_id,
                "session_id": f"{task_id}:{seed}",
                "status": "completed",
                "reward": float(row.get("final_task_score") or 0.0),
                "traces": traces,
                "metadata": {
                    "source_attempt": str(attempts_path),
                    "source_attempts_sha256": _sha256(attempts_path),
                    "rollout_seed": seed,
                    "source_task_id": task_id,
                    "source_route": "qwen2.5-3b-self-generated-success",
                    "qualified_positive": True,
                },
            })

    success_tasks = sorted({str(row["task_id"]) for row in episodes})
    if len(success_tasks) < minimum_success_task_count:
        raise ValueError(f"capability gate failed: {len(success_tasks)} success task identities, need {minimum_success_task_count}")
    if len(episodes) < minimum_success_episode_count:
        raise ValueError(f"capability gate failed: {len(episodes)} success episodes, need {minimum_success_episode_count}")
    episodes.sort(key=lambda row: (row["task_id"], row["session_id"], row["event_id"]))
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    records_bytes = b"".join(_record_json(row) + b"\n" for row in episodes)
    records_path = output_dir / "records.jsonl"
    records_path.write_bytes(records_bytes)
    records_path.chmod(0o600)
    manifest = {
        "name": "seed3090-webshop-h1.29-self-generated-success",
        "schema": "openevo.dataset.v1",
        "source": "qwen2.5-3b-self-generated-success-trajectories",
        "teacher_identity": None,
        "train_task_ids": sorted(train_task_ids),
        "evaluation_task_ids": sorted(evaluation_task_ids),
        "qualified_success_task_ids": success_tasks,
        "qualified_success_episode_count": len(episodes),
        "record_count": len(episodes),
        "trace_count": sum(len(row["traces"]) for row in episodes),
        "records_path": "records.jsonl",
        "records_uri": records_path.resolve().as_uri(),
        "records_byte_size": len(records_bytes),
        "records_sha256": hashlib.sha256(records_bytes).hexdigest(),
        "source_attempt_files": source_files,
        "source_route": "self-evolution; no GPT or external teacher data",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode()
    manifest_path.write_bytes(manifest_bytes)
    manifest_path.chmod(0o600)
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "manifest_file_sha256": _sha256(manifest_path),
        "records": str(records_path),
        "records_sha256": manifest["records_sha256"],
        "record_count": manifest["record_count"],
        "trace_count": manifest["trace_count"],
        "qualified_success_task_ids": success_tasks,
        "qualified_success_episode_count": len(episodes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-root", action="append", type=Path, required=True)
    parser.add_argument("--train-task-id", action="append", required=True)
    parser.add_argument("--evaluation-task-id", action="append", required=True)
    parser.add_argument("--minimum-success-task-count", type=int, default=2)
    parser.add_argument("--minimum-success-episode-count", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_dataset(
        [path.expanduser().resolve() for path in args.cell_root],
        args.output_dir.expanduser().resolve(),
        train_task_ids=set(args.train_task_id),
        evaluation_task_ids=set(args.evaluation_task_id),
        minimum_success_task_count=args.minimum_success_task_count,
        minimum_success_episode_count=args.minimum_success_episode_count,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
