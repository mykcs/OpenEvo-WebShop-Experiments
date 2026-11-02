#!/usr/bin/env python3
"""Build an OpenEvo dataset whose responses contain parsed actions only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def _record_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path, *, minimum_reward: float) -> dict[str, Any]:
    source = json.loads(path.read_text(encoding="utf-8"))
    raw = source.get("episode") if isinstance(source.get("episode"), dict) else source
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: episode must be an object")
    reward = raw.get("task_score", raw.get("reward"))
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise ValueError(f"{path}: missing numeric reward")
    reward = float(reward)
    if not math.isfinite(reward) or reward < minimum_reward:
        raise ValueError(f"{path}: reward {reward} is below minimum {minimum_reward}")
    traces = []
    for index, step in enumerate(raw.get("trace") or []):
        if not isinstance(step, dict) or step.get("used_fallback") or not step.get("action_valid"):
            raise ValueError(f"{path}: trace[{index}] is not fallback-free and action-valid")
        prompt = step.get("prompt")
        action = step.get("action")
        if not isinstance(prompt, str) or not prompt.strip() or not isinstance(action, str) or not action.strip():
            raise ValueError(f"{path}: trace[{index}] lacks prompt or parsed action")
        traces.append(
            {
                "prompt_messages": [{"role": "user", "content": prompt}],
                "response_messages": [{"role": "assistant", "content": action}],
                "metadata": {"source_step": step.get("step", index), "action_only_projection": True},
            }
        )
    if not traces:
        raise ValueError(f"{path}: trace must be non-empty")
    task_id = raw.get("task_id") or source.get("task_id") or f"webshop-{raw.get('episode', path.stem)}"
    return {
        "event_id": f"webshop-action-only:{path.stem}:{_sha256(path)[:16]}",
        "task_id": str(task_id),
        "session_id": str(raw.get("session_id") or path.stem),
        "status": "completed",
        "reward": reward,
        "traces": traces,
        "metadata": {
            "source_episode": path.name,
            "source_episode_sha256": _sha256(path),
            "action_only_projection": True,
            "rollout_seed": raw.get("rollout_seed", source.get("rollout_seed")),
        },
    }


def build_dataset(inputs: list[Path], output_dir: Path, *, minimum_reward: float) -> dict[str, Any]:
    records = sorted((_record(path, minimum_reward=minimum_reward) for path in inputs), key=lambda row: row["event_id"])
    if not records:
        raise ValueError("no episode files supplied")
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    records_bytes = b"".join(_record_json(record) + b"\n" for record in records)
    records_path = output_dir / "records.jsonl"
    records_path.write_bytes(records_bytes)
    records_path.chmod(0o600)
    manifest = {
        "name": "seed3090-webshop-action-only-projection",
        "schema": "openevo.dataset.v1",
        "source": "seed3090-webshop-real-episodes-action-only-projection",
        "minimum_reward": minimum_reward,
        "action_only_projection": True,
        "record_count": len(records),
        "trace_count": sum(len(record["traces"]) for record in records),
        "records_path": "records.jsonl",
        "records_uri": records_path.resolve().as_uri(),
        "records_byte_size": len(records_bytes),
        "records_sha256": hashlib.sha256(records_bytes).hexdigest(),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_bytes(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False).encode())
    manifest_path.chmod(0o600)
    return {"manifest": str(manifest_path), "manifest_sha256": hashlib.sha256(_canonical(manifest)).hexdigest(), **manifest}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-reward", type=float, default=0.0)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.episode, args.output_dir, minimum_reward=args.minimum_reward), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
