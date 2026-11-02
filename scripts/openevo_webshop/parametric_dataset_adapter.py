#!/usr/bin/env python3
"""Convert real WebShop episode JSON into the official OpenEvo dataset contract.

This is a downstream benchmark adapter. It intentionally does not modify the
OpenEvo Core method or invent another artifact schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _digest_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def _record_json(value: Any) -> bytes:
    # OpenEvo's strict dataset reader intentionally uses the standard
    # ``json.dumps(sort_keys=True)`` representation for each JSONL record.
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_record(
    path: Path, *, minimum_reward: float, pair_arm: str | None = None
) -> dict[str, Any]:
    source = json.loads(path.read_text(encoding="utf-8"))
    raw = source
    if pair_arm is not None:
        if pair_arm not in {"before", "after"}:
            raise ValueError(f"unsupported paired rollout arm: {pair_arm}")
        raw = source.get(pair_arm) if isinstance(source, dict) else None
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: paired rollout has no {pair_arm!r} episode")
    elif isinstance(source, dict) and isinstance(source.get("episode"), dict):
        # H0 stores an evidence wrapper around the benchmark episode.  Unwrap
        # it here while retaining the wrapper identity and qualification fields
        # in metadata; the official OpenEvo dataset contract still receives the
        # same prompt/response trace shape.
        raw = dict(source["episode"])
        for key in ("task_id", "rollout_seed", "attempt_id", "qualified_positive"):
            if key in source and key not in raw:
                raw[key] = source[key]
        raw["source_reward"] = source.get("source_reward")
        raw["wrapper_identity_ok"] = source.get("identity_ok")
        raw["wrapper_trace_complete"] = source.get("trace_complete")
        raw["wrapper_parser_or_fallback_failure"] = source.get(
            "parser_or_fallback_failure"
        )
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: episode must be an object")
    reward = raw.get("task_score", raw.get("reward"))
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise ValueError(f"{path}: missing numeric task_score/reward")
    reward = float(reward)
    if not math.isfinite(reward) or reward < minimum_reward:
        raise ValueError(f"{path}: reward {reward} is below minimum {minimum_reward}")
    trace_rows = raw.get("trace")
    if not isinstance(trace_rows, list) or not trace_rows:
        raise ValueError(f"{path}: trace must be a non-empty list")

    traces: list[dict[str, Any]] = []
    for index, step in enumerate(trace_rows):
        if not isinstance(step, dict):
            raise ValueError(f"{path}: trace[{index}] is not an object")
        if step.get("used_fallback"):
            raise ValueError(f"{path}: trace[{index}] used a fallback action")
        prompt = step.get("prompt")
        completion = step.get("raw_completion")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{path}: trace[{index}] has no prompt")
        if not isinstance(completion, str) or not completion.strip():
            raise ValueError(f"{path}: trace[{index}] has no model completion")
        traces.append(
            {
                "prompt_messages": [{"role": "user", "content": prompt}],
                "response_messages": [{"role": "assistant", "content": completion}],
                "metadata": {
                    "source_step": step.get("step", index),
                    "environment_seed": step.get(
                        "environment_seed",
                        source.get("environment_seed") if isinstance(source, dict) else None,
                    ),
                    "generation_seed": step.get("generation_seed"),
                    "action_valid": step.get("action_valid"),
                },
            }
        )

    binding = source.get("task_binding") if isinstance(source, dict) else None
    task_id = raw.get("task_id") or (
        binding.get("task_id") if isinstance(binding, dict) else None
    ) or f"webshop-{raw.get('episode', path.stem)}"
    session_id = raw.get("session_id") or (
        f"{path.stem}:{pair_arm}" if pair_arm is not None else path.stem
    )
    return {
        "event_id": (
            f"webshop-parametric:{path.stem}:{pair_arm or 'episode'}:{_sha256(path)[:16]}"
        ),
        "task_id": str(task_id),
        "session_id": str(session_id),
        "status": "completed",
        "reward": reward,
        "traces": traces,
        "metadata": {
            "source_episode": path.name,
            "source_episode_sha256": _sha256(path),
            "source_pair_arm": pair_arm,
            "source_reward": raw.get("source_reward"),
            "wrapper_identity_ok": raw.get("wrapper_identity_ok"),
            "wrapper_trace_complete": raw.get("wrapper_trace_complete"),
            "wrapper_parser_or_fallback_failure": raw.get(
                "wrapper_parser_or_fallback_failure"
            ),
            "rollout_seed": raw.get(
                "rollout_seed",
                source.get("rollout_seed") if isinstance(source, dict) else None,
            ),
            "rollout_id": raw.get("rollout_id"),
        },
    }


def build_dataset(
    inputs: list[Path],
    output_dir: Path,
    *,
    minimum_reward: float,
    pair_arm: str | None = None,
) -> dict[str, Any]:
    if not inputs:
        raise ValueError("no episode files supplied")
    records: list[dict[str, Any]] = []
    for path in inputs:
        records.append(
            _episode_record(path, minimum_reward=minimum_reward, pair_arm=pair_arm)
        )
    records.sort(key=lambda row: (row["task_id"], row["session_id"], row["event_id"]))
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    records_bytes = b"".join(_record_json(record) + b"\n" for record in records)
    records_path = output_dir / "records.jsonl"
    records_path.write_bytes(records_bytes)
    records_path.chmod(0o600)
    manifest = {
        "name": "seed3090-webshop-parametric-smoke",
        "schema": "openevo.dataset.v1",
        "source": "seed3090-webshop-real-episodes",
        "minimum_reward": minimum_reward,
        "source_pair_arm": pair_arm,
        "record_count": len(records),
        "trace_count": sum(len(row["traces"]) for row in records),
        "records_path": "records.jsonl",
        "records_uri": records_path.resolve().as_uri(),
        "records_byte_size": len(records_bytes),
        "records_sha256": hashlib.sha256(records_bytes).hexdigest(),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False).encode()
    manifest_path.write_bytes(manifest_bytes)
    manifest_path.chmod(0o600)
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(_digest_json(manifest)).hexdigest(),
        "records": str(records_path),
        "records_sha256": manifest["records_sha256"],
        "record_count": manifest["record_count"],
        "trace_count": manifest["trace_count"],
        "minimum_reward": minimum_reward,
        "source_pair_arm": pair_arm,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-reward", type=float, default=0.0)
    parser.add_argument(
        "--paired-rollout-arm",
        choices=("before", "after"),
        help="Select one arm from a paired run_parametric_rollout.py artifact.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_dataset(
                args.episode,
                args.output_dir,
                minimum_reward=args.minimum_reward,
                pair_arm=args.paired_rollout_arm,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
