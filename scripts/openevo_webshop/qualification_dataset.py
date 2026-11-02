#!/usr/bin/env python3
"""Build the deterministic synthetic dataset used by the native trainer gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "openevo.synthetic-trainer-qualification-dataset.v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _record(index: int, query: str) -> dict[str, Any]:
    event = f"fast-start-trainer-qualification-{index}"
    prompt = (
        "You are an autonomous agent operating in WebShop.\n"
        "This is synthetic engineering qualification data, not a benchmark task.\n"
        f"Synthetic request: {query}.\n"
        "Output exactly <think>...</think> then <action>search[query]</action>."
    )
    response = f"<think>qualification</think>\n<action>search[{query}]</action>"
    return {
        "event_id": f"qualification:{event}",
        "metadata": {
            "qualification_only": True,
            "synthetic": True,
            "formal_schedule_position": None,
            "formal_panel_member": False,
            "wrapper_identity_ok": True,
            "wrapper_parser_or_fallback_failure": False,
            "wrapper_trace_complete": True,
        },
        "reward": 1.0,
        "session_id": event,
        "status": "completed",
        "task_id": event,
        "traces": [
            {
                "metadata": {
                    "action_valid": True,
                    "environment_seed": None,
                    "generation_seed": 990000 + index,
                    "source_step": 0,
                    "qualification_only": True,
                },
                "prompt_messages": [{"role": "user", "content": prompt}],
                "response_messages": [{"role": "assistant", "content": response}],
            }
        ],
    }


def canonical_records() -> list[dict[str, Any]]:
    return [
        _record(0, "red cotton shirt"),
        _record(1, "blue running shoes"),
    ]


def build(root: Path) -> dict[str, Any]:
    root = root.resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    records_path = root / "records.jsonl"
    manifest_path = root / "manifest.json"
    records = canonical_records()
    encoded = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in records
    )
    records_path.write_bytes(encoded)
    records_path.chmod(0o600)
    manifest = {
        "schema": "openevo.dataset.v1",
        "name": "openevo-fast-start-synthetic-trainer-qualification",
        "source": SCHEMA,
        "qualification_only": True,
        "synthetic": True,
        "formal_schedule_positions_consumed": 0,
        "formal_panel_members": 0,
        "record_count": len(records),
        "trace_count": sum(len(row["traces"]) for row in records),
        "minimum_reward": 1.0,
        "records_path": records_path.name,
        "records_uri": records_path.as_uri(),
        "records_byte_size": len(encoded),
        "records_sha256": _sha256_bytes(encoded),
    }
    manifest_identity = {key: value for key, value in manifest.items() if key != "records_uri"}
    dataset_content_sha256 = _sha256_bytes(
        json.dumps(manifest_identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\0"
        + encoded
    )
    # Frozen OpenEvo strict snapshots require byte-for-byte canonical manifest
    # serialization: pretty JSON with sorted keys and no trailing newline.
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    manifest_path.chmod(0o600)
    return {
        "schema": SCHEMA,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_bytes(manifest_path.read_bytes()),
        "dataset_content_sha256": dataset_content_sha256,
        "records": str(records_path),
        "records_sha256": manifest["records_sha256"],
        "record_count": len(records),
        "formal_schedule_positions_consumed": 0,
        "formal_panel_members": 0,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
