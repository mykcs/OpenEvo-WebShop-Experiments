#!/usr/bin/env python3
"""Build the matched H1.28 oracle-action treatment from H1.18 records."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from scripts.openevo_webshop.audit_h128_experience_content import canonical_action


QUOTED_ACTION = re.compile(r"'((?:search|click)\[[^'\n]+\])'", re.IGNORECASE)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def _single_message(trace: dict[str, Any], key: str) -> dict[str, Any]:
    messages = trace.get(key)
    if not isinstance(messages, list) or len(messages) != 1 or not isinstance(messages[0], dict):
        raise ValueError(f"each trace must have exactly one {key} object")
    return messages[0]


def _oracle_is_admissible(prompt: str, oracle_action: str) -> bool:
    allowed = {canonical_action(value) for value in QUOTED_ACTION.findall(prompt)}
    if oracle_action.startswith("search["):
        return "search[<your query>]" in allowed or oracle_action in allowed
    return oracle_action in allowed


def build_dataset(source_path: Path, output_dir: Path, *, expected_count: int = 96) -> dict[str, Any]:
    source_rows = _load_jsonl(source_path)
    if len(source_rows) != expected_count:
        raise ValueError(f"expected {expected_count} source records, found {len(source_rows)}")
    source_sha = _sha256_bytes(source_path.read_bytes())
    records: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    prompt_hashes: list[str] = []
    changed = 0
    teacher_oracle_exact = 0
    for source in source_rows:
        event_id = str(source.get("event_id") or "")
        if not event_id or event_id in event_ids:
            raise ValueError(f"missing or duplicate event_id: {event_id!r}")
        event_ids.add(event_id)
        traces = source.get("traces")
        if not isinstance(traces, list) or len(traces) != 1:
            raise ValueError(f"{event_id}: expected exactly one trace")
        source_trace = traces[0]
        prompt_message = _single_message(source_trace, "prompt_messages")
        response_message = _single_message(source_trace, "response_messages")
        prompt = prompt_message.get("content")
        teacher_raw = response_message.get("content")
        metadata = source_trace.get("metadata")
        if not isinstance(prompt, str) or not isinstance(teacher_raw, str) or not isinstance(metadata, dict):
            raise ValueError(f"{event_id}: malformed prompt, response, or metadata")
        oracle_raw = metadata.get("oracle_action")
        if not isinstance(oracle_raw, str) or not oracle_raw.strip():
            raise ValueError(f"{event_id}: missing metadata.oracle_action")
        teacher_action = canonical_action(teacher_raw)
        oracle_action = canonical_action(oracle_raw)
        if not (oracle_action.startswith("search[") or oracle_action.startswith("click[")):
            raise ValueError(f"{event_id}: non-canonical oracle action {oracle_action!r}")
        if not _oracle_is_admissible(prompt, oracle_action):
            raise ValueError(f"{event_id}: oracle action is absent from the prompt's allowed actions")
        teacher_oracle_exact += int(teacher_action == oracle_action)
        changed += int(teacher_action != oracle_action)
        record = copy.deepcopy(source)
        trace = record["traces"][0]
        trace["response_messages"][0]["content"] = oracle_action
        trace["metadata"]["source_teacher_action"] = teacher_action
        trace["metadata"]["supervision_target_source"] = "deterministic-oracle-success-path"
        record.setdefault("metadata", {})["h1_28_variant"] = "oracle-action"
        record["metadata"]["source_teacher_action_records_sha256"] = source_sha
        records.append(record)
        prompt_hashes.append(_sha256_bytes(prompt.encode()))

    records.sort(key=lambda row: str(row["event_id"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    records_bytes = b"".join(
        json.dumps(row, sort_keys=True, allow_nan=False).encode() + b"\n"
        for row in records
    )
    records_path = output_dir / "records.jsonl"
    temporary = records_path.with_name(records_path.name + ".tmp")
    temporary.write_bytes(records_bytes)
    temporary.replace(records_path)
    manifest = {
        "schema": "openevo.webshop-h1.28-oracle-action-dataset.v1",
        "name": "h1.28-matched-oracle-action-supervision",
        "source_records_sha256": source_sha,
        "records_path": "records.jsonl",
        "records_uri": records_path.resolve().as_uri(),
        "records_byte_size": len(records_bytes),
        "records_sha256": _sha256_bytes(records_bytes),
        "record_count": len(records),
        "trace_count": len(records),
        "unique_event_count": len(event_ids),
        "unique_prompt_count": len(set(prompt_hashes)),
        "prompt_sequence_sha256": _sha256_bytes("\n".join(prompt_hashes).encode()),
        "teacher_oracle_exact_count": teacher_oracle_exact,
        "changed_target_count": changed,
        "independent_variable": "next-action target: teacher action versus exact deterministic-oracle action",
        "teacher_evaluation_allowed": False,
        "terminal_state_coverage_changed": False,
    }
    # OpenEvo's strict snapshot reader compares the manifest bytes with this
    # exact canonical JSON serialization (which intentionally has no newline).
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode()
    manifest_path = output_dir / "manifest.json"
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary_manifest.write_bytes(manifest_bytes)
    temporary_manifest.replace(manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=96)
    args = parser.parse_args()
    result = build_dataset(
        args.source_records.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        expected_count=args.expected_count,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
