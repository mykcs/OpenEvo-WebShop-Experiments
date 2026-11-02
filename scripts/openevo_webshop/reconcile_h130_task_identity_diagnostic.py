#!/usr/bin/env python3
"""Reconcile the preregistered H1.30 task-identity diagnostic."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from scripts.openevo_webshop.summarize_h130_failure_taxonomy import _load, classify
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from summarize_h130_failure_taxonomy import _load, classify


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _score(row: dict[str, Any]) -> float | None:
    value = row.get("final_task_score")
    if value is None and isinstance(row.get("episode"), dict):
        value = row["episode"].get("task_score")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    categories = collections.Counter(classify(row) for row in rows)
    valid = [row for row in rows if not row.get("parser_or_fallback_failure") and not row.get("error")]
    scores = [score for score in (_score(row) for row in valid) if score is not None]
    return {
        "attempts": len(rows),
        "valid_attempts": len(valid),
        "parser_or_fallback_failures": sum(bool(row.get("parser_or_fallback_failure")) for row in rows),
        "parser_repair_count": sum(bool(row.get("parser_repair_used")) for row in rows),
        "parser_retry_count": sum(bool(row.get("parser_retry_used")) for row in rows),
        "runtime_errors": sum(bool(row.get("error")) for row in rows),
        "qualified_positive_count": sum(bool(row.get("qualified_positive")) for row in rows),
        "qualified_positive_task_count": len({str(row.get("task_id")) for row in rows if row.get("qualified_positive")}),
        "valid_mean_task_score": sum(scores) / len(scores) if scores else None,
        "categories": dict(sorted(categories.items())),
    }


def reconcile(manifest_path: Path, run_root: Path, output: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups = manifest.get("diagnostic_groups")
    if not isinstance(groups, dict) or set(groups) != {"prior_positive", "v3_nonpositive"}:
        raise ValueError("manifest must contain exactly the preregistered diagnostic groups")
    group_by_task = {task_id: name for name, task_ids in groups.items() for task_id in task_ids}
    if len(group_by_task) != 16 or any(len(task_ids) != 8 for task_ids in groups.values()):
        raise ValueError("diagnostic groups must contain 8 disjoint task IDs each")

    attempt_paths = sorted(run_root.glob("harvest/**/attempts.jsonl"))
    if len(attempt_paths) != 16:
        raise ValueError(f"expected 16 attempt files, found {len(attempt_paths)}")
    rows: list[dict[str, Any]] = []
    source_files = []
    for path in attempt_paths:
        source_files.append({"path": str(path.resolve()), "sha256": _sha256(path)})
        rows.extend(_load(path))
    if len(rows) != 64:
        raise ValueError(f"expected 64 attempts, found {len(rows)}")
    counts = collections.Counter(str(row.get("task_id")) for row in rows)
    if set(counts) != set(group_by_task) or any(count != 4 for count in counts.values()):
        raise ValueError(f"each preregistered task must have exactly four attempts: {dict(counts)}")

    group_rows = {
        name: [row for row in rows if group_by_task.get(str(row.get("task_id"))) == name]
        for name in groups
    }
    receipt_path = run_root / "evidence" / "h1.30-open-evo-capability-gate.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else None
    result = {
        "schema": "openevo.webshop-h1.30-task-identity-diagnostic-reconciliation.v1",
        "status": "pass",
        "diagnostic_only": True,
        "formal_task_consumption_allowed": False,
        "scientific_result": True,
        "generic_capability_gate_status": receipt.get("status") if receipt else "not-found",
        "manifest": {
            "path": str(manifest_path.resolve()),
            "file_sha256": _sha256(manifest_path),
            "manifest_sha256": manifest.get("manifest_sha256"),
        },
        "contract": manifest.get("contract"),
        "groups": {name: {"task_ids": groups[name], **_metrics(group_rows[name])} for name in sorted(groups)},
        "overall": _metrics(rows),
        "protocol_fingerprints": sorted({str(row.get("protocol_fingerprint")) for row in rows if row.get("protocol_fingerprint")}),
        "source_files": source_files,
        "launcher_receipt": {
            "path": str(receipt_path.resolve()),
            "sha256": _sha256(receipt_path),
            "status": receipt.get("status"),
            "exit_code": receipt.get("exit_code"),
            "gpu_workload_launched": receipt.get("gpu_workload_launched"),
        } if receipt else None,
        "claim_boundary": [
            "This is a preregistered train-only association diagnostic under the H1.30 temp0/30-step contract.",
            "It does not establish causal task difficulty, OpenEvo-vs-SEED superiority, or a formal evaluation result.",
            "The generic capability-gate failure is an expected threshold mismatch for this diagnostic and is not a runtime failure.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reconcile(args.manifest.resolve(), args.run_root.resolve(), args.output.resolve()), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
