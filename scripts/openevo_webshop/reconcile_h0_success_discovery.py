#!/usr/bin/env python3
"""Reconcile every H0.1 success-discovery attempt without scoring omissions."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.openevo_webshop.build_h0_success_discovery_manifest import validate_manifest


def _failure_mode(row: dict[str, Any]) -> str:
    if row.get("error"):
        return "runtime-failure"
    if not row.get("identity_ok") or not row.get("trace_complete"):
        return "protocol-invalid"
    if row.get("parser_or_fallback_failure"):
        return "parser-failure"
    if bool(row.get("qualified_positive")):
        return "qualified-positive"
    if bool(row.get("partial_positive")):
        return "partial-positive"
    return "zero-reward"


def reconcile(manifest: dict[str, Any], run_root: Path) -> dict[str, Any]:
    validate_manifest(manifest)
    expected = {
        (row["task_id"], int(seed))
        for row in manifest["tasks"]
        for seed in manifest["controls"]["rollout_seeds"]
    }
    records: list[dict[str, Any]] = []
    cell_paths = sorted((run_root / "harvest").glob("wave*/task*/attempts.jsonl"))
    for path in cell_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["_source"] = str(path)
                records.append(row)
    observed = {(row.get("task_id"), int(row.get("rollout_seed"))) for row in records}
    missing = sorted(expected - observed)
    extras = sorted(observed - expected)
    duplicates = len(records) - len(observed)
    by_mode = Counter(_failure_mode(row) for row in records)
    positive_tasks = sorted({row["task_id"] for row in records if row.get("qualified_positive")})
    scores = [float(row.get("final_task_score") or 0.0) for row in records]
    return {
        "schema": "openevo.webshop-h0.1-success-discovery-reconciliation.v1",
        "experiment": manifest["experiment"],
        "status": "pass" if not missing and not extras and duplicates == 0 else "fail",
        "formal_task_consumption_allowed": False,
        "scientific_result": bool(not missing and not extras and duplicates == 0),
        "manifest_sha256": manifest["manifest_sha256"],
        "expected_attempts": len(expected),
        "observed_attempts": len(records),
        "missing_attempts": missing,
        "extra_attempts": extras,
        "duplicate_record_count": duplicates,
        "qualified_positive_count": sum(bool(row.get("qualified_positive")) for row in records),
        "qualified_positive_task_count": len(positive_tasks),
        "qualified_positive_task_ids": positive_tasks,
        "qualified_positive_rate": (
            sum(bool(row.get("qualified_positive")) for row in records) / len(records) if records else 0.0
        ),
        "nonzero_reward_count": sum(score > 0.0 for score in scores),
        "mean_task_score": sum(scores) / len(scores) if scores else 0.0,
        "failure_modes": dict(sorted(by_mode.items())),
        "teacher_trigger": bool(not missing and not extras and duplicates == 0 and not positive_tasks),
        "claim_boundary": "Qwen native success-discovery diagnostic only; no OpenEvo efficacy or SEED comparison claim",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.expanduser().resolve().read_text(encoding="utf-8"))
    receipt = reconcile(manifest, args.run_root.expanduser().resolve())
    output = args.output.expanduser().resolve()
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(receipt, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(output)
    print(f"H0.1_RECONCILIATION={output}")
    print(f"H0.1_RECONCILIATION_STATUS={receipt['status']}")
    print(f"H0.1_QUALIFIED_POSITIVES={receipt['qualified_positive_count']}")
    return 0 if receipt["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
