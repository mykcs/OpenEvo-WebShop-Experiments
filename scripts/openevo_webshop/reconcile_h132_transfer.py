#!/usr/bin/env python3
"""Reconcile the H1.32 BASE versus frozen H1.29-adapter transfer panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ARMS = {"base": "BASE", "openevo-transfer": "H129_SELF_ADAPTER"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _valid(row: dict[str, Any]) -> bool:
    return bool(
        row.get("identity_ok")
        and row.get("trace_complete")
        and not row.get("parser_or_fallback_failure")
        and not row.get("error")
    )


def reconcile(config: dict[str, Any], manifest: dict[str, Any], cells: list[tuple[str, int, Path]]) -> dict[str, Any]:
    failures: list[str] = []
    panel = config.get("panel", {})
    expected_manifest = panel.get("source_manifest_sha256")
    observed_manifest = manifest.get("manifest_sha256")
    if expected_manifest != observed_manifest:
        failures.append(f"manifest identity mismatch: expected {expected_manifest!r}, observed {observed_manifest!r}")
    tasks = {int(row["goal_idx"]): row for row in manifest.get("tasks", [])}
    expected_tasks = {int(value) for value in panel.get("task_ids", [])}
    if set(tasks) != expected_tasks:
        failures.append(f"task identity mismatch: expected {sorted(expected_tasks)}, observed {sorted(tasks)}")
    seeds = tuple(int(value) for value in config.get("frozen_contract", {}).get("rollout_seeds", []))
    expected_keys = {(arm, task, seed) for arm in ARMS for task in expected_tasks for seed in seeds}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    failure_counts: Counter[str] = Counter()
    for arm, task, root in cells:
        if arm not in ARMS:
            failures.append(f"unsupported arm: {arm}")
            continue
        episodes = root / "episodes"
        for path in sorted(episodes.glob("*.json")):
            row = load_json(path)
            key = (arm, int(row.get("goal_idx", -1)), int(row.get("rollout_seed", -1)))
            if key in seen:
                failures.append(f"duplicate episode: {key}")
            seen.add(key)
            if key[0] != row.get("arm") or key[1] != task or key[1] not in tasks or key[2] not in seeds:
                failures.append(f"episode identity mismatch at {path}")
            rows.append(row)
            if row.get("error"):
                failure_counts["runtime-error"] += 1
            elif row.get("parser_or_fallback_failure"):
                failure_counts["parser-or-fallback"] += 1
            elif not row.get("identity_ok"):
                failure_counts["identity-mismatch"] += 1
            elif not row.get("trace_complete"):
                failure_counts["incomplete-trace"] += 1
            elif float(row.get("final_task_score") or 0.0) == 0.0:
                failure_counts["zero-reward-other"] += 1
            else:
                failure_counts["qualified-positive"] += 1
    if seen != expected_keys:
        missing = sorted(expected_keys - seen)
        extra = sorted(seen - expected_keys)
        if missing:
            failures.append(f"missing episodes: {missing[:8]}" + (" ..." if len(missing) > 8 else ""))
        if extra:
            failures.append(f"unexpected episodes: {extra[:8]}" + (" ..." if len(extra) > 8 else ""))

    by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for row in rows:
        by_arm.setdefault(str(row.get("arm")), []).append(row)
    arm_summary: dict[str, Any] = {}
    for arm, label in ARMS.items():
        arm_rows = by_arm[arm]
        valid_rows = [row for row in arm_rows if _valid(row)]
        scores = [float(row.get("final_task_score") or 0.0) for row in valid_rows]
        arm_summary[label] = {
            "arm": arm,
            "attempt_count": len(arm_rows),
            "valid_count": len(valid_rows),
            "invalid_count": len(arm_rows) - len(valid_rows),
            "mean_valid_task_score": sum(scores) / len(scores) if scores else 0.0,
            "valid_win_rate": sum(bool(row.get("won")) for row in valid_rows) / len(valid_rows) if valid_rows else 0.0,
        }

    indexed = {(str(row.get("arm")), int(row.get("goal_idx")), int(row.get("rollout_seed"))): row for row in rows}
    paired: list[dict[str, Any]] = []
    for task in sorted(expected_tasks):
        for seed in seeds:
            base = indexed.get(("base", task, seed))
            adapter = indexed.get(("openevo-transfer", task, seed))
            if base is None or adapter is None:
                continue
            paired.append({
                "goal_idx": task,
                "rollout_seed": seed,
                "base_valid": _valid(base),
                "adapter_valid": _valid(adapter),
                "base_score": float(base.get("final_task_score") or 0.0),
                "adapter_score": float(adapter.get("final_task_score") or 0.0),
                "delta": float(adapter.get("final_task_score") or 0.0) - float(base.get("final_task_score") or 0.0),
            })
    valid_pairs = [row for row in paired if row["base_valid"] and row["adapter_valid"]]
    paired_delta = sum(row["delta"] for row in valid_pairs) / len(valid_pairs) if valid_pairs else 0.0
    base_invalid = 1.0 - arm_summary["BASE"]["valid_count"] / max(1, arm_summary["BASE"]["attempt_count"])
    adapter_invalid = 1.0 - arm_summary["H129_SELF_ADAPTER"]["valid_count"] / max(1, arm_summary["H129_SELF_ADAPTER"]["attempt_count"])
    return {
        "schema": "openevo.webshop-h1.32-transfer-reconciliation.v1",
        "status": "pass" if not failures else "blocked",
        "formal_task_consumption_allowed": False,
        "scientific_result": not failures,
        "manifest_sha256": observed_manifest,
        "manifest_file_sha256": sha256_file(Path(panel["source_manifest"])) if Path(panel.get("source_manifest", "")).is_file() else None,
        "arms": arm_summary,
        "paired": {"pair_count": len(paired), "valid_pair_count": len(valid_pairs), "mean_adapter_minus_base_task_score": paired_delta},
        "invalid_rate": {"base": base_invalid, "adapter": adapter_invalid, "adapter_minus_base": adapter_invalid - base_invalid},
        "failure_taxonomy": dict(failure_counts),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cell", action="append", required=True, help="ARM:TASK_ID:OUTPUT_ROOT")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    manifest = load_json(args.manifest)
    cells: list[tuple[str, int, Path]] = []
    for value in args.cell:
        arm, task, root = value.split(":", 2)
        cells.append((arm, int(task), Path(root)))
    result = reconcile(config, manifest, cells)
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
