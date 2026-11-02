#!/usr/bin/env python3
"""Materialize the frozen 8x2 E2 selection as eight SD-LoRA datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from scripts.openevo_webshop.build_self_evolution_dataset import build_dataset


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest_value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest_value.update(block)
    return digest_value.hexdigest()


def prepare(
    harvest_jsonl: Path,
    selection_json: Path,
    output_root: Path,
    records_uri_resolver: Callable[[Path], str] | None = None,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"G2 input root is not empty: {output_root}")
    selection = json.loads(selection_json.read_text(encoding="utf-8"))
    if selection.get("status") != "pass" or selection.get("record_count") != 16:
        raise ValueError("E2 selection is not the frozen passing 16-record selection")
    rows = [json.loads(line) for line in harvest_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_digest = {digest(row): row for row in rows}
    selected = selection.get("selected_records")
    if not isinstance(selected, list) or len(selected) != 16:
        raise ValueError("E2 selection record list must contain exactly 16 rows")
    task_order = selection.get("selected_task_ids")
    if not isinstance(task_order, list) or len(task_order) != 8 or len(set(task_order)) != 8:
        raise ValueError("E2 selection task order must contain exactly 8 unique identities")
    selected_by_task: dict[str, list[dict[str, Any]]] = {str(task): [] for task in task_order}
    for item in selected:
        key = str(item.get("record_sha256"))
        row = by_digest.get(key)
        if row is None:
            raise ValueError(f"selected E2 record is absent from the reconciled harvest: {key}")
        task_id = str(row.get("task_id"))
        if task_id not in selected_by_task or not bool(row.get("qualified_positive")):
            raise ValueError(f"selected E2 row is not a qualified row in the selected task order: {task_id}")
        selected_by_task[task_id].append(row)
    if any(len(rows_for_task) != 2 for rows_for_task in selected_by_task.values()):
        raise ValueError("every G2 increment must contain exactly two selected E2 records")

    output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    increments: list[dict[str, Any]] = []
    for increment, task_id in enumerate(task_order):
        source = output_root / "sources" / f"increment-{increment:02d}-{task_id}"
        source.mkdir(mode=0o700, parents=True, exist_ok=False)
        source_rows = sorted(selected_by_task[task_id], key=lambda row: int(row["rollout_seed"]))
        source_attempts = source / "attempts.jsonl"
        source_attempts.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in source_rows), encoding="utf-8")
        source_attempts.chmod(0o600)
        dataset = output_root / "datasets" / f"increment-{increment:02d}-{task_id}"
        dataset_result = build_dataset(
            [source], dataset, train_task_ids={task_id}, evaluation_task_ids=set(),
            minimum_success_task_count=1, minimum_success_episode_count=2,
        )
        manifest_path = dataset / "manifest.json"
        if records_uri_resolver is not None:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["records_uri"] = records_uri_resolver(dataset / "records.jsonl")
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            manifest_path.chmod(0o600)
        increments.append({
            "increment_index": increment,
            "task_id": task_id,
            "rollout_seeds": [int(row["rollout_seed"]) for row in source_rows],
            "selected_record_sha256": [digest(row) for row in source_rows],
            "source_attempts_sha256": sha256_file(source_attempts),
            "dataset_manifest": str(dataset / "manifest.json"),
            "dataset_manifest_sha256": sha256_file(manifest_path),
            "dataset_records_sha256": dataset_result["records_sha256"],
            "record_count": dataset_result["record_count"],
        })
    result = {
        "schema": "openevo.webshop-h1.40-g2-training-inputs.v1",
        "status": "pass",
        "selection_file": str(selection_json.resolve()),
        "selection_file_sha256": sha256_file(selection_json),
        "harvest_jsonl": str(harvest_jsonl.resolve()),
        "harvest_jsonl_sha256": sha256_file(harvest_jsonl),
        "increment_count": len(increments),
        "record_count": sum(int(item["record_count"]) for item in increments),
        "increments": increments,
    }
    (output_root / "training-inputs.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harvest-jsonl", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.harvest_jsonl.resolve(), args.selection_json.resolve(), args.output_root.resolve())
    print(json.dumps({"status": result["status"], "increment_count": result["increment_count"], "record_count": result["record_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
