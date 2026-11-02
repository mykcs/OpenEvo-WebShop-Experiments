#!/usr/bin/env python3
"""Freeze H1.38B D8 increments and a new task-ID-disjoint evaluation panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

try:
    from scripts.openevo_webshop.build_h1_30_manifest import collect_prior_indices
    from scripts.openevo_webshop.webshop_task_manifest import canonical_bytes, live_goals, sha256_text
except ModuleNotFoundError:  # pragma: no cover - direct container execution
    from build_h1_30_manifest import collect_prior_indices  # type: ignore
    from webshop_task_manifest import canonical_bytes, live_goals, sha256_text  # type: ignore


SCHEMA = "openevo.webshop-h1.38b-method-control-inputs.v1"
CAMPAIGN_ID = "20260820-0129-h138b-method-control"
D8_RECORDS_SHA256 = "d206c20e4d08a9c1785a6af9c2edb510db4a478d5d459395537706a2d7d8191c"
D8_SELECTION_SHA256 = "b8c35b51aed3bf942cb2e45e7c9c0e68239d6efc27a3d7be8bc6a5ed54ff70c7"
FRESH_TASK_COUNT = 48
FRESH_SELECTION_SEED = 202608200138
EXPECTED_PRIOR_IDENTITY_COUNT = 493
EXPECTED_PRIOR_IDENTITY_DIGEST = "24efaed40ff35f5624fc82b4183ce3d58cb72690d686b8627aee2772d7719eeb"
ACQUISITION_SEEDS = [2026090600, 2026090601]
RETENTION_SEEDS = [2026090610, 2026090611, 2026090612, 2026090613]
FRESH_SEEDS = [2026090620, 2026090621, 2026090622, 2026090623]
ARMS = ["base", "ordinary-lora", "sd-lora"]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_hash(value: Any) -> str:
    return _sha256_bytes(canonical_bytes(value))


def openevo_dataset_manifest_bytes(value: Any) -> bytes:
    """Match pinned OpenEvo strict dataset-manifest byte serialization."""

    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")


def _index_digest(indices: set[int]) -> str:
    return _sha256_bytes("".join(f"{value}\n" for value in sorted(indices)).encode())


def _record_identity(record: dict[str, Any]) -> tuple[str, int, str]:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("D8 record lacks metadata")
    task_id = str(record.get("task_id") or "")
    rollout_seed = metadata.get("rollout_seed")
    source_hash = str(metadata.get("source_record_sha256") or "")
    if not task_id or isinstance(rollout_seed, bool) or not isinstance(rollout_seed, int):
        raise ValueError("D8 record identity is malformed")
    if len(source_hash) != 64:
        raise ValueError("D8 record source hash is malformed")
    return task_id, rollout_seed, source_hash


def load_sealed_d8(manifest_path: Path) -> tuple[dict[str, Any], list[bytes], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records_path = manifest_path.with_name(str(manifest.get("records_path") or "records.jsonl"))
    records_bytes = records_path.read_bytes()
    if _sha256_bytes(records_bytes) != D8_RECORDS_SHA256:
        raise ValueError("sealed D8 records SHA-256 mismatch")
    if manifest.get("records_sha256") != D8_RECORDS_SHA256:
        raise ValueError("D8 manifest records SHA-256 mismatch")
    if manifest.get("selection_sha256") != D8_SELECTION_SHA256:
        raise ValueError("D8 selection SHA-256 mismatch")
    if int(manifest.get("record_count") or 0) != 16 or int(manifest.get("identity_count") or 0) != 8:
        raise ValueError("D8 manifest count mismatch")
    raw_lines = records_bytes.splitlines(keepends=True)
    if len(raw_lines) != 16 or any(not line.endswith(b"\n") for line in raw_lines):
        raise ValueError("D8 records must contain exactly 16 newline-terminated rows")
    records = [json.loads(line) for line in raw_lines]
    selection = manifest.get("selection")
    if not isinstance(selection, list) or len(selection) != 16:
        raise ValueError("D8 selection must contain 16 rows")
    if _canonical_hash(selection) != D8_SELECTION_SHA256:
        raise ValueError("D8 selection canonical identity mismatch")
    observed = [_record_identity(record) for record in records]
    expected = [
        (str(item.get("task_id") or ""), int(item.get("rollout_seed")), str(item.get("record_sha256") or ""))
        for item in selection
    ]
    if observed != expected:
        raise ValueError("D8 record order/identity differs from the sealed selection")
    return manifest, raw_lines, records


def write_increments(manifest_path: Path, output_root: Path) -> dict[str, Any]:
    parent, raw_lines, records = load_sealed_d8(manifest_path)
    selection = parent["selection"]
    task_order: list[str] = []
    for record in records:
        task_id = _record_identity(record)[0]
        if task_id not in task_order:
            task_order.append(task_id)
    if len(task_order) != 8:
        raise ValueError("D8 must yield exactly eight ordered task increments")
    increments: list[dict[str, Any]] = []
    reconstructed = b""
    for increment_index, task_id in enumerate(task_order):
        positions = [index for index, record in enumerate(records) if _record_identity(record)[0] == task_id]
        if len(positions) != 2 or positions != list(range(positions[0], positions[0] + 2)):
            raise ValueError(f"{task_id} is not one contiguous two-record increment")
        seeds = [_record_identity(records[index])[1] for index in positions]
        if len(set(seeds)) != 2:
            raise ValueError(f"{task_id} does not contain two independent rollout seeds")
        payload = b"".join(raw_lines[index] for index in positions)
        reconstructed += payload
        increment_dir = output_root / f"increment-{increment_index:02d}-{task_id}"
        increment_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
        records_path = increment_dir / "records.jsonl"
        records_path.write_bytes(payload)
        records_path.chmod(0o600)
        selected = [selection[index] for index in positions]
        trace_count = sum(len(records[index].get("traces") or []) for index in positions)
        document = {
            "schema": "openevo.dataset.v1",
            "name": f"h138b-increment-{increment_index:02d}-{task_id}",
            "source": parent.get("source"),
            "source_route": parent.get("source_route"),
            "teacher_identity": None,
            "record_count": 2,
            "identity_count": 1,
            "trace_count": trace_count,
            "records_path": "records.jsonl",
            "records_uri": records_path.resolve().as_uri(),
            "records_byte_size": len(payload),
            "records_sha256": _sha256_bytes(payload),
            "selection": selected,
            "selection_sha256": _canonical_hash(selected),
            "parent_records_sha256": D8_RECORDS_SHA256,
            "parent_selection_sha256": D8_SELECTION_SHA256,
            "increment_index": increment_index,
            "task_id": task_id,
            "goal_idx": int(selected[0]["goal_idx"]),
            "rollout_seeds": seeds,
        }
        increment_manifest = increment_dir / "manifest.json"
        increment_manifest.write_bytes(openevo_dataset_manifest_bytes(document))
        increment_manifest.chmod(0o600)
        increments.append(
            {
                "increment_index": increment_index,
                "task_id": task_id,
                "goal_idx": document["goal_idx"],
                "rollout_seeds": seeds,
                "records_sha256": document["records_sha256"],
                "manifest_file_sha256": _sha256_bytes(increment_manifest.read_bytes()),
                "manifest_path": str(increment_manifest),
            }
        )
    if reconstructed != b"".join(raw_lines) or _sha256_bytes(reconstructed) != D8_RECORDS_SHA256:
        raise ValueError("increment concatenation is not byte-identical to sealed D8")
    return {"task_order": task_order, "increments": increments}


def build_evaluation_manifest(
    goals: list[dict[str, Any]],
    *,
    prior_indices: set[int],
    d8_selection: list[dict[str, Any]],
    experiment_git_sha: str,
) -> dict[str, Any]:
    prior_digest = _index_digest(prior_indices)
    if len(prior_indices) != EXPECTED_PRIOR_IDENTITY_COUNT:
        raise ValueError("pre-H1.38B prior-identity count mismatch")
    if prior_digest != EXPECTED_PRIOR_IDENTITY_DIGEST:
        raise ValueError("pre-H1.38B prior-identity digest mismatch")
    train_indices = [int(item["goal_idx"]) for item in d8_selection[::2]]
    if len(train_indices) != 8 or len(set(train_indices)) != 8:
        raise ValueError("D8 training identity contract mismatch")
    if not set(train_indices).issubset(prior_indices):
        raise ValueError("D8 training tasks are absent from the prior-identity ledger")
    pool = [index for index in range(500, len(goals)) if index not in prior_indices]
    random.Random(FRESH_SELECTION_SEED).shuffle(pool)
    fresh_indices = sorted(pool[:FRESH_TASK_COUNT])
    if len(fresh_indices) != FRESH_TASK_COUNT or set(fresh_indices) & prior_indices:
        raise ValueError("H1.38B fresh-task selection is not task-ID-disjoint")

    def task_row(goal_idx: int, split: str) -> dict[str, Any]:
        description = str(goals[goal_idx].get("instruction_text") or "").strip()
        if not description:
            raise ValueError(f"empty WebShop task {goal_idx}")
        return {
            "task_id": f"webshop_{goal_idx:06d}",
            "goal_idx": goal_idx,
            "split": split,
            "task_description": description,
            "task_description_sha256": sha256_text(description),
        }

    training_tasks = [task_row(index, "h1.38b-train-identity") for index in train_indices]
    fresh_tasks = [task_row(index, "h1.38b-fresh-task-disjoint") for index in fresh_indices]
    document: dict[str, Any] = {
        "schema": "openevo.webshop-h1.38b-method-control-eval-manifest.v1",
        "campaign_id": CAMPAIGN_ID,
        "source": {
            "experiment_git_sha": experiment_git_sha,
            "d8_records_sha256": D8_RECORDS_SHA256,
            "d8_selection_sha256": D8_SELECTION_SHA256,
        },
        "task_order": train_indices,
        "training_tasks": training_tasks,
        "fresh_tasks": fresh_tasks,
        "tasks": [*training_tasks, *fresh_tasks],
        "acquisition": {"arms": ARMS[1:], "rollout_seeds": ACQUISITION_SEEDS, "cells": 16, "attempts": 32},
        "retention": {"arms": ARMS, "rollout_seeds": RETENTION_SEEDS, "cells": 24, "attempts": 96},
        "fresh": {"arms": ARMS, "rollout_seeds": FRESH_SEEDS, "cells": 144, "attempts": 576},
        "planned_attempts": 704,
        "task_selection": {
            "selection_seed": FRESH_SELECTION_SEED,
            "fresh_task_count": FRESH_TASK_COUNT,
            "prior_identity_count": len(prior_indices),
            "prior_identity_digest": prior_digest,
            "prior_identity_indices": sorted(prior_indices),
            "policy": "exclude every prior committed train/eval identity plus the aborted pre-protocol ledger",
        },
        "contamination": {
            "training_fresh_overlap": False,
            "prior_fresh_overlap": False,
            "evaluation_outcomes_used_for_training": False,
            "teacher_oracle_gpt_used": False,
        },
        "manifest_sha256": "",
    }
    document["manifest_sha256"] = _canonical_hash(
        {key: value for key, value in document.items() if key != "manifest_sha256"}
    )
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d8-manifest", type=Path, required=True)
    parser.add_argument("--increment-root", type=Path, required=True)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--aborted-ledger", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--experiment-git-sha", required=True)
    args = parser.parse_args()

    parent, _, _ = load_sealed_d8(args.d8_manifest.resolve())
    increments = write_increments(args.d8_manifest.resolve(), args.increment_root.resolve())
    ledger = json.loads(args.aborted_ledger.read_text(encoding="utf-8"))
    extra = ledger.get("task_ids") if isinstance(ledger, dict) else None
    if not isinstance(extra, list) or len(extra) != 64:
        raise SystemExit("aborted reservoir ledger malformed")
    goals, _ = live_goals(
        args.seed_root.resolve(),
        2026,
        data_dir=args.data_dir.resolve(),
        index_dir=args.index_dir.resolve(),
    )
    prior = collect_prior_indices([args.prior_root.resolve()]) | {int(item) for item in extra}
    evaluation = build_evaluation_manifest(
        goals,
        prior_indices=prior,
        d8_selection=parent["selection"],
        experiment_git_sha=args.experiment_git_sha,
    )
    args.evaluation_manifest.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.evaluation_manifest.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output = {"schema": SCHEMA, "increments": increments, "evaluation": evaluation}
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
