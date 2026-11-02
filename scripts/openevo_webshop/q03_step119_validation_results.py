"""Validate and persist one row per frozen Q03 Step119 validation identity."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "openevo.q03-step119-validation-results.v1"
ROW_FIELDS = (
    "q03_panel_slot",
    "q03_task_id",
    "q03_sampled_session_index",
    "q03_worker_seed",
    "q03_generation_seed",
    "q03_instruction_sha256",
    "q03_validation_task_score",
    "q03_validation_exact_success",
)


def collect_batch_rows(columns: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    missing = [name for name in ROW_FIELDS if name not in columns]
    if missing:
        raise ValueError(f"Step119 output lacks per-task result columns: {missing}")
    lengths = {len(columns[name]) for name in ROW_FIELDS}
    if len(lengths) != 1:
        raise ValueError("Step119 per-task result column lengths differ")

    by_slot: dict[int, dict[str, Any]] = {}
    for index in range(next(iter(lengths), 0)):
        def integer(name: str) -> int:
            value = columns[name][index]
            if isinstance(value, bool) or not hasattr(value, "__index__"):
                raise ValueError(f"Step119 {name} is not an integer")
            return int(value)

        slot = integer("q03_panel_slot")
        session = integer("q03_sampled_session_index")
        worker_seed = integer("q03_worker_seed")
        generation_seed = integer("q03_generation_seed")
        raw_success = float(columns["q03_validation_exact_success"][index])
        if raw_success not in (0.0, 1.0):
            raise ValueError("Step119 exact success value must be boolean")
        row = {
            "panel_slot": slot,
            "task_id": str(columns["q03_task_id"][index]),
            "sampled_session_index": session,
            "worker_seed": worker_seed,
            "generation_seed": generation_seed,
            "instruction_sha256": str(columns["q03_instruction_sha256"][index]),
            "task_score": float(columns["q03_validation_task_score"][index]),
            "exact_success": bool(raw_success),
        }
        if not 0 <= slot < 128 or not 0 <= session < 500:
            raise ValueError("Step119 panel slot or sampled session is out of range")
        if not math.isfinite(row["task_score"]) or not 0.0 <= row["task_score"] <= 1.0:
            raise ValueError("Step119 task score must be finite and within [0, 1]")
        if row["exact_success"] != (row["task_score"] == 1.0):
            raise ValueError("Step119 exact success disagrees with the terminal task score")
        if row["task_id"] != (
            f"webshop_seed_source_faithful_slot{slot:03d}_"
            f"session{row['sampled_session_index']:05d}"
        ):
            raise ValueError("Step119 task ID does not match its slot and session")
        if row["worker_seed"] != 1000 + slot:
            raise ValueError("Step119 worker seed does not match its frozen slot")
        if row["generation_seed"] != 2026082500 + slot:
            raise ValueError("Step119 generation seed does not match its frozen slot")
        if len(row["instruction_sha256"]) != 64 or any(
            character not in "0123456789abcdef" for character in row["instruction_sha256"]
        ):
            raise ValueError("Step119 instruction digest is malformed")
        prior = by_slot.setdefault(slot, row)
        if prior != row:
            raise ValueError("Step119 repeated rollout rows disagree for one task")
    return [by_slot[slot] for slot in sorted(by_slot)]


def build_document(
    rows: Sequence[Mapping[str, Any]],
    *,
    task_manifest_sha256: str,
    parquet_sha256: str,
    experiment_id: str,
) -> dict[str, Any]:
    ordered = sorted((dict(row) for row in rows), key=lambda row: row["panel_slot"])
    if [row.get("panel_slot") for row in ordered] != list(range(128)):
        raise ValueError("Step119 final results must contain each frozen slot 0..127 exactly once")
    if len({row.get("task_id") for row in ordered}) != 128:
        raise ValueError("Step119 final results contain duplicate task IDs")
    scores = [float(row["task_score"]) for row in ordered]
    successes = sum(bool(row["exact_success"]) for row in ordered)
    return {
        "schema": SCHEMA,
        "status": "PASS",
        "experiment_id": experiment_id,
        "task_manifest_sha256": task_manifest_sha256,
        "validation_parquet_sha256": parquet_sha256,
        "denominator": 128,
        "observed_rows": len(ordered),
        "aggregates": {
            "mean_task_score": sum(scores) / len(scores),
            "exact_success_count": successes,
            "exact_success_rate": successes / len(ordered),
        },
        "results": ordered,
    }


def write_document(run_root: Path, document: Mapping[str, Any]) -> Path:
    if run_root.is_symlink() or not run_root.is_dir():
        raise ValueError("Step119 result output root must be a real directory")
    target = run_root / "q03-step119-validation-results.json"
    payload = (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(target, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)
    return target
