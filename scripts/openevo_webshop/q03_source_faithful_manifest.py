"""Validation helpers for the Step119 source-faithful WebShop task manifest.

This module intentionally uses only the Python standard library so the frozen
task identity can be checked before any model/runtime dependencies are loaded.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "openevo.q03-step119-public-validation-id-manifest.v1"
GENERATION_SEED_BASE = 2026082500


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_file_sha256(path: Path, expected_sha256: str | None) -> str:
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if not expected_sha256 or observed != expected_sha256:
        raise ValueError("source-faithful Q03 task manifest file SHA-256 differs from the frozen release")
    return observed


def validate_and_select(manifest: Mapping[str, Any], selected_slots: Sequence[int] | None = None) -> list[dict[str, Any]]:
    """Validate the frozen semantic panel and return tasks in manifest order."""
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unexpected Step119 source-faithful manifest schema")
    payload_sha = manifest.get("payload_sha256")
    body = {key: value for key, value in manifest.items() if key != "payload_sha256"}
    if not isinstance(payload_sha, str) or canonical_sha256(body) != payload_sha:
        raise ValueError("Step119 task manifest payload SHA-256 mismatch")
    for key in ("source_manifest_file_sha256", "source_manifest_content_sha256", "selected_panel_digest"):
        if not isinstance(manifest.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", manifest[key]):
            raise ValueError(f"Step119 task manifest has invalid {key}")
    if manifest.get("task_count") != 128:
        raise ValueError("Step119 source-faithful manifest must declare 128 tasks")
    rows = manifest.get("tasks")
    if not isinstance(rows, list) or len(rows) != 128:
        raise ValueError("Step119 source-faithful manifest must contain exactly 128 tasks")

    fingerprint: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_sessions: set[int] = set()
    for slot, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("Step119 task slots must preserve the frozen 0..127 order")
        row_slot = row.get("slot")
        worker_seed = row.get("worker_seed")
        session_index = row.get("sampled_session_index")
        if (
            isinstance(row_slot, bool)
            or not isinstance(row_slot, int)
            or row_slot != slot
            or isinstance(worker_seed, bool)
            or not isinstance(worker_seed, int)
            or isinstance(session_index, bool)
            or not isinstance(session_index, int)
        ):
            raise ValueError("Step119 task slot, worker seed, and sampled session must be integers")
        task_id = row.get("task_id")
        if worker_seed != 1000 + slot or not 0 <= session_index < 500:
            raise ValueError(f"invalid Step119 public task seed/session at slot {slot}")
        if row.get("split") != "held_out":
            raise ValueError(f"Step119 task at slot {slot} is outside the public held-out split")
        expected_id = f"webshop_seed_source_faithful_slot{slot:03d}_session{session_index:05d}"
        if task_id != expected_id or task_id in seen_ids:
            raise ValueError(f"invalid or duplicate Step119 public task ID at slot {slot}")
        if session_index in seen_sessions:
            raise ValueError(f"duplicate held-out session index at slot {slot}")
        instruction_sha = row.get("instruction_sha256")
        if not isinstance(instruction_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", instruction_sha):
            raise ValueError(f"missing public task instruction digest at slot {slot}")
        generation_seed = row.get("generation_seed")
        if isinstance(generation_seed, bool) or not isinstance(generation_seed, int):
            raise ValueError(f"missing frozen generation seed at slot {slot}")
        if generation_seed != GENERATION_SEED_BASE + slot:
            raise ValueError(f"Q03 generation seed differs from the frozen per-slot schedule at slot {slot}")
        seen_ids.add(task_id)
        seen_sessions.add(session_index)
        fingerprint.append({
            "slot": slot,
            "worker_seed": worker_seed,
            "sampled_session_index": session_index,
            "instruction_sha256": instruction_sha,
        })

    if canonical_sha256(fingerprint) != manifest["selected_panel_digest"]:
        raise ValueError("Step119 selected panel digest does not match its task identities")
    if selected_slots is None:
        selected = set(range(128))
    else:
        selected_values = list(selected_slots)
        if any(isinstance(slot, bool) or not isinstance(slot, int) for slot in selected_values):
            raise ValueError("requested Step119 task slots must be integers")
        selected = set(selected_values)
        if len(selected) != len(selected_values):
            raise ValueError("requested Step119 task slots must be unique")
    if any(slot < 0 or slot >= 128 for slot in selected):
        raise ValueError("requested Step119 task slot is outside 0..127")
    return [dict(row) for row in rows if int(row["slot"]) in selected]


def to_evaluation_tasks(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Map frozen semantic task identities to the evaluator's per-episode input."""
    return [
        {
            "goal_idx": int(row["sampled_session_index"]),
            "task_id": str(row["task_id"]),
            "instruction_sha256": str(row["instruction_sha256"]),
            "generation_seed": int(row["generation_seed"]),
            "panel_ordinal": int(row["slot"]),
            "worker_seed": int(row["worker_seed"]),
            "sampled_session_index": int(row["sampled_session_index"]),
        }
        for row in rows
    ]


def environment_identity(task: Mapping[str, Any], default_seed: int) -> tuple[int, int]:
    """Return (worker seed, reset session index) for one evaluation task."""
    worker_seed = int(task.get("worker_seed", default_seed))
    session_index = int(task.get("sampled_session_index", task.get("goal_idx", -1)))
    if session_index < 0:
        raise ValueError("evaluation task lacks a non-negative session/goal index")
    return worker_seed, session_index


def step_generation_seed(task: Mapping[str, Any], step_index: int) -> int:
    """Return the frozen per-episode plus step-index generation seed."""
    if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
        raise ValueError("Q03 generation step index must be a non-negative integer")
    seed = task.get("generation_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < GENERATION_SEED_BASE:
        raise ValueError("Q03 task lacks a valid frozen per-episode generation seed")
    return seed + step_index


def bind_validation_carrier_rows(
    carrier_rows: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Bind ordered parquet carrier rows to source-faithful environment tasks.

    The public SEED test parquet has 128 carrier rows indexed 0..127. Those
    indexes identify validation *slots*, while each task's
    ``sampled_session_index`` is selected separately by the WebShop
    environment's seeded RNG. Treating a sampled session as a parquet row
    index would silently change the evaluated task.
    """
    if len(carrier_rows) != 128 or len(tasks) != 128:
        raise ValueError("Q03 Step119 requires exactly 128 validation carrier rows and tasks")

    bound: list[dict[str, Any]] = []
    for slot, (carrier, task) in enumerate(zip(carrier_rows, tasks, strict=True)):
        if not isinstance(carrier, Mapping):
            raise ValueError(f"Q03 validation carrier slot {slot} is not a row mapping")
        extra = carrier.get("extra_info")
        if not isinstance(extra, Mapping):
            raise ValueError(f"Q03 validation carrier slot {slot} lacks producer extra_info")
        if extra.get("split") != "test" or extra.get("index") != slot:
            raise ValueError(f"Q03 validation carrier slot {slot} is not the frozen ordered test row")
        task_slot = task.get("slot")
        if isinstance(task_slot, bool) or not isinstance(task_slot, int) or task_slot != slot:
            raise ValueError(f"Q03 task order differs from the validation carrier at slot {slot}")

        row = dict(carrier)
        row.update({
            "q03_panel_slot": slot,
            "q03_task_id": str(task["task_id"]),
            "q03_worker_seed": int(task["worker_seed"]),
            "q03_sampled_session_index": int(task["sampled_session_index"]),
            "q03_generation_seed": int(task["generation_seed"]),
            "q03_instruction_sha256": str(task["instruction_sha256"]),
        })
        bound.append(row)
    return bound


def validate_environment_session_draw(
    tasks: Sequence[Mapping[str, Any]],
    validation_base_seed: int = 1000,
) -> None:
    """Prove that SEED's validation RNG selects the frozen session order."""
    if isinstance(validation_base_seed, bool) or not isinstance(validation_base_seed, int):
        raise ValueError("Q03 validation environment seed must be an integer")
    if len(tasks) != 128:
        raise ValueError("Q03 Step119 requires exactly 128 source-faithful environment tasks")
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required to verify the pinned SEED environment draw") from exc
    expected = [
        int(value)
        for value in np.random.RandomState(validation_base_seed).choice(
            np.arange(500), size=128, replace=False
        )
    ]
    observed = []
    for slot, task in enumerate(tasks):
        task_slot = task.get("slot")
        worker_seed = task.get("worker_seed")
        session_index = task.get("sampled_session_index")
        if isinstance(task_slot, bool) or not isinstance(task_slot, int) or task_slot != slot:
            raise ValueError("Q03 task order differs from the source-faithful validation slots")
        if isinstance(worker_seed, bool) or not isinstance(worker_seed, int) or worker_seed != validation_base_seed + slot:
            raise ValueError(f"Q03 WebShop worker seed mismatch at slot {slot}")
        if isinstance(session_index, bool) or not isinstance(session_index, int) or not 0 <= session_index < 500:
            raise ValueError(f"Q03 sampled WebShop session is invalid at slot {slot}")
        observed.append(session_index)
    if observed != expected:
        raise ValueError("Q03 manifest session order differs from the pinned SEED validation RNG draw")
