"""Read-only comparison of GPU handoff receipts from candidate backends.

No GPU probes, dispatch, subprocesses, scheduler control, or scientific writes.
Only same-host monotonic-clock intervals can establish useful-GPU handoff.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCHEMA = "openevo.gpu-handoff-observation.v1"
RESULT_SCHEMA = "openevo.gpu-handoff-shadow-score.v1"
GPU_RE = re.compile(
    r"^GPU-(?:[0-9A-Fa-f]{32}|[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12})$"
)
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
TIMES = (
    "previous_last_useful_kernel_end_ns",
    "next_science_ready_ns",
    "dispatch_detected_ns",
    "spawned_ns",
    "next_first_useful_kernel_ns",
    "next_first_scientific_output_ns",
)
PROOFS = (
    "previous_kernel_evidence_ref",
    "next_kernel_evidence_ref",
    "next_scientific_output_receipt_ref",
)
MIN_TRIALS = 30
TARGET_MS = 5000.0


class ObservationError(ValueError):
    """Invalid or unreconcilable receipt; fail the complete evaluation."""


def _name(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not NAME_RE.fullmatch(value):
        raise ObservationError(f"{field}: required bounded non-empty identifier")
    return value


def _proof(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip() or len(value) > 2048:
        raise ObservationError(f"{field}: missing evidence reference")
    if value.startswith("example:") or value.startswith("placeholder:"):
        raise ObservationError(f"{field}: placeholder is not evidence")
    return value


def validate(row: Any, allow_synthetic: bool = False) -> dict[str, Any]:
    if not isinstance(row, dict) or row.get("schema") != SCHEMA:
        raise ObservationError(f"expected {SCHEMA}")
    for key in ("backend", "workload_class", "clock_domain", "previous_execution", "next_execution"):
        _name(row, key)
    gpu = row.get("gpu_uuid")
    if not isinstance(gpu, str) or not GPU_RE.fullmatch(gpu):
        raise ObservationError("gpu_uuid: exact physical NVIDIA GPU UUID required")
    if row["previous_execution"] == row["next_execution"]:
        raise ObservationError("handoff requires distinct execution identities")
    if type(row.get("warm_model")) is not bool or type(row.get("synthetic")) is not bool:
        raise ObservationError("warm_model and synthetic must be explicit booleans")
    if row["synthetic"] and not allow_synthetic:
        raise ObservationError("synthetic observations require --allow-synthetic")
    times = {}
    for field in TIMES:
        value = row.get(field)
        if type(value) is not int or value < 0:
            raise ObservationError(f"{field}: nonnegative monotonic nanoseconds required")
        times[field] = value
    end = times[TIMES[0]]
    ready, detected, spawned, useful, scientific = (times[k] for k in TIMES[1:])
    if not (ready <= detected <= spawned <= useful <= scientific and end <= useful):
        raise ObservationError("impossible event order or overlapping useful-kernel handoff")
    for field in PROOFS:
        _proof(row, field)
    if row["synthetic"] and any(not row[k].startswith("synthetic:") for k in PROOFS):
        raise ObservationError("synthetic input must label all three proof fields synthetic:")
    if not row["synthetic"] and any(row[k].startswith("synthetic:") for k in PROOFS):
        raise ObservationError("real observation cannot carry synthetic evidence")
    return row


def _quantile_nearest_rank(values: list[int], proportion: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(proportion * len(ordered)) - 1)]


def analyze(rows: list[dict[str, Any]], *, allow_synthetic: bool = False) -> dict[str, Any]:
    seen: set[tuple[str, str, str, str]] = set()
    groups: dict[tuple[str, str, bool, bool], list[dict[str, Any]]] = defaultdict(list)
    for pos, original in enumerate(rows, start=1):
        try:
            row = validate(original, allow_synthetic=allow_synthetic)
        except ObservationError as exc:
            raise ObservationError(f"observation {pos}: {exc}") from exc
        identity = (
            row["clock_domain"], row["gpu_uuid"].casefold(),
            row["previous_execution"], row["next_execution"],
        )
        if identity in seen:
            raise ObservationError(f"observation {pos}: duplicate handoff identity")
        seen.add(identity)
        ready_at_handoff = row["next_science_ready_ns"] <= row["previous_last_useful_kernel_end_ns"]
        key = (row["backend"], row["workload_class"], row["warm_model"], ready_at_handoff)
        groups[key].append(row)
    results = []
    for (backend, klass, warm, pre_ready), members in sorted(groups.items()):
        real = [x for x in members if not x["synthetic"]]
        durations_ns = [
            x["next_first_useful_kernel_ns"] - x["previous_last_useful_kernel_end_ns"]
            for x in real
        ]
        science_durations_ns = [
            x["next_first_scientific_output_ns"] - x["previous_last_useful_kernel_end_ns"]
            for x in real
        ]
        eligible = bool(warm and pre_ready)
        enough = eligible and len(durations_ns) >= MIN_TRIALS
        p95_ns = _quantile_nearest_rank(durations_ns, 0.95) if durations_ns else None
        if not eligible:
            status = "DIAGNOSTIC_ONLY_NOT_READY_OR_COLD"
        elif not enough:
            status = "INSUFFICIENT_REAL_OBSERVATIONS"
        elif p95_ns <= int(TARGET_MS * 1_000_000):
            status = "TIMING_TARGET_MET_SHADOW_ONLY"
        else:
            status = "TIMING_TARGET_MISSED"
        results.append({
            "backend": backend,
            "workload_class": klass,
            "warm_model": warm,
            "ready_at_handoff": pre_ready,
            "real_n": len(real),
            "synthetic_n": len(members) - len(real),
            "p50_useful_gap_ms": round(_quantile_nearest_rank(durations_ns, 0.5) / 1_000_000, 6) if durations_ns else None,
            "p95_useful_gap_ms": round(p95_ns / 1_000_000, 6) if p95_ns is not None else None,
            "max_useful_gap_ms": round(max(durations_ns) / 1_000_000, 6) if durations_ns else None,
            "count_useful_gap_gt_5s": sum(d > 5_000_000_000 for d in durations_ns),
            "count_useful_gap_gt_30s": sum(d > 30_000_000_000 for d in durations_ns),
            "p95_first_scientific_output_gap_ms": (
                round(_quantile_nearest_rank(science_durations_ns, 0.95) / 1_000_000, 6)
                if science_durations_ns else None
            ),
            "status": status,
        })
    return {
        "schema": RESULT_SCHEMA,
        "threshold_ms": TARGET_MS,
        "min_real_observations_per_class": MIN_TRIALS,
        "observations_total": len(rows),
        "backend_groups": results,
        "production_cutover_authorized": False,
        "independent_proof_validation_required": True,
        "notes": "Real means self-reported non-synthetic receipt, not independently verified evidence. Separate live GPU UUID/owner, science parity, crash recovery and peer handback gates remain mandatory.",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="UTF-8 physical-LF JSONL receipt set")
    p.add_argument("--allow-synthetic", action="store_true", help="for CPU fixtures only; never counts as real")
    args = p.parse_args(argv)
    try:
        raw = args.input.read_bytes()
        if b"\r" in raw:
            raise ObservationError("input must use LF physical records")
        rows = []
        for i, line in enumerate(raw.split(b"\n"), start=1):
            if not line:
                if i != len(raw.split(b"\n")):
                    raise ObservationError(f"blank physical record {i}")
                continue
            try:
                rows.append(json.loads(line.decode("utf-8")))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ObservationError(f"physical record {i}: {exc}") from exc
        print(json.dumps(analyze(rows, allow_synthetic=args.allow_synthetic), sort_keys=True, indent=2))
        return 0
    except (OSError, ObservationError) as exc:
        print(f"gpu-handoff shadow score blocked: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
