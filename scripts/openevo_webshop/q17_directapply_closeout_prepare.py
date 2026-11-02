#!/usr/bin/env python3
"""Prepare read-only Q17 DirectApply Stage2 closeout inputs for D1/final evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_root.resolve()
    out = args.output_root.resolve()
    state_path = run / "execution-state.json"
    state = load(state_path)
    if state.get("status") != "STAGE2_SEALED":
        raise RuntimeError("Q17 Stage2 is not sealed")
    if int(state.get("next_round", -1)) != 160 or int(state.get("rollouts_consumed", -1)) != 20480:
        raise RuntimeError("Q17 Stage2 denominator drift")
    if list(state.get("rounds_complete") or []) != list(range(160)):
        raise RuntimeError("Q17 sealed round coverage drift")
    if int(state.get("external_teacher_calls", -1)) != 0 or int(state.get("final_panel_access_count", -1)) != 0:
        raise RuntimeError("Q17 closeout budget boundary drift")

    seal = {
        "schema": "openevo.q17-directapply-stage2-seal.202609120000",
        "status": "PASS",
        "run_root": str(run),
        "execution_state_path": str(state_path),
        "execution_state_sha256": sha256(state_path),
        "rounds": 160,
        "rollouts": 20480,
        "sd_generation_count": int(state.get("sd_generation_count", -1)),
        "sd_component_count": int(state.get("sd_component_count", -1)),
        "final_state_id": state.get("state_id"),
        "final_sd_adapter": state.get("current_sd_adapter"),
        "memory": state.get("memory"),
        "skill": state.get("skill"),
        "agent_system": state.get("agent_system"),
        "external_teacher_calls": 0,
        "final_panel_access_count": 0,
    }
    seal_path = out / "STAGE2_SEAL_Q17_DIRECTAPPLY.json"
    atomic_json(seal_path, seal)

    entries: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    source_files: list[dict[str, str]] = []
    expected_component = 1
    for round_index in range(160):
        rr = run / f"round-{round_index:03d}"
        complete_path = rr / "ROUND_COMPLETE.json"
        if not complete_path.is_file():
            raise RuntimeError(f"missing sealed round receipt: {complete_path}")
        complete = load(complete_path)
        if int(complete.get("round_index", -1)) != round_index or complete.get("status") != "PASS":
            raise RuntimeError(f"invalid round receipt: {complete_path}")
        geometry_path = rr / "task-vector-geometry.json"
        if geometry_path.is_file():
            geometry = load(geometry_path)
            count = int(geometry.get("component_count", -1))
            if count != expected_component:
                raise RuntimeError(f"non-sequential Task-Vector component count at round {round_index}: {count} != {expected_component}")
            entries.append({
                "round_index": round_index,
                "component_count": count,
                "accepted_state_id_before": complete.get("state_id_before"),
                "accepted_state_id_after": complete.get("state_id_after"),
                "geometry_path": str(geometry_path),
                "geometry_sha256": sha256(geometry_path),
            })
            expected_component += 1
        records_path = rr / "sd-lora" / "dataset" / "records.jsonl"
        if records_path.is_file():
            source_files.append({"path": str(records_path), "sha256": sha256(records_path)})
            for line in records_path.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                for trace_index, trace in enumerate(record.get("traces") or []):
                    metadata = trace.get("metadata") or {}
                    source_rows.append({
                        "task_id": record.get("task_id"),
                        "source_schedule_position": int(metadata.get("source_schedule_position", -1)),
                        "step_idx": int(metadata.get("source_step", trace_index)),
                        "trajectory_id": record.get("session_id") or record.get("event_id"),
                        "prompt_messages": trace.get("prompt_messages"),
                        "response_messages": trace.get("response_messages"),
                        "source_round": round_index,
                        "final_panel": False,
                        "heldout_final": False,
                    })

    if len(entries) != int(state.get("sd_generation_count", -1)) or len(entries) != 159:
        raise RuntimeError(f"Task-Vector geometry coverage drift: {len(entries)}")
    manifest_path = out / "D1_ACCEPTED_GEOMETRY_MANIFEST.json"
    atomic_json(manifest_path, {
        "schema": "openevo.ceiling1-gdn-d1-accepted-geometry-manifest.v1",  # legacy-revision-compat: historical serialized schema identity
        "status": "PASS",
        "arm": "qwen3-1p7b",
        "stage2_sealed": True,
        "stage2_seal_path": str(seal_path),
        "stage2_seal_sha256": sha256(seal_path),
        "final_panel_access_count": 0,
        "external_teacher_calls_delta": 0,
        "new_environment_calls_delta": 0,
        "entries": entries,
    })
    rows_path = out / "D1_SOURCE_ROWS.jsonl"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in source_rows))
    atomic_json(out / "D1_SOURCE_ROWS_MANIFEST.json", {
        "schema": "openevo.q17-directapply-d1-source-rows.202609120000",
        "status": "PASS",
        "source_file_count": len(source_files),
        "source_files": source_files,
        "flattened_row_count": len(source_rows),
        "rows_path": str(rows_path),
        "rows_sha256": sha256(rows_path),
        "final_panel_access": 0,
        "new_environment_calls": 0,
        "external_teacher_calls": 0,
    })
    print(json.dumps({"status": "PASS", "geometry_entries": len(entries), "source_rows": len(source_rows), "stage2_seal": str(seal_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
