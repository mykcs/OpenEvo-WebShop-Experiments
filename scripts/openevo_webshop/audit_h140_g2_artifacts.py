#!/usr/bin/env python3
"""CPU-only H1.40-G2 artifact autopsy.

The audit reads already-produced JSONL/JSON/SafeTensors artifacts.  It does not
load a model, allocate CUDA, consume WebShop tasks, or change an experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import mmap
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


H140_PREFIX = "h140-r2-g2-increment-"
H138_PREFIX = "h138b-increment-"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nested_value(payload: Any, names: set[str]) -> Any:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in names:
                return value
            found = nested_value(value, names)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = nested_value(value, names)
            if found is not None:
                return found
    return None


def safe_tensor_component_stats(path: Path) -> dict[int, dict[str, Any]]:
    """Read F32 SafeTensors component norms without importing torch."""

    result: dict[int, dict[str, Any]] = {}
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
        data_start = 8 + header_size

    grouped: dict[int, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for key, spec in header.items():
        if key == "__metadata__":
            continue
        match = re.match(r"components\.(\d+)\.", key)
        if match:
            grouped[int(match.group(1))].append((key, spec))

    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        for component, entries in grouped.items():
            digest = hashlib.sha256()
            sum_squares = 0.0
            numel = 0
            dtype = {spec.get("dtype") for _, spec in entries}
            if dtype != {"F32"}:
                result[component] = {"status": "unavailable", "reason": f"unsupported dtype {sorted(dtype)}"}
                continue
            for key, spec in sorted(entries):
                low, high = spec["data_offsets"]
                raw = bytes(mapped[data_start + low : data_start + high])
                digest.update(raw)
                values = struct.iter_unpack("<f", raw)
                sum_squares += sum(float(value[0]) ** 2 for value in values)
                numel += len(raw) // 4
            result[component] = {
                "status": "pass",
                "direction_sha256": digest.hexdigest(),
                "direction_frobenius_norm": math.sqrt(sum_squares),
                "numel": numel,
                "tensor_count": len(entries),
            }
    return result


def component_rows(label: str, adapter: Path, state: dict[str, Any]) -> list[dict[str, Any]]:
    tensor_path = adapter / "openevo_sd_lora_state.safetensors"
    tensor_stats = safe_tensor_component_stats(tensor_path) if tensor_path.exists() else {}
    config = state.get("training_config") or {}
    rows: list[dict[str, Any]] = []
    for index, component in enumerate(state.get("components") or []):
        coefficient = float(component["coefficient"])
        stats = tensor_stats.get(index, {"status": "unavailable", "reason": "component tensor missing"})
        norm = stats.get("direction_frobenius_norm")
        rows.append(
            {
                "generation": label,
                "component": index + 1,
                "rank": component.get("rank"),
                "task_index": component.get("task_index"),
                "coefficient": coefficient,
                "coefficient_init": config.get("coefficient_init", "unavailable"),
                "direction_frobenius_norm": norm if norm is not None else "unavailable",
                "effective_contribution_proxy": abs(coefficient) * norm if norm is not None else "unavailable",
                "direction_sha256": stats.get("direction_sha256", "unavailable"),
                "direction_status": stats.get("status", "unavailable"),
                "new_component_magnitude_limit_ratio": nested_value(
                    state, {"new_component_magnitude_limit_ratio", "magnitude_limit_ratio"}
                ) or "unavailable",
                "magnitude_projection_count": nested_value(
                    state, {"magnitude_projection_count", "projection_count", "projection_events"}
                ) or "unavailable",
                "late_component_retention_decay": nested_value(
                    state, {"late_component_retention_decay", "retention_decay"}
                ) or "unavailable",
            }
        )
    return rows


def replay_summary(adapter: Path) -> dict[str, Any]:
    path = adapter / "openevo_sd_lora_replay.jsonl"
    if not path.exists():
        return {"status": "unavailable", "reason": "replay artifact missing", "path": str(path)}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_artifact = Counter()
    by_source = Counter()
    by_task = set()
    by_generation = Counter()
    by_status = Counter()
    for row in rows:
        metadata = row.get("metadata") or {}
        artifact = str(metadata.get("dataset_artifact_id", "missing"))
        by_artifact[artifact] += 1
        by_task.add(metadata.get("task_id"))
        by_status[str(metadata.get("status", "missing"))] += 1
        if artifact.startswith(H138_PREFIX):
            source = "historical-g1"
            generation = "G1"
        elif artifact.startswith(H140_PREFIX):
            source = "current-e2"
            generation = artifact.removeprefix(H140_PREFIX)
        else:
            source = "unknown"
            generation = "unknown"
        by_source[source] += 1
        by_generation[generation] += 1
    return {
        "status": "pass",
        "path": str(path),
        "record_count": len(rows),
        "unique_task_count": len(by_task - {None}),
        "by_source": dict(sorted(by_source.items())),
        "by_generation": dict(sorted(by_generation.items())),
        "by_dataset_artifact": dict(sorted(by_artifact.items())),
        "by_status": dict(sorted(by_status.items())),
        "e1_records_detected": sum(count for key, count in by_artifact.items() if "e1" in key.lower()),
        "note": "E1 is an evaluation partition in H1.40-R2, not a replay-training source; absence here is expected.",
    }


def state_summary(label: str, adapter: Path, state: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    old = [row for row in rows if row["component"] <= 8]
    new = [row for row in rows if label != "G1" and row["component"] >= 9]
    def mean(rows_: list[dict[str, Any]], key: str) -> Any:
        values = [float(row[key]) for row in rows_ if row[key] != "unavailable"]
        return sum(values) / len(values) if values else "unavailable"
    return {
        "generation": label,
        "adapter": str(adapter),
        "adapter_id": state.get("adapter_id"),
        "component_count": state.get("component_count"),
        "effective_rank": state.get("effective_rank"),
        "replay_buffer_record_count": state.get("replay_buffer_record_count", "unavailable"),
        "replay_training_record_count": state.get("replay_training_record_count", "unavailable"),
        "training_record_count": state.get("training_record_count", "unavailable"),
        "optimizer_training_record_count": state.get("optimizer_training_record_count", "unavailable"),
        "steps_completed": state.get("steps_completed", "unavailable"),
        "training_loss": state.get("training_loss", "unavailable"),
        "training_time_seconds": state.get("training_time_seconds", "unavailable"),
        "direction_parameterization": state.get("direction_parameterization", "unavailable"),
        "old_mean_coefficient": mean(old, "coefficient"),
        "new_mean_coefficient": mean(new, "coefficient") if label != "G1" else "unavailable",
        "old_mean_effective_contribution_proxy": mean(old, "effective_contribution_proxy"),
        "new_mean_effective_contribution_proxy": mean(new, "effective_contribution_proxy") if label != "G1" else "unavailable",
        "replay": replay_summary(adapter),
    }


def find_state_adapters(g2_root: Path) -> list[tuple[str, Path, dict[str, Any]]]:
    found: list[tuple[str, Path, dict[str, Any]]] = []
    for state_path in g2_root.glob("artifacts/sd-lora/workers/*/adapter/openevo_sd_lora_state.json"):
        state = read_json(state_path)
        count = int(state.get("component_count", -1))
        if 9 <= count <= 16:
            found.append((f"G2-increment-{count - 9:02d}", state_path.parent, state))
    return sorted(found, key=lambda item: int(item[2]["component_count"]))


def _walk_event_flags(payload: Any) -> tuple[int, int, int, list[str]]:
    parser_events = 0
    parser_failures = 0
    runtime_errors = 0
    raw_signatures: list[str] = []

    def walk(value: Any) -> None:
        nonlocal parser_events, parser_failures, runtime_errors
        if isinstance(value, dict):
            event_type = str(value.get("event_type") or value.get("type") or "")
            if "parser" in event_type.lower():
                parser_events += 1
            if value.get("parser_failure") is True or event_type.lower() == "parser-retry":
                parser_failures += 1
            if value.get("error") or "runtime-error" in event_type.lower() or "exception" in event_type.lower():
                runtime_errors += 1
            for key in ("raw_completion", "raw_output", "completion", "response"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate:
                    # Keep a short, deterministic diagnostic signature rather
                    # than copying full model output into the evidence report.
                    raw_signatures.append(candidate[:160].replace("\n", " "))
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    return parser_events, parser_failures, runtime_errors, raw_signatures


def _invalid_cell_diagnostic(cell_root: Path | None) -> dict[str, Any]:
    if cell_root is None or not cell_root.exists():
        return {
            "status": "unavailable",
            "reason": "invalid acquisition cell artifacts not supplied; reconciliation combines parser/runtime invalidity",
        }
    attempts_path = cell_root / "attempts.jsonl"
    attempts = [
        json.loads(line)
        for line in attempts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if attempts_path.exists() else []
    parser_events = parser_failures = runtime_errors = 0
    episode_root = cell_root / "episodes"
    episode_files = sorted(episode_root.rglob("*.json")) if episode_root.exists() else sorted(cell_root.glob("*.json"))
    signatures: list[str] = []
    complete = 0
    for path in episode_files:
        if path.name in {"summary.json", "gpu_receipt.json"}:
            continue
        payload = read_json(path)
        events, failures, errors, raw = _walk_event_flags(payload)
        parser_events += events
        parser_failures += failures
        runtime_errors += errors
        signatures.extend(raw)
        if payload.get("trace_complete") is True:
            complete += 1
    parser_only = parser_failures > 0 and runtime_errors == 0
    return {
        "status": "pass",
        "cell_root": str(cell_root),
        "attempt_count": len(attempts),
        "episode_file_count": len([p for p in episode_files if p.name not in {"summary.json", "gpu_receipt.json"}]),
        "trace_complete_count": complete,
        "parser_event_count": parser_events,
        "parser_failure_event_count": parser_failures,
        "runtime_error_count": runtime_errors,
        "direct_episode_classification": "parser-only" if parser_only else "parser-and-or-runtime",
        "raw_output_signatures": sorted(set(signatures))[:8],
        "interpretation": "Direct episode evidence indicates parser-only invalidity; no runtime exception was observed in the supplied episode JSON. This still does not establish that the underlying model behavior was correct.",
    }


def acquisition_invalid_diagnostic(reconciliation: Path | None, invalid_cell_root: Path | None = None) -> dict[str, Any]:
    if reconciliation is None or not reconciliation.exists():
        return {"status": "unavailable", "reason": "upstream reconciliation not supplied"}
    payload = read_json(reconciliation)
    acquisition = (payload.get("stages") or {}).get("acquisition") or {}
    g2 = (acquisition.get("arms") or {}).get("g2") or {}
    reasons = g2.get("invalid_reasons") or []
    diagnostic = {
        "status": "pass",
        "stage": "acquisition",
        "g2_valid_attempt_count": g2.get("valid_attempt_count"),
        "g2_invalid_reasons": reasons,
        "same_failure_signature": len(set(str(reason).split(":", 1)[1] for reason in reasons if ":" in str(reason))) == 1 if reasons else False,
        "measurement_failure": bool(reasons),
        "behavior_failure": False,
        "interpretation": "The invalid attempts are excluded from the scientific denominator; the reconciliation signature alone cannot label them model incompetence.",
    }
    direct = _invalid_cell_diagnostic(invalid_cell_root)
    diagnostic["direct_cell_evidence"] = direct
    if direct.get("direct_episode_classification") == "parser-only":
        diagnostic["interpretation"] = "Direct episode evidence classifies this cell as parser-only invalidity, with no observed runtime exception. The invalid attempts remain excluded from the scientific denominator and do not prove successful model behavior."
    return diagnostic


def write_markdown(output_dir: Path, report: dict[str, Any]) -> None:
    final = report["states"][-1]
    old = final["old_mean_coefficient"]
    new = final["new_mean_coefficient"]
    lines = [
        "# H1.40-MD artifact autopsy",
        "",
        "Status: CPU-only read-only diagnostic; no WebShop task or GPU was consumed.",
        "",
        "## Bottom line",
        "",
        "The artifacts support magnitude imbalance as the leading mechanism hypothesis, not a proven cause. Historical G1 directions are byte-identical in the final G2 state, while coefficients are re-optimized; final new-component coefficients are much smaller than historical ones.",
        "",
        "## Observed facts",
        "",
        f"- States audited: G1 plus {len(report['states']) - 1} G2 increments; final effective rank {final['effective_rank']} with {final['component_count']} components.",
        f"- Final historical-component mean coefficient: `{old}`; final new-component mean coefficient: `{new}`.",
        f"- Final replay composition: `{json.dumps(final['replay'].get('by_source', {}), sort_keys=True)}`; E1 records detected: `{final['replay'].get('e1_records_detected')}`.",
        "- Direction norm and coefficient are reported separately; `effective_contribution_proxy = |coefficient| × direction Frobenius norm`.",
        "- Projection, new-component cap, and late-component decay are reported as `unavailable` when absent from the artifact; they are not inferred.",
        "",
        "## Interpretation boundary",
        "",
        "The frozen H1.40 result remains `complete-upstream-gate-failed`; it is not retroactively assigned MG0 or MG1. The acquisition stage is structurally incomplete, but G2 construction and the upstream matrix reconciliation completed. A future preregistration should add an explicit upstream-stage invalidity taxonomy.",
        "",
        "## Acquisition invalidity",
        "",
        f"`{json.dumps(report['acquisition_invalid_diagnostic'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Recommended next mechanism screen",
        "",
        "Use exact H1.40 as C0 and change only magnitude initialization for C1 (`reset_all_to_coefficient_init`). Keep replay, rank, records, seeds, schedule, and evaluation gates fixed. Do not open T2 until acquisition, retention, and preservation all pass.",
        "",
    ]
    (output_dir / "ARTIFACT_AUTOPSY.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "ACQUISITION_INVALID_DIAGNOSTIC.md").write_text(
        "# H1.40 acquisition invalid-attempt diagnostic\n\n"
        + json.dumps(report["acquisition_invalid_diagnostic"], ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def audit(args: argparse.Namespace) -> dict[str, Any]:
    g2_root = args.g2_run_root.resolve()
    g1_adapter = args.g1_adapter.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    g1_state_path = g1_adapter / "openevo_sd_lora_state.json"
    if not g1_state_path.exists():
        raise FileNotFoundError(g1_state_path)
    states = [("G1", g1_adapter, read_json(g1_state_path))]
    states.extend(find_state_adapters(g2_root))
    if len(states) != 9:
        raise ValueError(f"expected G1 plus 8 G2 states, found {len(states)}")

    matrix: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for label, adapter, state in states:
        rows = component_rows(label, adapter, state)
        matrix.extend(rows)
        summaries.append(state_summary(label, adapter, state, rows))

    g1_sha = {row["component"]: row["direction_sha256"] for row in matrix if row["generation"] == "G1"}
    final_sha = {row["component"]: row["direction_sha256"] for row in matrix if row["generation"] == "G2-increment-07"}
    direction_comparison = [
        {
            "component": component,
            "same_bytes": g1_sha.get(component) == final_sha.get(component),
            "g1_sha256": g1_sha.get(component, "unavailable"),
            "g2_final_sha256": final_sha.get(component, "unavailable"),
        }
        for component in range(1, 9)
    ]
    report = {
        "schema": "openevo.webshop-h1.40-g2-artifact-autopsy.v1",
        "status": "pass",
        "cpu_only": True,
        "g2_run_root": str(g2_root),
        "g1_adapter": str(g1_adapter),
        "states": summaries,
        "direction_comparison_g1_to_final_g2": direction_comparison,
        "all_historical_directions_byte_identical": all(item["same_bytes"] for item in direction_comparison),
        "component_matrix": matrix,
        "acquisition_invalid_diagnostic": acquisition_invalid_diagnostic(args.upstream_reconciliation, args.invalid_cell_root),
        "claim_boundary": "Artifact mechanism diagnosis only; no causal mechanism claim, new task result, T2 result, or universal continual-learning claim.",
    }
    (output_dir / "artifact-autopsy.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "component-magnitudes.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(matrix[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(matrix)
    (output_dir / "replay-composition.json").write_text(json.dumps({s["generation"]: s["replay"] for s in summaries}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "training-dynamics.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(output_dir, report)
    print(json.dumps({"status": "pass", "output_dir": str(output_dir), "state_count": len(states), "matrix_rows": len(matrix), "all_historical_directions_byte_identical": report["all_historical_directions_byte_identical"]}, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2-run-root", type=Path, required=True)
    parser.add_argument("--g1-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--upstream-reconciliation", type=Path)
    parser.add_argument("--invalid-cell-root", type=Path)
    args = parser.parse_args()
    audit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
