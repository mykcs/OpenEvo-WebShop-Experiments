#!/usr/bin/env python3
"""Freeze H1.38A diversity datasets and a fresh held-out WebShop manifest.

The selection algorithm intentionally uses only the sealed reservoir analysis:
task IDs are ordered by a domain-separated SHA-256 key and successes within an
identity are ordered by their canonical full-record SHA-256.  Raw harvest rows
are then required to reproduce those hashes before their traces can be used.
"""

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
except ModuleNotFoundError:  # pragma: no cover
    from build_h1_30_manifest import collect_prior_indices  # type: ignore
    from webshop_task_manifest import canonical_bytes, live_goals, sha256_text  # type: ignore


SCHEMA = "openevo.webshop-h1.38a-pooled-diversity-inputs.v1"
RESERVOIR_ANALYSIS_SHA256 = "d433992f243ab80304caa8bf615185d1d300a080f82218de9d2932eb0f8d4b40"
IDENTITY_KEY_DOMAIN = "h138a-identity-v1"
EVALUATION_TASK_COUNT = 48
EVALUATION_SELECTION_SEED = 20260818138
EVALUATION_SEEDS = [2026090500, 2026090501, 2026090502, 2026090503]
EXPECTED_PRIOR_COUNT = 445
EXPECTED_PRIOR_DIGEST = "48d59ea7a8f3aeee8e760e26a6650c3036ba052bc0016fdb54719c7eb3526fdc"
ARM_RECORDS = {"diversity-4": (4, 4), "diversity-8": (8, 2), "diversity-16": (16, 1)}


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _record_hash(row: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _index_digest(indices: set[int]) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in sorted(indices)).encode()).hexdigest()


def _identity_order(task_id: str) -> str:
    return hashlib.sha256(f"{IDENTITY_KEY_DOMAIN}:{task_id}".encode()).hexdigest()


def select_records(analysis: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Select 16 qualified record descriptors per arm, without raw trace access."""
    ledger = analysis.get("identity_ledger")
    if not isinstance(ledger, dict):
        raise ValueError("reservoir analysis lacks identity ledger")
    selections: dict[str, list[dict[str, Any]]] = {}
    for arm, (identity_count, records_per_identity) in ARM_RECORDS.items():
        eligible: list[tuple[str, dict[str, Any]]] = []
        for task_id, item in ledger.items():
            if isinstance(item, dict) and int(item.get("qualified_successes") or 0) >= records_per_identity:
                eligible.append((str(task_id), item))
        eligible.sort(key=lambda pair: _identity_order(pair[0]))
        selected = eligible[:identity_count]
        if len(selected) != identity_count:
            raise ValueError(f"{arm} lacks {identity_count} eligible identities")
        rows: list[dict[str, Any]] = []
        for task_id, item in selected:
            hashes = item.get("qualified_success_record_sha256")
            seeds = item.get("qualified_success_rollout_seeds")
            if not isinstance(hashes, list) or not isinstance(seeds, list) or len(hashes) != len(seeds):
                raise ValueError(f"{task_id} has malformed success ledger")
            pairs = sorted((str(digest), int(seed)) for digest, seed in zip(hashes, seeds))[:records_per_identity]
            if len(pairs) != records_per_identity:
                raise ValueError(f"{task_id} has insufficient records")
            for digest, seed in pairs:
                rows.append({"task_id": task_id, "goal_idx": int(item["goal_idx"]), "rollout_seed": seed, "record_sha256": digest, "task_description_sha256": item.get("task_description_sha256")})
        if len(rows) != 16 or len({row["task_id"] for row in rows}) != identity_count:
            raise ValueError(f"{arm} record/identity count mismatch")
        selections[arm] = sorted(rows, key=lambda row: (row["task_id"], row["rollout_seed"], row["record_sha256"]))
    return selections


def _load_attempt_rows(harvest_root: Path, selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {(int(row["goal_idx"]), int(row["rollout_seed"])): row for row in selected}
    found: dict[tuple[int, int], dict[str, Any]] = {}
    for goal_idx in sorted({key[0] for key in wanted}):
        attempts = harvest_root / "harvest" / f"task{goal_idx}" / "attempts.jsonl"
        if not attempts.is_file():
            raise ValueError(f"missing reservoir attempts: {attempts}")
        for line in attempts.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row.get("goal_idx", -1)), int(row.get("rollout_seed", -1)))
            if key not in wanted:
                continue
            if key in found:
                raise ValueError(f"duplicate reservoir attempt {key}")
            expected = wanted[key]
            if _record_hash(row) != expected["record_sha256"]:
                raise ValueError(f"reservoir record hash mismatch for {key}")
            if not (bool(row.get("identity_ok")) and bool(row.get("trace_complete")) and not bool(row.get("parser_or_fallback_failure")) and row.get("error") is None and float(row.get("final_task_score") or 0.0) > 0):
                raise ValueError(f"selected reservoir row is not qualified: {key}")
            found[key] = row
    if set(found) != set(wanted):
        raise ValueError("selected reservoir record not found")
    return [found[(int(item["goal_idx"]), int(item["rollout_seed"]))] for item in selected]


def _dataset_record(row: dict[str, Any]) -> dict[str, Any]:
    trace = ((row.get("episode") or {}).get("trace") or [])
    if not isinstance(trace, list) or not trace:
        raise ValueError("qualified reservoir record lacks trace")
    traces: list[dict[str, Any]] = []
    for index, step in enumerate(trace):
        if not isinstance(step, dict) or not bool(step.get("action_valid")) or bool(step.get("used_fallback")) or bool(step.get("parser_failure")):
            raise ValueError("selected reservoir trace contains invalid action")
        prompt, completion = step.get("prompt"), step.get("raw_completion")
        if not isinstance(prompt, str) or not prompt.strip() or not isinstance(completion, str) or not completion.strip():
            raise ValueError("selected reservoir trace lacks prompt/completion")
        traces.append({"prompt_messages": [{"role": "user", "content": prompt}], "response_messages": [{"role": "assistant", "content": completion}], "metadata": {"source_step": step.get("step", index), "generation_seed": step.get("generation_seed"), "environment_seed": step.get("environment_seed"), "action": step.get("action")}})
    task_id, seed = str(row["task_id"]), int(row["rollout_seed"])
    return {"event_id": f"h138a:{task_id}:{seed}:{_record_hash(row)[:16]}", "task_id": task_id, "session_id": f"{task_id}:{seed}", "status": "completed", "reward": float(row.get("final_task_score") or 0.0), "traces": traces, "metadata": {"source_route": "qwen2.5-7b-endogenous-reservoir-v2", "qualified_positive": True, "rollout_seed": seed, "source_task_id": task_id, "source_record_sha256": _record_hash(row)}}


def write_datasets(analysis: dict[str, Any], harvest_root: Path, output_root: Path) -> dict[str, Any]:
    selections = select_records(analysis)
    results: dict[str, Any] = {}
    for arm, selected in selections.items():
        source = _load_attempt_rows(harvest_root, selected)
        records = [_dataset_record(row) for row in source]
        # Match the pinned OpenEvo strict-snapshot serializer byte-for-byte.
        record_bytes = b"".join(json.dumps(row, sort_keys=True, allow_nan=False).encode() + b"\n" for row in records)
        arm_dir = output_root / arm
        arm_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        records_path = arm_dir / "records.jsonl"
        records_path.write_bytes(record_bytes)
        records_path.chmod(0o600)
        manifest = {"schema": "openevo.dataset.v1", "name": f"h138a-{arm}-endogenous-diversity", "source": "qwen2.5-7b-endogenous-reservoir-v2", "teacher_identity": None, "record_count": len(records), "identity_count": len({row["task_id"] for row in records}), "trace_count": sum(len(row["traces"]) for row in records), "records_path": "records.jsonl", "records_uri": records_path.resolve().as_uri(), "records_byte_size": len(record_bytes), "records_sha256": hashlib.sha256(record_bytes).hexdigest(), "selection": selections[arm], "selection_sha256": _canonical_hash(selections[arm]), "reservoir_analysis_sha256": RESERVOIR_ANALYSIS_SHA256, "source_route": "self-generated only; no teacher or held-out evaluation"}
        manifest_path = arm_dir / "manifest.json"
        manifest_path.write_bytes(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False).encode())
        manifest_path.chmod(0o600)
        results[arm] = {"manifest": str(manifest_path), "manifest_file_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "records_sha256": manifest["records_sha256"], "selection_sha256": manifest["selection_sha256"], "record_count": 16, "identity_count": manifest["identity_count"]}
    return results


def build_evaluation_manifest(goals: list[dict[str, Any]], *, prior_indices: set[int], experiment_git_sha: str) -> dict[str, Any]:
    if len(prior_indices) != EXPECTED_PRIOR_COUNT or _index_digest(prior_indices) != EXPECTED_PRIOR_DIGEST:
        raise ValueError("prior task identity exclusion set drifted")
    pool = [idx for idx in range(500, len(goals)) if idx not in prior_indices]
    random.Random(EVALUATION_SELECTION_SEED).shuffle(pool)
    ids = sorted(pool[:EVALUATION_TASK_COUNT])
    if len(ids) != EVALUATION_TASK_COUNT or set(ids) & prior_indices:
        raise ValueError("fresh H1.38A evaluation selection failed")
    tasks = []
    for sample_index, goal_idx in enumerate(ids):
        description = str(goals[goal_idx].get("instruction_text") or "").strip()
        if not description:
            raise ValueError(f"empty WebShop task {goal_idx}")
        tasks.append({"sample_index": sample_index, "task_id": f"webshop_{goal_idx:06d}", "goal_idx": goal_idx, "split": "h1.38a-fresh-heldout-evaluation", "task_description": description, "task_description_sha256": sha256_text(description)})
    document: dict[str, Any] = {"schema": "openevo.webshop-h1.38a-pooled-diversity-eval-manifest.v1", "campaign_id": "20260819-0015-h138a-pooled-diversity", "source": {"experiment_git_sha": experiment_git_sha, "reservoir_analysis_sha256": RESERVOIR_ANALYSIS_SHA256}, "evaluation_tasks": ids, "tasks": tasks, "evaluation": {"arms": ["base", *ARM_RECORDS], "rollout_seeds": EVALUATION_SEEDS, "cells": len(ids) * 4, "attempts": len(ids) * 4 * len(EVALUATION_SEEDS)}, "task_selection": {"selection_seed": EVALUATION_SELECTION_SEED, "prior_identity_count": len(prior_indices), "prior_identity_digest": _index_digest(prior_indices), "policy": "exclude all prior committed identities, both reservoir task universes, and the aborted pre-protocol reservoir ledger"}, "contamination": {"training_evaluation_overlap": False, "evaluation_outcomes_used_for_training": False, "teacher_oracle_gpt_used": False, "protocol_class": "pooled_endogenous_task_identity_diversity", "upstream_v16_equivalent": False}, "manifest_sha256": ""}
    document["manifest_sha256"] = _canonical_hash({key: value for key, value in document.items() if key != "manifest_sha256"})
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--harvest-root", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--seed-root", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument("--prior-root", type=Path)
    parser.add_argument("--aborted-ledger", type=Path)
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument("--experiment-git-sha")
    args = parser.parse_args()
    raw = args.analysis.read_bytes()
    if hashlib.sha256(raw).hexdigest() != RESERVOIR_ANALYSIS_SHA256:
        raise SystemExit("sealed reservoir analysis sha256 mismatch")
    analysis = json.loads(raw)
    output: dict[str, Any] = {"schema": SCHEMA, "selection_preview": select_records(analysis)}
    if args.harvest_root or args.dataset_root:
        if not args.harvest_root or not args.dataset_root:
            raise SystemExit("--harvest-root and --dataset-root are required together")
        output["datasets"] = write_datasets(analysis, args.harvest_root.resolve(), args.dataset_root.resolve())
    if args.evaluation_manifest or args.seed_root or args.data_dir or args.index_dir or args.prior_root or args.aborted_ledger:
        required = (args.evaluation_manifest, args.seed_root, args.data_dir, args.index_dir, args.prior_root, args.aborted_ledger, args.experiment_git_sha)
        if not all(required):
            raise SystemExit("evaluation manifest arguments must be supplied together")
        ledger = json.loads(args.aborted_ledger.read_text(encoding="utf-8"))
        extra = ledger.get("task_ids") if isinstance(ledger, dict) else None
        if not isinstance(extra, list) or len(extra) != 64:
            raise SystemExit("aborted reservoir ledger malformed")
        goals, _ = live_goals(args.seed_root.resolve(), 2026, data_dir=args.data_dir.resolve(), index_dir=args.index_dir.resolve())
        manifest = build_evaluation_manifest(goals, prior_indices=collect_prior_indices([args.prior_root.resolve()]) | set(int(item) for item in extra), experiment_git_sha=args.experiment_git_sha)
        args.evaluation_manifest.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        args.evaluation_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        output["evaluation_manifest"] = {"path": str(args.evaluation_manifest), "manifest_sha256": manifest["manifest_sha256"], "file_sha256": hashlib.sha256(args.evaluation_manifest.read_bytes()).hexdigest(), "tasks": manifest["evaluation_tasks"]}
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
