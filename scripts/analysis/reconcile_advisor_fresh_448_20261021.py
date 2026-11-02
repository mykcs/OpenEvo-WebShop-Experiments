#!/usr/bin/env python3
"""Read-only final WebShop reconciliation: seven frozen checkpoints x 64 tasks.

Runs on the established experiment-host research workspace. No GPU, generation, training,
resumption, task writes, panel selection or server output mutation. Uses the
original frozen preregistration for every statistical hyperparameter.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random

WORKSPACE = Path("/workspace")
EXECUTION_SOURCE = WORKSPACE / "control/advisor-fresh-followups-20261020-source"
MANIFEST_REL = "configs/experiment/manifests/advisor-fresh-checkpoints-20261020.json"
PREREG_REL = "configs/experiment/preregistrations/advisor-fresh-checkpoints-20261020.json"

# Run identities are immutable historical facts from the two canonical launch waves.
LANES = (
    ("advisor-fresh-202610201610-gpu1", ("bounded150", "ordinary120")),
    ("advisor-fresh-202610201610-gpu3", ("bounded160",)),
    ("advisor-fresh-202610201610-gpu5", ("gdr130", "gdr160")),
    ("advisor-fresh-202610201758-ordinary-gpu5", ("ordinary160",)),
    ("advisor-fresh-202610201758-reverse-gpu3", ("reverse156",)),
)


def require(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError("Fresh 448 reconciliation failed: " + message)


def load(path: Path) -> dict:
    return json.loads(path.read_bytes())


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def percentile(sorted_numbers: list[float], p: float) -> float:
    """Linear interpolation between the (n-1)*p adjacent order statistics."""
    require(len(sorted_numbers) > 0 and 0 <= p <= 1, "percentile input")
    point = (len(sorted_numbers) - 1) * p
    lo, hi = math.floor(point), math.ceil(point)
    return (sorted_numbers[lo] + (sorted_numbers[hi] - sorted_numbers[lo]) * (point - lo))


def paired_bootstrap(deltas: list[list[float]], draws: int, seed: int,
                     tail: float) -> list[tuple[float, float]]:
    """One shared resampled 64-task index vector for all 8 endpoints per draw."""
    require(len(deltas) == 8 and all(len(x) == 64 for x in deltas),
            "expected 4 contrasts times 2 paired endpoints on 64 tasks")
    require(draws == 50000 and seed == 20261020 and tail == 0.003125,
            "prospective bootstrap settings drift")
    rng = random.Random(seed)
    boot = [[] for _ in deltas]
    for _ in range(draws):
        sample = [rng.randrange(64) for _ in range(64)]
        for b, ds in zip(boot, deltas):
            b.append(sum(ds[i] for i in sample) / 64)
    return [(percentile(sorted(b), tail), percentile(sorted(b), 1-tail)) for b in boot]


def reconcile(workspace: Path = WORKSPACE, source: Path = EXECUTION_SOURCE) -> dict:
    manifest_path = source / MANIFEST_REL
    prereg_path = source / PREREG_REL
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    prereg = load(prereg_path)
    manifest_sha = digest(manifest_bytes)
    require(manifest_sha == prereg["manifest_sha256"],
            "frozen prospective bytes differ from preregistration")
    # The 2026-10-20 execution checkout predates the additional canonical
    # frozen_scientific_contract wrapper introduced on the PR branch. The
    # scientific manifest/analysis settings are the same pinned bytes.
    frozen_wrapper = prereg.get("frozen_scientific_contract")
    if frozen_wrapper is not None:
        require(frozen_wrapper["manifest"]["sha256"] == manifest_sha,
                "optional canonical wrapper disagrees with frozen manifest")
    require(len(manifest["tasks"]) == prereg["episodes_per_checkpoint"] == 64,
            "frozen 64-task count drift")
    require(len(manifest["checkpoints"]) == prereg["checkpoint_count"] == 7 and
            prereg["episodes"] == 448 and prereg["joint_family_endpoint_count"] == 8,
            "frozen checkpoint or family size drift")
    require(prereg["endpoints"] == ["task_score", "success"] and
            len(prereg["contrasts"]) == 4, "frozen contrast/endpoints drift")
    require(prereg["status"] == "PROSPECTIVELY_FROZEN_BEFORE_ANY_MODEL_CALL",
            "frozen prereg status drift")
    require(prereg["final_panel_access"] == prereg["optimizer_updates"] ==
            prereg["teacher_calls"] == 0 and
            manifest["final_panel_access_allowed"] is False and
            manifest["optimizer_updates"] == 0, "frozen no-training/no-final-panel contract drift")
    require(prereg["analysis"]["paired_bootstrap_draws"] == 50000 and
            prereg["analysis"]["quantile_tail_probability"] == 0.003125 and
            prereg["analysis"]["seed"] == 20261020, "frozen statistical settings drift")
    expected_panel_sha = digest(json.dumps(manifest["tasks"], ensure_ascii=False,
                                 sort_keys=True, separators=(",", ":")).encode())
    require(manifest["panel_sha256"] == expected_panel_sha, "panel task content drift")
    expected_tasks = {row["probe_ordinal"]: row for row in manifest["tasks"]}
    require(set(expected_tasks) == set(range(64)), "frozen ordinals missing or repeated")
    metrics = {}
    arrays = {}
    provenance = {}
    all_rows = 0
    for run_id, checkpoints in LANES:
        lane_root = workspace / "runs" / run_id / "evaluation"
        lane_start_path = lane_root / "START.json"
        lane_start = load(lane_start_path)
        formal = lane_start["formal_start"]
        selected = lane_start["checkpoint_ids"]
        # GPU3 originally named reverse156 as a prospective second cell, but
        # no task START occurred there. Its separate 17:58 successor is the
        # actual and only 64-task reverse156 execution.
        if run_id == "advisor-fresh-202610201610-gpu3":
            unstarted = lane_root / "reverse156"
            require(selected == ["bounded160", "reverse156"] and
                    unstarted.is_dir() and not list(unstarted.iterdir()),
                    "GPU3 initial reverse156 must be an unstarted empty cell")
        else:
            require(selected == list(checkpoints),
                    f"{run_id}: selected checkpoint set/order changed")
        require(lane_start["manifest_sha256"] == manifest_sha,
                f"{run_id}: lane manifest differs")
        require(formal["formal_start_committed"] is True and
                formal["formal_task_consumption_may_begin"] is True and
                formal["run_id"] == run_id and
                formal["selected_gpu_uuids"] == [lane_start["gpu_uuid"]] and
                formal["image_id"] == manifest["image_id"],
                f"{run_id}: canonical Formal Start identity drift")
        provenance[run_id] = {
            "formal_start_receipt_sha256": formal.get("receipt_sha256"),
            "lane_start_file_sha256": digest(lane_start_path.read_bytes()),
            "gpu_uuid": lane_start["gpu_uuid"],
            "checkpoints": list(checkpoints),
            "original_lane_selection": selected,
        }
        for checkpoint in checkpoints:
            root = lane_root / checkpoint
            report_path = root / "COMPLETE.json"
            raw = report_path.read_bytes()
            report = json.loads(raw)
            rows = report["rows"]
            require(report["status"] == "COMPLETE" and
                    report["checkpoint_id"] == checkpoint and
                    report["tasks"] == len(rows) == 64 and
                    report["manifest_sha256"] == manifest_sha and
                    report["gpu_uuid"] == lane_start["gpu_uuid"] and
                    report["optimizer_updates"] == report["final_panel_access"] == 0 and
                    report["source_files"] == manifest["source_files"],
                    f"{checkpoint}: final science receipt/identity drift")
            task_dirs = {x.name for x in root.glob("task-*") if x.is_dir()}
            require(task_dirs == {f"task-{i:03d}" for i in range(64)},
                    f"{checkpoint}: task set differs from 64 frozen ordinals")
            require((root / "START.json").is_file(), f"{checkpoint}: checkpoint start missing")
            lookup = {}
            for row in rows:
                ordinal = row["probe_ordinal"]
                require(ordinal in expected_tasks and ordinal not in lookup,
                        f"{checkpoint}: repeated or unregistered ordinal")
                for field in ("task_identity", "worker_seed", "paired_generation_seed",
                              "source_session_index", "instruction_sha256", "goal_sha256",
                              "instruction"):
                    require(row[field] == expected_tasks[ordinal][field],
                            f"{checkpoint}: {field} differs from frozen task {ordinal}")
                require(row["checkpoint_id"] == checkpoint and row["engineering_invalid"] is False,
                        f"{checkpoint}: row checkpoint/engineering drift")
                episode = row["episode"]
                require(isinstance(episode["won"], bool) and
                        isinstance(episode["score"], (float, int)) and
                        math.isfinite(episode["score"]) and
                        0 <= episode["score"] <= 1 and
                        episode["webshop_env_step_calls"] > 0 and
                        episode["environment_action_schema_violation"] is False and
                        episode["constraint_escape"] is False and
                        episode["invalid_terminated"] is False,
                        f"{checkpoint}: invalid WebShop science row {ordinal}")
                task_root = root / f"task-{ordinal:03d}"
                require(load(task_root / "START.json") == expected_tasks[ordinal],
                        f"{checkpoint}: per-task start does not match frozen task")
                require(load(task_root / "COMPLETE.json") == row,
                        f"{checkpoint}: per-task completion differs from aggregate")
                require(not (task_root / "ENGINEERING_FAILURE.json").exists(),
                        f"{checkpoint}: a task engineering failure exists")
                lookup[ordinal] = row
            require(set(lookup) == set(range(64)), f"{checkpoint}: 64 completion rows not exhaustive")
            seq = [lookup[i]["episode"] for i in range(64)]
            scores = [float(x["score"]) for x in seq]
            success = [int(x["won"]) for x in seq]
            require(math.isclose(sum(scores)/64, report["metrics"]["mean_score"], abs_tol=1e-12)
                    and math.isclose(sum(success)/64, report["metrics"]["success_rate"], abs_tol=1e-12)
                    and report["metrics"]["attempt_count"] == 64,
                    f"{checkpoint}: supplied summary differs from original 64 rows")
            arrays[checkpoint] = {"task_score": scores, "success": success}
            metrics[checkpoint] = {
                "completed": 64, "complete_sha256": digest(raw),
                "mean_task_score": sum(scores) / 64,
                "success_count": sum(success), "success_rate": sum(success) / 64,
                "engineering_failure_count": 0, "WebShop_environment_called_per_task": True,
            }
            all_rows += 64
    require(set(metrics) == set(manifest["checkpoints"]), "incomplete seven-checkpoint set")
    require(all_rows == prereg["episodes"] == 448, "not all 448 frozen tasks completed")

    deltas = []
    for late, early in prereg["contrasts"]:
        require(late in arrays and early in arrays, "preregistered checkpoint missing")
        for endpoint in prereg["endpoints"]:
            deltas.append([a-b for a,b in zip(arrays[late][endpoint], arrays[early][endpoint])])
    intervals = paired_bootstrap(deltas, prereg["analysis"]["paired_bootstrap_draws"],
                                 prereg["analysis"]["seed"],
                                 prereg["analysis"]["quantile_tail_probability"])
    contrasts = []
    for pair_index, (late, early) in enumerate(prereg["contrasts"]):
        entries = {}
        for endpoint_index, endpoint in enumerate(prereg["endpoints"]):
            k = pair_index * len(prereg["endpoints"]) + endpoint_index
            lower, upper = intervals[k]
            entries[endpoint] = {
                "late_minus_early": sum(deltas[k]) / 64,
                "familywise_corrected_lower": lower,
                "familywise_corrected_upper": upper,
                "paired_task_count": 64,
            }
        ordinary_comparison = pair_index < 3
        contrasts.append({
            "late": late, "early": early, "endpoints": entries,
            "early_stop_confirmed_under_preregistered_margins": (
                entries["task_score"]["familywise_corrected_upper"] <=
                prereg["analysis"]["score_noninferiority_margin"] and
                entries["success"]["familywise_corrected_upper"] <=
                prereg["analysis"]["success_noninferiority_margin"]
            ) if ordinary_comparison else None,
            "claim_limit": ("Reverse156 vs intact Bounded160, not direction vs norm causal isolation"
                            if not ordinary_comparison else
                            "Only this selected early/late pair, not the entire learning curve"),
        })
    return {
        "schema": "openevo.advisor-fresh-448-readonly-reconciliation.20261021.v1",
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_448_ACTUAL_WEBSHOP_TASKS",
        "zero_reexecuted_or_new_tasks_by_this_audit": True,
        "frozen_manifest_sha256": manifest_sha,
        "frozen_panel_sha256": manifest["panel_sha256"],
        "preregistration_sha256": digest(prereg_path.read_bytes()),
        "episodes": all_rows, "checkpoints": metrics, "formal_start_lanes": provenance,
        "statistical_method": {
            "paired_bootstrap_draws": prereg["analysis"]["paired_bootstrap_draws"],
            "seed": prereg["analysis"]["seed"],
            "shared_task_resamples_across_all_endpoints": True,
            "family_confidence": prereg["analysis"]["family_confidence"],
            "joint_family_endpoint_count": prereg["joint_family_endpoint_count"],
            "per_endpoint_two_sided_alpha": prereg["analysis"]["two_sided_per_endpoint_alpha"],
            "tail_probability": prereg["analysis"]["quantile_tail_probability"],
            "quantile_interpolation": "linear-(n-1)*p",
            "score_noninferiority_margin": prereg["analysis"]["score_noninferiority_margin"],
            "success_noninferiority_margin": prereg["analysis"]["success_noninferiority_margin"],
        },
        "contrasts": contrasts,
        "limitations": [
            "A complete pre-registered new-task assay is not proof of a global stopping optimum.",
            "Q06 synthetic scorer forward calls are not counted as WebShop tasks.",
            "This read-only analysis does not reassess live GPU occupancy or rerun any model.",
            "Reverse intervention has no matched removal control for directional causality.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--execution-source", type=Path, default=EXECUTION_SOURCE)
    args = parser.parse_args()
    print(json.dumps(reconcile(args.workspace, args.execution_source),
                     ensure_ascii=False, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
