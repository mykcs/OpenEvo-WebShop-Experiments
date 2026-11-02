#!/usr/bin/env python3
"""Export descriptive Q08 report artifacts without promoting partial learning evidence.

This CPU-only consumer takes already validated outcome rows and candidate audit
facts. It produces an interim report, never an execution authority or science seal.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from advisor_q08_retention_analysis import analyse, frozen_tasks


COMPLETE_PARTIAL = "EVALUATION_COMPLETE_SCIENCE_PARTIAL"


def bound_json(ref: dict) -> dict:
    root = Path(__file__).resolve().parents[2]
    path = (root / ref["path"]).resolve()
    path.relative_to(root)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ValueError("report source bytes changed")
    return json.loads(raw)


def bound_analyses(data: dict, seen: dict) -> dict:
    """Re-use audited training facts and bind the newly completed score rows."""
    results = {}
    for arm in ("ordinary", "bounded", "gdr"):
        ref = data["candidate_training_audits"][arm]
        payload = bound_json(ref)
        if payload["lineage_id"] != arm or payload["audit_status"] != ref["audit_status"]:
            raise ValueError("normalized training audit identity changed")
        expected_scores = {(r["task_id"], cp): (r["score"], r["evaluation_seed"], True)
                           for (lineage, cp), rows in seen.items() if lineage == arm for r in rows}
        observed_scores = {(r["task_id"], r["completed_rounds"]):
                           (r["score"], r["evaluation_seed"], r["engineering_valid"])
                           for r in payload["scores"]}
        if len(observed_scores) != len(payload["scores"]) or observed_scores != expected_scores:
            raise ValueError("normalized analysis scores differ from the verified report cells")
        results[arm] = analyse(payload, resamples=50000, seed=20261001)
    return results


def primary_scopes(analyses: dict) -> tuple[list[dict], list[dict]]:
    comparisons, diagnostics = [], []
    for arm, result in analyses.items():
        for cohort in ("EARLY_SEEN", "MID_SEEN", "LATE_SEEN"):
            row = {"lineage": arm, "cohort": cohort, "planned_tasks": 32}
            for metric in ("acquisition", "retention"):
                if result["status"] == "PARTIAL_AUDIT_INPUT":
                    row[metric] = {"status": "UNAVAILABLE_TRAINING_PROVENANCE_PARTIAL",
                                   "available_tasks": None, "mean": None, "ci95": None}
                    continue
                value = result["cohorts"][cohort][metric]
                if value["tasks"] == 32:
                    row[metric] = {**value, "available_tasks": 32}
                else:
                    row[metric] = {"status": "UNAVAILABLE_FIXED_COHORT_COVERAGE",
                                   "available_tasks": value["tasks"], "mean": None, "ci95": None}
                    if value["tasks"]:
                        diagnostics.append({"lineage": arm, "cohort": cohort, "metric": metric,
                                            "status": "DIAGNOSTIC_INCOMPLETE_COHORT",
                                            "planned_tasks": 32, "estimate": value})
            comparisons.append(row)
    return comparisons, diagnostics


def export(input_path: Path, output_dir: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    raw = input_path.read_bytes()
    data = json.loads(raw)
    complete = data.get("status") == COMPLETE_PARTIAL
    if not complete and data.get("status") != "INTERIM_RUNNING_NOT_SCIENTIFIC_SEAL":
        raise ValueError("explicit interim or completed evaluation/science-partial input required")
    expected = {t["task_id"]: t["cohort"] for t in frozen_tasks()}
    lineages = ("ordinary", "bounded", "gdr")
    checkpoints = (40, 80, 120, 160)
    cohorts = ("EARLY_SEEN", "MID_SEEN", "LATE_SEEN")
    seen = {}
    seeds = {}
    for cell in data["seen"]["cells"]:
        key = (cell["lineage"], cell["completed_rounds"])
        if key in seen or key[0] not in lineages or key[1] not in checkpoints:
            raise ValueError("duplicate or unknown seen cell")
        rows = cell["scores"]
        if len(rows) != 96 or len({r["task_id"] for r in rows}) != 96 or {r["task_id"]: r["cohort"] for r in rows} != expected:
            raise ValueError("seen cell does not contain the exact frozen96 task panel")
        for row in rows:
            if type(row["score"]) not in (int, float) or not math.isfinite(row["score"]) or not 0 <= row["score"] <= 1:
                raise ValueError("invalid Task Score")
            if type(row["evaluation_seed"]) is not int:
                raise ValueError("missing paired seed")
            seed_key = (key[0], row["task_id"])
            if seed_key in seeds and seeds[seed_key] != row["evaluation_seed"]:
                raise ValueError("seen seed changed across checkpoints")
            seeds[seed_key] = row["evaluation_seed"]
        seen[key] = rows
    if complete:
        required = {(a, cp) for a in lineages for cp in checkpoints} - {("ordinary", 40), ("ordinary", 120)}
        cost = data["seen"]["cost"]
        execution = bound_json(data["execution_audit"])
        closed_costs = bound_json(data["closed_cost_audit"])
        if (execution.get("status") != "TEN_COMPLETE_CELLS_EXECUTION_COMPLETE_SCIENCE_PARTIAL"
                or execution.get("source_sha") != data["execution_source_sha"]
                or execution.get("authority_sha") != data["authority_sha"]
                or execution.get("completed_seen_cells") != 10 or execution.get("completed_seen_episodes") != 960
                or execution.get("claim_count") != 10 or execution.get("claims_exactly_one_per_cell") is not True):
            raise ValueError("completed execution audit scope/claims do not bind this report")
        required_ids = {f"{arm}-r{cp:03d}" for arm, cp in required}
        sources = execution["all_cells_raw_sources"]
        if len(sources) != 10 or {s["cell_id"] for s in sources} != required_ids:
            raise ValueError("execution audit must bind exactly ten unique cell raw sources")
        if execution["whole_run_cost"] != cost or execution["all_cells_raw_sources"] != data["seen"]["raw_sources"]:
            raise ValueError("completed execution source/cost binding differs from report input")
        if (closed_costs.get("status") != "ALL_AVAILABLE_BOUND_LAUNCH_RECEIPTS_CLOSED"
                or closed_costs["formal_launch_gpu_hours"] != cost.get("gpu_hours")
                or not any(r["receipt"] == cost for r in closed_costs["deduplicated_launch_receipts"])):
            raise ValueError("formal cost reconciliation differs from native source")
        if (set(seen) != required or cost.get("status") != "succeeded" or type(cost.get("return_code")) is not int or cost["return_code"] != 0
                or not cost.get("end_utc") or type(cost.get("gpu_hours")) not in (int, float)
                or not math.isfinite(cost["gpu_hours"]) or cost["gpu_hours"] <= 0):
            raise ValueError("all ten completed cells and closed successful native costs required")
    fresh = data["fresh_reuse"]["cells"]
    if len(fresh) != 12 or {(c["lineage"], c["completed_rounds"]) for c in fresh} != {(a, cp) for a in lineages for cp in checkpoints}:
        raise ValueError("all12 exact historical fresh cells required")
    if any(len(c["rows"]) != 64 or c["engineering_invalid_tasks"] != 0 for c in fresh):
        raise ValueError("fresh reuse must contain768 valid historical rows")
    fresh_ref = data["fresh_reuse"]
    fresh_bytes = (Path(__file__).resolve().parents[2] / "docs/evidence/advisor-q08-retention-202610010523/fresh-reuse-audit.json").read_bytes()
    if hashlib.sha256(fresh_bytes).hexdigest() != fresh_ref["sha256"] or json.loads(fresh_bytes)["cells"] != fresh:
        raise ValueError("historical fresh source bytes changed")
    for cell in fresh:
        if any(type(r["score"]) not in (int, float) or not math.isfinite(r["score"]) or not 0 <= r["score"] <= 1 for r in cell["rows"]):
            raise ValueError("invalid historical Task Score")
    for cell in fresh:
        if any(not isinstance(r.get("task_identity"), str) or not r["task_identity"] or type(r.get("paired_generation_seed")) is not int for r in cell["rows"]):
            raise ValueError("invalid historical task identity or paired seed")
        if len({r["task_identity"] for r in cell["rows"]}) != 64:
            raise ValueError("historical fresh panel must have64 unique tasks per cell")
    if len({tuple((r["task_identity"], r["paired_generation_seed"]) for r in c["rows"]) for c in fresh}) != 1:
        raise ValueError("historical fresh task/seed panel changed")
    means = []
    for (arm, cp), rows in seen.items():
        for cohort in cohorts:
            values = [r["score"] for r in rows if r["cohort"] == cohort]
            means.append({"lineage": arm, "checkpoint": cp, "panel": cohort, "tasks": len(values), "mean_score": sum(values) / len(values)})
    for cell in fresh:
        values = [r["score"] for r in cell["rows"]]
        means.append({"lineage": cell["lineage"], "checkpoint": cell["completed_rounds"], "panel": "FRESH_HELDOUT", "tasks": 64, "mean_score": sum(values) / 64})
    analyses = bound_analyses(data, seen) if complete else {}
    comparisons, diagnostics = primary_scopes(analyses) if complete else (data["planned_comparisons"], [])
    if len(comparisons) != 9 or {(r["lineage"], r["cohort"]) for r in comparisons} != {(a, c) for a in lineages for c in cohorts}:
        raise ValueError("exact nine planned lineage/cohort comparisons required")
    allowed = {"UNAVAILABLE_TRAINING_PROVENANCE_PARTIAL", "UNAVAILABLE_FIXED_COHORT_COVERAGE"} if complete else {"UNAVAILABLE_TRAINING_PROVENANCE_PARTIAL"}
    if any(r.get("planned_tasks") != 32 or r[metric].get("status") not in allowed for r in comparisons for metric in ("acquisition", "retention")):
        raise ValueError("this partial report requires32 planned tasks and typed unavailable primary scopes")
    if any(r[metric][field] is not None for r in comparisons for metric in ("acquisition", "retention") for field in ("mean", "ci95")):
        raise ValueError("this descriptive exporter cannot accept primary learning estimates")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report-input.json").write_bytes(raw)
    score_rows = [{"lineage": arm, "checkpoint": cp, **row} for (arm, cp), rows in seen.items() for row in rows]
    if score_rows:
        with (output_dir / "seen-task-scores.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(score_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(score_rows)
    with (output_dir / "descriptive-checkpoint-means.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(means[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(means)
    (output_dir / "planned-primary-comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    if complete:
        (output_dir / "incomplete-cohort-diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
        for arm, result in analyses.items():
            (output_dir / f"{arm}-analysis.json").write_text(json.dumps(result, indent=2) + "\n")
        exposure = []
        for arm, result in analyses.items():
            observed = {r["task_id"]: r for r in result.get("task_rows", [])}
            for task in frozen_tasks():
                row = observed.get(task["task_id"])
                exposure.append({"lineage": arm, **task,
                                 "status": row["status"] if row else "TRAINING_INCLUSION_UNBOUND",
                                 "accepted_task_update_count": row["accepted_task_update_count"] if row else None,
                                 "first_effective_exposure": row.get("first_effective_exposure") if row else None,
                                 "last_effective_exposure": row.get("last_effective_exposure") if row else None,
                                 **{f"accepted_before_R{cp}": row.get("exposure_counts_by_checkpoint", {}).get(str(cp), 0) if row else None for cp in checkpoints}})
        with (output_dir / "task-training-exposure.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(exposure[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(exposure)
        fig, axes = plt.subplots(1, 3, figsize=(10, 9), sharey=True)
        max_count = max(r["accepted_task_update_count"] or 0 for r in exposure) or 1
        for ax, arm in zip(axes, lineages):
            matrix = np.full((96, 4), np.nan)
            for i, row in enumerate(r for r in exposure if r["lineage"] == arm):
                for j, cp in enumerate(checkpoints):
                    value = row[f"accepted_before_R{cp}"]
                    if value is not None:
                        matrix[i, j] = value
            image = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=max_count, aspect="auto", cmap="viridis")
            ax.set(title=arm, xticks=range(4), xticklabels=checkpoints, xlabel="Completed rounds")
        axes[0].set_ylabel("Frozen task order; blank = training inclusion unbound")
        fig.suptitle("Verified accepted task updates; rollout counts are not training exposure", fontsize=10)
        fig.tight_layout(rect=(0, 0, .94, 1))
        fig.colorbar(image, cax=fig.add_axes((.95, .15, .015, .7)), label="Accepted task updates")
        fig.savefig(output_dir / "task-training-exposure-heatmap.png", dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, arm in zip(axes, lineages):
        for panel in (*cohorts, "FRESH_HELDOUT"):
            values = {r["checkpoint"]: r["mean_score"] for r in means if r["lineage"] == arm and r["panel"] == panel}
            ax.plot(checkpoints, [values.get(cp, np.nan) for cp in checkpoints], marker="o", label=panel)
        ax.set(title=arm, xlabel="Completed historical rounds", xticks=checkpoints, ylim=(0, 1))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Mean Task Score (descriptive)")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Evaluation complete; learning claims PARTIAL" if complete else "INTERIM: complete seen cells only; exact historical fresh reuse; learning claims PARTIAL", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "checkpoint-curves.png", dpi=180)
    fig.savefig(output_dir / "checkpoint-curves.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(10, 9), sharey=True)
    tasks = list(expected)
    for ax, arm in zip(axes, lineages):
        matrix = np.full((96, 4), np.nan)
        for j, cp in enumerate(checkpoints):
            for row in seen.get((arm, cp), []):
                matrix[tasks.index(row["task_id"]), j] = row["score"]
        image = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, aspect="auto", cmap="viridis")
        ax.set(title=arm, xticks=range(4), xticklabels=checkpoints, xlabel="Completed rounds")
    axes[0].set_ylabel("Original frozen task order; blank = missing/uncompleted")
    fig.suptitle("Per-task observed scores; learning claims PARTIAL" if complete else "INTERIM: per-task observed scores; this is not a training exposure or forgetting heatmap", fontsize=10)
    fig.tight_layout(rect=(0, 0, .94, 1))
    color_axis = fig.add_axes((.95, .15, .015, .7))
    fig.colorbar(image, cax=color_axis, label="Task Score")
    fig.savefig(output_dir / "seen-task-score-heatmap.png", dpi=180)
    plt.close(fig)
    limitations = (["Ordinary source audit is COMPLETE; fixed primary cohort/acquisition coverage is incomplete.",
                    "Bounded/GDR historical training inclusion remains PARTIAL.",
                    "Incomplete-cohort intervals are diagnostic; no primary or catastrophic-forgetting closure."] if complete else
                   ["Historical training inclusion remains PARTIAL; primary acquisition/retention intervals are unavailable.",
                    "Whole-run costs are not closed while the canonical controller runs."])
    result = {"status": COMPLETE_PARTIAL if complete else "INTERIM_RUNNING_NOT_SCIENTIFIC_SEAL", "input_sha256": hashlib.sha256(raw).hexdigest(),
              "report_producer_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__,
              "complete_seen_cells": len(seen), "seen_rows": 96 * len(seen), "fresh_exact_reused_rows": 768,
              "final_panel_access": 0, "new_model_calls": 0,
              "limitations": limitations + ["One historical training lineage per mechanism; no causal comparison or training-seed replication."],
              "sources": data["seen"]["raw_sources"],
              "artifacts": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output_dir.iterdir())}}
    if complete:
        result.update(execution_audit=data["execution_audit"], closed_cost_audit=data["closed_cost_audit"],
                      native_launch_cost=cost, evaluation_model_calls=execution["completed_model_calls"],
                      report_model_calls=0, new_model_calls_scope="CPU report export only",
                      candidate_training_audits=data["candidate_training_audits"],
                      cost_accounting={k: v for k, v in closed_costs.items() if k != "deduplicated_launch_receipts"})
    (output_dir / "report-index.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.input, args.output_dir)))
