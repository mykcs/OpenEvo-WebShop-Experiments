#!/usr/bin/env python3
"""CPU-only Q08 analysis of normalized accepted-update/exposure and score facts.

The input adapter remains a separate, provenance-bound task. No rollout is
treated as learning; missing planned checkpoints are never replaced silently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

PANEL_SHA256 = "0780dbe68dcc1cafb013a8765f2ecb4b91bfd2a0d92ac3ac9fe4fde7453fccc3"
PANEL_PATH = Path(__file__).resolve().parents[2] / "docs/evidence/advisor-q08-retention-202610010523/seen-panel-manifest.json"


def frozen_tasks() -> list[dict]:
    data = PANEL_PATH.read_bytes()
    if hashlib.sha256(data).hexdigest() != PANEL_SHA256:
        raise ValueError("frozen seen-panel manifest bytes changed")
    tasks = [{"task_id": t["task_identity"], "cohort": t["cohort"]} for t in json.loads(data)["tasks"]]
    if len(tasks) != 96 or any(sum(t["cohort"] == c for t in tasks) != 32 for c in ("EARLY_SEEN", "MID_SEEN", "LATE_SEEN")):
        raise ValueError("frozen manifest must contain the original three32-task cohorts")
    return tasks


def number(value) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("finite numeric value required")
    return float(value)


def paired_interval(values: list[float], resamples: int, seed: int) -> dict:
    if not values:
        return {"status": "UNAVAILABLE", "tasks": 0, "mean": None, "ci95": None}
    if resamples < 1000:
        raise ValueError("at least 1000 task bootstrap resamples required")
    rng = random.Random(seed)
    n = len(values)
    boot = sorted(statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(resamples))
    tail = .05 / (2 * 9)  # separate nine-comparison acquisition/retention families.
    return {"status": "FIXED_LINEAGE_TASK_INTERVAL", "tasks": n,
            "mean": statistics.fmean(values), "ci95": [boot[int(tail * (resamples - 1))], boot[int((1 - tail) * (resamples - 1))]],
            "interval_method": "task-bootstrap Bonferroni; familywise coverage .95; family size 9",
            "resamples": resamples, "seed": seed, "training_replicates": 1}


def analyse(payload: dict, *, resamples: int = 10000, seed: int = 20261001) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("normalized payload must be an object")
    if payload.get("panel_role") != "development" or not payload.get("lineage_id"):
        raise ValueError("single explicit development lineage required")
    planned = payload.get("planned_checkpoints")
    if planned not in ([40, 80, 120, 160], [0, 40, 80, 120, 160]) or any(type(p) is not int for p in planned):
        raise ValueError("frozen checkpoints40/80/120/160 required; only completed0 may be added")
    late = payload.get("late_checkpoint")
    if late != planned[-1]:
        raise ValueError("late checkpoint must be the frozen final planned checkpoint")
    statuses = payload.get("checkpoint_status", {})
    if set(statuses) != {str(p) for p in planned} or any(v not in ("AVAILABLE", "MISSING_PAYLOAD") for v in statuses.values()):
        raise ValueError("explicit status for every planned checkpoint required")
    tasks = payload.get("tasks", [])
    if not isinstance(tasks, list) or len(tasks) != 96 or any(not isinstance(t, dict) or not isinstance(t.get("task_id"), str) for t in tasks) or len({t["task_id"] for t in tasks}) != 96:
        raise ValueError("original96 unique outcome-blind task panel required")
    if any(t.get("cohort") not in ("EARLY_SEEN", "MID_SEEN", "LATE_SEEN") for t in tasks):
        raise ValueError("only the three frozen seen cohorts are eligible")
    expected = {t["task_id"]: t["cohort"] for t in frozen_tasks()}
    if {t["task_id"]: t["cohort"] for t in tasks} != expected:
        raise ValueError("tasks/cohorts do not match the exact frozen manifest")
    task_ids = {t["task_id"] for t in tasks}
    scores = {}
    eval_seeds = {}
    for row in payload.get("scores", []):
        tid, cp = row["task_id"], row["completed_rounds"]
        if tid not in task_ids or cp not in planned or statuses[str(cp)] != "AVAILABLE":
            raise ValueError("score uses unknown task/checkpoint or missing payload")
        key = (tid, cp)
        if key in scores or type(row.get("engineering_valid")) is not bool:
            raise ValueError("duplicate score or missing engineering-valid fact")
        if type(row.get("evaluation_seed")) is not int:
            raise ValueError("paired evaluation seed required")
        if tid in eval_seeds and eval_seeds[tid] != row["evaluation_seed"]:
            raise ValueError("task evaluation seed changed across checkpoints")
        eval_seeds[tid] = row["evaluation_seed"]
        raw_score = row.get("score")
        if row["engineering_valid"]:
            score = number(raw_score)
            if not 0 <= score <= 1:
                raise ValueError("Task Score outside [0,1]")
            scores[key] = score
        else:
            if raw_score is not None:
                score = number(raw_score)
                if not 0 <= score <= 1:
                    raise ValueError("Task Score outside [0,1]")
            scores[key] = None
    observations = {}
    for cp in planned:
        cohorts = {}
        for cohort in sorted(set(expected.values())):
            values = [scores[(tid, cp)] for tid, group in expected.items()
                      if group == cohort and scores.get((tid, cp)) is not None]
            observed = sum((tid, cp) in scores for tid, group in expected.items() if group == cohort)
            cohorts[cohort] = {"planned_tasks": 32, "observed_tasks": observed,
                               "valid_tasks": len(values), "coverage_complete": len(values) == 32,
                               "mean_score": statistics.fmean(values) if values else None}
        observations[str(cp)] = {"checkpoint_status": statuses[str(cp)], "cohorts": cohorts}
    evaluation_observations = {
        "status": "DESCRIPTIVE_EVALUATION_ONLY", "checkpoints": observations,
        "interpretation": "Observed checkpoint scores do not establish training inclusion, acquisition, retention, or interference causality."}
    if payload.get("audit_status") != "COMPLETE" or payload.get("audit_gaps", []) != []:
        return {"schema": "openevo.q08.normalized-retention-analysis.v1", "status": "PARTIAL_AUDIT_INPUT",
                "lineage_id": payload["lineage_id"], "audit_gaps": payload.get("audit_gaps", []),
                "audit_reason": "audit_status is missing/non-COMPLETE or audit_gaps is nonempty",
                "evaluation_observations": evaluation_observations,
                "source_provenance": payload.get("source_provenance"),
                "final_panel_access": 0, "model_calls": 0}
    events = {}
    event_ids = set()
    for event in payload.get("events", []):
        tid = event["task_id"]
        if tid not in task_ids or type(event.get("round")) is not int or not 0 <= event["round"] < late:
            raise ValueError("unknown task or malformed event round")
        eid = (tid, event.get("update_id"), event["round"])
        if not event.get("update_id") or eid in event_ids:
            raise ValueError("explicit unique task/update event required")
        event_ids.add(eid)
        for flag in ("rollout_consumed", "training_included", "accepted_update"):
            if type(event.get(flag)) is not bool:
                raise ValueError("rollout/training/acceptance facts must be explicit booleans")
        if event["accepted_update"]:
            if not event["training_included"] or not event["rollout_consumed"]:
                raise ValueError("accepted task update requires actual training inclusion")
            before, after = event.get("state_before"), event.get("state_after")
            sha = event.get("update_receipt_sha256", "")
            digest_before, digest_after = event.get("adapter_digest_before"), event.get("adapter_digest_after")
            valid_digests = all(isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)
                                for v in (digest_before, digest_after))
            if (not before or not after or before == after or not valid_digests or digest_before == digest_after
                    or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)):
                raise ValueError("accepted update requires changed adapter content, state, and exact receipt")
        events.setdefault(tid, []).append(event)
    output = []
    for task in tasks:
        tid = task["task_id"]
        consumed = sorted(e["round"] for e in events.get(tid, []) if e["rollout_consumed"])
        effective = sorted(e["round"] for e in events.get(tid, []) if e["accepted_update"] and e["round"] < late)
        row = {"task_id": tid, "cohort": task["cohort"], "first_rollout_exposure": min(consumed) if consumed else None,
               "first_effective_exposure": effective[0] if effective else None,
               "last_effective_exposure": effective[-1] if effective else None,
               "accepted_task_update_count": len(effective), "acquisition_delta": None,
               "retention_delta": None, "status": "NO_EFFECTIVE_EXPOSURE"}
        if effective:
            first = effective[0]
            post = next((p for p in planned if p > first), None)
            pre = max((p for p in planned if p <= first), default=None)
            row.update(first_post_checkpoint=post, pre_exposure_checkpoint=pre,
                       exposure_counts_by_checkpoint={str(p): sum(r < p for r in effective) for p in planned})
            if pre is not None and post is not None and scores.get((tid, pre)) is not None and scores.get((tid, post)) is not None:
                row["acquisition_delta"] = scores[(tid, post)] - scores[(tid, pre)]
            if post is None or statuses[str(post)] == "MISSING_PAYLOAD":
                row["status"] = "PARTIAL_MISSING_FIRST_POST_CHECKPOINT"
            elif post == late:
                row["status"] = "PARTIAL_NO_LATER_CHECKPOINT"
            elif scores.get((tid, post)) is None or scores.get((tid, late)) is None:
                row["status"] = "PARTIAL_MISSING_OR_INVALID_EVALUATION"
            else:
                row["retention_delta"] = scores[(tid, late)] - scores[(tid, post)]
                row["status"] = "RETENTION_MEASURED_ACQUISITION_UNVERIFIED"
                if row["acquisition_delta"] is not None:
                    row["status"] = "ACQUISITION_AND_RETENTION_MEASURED"
        output.append(row)
    cohorts = {}
    for idx, cohort in enumerate(sorted({t["cohort"] for t in tasks})):
        group = [r for r in output if r["cohort"] == cohort]
        acquisition = paired_interval([r["acquisition_delta"] for r in group if r["acquisition_delta"] is not None], resamples, seed + 2 * idx)
        retention = paired_interval([r["retention_delta"] for r in group if r["retention_delta"] is not None], resamples, seed + 2 * idx + 1)
        acq = acquisition["ci95"] is not None and acquisition["ci95"][0] > .05
        loss = retention["ci95"] is not None and retention["ci95"][1] < -.05
        cohorts[cohort] = {"planned_tasks": len(group), "acquisition": acquisition, "retention": retention,
                           "coverage_complete": retention["tasks"] == len(group),
                           "routing_hint": "INTERFERENCE_EXPERIMENT_CANDIDATE" if acq and loss and acquisition["tasks"] == retention["tasks"] == len(group) else "PARTIAL_OR_NO_CONFIRMED_ACQUISITION_AND_LOSS"}
    return {"schema": "openevo.q08.normalized-retention-analysis.v1", "status": "CPU_ANALYSIS_NOT_SCIENTIFIC_SEAL",
            "lineage_id": payload["lineage_id"], "task_rows": output, "cohorts": cohorts,
            "evaluation_observations": evaluation_observations,
            "source_provenance": payload.get("source_provenance"),
            "final_panel_access": 0, "model_calls": 0,
            "limitations": ["Task bootstrap conditions on one historical training lineage; it is not a training-seed replicate.",
                            "Acquisition and retention each use a frozen nine-comparison historical lineage/cohort family; no training-population inference.",
                            "Acquisition selection is never used to remove tasks from primary cohort retention.",
                            "Retention loss alone does not identify interference causality or catastrophic forgetting."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    print(json.dumps(analyse(json.loads(args.input.read_text(encoding="utf-8")), resamples=args.resamples, seed=args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
