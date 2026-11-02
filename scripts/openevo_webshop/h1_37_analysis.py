#!/usr/bin/env python3
"""H1.37 preregistered statistical analysis for the frozen-adapter fresh-task eval.

Consumes the reconcile output (paired_rows with per-attempt self_evolution_minus_base)
and computes the preregistered primary estimand and inference:

  primary estimand  Delta = (1/N) * sum_i (R_self,i - R_base,i)   (mean paired reward
                    difference per episode over all valid paired attempts)
  inference unit    the task (cluster): 64 task-level paired effects, NOT 256
                    independent episodes. Uncertainty is computed with a task-cluster
                    bootstrap and a task-level sign-flip (paired randomization) test.

The three preregistered interpretation zones are evaluated here so the verdict is
computed, not narrated after the fact:
  - "transfer-positive":   lower bound of the 95% task-cluster CI > 0
  - "clean-negative":      CI half-width <= NARROW_HALF_WIDTH and the CI contains 0
  - "underpowered":        CI half-width > NARROW_HALF_WIDTH (cannot resolve the effect)

All resampling uses a fixed seed for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

BOOTSTRAP_ROUNDS = 10000
PERMUTATION_ROUNDS = 10000
RESAMPLE_SEED = 20260818
# Preregistered resolution threshold: a 95% CI with half-width <= 0.05 reward points
# is "narrow" enough to call a near-zero delta a clean negative rather than a wash.
NARROW_HALF_WIDTH = 0.05


def _task_clusters(paired_rows: list[dict[str, Any]]) -> dict[int, list[float]]:
    clusters: dict[int, list[float]] = {}
    for row in paired_rows:
        clusters.setdefault(int(row["goal_idx"]), []).append(float(row["self_evolution_minus_base"]))
    return clusters


def _mean_paired_delta(paired_rows: list[dict[str, Any]]) -> float:
    if not paired_rows:
        return 0.0
    return sum(float(r["self_evolution_minus_base"]) for r in paired_rows) / len(paired_rows)


def _bootstrap_ci(clusters: dict[int, list[float]]) -> tuple[float, float, float]:
    """Task-cluster bootstrap 95% percentile CI for the mean paired per-episode delta.

    Resample tasks (clusters) with replacement; within a resampled task keep all of
    its per-episode deltas. The statistic is the mean over all resampled episodes,
    matching the primary estimand's per-episode weighting while respecting clustering.
    """
    task_ids = sorted(clusters)
    if not task_ids:
        return (0.0, 0.0, 0.0)
    rng = random.Random(RESAMPLE_SEED)
    stats: list[float] = []
    for _ in range(BOOTSTRAP_ROUNDS):
        sampled = [rng.choice(task_ids) for _ in task_ids]
        deltas = [d for tid in sampled for d in clusters[tid]]
        stats.append(sum(deltas) / len(deltas))
    stats.sort()
    lo = stats[int(0.025 * BOOTSTRAP_ROUNDS)]
    hi = stats[min(int(0.975 * BOOTSTRAP_ROUNDS), BOOTSTRAP_ROUNDS - 1)]
    half_width = (hi - lo) / 2.0
    return (lo, hi, half_width)


def _permutation_pvalue(clusters: dict[int, list[float]]) -> tuple[float, float]:
    """Task-level sign-flip (paired randomization) test, two-sided.

    Under the null of no self-evolution effect, each task's mean paired delta is
    symmetric about 0, so flipping its sign is exchangeable. The statistic is the
    mean of task-mean deltas (the task-level estimand). Returns (observed, p_value).
    """
    task_means = [sum(v) / len(v) for _, v in sorted(clusters.items())]
    if not task_means:
        return (0.0, 1.0)
    observed = sum(task_means) / len(task_means)
    rng = random.Random(RESAMPLE_SEED + 1)
    extreme = 0
    for _ in range(PERMUTATION_ROUNDS):
        perm = sum(tm if rng.random() < 0.5 else -tm for tm in task_means) / len(task_means)
        if abs(perm) >= abs(observed):
            extreme += 1
    return (observed, (extreme + 1) / (PERMUTATION_ROUNDS + 1))


def analyze(reconcile: dict[str, Any]) -> dict[str, Any]:
    paired_rows = reconcile.get("paired_rows") or []
    clusters = _task_clusters(paired_rows)
    n_attempts = len(paired_rows)
    n_tasks = len(clusters)

    base_mean = sum(float(r["base_score"]) for r in paired_rows) / n_attempts if n_attempts else 0.0
    self_mean = sum(float(r["self-evolution_score"]) for r in paired_rows) / n_attempts if n_attempts else 0.0
    mean_delta = _mean_paired_delta(paired_rows)

    task_mean_deltas = {tid: sum(v) / len(v) for tid, v in clusters.items()}
    pos = sum(1 for v in task_mean_deltas.values() if v > 0)
    neg = sum(1 for v in task_mean_deltas.values() if v < 0)
    tie = sum(1 for v in task_mean_deltas.values() if v == 0)

    base_success = sum(1 for r in paired_rows if float(r["base_score"]) >= 1.0)
    self_success = sum(1 for r in paired_rows if float(r["self-evolution_score"]) >= 1.0)

    ci_lo, ci_hi, half_width = _bootstrap_ci(clusters)
    perm_observed, perm_p = _permutation_pvalue(clusters)

    if ci_lo > 0:
        zone = "transfer-positive"
    elif half_width <= NARROW_HALF_WIDTH:
        zone = "clean-negative"
    else:
        zone = "underpowered"

    return {
        "schema": "openevo.webshop-h1.37-analysis.v1",
        "experiment": reconcile.get("experiment"),
        "primary_estimand": "Delta = mean paired per-episode reward difference (self-evolution - base)",
        "inference_unit": "task (cluster); task-cluster bootstrap + task-level sign-flip permutation",
        "resample_seed": RESAMPLE_SEED,
        "bootstrap_rounds": BOOTSTRAP_ROUNDS,
        "permutation_rounds": PERMUTATION_ROUNDS,
        "narrow_half_width_threshold": NARROW_HALF_WIDTH,
        "paired_attempt_count": n_attempts,
        "paired_task_count": n_tasks,
        "base_mean_reward": base_mean,
        "self_evolution_mean_reward": self_mean,
        "mean_paired_delta_per_episode": mean_delta,
        "task_level_mean_paired_delta": perm_observed,
        "task_positive_tie_negative": {"positive": pos, "tie": tie, "negative": neg},
        "success_counts": {"base": base_success, "self_evolution": self_success},
        "task_cluster_bootstrap_ci95": {"lower": ci_lo, "upper": ci_hi, "half_width": half_width},
        "paired_randomization_p_value": perm_p,
        "interpretation_zone": zone,
        "zone_definitions": {
            "transfer-positive": "task-cluster CI95 lower bound > 0 -> frozen self-generated SD-LoRA update transfers to unseen WebShop tasks",
            "clean-negative": "CI95 half-width <= threshold and CI contains 0 -> 7B generates and internalizes endogenous data but the update does not improve task-level generalization; motivates the H1.38 task-diversity campaign",
            "underpowered": "CI95 half-width > threshold -> still cannot resolve the effect; NOT a negative result",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconcile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reconcile = json.loads(args.reconcile.expanduser().resolve().read_text(encoding="utf-8"))
    result = analyze(reconcile)
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
