#!/usr/bin/env python3
"""CPU-only nested Q06 prediction and task-cluster inference.

All feature encoding, scaling, hyperparameter selection, and model fitting are
recomputed inside each training partition.  Only the supplied outer split
decides which labels are held out.  This is a historical diagnostic, never a
controller or causal claim.
"""
from __future__ import annotations

import hashlib
import math
import random
import statistics
from collections import defaultdict
from typing import Any

from scripts.analysis.advisor_q06_raw_adapter import (
    BASELINE_NUMERIC_FEATURES,
    SCHEMA as ROW_SCHEMA,
    UNCERTAINTY_FEATURES,
)

BOOTSTRAPS = 50_000
PRACTICAL_THRESHOLD = 0.02
SECONDARY_BRIER_THRESHOLD = 0.01
RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_normalized_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("normalized development rows are required")
    seen: set[tuple[str, str, str, int]] = set()
    allowed = BASELINE_NUMERIC_FEATURES | UNCERTAINTY_FEATURES
    for row in rows:
        if row.get("schema") != ROW_SCHEMA:
            raise ValueError("normalized row schema mismatch")
        if row.get("panel_role") != "development" or row.get("feature_phase") != "pre_action":
            raise ValueError("development/pre_action rows are required")
        if row.get("target_phase") != "post_action_or_later":
            raise ValueError("target must be observed after the decision")
        for key in ("task_id", "lineage_id", "checkpoint_id", "state_type"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"missing row identity/category: {key}")
        if not _valid_sha(row.get("feature_provenance_sha256")) or not _valid_sha(row.get("source_file_sha256")):
            raise ValueError("source-byte provenance hashes are required")
        record_number = row.get("source_record_number")
        if type(record_number) is not int or record_number < 1:
            raise ValueError("source_record_number must be positive")
        if not _valid_sha(row.get("pre_action_context_sha256")):
            raise ValueError("pre-action context hash is required")
        event_indices = [row.get(k) for k in ("feature_event_index", "decision_event_index", "action_event_index")]
        if any(type(v) is not int or v < 0 for v in event_indices) or not event_indices[0] < event_indices[1] <= event_indices[2]:
            raise ValueError("source-bound feature event must precede the decision/action events")
        steps = [row.get(k) for k in ("decision_step", "feature_observed_step", "target_observed_step", "completed_rounds")]
        if any(type(v) is not int or v < 0 for v in steps):
            raise ValueError("nonnegative integer decision/feature/target steps and round are required")
        decision, observed, target, _round = steps
        if observed > decision or target <= decision:
            raise ValueError("feature must be pre-action and target must follow decision")
        ident = (row["lineage_id"], row["checkpoint_id"], row["task_id"], decision)
        if ident in seen:
            raise ValueError("duplicate normalized decision identity")
        seen.add(ident)
        features = row.get("features")
        if not isinstance(features, dict) or set(features) != allowed:
            missing, extra = allowed - set(features or {}), set(features or {}) - allowed
            raise ValueError(f"feature registry mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
        for key, value in features.items():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"feature {key} must be a finite numeric scalar")
        for key in ("round", "choice_count", "step_index", "prompt_tokens", "history_tokens"):
            if type(features[key]) is not int or features[key] < 0:
                raise ValueError(f"feature {key} must be a nonnegative integer")
        if features["choice_count"] < 2 or features["history_tokens"] > features["prompt_tokens"]:
            raise ValueError("choice count or prompt/history length is outside its domain")
        if features["reasoning_entropy_mean"] < 0 or features["action_entropy_mean"] < 0:
            raise ValueError("entropy features must be nonnegative")
        if features["step_index"] != decision or features["round"] != row["completed_rounds"]:
            raise ValueError("round/step features do not bind the observed decision")
        if not 0 <= features["click_entropy_normalized"] <= 1 + 1e-8:
            raise ValueError("choice count or normalized click entropy is outside its domain")
        if type(row.get("target")) not in (int, float) or not math.isfinite(row["target"]) or not 0 <= row["target"] <= 1:
            raise ValueError("Task Score target must be finite and in [0,1]")
        secondary = row.get("secondary_targets")
        if not isinstance(secondary, dict) or set(secondary) != {"exact_success"} or type(secondary["exact_success"]) is not int or secondary["exact_success"] not in (0, 1):
            raise ValueError("explicit exact_success secondary target (0/1) is required")


def outer_partition(rows: list[dict[str, Any]], spec: dict[str, Any]) -> tuple[list[int], list[int]]:
    mode = spec.get("mode")
    if mode in ("task", "lineage"):
        field = "task_id" if mode == "task" else "lineage_id"
        ids = spec.get("heldout_ids")
        if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
            raise ValueError(f"{mode} outer split requires unique heldout_ids")
        if not set(ids) <= {row[field] for row in rows}:
            raise ValueError(f"{mode} outer split includes unknown ids")
        held = set(ids)
        train = [i for i, row in enumerate(rows) if row[field] not in held]
        test = [i for i, row in enumerate(rows) if row[field] in held]
        if {rows[i][field] for i in train} & {rows[i][field] for i in test}:
            raise ValueError("outer group leakage")
    elif mode == "time":
        cutoff = spec.get("train_max_round")
        if type(cutoff) is not int or cutoff < 0:
            raise ValueError("time outer split requires a frozen train_max_round")
        test = [i for i, row in enumerate(rows) if row["completed_rounds"] > cutoff]
        # A task-level final score is repeated on every decision row.  Never let
        # an earlier row for a held-out task's later state enter training.
        heldout_tasks = {rows[i]["task_id"] for i in test}
        train = [i for i, row in enumerate(rows)
                 if row["completed_rounds"] <= cutoff and row["task_id"] not in heldout_tasks]
    else:
        raise ValueError("outer split mode must be task, time or lineage")
    if not train or not test:
        raise ValueError("outer split must have nonempty train and test rows")
    return train, test


def _feature_keys(model: str) -> tuple[str, ...]:
    base = tuple(sorted(BASELINE_NUMERIC_FEATURES))
    if model == "baseline":
        return base
    if model == "uncertainty":
        return base + tuple(sorted(UNCERTAINTY_FEATURES))
    raise ValueError("model must be baseline or uncertainty")


def _task_row_weights(rows: list[dict[str, Any]]) -> list[float]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["task_id"]] += 1
    return [1.0 / counts[row["task_id"]] for row in rows]


def _encode(train: list[dict[str, Any]], rows: list[dict[str, Any]], model: str) -> tuple[list[list[float]], list[list[float]]]:
    keys = _feature_keys(model)
    categories = sorted({r["state_type"] for r in train})
    weights = _task_row_weights(train)
    total_weight = sum(weights)
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for key in keys:
        vals = [float(r["features"][key]) for r in train]
        means[key] = sum(w * v for w, v in zip(weights, vals)) / total_weight
        sd = math.sqrt(sum(w * (v - means[key]) ** 2 for w, v in zip(weights, vals)) / total_weight)
        scales[key] = sd if sd > 1e-12 else 1.0

    def one(row: dict[str, Any]) -> list[float]:
        numeric = [(float(row["features"][key]) - means[key]) / scales[key] for key in keys]
        categorical = [1.0 if row["state_type"] == cat else 0.0 for cat in categories]
        return [1.0, *numeric, *categorical]

    return [one(r) for r in train], [one(r) for r in rows]


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Small dense Gauss-Jordan solve; ridge makes the system nonsingular."""
    n = len(b)
    aug = [list(a[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-14:
            raise ValueError("singular ridge system")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col]
        aug[col] = [x / div for x in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor:
                aug[row] = [x - factor * y for x, y in zip(aug[row], aug[col])]
    return [aug[i][-1] for i in range(n)]


def _fit_predict(train_rows: list[dict[str, Any]], predict_rows: list[dict[str, Any]], model: str, alpha: float,
                 *, target_key: str = "target") -> list[float]:
    x_train, x_predict = _encode(train_rows, predict_rows, model)
    row_weights = _task_row_weights(train_rows)
    p = len(x_train[0])
    gram = [[sum(weight * row[i] * row[j] for weight, row in zip(row_weights, x_train)) for j in range(p)] for i in range(p)]
    # Do not penalize intercept.
    for i in range(1, p):
        gram[i][i] += alpha
    y = [float(r[target_key]) for r in train_rows] if target_key == "target" else [float(r["secondary_targets"][target_key]) for r in train_rows]
    rhs = [sum(weight * row[i] * target for weight, row, target in zip(row_weights, x_train, y)) for i in range(p)]
    weights = _solve(gram, rhs)
    return [min(1.0, max(0.0, sum(w * x for w, x in zip(weights, row)))) for row in x_predict]


def _folds(rows: list[dict[str, Any]], k: int) -> list[tuple[list[int], list[int]]]:
    group_rows: dict[str, list[int]] = defaultdict(list)
    for local_idx, row in enumerate(rows):
        group_rows[row["task_id"]].append(local_idx)
    task_ids = sorted(group_rows)
    if len(task_ids) < 2:
        raise ValueError("nested inner fitting needs at least two training task groups")
    k = min(k, len(task_ids))
    buckets: list[list[str]] = [[] for _ in range(k)]
    # Stable assignment without random state or target-dependent balancing.
    for idx, task_id in enumerate(task_ids):
        buckets[idx % k].append(task_id)
    out = []
    for bucket in buckets:
        val_ids = set(bucket)
        val = [i for tid in bucket for i in group_rows[tid]]
        trn = [i for tid in task_ids if tid not in val_ids for i in group_rows[tid]]
        if trn and val:
            out.append((trn, val))
    return out


def _tune_alpha(rows: list[dict[str, Any]], model: str, *, inner_folds: int,
                target_key: str = "target", loss: str = "mae") -> tuple[float, dict[str, float]]:
    folds = _folds(rows, inner_folds)
    losses: dict[float, list[float]] = {alpha: [] for alpha in RIDGE_GRID}
    for train_idx, valid_idx in folds:
        train = [rows[i] for i in train_idx]
        valid = [rows[i] for i in valid_idx]
        for alpha in RIDGE_GRID:
            pred = _fit_predict(train, valid, model, alpha, target_key=target_key)
            losses[alpha].append(_task_balanced_loss(valid, pred, target_key=target_key, loss=loss))
    means = {str(alpha): statistics.fmean(vals) for alpha, vals in losses.items()}
    # Tie-break toward the stronger regularizer.
    selected = min(RIDGE_GRID, key=lambda alpha: (statistics.fmean(losses[alpha]), -alpha))
    return selected, means


def _cluster_effects(rows: list[dict[str, Any]], baseline: list[float], augmented: list[float],
                     *, target_key: str = "target", loss: str = "mae") -> dict[str, float]:
    by_task: dict[str, list[float]] = defaultdict(list)
    for row, base, unc in zip(rows, baseline, augmented):
        target = float(row["target"]) if target_key == "target" else float(row["secondary_targets"][target_key])
        base_loss = abs(target - base) if loss == "mae" else (target - base) ** 2
        unc_loss = abs(target - unc) if loss == "mae" else (target - unc) ** 2
        by_task[row["task_id"]].append(base_loss - unc_loss)
    return {task: statistics.fmean(values) for task, values in by_task.items()}


def _task_balanced_loss(rows: list[dict[str, Any]], predictions: list[float], *,
                        target_key: str = "target", loss: str = "mae") -> float:
    by_task: dict[str, list[float]] = defaultdict(list)
    for row, pred in zip(rows, predictions):
        target = float(row["target"]) if target_key == "target" else float(row["secondary_targets"][target_key])
        if loss == "mae":
            value = abs(target - pred)
        elif loss == "brier":
            value = (target - pred) ** 2
        else:
            raise ValueError("loss must be mae or brier")
        by_task[row["task_id"]].append(value)
    return statistics.fmean(statistics.fmean(values) for values in by_task.values())


def _quantile(sorted_values: list[float], q: float) -> float:
    position = (len(sorted_values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (position - lo)


def _bootstrap_ci(effects: list[float], seed: int, alpha: float) -> list[float]:
    rng = random.Random(seed)
    n = len(effects)
    draws = []
    for _ in range(BOOTSTRAPS):
        draws.append(sum(effects[rng.randrange(n)] for _ in range(n)) / n)
    draws.sort()
    return [_quantile(draws, alpha / 2), _quantile(draws, 1 - alpha / 2)]


def _sign_flip_p(effects: list[float], seed: int) -> float:
    if len(effects) == 1:
        return 1.0
    observed = abs(statistics.fmean(effects))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(BOOTSTRAPS):
        value = abs(sum(x if rng.getrandbits(1) else -x for x in effects) / len(effects))
        extreme += value >= observed - 1e-15
    return (extreme + 1) / (BOOTSTRAPS + 1)


def holm_adjust(p_values: dict[str, float], *, family_size: int | None = None) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda key: p_values[key])
    m = len(ordered) if family_size is None else family_size
    if m < len(ordered):
        raise ValueError("family_size cannot be smaller than the observed p-value count")
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p_values[key]))
        adjusted[key] = running
    return adjusted


def _run_protocol(rows: list[dict[str, Any]], name: str, spec: dict[str, Any], *, inner_folds: int) -> dict[str, Any]:
    if spec.get("mode") != name:
        raise ValueError(f"{name} protocol spec must declare mode={name}")
    train_idx, test_idx = outer_partition(rows, spec)
    train_rows = [rows[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]
    # The two candidates tune on identical, task-grouped inner folds drawn only
    # from outer-train rows. No outer target is passed to the tuner.
    selected: dict[str, float] = {}
    tuning: dict[str, Any] = {}
    for model in ("baseline", "uncertainty"):
        alpha, loss = _tune_alpha(train_rows, model, inner_folds=inner_folds)
        selected[model] = alpha
        tuning[model] = {"selected_alpha": alpha, "inner_mae_by_alpha": loss}
    pred_baseline = _fit_predict(train_rows, test_rows, "baseline", selected["baseline"])
    pred_uncertainty = _fit_predict(train_rows, test_rows, "uncertainty", selected["uncertainty"])
    effects_by_task = _cluster_effects(test_rows, pred_baseline, pred_uncertainty)
    effects = [effects_by_task[task] for task in sorted(effects_by_task)]
    improvement = statistics.fmean(effects)
    test_lineage_effects: dict[str, list[float]] = defaultdict(list)
    for row, base, unc in zip(test_rows, pred_baseline, pred_uncertainty):
        test_lineage_effects[row["lineage_id"]].append(abs(float(row["target"]) - base) - abs(float(row["target"]) - unc))
    lineage_effects = {lineage: statistics.fmean(vals) for lineage, vals in test_lineage_effects.items()}
    unique_train_lineages = sorted({r["lineage_id"] for r in train_rows})
    enough_independent_coverage = len(unique_train_lineages) >= 2 and len(lineage_effects) >= 2
    status = "ESTIMATED" if enough_independent_coverage and len(effects) >= 2 else "PARTIAL"
    reason = None if status == "ESTIMATED" else "independent trajectory coverage is insufficient for a predictive claim"
    # Secondary exact-success Brier family has its own inner tuning, paired
    # task bootstrap and multiplicity correction; it never substitutes for MAE.
    success_selected: dict[str, float] = {}
    success_tuning: dict[str, Any] = {}
    for model in ("baseline", "uncertainty"):
        alpha, loss = _tune_alpha(train_rows, model, inner_folds=inner_folds,
                                  target_key="exact_success", loss="brier")
        success_selected[model] = alpha
        success_tuning[model] = {"selected_alpha": alpha, "inner_brier_by_alpha": loss}
    pred_success_baseline = _fit_predict(train_rows, test_rows, "baseline", success_selected["baseline"], target_key="exact_success")
    pred_success_uncertainty = _fit_predict(train_rows, test_rows, "uncertainty", success_selected["uncertainty"], target_key="exact_success")
    success_effects = _cluster_effects(test_rows, pred_success_baseline, pred_success_uncertainty,
                                      target_key="exact_success", loss="brier")
    success_effect_vector = [success_effects[task] for task in sorted(success_effects)]
    success_improvement = statistics.fmean(success_effect_vector)
    return {
        "protocol": name,
        "outer_split": spec,
        "status": status,
        "partial_reason": reason,
        "train_rows": len(train_rows),
        "heldout_rows": len(test_rows),
        "train_task_groups": len({r["task_id"] for r in train_rows}),
        "heldout_task_clusters": len(effects),
        "train_lineages": unique_train_lineages,
        "heldout_lineage_effects": lineage_effects,
        "heldout_mae_baseline": _task_balanced_loss(test_rows, pred_baseline),
        "heldout_mae_uncertainty": _task_balanced_loss(test_rows, pred_uncertainty),
        "mae_improvement_baseline_minus_uncertainty": improvement,
        "task_cluster_effects": effects_by_task,
        "bootstrap_ci95": _bootstrap_ci(effects, 20261001 + len(name), 0.05),
        "sign_flip_p_unadjusted": _sign_flip_p(effects, 20262001 + len(name)),
        "bootstrap_resamples": BOOTSTRAPS,
        "tuning": tuning,
        "secondary_exact_success_brier": {
            "heldout_brier_baseline": _task_balanced_loss(test_rows, pred_success_baseline, target_key="exact_success", loss="brier"),
            "heldout_brier_uncertainty": _task_balanced_loss(test_rows, pred_success_uncertainty, target_key="exact_success", loss="brier"),
            "brier_improvement_baseline_minus_uncertainty": success_improvement,
            "task_cluster_effects": success_effects,
            "bootstrap_ci95": _bootstrap_ci(success_effect_vector, 20401001 + len(name), 0.05),
            "sign_flip_p_unadjusted": _sign_flip_p(success_effect_vector, 20402001 + len(name)),
            "tuning": success_tuning,
            "practical_effect_threshold": SECONDARY_BRIER_THRESHOLD,
        },
        "claim": "heldout historical prediction diagnostic; not causal and not controller validation",
    }


def run_nested_prediction(
    rows: list[dict[str, Any]], outer_splits: dict[str, dict[str, Any]], *, inner_folds: int = 5,
) -> dict[str, Any]:
    """Fit nested baseline/uncertainty predictors for frozen outer protocols.

    Expected outer split keys are ``task``, ``time`` and ``lineage``. Missing
    or unestimable protocols remain PARTIAL; no split is created from labels.
    """
    validate_normalized_rows(rows)
    if type(inner_folds) is not int or inner_folds < 2:
        raise ValueError("inner_folds must be at least 2")
    required = {"task", "time", "lineage"}
    if not isinstance(outer_splits, dict):
        raise ValueError("outer_splits must be a frozen protocol mapping")
    unknown = set(outer_splits) - required
    if unknown:
        raise ValueError(f"unknown outer protocol(s): {sorted(unknown)}")
    results: dict[str, Any] = {}
    p_values: dict[str, float] = {}
    secondary_p_values: dict[str, float] = {}
    for name in ("task", "time", "lineage"):
        if name not in outer_splits:
            results[name] = {"protocol": name, "status": "PARTIAL", "partial_reason": "frozen outer split not supplied"}
            continue
        try:
            result = _run_protocol(rows, name, outer_splits[name], inner_folds=inner_folds)
            results[name] = result
            p_values[name] = result["sign_flip_p_unadjusted"]
            secondary_p_values[name] = result["secondary_exact_success_brier"]["sign_flip_p_unadjusted"]
        except ValueError as exc:
            results[name] = {"protocol": name, "status": "PARTIAL", "partial_reason": str(exc), "outer_split": outer_splits[name]}
    # The frozen family remains three protocols even when a protocol is not
    # estimable; absent tests cannot shrink the multiplicity penalty.
    adjusted = holm_adjust(p_values, family_size=3) if p_values else {}
    secondary_adjusted = holm_adjust(secondary_p_values, family_size=3) if secondary_p_values else {}
    for name, result in results.items():
        if name not in p_values:
            continue
        result["sign_flip_p_holm"] = adjusted[name]
        # Conservative family-wise simultaneous intervals: Bonferroni percentile
        # task bootstrap, using the already task-clustered effect vector.
        effects = list(result["task_cluster_effects"].values())
        result["simultaneous_ci95"] = _bootstrap_ci(effects, 20301001 + len(name), 0.05 / 3)
        lo, hi = result["simultaneous_ci95"]
        signs_agree = bool(result["heldout_lineage_effects"]) and all(v > 0 for v in result["heldout_lineage_effects"].values())
        if result["status"] != "ESTIMATED":
            result["decision"] = "PARTIAL"
        elif lo > PRACTICAL_THRESHOLD and signs_agree:
            result["decision"] = "HISTORICAL_PREDICTIVE_CANDIDATE"
        elif lo <= PRACTICAL_THRESHOLD <= hi:
            result["decision"] = "NEED_ONE_MORE_DISAMBIGUATING_EXPERIMENT"
        elif hi <= PRACTICAL_THRESHOLD:
            result["decision"] = "STOP_CURRENT_SIGNAL_EXPANSION"
        else:
            result["decision"] = "PARTIAL"
        secondary = result["secondary_exact_success_brier"]
        secondary["sign_flip_p_holm"] = secondary_adjusted[name]
        secondary_effects = list(secondary["task_cluster_effects"].values())
        secondary["simultaneous_ci95"] = _bootstrap_ci(secondary_effects, 20403001 + len(name), 0.05 / 3)
        sec_lo, sec_hi = secondary["simultaneous_ci95"]
        secondary_decision = (
            "SECONDARY_SIGNAL" if sec_lo > SECONDARY_BRIER_THRESHOLD else
            "NEED_ONE_MORE_DISAMBIGUATING_EXPERIMENT" if sec_lo <= SECONDARY_BRIER_THRESHOLD <= sec_hi else
            "STOP_SECONDARY_SIGNAL" if sec_hi <= SECONDARY_BRIER_THRESHOLD else "PARTIAL"
        )
        secondary["decision"] = secondary_decision if result["status"] == "ESTIMATED" else "PARTIAL"
    return {
        "schema": "openevo.q06.nested-prediction-statistics.v1",
        "status": "COMPLETE_DIAGNOSTIC" if len(p_values) == 3 and all(results[k]["status"] == "ESTIMATED" for k in required) else "PARTIAL",
        "estimand": "heldout MAE reduction = baseline MAE - uncertainty-augmented MAE",
        "practical_effect_threshold": PRACTICAL_THRESHOLD,
        "outer_protocol_family": ["task", "time", "lineage"],
        "holm_family_size": 3,
        "computed_protocol_count": len(p_values),
        "secondary_exact_success_brier_family": {
            "holm_family_size": 3,
            "computed_protocol_count": len(secondary_p_values),
            "practical_effect_threshold": SECONDARY_BRIER_THRESHOLD,
            "independent_of_primary_mae_decision": True,
        },
        "confidence_intervals": "task-cluster percentile bootstrap; Bonferroni family-wise 95% intervals",
        "bootstrap_resamples": BOOTSTRAPS,
        "sign_flip_resamples": BOOTSTRAPS,
        "protocols": results,
        "model_calls": 0,
        "environment_calls": 0,
        "final_panel_access": 0,
        "claim": "historical predictive screening only; prospective independent trajectories are required before controller use",
    }
