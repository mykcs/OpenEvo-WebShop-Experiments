#!/usr/bin/env python3
"""Paired task-level descriptive analysis for sealed Q06/Q17 attempts JSONL.

No model, environment, or GPU calls are made. Entropy/NLL means are token-
weighted from the per-generation sums and counts retained by the probe.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

BOOTSTRAPS = 50_000
BINS = ((5, 8, "5-8"), (9, 16, "9-16"), (17, 32, "17-32"), (33, 10**9, "33+"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_attempts(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def quantile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def paired_ci(deltas: list[float], name: str) -> dict:
    rng = random.Random(20261001 + len(name))
    n = len(deltas)
    means = []
    for _ in range(BOOTSTRAPS):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return {
        "mean_delta_beta_minus_bounded": statistics.fmean(deltas),
        "ci95": [quantile(means, 0.025), quantile(means, 0.975)],
        "paired_tasks": n,
        "bootstrap_resamples": BOOTSTRAPS,
        "bootstrap_seed": 20261001 + len(name),
    }


def average(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize_row(row: dict) -> dict:
    episode = row["episode"]
    traces = episode["trace"]
    result = {
        "task_identity": row["task_identity"],
        "worker_seed": row["worker_seed"],
        "paired_generation_seed": row["paired_generation_seed"],
        "score": float(episode["score"]),
        "success": int(bool(episode["won"])),
        "steps": len(traces),
        "generation_tokens": 0,
        "reasoning_tokens": 0,
        "action_tokens": 0,
        "reasoning_entropy_mean_per_token": None,
        "action_entropy_mean_per_token": None,
        "reasoning_nll_mean_per_token": None,
        "action_nll_mean_per_token": None,
        "reasoning_margin_mean": None,
        "action_margin_mean": None,
        "reasoning_generation_p90_mean": None,
        "action_generation_p90_mean": None,
        "generation_entropy_p90": [],
        "click": [],
        "sentinel_admissible": 0,
        "sentinel_selected": 0,
        "engineering_invalid": bool(row["engineering_invalid"]),
    }
    sums = {k: 0.0 for k in ("reasoning_entropy", "action_entropy", "reasoning_nll", "action_nll")}
    counts = {k: 0 for k in ("reasoning", "action")}
    margins = {"reasoning": [], "action": []}
    generation_p90s = {"reasoning": [], "action": []}
    for trace in traces:
        diag = trace["generation_diagnostics"]
        for channel in ("reasoning", "action"):
            d = diag[channel]
            count = int(d.get("generated_token_count", 0))
            counts[channel] += count
            result[f"{channel}_tokens"] += count
            sums[f"{channel}_entropy"] += float(d.get("entropy_sum_nats", 0.0))
            sums[f"{channel}_nll"] += float(d.get("chosen_nll_sum_nats", 0.0))
            margins[channel].extend(float(x) for x in d.get("top1_top2_logit_margins", []) if x is not None)
            if d.get("entropy_p90_nats") is not None:
                p90_value = float(d["entropy_p90_nats"])
                generation_p90s[channel].append(p90_value)
                result["generation_entropy_p90"].append({
                    "step": int(trace["step"]),
                    "channel": channel,
                    "token_count": count,
                    "within_generation_entropy_p90_nats": p90_value,
                })
        result["generation_tokens"] += int(diag["combined"]["generated_token_count"])
        click = diag["click_choice"]
        sentinel = click.get("click_search_sentinel", {})
        result["sentinel_admissible"] += int(sentinel.get("status") == "ADMISSIBLE")
        result["sentinel_selected"] += int(bool(sentinel.get("selected")))
        n = int(click.get("candidate_count", 0))
        if n > 0:
            h = float(click["entropy_nats"])
            probabilities = [float(c["conditional_probability_given_click"]) for c in click["candidates"]]
            selected = float(click["selected_action_conditional_probability"])
            ranked = sorted(probabilities, reverse=True)
            selected_rank = 1 + sum(p > selected for p in probabilities)
            result["click"].append({
                "n": n,
                "entropy": h,
                "normalized_entropy": h / math.log(n) if n > 1 else 0.0,
                "hhi": float(click["hhi"]),
                "top1_probability": ranked[0],
                "top1_top2_probability_margin": ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0],
                "selected_probability": selected,
                "selected_rank": selected_rank,
            })
    for channel in ("reasoning", "action"):
        if counts[channel]:
            result[f"{channel}_entropy_mean_per_token"] = sums[f"{channel}_entropy"] / counts[channel]
            result[f"{channel}_nll_mean_per_token"] = sums[f"{channel}_nll"] / counts[channel]
        result[f"{channel}_margin_mean"] = average(margins[channel])
        result[f"{channel}_generation_p90_mean"] = average(generation_p90s[channel])
    return result


def state_average(row: dict, key: str, lo: int | None = None, hi: int | None = None) -> float | None:
    vals = [x[key] for x in row["click"] if lo is None or lo <= x["n"] <= hi]
    return average(vals)


def mean_or_none(values: list[float | None]) -> float | None:
    return average([float(x) for x in values if x is not None])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bounded", type=Path, required=True)
    p.add_argument("--beta", type=Path, required=True)
    p.add_argument("--bounded-summary", type=Path, required=True)
    p.add_argument("--beta-summary", type=Path, required=True)
    p.add_argument("--failed-bounded", type=Path, required=True)
    p.add_argument("--failed-bounded-summary", type=Path, required=True)
    p.add_argument("--candidate-cost-receipt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--p90-csv", type=Path, required=True)
    args = p.parse_args()
    raw = {"bounded": read_attempts(args.bounded), "beta": read_attempts(args.beta)}
    summaries = {"bounded": json.loads(args.bounded_summary.read_text()), "beta": json.loads(args.beta_summary.read_text())}
    cost_receipt = json.loads(args.candidate_cost_receipt.read_text())
    if len(raw["bounded"]) != 64 or len(raw["beta"]) != 64:
        raise SystemExit("expected exactly 64 sealed rows per endpoint")
    if any(summaries[a].get("status") != "PASS" or summaries[a].get("task_count") != 64 for a in ("bounded", "beta")):
        raise SystemExit("primary summary is not a sealed 64-task PASS")
    identity_keys = ("checkpoint_manifest_sha256", "panel_sha256", "task_panel_manifest_sha256", "task_panel_task_identity_sha256", "runtime_execution_sha", "probe_script_sha256", "prompt_contract_sha256", "base_model", "base_model_revision")
    if any(summaries["bounded"].get(k) != summaries["beta"].get(k) for k in identity_keys):
        raise SystemExit("endpoint execution identity differs")
    if cost_receipt.get("schema") != "openevo.q06.candidate-scoring-cost-derivation.v1" or cost_receipt.get("model_loaded") is not False or cost_receipt.get("model_calls") != 0 or cost_receipt.get("environment_calls") != 0:
        raise SystemExit("candidate-cost receipt is not an inert derived-cost record")
    for arm, path in (("bounded", args.bounded), ("beta", args.beta)):
        cost_arm = cost_receipt.get("arms", {}).get(arm, {})
        if cost_arm.get("input_attempts_sha256") != sha256(path) or cost_arm.get("states_scored") != sum(len(r["episode"]["trace"]) for r in raw[arm]):
            raise SystemExit(f"candidate-cost receipt does not bind {arm} input/state count")
    rows = {arm: {x["task_identity"]: summarize_row(x) for x in data} for arm, data in raw.items()}
    if set(rows["bounded"]) != set(rows["beta"]):
        raise SystemExit("task identity sets differ")
    ids = sorted(rows["bounded"])
    for task in ids:
        b, e = rows["bounded"][task], rows["beta"][task]
        if (b["worker_seed"], b["paired_generation_seed"]) != (e["worker_seed"], e["paired_generation_seed"]):
            raise SystemExit(f"seed mismatch for {task}")
        if b["engineering_invalid"] or e["engineering_invalid"]:
            raise SystemExit(f"primary analysis includes invalid attempt: {task}")

    pairs = [(rows["bounded"][i], rows["beta"][i]) for i in ids]
    metrics = {
        "task_score": "score",
        "exact_success": "success",
        "episode_steps": "steps",
        "generation_tokens": "generation_tokens",
        "reasoning_entropy_mean_per_token": "reasoning_entropy_mean_per_token",
        "action_entropy_mean_per_token": "action_entropy_mean_per_token",
        "reasoning_nll_mean_per_token": "reasoning_nll_mean_per_token",
        "action_nll_mean_per_token": "action_nll_mean_per_token",
        "reasoning_margin_mean": "reasoning_margin_mean",
        "action_margin_mean": "action_margin_mean",
        "reasoning_generation_p90_mean": "reasoning_generation_p90_mean",
        "action_generation_p90_mean": "action_generation_p90_mean",
    }
    output = {
        "schema": "openevo.q06.paired-attempt-analysis.v1",
        "population": {"tasks_per_endpoint": 64, "paired_tasks": len(ids), "primary_attempts": 128, "analysis_unit": "task_identity; bootstrap resamples tasks"},
        "input_files": {
            "bounded": {"basename": args.bounded.name, "sha256": sha256(args.bounded)},
            "beta": {"basename": args.beta.name, "sha256": sha256(args.beta)},
            "bounded_summary": {"basename": args.bounded_summary.name, "sha256": sha256(args.bounded_summary)},
            "beta_summary": {"basename": args.beta_summary.name, "sha256": sha256(args.beta_summary)},
            "candidate_cost_receipt": {"basename": args.candidate_cost_receipt.name, "sha256": sha256(args.candidate_cost_receipt)},
        },
        "canonical_server_sources": {
            "bounded_attempts": "/workspace/runs/post-advisor-unanswered-20260928/e2-mechanism-entropy-20260930/bounded_r159_rank128-amendment-retry1-202610010527/attempts.jsonl",
            "bounded_summary": "/workspace/runs/post-advisor-unanswered-20260928/e2-mechanism-entropy-20260930/bounded_r159_rank128-amendment-retry1-202610010527/summary.json",
            "beta_attempts": "/workspace/runs/post-advisor-unanswered-20260928/e2-mechanism-entropy-20260930/e1_beta_direction_r159/attempts.jsonl",
            "beta_summary": "/workspace/runs/post-advisor-unanswered-20260928/e2-mechanism-entropy-20260930/e1_beta_direction_r159/summary.json",
            "failed_initial_bounded_attempts": "/workspace/runs/post-advisor-unanswered-20260928/e2-mechanism-entropy-20260930/bounded_r159_rank128/attempts.jsonl",
            "failed_initial_bounded_summary": "/workspace/runs/post-advisor-unanswered-20260928/e2-mechanism-entropy-20260930/bounded_r159_rank128/summary.json",
            "candidate_cost_receipt": "/workspace/maintenance/q06-e2-click-scoring-cost-derivation-20261001/candidate-cost-receipt-v2.json",
        },
        "execution_identity": {k: summaries["bounded"].get(k) for k in identity_keys},
        "method": {"bootstrap_resamples": BOOTSTRAPS, "ci": "two-sided percentile 95%", "delta_direction": "beta_minus_bounded", "seed_rule": "20261001 + length(metric key)", "entropy_and_nll": "per-task token-weighted means from saved per-generation sum/count; task-level means and paired bootstrap weight each task equally", "click_features": "state means first averaged within task, then paired bootstrap across tasks"},
        "endpoint_summary": {},
        "paired_metrics": {},
        "click_metrics": {},
        "generation_entropy_p90": {},
        "choice_count_strata": {},
        "sentinels": {},
        "discordance": {},
        "limitations": [
            "The frozen per-generation entropy p90 requirement is met from each generation_diagnostics[channel].entropy_p90_nats field. The retained data do not support an additional pooled-across-generations token-level entropy p90, which is not required by the frozen Q06 contract.",
            "Different policies follow different trajectories; click-state summaries are paired by task, not matched-state causal contrasts.",
            "Candidate scoring forward-call counts were not observed by runtime instrumentation; any supplied trie/tokenizer derivation is separately labeled as derived.",
            "These two endpoints do not complete the third historical GDR context endpoint in the frozen Q06 design.",
        ],
    }
    for arm in ("bounded", "beta"):
        vals = list(rows[arm].values())
        all_clicks = [x for task in vals for x in task["click"]]
        p90_by_channel = {
            channel: [float(g["within_generation_entropy_p90_nats"])
                      for task in vals for g in task["generation_entropy_p90"]
                      if g["channel"] == channel]
            for channel in ("reasoning", "action")
        }
        output["endpoint_summary"][arm] = {
            "tasks": len(vals),
            "exact_successes": sum(x["success"] for x in vals),
            "exact_success_rate": statistics.fmean(x["success"] for x in vals),
            "mean_task_score": statistics.fmean(x["score"] for x in vals),
            "mean_steps": statistics.fmean(x["steps"] for x in vals),
            "generation_tokens": sum(x["generation_tokens"] for x in vals),
            "reasoning_tokens": sum(x["reasoning_tokens"] for x in vals),
            "action_tokens": sum(x["action_tokens"] for x in vals),
            "task_mean_reasoning_entropy_per_token": mean_or_none([x["reasoning_entropy_mean_per_token"] for x in vals]),
            "task_mean_action_entropy_per_token": mean_or_none([x["action_entropy_mean_per_token"] for x in vals]),
            "task_mean_reasoning_nll_per_token": mean_or_none([x["reasoning_nll_mean_per_token"] for x in vals]),
            "task_mean_action_nll_per_token": mean_or_none([x["action_nll_mean_per_token"] for x in vals]),
            "click_states": len(all_clicks),
            "observed_action_steps": sum(x["steps"] for x in vals),
            "raw_click_entropy_mean": mean_or_none([x["entropy"] for x in all_clicks]),
            "h_over_log_n_mean": mean_or_none([x["normalized_entropy"] for x in all_clicks]),
            "task_mean_h_over_log_n": mean_or_none([state_average(x, "normalized_entropy") for x in vals]),
            "task_mean_click_entropy": mean_or_none([state_average(x, "entropy") for x in vals]),
            "hhi_mean": mean_or_none([x["hhi"] for x in all_clicks]),
            "sentinel_admissible": sum(x["sentinel_admissible"] for x in vals),
            "sentinel_selected": sum(x["sentinel_selected"] for x in vals),
            "engineering_invalid": sum(x["engineering_invalid"] for x in vals),
            "within_generation_entropy_p90": {
                channel: {
                    "generation_calls": len(values),
                    "mean_across_generation_calls": average(values),
                    "median_across_generation_calls": quantile(sorted(values), 0.5),
                    "p90_of_generation_call_p90s": quantile(sorted(values), 0.9),
                    "task_equal_mean_of_call_p90s": mean_or_none([task[f"{channel}_generation_p90_mean"] for task in vals]),
                }
                for channel, values in p90_by_channel.items()
            },
        }
    for name, key in metrics.items():
        deltas = [float(beta[key]) - float(bounded[key]) for bounded, beta in pairs]
        output["paired_metrics"][name] = paired_ci(deltas, name)
    output["generation_entropy_p90"] = {
        channel: {
            "metric": "within-generation token entropy p90; each value is the p90 among tokens generated in that one call",
            "paired_task_mean_delta_beta_minus_bounded": output["paired_metrics"][f"{channel}_generation_p90_mean"],
        }
        for channel in ("reasoning", "action")
    }
    for key in ("entropy", "normalized_entropy", "hhi", "top1_probability", "top1_top2_probability_margin", "selected_probability", "selected_rank"):
        bmeans = [state_average(b, key) for b, _ in pairs]
        emean = [state_average(e, key) for _, e in pairs]
        valid = [(float(b), float(e)) for b, e in zip(bmeans, emean) if b is not None and e is not None]
        output["click_metrics"][key] = paired_ci([e-b for b, e in valid], "click_" + key) if valid else {"paired_tasks": 0, "mean_delta_beta_minus_bounded": None, "ci95": None}
    for lo, hi, label in BINS:
        b_clicks = [c for rows_ in rows["bounded"].values() for c in rows_["click"] if lo <= c["n"] <= hi]
        e_clicks = [c for rows_ in rows["beta"].values() for c in rows_["click"] if lo <= c["n"] <= hi]
        b_by_task = {task: state_average(r, "normalized_entropy", lo, hi) for task, r in rows["bounded"].items()}
        e_by_task = {task: state_average(r, "normalized_entropy", lo, hi) for task, r in rows["beta"].items()}
        valid = [(b_by_task[t], e_by_task[t]) for t in ids if b_by_task[t] is not None and e_by_task[t] is not None]
        result = {
            "bounded_click_states": len(b_clicks),
            "beta_click_states": len(e_clicks),
            "paired_task_count": len(valid),
            "mean_h_over_log_n_delta_beta_minus_bounded": statistics.fmean(e-b for b, e in valid) if valid else None,
            "ci95_task_cluster": paired_ci([e-b for b, e in valid], "stratum_" + label)["ci95"] if valid else None,
        }
        output["choice_count_strata"][label] = result
    bs = sum(x["success"] for x in rows["bounded"].values())
    es = sum(x["success"] for x in rows["beta"].values())
    both = sum(rows["bounded"][t]["success"] and rows["beta"][t]["success"] for t in ids)
    beta_only = sum(not rows["bounded"][t]["success"] and rows["beta"][t]["success"] for t in ids)
    bounded_only = sum(rows["bounded"][t]["success"] and not rows["beta"][t]["success"] for t in ids)
    neither = len(ids) - both - beta_only - bounded_only
    output["discordance"] = {"both_success": both, "beta_only_success": beta_only, "bounded_only_success": bounded_only, "neither_success": neither, "mcnemar_exact_two_sided_p": min(1.0, 2.0 * sum(math.comb(beta_only+bounded_only, k) for k in range(min(beta_only, bounded_only)+1)) / (2 ** (beta_only+bounded_only)))}
    output["sentinels"] = {"bounded_admissible": sum(x["sentinel_admissible"] for x in rows["bounded"].values()), "bounded_selected": sum(x["sentinel_selected"] for x in rows["bounded"].values()), "beta_admissible": sum(x["sentinel_admissible"] for x in rows["beta"].values()), "beta_selected": sum(x["sentinel_selected"] for x in rows["beta"].values())}
    failed_rows = read_attempts(args.failed_bounded)
    failed_summary = json.loads(args.failed_bounded_summary.read_text())
    output["separate_failed_initial_bounded_attempt"] = {
        "summary_basename": args.failed_bounded_summary.name,
        "summary_sha256": sha256(args.failed_bounded_summary),
        "attempts_basename": args.failed_bounded.name,
        "attempts_sha256": sha256(args.failed_bounded),
        "summary_status": failed_summary.get("status"),
        "task_count": len(failed_rows),
        "engineering_invalid_count": sum(bool(r.get("engineering_invalid")) for r in failed_rows),
        "first_error": failed_rows[0].get("error") if failed_rows else None,
        "environment_steps": sum(int(r.get("episode", {}).get("webshop_env_step_calls", 0)) for r in failed_rows),
        "generated_tokens": sum(int(t.get("generation_diagnostics", {}).get("combined", {}).get("generated_token_count", 0)) for r in failed_rows for t in r.get("episode", {}).get("trace", [])),
        "classification": "preserved failed engineering attempt; excluded from outcome denominator; its separately authorized one-time retry is the primary Bounded endpoint above",
    }
    if output["separate_failed_initial_bounded_attempt"]["task_count"] != 64 or output["separate_failed_initial_bounded_attempt"]["engineering_invalid_count"] != 64 or output["separate_failed_initial_bounded_attempt"]["environment_steps"] != 0 or output["separate_failed_initial_bounded_attempt"]["generated_tokens"] != 0:
        raise SystemExit("initial failed attempt does not match expected zero-consumption evidence")
    output["derived_candidate_scoring"] = {
        "classification": cost_receipt["classification"],
        "receipt_basename": args.candidate_cost_receipt.name,
        "receipt_sha256": sha256(args.candidate_cost_receipt),
        "source_identity": cost_receipt["source_identity"],
        "tokenizer": cost_receipt["tokenizer"],
        "scorer_counting_rule": cost_receipt["scorer_counting_rule"],
        "arms": {
            arm: {
                "states_scored": cost_receipt["arms"][arm]["states_scored"],
                "groups": cost_receipt["arms"][arm]["groups"],
                "expected_scoring_forward_calls": sum(x["estimated_forward_calls"] for x in cost_receipt["arms"][arm]["groups"].values()),
                "expected_scoring_input_tokens": sum(x["estimated_input_tokens"] for x in cost_receipt["arms"][arm]["groups"].values()),
            }
            for arm in ("bounded", "beta")
        },
    }
    output["budget_accounting"] = {
        "frozen_q06_base_ceiling_episodes": 192,
        "frozen_base_cells": {"bounded_rank128": 64, "beta_direction": 64, "historical_gdr_context": 64},
        "base_cell_attempts_completed": {"bounded_rank128_initial_attempt_failed_engineering": 64, "beta_direction": 64},
        "base_cells_remaining": {"historical_gdr_context": 64},
        "explicit_one_time_engineering_retry_addition": {"episodes": 64, "status": "consumed by the corrected Bounded retry; this supplies valid Bounded endpoint evidence"},
        "attempted_episode_budget_so_far_including_invalid": 192,
        "planned_total_if_remaining_gdr_base_cell_runs": 256,
        "note": "The failed initial Bounded group and Beta group account for 128 base attempts. The corrected Bounded retry is an explicit extra 64. The remaining base GDR cell is still 64 and would bring the total attempted episode count to 256. Invalid rows are excluded from outcome denominators, not erased from budget accounting.",
    }
    output["within_generation_entropy_p90_csv"] = args.p90_csv.name
    csv_fields = (
        "endpoint", "channel", "scope", "task_count", "generation_call_count",
        "mean_of_within_generation_p90_nats", "median_of_within_generation_p90_nats",
        "p90_of_generation_call_p90s_nats", "task_equal_mean_of_call_p90s_nats",
        "paired_delta_beta_minus_bounded_nats", "paired_ci95_low_nats", "paired_ci95_high_nats",
    )
    with args.p90_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for arm in ("bounded", "beta"):
            for channel in ("reasoning", "action"):
                summary = output["endpoint_summary"][arm]["within_generation_entropy_p90"][channel]
                ci = output["paired_metrics"][f"{channel}_generation_p90_mean"]
                writer.writerow({
                    "endpoint": arm,
                    "channel": channel,
                    "scope": "within_generation_entropy_p90; distribution across calls is separate",
                    "task_count": 64,
                    "generation_call_count": summary["generation_calls"],
                    "mean_of_within_generation_p90_nats": summary["mean_across_generation_calls"],
                    "median_of_within_generation_p90_nats": summary["median_across_generation_calls"],
                    "p90_of_generation_call_p90s_nats": summary["p90_of_generation_call_p90s"],
                    "task_equal_mean_of_call_p90s_nats": summary["task_equal_mean_of_call_p90s"],
                    "paired_delta_beta_minus_bounded_nats": ci["mean_delta_beta_minus_bounded"],
                    "paired_ci95_low_nats": ci["ci95"][0],
                    "paired_ci95_high_nats": ci["ci95"][1],
                })
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
