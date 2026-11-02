#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

BOOTSTRAP_SEED = 20260911
BOOTSTRAP_DRAWS = 10000


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * q))
    return ordered[max(0, min(len(ordered) - 1, index))]


def bootstrap_ci(diffs: list[float]) -> list[float] | None:
    if not diffs:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(diffs)
    means = []
    for _ in range(BOOTSTRAP_DRAWS):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    return [quantile(means, 0.025), quantile(means, 0.975)]


def row_fingerprint(row: dict[str, Any]) -> dict[str, Any]:
    episode = row["episode"]
    return {
        "score": float(episode["score"]),
        "won": bool(episode["won"]),
        "steps": int(episode["steps"]),
        "invalid_terminated": bool(episode["invalid_terminated"]),
        "constraint_escape": bool(episode["constraint_escape"]),
        "actions": [step.get("action") for step in episode["trace"]],
        "completion_sha256": [step.get("raw_completion_sha256") for step in episode["trace"]],
        "generation_seeds": [step.get("generation_seed") for step in episode["trace"]],
    }


def aggregate_entropy(row: dict[str, Any], phase: str) -> tuple[float, int]:
    entropy_sum = 0.0
    tokens = 0
    for step in row["episode"]["trace"]:
        diag = step.get("generation_diagnostics")
        if not diag:
            continue
        call = diag[phase]
        entropy_sum += float(call["entropy_sum_nats"])
        tokens += int(call["generated_token_count"])
    return entropy_sum, tokens


def episode_output_tokens(row: dict[str, Any]) -> int:
    return sum(int(step.get("raw_completion_token_count") or 0) for step in row["episode"]["trace"])


def row_entropy_mean(row: dict[str, Any], phase: str) -> float | None:
    entropy_sum, tokens = aggregate_entropy(row, phase)
    return entropy_sum / tokens if tokens else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-equivalent", action="store_true")
    args = parser.parse_args()

    left = load(args.left)
    right = load(args.right)
    left_rows = {row["task_identity"]: row for row in left["rows"]}
    right_rows = {row["task_identity"]: row for row in right["rows"]}
    if left.get("panel_sha256") != right.get("panel_sha256"):
        raise RuntimeError("panel SHA mismatch")
    if list(left_rows) != list(right_rows):
        raise RuntimeError("task order/identity mismatch")
    identities = list(left_rows)
    for task_id in identities:
        a, b = left_rows[task_id], right_rows[task_id]
        invariant = (a["source_session_index"], a["worker_seed"], a["paired_generation_seed"])
        if invariant != (b["source_session_index"], b["worker_seed"], b["paired_generation_seed"]):
            raise RuntimeError(f"paired task/seed mismatch: {task_id}")

    score_diffs = [float(right_rows[k]["episode"]["score"]) - float(left_rows[k]["episode"]["score"]) for k in identities]
    success_diffs = [float(bool(right_rows[k]["episode"]["won"])) - float(bool(left_rows[k]["episode"]["won"])) for k in identities]
    step_diffs = [float(right_rows[k]["episode"]["steps"]) - float(left_rows[k]["episode"]["steps"]) for k in identities]
    token_diffs = [float(episode_output_tokens(right_rows[k]) - episode_output_tokens(left_rows[k])) for k in identities]
    transitions = {"win_to_loss": 0, "loss_to_win": 0, "both_win": 0, "both_loss": 0}
    exact_equivalent = True
    paired_rows = []
    for task_id in identities:
        a, b = left_rows[task_id], right_rows[task_id]
        aw, bw = bool(a["episode"]["won"]), bool(b["episode"]["won"])
        if aw and not bw:
            transitions["win_to_loss"] += 1
        elif not aw and bw:
            transitions["loss_to_win"] += 1
        elif aw and bw:
            transitions["both_win"] += 1
        else:
            transitions["both_loss"] += 1
        same = row_fingerprint(a) == row_fingerprint(b)
        exact_equivalent = exact_equivalent and same
        paired_rows.append({
            "task_identity": task_id,
            "left_score": float(a["episode"]["score"]),
            "right_score": float(b["episode"]["score"]),
            "left_won": aw,
            "right_won": bw,
            "left_steps": int(a["episode"]["steps"]),
            "right_steps": int(b["episode"]["steps"]),
            "left_output_tokens": episode_output_tokens(a),
            "right_output_tokens": episode_output_tokens(b),
            "left_reasoning_entropy_mean_nats": row_entropy_mean(a, "reasoning"),
            "right_reasoning_entropy_mean_nats": row_entropy_mean(b, "reasoning"),
            "left_action_entropy_mean_nats": row_entropy_mean(a, "action"),
            "right_action_entropy_mean_nats": row_entropy_mean(b, "action"),
            "score_diff_right_minus_left": float(b["episode"]["score"]) - float(a["episode"]["score"]),
            "same_behavior_fingerprint": same,
        })

    def phase_mean(rows: dict[str, Any], phase: str) -> float | None:
        total_entropy = 0.0
        total_tokens = 0
        for row in rows.values():
            entropy_sum, tokens = aggregate_entropy(row, phase)
            total_entropy += entropy_sum
            total_tokens += tokens
        return total_entropy / total_tokens if total_tokens else None
    result = {
        "schema": "openevo.q17.advisor-fixed-checkpoint-comparison.202609111535",
        "left_checkpoint_id": left["checkpoint_id"],
        "right_checkpoint_id": right["checkpoint_id"],
        "task_count": len(identities),
        "panel_sha256": left["panel_sha256"],
        "paired_seed_identity_pass": True,
        "exact_behavior_equivalent": exact_equivalent,
        "transitions": transitions,
        "mean_score_diff_right_minus_left": statistics.mean(score_diffs),
        "mean_score_diff_bootstrap_95ci": bootstrap_ci(score_diffs),
        "success_rate_diff_right_minus_left": statistics.mean(success_diffs),
        "success_rate_diff_bootstrap_95ci": bootstrap_ci(success_diffs),
        "mean_steps_diff_right_minus_left": statistics.mean(step_diffs),
        "mean_output_tokens_per_episode_diff_right_minus_left": statistics.mean(token_diffs),
        "left_reasoning_entropy_mean_nats": phase_mean(left_rows, "reasoning"),
        "right_reasoning_entropy_mean_nats": phase_mean(right_rows, "reasoning"),
        "left_action_entropy_mean_nats": phase_mean(left_rows, "action"),
        "right_action_entropy_mean_nats": phase_mean(right_rows, "action"),
        "formal_task_consumption": 0,
        "final_panel_access": 0,
        "paired_rows": paired_rows,
    }
    if args.expect_equivalent and not exact_equivalent:
        result["status"] = "FAIL"
        result["failure_reason"] = "instrumentation changed paired behavior"
    else:
        result["status"] = "PASS"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"paired_rows"}}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
