#!/usr/bin/env python3
"""Reconcile the H1.11 matched-supervision closed-loop evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def _metrics(trace: list[dict]) -> dict[str, int]:
    actions = [str(step.get("action") or "") for step in trace]
    return {
        "steps": len(trace),
        "next_count": sum("next >" in action.lower() for action in actions),
        "buy_now_count": sum("buy now" in action.lower() for action in actions),
        "parser_failure_steps": sum(
            bool(step.get("parser_failure")) or bool(step.get("used_fallback"))
            for step in trace
        ),
    }


def _rows(run_root: Path, expected_attempts: int) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(run_root.glob("cells/*/episodes/*.json")):
        arm, partition = path.parts[-3].split("-", 1)
        payload = json.loads(path.read_text(encoding="utf-8"))
        episode = payload.get("episode") or {}
        trace = episode.get("trace") or []
        rows.append(
            {
                "key": (
                    partition,
                    arm,
                    int(payload["goal_idx"]),
                    int(payload["rollout_seed"]),
                ),
                "arm": arm,
                "partition": partition,
                "task": int(payload["goal_idx"]),
                "rollout_seed": int(payload["rollout_seed"]),
                "valid": bool(
                    payload.get("identity_ok")
                    and payload.get("trace_complete")
                    and not payload.get("parser_or_fallback_failure")
                    and not payload.get("error")
                ),
                "qualified_positive": bool(payload.get("qualified_positive")),
                "score": float(payload.get("final_task_score") or 0.0),
                "runtime_error": bool(payload.get("error")),
                **_metrics(trace),
            }
        )
    if len(rows) != expected_attempts:
        raise ValueError(f"expected {expected_attempts} attempts, found {len(rows)}")
    if len({row["key"] for row in rows}) != len(rows):
        raise ValueError("duplicate partition/arm/task/seed key")
    return rows


def _summary(rows: list[dict]) -> dict:
    def average(name: str) -> float:
        return round(mean(row[name] for row in rows), 6) if rows else 0.0

    return {
        "attempts": len(rows),
        "valid": sum(row["valid"] for row in rows),
        "qualified_positive": sum(row["qualified_positive"] for row in rows),
        "runtime_errors": sum(row["runtime_error"] for row in rows),
        "score_sum": round(sum(row["score"] for row in rows), 6),
        "mean_score_all": average("score"),
        "mean_steps": average("steps"),
        "mean_next_count": average("next_count"),
        "mean_buy_now_count": average("buy_now_count"),
        "parser_failure_steps": sum(row["parser_failure_steps"] for row in rows),
        "positive_tasks": dict(
            sorted(Counter(row["task"] for row in rows if row["qualified_positive"]).items())
        ),
    }


def _paired(rows: list[dict]) -> dict:
    index = {row["key"]: row for row in rows}
    output: dict[str, dict] = {}
    for partition in ("A", "B"):
        for arm in ("failure", "success"):
            deltas = []
            for row in rows:
                if row["partition"] != partition or row["arm"] != arm:
                    continue
                base = index[(partition, "base", row["task"], row["rollout_seed"])]
                deltas.append(
                    {
                        "valid_delta": int(row["valid"]) - int(base["valid"]),
                        "qualified_positive_delta": int(row["qualified_positive"])
                        - int(base["qualified_positive"]),
                        "score_delta": row["score"] - base["score"],
                        "next_count_delta": row["next_count"] - base["next_count"],
                        "buy_now_delta": row["buy_now_count"] - base["buy_now_count"],
                    }
                )
            output[f"{partition}__{arm}_minus_base"] = {
                "n": len(deltas),
                "valid_delta_sum": sum(item["valid_delta"] for item in deltas),
                "qualified_positive_delta_sum": sum(
                    item["qualified_positive_delta"] for item in deltas
                ),
                "score_delta_sum": round(sum(item["score_delta"] for item in deltas), 6),
                "mean_score_delta": round(mean(item["score_delta"] for item in deltas), 6)
                if deltas
                else 0.0,
                "mean_next_count_delta": round(
                    mean(item["next_count_delta"] for item in deltas), 6
                )
                if deltas
                else 0.0,
                "buy_now_delta_sum": sum(item["buy_now_delta"] for item in deltas),
            }
    return output


def reconcile(
    run_root: Path,
    *,
    code_sha: str,
    evaluation_manifest_sha256: str,
    expected_attempts: int,
    excluded_runtime_attempts: int,
) -> dict:
    rows = _rows(run_root, expected_attempts)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[f"{row['partition']}__{row['arm']}"].append(row)
    summaries = {key: _summary(value) for key, value in sorted(groups.items())}
    success_positive = sum(
        summaries[f"{partition}__success"]["qualified_positive"] for partition in ("A", "B")
    )
    base_positive = sum(
        summaries[f"{partition}__base"]["qualified_positive"] for partition in ("A", "B")
    )
    failure_positive = sum(
        summaries[f"{partition}__failure"]["qualified_positive"] for partition in ("A", "B")
    )
    if success_positive > max(base_positive, failure_positive):
        classification = "positive-supervision-closed-loop-supported"
    elif success_positive == base_positive == failure_positive:
        classification = "no-supervision-arm-separation"
    else:
        classification = "positive-supervision-closed-loop-not-supported"
    return {
        "schema": "seed3090.webshop.h1.11-reconciliation.v1",
        "experiment_group": "H1.11-new-positive-supervision-fit-v1",
        "run_root": str(run_root),
        "code_sha": code_sha,
        "evaluation_manifest_sha256": evaluation_manifest_sha256,
        "attempts": len(rows),
        "valid": sum(row["valid"] for row in rows),
        "qualified_positive": sum(row["qualified_positive"] for row in rows),
        "runtime_errors": sum(row["runtime_error"] for row in rows),
        "excluded_runtime_attempts": excluded_runtime_attempts,
        "group_summaries": summaries,
        "paired_deltas_vs_base": _paired(rows),
        "positive_task_keys": sorted(
            {
                (row["partition"], row["arm"], row["task"], row["rollout_seed"])
                for row in rows
                if row["qualified_positive"]
            }
        ),
        "classification": classification,
        "claim_boundary": (
            "Matched new-supervision fit and fresh-task closed-loop test only; "
            "no benchmark/generalization claim and no D0 opening."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--evaluation-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-attempts", type=int, default=96)
    parser.add_argument("--excluded-runtime-attempts", type=int, default=0)
    args = parser.parse_args()
    result = reconcile(
        args.run_root,
        code_sha=args.code_sha,
        evaluation_manifest_sha256=args.evaluation_manifest_sha256,
        expected_attempts=args.expected_attempts,
        excluded_runtime_attempts=args.excluded_runtime_attempts,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: result[key] for key in ("attempts", "valid", "qualified_positive", "classification")}))


if __name__ == "__main__":
    main()
