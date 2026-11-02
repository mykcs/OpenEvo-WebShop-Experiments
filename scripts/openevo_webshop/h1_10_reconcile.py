#!/usr/bin/env python3
"""Reconcile H1.10 new-task transfer from atomic episode artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def _metrics(trace: list[dict]) -> dict[str, int]:
    actions = [str(step.get("action") or "") for step in trace]
    kinds = [action.lower().split("[", 1)[0] for action in actions]
    return {
        "steps": len(trace),
        "generation_words": sum(len(str(step.get("raw_completion") or "").split()) for step in trace),
        "next_count": sum("next >" in action.lower() for action in actions),
        "product_click_count": sum(
            kind == "click" and action.lower().split("[", 1)[-1].rstrip("]") not in {"back to search", "next >", "prev"}
            for action, kind in zip(actions, kinds)
        ),
        "buy_now_count": sum("buy now" in action.lower() for action in actions),
        "parser_failure_count": sum(bool(step.get("parser_failure")) or bool(step.get("used_fallback")) for step in trace),
    }


def _rows(run_root: Path, expected_attempts: int) -> list[dict]:
    rows = []
    for path in sorted(run_root.glob("cells/*/episodes/*.json")):
        partition, prompt_tag, arm = path.parts[-3].split("__")
        payload = json.loads(path.read_text(encoding="utf-8"))
        episode = payload["episode"]
        trace = episode.get("trace") or []
        rows.append({
            "key": (partition, prompt_tag, arm, int(episode["episode"]), int(payload["rollout_seed"])),
            "partition": partition,
            "prompt_tag": prompt_tag,
            "prompt_contract": episode.get("prompt_contract"),
            "arm": arm,
            "task": int(episode["episode"]),
            "rollout_seed": int(payload["rollout_seed"]),
            "valid": bool(payload.get("identity_ok") and payload.get("trace_complete") and not payload.get("parser_or_fallback_failure") and not payload.get("error")),
            "qualified_positive": bool(payload.get("qualified_positive")),
            "score": float(payload.get("final_task_score") or 0.0),
            "runtime_error": bool(payload.get("error")),
            **_metrics(trace),
        })
    if len(rows) != expected_attempts:
        raise ValueError(f"expected {expected_attempts} attempts, found {len(rows)}")
    if len({row["key"] for row in rows}) != len(rows):
        raise ValueError("duplicate partition/prompt/arm/task/seed key")
    return rows


def _summary(rows: list[dict]) -> dict:
    def avg(name: str) -> float:
        return round(mean(row[name] for row in rows), 6) if rows else 0.0

    return {
        "attempts": len(rows),
        "valid": sum(row["valid"] for row in rows),
        "qualified_positive": sum(row["qualified_positive"] for row in rows),
        "runtime_errors": sum(row["runtime_error"] for row in rows),
        "mean_task_score_all": avg("score"),
        "mean_steps": avg("steps"),
        "mean_generation_words": avg("generation_words"),
        "mean_next_count": avg("next_count"),
        "mean_product_click_count": avg("product_click_count"),
        "mean_buy_now_count": avg("buy_now_count"),
        "parser_failure_steps": sum(row["parser_failure_count"] for row in rows),
        "positive_tasks": dict(sorted(Counter(row["task"] for row in rows if row["qualified_positive"]).items())),
    }


def _paired(rows: list[dict]) -> dict:
    index = {row["key"]: row for row in rows}
    output = {}
    for partition in ("A", "B"):
        for arm in ("base", "success"):
            deltas = []
            for row in rows:
                if row["partition"] != partition or row["prompt_tag"] != "compact" or row["arm"] != arm:
                    continue
                treatment = index[(partition, "stateaware", arm, row["task"], row["rollout_seed"])]
                deltas.append({
                    "valid_delta": int(treatment["valid"]) - int(row["valid"]),
                    "qualified_positive_delta": int(treatment["qualified_positive"]) - int(row["qualified_positive"]),
                    "score_delta": treatment["score"] - row["score"],
                    "next_count_delta": treatment["next_count"] - row["next_count"],
                    "buy_now_delta": treatment["buy_now_count"] - row["buy_now_count"],
                })
            output[f"{partition}__{arm}"] = {
                "n": len(deltas),
                "valid_delta_sum": sum(row["valid_delta"] for row in deltas),
                "qualified_positive_delta_sum": sum(row["qualified_positive_delta"] for row in deltas),
                "score_delta_sum": round(sum(row["score_delta"] for row in deltas), 6),
                "mean_next_count_delta": round(mean(row["next_count_delta"] for row in deltas), 6) if deltas else 0.0,
                "buy_now_delta_sum": sum(row["buy_now_delta"] for row in deltas),
            }
    return output


def reconcile(run_root: Path, code_sha: str, prelaunch_sha: str, expected_attempts: int) -> dict:
    rows = _rows(run_root, expected_attempts)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[f"{row['partition']}__{row['prompt_tag']}__{row['arm']}"].append(row)
    summaries = {key: _summary(value) for key, value in sorted(groups.items())}
    state_success = {partition: summaries[f"{partition}__stateaware__success"]["qualified_positive"] for partition in ("A", "B")}
    compact_success = {partition: summaries[f"{partition}__compact__success"]["qualified_positive"] for partition in ("A", "B")}
    base_state = {partition: summaries[f"{partition}__stateaware__base"]["qualified_positive"] for partition in ("A", "B")}
    selective = {
        partition: state_success[partition] >= 1 and state_success[partition] > compact_success[partition] and base_state[partition] == 0
        for partition in ("A", "B")
    }
    if all(selective.values()):
        classification = "new-task-state-aware-transfer-supported"
    elif any(selective.values()):
        classification = "partial-selective-new-task-state-aware-transfer"
    elif any(state_success.values()) or any(compact_success.values()):
        classification = "new-task-positive-not-state-aware-selective"
    else:
        classification = "no-new-task-state-aware-transfer"
    return {
        "schema": "seed3090.webshop.h1.10-reconciliation.v1",
        "experiment_group": "H1.10-new-task-state-aware-transfer-v1",
        "run_root": str(run_root),
        "code_sha": code_sha,
        "prelaunch_manifest_sha256": prelaunch_sha,
        "attempts": len(rows),
        "valid": sum(row["valid"] for row in rows),
        "qualified_positive": sum(row["qualified_positive"] for row in rows),
        "runtime_errors": sum(row["runtime_error"] for row in rows),
        "group_summaries": summaries,
        "paired_prompt_deltas": _paired(rows),
        "positive_task_keys": sorted({(row["partition"], row["prompt_tag"], row["arm"], row["task"], row["rollout_seed"]) for row in rows if row["qualified_positive"]}),
        "classification": classification,
        "claim_boundary": "New-task transfer test only; no benchmark/generalization claim and no D0 opening without independent scientific review.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--prelaunch-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-attempts", type=int, default=128)
    args = parser.parse_args()
    result = reconcile(args.run_root, args.code_sha, args.prelaunch_sha, args.expected_attempts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "attempts": result["attempts"], "valid": result["valid"], "qualified_positive": result["qualified_positive"], "classification": result["classification"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
