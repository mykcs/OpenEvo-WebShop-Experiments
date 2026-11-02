#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def pctl(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def action_type(action: str | None) -> str:
    text = (action or "").strip().lower()
    if text.startswith("search["):
        return "search"
    if text.startswith("click["):
        return "click"
    return "other" if text else "none"


def entropy_nats(counter: Counter[str]) -> float | None:
    total = sum(counter.values())
    if not total:
        return None
    return -sum((n / total) * math.log(n / total) for n in counter.values() if n)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    per_round: list[dict[str, Any]] = []
    active_unsealed: list[int] = []
    fallback_rounds: list[int] = []
    for round_dir in sorted(args.run_root.glob("round-*")):
        if not round_dir.is_dir():
            continue
        try:
            round_index = int(round_dir.name.split("-")[-1])
        except ValueError:
            continue
        if not (round_dir / "ROUND_COMPLETE.json").is_file():
            active_unsealed.append(round_index)
            continue
        attempts_path = round_dir / "attempts.jsonl"
        if not attempts_path.is_file():
            continue
        try:
            attempts = [json.loads(line) for line in attempts_path.read_text().splitlines() if line.strip()]
        except json.JSONDecodeError:
            episode_paths = sorted(round_dir.glob("shard-*/episodes/*.json"))
            attempts = [json.loads(path.read_text()) for path in episode_paths]
            positions = [int(row.get("schedule_position", -1)) for row in attempts]
            if len(attempts) != 128 or len(set(positions)) != 128:
                raise RuntimeError(f"round {round_index}: JSONL framing failed and episode fallback is incomplete")
            fallback_rounds.append(round_index)
        step_tokens: list[int] = []
        episode_tokens: list[int] = []
        steps: list[int] = []
        actions = Counter()
        successes = positives = invalid = repairs = escapes = 0
        for row in attempts:
            episode = row.get("episode") or {}
            score = float(row.get("final_task_score") or episode.get("task_score") or 0.0)
            won = bool(row.get("won")) or bool(episode.get("won")) or score >= 1.0
            successes += int(won)
            positives += int(score > 0)
            invalid += int(bool(row.get("engineering_invalid")) or bool(episode.get("terminated_on_invalid_action")))
            repairs += int(bool(row.get("parser_repair_used")) or bool(episode.get("parser_repair_used")))
            escapes += int(bool(episode.get("constraint_escape")))
            trace = episode.get("trace") or []
            steps.append(int(episode.get("steps") or len(trace)))
            ep_tokens = 0
            for step in trace:
                raw = str(step.get("raw_completion") or "")
                count = len(tokenizer(raw, add_special_tokens=False)["input_ids"]) if raw else 0
                step_tokens.append(count)
                ep_tokens += count
                actions[action_type(step.get("action"))] += 1
            episode_tokens.append(ep_tokens)
        update_path = round_dir / "sd-lora" / "update.json"
        geometry_path = round_dir / "task-vector-geometry.json"
        loss = None
        cosine = None
        update_status = None
        if update_path.is_file():
            update = json.loads(update_path.read_text())
            loss = update.get("training_loss")
            update_status = update.get("status")
        if geometry_path.is_file():
            cosine = json.loads(geometry_path.read_text()).get("adjacent_update_cosine")
        n = len(attempts)
        per_round.append({
            "round": round_index,
            "attempt_count": n,
            "success_rate_pct": 100.0 * successes / n if n else None,
            "positive_reward_rate_pct": 100.0 * positives / n if n else None,
            "mean_episode_steps": statistics.mean(steps) if steps else None,
            "mean_output_tokens_per_step": statistics.mean(step_tokens) if step_tokens else None,
            "p90_output_tokens_per_step": pctl([float(x) for x in step_tokens], 0.90),
            "mean_output_tokens_per_episode": statistics.mean(episode_tokens) if episode_tokens else None,
            "p90_output_tokens_per_episode": pctl([float(x) for x in episode_tokens], 0.90),
            "invalid_or_terminated_count": invalid,
            "parser_repair_count": repairs,
            "constraint_escape_count": escapes,
            "action_type_entropy_nats": entropy_nats(actions),
            "search_action_count": actions["search"],
            "click_action_count": actions["click"],
            "other_action_count": actions["other"],
            "sd_lora_training_loss": loss,
            "sd_lora_update_status": update_status,
            "adjacent_update_cosine": cosine,
        })

    if not per_round:
        raise RuntimeError("no sealed rounds found")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    start, end = per_round[0]["round"], per_round[-1]["round"]
    payload = {
        "schema": "openevo.q17.advisor-existing-log-metrics.202609111535",
        "round_start": start,
        "round_end": end,
        "sealed_round_count": len(per_round),
        "active_unsealed_rounds": active_unsealed,
        "episode_fallback_rounds": fallback_rounds,
        "final_panel_access": 0,
        "model_calls": 0,
        "per_round": per_round,
    }
    json_path = args.out_dir / f"q17_advisor_existing_log_metrics_R{start:03d}_R{end:03d}.json"
    csv_path = args.out_dir / f"q17_advisor_existing_log_metrics_R{start:03d}_R{end:03d}.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_round[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(per_round)
    selected = {row["round"]: row for row in per_round if row["round"] in {70, 110, 127, 128, end}}
    print(json.dumps({
        "sealed_round_count": len(per_round),
        "round_end": end,
        "active_unsealed_rounds": active_unsealed,
        "episode_fallback_rounds": fallback_rounds,
        "selected_rounds": selected,
        "json": str(json_path),
        "csv": str(csv_path),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
