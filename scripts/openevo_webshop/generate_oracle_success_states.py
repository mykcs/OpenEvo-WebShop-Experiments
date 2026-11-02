#!/usr/bin/env python3
"""Capture deterministic, train-only WebShop success-path states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.openevo_webshop.webshop_comparison_contract import render_compact_webshop_prompt
from openevo_webshop.env_compat import WebshopEnv, WebshopEnvConfig


ENVIRONMENT_SEED = 2026
STATES_PER_SEED = 4
DEFAULT_SEEDS = (2026082600, 2026082601, 2026082602, 2026082603)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _action_for_state(
    env: Any,
    *,
    query: str,
    target_asin: str,
    goal_options: list[str],
    searched: bool,
    target_found: bool,
    selected_options: list[str],
) -> str:
    clickables = [str(value) for value in (env.get_available_actions() or {}).get("clickables", [])]
    lowered = {value.lower(): value for value in clickables}
    if not searched:
        return f"search[{query}]"
    if not target_found:
        target = lowered.get(target_asin.lower())
        if target is not None:
            return f"click[{target}]"
        if "next >" in lowered:
            return f"click[{lowered['next >']}]"
        raise ValueError(f"target {target_asin} is not reachable from the current result page")
    for option in goal_options:
        if option.lower() not in {value.lower() for value in selected_options}:
            match = lowered.get(option.lower())
            if match is None:
                raise ValueError(f"goal option {option!r} is not available")
            return f"click[{match}]"
    buy = lowered.get("buy now")
    if buy is None:
        raise ValueError("buy now is not available after selecting exact goal options")
    return f"click[{buy}]"


def capture(
    seed_root: Path,
    manifest_path: Path,
    output_records: Path,
    output_summary: Path,
    *,
    task_ids: set[str],
    rollout_seeds: tuple[int, ...] = DEFAULT_SEEDS,
    states_per_seed: int = STATES_PER_SEED,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = [task for task in manifest["tasks"] if task["task_id"] in task_ids]
    if len(tasks) != len(task_ids) or len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("oracle task selection is missing or duplicated")
    if states_per_seed <= 0:
        raise ValueError("states_per_seed must be positive")

    env = WebshopEnv(WebshopEnvConfig(seed_root=str(seed_root), num_products=1000, seed=ENVIRONMENT_SEED))
    raw = env._build()
    rows: list[dict[str, Any]] = []
    try:
        for task in sorted(tasks, key=lambda value: int(value["goal_idx"])):
            goal_idx = int(task["goal_idx"])
            goal = raw.server.goals[goal_idx]
            description = str(goal.get("instruction_text") or "")
            if description != task["task_description"] or _sha256_text(description) != task["task_description_sha256"]:
                raise ValueError(f"task identity mismatch for goal {goal_idx}")
            query = str(goal.get("query") or "")
            target_asin = str(goal.get("asin") or "")
            goal_options = [str(value) for value in (goal.get("goal_options") or {}).values()]
            for rollout_seed in rollout_seeds:
                obs, _ = env.reset(goal_idx, seed=int(rollout_seed))
                history: list[tuple[str, str]] = []
                searched = False
                target_found = False
                selected_options: list[str] = []
                state_count = 0
                won = False
                for step_index in range(30):
                    available = env.get_available_actions() or {}
                    action = _action_for_state(
                        env,
                        query=query,
                        target_asin=target_asin,
                        goal_options=goal_options,
                        searched=searched,
                        target_found=target_found,
                        selected_options=selected_options,
                    )
                    if action.lower() == "click[buy now]":
                        next_obs, reward, _, info = env.step(action)
                        won = bool((info or {}).get("won"))
                        if not won:
                            raise ValueError(f"oracle buy did not win goal {goal_idx}")
                        break
                    prompt = render_compact_webshop_prompt(
                        task_description=description,
                        current_observation=obs,
                        available_actions=available,
                        step_count=step_index,
                        history_length=2,
                        action_history="\n".join(
                            f"[Observation {i + 1}: '{old_obs}', Action {i + 1}: '{old_action}']"
                            for i, (old_obs, old_action) in enumerate(history[-2:])
                        ),
                    )
                    if state_count < states_per_seed:
                        rows.append({
                            "task_id": task["task_id"],
                            "goal_idx": goal_idx,
                            "rollout_seed": int(rollout_seed),
                            "source_step": step_index,
                            "source_state": "deterministic-oracle-success-path",
                            "oracle_action": action,
                            "prompt": prompt,
                            "task_description": description,
                            "task_description_sha256": task["task_description_sha256"],
                            "environment_seed": ENVIRONMENT_SEED,
                        })
                        state_count += 1
                    next_obs, _, _, _ = env.step(action)
                    history.append((str(obs), action))
                    obs = next_obs
                    searched = True
                    if action.lower().startswith("click["):
                        click_value = action[6:-1]
                        if click_value.lower() == target_asin.lower():
                            target_found = True
                        elif target_found and click_value.lower() in {value.lower() for value in goal_options}:
                            selected_options.append(click_value)
                if not won or state_count != states_per_seed:
                    raise ValueError(f"oracle path incomplete for goal {goal_idx}, seed {rollout_seed}")
    finally:
        env.close()

    rows.sort(key=lambda value: (value["task_id"], value["rollout_seed"], value["source_step"]))
    output_records.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = output_records.with_name(output_records.name + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(output_records)
    output_records.chmod(0o600)
    summary = {
        "schema": "seed3090.webshop-h1.18-oracle-success-states.v1",
        "manifest_sha256": manifest.get("manifest_sha256"),
        "task_ids": sorted(task_ids),
        "rollout_seeds": list(rollout_seeds),
        "states_per_task_per_seed": states_per_seed,
        "record_count": len(rows),
        "records_sha256": hashlib.sha256(output_records.read_bytes()).hexdigest(),
        "teacher_evaluation_allowed": False,
        "model_inference_used": False,
    }
    output_summary.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_summary.chmod(0o600)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--task-id", action="append", required=True)
    parser.add_argument("--rollout-seed", action="append", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--states-per-seed", type=int, default=STATES_PER_SEED)
    parser.add_argument("--output-records", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(capture(args.seed_root.resolve(), args.task_manifest.resolve(), args.output_records.resolve(), args.output_summary.resolve(), task_ids=set(args.task_id), rollout_seeds=tuple(args.rollout_seed), states_per_seed=args.states_per_seed), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
