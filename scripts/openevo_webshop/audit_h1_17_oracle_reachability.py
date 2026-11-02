#!/usr/bin/env python3
"""Audit deterministic WebShop task reachability without model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from openevo_webshop.env_compat import WebshopEnv, WebshopEnvConfig


ENVIRONMENT_SEED = 2026


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def audit(seed_root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    env = WebshopEnv(WebshopEnvConfig(seed_root=str(seed_root), num_products=1000, seed=ENVIRONMENT_SEED))
    raw = env._build()
    results: list[dict[str, Any]] = []
    try:
        for task in manifest["tasks"]:
            goal_idx = int(task["goal_idx"])
            goal = raw.server.goals[goal_idx]
            description = str(goal.get("instruction_text") or "")
            if description != task["task_description"] or _sha256_text(description) != task["task_description_sha256"]:
                raise ValueError(f"task identity mismatch for goal {goal_idx}")
            query = str(goal.get("query") or "")
            asin = str(goal.get("asin") or "").lower()
            obs, _ = env.reset(goal_idx, seed=ENVIRONMENT_SEED)
            obs, _, _, _ = env.step(f"search[{query}]")
            page_count = 0
            target_found = False
            while page_count < 8:
                clickables = (env.get_available_actions() or {}).get("clickables", [])
                target_action = next((value for value in clickables if str(value).lower() == asin), None)
                if target_action is not None:
                    obs, _, _, _ = env.step(f"click[{target_action}]")
                    target_found = True
                    break
                if "next >" not in clickables:
                    break
                obs, _, _, _ = env.step("click[next >]")
                page_count += 1
            selected: list[str] = []
            missing: list[str] = []
            reward = 0.0
            task_score = 0.0
            won = False
            buy_available = False
            if target_found:
                for option in (goal.get("goal_options") or {}).values():
                    clickables = (env.get_available_actions() or {}).get("clickables", [])
                    match = next((value for value in clickables if str(value).lower() == str(option).lower()), None)
                    if match is None:
                        missing.append(str(option))
                        continue
                    obs, _, _, _ = env.step(f"click[{match}]")
                    selected.append(str(match))
                clickables = (env.get_available_actions() or {}).get("clickables", [])
                buy_available = any(str(value).lower() == "buy now" for value in clickables)
                if buy_available:
                    obs, reward, _, info = env.step("click[buy now]")
                    task_score = float((info or {}).get("task_score") or 0.0)
                    won = bool((info or {}).get("won"))
            results.append({
                "goal_idx": goal_idx,
                "task_id": task["task_id"],
                "partition": task.get("partition"),
                "target_asin": goal["asin"],
                "query": query,
                "target_found": target_found,
                "search_pages_before_target": page_count + 1 if target_found else page_count,
                "selected_options": selected,
                "missing_options": missing,
                "buy_available": buy_available,
                "task_score": task_score,
                "reward": reward,
                "won": won,
            })
    finally:
        env.close()
    return {
        "schema": "seed3090.webshop-h1.17-oracle-reachability.v1",
        "environment_seed": ENVIRONMENT_SEED,
        "manifest_sha256": manifest.get("manifest_sha256"),
        "method": "exact goal query, deterministic target-ASIN click, exact goal-option clicks, buy-now when available; no model inference",
        "results": results,
        "target_found_count": sum(bool(row["target_found"]) for row in results),
        "qualified_oracle_success_count": sum(bool(row["won"]) for row in results),
        "partial_oracle_count": sum(bool(row["target_found"] and not row["won"] and row["task_score"] > 0) for row in results),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(args.seed_root.resolve(), args.task_manifest.resolve())
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.chmod(0o600)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
