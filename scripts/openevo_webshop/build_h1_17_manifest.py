#!/usr/bin/env python3
"""Derive the H1.17 horizon-only diagnostic manifest from H1.16 tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from parametric_rnd import atomic_json, digest_json


ROLLOUT_SEEDS = [2026082600, 2026082601, 2026082602, 2026082603]


def build_manifest(source: Path, *, experiment_sha: str) -> dict[str, Any]:
    original = json.loads(source.read_text(encoding="utf-8"))
    if original.get("schema") != "seed3090.webshop-h1.16-carrier-confirmation-manifest.v1":
        raise ValueError("H1.17 requires the frozen H1.16 task manifest")
    tasks = original["tasks"]
    partitions = original["task_selection"]["partitions"]
    payload: dict[str, Any] = {
        "schema": "seed3090.webshop-h1.17-horizon-manifest.v1",
        "run_kind": "phase_h1.17_horizon_action_interface_diagnostic",
        "source": {
            "experiment_git_sha": experiment_sha,
            "source_h1_16_manifest_sha256": original["manifest_sha256"],
            "official_open_evo_stable": "796e6248a61112a61c880885844d3b0420c0b6ae",
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "model_revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        },
        "benchmark": {
            "name": "webshop",
            "catalog": "indexes_1k",
            "num_products": 1000,
            "environment_seed": 2026,
            "max_steps_conditions": {"base-15": 15, "base-30": 30},
            "max_new_tokens": 512,
            "temperature": 0.0,
            "prompt_contract": "seed3090.webshop-compact-action-prompt.v2",
            "action_contract": "seed3090.webshop-action-parser.v1",
        },
        "task_selection": {
            "source": "H1.16 frozen fresh-task panel",
            "selected_goal_indices": original["task_selection"]["selected_goal_indices"],
            "partitions": partitions,
            "split": "h1.17-interface-horizon-diagnostic",
        },
        "rollout": {
            "attempts": 64,
            "tasks_per_partition": 2,
            "seeds": ROLLOUT_SEEDS,
            "independent_seed_only": True,
        },
        "arms": [
            {"name": "base-15", "runner_arm": "base", "max_steps": 15, "adapter": "none"},
            {"name": "base-30", "runner_arm": "base-30", "max_steps": 30, "adapter": "none"},
        ],
        "tasks": tasks,
        "stop_rule": "one horizon diagnostic; no prompt, adapter, ranking, or parser changes",
    }
    payload["manifest_sha256"] = digest_json(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--experiment-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_manifest(args.source_manifest.resolve(), experiment_sha=args.experiment_sha)
    atomic_json(args.output.resolve(), payload)
    print({"manifest": str(args.output.resolve()), "manifest_sha256": payload["manifest_sha256"], "seeds": ROLLOUT_SEEDS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
