#!/usr/bin/env python3
"""Build the frozen SEED-compatible official-held-out 128-task panel manifest.

This script runs on the server (it needs the real WebShop env + JDK). It reads
the SimServer goal list for the canonical ``env.seed=0`` ordering directly from
the constructed environment's attribute state. It does NOT run any model, take
any environment step, or start any episode: it only reads goal metadata that is
already materialized at SimServer construction. That is catalog inspection, not
scientific task consumption, and it touches only the held-out pool 0-499 for
selection (never goal_idx >= 500).

Selection (outcome-blind, pre-frozen, mirrors SEED's first-validation draw):
    np.random.RandomState(1000).choice(np.arange(500), size=128, replace=False)
seed 1000 = env.seed(0) + 1000, matching SEED's build_webshop_envs(seed=env.seed+1000).
Draw order is preserved as the selection/panel order. This is a SEED-COMPATIBLE
panel, explicitly NOT the exact SEED paper evaluation denominator (which is
unrecoverable because SEED's validation RandomState advances per call).

The manifest records, per selected task: panel_ordinal, goal_idx, instruction,
instruction_sha256, asin, category, env_seed, selection_seed, selection order,
and a frozen per-episode generation seed. At the top level it records the data
file hashes, the full ordered-goals digest, the selected-panel digest, and the
manifest content SHA-256. It is committed/frozen BEFORE any formal GPU task
consumption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

EXPECTED_GOALS_SMALL = 6910
HELD_OUT_POOL_SIZE = 500  # goal_idx 0..499; 500+ is the train pool (forbidden)
PANEL_SIZE = 128
SELECTION_SEED = 1000  # env.seed(0) + 1000, mirrors SEED build_webshop_envs
GENERATION_SEED_BASE = 2026082500
SCHEMA = "openevo.webshop-seed-heldout-panel.v1"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_digest(value: Any) -> str:
    return _sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _load_goals(seed_root: Path, num_products: int, environment_seed: int) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Construct the real env and read the shuffled SimServer goal list.

    Returns (goals, data_hashes). Attribute access only; no episode is started.
    """
    from openevo_webshop.env_compat import WebshopEnv, WebshopEnvConfig

    cfg = WebshopEnvConfig(seed_root=str(seed_root), num_products=num_products, seed=environment_seed)
    env = WebshopEnv(cfg)
    raw = env._build()  # noqa: SLF001 - builder introspects construction state
    base = getattr(raw, "unwrapped", raw)
    server = getattr(base, "server", None)
    if server is None or not hasattr(server, "goals"):
        env.close()
        raise SystemExit("could not access SimServer.goals from the constructed env")
    goals = list(server.goals)
    data_dir = Path(cfg.data_dir)
    data_hashes = {}
    for name in ("items_shuffle_1000.json", "items_ins_v2_1000.json"):
        path = data_dir / name
        if not path.is_file():
            env.close()
            raise SystemExit(f"missing WebShop data file: {path}")
        data_hashes[name] = _sha256_bytes(path.read_bytes())
    env.close()
    return goals, data_hashes


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    goals, data_hashes = _load_goals(args.seed_root, args.num_products, args.environment_seed)
    goals_total = len(goals)
    if args.environment_seed == 0 and goals_total != EXPECTED_GOALS_SMALL:
        raise SystemExit(
            f"expected {EXPECTED_GOALS_SMALL} synthetic goals for the small WebShop, observed {goals_total}"
        )
    if goals_total < HELD_OUT_POOL_SIZE:
        raise SystemExit(f"goal list too small for the held-out pool: {goals_total}")

    # Full ordered-goals digest over (index, instruction_text, asin) so the exact
    # env-seed ordering is hash-bound even beyond the 128 selected tasks.
    ordered_fingerprint = [
        {"goal_idx": i, "instruction_sha256": _sha256_text(str(g.get("instruction_text", ""))), "asin": g.get("asin")}
        for i, g in enumerate(goals)
    ]
    goals_list_digest = _canonical_digest(ordered_fingerprint)

    selection = np.random.RandomState(SELECTION_SEED).choice(
        np.arange(HELD_OUT_POOL_SIZE), size=PANEL_SIZE, replace=False
    )
    selected_goal_idxs = [int(x) for x in selection]
    if len(set(selected_goal_idxs)) != PANEL_SIZE:
        raise SystemExit("selection produced duplicate goal_idx")

    tasks = []
    for ordinal, goal_idx in enumerate(selected_goal_idxs):
        goal = goals[goal_idx]
        instruction = str(goal.get("instruction_text", ""))
        if not instruction:
            raise SystemExit(f"empty instruction_text at goal_idx {goal_idx}")
        tasks.append(
            {
                "panel_ordinal": ordinal,
                "task_id": f"webshop_seed0_goal{goal_idx:05d}",
                "goal_idx": goal_idx,
                "split": "held_out",
                "instruction": instruction,
                "instruction_sha256": _sha256_text(instruction),
                "asin": goal.get("asin"),
                "category": goal.get("category"),
                "environment_seed": int(args.environment_seed),
                "selection_seed": SELECTION_SEED,
                "selection_order": ordinal,
                "generation_seed": GENERATION_SEED_BASE + ordinal,
            }
        )

    selected_panel_digest = _canonical_digest(
        [{"goal_idx": t["goal_idx"], "instruction_sha256": t["instruction_sha256"]} for t in tasks]
    )

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign_id": args.campaign_id,
        "title": "SEED-compatible official-held-out 128-task WebShop panel",
        "created_utc": args.created_utc,
        "benchmark": {
            "name": "WebShop small (official held-out)",
            "environment": "WebAgentTextEnv-v0",
            "num_products": int(args.num_products),
            "human_goals": False,
            "goals_total": goals_total,
            "held_out_pool": [0, HELD_OUT_POOL_SIZE - 1],
            "train_pool_start": HELD_OUT_POOL_SIZE,
            "data_files": data_hashes,
            "environment_seed": int(args.environment_seed),
            "goals_list_digest": goals_list_digest,
        },
        "selection": {
            "method": "numpy.random.RandomState(1000).choice(arange(500), size=128, replace=False)",
            "selection_seed": SELECTION_SEED,
            "selection_basis": "SEED first-validation draw (env.seed+1000), outcome-blind, draw order preserved",
            "panel_size": PANEL_SIZE,
            "label": "SEED-compatible official-held-out 128-task panel",
            "not_exact_paper_denominator": True,
            "selected_panel_digest": selected_panel_digest,
        },
        "generation": {
            "generation_seed_base": GENERATION_SEED_BASE,
            "per_episode_seed": "generation_seed_base + panel_ordinal",
            "per_step_seed": "per_episode_seed + step_index",
        },
        "tasks": tasks,
    }
    # content hash computed over the manifest with the hash field absent
    manifest["manifest_content_sha256"] = _canonical_digest(
        {k: v for k, v in manifest.items() if k != "manifest_content_sha256"}
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-seed", type=int, default=0)
    parser.add_argument("--num-products", type=int, default=1000)
    parser.add_argument("--created-utc", required=True, help="ISO-8601 UTC timestamp (passed in for determinism)")
    args = parser.parse_args()
    manifest = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    file_sha = _sha256_bytes(args.output.read_bytes())
    print(json.dumps({
        "output": str(args.output),
        "goals_total": manifest["benchmark"]["goals_total"],
        "panel_size": len(manifest["tasks"]),
        "selected_panel_digest": manifest["selection"]["selected_panel_digest"],
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "manifest_file_sha256": file_sha,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
