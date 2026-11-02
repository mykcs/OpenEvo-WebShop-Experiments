#!/usr/bin/env python3
"""Analyze the SEED held-out comparison (primary contract only for the table).

Reads the per-episode records for the PRIMARY contract (both arms) and produces:

- per-arm aggregates: mean raw task_score, mean score x100, exact success rate,
  valid/total, invalid rate + invalid taxonomy (terminal reasons, invalid steps);
- the paired primary estimand OPEN_EVO_SD - BASE on identical held-out task
  identities: task-weighted paired mean delta, bootstrap 95% CI (10000 rounds,
  seed 2026082501), and paired positive/tie/negative counts;
- BOTH valid-only and ITT/all-scheduled estimands (parser-invalid episodes are
  never silently dropped);
- the SEED paper-reported reference block (89.7 / 78.1) with the numerical,
  non-paired gaps OpenEvo - SEED_reported (no bootstrap against the paper).

The DIAGNOSTIC contract is summarized separately and never enters the comparison
table. Emits an analysis JSON and a claim-boundary-checked conclusion text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

BOOTSTRAP_ROUNDS = 10_000
BOOTSTRAP_SEED = 2026082501
SEED_PAPER = {"score": 89.7, "success": 78.1}
SEED_PAPER_GRPO = {"score": 80.9, "success": 72.6}


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_rows(cell_dir: Path) -> list[dict[str, Any]]:
    episodes_dir = cell_dir / "episodes"
    rows = []
    for path in sorted(episodes_dir.glob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _arm_aggregates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    valid_rows = [r for r in rows if r.get("valid_episode")]
    scores = [float(r.get("final_task_score", 0.0)) for r in rows]
    wins = sum(1 for r in rows if r.get("won"))
    valid_wins = sum(1 for r in valid_rows if r.get("won"))
    valid_scores = [float(r.get("final_task_score", 0.0)) for r in valid_rows]
    terminal = Counter(str(r.get("terminal_reason")) for r in rows)
    invalid_step_hist = Counter(int(r.get("invalid_action_steps", 0)) for r in rows)
    return {
        "total_episodes": total,
        "valid_episodes": len(valid_rows),
        "invalid_episodes": total - len(valid_rows),
        "invalid_rate": (total - len(valid_rows)) / total if total else 0.0,
        "mean_task_score_itt": sum(scores) / total if total else 0.0,
        "mean_score_x100_itt": (sum(scores) / total * 100.0) if total else 0.0,
        "exact_success_rate_itt": wins / total if total else 0.0,
        "mean_task_score_valid_only": sum(valid_scores) / len(valid_rows) if valid_rows else 0.0,
        "mean_score_x100_valid_only": (sum(valid_scores) / len(valid_rows) * 100.0) if valid_rows else 0.0,
        "exact_success_rate_valid_only": valid_wins / len(valid_rows) if valid_rows else 0.0,
        "terminal_reason_taxonomy": dict(terminal),
        "invalid_action_steps_histogram": {str(k): v for k, v in sorted(invalid_step_hist.items())},
    }


def _paired(base_rows: list[dict[str, Any]], sd_rows: list[dict[str, Any]], valid_only: bool) -> dict[str, Any]:
    base = {int(r["goal_idx"]): r for r in base_rows}
    sd = {int(r["goal_idx"]): r for r in sd_rows}
    common = sorted(set(base) & set(sd))
    deltas = []
    pos = tie = neg = 0
    for g in common:
        rb, rs = base[g], sd[g]
        if valid_only and not (rb.get("valid_episode") and rs.get("valid_episode")):
            continue
        d = float(rs.get("final_task_score", 0.0)) - float(rb.get("final_task_score", 0.0))
        deltas.append(d)
        if d > 1e-9:
            pos += 1
        elif d < -1e-9:
            neg += 1
        else:
            tie += 1
    n = len(deltas)
    mean_delta = sum(deltas) / n if n else 0.0
    rng = random.Random(BOOTSTRAP_SEED)
    boots = []
    for _ in range(BOOTSTRAP_ROUNDS):
        sample = [deltas[rng.randrange(n)] for _ in range(n)] if n else []
        boots.append(sum(sample) / n if n else 0.0)
    boots.sort()
    lo = boots[int(0.025 * BOOTSTRAP_ROUNDS)] if n else 0.0
    hi = boots[int(0.975 * BOOTSTRAP_ROUNDS) - 1] if n else 0.0
    return {
        "paired_tasks": n,
        "mean_delta": mean_delta,
        "mean_delta_x100": mean_delta * 100.0,
        "bootstrap": {"rounds": BOOTSTRAP_ROUNDS, "seed": BOOTSTRAP_SEED, "ci95": [lo, hi], "ci95_x100": [lo * 100.0, hi * 100.0]},
        "positive": pos,
        "tie": tie,
        "negative": neg,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conclusion", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.task_manifest.read_text(encoding="utf-8"))
    run_root = args.run_root

    primary_base = _load_rows(run_root / "primary" / "base")
    primary_sd = _load_rows(run_root / "primary" / "sd-lora")
    diag_base = _load_rows(run_root / "diagnostic" / "base")
    diag_sd = _load_rows(run_root / "diagnostic" / "sd-lora")

    analysis: dict[str, Any] = {
        "schema": "openevo.webshop-seed-heldout-analysis.v1",
        "campaign_id": args.campaign_id,
        "task_manifest_file_sha256": hashlib.sha256(args.task_manifest.read_bytes()).hexdigest(),
        "selected_panel_digest": manifest["selection"]["selected_panel_digest"],
        "panel_label": "SEED-compatible official-held-out 128-task panel (not the exact SEED paper denominator)",
        "primary": {
            "base": _arm_aggregates(primary_base),
            "sd_lora": _arm_aggregates(primary_sd),
            "paired_itt": _paired(primary_base, primary_sd, valid_only=False),
            "paired_valid_only": _paired(primary_base, primary_sd, valid_only=True),
        },
        "diagnostic_openevo_native": {
            "note": "OpenEvo-native contract; does NOT enter the comparison table.",
            "base": _arm_aggregates(diag_base),
            "sd_lora": _arm_aggregates(diag_sd),
        },
        "seed_paper_reference": {
            "label": "SEED paper-reported (Table 1, Qwen2.5-7B-Instruct); training/checkpoint NOT locally reproduced",
            "seed": SEED_PAPER,
            "grpo": SEED_PAPER_GRPO,
        },
    }

    p = analysis["primary"]
    sd_score = p["sd_lora"]["mean_score_x100_itt"]
    sd_succ = p["sd_lora"]["exact_success_rate_itt"] * 100.0
    base_score = p["base"]["mean_score_x100_itt"]
    base_succ = p["base"]["exact_success_rate_itt"] * 100.0
    analysis["seed_paper_reference"]["gap_sd_minus_seed_reported"] = {
        "score_x100": sd_score - SEED_PAPER["score"],
        "success_pct": sd_succ - SEED_PAPER["success"],
        "note": "numerical, non-paired gap; no bootstrap computed against the paper reference",
    }

    analysis["analysis_sha256"] = _digest({k: v for k, v in analysis.items() if k != "analysis_sha256"})

    conclusion = f"""# SEED held-out comparison — conclusion (campaign {args.campaign_id})

On our frozen SEED-compatible official-held-out WebShop panel (128 tasks, env.seed=0,
goal_idx 0-499), the frozen OpenEvo Qwen2.5-7B policy obtains
**{sd_score:.1f} task score** and **{sd_succ:.1f}% exact success** (ITT, primary
SEED-strict contract), compared with the SEED paper-reported reference of
**89.7 score** and **78.1% success**.

- Local BASE (Qwen2.5-7B-Instruct): {base_score:.1f} score / {base_succ:.1f}% success.
- Local frozen OpenEvo (SD-LoRA): {sd_score:.1f} score / {sd_succ:.1f}% success.
- SEED paper-reported: {SEED_PAPER['score']} score / {SEED_PAPER['success']}% success.
- Numerical gap OpenEvo - SEED_reported: {sd_score - SEED_PAPER['score']:+.1f} score / {sd_succ - SEED_PAPER['success']:+.1f} success (non-paired).

Primary paired estimand (OPEN_EVO_SD - BASE, ITT): mean delta
{p['paired_itt']['mean_delta_x100']:+.2f} score x100 over {p['paired_itt']['paired_tasks']} paired tasks,
bootstrap 95% CI [{p['paired_itt']['bootstrap']['ci95_x100'][0]:+.2f}, {p['paired_itt']['bootstrap']['ci95_x100'][1]:+.2f}]
(10000 rounds, seed {BOOTSTRAP_SEED}); pos/tie/neg = {p['paired_itt']['positive']}/{p['paired_itt']['tie']}/{p['paired_itt']['negative']}.

SEED value is paper-reported; SEED training/checkpoint was not locally reproduced.
This panel is SEED-compatible, not the exact SEED paper evaluation denominator.
"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.conclusion.parent.mkdir(parents=True, exist_ok=True)
    args.conclusion.write_text(conclusion, encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "conclusion": str(args.conclusion),
        "analysis_sha256": analysis["analysis_sha256"],
        "sd_score_x100_itt": sd_score,
        "sd_success_pct_itt": sd_succ,
        "base_score_x100_itt": base_score,
        "paired_mean_delta_x100_itt": p["paired_itt"]["mean_delta_x100"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
