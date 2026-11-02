from __future__ import annotations

import json
from pathlib import Path

from scripts.openevo_webshop.reconcile_carrier_confirmation import reconcile


def _row(arm: str, task: int, seed: int, score: float = 0.0) -> dict:
    return {
        "arm": arm,
        "goal_idx": task,
        "rollout_seed": seed,
        "identity_ok": True,
        "trace_complete": True,
        "parser_or_fallback_failure": False,
        "error": None,
        "qualified_positive": score > 0,
        "final_task_score": score,
    }


def test_confirmation_reconciliation_pairs_three_arms(tmp_path: Path) -> None:
    manifest = {
        "manifest_sha256": "test",
        "rollout": {"seeds": [1, 2]},
        "task_selection": {"partitions": {"P1": [101, 102]}},
        "tasks": [
            {"goal_idx": 101, "partition": "P1"},
            {"goal_idx": 102, "partition": "P1"},
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    paths = {}
    for arm in ("base", "text-memory", "parametric"):
        path = tmp_path / f"{arm}-P1"
        path.mkdir()
        rows = [_row(arm, task, seed, score=1.0 if arm == "parametric" else 0.0) for task in (101, 102) for seed in (1, 2)]
        (path / "attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        paths[f"{arm}-P1"] = path

    result = reconcile(tmp_path, manifest_path, paths)

    assert result["status"] == "pass"
    assert result["expected_attempt_count"] == 12
    assert result["scientific_valid_attempt_count"] == 12
    assert len(result["paired"]["parametric_minus_base"]) == 4
