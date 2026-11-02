from __future__ import annotations

import json
from pathlib import Path

from scripts.openevo_webshop.reconcile_horizon_diagnostic import reconcile


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


def test_horizon_reconciliation_pairs_two_conditions(tmp_path: Path) -> None:
    manifest = {
        "manifest_sha256": "test",
        "rollout": {"seeds": [1, 2]},
        "task_selection": {"partitions": {"P1": [101, 102], "P2": [], "P3": [], "P4": []}},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    paths = {}
    for condition, arm in (("base-15", "base"), ("base-30", "base-30")):
        path = tmp_path / condition
        path.mkdir()
        rows = [_row(arm, task, seed, score=1.0 if condition == "base-30" else 0.0) for task in (101, 102) for seed in (1, 2)]
        (path / "attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        paths[f"{condition}-P1"] = path
    for condition in ("base-15", "base-30"):
        for partition in ("P2", "P3", "P4"):
            path = tmp_path / f"{condition}-{partition}"
            path.mkdir()
            (path / "attempts.jsonl").write_text("")
            paths[f"{condition}-{partition}"] = path

    result = reconcile(tmp_path, manifest_path, paths)

    assert result["status"] == "pass"
    assert result["expected_attempt_count"] == 8
    assert result["scientific_valid_attempt_count"] == 8
    assert all(row["delta_base30_minus_base15"] == 1.0 for row in result["paired"])
