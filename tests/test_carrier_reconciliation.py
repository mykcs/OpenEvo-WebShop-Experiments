from __future__ import annotations

import json
from pathlib import Path

from scripts.openevo_webshop.reconcile_carrier_diagnostic import CELLS, reconcile


def _row(arm: str, task: int, seed: int, *, valid: bool = True, score: float = 0.0) -> dict:
    return {
        "arm": arm,
        "goal_idx": task,
        "task_id": f"webshop_{task:06d}",
        "rollout_seed": seed,
        "identity_ok": valid,
        "trace_complete": valid,
        "parser_or_fallback_failure": not valid,
        "error": None if valid else "runtime",
        "qualified_positive": False,
        "final_task_score": score,
    }


def _cell(path: Path, arm: str, tasks: list[int], *, valid: bool = True) -> None:
    path.mkdir()
    rows = [_row(arm, task, seed, valid=valid) for task in tasks for seed in (2026082400, 2026082401, 2026082402, 2026082403)]
    (path / "attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_reconciliation_excludes_runtime_failures(tmp_path: Path) -> None:
    tasks = {"A": [509, 714, 2709, 4111], "B": [4170, 4849, 5678, 6602]}
    paths = {}
    for cell in CELLS:
        arm, partition = cell.rsplit("-", 1)
        path = tmp_path / cell
        _cell(path, arm, tasks[partition], valid=cell != "text-memory-B")
        paths[cell] = path

    result = reconcile(tmp_path, paths)

    assert result["status"] == "incomplete-or-invalid"
    assert result["observed_attempt_count"] == 96
    assert result["scientific_valid_attempt_count"] == 80
    assert result["cells"]["text-memory-B"]["runtime_failures"]
    assert result["paired_excluded_count"] > 0
