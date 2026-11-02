from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from scripts.openevo_webshop.ceiling1_stage2_arm_controller import Controller
from scripts.openevo_webshop.run_ceiling1_stage2_arm_202609042025 import CurrentStage2Controller, exact_h2011_constraint_escape_terminal


def sentinel_step() -> dict:
    return {
        "step": 6,
        "base_prompt": "prompt",
        "generation_seed": 148618385,
        "step_seed_policy": "injective-v1",  # legacy-revision-compat: frozen field value
        "constraint_escape": True,
        "parser_repair_used": False,
        "error": "RuntimeError: Harness201.1 action prefix escaped all current action schemas",
    }


def valid_row() -> dict:
    return {"episode": {
        "task_score": 0.0,
        "won": False,
        "trace_complete": False,
        "terminated_on_invalid_action": True,
        "constraint_escape": True,
        "environment_action_schema_violation": False,
        "interface_failure_kind": "constraint_escape",
        "trace": [
            {"step": 0, "runtime_harness": "201.1", "constraint_escape": False, "action": "search[x]", "verifier_state": {}},
            sentinel_step(),
        ],
    }}


def test_exact_terminal_is_not_mechanical_hold() -> None:
    assert exact_h2011_constraint_escape_terminal(valid_row()) is True


def test_near_miss_terminal_remains_blocked() -> None:
    row = valid_row(); row["episode"]["trace"][-1]["error"] = "RuntimeError: unrelated"
    assert exact_h2011_constraint_escape_terminal(row) is False
    row = valid_row(); row["episode"]["trace"][-1]["action"] = "click[1]"
    assert exact_h2011_constraint_escape_terminal(row) is False
    row = valid_row(); row["episode"]["terminated_on_invalid_action"] = False
    assert exact_h2011_constraint_escape_terminal(row) is False
    row = valid_row(); del row["episode"]["trace"][-1]["base_prompt"]
    assert exact_h2011_constraint_escape_terminal(row) is False
    row = valid_row(); row["episode"]["trace"][-1]["unexpected"] = 123
    assert exact_h2011_constraint_escape_terminal(row) is False


def test_run_rollout_allows_only_exact_terminal(tmp_path: Path) -> None:
    rr = tmp_path / "round-132"; rr.mkdir()
    rows = [valid_row()]
    (rr / "attempts.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    c = CurrentStage2Controller.__new__(CurrentStage2Controller)
    with patch.object(Controller, "run_rollout", return_value=None):
        c.run_rollout(132, rr)
    rows[0]["episode"]["trace"][-1]["error"] = "RuntimeError: unrelated"
    (rr / "attempts.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    with patch.object(Controller, "run_rollout", return_value=None):
        with pytest.raises(RuntimeError, match="mechanical hold escape=1"):
            c.run_rollout(132, rr)
