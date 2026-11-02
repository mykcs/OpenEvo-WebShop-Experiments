from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.openevo_webshop.ceiling1_stage2_arm_controller import (
    Controller,
    harness_runtime_step_readback_ok,
)


PREFIX_ESCAPE = "RuntimeError: Harness201.1 action prefix escaped all current action schemas"
NO_SCHEMA = "RuntimeError: current environment exposes no compilable action schema"


def terminal(error: str = PREFIX_ESCAPE) -> dict:
    return {
        "step": 6,
        "base_prompt": "prompt",
        "generation_seed": 148618385,
        "step_seed_policy": "injective-v1",  # legacy-revision-compat: frozen field name
        "constraint_escape": True,
        "parser_repair_used": False,
        "error": error,
    }


def test_only_exact_harness2011_constraint_terminal_may_lack_runtime_receipt() -> None:
    assert harness_runtime_step_readback_ok({"runtime_harness": "201.1"}, "201.1")
    assert harness_runtime_step_readback_ok(terminal(PREFIX_ESCAPE), "201.1")
    assert harness_runtime_step_readback_ok(terminal(NO_SCHEMA), "201.1")
    assert not harness_runtime_step_readback_ok({"step": 1}, "201.1")
    assert not harness_runtime_step_readback_ok(terminal("RuntimeError: unrelated"), "201.1")
    assert not harness_runtime_step_readback_ok(terminal(PREFIX_ESCAPE), "2.0.1-thin")
    for missing in ("step", "base_prompt", "generation_seed"):
        bad = terminal()
        del bad[missing]
        assert not harness_runtime_step_readback_ok(bad, "201.1")
    bad = terminal()
    bad["unexpected"] = 123
    assert not harness_runtime_step_readback_ok(bad, "201.1")
    for key, value in (("step", True), ("step", -1), ("base_prompt", None), ("generation_seed", True)):
        bad = terminal()
        bad[key] = value
        assert not harness_runtime_step_readback_ok(bad, "201.1")
    for forbidden in ("runtime_context_receipt", "action", "action_valid", "environment_accepted"):
        bad = terminal()
        bad[forbidden] = {} if forbidden == "runtime_context_receipt" else False
        assert not harness_runtime_step_readback_ok(bad, "201.1")


def test_rollout_producer_terminal_matches_consumer_exception_contract() -> None:
    source = (Path(__file__).parents[1] / "scripts/openevo_webshop/ceiling1_stage2_rollout_worker.py").read_text()
    assert "if 'action prefix escaped' in str(exc) or 'no compilable action schema' in str(exc):" in source
    assert "'constraint_escape':True" in source
    assert "'parser_repair_used':False" in source
    assert "'step_seed_policy':step_seed_policy" in source
    assert "'interface_failure_kind':'constraint_escape'" in source
    assert "'terminated_on_invalid_action':True" in source



def write_round(path: Path, bad_step: dict | None = None) -> dict:
    normal = {"runtime_harness": "201.1"}
    rows = []
    for index in range(128):
        step = bad_step if index == 64 and bad_step is not None else normal
        rows.append({
            "harness_profile": "201.1",
            "step_seed_policy": "injective-v1",  # legacy-revision-compat: frozen field name
            "prompt_contract_sha256": "8efcba565fc11e485192f6ca85cb62119f7322662cff1a5e6597de0608d441b5",
            "episode": {"trace": [step]},
        })
    (path / "attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return {"harness_profile": "201.1", "step_seed_policy": "injective-v1"}  # legacy-revision-compat


def controller() -> Controller:
    value = object.__new__(Controller)
    value.harness_profile = "201.1"
    value.step_seed_policy = "injective-v1"  # legacy-revision-compat: frozen field value
    value.harness201_successor = True
    return value


def test_round_readback_accepts_exact_terminal_without_weakening_other_missing_receipts(tmp_path: Path) -> None:
    summary = write_round(tmp_path, terminal())
    controller().validate_rollout_harness_readback(summary, tmp_path)

    summary = write_round(tmp_path, {"step": 6, "constraint_escape": True})
    with pytest.raises(RuntimeError, match="runtime receipt readback mismatch"):
        controller().validate_rollout_harness_readback(summary, tmp_path)
