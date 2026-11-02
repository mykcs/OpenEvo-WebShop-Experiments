"""Contract tests for WebshopTrajectoryEvaluator."""

from __future__ import annotations

import asyncio
import json

import pytest

from openevo_webshop.evaluator import WebshopTrajectoryEvaluator


class _FakeTrajectory:
    pass


def _run(ev, traj, **runtime):
    return asyncio.run(ev.evaluate(traj, **runtime))


def test_evaluate_reads_score_file(tmp_path):
    score = {"task_score": 0.75, "won": False, "steps": 12, "episodes": 1}
    (tmp_path / "webshop_score.json").write_text(json.dumps(score), encoding="utf-8")
    ev = WebshopTrajectoryEvaluator()
    result = _run(ev, _FakeTrajectory(), session_dir=str(tmp_path))
    assert result.outcome_reward == 0.75
    assert isinstance(result.outcome_reward, float)
    assert result.metadata["won"] is False
    assert result.metadata["steps"] == 12
    assert result.metadata["source"] == "webshop"


def test_evaluate_continuous_partial_score(tmp_path):
    (tmp_path / "webshop_score.json").write_text(
        json.dumps({"task_score": 0.333, "won": False, "steps": 5}), encoding="utf-8"
    )
    ev = WebshopTrajectoryEvaluator()
    result = _run(ev, _FakeTrajectory(), artifacts_dir=str(tmp_path))
    assert result.outcome_reward == pytest.approx(0.333)


def test_evaluate_agent_result_inline_source():
    ev = WebshopTrajectoryEvaluator()
    result = _run(
        ev,
        _FakeTrajectory(),
        agent_result={"task_score": 1.0, "won": True, "steps": 3},
    )
    assert result.outcome_reward == 1.0
    assert result.metadata["won"] is True


def test_evaluate_zero_score_is_valid_not_none():
    ev = WebshopTrajectoryEvaluator()
    result = _run(
        ev,
        _FakeTrajectory(),
        agent_result={"task_score": 0.0, "won": False, "steps": 15},
    )
    assert result.outcome_reward == 0.0
    assert result.outcome_reward is not None


def test_evaluate_missing_score_returns_none_outcome():
    ev = WebshopTrajectoryEvaluator()
    result = _run(ev, _FakeTrajectory(), session_dir=None)
    assert result.outcome_reward is None
    assert result.metadata["source"] == "webshop"


def test_evaluator_config_score_relpath(tmp_path):
    (tmp_path / "custom.json").write_text(
        json.dumps({"task_score": 0.5, "won": False, "steps": 1}), encoding="utf-8"
    )
    ev = WebshopTrajectoryEvaluator(score_relpath="custom.json")
    result = _run(ev, _FakeTrajectory(), session_dir=str(tmp_path))
    assert result.outcome_reward == 0.5
