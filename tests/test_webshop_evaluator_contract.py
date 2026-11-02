from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "vendor/openevo-webshop/src/openevo_webshop/evaluator.py"


class _BaseTrajectoryEvaluator:
    def __init__(self, **config):
        self.config = config


class _EvalResult:
    def __init__(self, *, outcome_reward, trace_rewards, metadata):
        self.outcome_reward = outcome_reward
        self.trace_rewards = trace_rewards
        self.metadata = metadata


def _load_evaluator(monkeypatch):
    module_names = (
        "openevo",
        "openevo.trajectory",
        "openevo.trajectory.evaluator",
        "openevo.trajectory.evaluator.base",
        "openevo.trajectory.models",
    )
    modules = {name: types.ModuleType(name) for name in module_names}
    modules["openevo.trajectory.evaluator.base"].BaseTrajectoryEvaluator = (
        _BaseTrajectoryEvaluator
    )
    modules["openevo.trajectory.models"].EvalResult = _EvalResult
    modules["openevo.trajectory.models"].Trajectory = object
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("private-source_evaluator_contract", EVALUATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WebshopTrajectoryEvaluator


def test_evaluator_reads_real_runner_summary_shape(tmp_path, monkeypatch):
    evaluator_type = _load_evaluator(monkeypatch)
    score_path = tmp_path / "webshop_score.json"
    score_path.write_text(
        json.dumps(
            {
                "n_episodes": 1,
                "episodes": [
                    {
                        "steps": 4,
                        "task_score": 0.625,
                        "won": False,
                    }
                ],
                "avg_task_score": 0.625,
                "win_rate": 0.0,
            }
        ),
        encoding="utf-8",
    )

    result = asyncio.run(
        evaluator_type().evaluate(object(), session_dir=str(tmp_path))
    )

    assert result.outcome_reward == 0.625
    assert result.metadata == {
        "won": False,
        "steps": 4,
        "episodes": 1,
        "win_rate": 0.0,
        "source": "webshop",
    }
