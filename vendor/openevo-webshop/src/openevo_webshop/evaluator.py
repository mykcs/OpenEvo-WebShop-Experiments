"""WebShop trajectory evaluator.

Custom subclass of ``BaseTrajectoryEvaluator`` (decision D3): WebShop yields a
continuous 0..1 reward (``info['task_score']``), so the built-in
``test_on_output`` string-matching evaluator is not applicable.

The runner (inside the container) writes a small JSON score file under
``/openevo/session/...``; this evaluator reads it back via the runtime context
(``session_dir`` / ``artifacts_dir``) and returns it as ``outcome_reward``.
"""

from __future__ import annotations

import json
import os
from typing import Any

from openevo.trajectory.evaluator.base import BaseTrajectoryEvaluator
from openevo.trajectory.models import EvalResult, Trajectory

# Where the in-container runner writes its score, relative to the session dir.
DEFAULT_SCORE_RELPATH = "webshop_score.json"


class WebshopTrajectoryEvaluator(BaseTrajectoryEvaluator):
    """Score a WebShop trajectory from the runner's written score file.

    Config (via EvaluatorSpec.config):
        score_relpath: path of the score json relative to the session dir.
    """

    def __init__(self, **config: Any) -> None:
        super().__init__(**config)
        self._score_relpath = config.get("score_relpath", DEFAULT_SCORE_RELPATH)

    def _score_path(self, runtime: dict[str, Any]) -> str | None:
        session_dir = runtime.get("session_dir") or runtime.get("artifacts_dir")
        if not session_dir:
            return None
        return os.path.join(str(session_dir), self._score_relpath)

    def _read_score(self, runtime: dict[str, Any]) -> dict[str, Any]:
        # Contract runtime kwargs (BaseTrajectoryEvaluator docstring):
        # session_dir / artifacts_dir (score file) and agent_result (inline payload).
        path = self._score_path(runtime)
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        agent_result = runtime.get("agent_result")
        if isinstance(agent_result, dict) and "task_score" in agent_result:
            return agent_result
        return {}

    async def evaluate(self, trajectory: Trajectory, **runtime: Any) -> EvalResult:
        data = self._read_score(runtime)
        score = data.get("task_score")
        if score is None:
            score = data.get("avg_task_score")
        outcome = float(score) if score is not None else None

        episodes = data.get("episodes")
        if isinstance(episodes, list):
            episode_count = len(episodes)
            steps = sum(int(item.get("steps", 0)) for item in episodes)
        else:
            episode_count = data.get("n_episodes", episodes)
            steps = int(data.get("steps", 0))

        won = data.get("won", False)
        if isinstance(episodes, list) and len(episodes) == 1:
            won = episodes[0].get("won", won)

        return EvalResult(
            outcome_reward=outcome,
            trace_rewards=None,
            metadata={
                "won": bool(won),
                "steps": steps,
                "episodes": episode_count,
                "win_rate": data.get("win_rate"),
                "source": "webshop",
            },
        )
