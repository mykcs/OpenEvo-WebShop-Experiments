"""Official OpenEvo harness for the WebShop benchmark.

``agent.import_path`` in a TaskRequest is loaded by
``openevo.harness.factory`` which requires a ``BaseHarness`` subclass (not a
plain function). This module provides that subclass so the WebShop adapter
plugs into the official container end-to-end path:

  AgentSpec(import_path="openevo_webshop.harness:WebshopHarness", ...)

The gateway node executes the ``ExecInput`` returned by ``run_steps`` inside
the Docker runtime. We point it at the module entrypoint
``python -m openevo_webshop.runner``, which runs the episode loop in-container
and writes ``webshop_score.json`` for the evaluator to read back.

Config knobs are read from ``AgentSpec.settings`` / ``AgentSpec.env`` so the
same image can be reused across seeds/models without a rebuild:

  settings.n_episodes / settings.max_steps
  env WEBSHOP_SEED_ROOT / WEBSHOP_SCORE_PATH / OPENAI_BASE_URL (gateway endpoint)
"""

from __future__ import annotations

from openevo.harness.base import BaseHarness
from openevo.runtime.models import ExecInput

# In-container defaults; overridable via AgentSpec.env.
DEFAULT_SEED_ROOT = "/openevo/session/seed"
DEFAULT_SCORE_PATH = "webshop_score.json"


class WebshopHarness(BaseHarness):
    """Run the WebShop episode loop inside the OpenEvo container runtime."""

    def _env_int(self, settings: dict, key: str, default: int) -> int:
        raw = settings.get(key)
        try:
            return int(raw) if raw is not None else default
        except (TypeError, ValueError):
            return default

    def run_steps(self, instruction: str) -> list[ExecInput]:
        s = self.settings or {}
        env = self.env or {}

        n_episodes = self._env_int(s, "n_episodes", 1)
        max_steps = self._env_int(s, "max_steps", 15)
        seed_root = env.get("WEBSHOP_SEED_ROOT", DEFAULT_SEED_ROOT)
        score_path = env.get("WEBSHOP_SCORE_PATH", DEFAULT_SCORE_PATH)
        model_name = self.model_name or "Qwen/Qwen2.5-3B-Instruct"

        # In-container the model is served by the gateway node; the runner
        # talks to it over the OpenAI-compatible endpoint (GatewayBackend).
        # Reuse the cli rollout entrypoint so argparse stays single-sourced.
        command = (
            "python -m openevo_webshop.cli rollout "
            f"--seed-root {seed_root} "
            f"--model {model_name} "
            f"--backend gateway "
            f"--episodes {n_episodes} "
            f"--max-steps {max_steps} "
            f"--out {score_path}"
        )
        return [ExecInput(command=f"bash -lc '{command}'")]
