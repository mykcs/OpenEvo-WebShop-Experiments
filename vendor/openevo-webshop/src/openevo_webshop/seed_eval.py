"""SEED public-code inference episode loop (byte-exact port, SEED @2cf2fad).

This mirrors SEED's WebshopEnvironmentManager reset/step flow exactly:

- ``reset``  -> extract_task (positional split) -> format_obs (reformatted obs)
  -> build NO_HIS prompt; memory empty.
- each step  -> build_text_obs (NO_HIS when no history records, else the history
  template with ``history_length = valid_len``) -> generate -> webshop_projection
  -> store (pre_step reformatted obs, projected action) -> env.step(projected
  action) with PASS-THROUGH semantics (an invalid projected action is still sent
  to the env, which no-ops it; nothing is substituted or repaired).

The episode ``task_score`` is the final step's raw continuous score (0 until the
terminal purchase step), and ``won`` is SEED's ``done and task_score == 1.0``.
Both the raw score and the exact-success flag are preserved separately so the
analysis layer never has to reconstruct one from the other.
"""

from __future__ import annotations

import hashlib
from typing import Any

from openevo_webshop.comparison_contract import (
    format_webshop_observation,
    render_seed_eval_prompt,
    seed_extract_task,
    seed_project_action,
)
from openevo_webshop.env_compat import WebshopEnv
from openevo_webshop.inference import InferenceBackend

SEED_HISTORY_LENGTH = 2


def run_seed_eval_episode(
    env: WebshopEnv,
    backend: InferenceBackend,
    goal_idx: int,
    *,
    environment_seed: int = 0,
    generation_seed: int | None = None,
    max_steps: int = 15,
    max_new: int = 512,
    temperature: float = 0.4,
    history_length: int = SEED_HISTORY_LENGTH,
    expected_task_description_sha256: str | None = None,
) -> dict[str, Any]:
    """Run one SEED-strict WebShop episode and return a per-episode record."""
    obs_raw, _ = env.reset(goal_idx, seed=environment_seed)
    task = seed_extract_task(obs_raw)
    if expected_task_description_sha256 is not None:
        observed = hashlib.sha256(task.encode("utf-8")).hexdigest()
        if observed != expected_task_description_sha256:
            raise ValueError(
                "manifest drift: task description hash mismatch "
                f"(expected {expected_task_description_sha256}, observed {observed})"
            )

    history: list[tuple[str, str]] = []
    per_step: list[dict[str, Any]] = []
    steps = 0
    done = False
    won = False
    task_score = 0.0
    terminal_reason = "max_steps"

    for st in range(max_steps):
        text_obs = format_webshop_observation(obs_raw, task)
        available = env.get_available_actions()
        prompt = render_seed_eval_prompt(
            task_description=task,
            current_observation=text_obs,
            available_actions=available,
            history_records=history,
            history_length=history_length,
        )
        step_seed = None if generation_seed is None else int(generation_seed) + st
        raw = backend.generate(
            prompt,
            max_new=max_new,
            temperature=temperature,
            generation_seed=step_seed,
        )
        action, is_valid = seed_project_action(raw, require_think=True)
        # SEED pass-through: the projected action is sent to the env even when
        # invalid; the env no-ops it. Nothing is substituted or repaired.
        history.append((text_obs, action))
        obs_raw, reward, done, info = env.step(action)
        info = info or {}
        step_score = float(info.get("task_score", reward) or 0.0)
        task_score = step_score  # raw score is 0 until the terminal purchase step
        steps = st + 1
        per_step.append(
            {
                "step": st,
                "prompt": prompt,
                "raw_completion": raw,
                "action": action,
                "is_action_valid": bool(is_valid),
                "generation_seed": step_seed,
                "task_score": step_score,
                "done": bool(done),
            }
        )
        if done:
            won = bool(info.get("won", step_score == 1.0))
            terminal_reason = "done"
            break

    return {
        "episode": int(goal_idx),
        "goal": task,
        "task_description_sha256": hashlib.sha256(task.encode("utf-8")).hexdigest(),
        "steps": steps,
        "task_score": float(task_score),
        "task_score_x100": float(task_score) * 100.0,
        "won": bool(won),
        "done": bool(done),
        "terminal_reason": terminal_reason,
        "environment_seed": int(environment_seed),
        "generation_seed": generation_seed,
        "temperature": float(temperature),
        "max_steps": int(max_steps),
        "max_new_tokens": int(max_new),
        "history_length": int(history_length),
        "invalid_action_steps": sum(1 for s in per_step if not s["is_action_valid"]),
        "trace": per_step,
    }
