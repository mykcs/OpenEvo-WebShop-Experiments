"""Episode runner for WebShop rollouts.

Ported from webshop_rollout.py:113-156 (SEED, verified avg_task_score=0.0 /
win_rate=0.0 on Qwen2.5-3B-Instruct). Single-process, backend-agnostic: the
model only appears through the ``generate`` protocol (decision D2).

Evolution injection point (decision D6): ``inject_evolution`` is a standalone,
auditable pure function applied to the built prompt AFTER WEBSHOP_TEMPLATE
rendering and BEFORE generate(). The evolution product (memory / skill text)
is therefore present at the string level of the final prompt, which is how we
verify "the memory actually entered the prompt" (acceptance C3).
"""

from __future__ import annotations

import json
import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from openevo_webshop.env_compat import WebshopEnv, WebshopEnvConfig
from openevo_webshop.inference import InferenceBackend
from openevo_webshop.comparison_contract import (
    ACTION_CONTRACT,
    COMPACT_PROMPT_CONTRACT,
    PRODUCT_GROUNDED_PROMPT_CONTRACT,
    PROMPT_CONTRACT,
    STATE_AWARE_PROMPT_CONTRACT,
    extract_goal,
    fallback_webshop_action,
    parse_webshop_action,
    render_compact_webshop_prompt,
    render_product_grounded_webshop_prompt,
    render_state_aware_webshop_prompt,
    render_webshop_prompt,
)

# SEED training-side hyperparameters (reused as rollout defaults).
MAX_STEPS = 15
HISTORY_LENGTH = 2
MAX_NEW = 128
OBS_TRUNCATE = 600


def inject_evolution(prompt: str, evolution_text: str | None) -> str:
    """Append evolution memory/skill to a built prompt (decision D6).

    Standalone and auditable: callers can grep the returned prompt for the
    injected content to prove it entered the model input (acceptance C3).
    """
    if not evolution_text:
        return prompt
    return f"{prompt}\n\n[evolution]\n{evolution_text}\n[/evolution]\n"


@dataclass
class EpisodeResult:
    episode: int
    goal: str
    steps: int
    task_score: float
    won: bool
    reward: float
    error: bool = False
    prompt_contract: str = PROMPT_CONTRACT
    action_contract: str = ACTION_CONTRACT
    trace: list[dict[str, Any]] = field(default_factory=list)


def run_episode(
    env: WebshopEnv,
    backend: InferenceBackend,
    idx: int,
    evolution_text: str | None = None,
    max_steps: int = MAX_STEPS,
    max_new: int = MAX_NEW,
    temperature: float = 0.4,
    generation_seed: int | None = None,
    environment_seed: int = 2026,
    prompt_contract: str = PROMPT_CONTRACT,
    expected_task_description_sha256: str | None = None,
) -> EpisodeResult:
    """Run one WebShop episode. Mirrors webshop_rollout.py:115-147."""
    obs, info = env.reset(idx, seed=environment_seed)
    goal = extract_goal(obs)
    if expected_task_description_sha256 is not None:
        observed_hash = hashlib.sha256(goal.encode("utf-8")).hexdigest()
        if observed_hash != expected_task_description_sha256:
            raise ValueError(
                "manifest drift: task description hash mismatch "
                f"(expected {expected_task_description_sha256}, observed {observed_hash})"
            )
    hist: list[str] = []
    steps, won, task_score, total_r = 0, False, 0.0, 0.0
    trace: list[dict[str, Any]] = []

    for st in range(max_steps):
        av = env.get_available_actions()
        prompt_kwargs = {
            "task_description": goal or "buy the required product",
            "current_observation": obs,
            "available_actions": av,
            "step_count": st,
            "history_length": HISTORY_LENGTH,
            "action_history": " | ".join(hist[-HISTORY_LENGTH:]) or "(none)",
            "max_prompt_chars": 20_000,
        }
        if prompt_contract == PROMPT_CONTRACT:
            prompt = render_webshop_prompt(**prompt_kwargs)
        elif prompt_contract == COMPACT_PROMPT_CONTRACT:
            prompt = render_compact_webshop_prompt(**prompt_kwargs)
        elif prompt_contract == STATE_AWARE_PROMPT_CONTRACT:
            prompt = render_state_aware_webshop_prompt(**prompt_kwargs)
        elif prompt_contract == PRODUCT_GROUNDED_PROMPT_CONTRACT:
            prompt = render_product_grounded_webshop_prompt(**prompt_kwargs)
        else:
            raise ValueError(f"unsupported WebShop prompt contract: {prompt_contract}")
        prompt = inject_evolution(prompt, evolution_text)  # D6 injection point
        step_seed = None if generation_seed is None else int(generation_seed) + st
        raw = backend.generate(
            prompt,
            max_new=max_new,
            temperature=temperature,
            generation_seed=step_seed,
        )
        act, action_valid = parse_webshop_action(raw, require_think=True)
        used_fallback = not action_valid
        if used_fallback:
            act = fallback_webshop_action(goal)

        obs, reward, done, info = env.step(act)
        info = info or {}
        hist.append(f"obs:{str(obs)[:60]} -> act:{act}")
        steps = st + 1
        task_score = max(task_score, float(info.get("task_score", reward) or 0))
        trace.append(
            {
                "step": st,
                "prompt": prompt,
                "raw_completion": raw,
                "action": act,
                "action_valid": bool(action_valid),
                "used_fallback": used_fallback,
                "parser_failure": not action_valid,
                "generation_seed": step_seed,
                "environment_seed": int(environment_seed),
                "task_score": float(info.get("task_score", reward) or 0),
            }
        )
        if done:
            won = bool(info.get("won", reward == 1.0))
            total_r = 10.0 if won else 0.0
            break

    return EpisodeResult(
        episode=idx,
        goal=goal[:80],
        steps=steps,
        task_score=round(task_score, 3),
        won=won,
        reward=total_r,
        prompt_contract=prompt_contract,
        trace=trace,
    )


def run_rollout(
    env_config: WebshopEnvConfig,
    backend: InferenceBackend,
    n_episodes: int,
    out_path: str,
    evolution_text: str | None = None,
    max_steps: int = MAX_STEPS,
    on_episode: Callable[[EpisodeResult], None] | None = None,
) -> dict[str, Any]:
    """Run N episodes, stream partial results to out_path, return summary.

    Writes base_model_scores.json-shaped summary (model/device/n_episodes/
    avg_task_score/win_rate/elapsed_s/episodes).
    """
    env = WebshopEnv(env_config)
    results: list[EpisodeResult] = []
    t_start = time.time()

    def flush() -> None:
        _write_summary(out_path, backend, results, t_start)

    for ep in range(n_episodes):
        try:
            res = run_episode(env, backend, ep, evolution_text, max_steps=max_steps)
        except Exception:  # noqa: BLE001 — record and continue to next episode
            res = EpisodeResult(
                episode=ep, goal="", steps=0, task_score=0.0, won=False,
                reward=0.0, error=True,
            )
        results.append(res)
        if on_episode:
            on_episode(res)
        flush()

    env.close()
    return _write_summary(out_path, backend, results, t_start)


def _write_summary(
    out_path: str,
    backend: InferenceBackend,
    results: list[EpisodeResult],
    t_start: float,
) -> dict[str, Any]:
    n_ok = [r for r in results if not r.error]
    summary = {
        "model": backend.model_name,
        "device": getattr(backend, "_device", "unknown"),
        "n_episodes": len(results),
        "avg_task_score": round(
            sum(r.task_score for r in n_ok) / max(1, len(n_ok)), 3
        ),
        "win_rate": round(sum(1 for r in n_ok if r.won) / max(1, len(n_ok)), 3),
        "elapsed_s": round(time.time() - t_start, 1),
        "episodes": [
            {
                "episode": r.episode,
                "goal": r.goal,
                "steps": r.steps,
                "task_score": r.task_score,
                "won": r.won,
                "reward": r.reward,
                "error": r.error,
            }
            for r in results
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    return summary
