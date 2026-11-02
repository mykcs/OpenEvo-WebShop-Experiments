"""Compatibility facade for the repository's single WebShop contract."""

from __future__ import annotations

from openevo_webshop.comparison_contract import (
    ACTION_CONTRACT,
    COMPACT_PROMPT_CONTRACT,
    PRODUCT_GROUNDED_PROMPT_CONTRACT,
    PROMPT_CONTRACT,
    STATE_AWARE_PROMPT_CONTRACT,
    WEBSHOP_TEMPLATE,
    extract_goal,
    fallback_webshop_action,
    format_webshop_actions,
    format_product_candidates,
    parse_webshop_action,
    render_compact_webshop_prompt,
    render_product_grounded_webshop_prompt,
    render_state_aware_webshop_prompt,
    render_webshop_prompt,
)


def fmt_actions(available_actions: dict) -> str:
    return "\n".join(format_webshop_actions(available_actions))


def parse_action(text: str) -> str | None:
    lowered = str(text).lower()
    if "<action>" not in lowered or "</action>" not in lowered:
        return None
    action, _ = parse_webshop_action(text, require_think=False)
    return action


def fallback_action(goal: str) -> str:
    return fallback_webshop_action(goal)


def build_prompt(
    task_description: str,
    step_count: int,
    history_length: int,
    action_history: str,
    current_observation: object,
    available_actions: dict,
    obs_truncate: int = 600,
) -> str:
    return render_webshop_prompt(
        task_description=task_description,
        step_count=step_count,
        history_length=history_length,
        action_history=action_history,
        current_observation=str(current_observation)[:obs_truncate],
        available_actions=available_actions,
    )


__all__ = [
    "ACTION_CONTRACT",
    "COMPACT_PROMPT_CONTRACT",
    "PRODUCT_GROUNDED_PROMPT_CONTRACT",
    "STATE_AWARE_PROMPT_CONTRACT",
    "PROMPT_CONTRACT",
    "WEBSHOP_TEMPLATE",
    "build_prompt",
    "extract_goal",
    "fallback_action",
    "fmt_actions",
    "format_product_candidates",
    "parse_action",
    "render_compact_webshop_prompt",
    "render_product_grounded_webshop_prompt",
    "render_state_aware_webshop_prompt",
]
