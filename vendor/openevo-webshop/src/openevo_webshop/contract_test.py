"""Dependency-free smoke checks shipped with the WebShop adapter."""

from __future__ import annotations

from openevo_webshop.comparison_contract import (
    extract_goal,
    fallback_webshop_action,
    format_webshop_actions,
    parse_webshop_action,
    render_webshop_prompt,
)


def run_contract_checks() -> int:
    actions = {"has_search_bar": True, "clickables": ["item - blue shoe"]}
    formatted = format_webshop_actions(actions)
    if formatted != ["search[<your query>]", "click[item - blue shoe]"]:
        raise RuntimeError("available-action formatting contract failed")

    prompt = render_webshop_prompt(
        task_description="buy a blue shoe",
        current_observation="search page",
        available_actions=actions,
    )
    if "<think>" not in prompt or "<action>" not in prompt:
        raise RuntimeError("prompt tag contract failed")

    action, valid = parse_webshop_action(
        "<think>search first</think><action>search[blue shoe]</action>"
    )
    if not valid or action != "search[blue shoe]":
        raise RuntimeError("action parser contract failed")

    goal = extract_goal("Instruction: [SEP] buy a blue shoe [SEP] observation")
    if goal != "buy a blue shoe":
        raise RuntimeError("goal extraction contract failed")

    if fallback_webshop_action(goal) != "search[buy]":
        raise RuntimeError("fallback action contract failed")

    return 5
