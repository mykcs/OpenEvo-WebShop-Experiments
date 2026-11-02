"""Unit tests for openevo_webshop.prompts (pure functions, no env required)."""

from __future__ import annotations

from openevo_webshop.prompts import (
    WEBSHOP_TEMPLATE,
    build_prompt,
    extract_goal,
    fallback_action,
    fmt_actions,
    parse_action,
)


def test_fmt_actions_search_and_clickables():
    av = {"has_search_bar": True, "clickables": ["buy now", "details"]}
    out = fmt_actions(av)
    assert "search[<your query>]" in out
    assert "click[buy now]" in out
    assert "click[details]" in out


def test_fmt_actions_no_search_bar():
    av = {"has_search_bar": False, "clickables": ["next >"]}
    out = fmt_actions(av)
    assert "search[" not in out
    assert "click[next >]" in out


def test_parse_action_basic():
    assert parse_action("blah <action>search[shoes]</action> blah") == "search[shoes]"


def test_parse_action_case_insensitive():
    assert parse_action("<ACTION>Click[Buy]</ACTION>") == "click[buy]"


def test_parse_action_missing_returns_none():
    assert parse_action("no tags here") is None
    assert parse_action("<action>only open") is None


def test_extract_goal_found():
    obs = "Instruction:  [SEP]  find a red shoe  [SEP] rest"
    assert extract_goal(obs) == "find a red shoe"


def test_extract_goal_missing():
    assert extract_goal("no instruction") == ""


def test_fallback_action_uses_first_word():
    assert fallback_action("Red Shoe") == "search[red]"
    assert fallback_action("") == "search[item]"


def test_build_prompt_renders_slots():
    p = build_prompt(
        task_description="buy a shoe",
        step_count=2,
        history_length=2,
        action_history="click[x]",
        current_observation="OBS-" + "y" * 1000,
        available_actions={"has_search_bar": True, "clickables": []},
        obs_truncate=10,
    )
    assert "buy a shoe" in p
    assert "2 step(s)" in p
    assert "OBS-yyyyyy" in p  # truncated to 10 chars
    assert "search[<your query>]" in p


def test_webshop_template_has_think_and_action_instruction():
    assert "<think>" in WEBSHOP_TEMPLATE
    assert "<action>" in WEBSHOP_TEMPLATE
