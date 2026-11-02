"""Pure WebShop prompt and action contract shared by every comparison arm."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any


PROMPT_CONTRACT = "seed3090.webshop-shared-prompt.v1"
COMPACT_PROMPT_CONTRACT = "seed3090.webshop-compact-action-prompt.v2"
STATE_AWARE_PROMPT_CONTRACT = "seed3090.webshop-state-aware-action-prompt.v3"
PRODUCT_GROUNDED_PROMPT_CONTRACT = "seed3090.webshop-product-grounded-action-prompt.v1"
ACTION_CONTRACT = "seed3090.webshop-action-parser.v1"

WEBSHOP_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

WEBSHOP_TEMPLATE = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

COMPACT_WEBSHOP_TEMPLATE = """
You are an autonomous agent operating in the WebShop e-commerce environment.
Task: {task_description}.
Recent history: {action_history}
Current observation: {current_observation}
Allowed actions:
[
{available_actions}
]

Keep reasoning under 30 words. Output exactly two tagged lines: <think>brief reason</think> and <action>one admissible action</action>. Do not list products, repeat the instruction, or write text after </action>.
"""

PRODUCT_GROUNDED_WEBSHOP_TEMPLATE = COMPACT_WEBSHOP_TEMPLATE.replace(
    "Keep reasoning under 30 words.",
    "Deterministic product candidates visible on this page:\n{product_candidates}\n\nKeep reasoning under 30 words.",
)

STATE_AWARE_WEBSHOP_TEMPLATE = COMPACT_WEBSHOP_TEMPLATE.replace(
    "Keep reasoning under 30 words.",
    "State-aware action policy: search once, prefer plausible product links, and avoid repeated identical actions.\n\nKeep reasoning under 30 words.",
)


def format_webshop_actions(available_actions: dict[str, Any]) -> list[str]:
    unknown = set(available_actions) - {"has_search_bar", "clickables"}
    if unknown:
        raise ValueError(f"Unknown keys in WebShop available actions: {sorted(unknown)}")
    actions = []
    if available_actions.get("has_search_bar"):
        actions.append("search[<your query>]")
    actions.extend(
        f"click[{text}]" for text in (available_actions.get("clickables") or [])
    )
    return actions


def format_product_candidates(
    observation: object,
    available_actions: dict[str, Any],
    max_candidates: int = 12,
    max_chars: int = 4_000,
) -> str:
    parts = [part.strip() for part in str(observation).split(" [SEP] ") if part.strip()]
    rows: list[str] = []
    for action in format_webshop_actions(available_actions):
        if not action.startswith("click[") or action in {"click[back to search]", "click[next >]", "click[< prev]"}:
            continue
        key = action[len("click["):-1].strip().lower()
        for index, part in enumerate(parts):
            if part.lower() == key and index + 1 < len(parts):
                title = parts[index + 1]
                price = parts[index + 2] if index + 2 < len(parts) else ""
                suffix = f" | {price}" if price.startswith("$") else ""
                rows.append(f"- {action} -> {title[:180]}{suffix}")
                break
        if len(rows) >= max_candidates:
            break
    return ("\n".join(rows) or "- (no product title/price pair was extracted; use Allowed actions)")[:max_chars]


def format_webshop_observation(observation: object, task_description: str) -> str:
    text = str(observation)
    parts = text.split(" [SEP] ")
    try:
        task_index = parts.index(str(task_description))
    except ValueError:
        return text
    return " [SEP] ".join(f"'{part}'" for part in parts[task_index + 1 :])


def format_webshop_history(
    records: Sequence[tuple[str, str]], history_length: int
) -> tuple[str, int]:
    history_length = max(0, int(history_length))
    if history_length == 0:
        return "", 0
    recent = list(records)[-history_length:]
    start_index = len(records) - len(recent)
    lines = [
        f"[Observation {start_index + offset + 1}: '{observation}', "
        f"Action {start_index + offset + 1}: '{action}']"
        for offset, (observation, action) in enumerate(recent)
    ]
    return "\n".join(lines), len(recent)


def render_webshop_prompt(
    *,
    task_description: str,
    current_observation: object,
    available_actions: dict[str, Any],
    step_count: int = 0,
    history_length: int = 0,
    action_history: str = "",
    max_prompt_chars: int = 20_000,
) -> str:
    action_text = "\n".join(
        f"'{action}'," for action in format_webshop_actions(available_actions)
    )
    step_count = int(step_count)
    history_length = int(history_length)
    values = {
        "task_description": str(task_description),
        "current_observation": str(current_observation),
        "available_actions": action_text,
    }
    if step_count <= 0 or history_length <= 0:
        prompt = WEBSHOP_TEMPLATE_NO_HIS.format(**values)
    else:
        prompt = WEBSHOP_TEMPLATE.format(
            **values,
            step_count=step_count,
            history_length=history_length,
            action_history=str(action_history),
            current_step=step_count + 1,
        )
    if len(prompt) > int(max_prompt_chars):
        prompt = WEBSHOP_TEMPLATE_NO_HIS.format(**values)
    return prompt


def _render_special_prompt(template: str, *, task_description: str, current_observation: object, available_actions: dict[str, Any], action_history: str = "", max_prompt_chars: int = 20_000, **extra: str) -> str:
    action_text = "\n".join(f"'{action}'," for action in format_webshop_actions(available_actions))
    values = dict(task_description=str(task_description), current_observation=str(current_observation), available_actions=action_text, action_history=str(action_history) or "(none)", **extra)
    prompt = template.format(**values)
    return prompt if len(prompt) <= int(max_prompt_chars) else template.format(**values, action_history="(omitted to fit prompt budget)")


def render_compact_webshop_prompt(**kwargs: Any) -> str:
    return _render_special_prompt(COMPACT_WEBSHOP_TEMPLATE, **kwargs)


def render_product_grounded_webshop_prompt(**kwargs: Any) -> str:
    kwargs["product_candidates"] = format_product_candidates(kwargs["current_observation"], kwargs["available_actions"])
    return _render_special_prompt(PRODUCT_GROUNDED_WEBSHOP_TEMPLATE, **kwargs)


def render_state_aware_webshop_prompt(**kwargs: Any) -> str:
    return _render_special_prompt(STATE_AWARE_WEBSHOP_TEMPLATE, **kwargs)


def parse_webshop_action(text: str, require_think: bool = True) -> tuple[str, bool]:
    original = str(text)
    lowered = original.lower()
    start_idx = lowered.find("<action>")
    end_idx = lowered.find("</action>")
    valid = start_idx != -1 and end_idx != -1 and end_idx >= start_idx
    action = (
        lowered[start_idx + len("<action>") : end_idx].strip().lower()
        if valid
        else lowered[-20:]
    )
    if require_think and (
        original.find("<think>") == -1 or original.find("</think>") == -1
    ):
        valid = False
    if re.search(r"[\u4e00-\u9fff]", original):
        valid = False
    return action, valid


def extract_goal(observation: object) -> str:
    match = re.search(r"Instruction:\s*\[SEP\]\s*(.+?)\s*\[SEP\]", str(observation))
    return match.group(1).strip() if match else ""


def fallback_webshop_action(task_description: str) -> str:
    first_word = str(task_description).split()[0].lower() if task_description else "item"
    return f"search[{first_word}]"


# ---------------------------------------------------------------------------
# SEED public-code inference contract (byte-exact port, SEED @2cf2fad).
#
# These constants/functions mirror, verbatim, the SEED public evaluation path:
#   agent_system/environments/prompts/webshop.py        (WEBSHOP_TEMPLATE*)
#   agent_system/environments/env_manager.py            (build_text_obs)
#   agent_system/memory/memory.py                       (SimpleMemory.fetch)
#   agent_system/environments/env_package/webshop/projection.py (webshop_projection)
#
# The templates are written with explicit "\n" / trailing-" " escapes (not
# literal triple-quoted trailing whitespace) so the byte content is exact while
# the source stays lint-clean (no W291). Do not "fix" the trailing spaces: they
# are part of SEED's prompt bytes. SEED_WEBSHOP_TEMPLATE_NO_HIS has a trailing
# space after "environment.", "are:", and "<think> </think> tags."; the history
# template has none.
# ---------------------------------------------------------------------------

SEED_EVAL_PROMPT_CONTRACT = "seed.webshop-public-eval-prompt.v1"
SEED_EVAL_ACTION_CONTRACT = "seed.webshop-public-eval-projection.v1"
SEED_EVAL_MAX_PROMPT_CHARS = 20_000

_E = "e‑commerce"  # "e-commerce" with U+2011 non-breaking hyphen, as in SEED

SEED_WEBSHOP_TEMPLATE_NO_HIS = (
    "\n"
    f"You are an expert autonomous agent operating in the WebShop {_E} environment. \n"
    "Your task is to: {task_description}.\n"
    "Your current observation is: {current_observation}.\n"
    "Your admissible actions of the current situation are: \n"
    "[\n"
    "{available_actions}\n"
    "].\n"
    "\n"
    "Now it's your turn to take one action for the current step.\n"
    "You should first reason step-by-step about the current situation, then think"
    " carefully which admissible action best advances the shopping goal. This"
    " reasoning process MUST be enclosed within <think> </think> tags. \n"
    "Once you've finished your reasoning, you should choose an admissible action"
    " for current step and present it within <action> </action> tags.\n"
)

SEED_WEBSHOP_TEMPLATE = (
    "\n"
    f"You are an expert autonomous agent operating in the WebShop {_E} environment.\n"
    "Your task is to: {task_description}.\n"
    "Prior to this step, you have already taken {step_count} step(s). Below are the"
    " most recent {history_length} observations and the corresponding actions you"
    " took: {action_history}\n"
    "You are now at step {current_step} and your current observation is:"
    " {current_observation}.\n"
    "Your admissible actions of the current situation are:\n"
    "[\n"
    "{available_actions}\n"
    "].\n"
    "\n"
    "Now it's your turn to take one action for the current step.\n"
    "You should first reason step-by-step about the current situation, then think"
    " carefully which admissible action best advances the shopping goal. This"
    " reasoning process MUST be enclosed within <think> </think> tags.\n"
    "Once you've finished your reasoning, you should choose an admissible action"
    " for current step and present it within <action> </action> tags.\n"
)


def render_seed_eval_prompt(
    *,
    task_description: str,
    current_observation: object,
    available_actions: dict[str, Any],
    history_records: Sequence[tuple[str, str]] = (),
    history_length: int,
    max_prompt_chars: int = SEED_EVAL_MAX_PROMPT_CHARS,
) -> str:
    """Build one SEED-strict prompt. Mirrors env_manager.build_text_obs.

    ``history_records`` are (pre_step_reformatted_obs, projected_action) tuples in
    chronological order, exactly what SEED stores via SimpleMemory. ``init`` is
    True iff there are no records yet; then (or when ``history_length <= 0``) the
    NO_HIS template is used. Otherwise the history template is filled with
    ``history_length = valid_len`` (the actual number of records included, which
    SEED takes from SimpleMemory.fetch) and ``step_count = len(records)``,
    ``current_step = len(records) + 1``. Any prompt longer than
    ``max_prompt_chars`` falls back to the NO_HIS template.
    """
    action_text = "\n".join(
        f"'{action}'," for action in format_webshop_actions(available_actions)
    )
    values = {
        "task_description": str(task_description),
        "current_observation": str(current_observation),
        "available_actions": action_text,
    }
    records = list(history_records)
    init = len(records) == 0
    history_length = int(history_length)
    if init or history_length <= 0:
        prompt = SEED_WEBSHOP_TEMPLATE_NO_HIS.format(**values)
    else:
        action_history, valid_len = format_webshop_history(records, history_length)
        prompt = SEED_WEBSHOP_TEMPLATE.format(
            **values,
            step_count=len(records),
            history_length=valid_len,
            action_history=action_history,
            current_step=len(records) + 1,
        )
    if len(prompt) > int(max_prompt_chars):
        prompt = SEED_WEBSHOP_TEMPLATE_NO_HIS.format(**values)
    return prompt


def seed_extract_task(observation: object) -> str:
    """SEED env_manager.extract_task, verbatim: split on " [SEP] ", require
    parts[1] == "Instruction:", return parts[2]. This is intentionally NOT the
    regex-based extract_goal; SEED uses positional split."""
    parts = str(observation).split(" [SEP] ")
    if len(parts) < 3 or parts[1] != "Instruction:":
        raise ValueError("SEED extract_task: observation does not match expected layout")
    return parts[2]


def seed_project_action(text: str, require_think: bool = True) -> tuple[str, bool]:
    """SEED webshop_projection, verbatim, for a single completion.

    Mirrors projection.webshop_projection: lowercase the whole completion; if the
    ``<action>``/``</action>`` tags are not both present the action is the last 20
    chars of the lowered completion and valid=0; otherwise the action is the
    extracted tag content (stripped, lowercased) and valid=1. When the tags are
    present, ``require_think`` then requires both ``<think>``/``</think>`` in the
    ORIGINAL completion, and any CJK character in the original forces valid=0.
    There is deliberately NO end>=start guard (SEED has none) and NO fallback
    substitution — the projected action is returned for pass-through to the env.
    """
    original = str(text)
    lowered = original.lower()
    start_idx = lowered.find("<action>")
    end_idx = lowered.find("</action>")
    valid = 0
    if start_idx == -1 or end_idx == -1:
        action = lowered[-20:]
    else:
        action = lowered[start_idx + len("<action>") : end_idx].strip().lower()
        valid = 1
        if require_think and (
            original.find("<think>") == -1 or original.find("</think>") == -1
        ):
            valid = 0
        if re.search(r"[\u4e00-\u9fff]", original):
            valid = 0
    return action, bool(valid)
