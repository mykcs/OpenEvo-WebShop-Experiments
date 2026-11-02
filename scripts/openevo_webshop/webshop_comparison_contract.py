#!/usr/bin/env python3
"""Repository facade for the installable WebShop comparison contract."""

from __future__ import annotations

import sys
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any


PROMPT_CONTRACT = "seed3090.webshop-shared-prompt.v1"
COMPACT_PROMPT_CONTRACT = "seed3090.webshop-compact-action-prompt.v2"
STATE_AWARE_PROMPT_CONTRACT = "seed3090.webshop-state-aware-action-prompt.v3"

PRODUCT_GROUNDED_PROMPT_CONTRACT = "seed3090.webshop-product-grounded-action-prompt.v1"
ACTION_CONTRACT = "seed3090.webshop-action-parser.v1"

VENDOR_SOURCE = Path(__file__).resolve().parents[2] / "vendor/openevo-webshop/src"
if str(VENDOR_SOURCE) not in sys.path:
    sys.path.insert(0, str(VENDOR_SOURCE))

from openevo_webshop.comparison_contract import (  # noqa: E402
    ACTION_CONTRACT,
    PROMPT_CONTRACT,
    SEED_EVAL_ACTION_CONTRACT,
    SEED_EVAL_MAX_PROMPT_CHARS,
    SEED_EVAL_PROMPT_CONTRACT,
    SEED_WEBSHOP_TEMPLATE,
    SEED_WEBSHOP_TEMPLATE_NO_HIS,
    WEBSHOP_TEMPLATE,
    WEBSHOP_TEMPLATE_NO_HIS,
    extract_goal,
    fallback_webshop_action,
    format_webshop_actions,
    format_webshop_history,
    format_webshop_observation,
    parse_webshop_action,
    render_seed_eval_prompt,
    render_webshop_prompt,
    seed_extract_task,
    seed_project_action,
)

TOLERANT_ACTION_CONTRACT = "seed3090.webshop-action-parser.v2-missing-close"
ACTION_COMMAND_CONTRACT = "seed3090.webshop-action-parser.v3-action-command"

_ACTION_COMMAND_RE = re.compile(r"(?:search|click|select|buy)\[[^\]\n]*\]")
_SEMANTIC_ACTION_RE = re.compile(r"(?P<verb>search|click|select|buy)\[(?P<arg>[^\]\n]+)\]")
_NESTED_ACTION_PREFIX_RE = re.compile(r"(?:search|click|select|buy)\[", re.IGNORECASE)
_CJK_RE = re.compile(r"[一-鿿]")


def is_semantic_webshop_action(action: str) -> bool:
    """Validate v3 command syntax while preserving legal WebShop ``<`` tokens."""

    match = _SEMANTIC_ACTION_RE.fullmatch(str(action).strip().lower())
    if not match:
        return False
    if match.group("verb") != "search" and _NESTED_ACTION_PREFIX_RE.search(match.group("arg")):
        return False
    return True


def parse_action_command_contract(text: str, require_think: bool = True) -> tuple[str, bool]:
    """Apply the frozen v3 wrapper parse plus semantic action-command guard."""

    action, valid = parse_webshop_action(text, require_think=require_think)
    return action, bool(valid and is_semantic_webshop_action(action))


def repair_missing_action_close(text: str) -> tuple[str, str, bool]:
    """Repair exactly one bounded missing ``</action>`` tag."""

    original = str(text)
    strict_action, strict_valid = parse_action_command_contract(original, require_think=True)
    if strict_valid:
        return original, strict_action, False
    lowered = original.lower()
    start_idx = lowered.find("<action>")
    if start_idx < 0 or lowered.find("</action>", start_idx + len("<action>")) >= 0:
        return original, strict_action, False
    action_text = lowered[start_idx + len("<action>") :].strip()
    if not re.fullmatch(r"(?:search|click|select|buy)\[[^\]\n<]+\]", action_text):
        return original, strict_action, False
    repaired = original + "</action>"
    action, valid = parse_webshop_action(repaired, require_think=True)
    if not valid:
        return original, strict_action, False
    return repaired, action, True


def repair_action_command(text: str) -> tuple[str, str, bool]:
    """Recover a well-formed action command emitted without a strict wrapper.

    The 7B base reliably emits a valid ``search[...]``/``click[...]`` command but
    wraps it inconsistently (``[action]cmd``, ``=search[...]``, ``<click[...]>``)
    instead of ``<action>cmd</action>``. Strict-first: if the strict parse already
    succeeds, return the text untouched (no repair). Otherwise scan only the region
    after ``</think>`` for the first well-formed command. Recovery is legitimate
    only when the think contract holds and no CJK char is present (otherwise the
    normalized text would still fail the strict re-parse), in which case the
    completion is rebuilt as ``<think>...</think>\\n<action>cmd</action>`` so the
    strict re-parse in ``run_episode`` succeeds and the step is no longer a
    fallback. Anything else is returned untouched so the caller can retry.
    """

    original = str(text)
    strict_action, strict_valid = parse_action_command_contract(original, require_think=True)
    if strict_valid:
        return original, strict_action, False
    think_open = original.find("<think>")
    think_close = original.find("</think>")
    if think_open == -1 or think_close == -1:
        return original, strict_action, False
    if _CJK_RE.search(original):
        return original, strict_action, False
    region = original[think_close + len("</think>") :]
    match = _ACTION_COMMAND_RE.search(region)
    if not match:
        return original, strict_action, False
    command = match.group(0).strip().lower()
    if not is_semantic_webshop_action(command):
        return original, strict_action, False
    think_block = original[think_open : think_close + len("</think>")]
    repaired = f"{think_block}\n<action>{command}</action>"
    action, valid = parse_action_command_contract(repaired, require_think=True)
    if not valid or action != command:
        return original, strict_action, False
    return repaired, action, True

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


PRODUCT_GROUNDED_WEBSHOP_TEMPLATE = """
You are an autonomous agent operating in the WebShop e-commerce environment.
Task: {task_description}.
Recent history: {action_history}
Current observation: {current_observation}
Allowed actions:
[
{available_actions}
]

Deterministic product candidates visible on this page:
{product_candidates}

Use the candidate title and price to match the task constraints. Choose exactly one allowed action.
Keep reasoning under 30 words. Output exactly two tagged lines: <think>brief reason</think> and <action>one admissible action</action>. Do not invent product attributes or write text after </action>.
"""

STATE_AWARE_WEBSHOP_TEMPLATE = """
You are an autonomous agent operating in the WebShop e-commerce environment.
Task: {task_description}.
Recent history: {action_history}
Current observation: {current_observation}
Allowed actions:
[
{available_actions}
]

State-aware action policy:
- If Search is admissible, search once using the task attributes.
- On a results page, prefer a plausible product link before clicking Next >.
- Click Next > only when no plausible product is shown, and never more than twice consecutively.
- On a product page, select required options when admissible, then click buy now.
- Do not repeat an identical action unless no other admissible action exists.

Keep reasoning under 30 words. Output exactly two tagged lines: <think>brief reason</think> and <action>one admissible action</action>. Do not list products, repeat the instruction, or write text after </action>.
"""



def format_product_candidates(
    observation: object,
    available_actions: dict[str, Any],
    max_candidates: int = 12,
    max_chars: int = 4_000,
) -> str:
    """Attach only visible title/price text to exact clickable product IDs."""
    parts = [part.strip() for part in str(observation).split(" [SEP] ") if part.strip()]
    product_actions = [
        action
        for action in format_webshop_actions(available_actions)
        if action.startswith("click[")
        and action not in {"click[back to search]", "click[next >]", "click[< prev]"}
    ]
    rows = []
    for action in product_actions:
        key = action[len("click["):-1].strip().lower()
        for index, part in enumerate(parts):
            if part.lower() != key:
                continue
            title = parts[index + 1] if index + 1 < len(parts) else ""
            price = parts[index + 2] if index + 2 < len(parts) else ""
            if title and len(title) < 240:
                suffix = f" | {price}" if price.startswith("$") else ""
                rows.append(f"- {action} -> {title[:180]}{suffix}")
            break
        if len(rows) >= max_candidates:
            break
    if not rows:
        return "- (no product title/price pair was extracted; use Allowed actions)"
    text = "\n".join(rows)
    return text[:max_chars]


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


def render_compact_webshop_prompt(
    *,
    task_description: str,
    current_observation: object,
    available_actions: dict[str, Any],
    step_count: int = 0,
    history_length: int = 0,
    action_history: str = "",
    max_prompt_chars: int = 20_000,
) -> str:
    """Render the frozen H1.4 compact-output prompt contract."""
    action_text = "\n".join(
        f"'{action}'," for action in format_webshop_actions(available_actions)
    )
    prompt = COMPACT_WEBSHOP_TEMPLATE.format(
        task_description=str(task_description),
        current_observation=str(current_observation),
        available_actions=action_text,
        action_history=str(action_history) if history_length and step_count else "(none)",
    )
    if len(prompt) > int(max_prompt_chars):
        prompt = COMPACT_WEBSHOP_TEMPLATE.format(
            task_description=str(task_description),
            current_observation=str(current_observation),
            available_actions=action_text,
            action_history="(omitted to fit prompt budget)",
        )
    return prompt



def render_product_grounded_webshop_prompt(
    *,
    task_description: str,
    current_observation: object,
    available_actions: dict[str, Any],
    step_count: int = 0,
    history_length: int = 0,
    action_history: str = "",
    max_prompt_chars: int = 20_000,
) -> str:
    """Render a deterministic product-title/price grounding interface."""
    action_text = "\n".join(
        f"'{action}'," for action in format_webshop_actions(available_actions)
    )
    prompt = PRODUCT_GROUNDED_WEBSHOP_TEMPLATE.format(
        task_description=str(task_description),
        current_observation=str(current_observation),
        available_actions=action_text,
        product_candidates=format_product_candidates(
            current_observation, available_actions
        ),
        action_history=str(action_history) if history_length and step_count else "(none)",
    )
    if len(prompt) > int(max_prompt_chars):
        prompt = PRODUCT_GROUNDED_WEBSHOP_TEMPLATE.format(
            task_description=str(task_description),
            current_observation=str(current_observation),
            available_actions=action_text,
            product_candidates=format_product_candidates(
                current_observation, available_actions, max_candidates=6, max_chars=2_000
            ),
            action_history="(omitted to fit prompt budget)",
        )
    return prompt


def render_state_aware_webshop_prompt(
    *,
    task_description: str,
    current_observation: object,
    available_actions: dict[str, Any],
    step_count: int = 0,
    history_length: int = 0,
    action_history: str = "",
    max_prompt_chars: int = 20_000,
) -> str:
    """Render H1.8's state/action-selection intervention prompt."""
    action_text = "\n".join(
        f"'{action}'," for action in format_webshop_actions(available_actions)
    )
    prompt = STATE_AWARE_WEBSHOP_TEMPLATE.format(
        task_description=str(task_description),
        current_observation=str(current_observation),
        available_actions=action_text,
        action_history=str(action_history) if history_length and step_count else "(none)",
    )
    if len(prompt) > int(max_prompt_chars):
        prompt = STATE_AWARE_WEBSHOP_TEMPLATE.format(
            task_description=str(task_description),
            current_observation=str(current_observation),
            available_actions=action_text,
            action_history="(omitted to fit prompt budget)",
        )
    return prompt


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

__all__ = [
    "ACTION_CONTRACT",
    "PROMPT_CONTRACT",
    "WEBSHOP_TEMPLATE",
    "WEBSHOP_TEMPLATE_NO_HIS",
    "extract_goal",
    "fallback_webshop_action",
    "format_webshop_actions",
    "format_webshop_history",
    "format_webshop_observation",
    "parse_webshop_action",
    "parse_action_command_contract",
    "is_semantic_webshop_action",
    "render_webshop_prompt",
    "TOLERANT_ACTION_CONTRACT",
    "ACTION_COMMAND_CONTRACT",
    "repair_missing_action_close",
    "repair_action_command",
    "SEED_EVAL_PROMPT_CONTRACT",
    "SEED_EVAL_ACTION_CONTRACT",
    "SEED_EVAL_MAX_PROMPT_CHARS",
    "SEED_WEBSHOP_TEMPLATE",
    "SEED_WEBSHOP_TEMPLATE_NO_HIS",
    "render_seed_eval_prompt",
    "seed_extract_task",
    "seed_project_action",
]
