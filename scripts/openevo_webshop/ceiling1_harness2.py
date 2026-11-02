#!/usr/bin/env python3
"""Pure Harness-2.0 contracts; no GPU launch and no formal-task authority."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from scripts.openevo_webshop.webshop_comparison_contract import (
    format_webshop_actions,
    is_semantic_webshop_action,
    parse_webshop_action,
)

DEFAULT_UI_LITERALS = ("Next >", "Back to Search", "Buy Now", "< Prev")
EXECUTABLE_ACTION_RE = re.compile(r"(?i)\b(?:click|select|buy)\[[^\]\n]+\]")
PRODUCT_ID_RE = re.compile(r"(?i)\bB0[A-Z0-9]{8}\b")
ACTION_LIKE_ABSTRACTION_RE = re.compile(r"(?i)\b(?:pagination|paginate|return[- ]to[- ]search|next_page|next page|previous_page|reverse[- ]pagination)\b")


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_<>-]+", str(text).lower())


def bigram_jaccard(a: str, b: str) -> float:
    def grams(text: str) -> set[tuple[str, str]]:
        w = _words(text)
        return set(zip(w, w[1:]))
    x, y = grams(a), grams(b)
    if not x and not y:
        return 0.0
    return len(x & y) / max(1, len(x | y))


def lint_carriers(
    *,
    memory: str,
    skill: str,
    agent: str,
    forbidden_literals: Iterable[str] = (),
    ui_literals: Iterable[str] = DEFAULT_UI_LITERALS,
    max_agent_skill_overlap: float = 0.55,
    forbid_action_like_abstractions: bool = False,
) -> dict[str, Any]:
    violations: list[str] = []
    texts = {"memory": memory, "skill": skill, "agent": agent}
    for name, text in texts.items():
        if EXECUTABLE_ACTION_RE.search(text):
            violations.append(f"{name}: executable historical UI command")
        if PRODUCT_ID_RE.search(text):
            violations.append(f"{name}: product-id literal")
        if forbid_action_like_abstractions:
            action_like = sorted({m.group(0).lower() for m in ACTION_LIKE_ABSTRACTION_RE.finditer(text)})
            if action_like:
                violations.append(f"{name}: action-like abstraction {action_like}")
        lower = text.lower()
        for literal in ui_literals:
            if str(literal).lower() in lower:
                violations.append(f"{name}: UI literal {literal!r}")
        for literal in forbidden_literals:
            value = str(literal).strip()
            if value and value.lower() in lower:
                violations.append(f"{name}: task literal {value!r}")
    overlap = bigram_jaccard(skill, agent)
    if overlap > float(max_agent_skill_overlap):
        violations.append(f"agent/skill overlap {overlap:.3f} exceeds {max_agent_skill_overlap:.3f}")
    return {"status": "PASS" if not violations else "FAIL", "violations": violations, "agent_skill_bigram_jaccard": overlap}


@dataclass(frozen=True)
class ContextCaps:
    total_tokens: int
    agent_tokens: int = 192
    skill_tokens: int = 256
    memory_tokens: int = 384
    carrier_fraction_max: float = 0.35


def govern_context(
    *,
    environment_text: str,
    recent_history: str,
    agent: str,
    skill: str,
    memory: str,
    token_counter: Callable[[str], int],
    caps: ContextCaps,
) -> dict[str, Any]:
    """Select whole carrier sections under a fixed total budget; never silently truncate."""
    env_n = int(token_counter(environment_text))
    hist_n = int(token_counter(recent_history)) if recent_history else 0
    mandatory = env_n + hist_n
    if mandatory > caps.total_tokens:
        raise ValueError("environment truth + recent history exceed total context budget")
    selected: dict[str, str] = {}
    counts = {"environment": env_n, "recent_history": hist_n, "agent_system": 0, "skill_bundle": 0, "text_memory": 0}
    requested_counts = {"agent_system": 0, "skill_bundle": 0, "text_memory": 0}
    decisions: dict[str, dict[str, Any]] = {}
    remaining = caps.total_tokens - mandatory
    carrier_cap = int(caps.total_tokens * float(caps.carrier_fraction_max))
    carrier_used = 0
    for name, text, per_cap in (
        ("agent_system", agent, caps.agent_tokens),
        ("skill_bundle", skill, caps.skill_tokens),
        ("text_memory", memory, caps.memory_tokens),
    ):
        if not text:
            decisions[name] = {"requested": False, "selected": False, "reason": "not_requested", "requested_tokens": 0, "selected_tokens": 0}
            continue
        n = int(token_counter(text))
        requested_counts[name] = n
        if n > int(per_cap):
            reason = "per_carrier_token_cap_exceeded"
        elif n > remaining:
            reason = "remaining_context_budget_exceeded"
        elif carrier_used + n > carrier_cap:
            reason = "carrier_fraction_cap_exceeded"
        else:
            selected[name] = text
            counts[name] = n
            remaining -= n
            carrier_used += n
            reason = "selected"
        decisions[name] = {
            "requested": True,
            "selected": name in selected,
            "reason": reason,
            "requested_tokens": n,
            "selected_tokens": n if name in selected else 0,
        }
    return {
        "status": "PASS",
        "selected": selected,
        "token_counts": counts,
        "requested_token_counts": requested_counts,
        "decisions": decisions,
        "total_tokens": mandatory + carrier_used,
        "carrier_tokens": carrier_used,
        "carrier_fraction": carrier_used / max(1, mandatory + carrier_used),
        "dropped": [x for x in ("agent_system", "skill_bundle", "text_memory") if x not in selected and {"agent_system":agent,"skill_bundle":skill,"text_memory":memory}[x]],
    }


def validate_preformal_resume_row(prior: dict[str, Any], task: dict[str, Any], mode: str, *, seed_base: int = 148_000_000, harness_profile: str | None = None) -> None:
    pos = int(task["position"])
    expected = {
        "schema": "openevo.ceiling1-harness2-preformal-row.v1",
        "qualification_only": True,
        "formal_task_consumption": 0,
        "final_panel_access": 0,
        "mode": mode,
        "schedule_position": pos,
        "task_id": task["task_id"],
        "worker_seed": int(task["worker_seed"]),
        "source_session_index": int(task["source_session_index"]),
        "instruction_sha256": task["instruction_sha256"],
        "rollout_seed": int(seed_base) + pos,
    }
    if harness_profile is not None:
        expected["harness_profile"] = harness_profile
    mismatches = [key for key, value in expected.items() if prior.get(key) != value]
    if mismatches:
        raise ValueError("preformal resume identity mismatch: " + ",".join(mismatches))
    if prior.get("engineering_invalid") or prior.get("error"):
        raise ValueError("preformal resume refuses engineering-invalid prior row")


def state_admissible(action: str, available_actions: dict[str, Any]) -> bool:
    if action.startswith("search["):
        return bool(available_actions.get("has_search_bar"))
    return action in format_webshop_actions(available_actions)


def classify_action_step(
    *,
    raw_completion: str,
    prepared_completion: str,
    available_actions: dict[str, Any],
    repair_used: bool,
    environment_accepted: bool | None = None,
) -> dict[str, Any]:
    parsed, format_valid = parse_webshop_action(prepared_completion, require_think=True)
    semantic = bool(format_valid and is_semantic_webshop_action(parsed))
    admissible = bool(semantic and state_admissible(parsed, available_actions))
    return {
        "format_valid": bool(format_valid),
        "semantic_command_valid": semantic,
        "state_admissible": admissible,
        "environment_accepted": environment_accepted,
        "repair_used": bool(repair_used),
        "parsed_action": parsed if semantic else None,
        "raw_completion_present": bool(str(raw_completion).strip()),
    }


def qualify_harness(
    *,
    baseline: dict[str, float],
    candidate: dict[str, float],
    lint_status: str,
    overlap_pass: bool,
    context_budget_pass: bool,
    runtime_readback_pass: bool,
    max_format_invalid_delta: float = 0.05,
    max_state_inadmissible_delta: float = 0.10,
) -> dict[str, Any]:
    """Interface-health gate. Success/reward is deliberately absent from selection logic."""
    reasons: list[str] = []
    if lint_status != "PASS": reasons.append("carrier lint failed")
    if not overlap_pass: reasons.append("carrier overlap failed")
    if not context_budget_pass: reasons.append("context budget failed")
    if not runtime_readback_pass: reasons.append("runtime readback failed")
    if candidate["format_invalid_rate"] > baseline["format_invalid_rate"] + max_format_invalid_delta:
        reasons.append("format-invalid regression")
    if candidate["state_inadmissible_rate"] > baseline["state_inadmissible_rate"] + max_state_inadmissible_delta:
        reasons.append("state-inadmissible regression")
    return {"status": "PASS" if not reasons else "FAIL", "reasons": reasons}


def interface_circuit_breaker(
    *,
    baseline_state_inadmissible_rate: float,
    sealed_round_rates: list[float],
    window_rounds: int = 3,
    absolute_delta_max: float = 0.10,
) -> dict[str, Any]:
    window = sealed_round_rates[-int(window_rounds):]
    trip = len(window) == int(window_rounds) and all(
        float(x) > float(baseline_state_inadmissible_rate) + float(absolute_delta_max) for x in window
    )
    return {
        "status": "TRIP" if trip else "PASS",
        "action": "pause-before-next-round-and-diagnose-only" if trip else "continue",
        "window": window,
        "automatic_scientific_change": False,
    }
