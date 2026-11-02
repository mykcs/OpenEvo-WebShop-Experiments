#!/usr/bin/env python3
"""Fail-closed adapter from retained Q06 attempts JSONL to pre-action rows.

The current sealed attempts schema retains outcomes and entropy diagnostics, but
does not retain enough pre-action state to construct the frozen predictor
baseline.  This adapter accepts that schema only when each trace row also has
an explicit, event-ordered ``q06_pre_action`` snapshot.  It never fills missing
features from episode totals or labels.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any


SCHEMA = "openevo.q06.normalized-pre-action-row.v1"
CONTEXT_SCHEMA = "openevo.q06.pre-action-state.v1"
UNCERTAINTY_FEATURES = frozenset({
    "click_entropy_normalized", "reasoning_entropy_mean", "action_entropy_mean",
})
BASELINE_NUMERIC_FEATURES = frozenset({
    "round", "choice_count", "step_index", "prompt_tokens", "history_tokens",
})
HEX = frozenset("0123456789abcdef")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in HEX for c in value)


def _number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _entropy_mean(channel: dict[str, Any], name: str, *, decision_event_index: int) -> float:
    if not isinstance(channel, dict):
        raise ValueError(f"{name} diagnostics must be an object")
    if channel.get("phase") != "pre_action":
        raise ValueError(f"{name} entropy diagnostics must explicitly be pre_action")
    event_index = channel.get("source_event_index")
    if type(event_index) is not int or event_index < 0 or event_index >= decision_event_index:
        raise ValueError(f"{name} entropy provenance event must precede the decision")
    count = channel.get("generated_token_count")
    total = channel.get("entropy_sum_nats")
    if type(count) is not int or count <= 0:
        raise ValueError(f"{name}.generated_token_count must be a positive integer")
    return _number(total, f"{name}.entropy_sum_nats", minimum=0) / count


def _validate_context(context: Any, trace_step: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(context, dict) or context.get("schema") != CONTEXT_SCHEMA:
        raise ValueError("explicit q06_pre_action snapshot is missing or has wrong schema")
    if context.get("phase") != "pre_action":
        raise ValueError("pre-action snapshot phase is not pre_action")
    step = trace_step.get("step")
    if type(step) is not int or step < 0 or context.get("step_index") != step:
        raise ValueError("pre-action snapshot step does not bind the trace decision")
    feature_event = context.get("feature_event_index")
    decision_event = context.get("decision_event_index")
    action_event = context.get("action_event_index")
    if any(type(x) is not int or x < 0 for x in (feature_event, decision_event, action_event)):
        raise ValueError("event-order indices are required for pre-action provenance")
    if not feature_event < decision_event <= action_event:
        raise ValueError("feature provenance is not strictly before the decision/action")

    for key in ("observation_sha256", "prompt_sha256", "history_sha256", "available_actions_sha256"):
        if not _is_sha(context.get(key)):
            raise ValueError(f"{key} must bind the exact pre-action bytes")
    round_no = context.get("completed_rounds")
    if type(round_no) is not int or round_no < 0:
        raise ValueError("completed_rounds must be an explicit nonnegative integer")
    prompt_tokens = context.get("prompt_tokens")
    history_tokens = context.get("history_tokens")
    if type(prompt_tokens) is not int or prompt_tokens < 0:
        raise ValueError("prompt_tokens must be an explicit nonnegative integer")
    if type(history_tokens) is not int or history_tokens < 0 or history_tokens > prompt_tokens:
        raise ValueError("history_tokens must be an explicit count within prompt_tokens")
    state_type = context.get("state_type")
    if not isinstance(state_type, str) or not state_type.strip():
        raise ValueError("state_type must be an explicit pre-action category")
    return {
        "round": round_no,
        "step_index": step,
        "prompt_tokens": prompt_tokens,
        "history_tokens": history_tokens,
        "state_type": state_type,
    }


def _row_from_attempt(
    attempt: dict[str, Any], line_bytes: bytes, *, source_file_sha256: str,
    source_record_number: int, lineage_id: str, checkpoint_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if attempt.get("engineering_invalid") is not False:
        raise ValueError("attempt is not explicitly engineering-valid")
    if attempt.get("panel_role") != "development":
        raise ValueError("attempt must explicitly bind to the development panel")
    task_id = attempt.get("task_identity")
    if attempt.get("lineage_id") != lineage_id or attempt.get("checkpoint_id") != checkpoint_id:
        raise ValueError("attempt does not carry the supplied bound lineage/checkpoint identity")
    episode = attempt.get("episode")
    if not isinstance(task_id, str) or not task_id or not isinstance(episode, dict):
        raise ValueError("actual attempts identity/episode schema is incomplete")
    score = _number(episode.get("score"), "episode.score", minimum=0)
    if score > 1:
        raise ValueError("episode.score must be a normalized Task Score in [0,1]")
    won = episode.get("won")
    if type(won) is not bool:
        raise ValueError("episode.won must be an explicit boolean for the secondary Brier target")
    trace = episode.get("trace")
    if not isinstance(trace, list) or not trace:
        raise ValueError("episode.trace must contain decision records")
    target_step = episode.get("steps")
    if type(target_step) is not int or target_step < 1:
        raise ValueError("episode.steps must bind the post-action target time")

    line_sha = _sha256(line_bytes)
    output: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for trace_row in trace:
        if not isinstance(trace_row, dict):
            raise ValueError("trace rows must be objects")
        step = trace_row.get("step")
        if type(step) is not int or step < 0 or step >= target_step:
            raise ValueError("decision step must precede the episode target observation")
        context = _validate_context(trace_row.get("q06_pre_action"), trace_row)
        diag = trace_row.get("generation_diagnostics")
        if not isinstance(diag, dict):
            raise ValueError("generation_diagnostics missing from decision")
        click = diag.get("click_choice")
        if not isinstance(click, dict):
            raise ValueError("click_choice missing from decision")
        n = click.get("candidate_count")
        if type(n) is not int:
            raise ValueError("click_choice.candidate_count must be an integer")
        if n < 0:
            raise ValueError("click_choice.candidate_count must be nonnegative")
        if n < 2:
            # Zero/one destination states are not eligible for normalized H/log(n).
            excluded.append({"source_record_number": source_record_number, "decision_step": step,
                             "candidate_count": n,
                             "reason": "click entropy undefined with fewer than two legal destinations"})
            continue
        decision_event_index = trace_row["q06_pre_action"]["decision_event_index"]
        if click.get("phase") != "pre_action":
            raise ValueError("click entropy diagnostics must explicitly be pre_action")
        click_event = click.get("source_event_index")
        if type(click_event) is not int or click_event < 0 or click_event >= decision_event_index:
            raise ValueError("click entropy provenance event must precede the decision")
        entropy = _number(click.get("entropy_nats"), "click_choice.entropy_nats", minimum=0)
        if entropy > math.log(n) + 1e-8:
            raise ValueError("click entropy exceeds log(choice_count)")
        features = {
            "round": context["round"],
            "choice_count": n,
            "step_index": step,
            "prompt_tokens": context["prompt_tokens"],
            "history_tokens": context["history_tokens"],
            "click_entropy_normalized": entropy / math.log(n),
            "reasoning_entropy_mean": _entropy_mean(diag.get("reasoning", {}), "reasoning", decision_event_index=decision_event_index),
            "action_entropy_mean": _entropy_mean(diag.get("action", {}), "action", decision_event_index=decision_event_index),
        }
        output.append({
            "schema": SCHEMA,
            "task_id": task_id,
            "lineage_id": lineage_id,
            "checkpoint_id": checkpoint_id,
            "identity_binding": "caller_asserted_against_raw_attempt_fields",
            "completed_rounds": context["round"],
            "decision_step": step,
            "feature_observed_step": step,
            "target_observed_step": target_step,
            "panel_role": "development",
            "feature_phase": "pre_action",
            "target_phase": "post_action_or_later",
            "feature_provenance_sha256": line_sha,
            "source_file_sha256": source_file_sha256,
            "source_record_number": source_record_number,
            "feature_event_index": trace_row["q06_pre_action"]["feature_event_index"],
            "decision_event_index": trace_row["q06_pre_action"]["decision_event_index"],
            "action_event_index": trace_row["q06_pre_action"]["action_event_index"],
            "pre_action_context_sha256": _sha256(json.dumps(
                trace_row["q06_pre_action"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")),
            "state_type": context["state_type"],
            "features": features,
            "target": score,
            "secondary_targets": {"exact_success": int(won)},
        })
    return output, excluded


def adapt_attempts_bytes(
    source_bytes: bytes, *, lineage_id: str, checkpoint_id: str,
) -> dict[str, Any]:
    """Normalize a retained attempts JSONL byte stream without repairing it.

    Rows missing explicit event-ordered pre-action snapshots are counted as
    rejected; eligible rows are never synthesized.  The current sealed Q06
    attempts schema lacks ``q06_pre_action`` and therefore yields a typed
    PARTIAL inventory until a source that actually retains this evidence exists.
    """
    if not source_bytes:
        raise ValueError("attempts source must be nonempty")
    if not lineage_id or not checkpoint_id:
        raise ValueError("lineage_id and checkpoint_id must be supplied from bound identity")
    file_sha = _sha256(source_bytes)
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    excluded_no_destination: list[dict[str, Any]] = []
    physical = source_bytes.split(b"\n")
    if physical and physical[-1] == b"":
        physical.pop()  # A trailing LF terminates the final record, not a blank row.
    for number, line in enumerate(physical, start=1):
        exact_record = line + (b"\n" if number < len(physical) or source_bytes.endswith(b"\n") else b"")
        payload = line[:-1] if line.endswith(b"\r") else line
        if not payload:
            rejected.append({"source_record_number": number, "reason": "blank physical JSONL record"})
            continue
        try:
            attempt = json.loads(payload)
            if not isinstance(attempt, dict):
                raise ValueError("attempt JSON record must be an object")
            adapted, excluded = _row_from_attempt(
                attempt, exact_record, source_file_sha256=file_sha,
                source_record_number=number, lineage_id=lineage_id,
                checkpoint_id=checkpoint_id,
            )
            rows.extend(adapted)
            excluded_no_destination.extend(excluded)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            rejected.append({"source_record_number": number, "reason": str(exc)})
    status = "CPU_ADAPTER_PASS" if rows and not rejected else "PARTIAL"
    return {
        "schema": "openevo.q06.raw-adapter-inventory.v1",
        "status": status,
        "source_file_sha256": file_sha,
        "source_byte_count": len(source_bytes),
        "source_record_count": len(physical),
        "normalized_row_count": len(rows),
        "excluded_no_destination_states": len(excluded_no_destination),
        "excluded_decisions": excluded_no_destination,
        "rows": rows,
        "rejected_records": rejected,
        "status_reason": None if status == "CPU_ADAPTER_PASS" else (
            "no pre-action eligible rows survived; see rejected record reasons"
            if rejected else "no states with at least two legal click destinations"
        ),
        "final_panel_access": 0,
        "model_calls": 0,
        "environment_calls": 0,
        "identity_binding": "caller_asserted_against_raw_attempt_fields; external identity receipt not verified",
        "claim": "source-byte-addressed pre-decision input normalization only; no prediction result",
    }


def adapt_attempts_file(path: str, *, lineage_id: str, checkpoint_id: str) -> dict[str, Any]:
    from pathlib import Path
    return adapt_attempts_bytes(Path(path).read_bytes(), lineage_id=lineage_id, checkpoint_id=checkpoint_id)
