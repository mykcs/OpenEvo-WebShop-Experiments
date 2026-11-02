"""Shared scientific-denominator classification for WebShop attempts."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping


def classify_attempt(row: Mapping[str, Any]) -> dict[str, Any]:
    """Classify runtime/protocol validity without treating agent failure as infra failure.

    Parser failures after at least one model/environment step remain valid agent
    outcomes. Zero-step exceptions, identity drift and incomplete traces do not.
    """

    status = str(row.get("status") or "").strip().lower()
    if status in {"error", "invalid", "invalid-runtime", "runtime-failure"}:
        return {"status": "invalid", "reason": "runtime-status"}

    episode = row.get("episode") if isinstance(row.get("episode"), Mapping) else row
    trace = episode.get("trace")
    if not isinstance(trace, list) or not trace:
        return {"status": "invalid", "reason": "zero-step-or-missing-trace"}
    if row.get("identity_ok") is False:
        return {"status": "invalid", "reason": "task-identity-mismatch"}
    if row.get("trace_complete") is False:
        return {"status": "invalid", "reason": "incomplete-trace"}
    if isinstance(row.get("error"), str) and row.get("error"):
        return {"status": "invalid", "reason": "runtime-exception"}
    return {"status": "valid", "reason": None}


def aggregate_valid_attempts(
    rows: Iterable[Mapping[str, Any]],
    *,
    score_field: str = "final_task_score",
    success_field: str = "won",
    planned_count: int | None = None,
) -> dict[str, Any]:
    materialized = list(rows)
    valid_rows: list[Mapping[str, Any]] = []
    excluded = Counter()
    for row in materialized:
        classification = classify_attempt(row)
        if classification["status"] == "valid":
            valid_rows.append(row)
        else:
            excluded[str(classification["reason"])] += 1

    def value(row: Mapping[str, Any], field: str, default: Any) -> Any:
        episode = row.get("episode") if isinstance(row.get("episode"), Mapping) else {}
        return row.get(field, episode.get(field, default))

    scores = [float(value(row, score_field, 0.0) or 0.0) for row in valid_rows]
    successes = [bool(value(row, success_field, False)) for row in valid_rows]
    return {
        "planned_count": planned_count,
        "attempted_count": len(materialized),
        "valid_count": len(valid_rows),
        "excluded_count": len(materialized) - len(valid_rows),
        "excluded_by_reason": dict(sorted(excluded.items())),
        "mean_score": sum(scores) / len(scores) if scores else None,
        "success_rate": sum(successes) / len(successes) if successes else None,
    }
