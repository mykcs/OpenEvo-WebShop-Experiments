"""Read-only replay of historical round receipts through shared runtime primitives.

This adapter reuses the #524-derived round transaction planner. It never launches,
mutates, retries, or promotes historical evidence into current execution authority.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from scripts.openevo_runtime.round_transaction import build_round_recovery_plan

REPLAY_SCHEMA = "openevo.execution-runtime.historical-receipt-replay.202609171515"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decision_class(plan: Mapping[str, Any]) -> str:
    verdict = plan["verdict"]
    recovery = plan["recovery_class"]
    if verdict == "COMPLETE_NO_REPLAY":
        return "SEALED"
    if recovery in {"ENGINEERING_RETRYABLE", "RESUME_REQUIRED"}:
        return "EXACT_RESUME"
    return "BLOCKED_INTEGRITY"


def replay_historical_round(
    round_root: Path,
    round_index: int,
    *,
    rollouts_per_round: int = 128,
) -> dict[str, Any]:
    """Replay one historical round without changing any source bytes."""
    root = round_root.expanduser().resolve()
    plan = build_round_recovery_plan(root, round_index, rollouts_per_round=rollouts_per_round)
    witnesses: list[dict[str, Any]] = []
    for row in plan["required_readbacks_before_resume"]:
        path = Path(row["path"])
        if path.is_file():
            observed = _sha(path)
            if row.get("sha256") is not None and observed != row["sha256"]:
                raise ValueError(f"historical receipt witness drift: {path}")
            witnesses.append({
                "phase": row["phase"],
                "path": str(path),
                "sha256": observed,
            })
    return {
        "schema": REPLAY_SCHEMA,
        "round_root": str(root),
        "round_index": round_index,
        "decision_class": _decision_class(plan),
        "reference_verdict": plan["verdict"],
        "reference_recovery_class": plan["recovery_class"],
        "target_phase": plan["target_phase"],
        "must_not_replay": list(plan["must_not_replay"]),
        "receipt_witnesses": witnesses,
        "integrity_errors": list(plan["integrity_errors"]),
        "read_only": True,
        "mutation": False,
        "execution_action": "NONE",
        "scientific_retry_authorized": False,
        "launch_authority": False,
        "scientific_authority_granted": False,
    }
