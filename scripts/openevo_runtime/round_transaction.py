#!/usr/bin/env python3
"""Read-only exactly-once ledger and recovery planner for one formal Stage2 round.

This module never launches Docker, allocates GPUs, mutates a run root, or grants
resume authority.  It classifies durable phase receipts so an executor can prove
what must *not* be replayed before a separate authority workflow resumes work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "openevo.formal-round-recovery-plan.202609161620"
MAX_JSON_BYTES = 32 * 1024 * 1024
FINAL_ATTEMPT_RE = re.compile(r"^a[0-9]{3}\.json$")
COMPLETED = "COMPLETED"
NOT_STARTED = "NOT_STARTED"
FAILED_NO_SIDE_EFFECT = "FAILED_NO_SIDE_EFFECT"
FAILED_SIDE_EFFECT_UNKNOWN = "FAILED_SIDE_EFFECT_UNKNOWN"
IN_FLIGHT_OR_UNKNOWN = "IN_FLIGHT_OR_UNKNOWN"
INTEGRITY_UNKNOWN = "INTEGRITY_UNKNOWN"


class RoundPlanError(ValueError):
    pass


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    receipt: str
    accepted_statuses: tuple[str, ...] | None = None
    attempt_dirs: tuple[str, ...] = ()


PHASES: tuple[PhaseSpec, ...] = (
    PhaseSpec("state_before", "state-before.json"),
    PhaseSpec("rollout", "ROLLOUT_PHASE_COMPLETE.json", ("PASS",), ("rollout",)),
    PhaseSpec("evidence", "evidence/evidence.json", ("PASS",)),
    PhaseSpec("text_memory", "text-memory/terminal-receipt.json", ("UPDATE", "NOOP"), ("reflector-text_memory",)),
    PhaseSpec("sd_lora", "sd-lora/terminal-receipt.json", ("UPDATE", "NOOP"), ("transition", "training", "sd_lora")),
    PhaseSpec("skill_bundle", "skill-bundle/terminal-receipt.json", ("UPDATE", "NOOP"), ("reflector-skill_bundle",)),
    PhaseSpec("agent_system", "agent-system/terminal-receipt.json", ("UPDATE", "NOOP"), ("reflector-agent_system",)),
    PhaseSpec("state_after", "state-after.json"),
    PhaseSpec("round_complete", "ROUND_COMPLETE.json", ("PASS",)),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RoundPlanError(f"receipt must be a regular non-symlink file: {path}")
    size = path.stat().st_size
    if size > MAX_JSON_BYTES:
        raise RoundPlanError(f"receipt exceeds bounded read limit: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RoundPlanError(f"receipt unreadable: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RoundPlanError(f"receipt must contain a JSON object: {path}")
    return value


def _round_identity_errors(doc: Mapping[str, Any], phase: str, round_index: int, rollouts_per_round: int) -> list[str]:
    errors: list[str] = []
    observed_round = doc.get("round_index")
    if observed_round is not None and observed_round != round_index:
        errors.append(f"{phase}: round_index mismatch expected={round_index} observed={observed_round}")
    if phase == "state_before":
        if not isinstance(doc.get("state_id"), str) or not doc.get("state_id"):
            errors.append("state_before: state_id missing")
    elif phase == "state_after":
        if not isinstance(doc.get("state_id"), str) or not doc.get("state_id"):
            errors.append("state_after: state_id missing")
        if doc.get("next_round") != round_index + 1:
            errors.append("state_after: next_round mismatch")
        expected_rollouts = (round_index + 1) * rollouts_per_round
        if type(doc.get("rollouts_consumed")) is int and doc.get("rollouts_consumed") != expected_rollouts:
            errors.append(
                f"state_after: rollouts_consumed mismatch expected={expected_rollouts} observed={doc.get('rollouts_consumed')}"
            )
    return errors


def _attempts(round_root: Path, spec: PhaseSpec) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    saw_intent = False
    base = round_root / "container-attempts"
    for dirname in spec.attempt_dirs:
        root = base / dirname
        if not root.is_dir():
            continue
        if any(root.glob("a*-intent.json")) or any(root.glob("a*-launcher.json")):
            saw_intent = True
        for path in sorted(root.iterdir()):
            if not FINAL_ATTEMPT_RE.fullmatch(path.name):
                continue
            try:
                doc = load_json(path)
                rows.append({
                    "path": str(path),
                    "sha256": sha256(path),
                    "status": doc.get("status"),
                    "container_exit_code": doc.get("container_exit_code"),
                    "oom_killed": doc.get("oom_killed"),
                    "side_effect_status": (
                        doc.get("side_effect_status")
                        if isinstance(doc.get("side_effect_status"), str)
                        else "NONE" if doc.get("side_effects_started") is False
                        else "STARTED" if doc.get("side_effects_started") is True
                        else "UNKNOWN"
                    ),
                })
            except RoundPlanError as exc:
                rows.append({"path": str(path), "status": INTEGRITY_UNKNOWN, "error": str(exc), "side_effect_status": "UNKNOWN"})
    return rows, saw_intent


def classify_phase(round_root: Path, spec: PhaseSpec, round_index: int, rollouts_per_round: int) -> dict[str, Any]:
    path = round_root / spec.receipt
    if path.exists():
        try:
            doc = load_json(path)
        except RoundPlanError as exc:
            return {"phase": spec.name, "status": INTEGRITY_UNKNOWN, "receipt_path": str(path), "errors": [str(exc)], "attempts": []}
        errors = _round_identity_errors(doc, spec.name, round_index, rollouts_per_round)
        if spec.accepted_statuses is not None and doc.get("status") not in spec.accepted_statuses:
            errors.append(
                f"{spec.name}: terminal status {doc.get('status')!r} not in {list(spec.accepted_statuses)!r}"
            )
        return {
            "phase": spec.name,
            "status": COMPLETED if not errors else INTEGRITY_UNKNOWN,
            "receipt_path": str(path),
            "receipt_sha256": sha256(path),
            "receipt_status": doc.get("status"),
            "errors": errors,
            "attempts": [],
            "document": doc,
        }

    attempts, saw_intent = _attempts(round_root, spec)
    if any(item.get("status") == INTEGRITY_UNKNOWN for item in attempts):
        status = INTEGRITY_UNKNOWN
    elif any(item.get("status") == "PASS" for item in attempts):
        # The container reports success but the canonical phase acknowledgement is absent.
        # A scientific side effect may already exist, so replay is forbidden until readback.
        status = FAILED_SIDE_EFFECT_UNKNOWN
    elif attempts and all(item.get("status") == "ENGINEERING_FAILURE" for item in attempts):
        status = (
            FAILED_NO_SIDE_EFFECT
            if all(item.get("side_effect_status") == "NONE" for item in attempts)
            else FAILED_SIDE_EFFECT_UNKNOWN
        )
    elif attempts or saw_intent:
        status = IN_FLIGHT_OR_UNKNOWN
    else:
        status = NOT_STARTED
    return {
        "phase": spec.name,
        "status": status,
        "receipt_path": str(path),
        "receipt_sha256": None,
        "receipt_status": None,
        "errors": [],
        "attempts": attempts,
        "launch_intent_observed": saw_intent,
    }


def _cross_phase_errors(phases: list[dict[str, Any]], round_index: int) -> list[str]:
    errors: list[str] = []
    by_name = {row["phase"]: row for row in phases}
    before = by_name["state_before"]
    rollout = by_name["rollout"]
    after = by_name["state_after"]
    complete = by_name["round_complete"]
    if before["status"] == COMPLETED and rollout["status"] == COMPLETED:
        sid = before["document"].get("state_id")
        observed = rollout["document"].get("state_id_before")
        if observed is not None and observed != sid:
            errors.append("rollout state_id_before disagrees with state-before")
    if before["status"] == COMPLETED and complete["status"] == COMPLETED:
        sid = before["document"].get("state_id")
        if complete["document"].get("state_id_before") != sid:
            errors.append("ROUND_COMPLETE state_id_before disagrees with state-before")
    if after["status"] == COMPLETED and complete["status"] == COMPLETED:
        sid = after["document"].get("state_id")
        if complete["document"].get("state_id_after") != sid:
            errors.append("ROUND_COMPLETE state_id_after disagrees with state-after")
    completed_gap = False
    saw_noncomplete = False
    for row in phases:
        if row["status"] == COMPLETED and saw_noncomplete:
            completed_gap = True
            break
        if row["status"] != COMPLETED:
            saw_noncomplete = True
    if completed_gap:
        errors.append("phase completion gap: a later phase is complete after an earlier phase is not complete")
    return errors


def build_round_recovery_plan(round_root: Path, round_index: int, *, rollouts_per_round: int = 128) -> dict[str, Any]:
    root = round_root.expanduser().resolve()
    if not root.is_dir():
        raise RoundPlanError(f"round root is not a directory: {root}")
    phases = [classify_phase(root, spec, round_index, rollouts_per_round) for spec in PHASES]
    errors = [error for row in phases for error in row.get("errors", [])]
    errors.extend(_cross_phase_errors(phases, round_index))
    complete_rows = [row for row in phases if row["status"] == COMPLETED]
    must_not_replay = [row["phase"] for row in complete_rows]
    readbacks = [
        {"phase": row["phase"], "path": row["receipt_path"], "sha256": row.get("receipt_sha256")}
        for row in complete_rows
    ]

    if errors or any(row["status"] == INTEGRITY_UNKNOWN for row in phases):
        verdict = "MANUAL_REVIEW_REQUIRED"
        recovery_class = "INTEGRITY_UNKNOWN"
        target_phase = next((row["phase"] for row in phases if row["status"] != COMPLETED), None)
    elif phases[-1]["status"] == COMPLETED:
        verdict = "COMPLETE_NO_REPLAY"
        recovery_class = "NONE"
        target_phase = None
    else:
        target = next(row for row in phases if row["status"] != COMPLETED)
        target_phase = target["phase"]
        if target["status"] == FAILED_NO_SIDE_EFFECT:
            verdict = "SAFE_TO_PREPARE_SAME_INPUT_RETRY"
            recovery_class = "ENGINEERING_RETRYABLE"
        elif target["status"] == NOT_STARTED and target_phase in {"state_after", "round_complete"}:
            verdict = "SAFE_TO_PREPARE_ACK_OR_COMPOSE_ONLY"
            recovery_class = "ENGINEERING_RETRYABLE"
        elif target["status"] == NOT_STARTED:
            verdict = "SAFE_TO_PREPARE_NEXT_PHASE"
            recovery_class = "RESUME_REQUIRED"
        else:
            verdict = "MANUAL_REVIEW_REQUIRED"
            recovery_class = "SIDE_EFFECT_STATUS_UNKNOWN"

    return {
        "schema": SCHEMA,
        "mutation": False,
        "execution_action": "NONE",
        "round_root": str(root),
        "round_index": round_index,
        "rollouts_per_round": rollouts_per_round,
        "verdict": verdict,
        "recovery_class": recovery_class,
        "target_phase": target_phase,
        "phases": [{key: value for key, value in row.items() if key != "document"} for row in phases],
        "must_not_replay": must_not_replay,
        "required_readbacks_before_resume": readbacks,
        "integrity_errors": errors,
        "authority_required_before_execution": True,
        "scientific_retry_authorized": False,
        "planner_contract": {
            "completed_phase_replay_forbidden": True,
            "failed_without_explicit_no-side-effect_proof_is_not_retryable": True,
            "successful_container_without_canonical_receipt_requires_readback": True,
            "planner_never_launches_or_mutates": True,
        },
    }


def render(plan: Mapping[str, Any]) -> str:
    lines = [
        f"Round root: {plan['round_root']}",
        f"Round: {plan['round_index']}",
        f"Verdict: {plan['verdict']}",
        f"Recovery class: {plan['recovery_class']}",
        f"Target phase: {plan.get('target_phase') or 'NONE'}",
        f"Must not replay: {', '.join(plan.get('must_not_replay') or []) or 'none'}",
        "Phases:",
    ]
    lines.extend(f"- {row['phase']}: {row['status']}" for row in plan["phases"])
    if plan.get("integrity_errors"):
        lines.append("Integrity errors:")
        lines.extend(f"- {error}" for error in plan["integrity_errors"])
    lines.extend([
        "Execution action: NONE — this planner is read-only.",
        "Any real resume still requires the current campaign/release/exact executor authority.",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-root", type=Path, required=True)
    parser.add_argument("--round-index", type=int, required=True)
    parser.add_argument("--rollouts-per-round", type=int, default=128)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = build_round_recovery_plan(
            args.round_root, args.round_index, rollouts_per_round=args.rollouts_per_round
        )
    except RoundPlanError as exc:
        print(f"formal-round-transaction: {exc}")
        return 2
    print(json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False) if args.json else render(plan))
    return 0 if plan["verdict"] != "MANUAL_REVIEW_REQUIRED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
