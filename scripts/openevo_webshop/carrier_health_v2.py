#!/usr/bin/env python3
"""Pure Carrier Contract v2 health ledger construction and evaluation.

No model, reward, task-selection, final-panel, launch, or treatment authority lives here.
A healthy NOOP is explicitly allowed; update frequency is never a health criterion.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

SCHEMA = "openevo.carrier-health.v2"
TERMINAL_DISPOSITIONS = {
    "UPDATE",
    "NOOP_GATE_INELIGIBLE",
    "NOOP_NO_CONTENT_CHANGE",
    "NOOP_KEEP_PRIOR_CONTRACT_REJECTED",
}
IDENTITY_KEYS = ("generator", "validator", "repair", "router", "governor")
CONTRACT_FAIL_REASON = "NOOP_KEEP_PRIOR_TYPED_CONTRACT_REJECTED_AFTER_REPAIR"


def _single(values: set[Any]) -> Any:
    return next(iter(values)) if len(values) == 1 else None


def _runtime_receipt(step: dict[str, Any]) -> dict[str, Any] | None:
    receipt = step.get("runtime_context_receipt")
    return receipt if isinstance(receipt, dict) else None


def aggregate_runtime_readbacks(rows: Iterable[dict[str, Any]], carrier: str) -> dict[str, Any]:
    requested_steps = 0
    selected_steps = 0
    reasons: Counter[str] = Counter()
    requested_shas: set[str] = set()
    selected_shas: set[str] = set()
    requested_tokens: set[int] = set()
    selected_tokens: set[int] = set()
    policies: set[str] = set()
    integrity: list[str] = []
    for row in rows:
        for step in row.get("episode", {}).get("trace", []):
            receipt = _runtime_receipt(step)
            if receipt is None:
                continue
            readback = (receipt.get("carrier_readback") or {}).get(carrier)
            if not isinstance(readback, dict) or not readback.get("requested"):
                continue
            requested_steps += 1
            reason = str(readback.get("governor_reason") or "unexplained_drop")
            reasons[reason] += 1
            requested_sha = readback.get("requested_payload_sha256") or readback.get("sha256")
            if requested_sha:
                requested_shas.add(str(requested_sha))
            if readback.get("requested_tokens") is not None:
                requested_tokens.add(int(readback["requested_tokens"]))
            route = receipt.get("thin_route") or {}
            policy = route.get("agent_system") if carrier == "agent_system" else "page-routed-non-global"
            if policy:
                policies.add(str(policy))
            if readback.get("selected"):
                selected_steps += 1
                selected_sha = (
                    readback.get("runtime_readback_payload_sha256")
                    or readback.get("selected_payload_sha256")
                    or readback.get("sha256")
                )
                if selected_sha:
                    selected_shas.add(str(selected_sha))
                if readback.get("selected_tokens") is not None:
                    selected_tokens.add(int(readback["selected_tokens"]))
    if len(requested_shas) > 1:
        integrity.append("MULTIPLE_REQUESTED_PAYLOAD_IDENTITIES")
    if len(selected_shas) > 1:
        integrity.append("MULTIPLE_SELECTED_PAYLOAD_IDENTITIES")
    requested = requested_steps > 0
    all_selected = requested and requested_steps == selected_steps
    reason = _single(set(reasons)) if requested else "not_requested"
    if requested and reason is None:
        reason = "mixed"
    return {
        "router_requested": requested,
        "router_selected": all_selected,
        "router_requested_steps": requested_steps,
        "router_selected_steps": selected_steps,
        "governor_reason": reason,
        "governor_reason_counts": dict(sorted(reasons.items())),
        "requested_payload_sha256": _single(requested_shas),
        "router_selected_payload_sha256": _single(selected_shas),
        "runtime_readback_payload_sha256": _single(selected_shas),
        "requested_tokens": _single(requested_tokens),
        "selected_tokens": _single(selected_tokens),
        "router_policy": _single(policies),
        "readback_integrity_reasons": integrity,
    }


def _producer_fields(terminal: dict[str, Any]) -> dict[str, Any]:
    receipt = terminal.get("reflector_receipt") or {}
    source = receipt if receipt else terminal
    attempted = bool(source.get("generation_attempted"))
    if not attempted:
        attempted = int(source.get("primary_calls", terminal.get("primary_calls", 0)) or 0) > 0
    return {
        "generation_attempted": attempted,
        "primary_payload_sha256": source.get("primary_payload_sha256"),
        "primary_contract_valid": source.get("primary_contract_valid"),
        "primary_scope_status": source.get("primary_scope_status"),
        "repair_requested": bool(source.get("repair_requested")) if attempted else False,
        "repair_payload_sha256": source.get("repair_payload_sha256"),
        "repair_changed_bytes": source.get("repair_changed_bytes"),
        "repair_contract_valid": source.get("repair_contract_valid"),
        "repair_scope_status": source.get("repair_scope_status"),
        "scope_manifest_sha256": source.get("scope_manifest_sha256"),
    }


def _scope_result(fields: dict[str, Any]) -> str:
    if not fields["generation_attempted"]:
        return "NOT_APPLICABLE"
    if fields.get("repair_requested"):
        repair = fields.get("repair_scope_status")
        return repair if repair in {"PASS", "FAIL"} else "NOT_EVALUATED"
    primary = fields.get("primary_scope_status")
    return primary if primary in {"PASS", "FAIL"} else "NOT_EVALUATED"


def _terminal_disposition(terminal: dict[str, Any]) -> tuple[str, str]:
    status = terminal.get("status")
    reason = str(terminal.get("reason") or "")
    if status == "UPDATE":
        return "UPDATE", reason or "UPDATE_ACCEPTED_CONTENT_CHANGE"
    if reason == "NOOP_NO_CONTENT_CHANGE":
        return "NOOP_NO_CONTENT_CHANGE", reason
    if reason == CONTRACT_FAIL_REASON:
        return "NOOP_KEEP_PRIOR_CONTRACT_REJECTED", reason
    return "NOOP_GATE_INELIGIBLE", reason or "EVIDENCE_GATE_INELIGIBLE"


def build_carrier_health_record(*, round_index: int, arm: str, carrier: str,
                                state_id_before: str | None, terminal: dict[str, Any],
                                runtime: dict[str, Any], identities: dict[str, str],
                                requested: bool | None = None) -> dict[str, Any]:
    producer = _producer_fields(terminal)
    disposition, reason = _terminal_disposition(terminal)
    record = {
        "schema": SCHEMA,
        "round_index": int(round_index), "arm": arm, "carrier": carrier,
        "state_id_before": state_id_before,
        "requested": producer["generation_attempted"] if requested is None else bool(requested),
        **producer,
        "terminal_disposition": disposition,
        "terminal_reason": reason,
        "accepted_payload_sha256": terminal.get("output_sha256"),
        "task_local_scope_result": _scope_result(producer),
        **runtime,
        "identities": dict(identities),
    }
    return evaluate_carrier_health(record)


def _missing(record: dict[str, Any], *keys: str) -> list[str]:
    return [key for key in keys if record.get(key) is None]


def evaluate_carrier_health(record: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = list(record.get("readback_integrity_reasons") or [])
    if record.get("schema") != SCHEMA:
        reasons.append("SCHEMA_MISMATCH")
    if record.get("terminal_disposition") not in TERMINAL_DISPOSITIONS:
        reasons.append("TERMINAL_DISPOSITION_MISSING_OR_UNKNOWN")
    if not record.get("terminal_reason"):
        reasons.append("TERMINAL_REASON_MISSING")
    if not record.get("accepted_payload_sha256"):
        reasons.append("ACCEPTED_PAYLOAD_IDENTITY_MISSING")
    requested = record.get("requested")
    if not isinstance(requested, bool):
        reasons.append("REQUESTED_MISSING")
    attempted = record.get("generation_attempted")
    if not isinstance(attempted, bool):
        reasons.append("GENERATION_ATTEMPTED_MISSING")
    elif attempted:
        if requested is False:
            reasons.append("GENERATION_WITHOUT_REQUEST")
        missing = _missing(record, "primary_payload_sha256", "primary_contract_valid")
        reasons.extend(f"{key.upper()}_MISSING" for key in missing)
        repair_requested = record.get("repair_requested")
        if not isinstance(repair_requested, bool):
            reasons.append("REPAIR_REQUESTED_MISSING")
        elif repair_requested:
            missing = _missing(record, "repair_payload_sha256", "repair_changed_bytes", "repair_contract_valid")
            reasons.extend(f"{key.upper()}_MISSING" for key in missing)
            if (
                record.get("primary_contract_valid") is False
                and record.get("repair_contract_valid") is False
                and record.get("repair_changed_bytes") is False
            ):
                reasons.append("REPAIR_IDENTICAL_INVALID")
        elif record.get("primary_contract_valid") is False:
            reasons.append("INVALID_PRIMARY_WITHOUT_REPAIR")
    else:
        if requested is True:
            reasons.append("REQUESTED_WITHOUT_GENERATION")
        if record.get("repair_requested") is True:
            reasons.append("REPAIR_WITHOUT_GENERATION")

    scope = record.get("task_local_scope_result")
    if attempted:
        if scope == "FAIL":
            reasons.append("TASK_LOCAL_CONTAMINATION")
        elif scope != "PASS":
            reasons.append("TASK_LOCAL_SCOPE_NOT_EVALUATED")
    elif scope not in {"NOT_APPLICABLE", "PASS"}:
        reasons.append("TASK_LOCAL_SCOPE_RESULT_INVALID")

    router_requested = record.get("router_requested")
    router_selected = record.get("router_selected")
    if not isinstance(router_requested, bool) or not isinstance(router_selected, bool):
        reasons.append("ROUTER_SELECTION_EVIDENCE_MISSING")
    elif router_selected and not router_requested:
        reasons.append("ROUTER_SELECTED_WITHOUT_REQUEST")
    elif router_requested:
        governor_reason = record.get("governor_reason")
        if router_selected:
            if governor_reason != "selected":
                reasons.append("SELECTED_GOVERNOR_REASON_MISMATCH")
            if not record.get("runtime_readback_payload_sha256"):
                reasons.append("RUNTIME_READBACK_PAYLOAD_MISSING")
            if record.get("runtime_readback_payload_sha256") != record.get("router_selected_payload_sha256"):
                reasons.append("RUNTIME_READBACK_PAYLOAD_MISMATCH")
        else:
            if governor_reason in {None, "", "selected", "not_requested", "unexplained_drop"}:
                reasons.append("REQUESTED_DROP_REASON_MISSING")
            if record.get("runtime_readback_payload_sha256") is not None:
                reasons.append("DROPPED_CARRIER_HAS_READBACK_PAYLOAD")
            if record.get("router_policy") == "always-global-authority":
                reasons.append("ALWAYS_GLOBAL_REQUESTED_NOT_SELECTED")

    identities = record.get("identities")
    if not isinstance(identities, dict):
        reasons.append("COMPONENT_IDENTITIES_MISSING")
    else:
        for key in IDENTITY_KEYS:
            if not identities.get(key):
                reasons.append(f"{key.upper()}_IDENTITY_MISSING")

    result = dict(record)
    result["health_status"] = "PASS" if not reasons else "FAIL"
    result["health_reasons"] = sorted(set(reasons))
    return result
