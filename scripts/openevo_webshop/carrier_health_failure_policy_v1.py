#!/usr/bin/env python3
"""Pure Carrier Health failure policy for matched studies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "openevo.carrier-health-failure-policy.v1"
POLICY_ID = "carrier-health-failure-policy-v1"
FORBIDDEN_MUTATIONS = (
    "raise_token_cap",
    "force_selected",
    "third_repair",
    "extra_reflector_call",
    "change_update_frequency",
    "arm_specific_fallback",
    "change_acceleration_semantics",
    "change_gdr",
    "reward_conditioned_repair",
)


def canonical_sha(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_policy(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != SCHEMA or payload.get("policy_id") != POLICY_ID:
        raise ValueError("carrier health failure policy identity mismatch")
    pre = payload.get("pre_round0") or {}
    if pre != {
        "required_status": "PASS",
        "on_fail": "BLOCKED",
        "formal_task_consumption_allowed": False,
    }:
        raise ValueError("carrier health pre-Round0 policy drift")
    formal = payload.get("formal_run") or {}
    if formal != {
        "evaluate_after_round_complete": True,
        "preserve_completed_transaction": True,
        "on_health_fail": "PAUSE_FOR_DIAGNOSTIC",
        "block_next_round": True,
    }:
        raise ValueError("carrier health formal-run policy drift")
    if payload.get("reward_conditioning_allowed") is not False or payload.get("final_panel_access_allowed") is not False:
        raise ValueError("carrier health policy may not depend on reward/final panel")
    mutations = payload.get("automatic_mutations") or {}
    if set(mutations) != set(FORBIDDEN_MUTATIONS) or any(mutations.get(key) is not False for key in FORBIDDEN_MUTATIONS):
        raise ValueError("carrier health automatic-mutation prohibition drift")
    resume = payload.get("resume") or {}
    if resume != {
        "requires_resolution_receipt": True,
        "automatic_resume": False,
        "same_contract_allowed": True,
        "new_contract_requires_prospective_authority": True,
        "single_arm_patch_allowed": False,
    }:
        raise ValueError("carrier health resume policy drift")
    return payload


def policy_identity(payload: dict[str, Any]) -> str:
    return canonical_sha(payload)


def pre_round0_decision(policy: dict[str, Any], checks: dict[str, str]) -> dict[str, Any]:
    failed = sorted(name for name, status in checks.items() if status != "PASS")
    blocked = bool(failed)
    return {
        "schema": "openevo.carrier-health-preflight-decision.v1",
        "policy_id": policy["policy_id"],
        "policy_identity_sha256": policy_identity(policy),
        "status": "BLOCKED" if blocked else "PASS",
        "formal_task_consumption_allowed": not blocked,
        "failed_checks": failed,
        "checks": dict(sorted(checks.items())),
    }


def formal_round_decision(policy: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any]:
    if ledger.get("schema") != "openevo.carrier-health-ledger.v2":
        raise ValueError("carrier health ledger schema mismatch")
    status = ledger.get("status")
    if status not in {"PASS", "FAIL"}:
        raise ValueError("carrier health ledger status invalid")
    failed_rows = [
        {"carrier": row.get("carrier"), "health_reasons": list(row.get("health_reasons") or [])}
        for row in (ledger.get("rows") or []) if row.get("health_status") != "PASS"
    ]
    pause = status == "FAIL"
    return {
        "schema": "openevo.carrier-health-boundary-decision.v1",
        "policy_id": policy["policy_id"],
        "policy_identity_sha256": policy_identity(policy),
        "round_index": int(ledger.get("round_index", -1)),
        "arm": ledger.get("arm"),
        "ledger_status": status,
        "action": "PAUSE_FOR_DIAGNOSTIC" if pause else "CONTINUE",
        "preserve_completed_transaction": True,
        "block_next_round": pause,
        "failed_rows": failed_rows,
        "automatic_mutations": {key: False for key in FORBIDDEN_MUTATIONS},
    }


def validate_resolution_receipt(
    policy: dict[str, Any], receipt: dict[str, Any], *,
    expected_arm: str, expected_round: int, expected_pause_receipt_sha256: str,
) -> dict[str, Any]:
    if receipt.get("schema") != "openevo.carrier-health-resolution.v1" or receipt.get("status") != "RESOLVED":
        raise ValueError("carrier health resolution receipt invalid")
    if receipt.get("policy_identity_sha256") != policy_identity(policy):
        raise ValueError("carrier health resolution policy identity drift")
    if receipt.get("arm") != expected_arm or int(receipt.get("through_round", -1)) != int(expected_round):
        raise ValueError("carrier health resolution arm/round mismatch")
    if receipt.get("pause_receipt_sha256") != expected_pause_receipt_sha256:
        raise ValueError("carrier health resolution pause identity drift")
    if receipt.get("automatic_scientific_change") is not False:
        raise ValueError("carrier health resolution cannot authorize automatic scientific change")
    if receipt.get("contract_transition") != "SAME_CONTRACT":
        raise ValueError("this controller resumes only under the same frozen carrier contract")
    return receipt
