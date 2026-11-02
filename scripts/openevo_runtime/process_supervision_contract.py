"""Declared process-supervision contract for prospective Execution Runtime workers.

This layer reuses the Phase-A systemd service bridge. It owns only OS process
lifetime: it does not authorize launch, resume, scientific retry, or workflow replay.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.openevo_runtime.service_supervision import (
    ServiceSpec,
    render_systemd_unit,
    service_receipt,
    validate_service_spec,
)
from scripts.openevo_webshop.experiment_passport import (
    PassportError,
    assert_no_credentials,
    canonical_bytes,
    validate_schema,
)

ROOT = Path(__file__).resolve().parents[2]
REV = "202609171050"
SCHEMA_ID = f"openevo.process-supervision-contract.{REV}"
SCHEMA_PATH = ROOT / f"schemas/process-supervision-contract-{REV}.schema.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ROLES = frozenset({"CONTROL_PLANE", "RESOURCE_BROKER", "ARM_SUPERVISOR", "PAIR_AUDIT", "WORKER"})


class ProcessSupervisionContractError(ValueError):
    """A process-lifetime declaration violates the non-authoritative bridge contract."""


def _digest(value: Mapping[str, Any]) -> str:
    assert_no_credentials(value)
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _schema() -> dict[str, Any]:
    try:
        value = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProcessSupervisionContractError("process-supervision schema is unreadable") from exc
    if not isinstance(value, dict):
        raise ProcessSupervisionContractError("process-supervision schema must be an object")
    return value


def build_service_member(spec: ServiceSpec, *, role: str) -> dict[str, Any]:
    validate_service_spec(spec)
    if role not in ROLES:
        raise ProcessSupervisionContractError(f"unsupported process role: {role}")
    unit = render_systemd_unit(spec)
    receipt = service_receipt(spec)
    if "Restart=no" not in unit or "[Install]" in unit:
        raise ProcessSupervisionContractError("service bridge no longer satisfies fail-closed lifetime policy")
    if receipt.get("scientific_authority") is not False or receipt.get("scientific_retry_authorized") is not False:
        raise ProcessSupervisionContractError("service receipt cannot grant scientific authority")
    return {
        "role": role,
        "service_name": spec.service_name,
        "working_directory": str(spec.working_directory),
        "argv": list(spec.argv),
        "user": spec.user,
        "group": spec.group,
        "unit_sha256": hashlib.sha256(unit.encode("utf-8")).hexdigest(),
        "process_supervisor": "systemd",
        "restart_policy": "NONE_FAIL_CLOSED",
        "manual_start_required": True,
        "terminal_session_lifetime_allowed": False,
        "boot_auto_start_allowed": False,
        "scientific_authority": False,
        "scientific_retry_authorized": False,
    }


def contract_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "schema",
        "execution_generation_id",
        "members",
        "lifetime_policy",
        "live_adoption",
        "scientific_authority",
        "scientific_retry_authorized",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ProcessSupervisionContractError(f"missing process-supervision field: {missing[0]}")
    if payload["schema"] != SCHEMA_ID:
        raise ProcessSupervisionContractError(f"schema must equal {SCHEMA_ID}")
    generation = payload["execution_generation_id"]
    if not isinstance(generation, str) or not HEX64.fullmatch(generation):
        raise ProcessSupervisionContractError("execution_generation_id must be a SHA-256 identity")
    members = payload["members"]
    if not isinstance(members, list) or not members:
        raise ProcessSupervisionContractError("members must be a non-empty list")
    names: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for member in members:
        if not isinstance(member, Mapping):
            raise ProcessSupervisionContractError("member rows must be objects")
        if member.get("role") not in ROLES:
            raise ProcessSupervisionContractError("member role is not declared")
        name = member.get("service_name")
        if not isinstance(name, str) or not name:
            raise ProcessSupervisionContractError("member service_name must be non-empty")
        if name in names:
            raise ProcessSupervisionContractError("service_name must be unique within one contract")
        names.add(name)
        if member.get("process_supervisor") != "systemd":
            raise ProcessSupervisionContractError("process lifetime must use the declared systemd bridge")
        if member.get("restart_policy") != "NONE_FAIL_CLOSED":
            raise ProcessSupervisionContractError("automatic process restart is not permitted")
        for key in (
            "manual_start_required",
            "terminal_session_lifetime_allowed",
            "boot_auto_start_allowed",
            "scientific_authority",
            "scientific_retry_authorized",
        ):
            if type(member.get(key)) is not bool:
                raise ProcessSupervisionContractError(f"member {key} must be boolean")
        if member["manual_start_required"] is not True:
            raise ProcessSupervisionContractError("service start must remain separately authorized")
        if any(member[key] is not False for key in (
            "terminal_session_lifetime_allowed",
            "boot_auto_start_allowed",
            "scientific_authority",
            "scientific_retry_authorized",
        )):
            raise ProcessSupervisionContractError("service member violates non-authoritative lifetime policy")
        unit_sha = member.get("unit_sha256")
        if not isinstance(unit_sha, str) or not HEX64.fullmatch(unit_sha):
            raise ProcessSupervisionContractError("member unit_sha256 must be a SHA-256 identity")
        argv = member.get("argv")
        if not isinstance(argv, list) or not argv or any(not isinstance(v, str) or not v for v in argv):
            raise ProcessSupervisionContractError("member argv must be a non-empty string vector")
        normalized.append(dict(member))
    expected_policy = {
        "single_declared_contract": True,
        "ssh_or_nohup_is_lifetime_authority": False,
        "process_death_implies_scientific_retry": False,
        "failure_requires_openevo_reconciliation": True,
        "service_state_is_scientific_authority": False,
    }
    if payload["lifetime_policy"] != expected_policy:
        raise ProcessSupervisionContractError("lifetime_policy drift")
    if payload["live_adoption"] is not False:
        raise ProcessSupervisionContractError("prospective contract cannot claim live adoption")
    if payload["scientific_authority"] is not False or payload["scientific_retry_authorized"] is not False:
        raise ProcessSupervisionContractError("process contract cannot grant scientific authority")
    projection = {key: payload[key] for key in required}
    projection["members"] = normalized
    assert_no_credentials(projection)
    canonical_bytes(projection)
    return projection


def build_process_supervision_contract(
    *, execution_generation_id: str, members: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = {
        "schema": SCHEMA_ID,
        "execution_generation_id": execution_generation_id,
        "members": [dict(member) for member in members],
        "lifetime_policy": {
            "single_declared_contract": True,
            "ssh_or_nohup_is_lifetime_authority": False,
            "process_death_implies_scientific_retry": False,
            "failure_requires_openevo_reconciliation": True,
            "service_state_is_scientific_authority": False,
        },
        "live_adoption": False,
        "scientific_authority": False,
        "scientific_retry_authorized": False,
    }
    projection = contract_projection(payload)
    result = {**payload, "contract_id": _digest(projection)}
    validate_process_supervision_contract(result)
    return result


def validate_process_supervision_contract(payload: Mapping[str, Any]) -> None:
    try:
        validate_schema(dict(payload), _schema())
    except PassportError as exc:
        raise ProcessSupervisionContractError(str(exc)) from exc
    projection = contract_projection(payload)
    if payload.get("contract_id") != _digest(projection):
        raise ProcessSupervisionContractError("contract_id does not match canonical projection")
