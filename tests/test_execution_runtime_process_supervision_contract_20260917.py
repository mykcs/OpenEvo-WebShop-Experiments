from pathlib import Path

import pytest

from scripts.openevo_runtime.process_supervision_contract import (
    ProcessSupervisionContractError,
    build_process_supervision_contract,
    build_service_member,
    validate_process_supervision_contract,
)
from scripts.openevo_runtime.service_supervision import ServiceSpec


def _spec(name: str, *argv: str) -> ServiceSpec:
    return ServiceSpec(
        service_name=name,
        description=f"OpenEVO {name}",
        working_directory=Path("/srv/openevo/execution"),
        argv=tuple(argv),
        user="openevo",
        group="openevo",
    )


def test_contract_reuses_fail_closed_systemd_bridge_for_long_lived_roles() -> None:
    members = [
        build_service_member(_spec("openevo-control", "/usr/bin/python3", "control.py"), role="CONTROL_PLANE"),
        build_service_member(_spec("openevo-broker", "/usr/bin/python3", "broker.py"), role="RESOURCE_BROKER"),
        build_service_member(_spec("openevo-off", "/usr/bin/python3", "arm.py", "--arm", "off"), role="ARM_SUPERVISOR"),
        build_service_member(_spec("openevo-on", "/usr/bin/python3", "arm.py", "--arm", "on"), role="ARM_SUPERVISOR"),
    ]
    contract = build_process_supervision_contract(execution_generation_id="a" * 64, members=members)
    validate_process_supervision_contract(contract)
    assert contract["live_adoption"] is False
    assert contract["scientific_authority"] is False
    assert contract["scientific_retry_authorized"] is False
    assert contract["lifetime_policy"]["ssh_or_nohup_is_lifetime_authority"] is False
    assert all(row["process_supervisor"] == "systemd" for row in contract["members"])
    assert all(row["restart_policy"] == "NONE_FAIL_CLOSED" for row in contract["members"])


def test_contract_identity_binds_exact_worker_command() -> None:
    first = build_service_member(
        _spec("openevo-worker", "/usr/bin/python3", "worker.py", "--round", "7"),
        role="WORKER",
    )
    second = build_service_member(
        _spec("openevo-worker", "/usr/bin/python3", "worker.py", "--round", "8"),
        role="WORKER",
    )
    a = build_process_supervision_contract(execution_generation_id="b" * 64, members=[first])
    b = build_process_supervision_contract(execution_generation_id="b" * 64, members=[second])
    assert a["contract_id"] != b["contract_id"]
    assert first["unit_sha256"] != second["unit_sha256"]


def test_duplicate_service_name_fails_closed() -> None:
    row = build_service_member(
        _spec("openevo-arm", "/usr/bin/python3", "arm.py"),
        role="ARM_SUPERVISOR",
    )
    with pytest.raises(ProcessSupervisionContractError, match="unique"):
        build_process_supervision_contract(execution_generation_id="c" * 64, members=[row, row])


def test_service_lifetime_cannot_be_promoted_into_retry_authority() -> None:
    row = build_service_member(
        _spec("openevo-control", "/usr/bin/python3", "control.py"),
        role="CONTROL_PLANE",
    )
    contract = build_process_supervision_contract(execution_generation_id="d" * 64, members=[row])
    contract["members"][0]["restart_policy"] = "ALWAYS"
    with pytest.raises(ProcessSupervisionContractError):
        validate_process_supervision_contract(contract)


def test_contract_cannot_claim_live_adoption() -> None:
    row = build_service_member(
        _spec("openevo-pair-audit", "/usr/bin/python3", "audit.py"),
        role="PAIR_AUDIT",
    )
    contract = build_process_supervision_contract(execution_generation_id="e" * 64, members=[row])
    contract["live_adoption"] = True
    with pytest.raises(ProcessSupervisionContractError):
        validate_process_supervision_contract(contract)
