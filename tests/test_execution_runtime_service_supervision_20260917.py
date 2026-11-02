from __future__ import annotations

from pathlib import Path

import pytest

from scripts.openevo_runtime.service_supervision import (
    ServiceSpec,
    ServiceSpecError,
    render_systemd_unit,
    service_receipt,
)


def spec() -> ServiceSpec:
    return ServiceSpec(
        service_name="openevo-off-supervisor",
        description="OpenEVO OFF arm control supervisor",
        working_directory=Path("/workspace/openevo-experiment"),
        argv=("/usr/bin/python3", "scripts/openevo_webshop/example.py", "--arm-key", "off"),
        user="1001",
        group="1001",
    )

def test_unit_survives_shell_logout_without_auto_restart() -> None:
    unit = render_systemd_unit(spec())
    assert "Type=simple" in unit
    assert "Restart=no" in unit
    assert "KillMode=control-group" in unit
    assert "WantedBy=" not in unit
    assert "ExecStart=" in unit


def test_receipt_cannot_be_mistaken_for_scientific_authority() -> None:
    receipt = service_receipt(spec())
    assert receipt["restart_policy"] == "NONE_FAIL_CLOSED"
    assert receipt["manual_start_required"] is True
    assert receipt["scientific_retry_authorized"] is False
    assert receipt["scientific_authority"] is False
    assert receipt["live_adoption"] is False


def test_argv_is_quoted_without_a_shell() -> None:
    with_space = ServiceSpec(
        **{**spec().__dict__, "argv": ("/usr/bin/python3", "/tmp/path with space/worker.py")}
    )
    unit = render_systemd_unit(with_space)
    assert 'ExecStart="/usr/bin/python3" "/tmp/path with space/worker.py"' in unit

def test_invalid_service_spec_fails_closed() -> None:
    bad = ServiceSpec(
        service_name="bad service name",
        description="bad",
        working_directory=Path("relative/path"),
        argv=(),
        user="1001",
        group="1001",
    )
    with pytest.raises(ServiceSpecError):
        render_systemd_unit(bad)
