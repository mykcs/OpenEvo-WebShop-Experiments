"""Fail-closed OS service bridge for long-lived OpenEVO control processes.

The bridge owns process lifetime only. It deliberately does not auto-restart failed
scientific control processes, classify recovery, or grant launch/resume authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Sequence

SERVICE_SCHEMA = "openevo.runtime-service-bridge.202609170105"
_NAME_RE = re.compile(r"^[A-Za-z0-9_.@-]+$")


class ServiceSpecError(ValueError):
    pass


@dataclass(frozen=True)
class ServiceSpec:
    service_name: str
    description: str
    working_directory: Path
    argv: tuple[str, ...]
    user: str
    group: str
    kill_mode: str = "control-group"

def _validate_atom(value: str, field_name: str) -> None:
    if not value or "\n" in value or "\x00" in value:
        raise ServiceSpecError(f"{field_name} must be a non-empty single-line string")


def validate_service_spec(spec: ServiceSpec) -> None:
    _validate_atom(spec.service_name, "service_name")
    if not _NAME_RE.fullmatch(spec.service_name):
        raise ServiceSpecError("service_name contains unsupported characters")
    _validate_atom(spec.description, "description")
    _validate_atom(spec.user, "user")
    _validate_atom(spec.group, "group")
    if spec.kill_mode not in {"control-group", "process"}:
        raise ServiceSpecError("kill_mode must be control-group or process")
    if not spec.working_directory.is_absolute():
        raise ServiceSpecError("working_directory must be absolute")
    if not spec.argv:
        raise ServiceSpecError("argv must not be empty")
    for index, arg in enumerate(spec.argv):
        _validate_atom(arg, f"argv[{index}]")


def _systemd_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'

def render_systemd_unit(spec: ServiceSpec) -> str:
    """Render a manual-start unit that survives logout but never auto-retries."""
    validate_service_spec(spec)
    exec_start = " ".join(_systemd_quote(arg) for arg in spec.argv)
    lines: Sequence[str] = (
        "[Unit]",
        f"Description={spec.description}",
        "",
        "[Service]",
        "Type=simple",
        f"User={spec.user}",
        f"Group={spec.group}",
        f"WorkingDirectory={_systemd_quote(str(spec.working_directory))}",
        f"ExecStart={exec_start}",
        "Restart=no",
        f"KillMode={spec.kill_mode}",
        "TimeoutStopSec=30",
        "StandardOutput=journal",
        "StandardError=journal",
        "",
    )
    return "\n".join(lines)


def service_receipt(spec: ServiceSpec) -> dict[str, object]:
    validate_service_spec(spec)
    return {
        "schema": SERVICE_SCHEMA,
        "service_name": spec.service_name,
        "process_supervisor": "systemd",
        "restart_policy": "NONE_FAIL_CLOSED",
        "manual_start_required": True,
        "scientific_retry_authorized": False,
        "scientific_authority": False,
        "live_adoption": False,
        "kill_mode": spec.kill_mode,
    }
