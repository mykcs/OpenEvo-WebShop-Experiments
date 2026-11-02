#!/usr/bin/env python3
"""Fail-closed host/container path validation before GPU allocation."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Callable


class PathContractError(ValueError):
    """The execution path contract cannot prove safe container semantics."""


def _expected_type(path: Path, expected: str, label: str) -> None:
    if expected == "dir" and not path.is_dir():
        raise PathContractError(f"{label}: expected directory: {path}")
    if expected == "file" and not path.is_file():
        raise PathContractError(f"{label}: expected file: {path}")
    if expected not in {"dir", "file"}:
        raise PathContractError(f"{label}: unsupported expected_type: {expected}")


def validate_host_path(spec: dict[str, Any], label: str) -> Path:
    raw = spec.get("host_path")
    if not isinstance(raw, str) or not raw.startswith("/"):
        raise PathContractError(f"{label}: host_path must be absolute")
    path = Path(raw)
    if not path.exists():
        if spec.get("must_exist", True) is not False or not path.parent.is_dir():
            raise PathContractError(f"{label}: host path does not exist: {path}")
        if spec.get("writable", False) and not os.access(path.parent, os.W_OK):
            raise PathContractError(f"{label}: parent directory is not writable for new path: {path}")
        return path
    _expected_type(path, str(spec.get("expected_type")), label)
    if spec.get("readable", True) and not os.access(path, os.R_OK):
        raise PathContractError(f"{label}: host path is not readable: {path}")
    if spec.get("writable", False) and not os.access(path, os.W_OK):
        raise PathContractError(f"{label}: host path is not writable: {path}")
    return path


def validate_mount_mapping(
    spec: dict[str, Any],
    label: str,
    container_root: Path | None = None,
) -> tuple[Path, PurePosixPath]:
    host = validate_host_path(spec, label)
    raw_container = spec.get("container_path")
    if not isinstance(raw_container, str) or not raw_container.startswith("/"):
        raise PathContractError(f"{label}: container_path must be absolute")
    container = PurePosixPath(raw_container)
    if str(container) == str(host) or str(container).startswith("/host/") or str(container).startswith("/Users/"):
        raise PathContractError(f"{label}: host-only path was exposed as a container mapping")
    access = spec.get("access")
    if access not in {"ro", "rw"}:
        raise PathContractError(f"{label}: access must be ro or rw")
    if spec.get("writable", False) and access != "rw":
        raise PathContractError(f"{label}: writable host path needs rw container mapping")
    if container_root is not None:
        visible = container_root / str(container).lstrip("/")
        _expected_type(visible, str(spec.get("expected_type")), f"{label} container-visible")
        if spec.get("readable", True) and not os.access(visible, os.R_OK):
            raise PathContractError(f"{label}: container-visible path is not readable: {container}")
        if spec.get("writable", False) and not os.access(visible, os.W_OK):
            raise PathContractError(f"{label}: container-visible path is not writable: {container}")
        if spec.get("verify_same_object", True):
            host_stat = host.stat()
            visible_stat = visible.stat()
            if (host_stat.st_dev, host_stat.st_ino) != (visible_stat.st_dev, visible_stat.st_ino):
                raise PathContractError(f"{label}: container-visible path is not the exact host mapping")
    return host, container


def run_cpu_no_gpu_probe(
    command: list[str] | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> None:
    """Run a bounded probe with GPU visibility explicitly disabled."""
    probe = command or [
        "python3",
        "-c",
        "import os; assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('', '-1')",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = runner(probe, check=False, env=env, capture_output=True, text=True)
    if getattr(result, "returncode", 1) != 0:
        raise PathContractError("CPU/no-GPU probe failed")


def validate_contract(
    contract: dict[str, Any],
    container_root: Path | None = None,
    probe: Callable[[], None] | None = None,
) -> dict[str, Any]:
    paths = contract.get("paths")
    if not isinstance(paths, dict) or not paths:
        raise PathContractError("path contract must contain named paths")
    for label, spec in paths.items():
        if not isinstance(spec, dict):
            raise PathContractError(f"{label}: path specification must be an object")
        validate_mount_mapping(spec, label, container_root=container_root)
    if probe is not None:
        probe()
    return {
        "status": "pass",
        "path_count": len(paths),
        "cpu_probe": probe is not None,
        "container_probe": container_root is not None,
    }


def load_and_validate(path: Path, container_root: Path | None = None) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PathContractError("path contract JSON must be an object")
    return validate_contract(value, container_root=container_root, probe=run_cpu_no_gpu_probe)
