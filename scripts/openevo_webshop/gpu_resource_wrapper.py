#!/usr/bin/env python3
"""Execute one GPU-owning command with an atomic, fail-closed resource receipt."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from scripts.openevo_webshop.path_contract import PathContractError, load_and_validate


SCHEMA = "openevo.gpu-resource-receipt.v1"
_LAST_RUN_RECEIPT: Path | None = None


class ResourceAccountingError(subprocess.CalledProcessError):
    """A payload may have run, but its resource receipt cannot be proven exact.

    This deliberately derives from CalledProcessError rather than the launcher's
    retryable GPU-acquisition error. A post-payload accounting failure must stop;
    it must never turn into an automatic scientific retry.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(125, ["gpu-resource-accounting"])

    def __str__(self) -> str:
        return self.reason


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _base_receipt(args: argparse.Namespace) -> dict[str, Any]:
    uuids = [value.strip() for value in args.gpu_uuids.split(",") if value.strip()]
    if not uuids or len(set(uuids)) != len(uuids):
        raise ValueError("gpu_uuids must contain a non-empty unique GPU UUID set")
    return {
        "schema": SCHEMA,
        "attempt_id": args.attempt_id,
        "campaign_id": args.campaign_id,
        "stage": args.stage,
        "command_identity": args.command_identity,
        "gpu_uuids": uuids,
        "gpu_count": len(uuids),
        "start_utc": None,
        "end_utc": None,
        "wall_seconds": None,
        "gpu_hours": None,
        "return_code": None,
        "status": "skeleton",
        "path_contract": args.path_contract,
        "wrapper_version": "openevo.gpu-resource-wrapper.v1",
    }


def _finalize(
    receipt: dict[str, Any],
    path: Path,
    start_utc: str,
    start_mono: float,
    status: str,
    return_code: int | None,
    error: str | None = None,
) -> dict[str, Any]:
    wall = max(0.0, time.monotonic() - start_mono)
    receipt.update(
        {
            "end_utc": utc_now(),
            "wall_seconds": wall,
            "gpu_hours": wall * receipt["gpu_count"] / 3600.0,
            "return_code": return_code,
            "status": status,
            "error": error,
        }
    )
    if receipt.get("start_utc") is None:
        receipt["start_utc"] = start_utc
    atomic_write(path, receipt)
    return receipt


def run(args: argparse.Namespace) -> int:
    global _LAST_RUN_RECEIPT
    receipt_path = Path(args.receipt).resolve()
    _LAST_RUN_RECEIPT = receipt_path
    receipt = _base_receipt(args)
    atomic_write(receipt_path, receipt)
    preflight_start = time.monotonic()
    preflight_utc = utc_now()
    try:
        if args.path_contract:
            load_and_validate(
                Path(args.path_contract),
                container_root=Path(args.container_root).resolve() if args.container_root else None,
            )
        start_utc = utc_now()
        start_mono = time.monotonic()
        receipt["start_utc"] = start_utc
        receipt["status"] = "running"
        atomic_write(receipt_path, receipt)
        return_code: int | None = None
        status = "wrapper-error"
        error: str | None = None
        try:
            completed = subprocess.run(args.command, check=False)
            return_code = int(completed.returncode)
            status = "succeeded" if return_code == 0 else "failed"
        except BaseException as exc:
            error = repr(exc)
        finally:
            _finalize(receipt, receipt_path, start_utc, start_mono, status, return_code, error)
        return return_code if return_code is not None else 125
    except (PathContractError, OSError, ValueError) as exc:
        receipt["start_utc"] = preflight_utc
        receipt["status"] = "preflight-failed"
        receipt["gpu_hours"] = 0.0
        receipt["wall_seconds"] = max(0.0, time.monotonic() - preflight_start)
        receipt["end_utc"] = utc_now()
        receipt["return_code"] = 125
        receipt["error"] = repr(exc)
        atomic_write(receipt_path, receipt)
        return 125


def account_receipt(value: dict[str, Any]) -> dict[str, Any]:
    """Accept completed success/failure receipts when accounting is exact."""
    status = value.get("status")
    if value.get("schema") != SCHEMA or status not in {
        "succeeded",
        "failed",
        "wrapper-error",
        "preflight-failed",
    }:
        return {"status": "unknown", "reason": "wrapper absent or corrupt"}
    wall = value.get("wall_seconds")
    gpu_hours = value.get("gpu_hours")
    count = value.get("gpu_count")
    if not all(
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(float(x))
        and float(x) >= 0
        for x in (wall, gpu_hours, count)
    ):
        return {"status": "unknown", "reason": "wrapper absent or corrupt"}
    if status == "preflight-failed":
        if not math.isclose(float(gpu_hours), 0.0, rel_tol=0.0, abs_tol=0.0):
            return {"status": "unknown", "reason": "preflight failure must account zero GPU-hours"}
        return {
            "status": "exact",
            "gpu_hours": 0.0,
            "return_code": value.get("return_code"),
            "pre_gpu_failure": True,
        }
    expected = float(wall) * float(count) / 3600.0
    if not math.isclose(float(gpu_hours), expected, rel_tol=1e-9, abs_tol=1e-9):
        return {"status": "unknown", "reason": "wrapper receipt accounting mismatch"}
    return {
        "status": "exact",
        "gpu_hours": float(gpu_hours),
        "return_code": value.get("return_code"),
        "pre_gpu_failure": False,
    }


def classify_receipt(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        accounting = {"status": "unknown", "reason": "wrapper absent or corrupt"}
    else:
        if not isinstance(value, dict):
            accounting = {"status": "unknown", "reason": "wrapper absent or corrupt"}
        else:
            accounting = account_receipt(value)
    if resolved == _LAST_RUN_RECEIPT and accounting.get("status") != "exact":
        raise ResourceAccountingError(
            f"GPU payload may have executed but resource accounting is not exact: {accounting}"
        )
    return accounting


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--command-identity", required=True)
    parser.add_argument("--gpu-uuids", required=True)
    parser.add_argument("--path-contract")
    parser.add_argument("--container-root")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a payload command is required")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())