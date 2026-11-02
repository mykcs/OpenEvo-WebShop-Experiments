#!/usr/bin/env python3
"""Run experiment cells with fail-closed fixed or dynamic GPU assignment.

The scheduler is transport-level infrastructure. Scientific cells, manifests,
and formal launch permission come from the versioned plan/receipt; GPU placement
is a separate resource concern.

Two resource modes are supported:

- ``fixed`` (legacy): every cell declares ``gpu_uuid`` and that exact
  mapping is preserved.
- ``dynamic-live-idle``: cells must not declare ``gpu_uuid``; worker lanes are
  created from the runtime candidate pool and idle GPUs pull the next unstarted
  cell. Busy GPUs therefore reduce concurrency instead of pinning a cell to a
  stale resource choice.

New ``openevo.gpu-plan.v2`` formal plans must explicitly declare a resource
assignment mode. Dynamic placement is the ordinary transport contract and does
not require a second reassignment bit. Historical plans keep their legacy
interpretation, including the old reassignment fields when present. The legacy
``--allow-formal`` flag is accepted only as a compatibility no-op; the release
receipt is the machine-verifiable authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from scripts.openevo_webshop.gpu_resource_gate import (
    idle as canonical_idle,
    snapshot_gpu as canonical_snapshot_gpu,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXED_MODE = "fixed"
DYNAMIC_MODE = "dynamic-live-idle"
SUPPORTED_MODES = {FIXED_MODE, DYNAMIC_MODE}
DEFAULT_MAX_IDLE_MEMORY_MIB = 16
DEFAULT_MAX_IDLE_UTIL_PERCENT = 0
PLAN_CONTRACT_VERSION = "openevo.gpu-plan.v2"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a JSON receipt without exposing a truncated file."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def claim_output_root(path: Path) -> None:
    """Atomically claim a fresh evidence namespace for exactly one scheduler."""
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise SystemExit(
            f"output root is not empty or already claimed; refusing to overwrite prior evidence: {path}"
        ) from exc


def parse_uuid_list(value: str) -> set[str]:
    return {item for item in value.replace(",", " ").split() if item}


def assignment_mode(document: dict[str, Any]) -> str:
    if document.get("contract_version") == PLAN_CONTRACT_VERSION and "resource_assignment_mode" not in document:
        raise ValueError("v2 plan must explicitly declare resource_assignment_mode")
    mode = str(document.get("resource_assignment_mode", FIXED_MODE))
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"unsupported resource_assignment_mode: {mode}")
    return mode


def load_plan(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("cells"), list):
        raise ValueError("plan must be an object with a cells list")
    mode = assignment_mode(document)
    if document.get("contract_version") == PLAN_CONTRACT_VERSION and mode == FIXED_MODE:
        reason = document.get("fixed_gpu_binding_reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("v2 fixed plan requires non-empty fixed_gpu_binding_reason")
    cells = document["cells"]
    seen_ids: set[str] = set()
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise ValueError(f"cells[{index}] must be an object")
        cell_id = cell.get("cell_id")
        gpu_uuid = cell.get("gpu_uuid")
        argv = cell.get("argv")
        if not isinstance(cell_id, str) or not cell_id or cell_id in seen_ids:
            raise ValueError(f"cells[{index}] has a duplicate or invalid cell_id")
        if mode == FIXED_MODE:
            if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-"):
                raise ValueError(f"cells[{index}] has an invalid gpu_uuid for fixed mode")
        elif gpu_uuid is not None:
            raise ValueError(f"cells[{index}] must omit gpu_uuid in dynamic-live-idle mode")
        if not isinstance(argv, list) or not argv or not all(isinstance(arg, str) for arg in argv):
            raise ValueError(f"cells[{index}] argv must be a non-empty string list")
        if "run_manifest" in cell and (
            not isinstance(cell["run_manifest"], str) or not cell["run_manifest"]
        ):
            raise ValueError(f"cells[{index}] run_manifest must be a non-empty path string")
        seen_ids.add(cell_id)
    return document, cells


def validate_formal_launch_gate(document: dict[str, Any], plan_path: Path) -> dict[str, Any]:
    """Require a separately materialized, hash-bound scientific launch receipt."""
    receipt_value = document.get("launch_gate_receipt")
    contract_sha = document.get("contract_sha256")
    if not isinstance(receipt_value, str) or not receipt_value:
        raise SystemExit("formal plan must declare launch_gate_receipt")
    if not isinstance(contract_sha, str) or not contract_sha.startswith("sha256:"):
        raise SystemExit("formal plan must declare contract_sha256")
    receipt_path = Path(receipt_value).expanduser()
    if not receipt_path.is_absolute():
        receipt_path = plan_path.parent / receipt_path
    try:
        receipt = json.loads(receipt_path.resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"formal launch-gate receipt is unreadable: {receipt_path}: {exc}") from exc
    if not isinstance(receipt, dict) or receipt.get("status") != "pass":
        raise SystemExit("formal launch-gate receipt must have status=pass")
    if receipt.get("formal_task_consumption_allowed") is not True:
        raise SystemExit("formal launch-gate receipt must explicitly permit task consumption")
    if receipt.get("contract_sha256") != contract_sha:
        raise SystemExit("formal plan contract_sha256 does not match launch-gate receipt")
    mode = assignment_mode(document)
    if mode == DYNAMIC_MODE and document.get("contract_version") != PLAN_CONTRACT_VERSION:
        if document.get("gpu_reassignment_allowed") is not True:
            raise SystemExit("historical formal dynamic plan must declare gpu_reassignment_allowed=true")
        if receipt.get("gpu_reassignment_allowed") is not True:
            raise SystemExit("historical formal dynamic launch-gate receipt must permit GPU reassignment")
    if mode == FIXED_MODE and document.get("contract_version") == PLAN_CONTRACT_VERSION:
        reason = document.get("fixed_gpu_binding_reason")
        if receipt.get("fixed_gpu_binding_reason") != reason:
            raise SystemExit("v2 fixed launch-gate receipt must preserve fixed_gpu_binding_reason")
    return {
        "basis": "hash-bound-launch-gate-receipt",
        "path": str(receipt_path.resolve()),
        "receipt_sha256": "sha256:" + hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "contract_sha256": contract_sha,
        "resource_assignment_mode": mode,
        "fixed_gpu_binding_reason": document.get("fixed_gpu_binding_reason"),
    }


def resolve_runtime_candidates(document: dict[str, Any], runtime_candidates: set[str]) -> set[str]:
    """Use the current parent standing-pool projection, never plan history."""
    del document
    candidates = {str(uuid) for uuid in runtime_candidates if str(uuid)}
    if not candidates:
        raise ValueError("current standing-pool runtime candidate set is empty")
    return candidates


def capacity_starvation_threshold_seconds(poll_seconds: float) -> float:
    return max(2 * poll_seconds, 60.0)


def capacity_starvation_event(
    *,
    candidate_uuid: str,
    pending_cell_count: int,
    eligible_since_monotonic: float,
    observed_monotonic: float,
    poll_seconds: float,
    assigned: bool,
    max_concurrent_cells: int | None,
    eligible_since_utc: str | None = None,
    observed_at_utc: str | None = None,
) -> dict[str, Any] | None:
    latency = round(max(0.0, observed_monotonic - eligible_since_monotonic), 3)
    if (
        pending_cell_count <= 0
        or assigned
        or max_concurrent_cells is not None
        or latency <= capacity_starvation_threshold_seconds(poll_seconds)
    ):
        return None
    return {
        "event": "capacity_starvation",
        "gpu_uuid": candidate_uuid,
        "pending_cell_count": pending_cell_count,
        "eligible_since_utc": eligible_since_utc,
        "observed_at_utc": observed_at_utc or utc_now(),
        "backfill_latency_seconds": latency,
    }


def gpu_snapshot(uuid: str) -> dict[str, Any]:
    return canonical_snapshot_gpu(uuid)


def idle(snapshot: dict[str, Any], *, max_memory_mib: int, max_util_percent: int) -> bool:
    return canonical_idle(snapshot, max_memory_mib=max_memory_mib, max_util_percent=max_util_percent)


def wait_for_idle(
    uuid: str,
    args: argparse.Namespace,
    *,
    work_remaining: Callable[[], bool] | None = None,
) -> dict[str, Any] | None:
    started = time.monotonic()
    while True:
        if work_remaining is not None and not work_remaining():
            return None
        snapshot = gpu_snapshot(uuid)
        if idle(snapshot, max_memory_mib=args.max_idle_memory_mib, max_util_percent=args.max_idle_util_percent):
            return snapshot
        if args.timeout_seconds and time.monotonic() - started >= args.timeout_seconds:
            raise TimeoutError(f"GPU did not become idle before timeout: {uuid}")
        time.sleep(args.poll_seconds)


def cell_command(cell: dict[str, Any], args: argparse.Namespace, cell_root: Path, cwd: Path) -> tuple[list[str], dict[str, str]]:
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in cell.get("env", {}).items()})
    command = cell["argv"]
    if cell.get("run_manifest"):
        manifest_path = Path(str(cell["run_manifest"])).expanduser().resolve()
        env["OPENEVO_RUN_MANIFEST"] = str(manifest_path)
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_with_manifest.py"),
            "--manifest",
            str(manifest_path),
            "--output-root",
            str(cell_root / "manifest-output"),
            "--cwd",
            str(cwd),
            "--",
            *cell["argv"],
        ]
    return command, env


def append_receipt(receipt: dict[str, Any], lock: threading.Lock, output_root: Path, row: dict[str, Any]) -> None:
    with lock:
        receipt["cells"].append(row)
        atomic_json(output_root / "scheduler-receipt.json", receipt)


def append_resource_event(receipt: dict[str, Any], lock: threading.Lock, output_root: Path, event: dict[str, Any]) -> None:
    with lock:
        receipt["resource_events"].append(event)
        atomic_json(output_root / "scheduler-receipt.json", receipt)


def run_cell(
    cell: dict[str, Any],
    gpu_uuid: str,
    args: argparse.Namespace,
    receipt: dict[str, Any],
    lock: threading.Lock,
    *,
    pre_gpu: dict[str, Any] | None = None,
) -> None:
    cell_id = cell["cell_id"]
    cell_root = args.output_root / cell_id
    cell_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    stdout_path = cell_root / "stdout.log"
    row: dict[str, Any] = {
        "cell_id": cell_id,
        "requested_gpu_uuid": cell.get("gpu_uuid"),
        "assigned_gpu_uuid": gpu_uuid,
        "argv": cell["argv"],
        "start_utc": utc_now(),
        "status": "waiting",
    }
    try:
        row["pre_gpu"] = pre_gpu or wait_for_idle(gpu_uuid, args)
        if row["pre_gpu"] is None:
            raise RuntimeError("resource lane ended before cell assignment")
        cwd = Path(cell.get("cwd", args.cwd or Path.cwd())).expanduser().resolve()
        row["cwd"] = str(cwd)
        command, env = cell_command(cell, args, cell_root, cwd)
        env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
        env["OPENEVO_SELECTED_GPU_UUIDS"] = gpu_uuid
        env["OPENEVO_SCHEDULER_CELL_ID"] = cell_id
        if cell.get("run_manifest"):
            row["run_manifest"] = str(Path(str(cell["run_manifest"])).expanduser().resolve())
        row["status"] = "running"
        with stdout_path.open("w", encoding="utf-8") as stream:
            process = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=args.cell_timeout_seconds or None,
                check=False,
                text=True,
            )
        row["returncode"] = process.returncode
        row["status"] = "pass" if process.returncode == 0 else "failed"
        row["post_gpu"] = gpu_snapshot(gpu_uuid)
        if not idle(row["post_gpu"], max_memory_mib=args.max_idle_memory_mib, max_util_percent=args.max_idle_util_percent):
            row["status"] = "failed_gpu_release"
    except Exception as exc:
        row["status"] = "blocked" if isinstance(exc, (TimeoutError, RuntimeError)) else "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["end_utc"] = utc_now()
    row["stdout"] = str(stdout_path)
    append_receipt(receipt, lock, args.output_root, row)


def run_fixed(
    cells: list[dict[str, Any]],
    args: argparse.Namespace,
    receipt: dict[str, Any],
    lock: threading.Lock,
) -> None:
    lanes: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        lanes.setdefault(str(cell["gpu_uuid"]), []).append(cell)

    def run_lane(uuid: str, lane: list[dict[str, Any]]) -> None:
        for cell in lane:
            run_cell(cell, uuid, args, receipt, lock)

    with ThreadPoolExecutor(max_workers=len(lanes) or 1) as executor:
        futures = [executor.submit(run_lane, uuid, lane) for uuid, lane in lanes.items()]
        for future in futures:
            future.result()


def run_dynamic(
    cells: list[dict[str, Any]],
    runtime_candidates: set[str],
    args: argparse.Namespace,
    receipt: dict[str, Any],
    lock: threading.Lock,
) -> None:
    pending: queue.Queue[dict[str, Any]] = queue.Queue()
    for cell in cells:
        pending.put(cell)

    def work_remaining() -> bool:
        return not pending.empty()

    deadline = time.monotonic() + args.timeout_seconds if args.timeout_seconds else None
    max_concurrent_cells = getattr(args, "max_concurrent_cells", None)
    if max_concurrent_cells is not None and max_concurrent_cells < 1:
        raise ValueError("max_concurrent_cells must be positive when set")
    sorted_candidates = sorted(runtime_candidates)
    concurrency_slots = threading.BoundedSemaphore(max_concurrent_cells) if max_concurrent_cells else None

    def worker(uuid: str) -> None:
        eligible_since_monotonic: float | None = None
        eligible_since_utc: str | None = None
        while work_remaining():
            if deadline is not None and time.monotonic() >= deadline:
                append_resource_event(
                    receipt,
                    lock,
                    args.output_root,
                    {"uuid": uuid, "status": "unavailable", "error": "scheduler timeout", "observed_at_utc": utc_now()},
                )
                return
            probe_args = argparse.Namespace(**vars(args))
            probe_args.timeout_seconds = max(args.poll_seconds, 0.001)
            try:
                pre_gpu = wait_for_idle(uuid, probe_args, work_remaining=work_remaining)
            except (TimeoutError, RuntimeError) as exc:
                append_resource_event(
                    receipt,
                    lock,
                    args.output_root,
                    {"uuid": uuid, "status": "poll_retry", "error": f"{type(exc).__name__}: {exc}", "observed_at_utc": utc_now()},
                )
                if deadline is not None and time.monotonic() >= deadline:
                    return
                time.sleep(args.poll_seconds)
                continue
            if pre_gpu is None:
                return
            if eligible_since_monotonic is None:
                eligible_since_monotonic = time.monotonic()
                eligible_since_utc = pre_gpu.get("observed_at_utc") or utc_now()
            try:
                cell = pending.get_nowait()
            except queue.Empty:
                return

            try:
                slot_acquired = False
                if concurrency_slots is not None and not concurrency_slots.acquire(blocking=False):
                    pending.put(cell)
                    time.sleep(args.poll_seconds)
                    continue
                slot_acquired = concurrency_slots is not None
                immediate = gpu_snapshot(uuid)
                if not idle(immediate, max_memory_mib=args.max_idle_memory_mib, max_util_percent=args.max_idle_util_percent):
                    pending.put(cell)
                    append_resource_event(
                    receipt,
                    lock,
                    args.output_root,
                        {
                            "uuid": uuid,
                            "cell_id": cell["cell_id"],
                            "status": "prelaunch-race-requeued",
                            "pending_cell_count": pending.qsize() + 1,
                            "observed_at_utc": utc_now(),
                        },
                    )
                    starvation = capacity_starvation_event(
                        candidate_uuid=uuid,
                        pending_cell_count=pending.qsize(),
                        eligible_since_monotonic=eligible_since_monotonic,
                        observed_monotonic=time.monotonic(),
                        poll_seconds=args.poll_seconds,
                        assigned=False,
                        max_concurrent_cells=max_concurrent_cells,
                        eligible_since_utc=eligible_since_utc,
                    )
                    if starvation is not None:
                        append_resource_event(receipt, lock, args.output_root, starvation)
                    time.sleep(args.poll_seconds)
                    continue
                assigned_at = utc_now()
                append_resource_event(
                    receipt,
                    lock,
                    args.output_root,
                    {
                        "uuid": uuid,
                        "cell_id": cell["cell_id"],
                        "status": "assigned",
                        "pending_cell_count": pending.qsize(),
                        "eligible_since_utc": eligible_since_utc,
                        "assigned_at_utc": assigned_at,
                        "backfill_latency_seconds": round(time.monotonic() - eligible_since_monotonic, 3),
                    },
                )
                run_cell(cell, uuid, args, receipt, lock, pre_gpu=immediate)
                eligible_since_monotonic = None
                eligible_since_utc = None
            finally:
                if slot_acquired and concurrency_slots is not None:
                    concurrency_slots.release()
                pending.task_done()

    with ThreadPoolExecutor(max_workers=len(sorted_candidates) or 1) as executor:
        futures = [executor.submit(worker, uuid) for uuid in sorted_candidates]
        for future in futures:
            future.result()

    while True:
        try:
            cell = pending.get_nowait()
        except queue.Empty:
            break
        append_receipt(
            receipt,
            lock,
            args.output_root,
            {
                "cell_id": cell["cell_id"],
                "requested_gpu_uuid": None,
                "assigned_gpu_uuid": None,
                "argv": cell["argv"],
                "start_utc": None,
                "end_utc": utc_now(),
                "status": "blocked_no_live_idle_gpu",
                "error": "no runtime candidate lane remained available",
            },
        )
        pending.task_done()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--runtime-gpu-uuids",
        "--authorized-gpu-uuids",
        dest="runtime_gpu_uuids",
        default=os.environ.get("OPENEVO_AUTHORIZED_GPU_UUIDS", ""),
        help="runtime projection/candidate UUID set from the parent standing pool; --authorized-gpu-uuids is a compatibility alias",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-concurrent-cells", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=0.0)
    parser.add_argument("--cell-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--max-idle-memory-mib", type=int, default=DEFAULT_MAX_IDLE_MEMORY_MIB)
    parser.add_argument("--max-idle-util-percent", type=int, default=DEFAULT_MAX_IDLE_UTIL_PERCENT)
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-formal",
        action="store_true",
        help="deprecated compatibility flag; formal permission is derived from the hash-bound launch-gate receipt",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan_path = args.plan.expanduser().resolve()
    document, cells = load_plan(plan_path)
    mode = assignment_mode(document)
    try:
        runtime_candidates = resolve_runtime_candidates(document, parse_uuid_list(args.runtime_gpu_uuids))
    except ValueError as exc:
        raise SystemExit("OPENEVO_AUTHORIZED_GPU_UUIDS / --runtime-gpu-uuids must provide the current in-pool runtime candidate UUIDs") from exc

    if mode == FIXED_MODE:
        outside_runtime_pool = sorted({str(cell["gpu_uuid"]) for cell in cells} - runtime_candidates)
        if outside_runtime_pool:
            raise SystemExit(f"fixed plan contains GPU UUIDs outside the current runtime candidate set: {', '.join(outside_runtime_pool)}")

    formal = document.get("formal_task_consumption_allowed") is True
    if formal:
        missing_manifest = [cell["cell_id"] for cell in cells if not cell.get("run_manifest")]
        if missing_manifest:
            raise SystemExit("formal plan cells must declare run_manifest: " + ", ".join(missing_manifest))
        formal_receipt = validate_formal_launch_gate(document, plan_path)
    else:
        formal_receipt = None

    if args.poll_seconds <= 0 or args.timeout_seconds < 0 or args.cell_timeout_seconds < 0:
        raise SystemExit("poll/timeout values are invalid")
    if args.max_concurrent_cells is not None and args.max_concurrent_cells < 1:
        raise SystemExit("max-concurrent-cells must be positive")
    if args.max_idle_memory_mib < 0 or args.max_idle_util_percent < 0:
        raise SystemExit("idle thresholds must be non-negative")

    args.output_root = args.output_root.expanduser().resolve()
    if args.dry_run:
        print(json.dumps({
            "status": "dry-run",
            "resource_assignment_mode": mode,
            "runtime_candidate_uuids": sorted(runtime_candidates),
            "formal_authorization": formal_receipt,
            "cells": cells,
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    claim_output_root(args.output_root)

    receipt: dict[str, Any] = {
        "schema": "openevo.gpu-scheduler-receipt.v2",
        "plan": str(plan_path),
        "started_utc": utc_now(),
        "resource_assignment_mode": mode,
        "runtime_candidate_uuids": sorted(runtime_candidates),
        "compat_authorized_gpu_uuids_semantics": "runtime projection of parent standing pool; not per-campaign human approval",
        "max_idle_memory_mib": args.max_idle_memory_mib,
        "max_idle_util_percent": args.max_idle_util_percent,
        "poll_seconds": args.poll_seconds,
        "max_concurrent_cells": args.max_concurrent_cells,
        "capacity_starvation_threshold_seconds": capacity_starvation_threshold_seconds(args.poll_seconds),
        "pending_cell_count": len(cells),
        "cells": [],
        "resource_events": [],
        "formal_task_consumption_allowed": formal,
        "formal_launch_gate": formal_receipt,
        "deprecated_allow_formal_flag_present": bool(args.allow_formal),
    }
    lock = threading.Lock()
    atomic_json(args.output_root / "scheduler-receipt.json", receipt)

    if mode == FIXED_MODE:
        run_fixed(cells, args, receipt, lock)
    else:
        run_dynamic(cells, runtime_candidates, args, receipt, lock)

    receipt["ended_utc"] = utc_now()
    atomic_json(args.output_root / "scheduler-receipt.json", receipt)
    failed = [row for row in receipt["cells"] if row["status"] != "pass"]
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
