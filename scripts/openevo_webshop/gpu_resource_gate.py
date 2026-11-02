#!/usr/bin/env python3
"""Shared fail-closed GPU eligibility semantics for OpenEvo launchers.

The legacy ``snapshot`` CLI/API remains available for historical receipts. New
launchers use the same ``idle`` predicate and richer live snapshot helpers.
"""

from __future__ import annotations

import os
import pwd
import re
import shutil
import subprocess
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


GPU_UUID_RE = re.compile(r"^GPU-[A-Za-z0-9-]+$")
DEFAULT_MAX_IDLE_MEMORY_MIB = 16
DEFAULT_MAX_IDLE_UTIL_PERCENT = 0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_gpu_rows(text: str) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) not in {4, 5} or not GPU_UUID_RE.fullmatch(fields[1]):
            raise ValueError(f"unexpected nvidia-smi GPU row: {line}")
        if len(fields) == 4:
            index, uuid, memory_used, utilization = fields
            cards.append({
                "index": int(index),
                "uuid": uuid,
                "memory_used_mib": int(memory_used),
                "utilization_gpu_percent": int(utilization),
            })
        else:
            index, uuid, memory_used, memory_total, utilization = fields
            cards.append({
                "index": int(index),
                "uuid": uuid,
                "memory_used_mib": int(memory_used),
                "memory_total_mib": int(memory_total),
                "utilization_gpu_percent": int(utilization),
            })
    return cards


def process_identity(proc: Path) -> dict[str, str]:
    """Return bounded, non-secret process identity evidence for a device FD."""
    try:
        uid = proc.stat().st_uid
        user = pwd.getpwuid(uid).pw_name
        comm = (proc / "comm").read_text(encoding="utf-8").strip()
        exe = os.readlink(proc / "exe")
        command_parts = (proc / "cmdline").read_bytes().split(b"\0")
        command_programs = sorted({Path(os.fsdecode(part)).name for part in command_parts if part})
        status = (proc / "status").read_text(encoding="utf-8")
        parent_pid = next(line.split()[1] for line in status.splitlines() if line.startswith("PPid:"))
        cgroup = (proc / "cgroup").read_text(encoding="utf-8").strip()
    except (KeyError, OSError, StopIteration, UnicodeDecodeError) as exc:
        return {"classification_error": f"{type(exc).__name__}: {exc}"}
    return {
        "user": user,
        "comm": comm,
        "exe": exe,
        "cmdline_programs": ",".join(command_programs),
        "parent_pid": parent_pid,
        "cgroup": cgroup,
    }


def visible_device_handles(index: int) -> list[dict[str, Any]]:
    device = f"/dev/nvidia{index}"
    handles: list[dict[str, Any]] = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            fds = list((proc / "fd").iterdir())
        except (FileNotFoundError, PermissionError):
            continue
        for fd in fds:
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if target == device:
                handles.append({"pid": proc.name, "fd": fd.name, "device": target, **process_identity(proc)})
    return handles


def is_trusted_monitor_handle(handle: Mapping[str, Any]) -> bool:
    """Only a verifiable root nvitop handle is monitoring-only."""
    if handle.get("classification_error"):
        return False
    programs = {value for value in str(handle.get("cmdline_programs", "")).split(",") if value}
    return (
        handle.get("user") == "root"
        and handle.get("comm") == "nvitop"
        and (Path(str(handle.get("exe", ""))).name == "nvitop" or "nvitop" in programs)
    )


def classify_device_handles(handles: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    monitoring: list[dict[str, Any]] = []
    blocking: list[dict[str, Any]] = []
    classification_failed = False
    for handle in handles:
        rendered = dict(handle)
        if rendered.get("classification_error"):
            classification_failed = True
            blocking.append(rendered)
        elif is_trusted_monitor_handle(rendered):
            monitoring.append(rendered)
        else:
            blocking.append(rendered)
    return monitoring, blocking, classification_failed


def idle(snapshot: Mapping[str, Any], *, max_memory_mib: int, max_util_percent: int) -> bool:
    """Strict, shared, fail-closed eligibility predicate."""
    try:
        memory = int(snapshot["memory_used_mib"])
        utilization_value = snapshot.get("utilization_percent")
        if utilization_value is None:
            utilization_value = snapshot["utilization_gpu_percent"]
        utilization = int(utilization_value)
    except (KeyError, TypeError, ValueError):
        return False
    if memory > max_memory_mib or utilization > max_util_percent:
        return False
    if snapshot.get("compute_apps_query_error"):
        return False
    compute_apps = snapshot.get("compute_apps")
    if compute_apps is not None and (not isinstance(compute_apps, list) or compute_apps):
        return False
    handles = snapshot.get("device_handles")
    if handles is not None:
        _, blocking, _ = classify_device_handles(handles)
        if blocking:
            return False
    return True


def running_gpu_container_owners() -> tuple[dict[str, list[dict[str, str]]], str | None]:
    """Map explicit Docker GPU UUID requests to running container identities.

    Device visibility without an explicit UUID request is deliberately not
    treated as ownership. The returned identities omit environment variables
    and other inspect data that may contain credentials.
    """
    if shutil.which("docker") is None:
        return {}, "docker is unavailable"
    listed = subprocess.run(
        ["docker", "ps", "--no-trunc", "--filter", "status=running", "--format", "{{.ID}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if listed.returncode != 0:
        return {}, listed.stderr.strip() or "docker ps failed"
    container_ids = [line.strip() for line in listed.stdout.splitlines() if line.strip()]
    if not container_ids:
        return {}, None
    inspected = subprocess.run(
        ["docker", "inspect", *container_ids],
        capture_output=True,
        text=True,
        check=False,
    )
    if inspected.returncode != 0:
        return {}, inspected.stderr.strip() or "docker inspect failed"
    try:
        containers = json.loads(inspected.stdout)
    except json.JSONDecodeError as exc:
        return {}, f"docker inspect returned invalid JSON: {type(exc).__name__}"
    owners: dict[str, list[dict[str, str]]] = {}
    for container in containers:
        container_id = str(container.get("Id") or "")
        name = str(container.get("Name") or "").lstrip("/")
        image = str(container.get("Config", {}).get("Image") or "")
        requests = container.get("HostConfig", {}).get("DeviceRequests") or []
        for request in requests:
            device_ids = request.get("DeviceIDs") or []
            for uuid in device_ids:
                if not isinstance(uuid, str) or not GPU_UUID_RE.fullmatch(uuid):
                    continue
                owners.setdefault(uuid, []).append({
                    "container_id": container_id[:12],
                    "container_name": name,
                    "image": image,
                })
    return owners, None


def co_resident_capacity(
    snapshot: Mapping[str, Any],
    *,
    minimum_free_memory_mib: int,
    max_util_percent: int = 100,
) -> tuple[bool, list[str]]:
    """Allow explicitly authorized co-residency after live capacity/owner checks."""
    failures: list[str] = []
    try:
        used = int(snapshot["memory_used_mib"])
        total = int(snapshot["memory_total_mib"])
        utilization_value = snapshot.get("utilization_percent")
        if utilization_value is None:
            utilization_value = snapshot["utilization_gpu_percent"]
        utilization = int(utilization_value)
    except (KeyError, TypeError, ValueError):
        return False, ["capacity telemetry is incomplete"]
    free = total - used
    if free < minimum_free_memory_mib:
        failures.append(f"free-memory={free}MiB<required={minimum_free_memory_mib}MiB")
    if utilization > max_util_percent:
        failures.append(f"utilization={utilization}%>limit={max_util_percent}%")
    if snapshot.get("compute_apps_query_error"):
        failures.append("compute-app-query-failed")
    apps = [str(line).strip() for line in (snapshot.get("compute_apps") or []) if str(line).strip()]
    app_pids: list[str] = []
    for line in apps:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2 or not fields[0].isdigit() or not fields[1]:
            failures.append("compute-app-owner-unresolved")
            continue
        app_pids.append(fields[0])
    handles = snapshot.get("device_handles") or []
    _, blocking, classification_failed = classify_device_handles(handles)
    if classification_failed:
        failures.append("device-handle-owner-unresolved")
    for handle in blocking:
        required = ("pid", "user", "comm", "exe", "cgroup")
        if any(not str(handle.get(key) or "").strip() for key in required):
            failures.append("device-handle-owner-unresolved")
            break
    owners = snapshot.get("gpu_owner_containers") or []
    owner_query_error = snapshot.get("gpu_owner_query_error")
    resolved_handle_pids = {
        str(handle.get("pid"))
        for handle in blocking
        if all(str(handle.get(key) or "").strip() for key in ("pid", "user", "comm", "exe", "cgroup"))
    }
    unresolved_apps = [pid for pid in app_pids if pid not in resolved_handle_pids]
    if unresolved_apps and not owners:
        failures.append("compute-app-owner-unresolved")
    if used > 16 and not app_pids and not blocking and not owners:
        failures.append("occupied-memory-owner-unresolved")
    if owner_query_error and used > 16 and not owners and not blocking and not app_pids:
        failures.append("docker-owner-query-failed")
    if used > 16 and not app_pids and not blocking and owner_query_error:
        failures.append("occupied-memory-owner-unresolved")
    return not failures, failures


def snapshot_gpu(uuid: str) -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi is unavailable")
    query = subprocess.run(
        ["nvidia-smi", "-i", uuid, "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if query.returncode != 0 or not query.stdout.strip():
        raise RuntimeError(query.stderr.strip() or f"GPU is not visible: {uuid}")
    rows = parse_gpu_rows(query.stdout)
    if len(rows) != 1 or rows[0]["uuid"] != uuid:
        raise RuntimeError(f"GPU identity mismatch for {uuid}: {query.stdout.strip()}")
    apps = subprocess.run(
        ["nvidia-smi", "-i", uuid, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if apps.returncode != 0:
        raise RuntimeError(apps.stderr.strip() or f"could not query compute apps: {uuid}")
    row = rows[0]
    row["utilization_percent"] = row.pop("utilization_gpu_percent")
    row["compute_apps"] = [line for line in apps.stdout.splitlines() if line.strip()]
    row["device_handles"] = visible_device_handles(int(row["index"]))
    row["monitoring_handles"], row["blocking_device_handles"], _ = classify_device_handles(row["device_handles"])
    row["observed_at_utc"] = utc_now()
    return row


# Historical API retained for old receipts/tests and standalone gate users.
def _run(args: list[str]) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def _gpu_row(uuid: str) -> dict[str, Any]:
    raw = _run([
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    for row in parse_gpu_rows(raw):
        if row["uuid"] == uuid and "memory_total_mib" in row:
            return row
    raise RuntimeError(f"GPU UUID not found: {uuid}")


def _compute_apps(uuid: str) -> list[str]:
    raw = _run([
        "nvidia-smi",
        "-i",
        uuid,
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ])
    return [line for line in raw.splitlines() if line.strip()]


def _device_handles(index: int) -> list[dict[str, Any]]:
    return visible_device_handles(index)


def snapshot(uuids: list[str], *, max_memory_mib: int = DEFAULT_MAX_IDLE_MEMORY_MIB) -> dict[str, Any]:
    cards = []
    for uuid in uuids:
        row = _gpu_row(uuid)
        apps = _compute_apps(uuid)
        handles = _device_handles(row["index"])
        card = {
            **row,
            "compute_apps": apps,
            "device_handles": handles,
            "eligible": idle(
                {**row, "compute_apps": apps, "device_handles": handles},
                max_memory_mib=max_memory_mib,
                max_util_percent=DEFAULT_MAX_IDLE_UTIL_PERCENT,
            ),
        }
        cards.append(card)
    return {
        "schema": "seed3090.webshop.gpu-resource-gate.v1",
        "observation_scope": "caller PID namespace; not a physical-host ownership audit",
        "ownership_attribution": "not attempted; this gate decides eligibility only",
        "hostname": _run(["hostname"]),
        "user": _run(["whoami"]),
        "observed_at_epoch": time.time(),
        "max_memory_mib": max_memory_mib,
        "cards": cards,
        "all_eligible": all(card["eligible"] for card in cards),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uuid", action="append", required=True, dest="uuids")
    parser.add_argument("--max-memory-mib", type=int, default=DEFAULT_MAX_IDLE_MEMORY_MIB)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = snapshot(args.uuids, max_memory_mib=args.max_memory_mib)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["all_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
