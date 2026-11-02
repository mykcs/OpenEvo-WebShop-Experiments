"""Read-only witness for formal-first opportunistic resource gaps.

The witness reuses the exact live formal runtime to ask which rollout requests are
pending, and reads the existing Ray placement state under its lock.  It never writes a
request, reserves a GPU, launches a process, or grants canary execution authority.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

from scripts.openevo_webshop.opportunistic_canary_guard import WITNESS_SCHEMA


class FormalPriorityWitnessError(RuntimeError):
    """The live formal resource witness cannot be proven read-only and exact."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalPriorityWitnessError(f"unreadable JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise FormalPriorityWitnessError(f"JSON root must be an object: {path}")
    return value


def _clean_git_head(repo: Path) -> str:
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True)
    if dirty.strip():
        raise FormalPriorityWitnessError("formal execution checkout is not clean")
    return head


def _probe_pending_rollouts(formal_repo: Path, ready: Path, release: Path) -> list[dict[str, Any]]:
    code = r'''
import json, sys
from pathlib import Path
from scripts.openevo_webshop.bounded_effective_state_async_launch_guard_20260916 import validate_launch_pair_async
from scripts.openevo_webshop.bounded_effective_state_async_runtime_20260916 import pending_rollout_requests
repo, ready, release = map(Path, sys.argv[1:4])
package, _ = validate_launch_pair_async(repo, ready, release)
rows = pending_rollout_requests(package, ready, release)
print(json.dumps(rows, sort_keys=True))
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(formal_repo)
    completed = subprocess.run(
        [sys.executable, "-c", code, str(formal_repo), str(ready), str(release)],
        cwd=formal_repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise FormalPriorityWitnessError(f"exact formal pending-request probe failed: {completed.stderr.strip()}")
    try:
        rows = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise FormalPriorityWitnessError("formal pending-request probe returned invalid JSON") from exc
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise FormalPriorityWitnessError("formal pending-request probe returned invalid rows")
    return rows


def _read_ray_state_snapshot(state_root: Path) -> dict[str, Any]:
    lock_path = state_root / "state.lock"
    state_path = state_root / "state.json"
    if not lock_path.is_file() or not state_path.is_file():
        raise FormalPriorityWitnessError("Ray placement state/lock missing")
    with lock_path.open("r", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_SH)
        try:
            state = _read_object(state_path)
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
    if state.get("schema") != "openevo.ceiling1-ray-placement-state.202609042025":
        raise FormalPriorityWitnessError("Ray placement state schema drift")
    return state


def _lease_rows(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    gpus = state.get("gpus")
    if not isinstance(gpus, Mapping):
        raise FormalPriorityWitnessError("Ray placement state missing GPU map")
    for row in gpus.values():
        if not isinstance(row, Mapping):
            raise FormalPriorityWitnessError("Ray GPU state row malformed")
        leases = row.get("leases") or []
        if not isinstance(leases, list):
            raise FormalPriorityWitnessError("Ray GPU lease list malformed")
        for lease in leases:
            if not isinstance(lease, dict) or not isinstance(lease.get("lease_id"), str):
                raise FormalPriorityWitnessError("Ray lease identity malformed")
            previous = unique.get(lease["lease_id"])
            if previous is not None and previous != lease:
                raise FormalPriorityWitnessError("same Ray lease ID has conflicting payloads")
            unique[lease["lease_id"]] = lease
    return [unique[key] for key in sorted(unique)]


def _final_panel_count(package: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    arms = package.get("arms")
    if not isinstance(arms, Mapping):
        raise FormalPriorityWitnessError("formal READY package missing arms")
    total = 0
    states: dict[str, Any] = {}
    for arm_key in ("off", "on"):
        arm = arms.get(arm_key)
        if not isinstance(arm, Mapping) or not isinstance(arm.get("output_root"), str):
            raise FormalPriorityWitnessError(f"formal READY package missing {arm_key} output root")
        state = _read_object(Path(arm["output_root"]) / "execution-state.json")
        count = state.get("final_panel_access_count")
        if type(count) is not int or count < 0:
            raise FormalPriorityWitnessError(f"{arm_key} final-panel counter invalid")
        total += count
        states[arm_key] = {
            "status": state.get("status"),
            "next_round": state.get("next_round"),
            "rollouts_consumed": state.get("rollouts_consumed"),
            "final_panel_access_count": count,
        }
    return total, states


def build_formal_priority_witness(
    *,
    formal_repo: Path,
    ready_path: Path,
    release_path: Path,
    ray_state_root: Path,
    expected_execution_git_sha: str,
    topology_id: str,
    candidate_gpu_indices: Sequence[int],
    head_reader: Callable[[Path], str] = _clean_git_head,
    pending_probe: Callable[[Path, Path, Path], list[dict[str, Any]]] = _probe_pending_rollouts,
    now_epoch: float | None = None,
) -> dict[str, Any]:
    formal_repo = formal_repo.resolve()
    ready_path = ready_path.resolve()
    release_path = release_path.resolve()
    ray_state_root = ray_state_root.resolve()
    source_head = head_reader(formal_repo)
    if source_head != expected_execution_git_sha:
        raise FormalPriorityWitnessError("formal execution checkout HEAD drift")

    package = _read_object(ready_path)
    release = _read_object(release_path)
    if package.get("status") != "FORMAL_READY_AWAITING_OWNER_LAUNCH":
        raise FormalPriorityWitnessError("formal READY status drift")
    if release.get("status") != "AUTHORIZED_FORMAL_LAUNCH":
        raise FormalPriorityWitnessError("formal launch release status drift")
    if package.get("execution_git_sha") != expected_execution_git_sha or release.get("execution_git_sha") != expected_execution_git_sha:
        raise FormalPriorityWitnessError("READY/release execution identity drift")

    before = pending_probe(formal_repo, ready_path, release_path)
    ray_state = _read_ray_state_snapshot(ray_state_root)
    after = pending_probe(formal_repo, ready_path, release_path)
    pending_stable = before == after
    leases = _lease_rows(ray_state)
    reservations = ray_state.get("reservations") or {}
    if not isinstance(reservations, Mapping):
        raise FormalPriorityWitnessError("Ray reservation map malformed")
    final_panel_access_count, arm_states = _final_panel_count(package)

    if not pending_stable:
        scheduler_status = "TRANSITIONING"
    elif leases or reservations:
        scheduler_status = "ACTIVE"
    elif after:
        scheduler_status = "PENDING"
    else:
        scheduler_status = "IDLE"

    candidates = list(candidate_gpu_indices)
    if not candidates or any(type(index) is not int for index in candidates) or len(candidates) != len(set(candidates)):
        raise FormalPriorityWitnessError("candidate GPU indices must be unique integers")
    observed = time.time() if now_epoch is None else float(now_epoch)
    witness: dict[str, Any] = {
        "schema": WITNESS_SCHEMA,
        "execution_git_sha": expected_execution_git_sha,
        "topology_id": topology_id,
        "candidate_gpu_indices": candidates,
        "observed_at_epoch": observed,
        "scheduler_status": scheduler_status,
        "pending_formal_gpu_requests": len(after),
        "pending_probe_stable": pending_stable,
        "active_formal_reservations": len(reservations),
        "active_formal_gpu_leases": len(leases),
        "active_formal_gpu_lease_ids": [lease["lease_id"] for lease in leases],
        "formal_arm_states": arm_states,
        "final_panel_access_count": final_panel_access_count,
        "formal_work_may_wait_for_canary": False,
        "read_only": True,
        "gpu_requests": 0,
        "formal_rows_consumed": 0,
        "model_calls": 0,
        "canary_execution_authority": False,
        "scientific_authority": False,
        "resource_topology_authority": False,
    }
    witness["witness_identity"] = hashlib.sha256(_canonical_bytes(witness)).hexdigest()
    return witness
