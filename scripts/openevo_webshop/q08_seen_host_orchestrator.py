#!/usr/bin/env python3
"""Canonical host runner for one Q08 run's ten fixed seen-checkpoint cells."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

from q08_seen_launch_guard import (
    CELL_IDS, CURRENT, MATRIX_SHA, PANEL_SHA, PROFILE, RELEASE, REUSE_SHA,
    ROUTER, _load_static_chain, bound_path, read_json, sha256_file,
)


def _canonical_hash(value: dict) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(data).hexdigest()


def _normalize_sha(value: object) -> str:
    if not isinstance(value, str):
        raise RuntimeError("expected SHA-256 text")
    normalized = value.removeprefix("sha256:")
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise RuntimeError("malformed SHA-256 reference")
    return normalized


def _load_formal_start(path: Path, profile: dict, selected: list[str], release: dict, repo: Path):
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"canonical FormalStart receipt is missing or unsafe: {path}")
    active_profile, active_release, active_router, active_current = _load_static_chain(repo)
    if active_profile != profile or active_release != release:
        raise RuntimeError("Q08 FormalStart inputs differ from the active static authority chain")
    release_path = bound_path(repo, RELEASE)
    router_path = bound_path(repo, ROUTER)
    current_path = bound_path(repo, CURRENT)
    release_sha = sha256_file(release_path)
    router_sha = sha256_file(router_path)
    # The candidate current file binds the active router and release as a
    # second, independently checked edge in the authority chain.
    if (active_router.get("release") != {"path": str(release_path.relative_to(repo.resolve())), "sha256": release_sha}
            or active_current.get("router") != {"path": str(router_path.relative_to(repo.resolve())), "sha256": router_sha}
            or active_current.get("release") != {"path": str(release_path.relative_to(repo.resolve())), "sha256": release_sha}):
        raise RuntimeError("Q08 router/current/release hash chain is inconsistent")
    start = read_json(path)
    projection = {key: value for key, value in start.items() if key != "receipt_sha256"}
    if (start.get("status") != "FORMAL_START_COMMITTED"
            or start.get("formal_task_consumption_before") is not False
            or start.get("formal_task_consumption_may_begin") is not True
            or start.get("receipt_sha256") != "sha256:" + _canonical_hash(projection)
            or start.get("campaign_id") != profile["campaign_id"]
            or start.get("run_id") != profile["run_id"]
            or start.get("selected_gpu_uuids") != selected
            or start.get("image_id") != profile["image_id"]):
        raise RuntimeError("canonical FormalStart scope/identity mismatch")
    expected_run_root = Path(profile["workspace"]) / profile["paths"]["runs"] / profile["run_id"]
    if Path(str(start.get("output_root") or "")).resolve() != expected_run_root.resolve():
        raise RuntimeError("canonical FormalStart output root differs from the fixed Q08 run root")
    readiness = start.get("readiness") or {}
    ready_ref = readiness.get("ready_receipt") or {}
    ready_path = Path(str(ready_ref.get("path") or ""))
    if not ready_path.is_file() or ready_path.is_symlink() or _normalize_sha(ready_ref.get("sha256")) != sha256_file(ready_path):
        raise RuntimeError("FormalStart READY receipt reference is invalid")
    ready = read_json(ready_path)
    if ready.get("verdict") != "READY":
        raise RuntimeError("FormalStart does not contain a READY receipt")
    authority = ready.get("authority") or {}
    if (authority.get("router_path") != str(current_path)
            or authority.get("router_release_pointer") != RELEASE
            or authority.get("router_status") != "launch-authorized"
            or authority.get("campaign_id") != profile["campaign_id"]):
        raise RuntimeError("READY receipt did not qualify the active exact Q08 activation router/release")
    release_ref = ready.get("release_receipt") or {}
    if (release_ref.get("path") != str(release_path)
            or _normalize_sha(release_ref.get("sha256")) != release_sha):
        raise RuntimeError("READY receipt release bytes differ from the active exact Q08 release")
    profile_ref = readiness.get("profile") or {}
    profile_path = bound_path(repo, PROFILE)
    if profile_ref.get("path") != str(profile_path) or _normalize_sha(profile_ref.get("sha256")) != sha256_file(profile_path):
        raise RuntimeError("FormalStart profile reference differs from Q08 authority profile")
    ready_profile = ready.get("profile") or {}
    if ready_profile.get("path") != str(profile_path) or _normalize_sha(ready_profile.get("sha256")) != sha256_file(profile_path):
        raise RuntimeError("READY receipt profile identity differs from Q08 authority profile")
    if release.get("status") != "launch-authorized":
        raise RuntimeError("Q08 release is not launch-authorized")
    from q08_seen_launch_guard import validate_engineering_recovery
    recovery = validate_engineering_recovery(repo, profile, release)
    if recovery is not None:
        expected_basename=(recovery.get("continuation") or {}).get("formal_start_basename")
        if path.name != expected_basename:
            raise RuntimeError("Q08 engineering recovery permits only its exact canonical FormalStart attempt")
        recovery_path = bound_path(repo, str(release["engineering_recovery"]["path"]))
        recovery_sha = sha256_file(recovery_path)
        q = read_json(Path(str(profile["frozen_inputs"]["qualification_path"])))
        if q.get("q08_binding", {}).get("engineering_recovery_sha256") != recovery_sha:
            raise RuntimeError("Q08 qualification does not bind the exact zero-consumption recovery receipt")
    # `release.engineering_recovery` is a descriptor (path, SHA, attempt),
    # while the validator returns the authenticated recovery document that
    # owns continuation and admission semantics. Keep that validated value
    # with FormalStart so execute() never treats the descriptor as proof.
    return start, ready, recovery


def _require_uid1001_dir(path: Path, *, create: bool) -> None:
    if path.is_symlink():
        raise RuntimeError(f"Q08 output/control directory must not be a symlink: {path}")
    if not path.exists() and create:
        missing = []
        cursor = path
        while not cursor.exists():
            if cursor.is_symlink():
                raise RuntimeError(f"Q08 output/control ancestor must not be a symlink: {cursor}")
            missing.append(cursor)
            cursor = cursor.parent
        for directory in reversed(missing):
            directory.mkdir(mode=0o750, exist_ok=False)
            os.chown(directory, 1001, 1001)
            os.chmod(directory, 0o750)
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != 1001 or info.st_gid != 1001
            or stat.S_IMODE(info.st_mode) not in (0o750, 0o700)):
        raise RuntimeError(f"Q08 output/control directory must be owned by UID/GID 1001 with a frozen private mode: {path}")


def _atomic_once(path: Path, value: dict) -> None:
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        os.fchown(fd, 1001, 1001)
        os.fchmod(fd, 0o640)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(tmp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _docker_argv(profile: dict, repo: Path, cell: str, selected: str, context_path: Path, qualification_path: Path) -> list[str]:
    lineage, rounds = cell.rsplit("-r", 1)
    run_root = Path(profile["workspace"]) / profile["paths"]["runs"] / profile["run_id"]
    output_parent = run_root / "cells" / cell
    output_leaf = output_parent / "seen-eval"
    claim = run_root / "control" / "claims" / f"{cell}.claim"
    frozen = profile["frozen_inputs"]
    source_a = Path(frozen["operational_source"]["path"]).resolve()
    bind = f"type=bind,src={Path(profile['workspace']).resolve()},dst={Path(profile['workspace']).resolve()},readonly"
    control_attempt = int(context_path.parent.parent.name.removeprefix("attempt-"))
    container_name = _container_name(profile, cell, control_attempt)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", container_name):
        raise RuntimeError("Q08 container name is not a safe fixed identifier")
    command = [
        "docker", "run", "--name", container_name, "--read-only", "--security-opt", "no-new-privileges",
        "--cap-drop", "ALL", "--pids-limit", "4096", "--shm-size", "8g",
        "--tmpfs", profile["container"]["tmpfs"],
        "--gpus", f"device={selected}", "--user", "1001:1001", "--network", "none",
        "--mount", bind,
        "--mount", f"type=bind,src={output_parent.resolve()},dst={output_parent.resolve()}",
        "--mount", f"type=bind,src={(run_root / 'control' / 'claims').resolve()},dst={(run_root / 'control' / 'claims').resolve()}",
        "--env", f"CUDA_VISIBLE_DEVICES={selected}",
        "--env", f"OPENEVO_ASSIGNED_GPU_UUIDS={selected}",
        "--env", "TMPDIR=/tmp", "--env", "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor",
        "--env", "HF_HOME=/tmp/hf-home", "--env", "TRANSFORMERS_CACHE=/tmp/hf-home/transformers",
        "--env", "GIT_CONFIG_COUNT=2",
        "--env", "GIT_CONFIG_KEY_0=safe.directory",
        "--env", f"GIT_CONFIG_VALUE_0={source_a}",
        "--env", "GIT_CONFIG_KEY_1=safe.directory",
        "--env", f"GIT_CONFIG_VALUE_1={repo.resolve()}",
        "--entrypoint", "python3", profile["image_id"], str(source_a / "scripts/openevo_webshop/q08_seen_checkpoint_probe.py"),
        "--panel-manifest", str(repo / frozen["panel_manifest"]["path"]),
        "--checkpoint-matrix", str(repo / frozen["checkpoint_matrix"]["path"]),
        "--runtime-source-root", frozen["runtime_source_root"], "--seed-root", frozen["seed_root"],
        "--memory", frozen["memory"], "--skill", frozen["skill"], "--agent", frozen["agent"],
        "--output-dir", str(output_leaf), "--lineage", lineage, "--completed-rounds", str(int(rounds)),
        "--model", frozen["model"], "--revision", frozen["model_revision"], "--gpu-uuid", selected,
        "--authority-repo", str(repo), "--launch-context", str(context_path), "--cell-claim", str(claim),
    ]
    if not output_parent.is_dir() or output_leaf.exists():
        raise RuntimeError(f"Q08 fixed output parent/leaf state is invalid: {output_parent}")
    if not qualification_path.is_file() or not context_path.parent.is_dir():
        raise RuntimeError("Q08 per-cell qualification/context inputs are not ready")
    return command


def _container_name(profile: dict, cell: str, control_attempt: int) -> str:
    suffix = "" if control_attempt == 1 else f"-a{control_attempt:04d}"
    return f"experiment-host-q08-{profile['run_id']}-{cell}{suffix}"


def _plan(repo: Path, selected: str | None):
    profile, release, router, current = _load_static_chain(repo)
    eligible = list(CELL_IDS)
    if selected and selected not in eligible:
        raise RuntimeError(f"Q08 cell is not eligible: {selected}")
    run_root = Path(profile["workspace"]) / profile["paths"]["runs"] / profile["run_id"]
    cells = [selected] if selected else eligible
    return profile, release, router, current, {
        "status": "Q08_FORMAL_PLAN",
        "campaign_id": profile["campaign_id"],
        "run_id": profile["run_id"],
        "eligible_cells": eligible,
        "selected_cells": cells,
        "rejected_missing_payload_cells": ["ordinary-r040", "ordinary-r120"],
        "episodes_per_cell": 96,
        "fresh_seen_episodes_max": 960,
        "reused_historical_episodes": 768,
        "program_total_maximum_including_reuse": 1920,
        "formal_start_required": True,
        "claim_path_root": str(run_root / "control" / "claims"),
        "output_root": str(run_root / "cells"),
        "no_resume": True,
    }


def execute(repo: Path, selected_cell: str | None, qualification_path: Path) -> int:
    if os.geteuid() != 0:
        raise RuntimeError("Q08 host orchestrator requires the canonical root Docker control plane")
    profile, release, router, current, plan = _plan(repo, selected_cell)
    start_value = os.environ.get("OPENEVO_FORMAL_START_RECEIPT", "")
    if not start_value:
        raise RuntimeError("canonical run-campaign must provide OPENEVO_FORMAL_START_RECEIPT")
    start_path = Path(start_value).resolve(strict=True)
    assigned = [item for item in os.environ.get("OPENEVO_ASSIGNED_GPU_UUIDS", "").split(",") if item]
    if len(assigned) != 1 or assigned[0] not in profile["gpu_gate"]["authorized_uuids"]:
        raise RuntimeError("Q08 requires one canonical run-campaign assigned UUID from the frozen pool")
    selected = assigned[0]
    start, ready, recovery = _load_formal_start(start_path, profile, assigned, release, repo)
    qualification_path = qualification_path.resolve(strict=True)
    q = read_json(qualification_path)
    if q.get("status") != "ZERO_FORMAL_QUALIFIED" or q.get("selected_gpu_uuids") != assigned or q.get("formal_task_budget") != 0 or q.get("model_calls") != 0 or q.get("task_bearing_calls") != 0 or q.get("final_panel_access") != 0:
        raise RuntimeError("Q08 zero-formal qualification receipt does not match assigned UUID")
    expected_qpath = Path(str(profile["frozen_inputs"]["qualification_path"])).resolve()
    if qualification_path != expected_qpath:
        raise RuntimeError("Q08 qualification path differs from the frozen profile")
    q_binding = q.get("q08_binding") or {}
    for key, expected in (("campaign_id", profile["campaign_id"]), ("run_id", profile["run_id"]), ("gpu_uuid", selected), ("profile_sha256", sha256_file(bound_path(repo, PROFILE))), ("release_sha256", sha256_file(bound_path(repo, RELEASE))), ("router_sha256", sha256_file(bound_path(repo, ROUTER))), ("panel_sha256", PANEL_SHA), ("matrix_sha256", MATRIX_SHA), ("reuse_audit_sha256", REUSE_SHA), ("image_id", profile["image_id"]), ("operational_source_sha", profile["frozen_inputs"]["operational_source"]["revision"])):
        if q_binding.get(key) != expected:
            raise RuntimeError(f"Q08 qualification scope mismatch: {key}")
    if set(q.get("authorized_cells", [])) != set(CELL_IDS):
        raise RuntimeError("Q08 qualification does not authorize exactly the ten retained checkpoint cells")
    b_head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    b_dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"], check=True, capture_output=True, text=True).stdout.strip()
    if b_dirty or q.get("authority_git_sha") != b_head:
        raise RuntimeError("Q08 qualification is not bound to this clean authority checkout")
    observed_image = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", profile["image"]], check=True, capture_output=True, text=True).stdout.strip()
    if observed_image != profile["image_id"]:
        raise RuntimeError("Q08 pinned scientific image identity changed at point of use")
    qsha = sha256_file(qualification_path)
    qtime = datetime.datetime.fromisoformat(str(q.get("observed_at_utc", "")).replace("Z", "+00:00"))
    now = datetime.datetime.now(datetime.timezone.utc)
    max_age = int(profile.get("gpu_gate", {}).get("qualification_max_age_seconds", 900))
    age = (now - qtime.astimezone(datetime.timezone.utc)).total_seconds()
    if age < -30 or age > max_age:
        raise RuntimeError(f"Q08 zero-formal qualification is not fresh at FormalStart: age_seconds={age:.1f}")
    if q.get("uid") != 1001 or q.get("gid") != 1001 or q.get("scientific_state_mutated") is not False:
        raise RuntimeError("Q08 qualification UID/scientific-state evidence is invalid")
    prelaunch_ref = q.get("prelaunch_receipt") or {}
    prelaunch_path = Path(str(prelaunch_ref.get("path") or ""))
    if (not prelaunch_path.is_file() or prelaunch_path.is_symlink()
            or _normalize_sha(prelaunch_ref.get("sha256")) != sha256_file(prelaunch_path)):
        raise RuntimeError("Q08 selected-GPU prelaunch receipt reference is invalid")
    prelaunch = read_json(prelaunch_path)
    if (prelaunch.get("status") != "PASS" or prelaunch.get("selected_gpu_uuids") != assigned
            or prelaunch.get("image_id") != profile["image_id"] or prelaunch.get("formal_task_consumed") is not False
            or prelaunch.get("scientific_budget_consumed") != 0 or prelaunch.get("model_calls") != 0
            or prelaunch.get("training_calls") != 0 or prelaunch.get("final_panel_access") != 0):
        raise RuntimeError("Q08 selected-GPU zero-formal prelaunch receipt is not a zero-task PASS")
    wrapper_ref = q.get("wrapper_identity_receipt") or {}
    wrapper_path = Path(str(wrapper_ref.get("path") or ""))
    if not wrapper_path.is_file() or _normalize_sha(wrapper_ref.get("sha256")) != sha256_file(wrapper_path):
        raise RuntimeError("Q08 UID1001 wrapper identity receipt reference is invalid")
    wrapper_receipt = read_json(wrapper_path)
    wrapper_result = wrapper_receipt.get("result") or {}
    if (wrapper_receipt.get("status") != "PASS" or wrapper_receipt.get("formal_task_consumed") is not False
            or wrapper_receipt.get("scientific_budget_consumed") != 0
            or wrapper_result.get("launch_guard_identity_validated") is not True
            or (wrapper_result.get("launch_guard_identity") or {}).get("claim_consumed") is not False):
        raise RuntimeError("Q08 UID1001 identity guard qualification is not a zero-task PASS")
    storage_gate = profile.get("storage_gate") or {}
    current_free = shutil.disk_usage(storage_gate["filesystem_path"]).free
    if current_free < int(storage_gate["minimum_free_bytes"]):
        raise RuntimeError("Q08 workspace capacity fell below the frozen minimum before FormalStart")
    run_root = Path(profile["workspace"]) / profile["paths"]["runs"] / profile["run_id"]
    launch_receipt_path = run_root / "launch-receipt.json"
    if not launch_receipt_path.is_file():
        raise RuntimeError("canonical run-campaign launch receipt is missing")
    launch_receipt = read_json(launch_receipt_path)
    start_ref = launch_receipt.get("formal_start") or {}
    if (launch_receipt.get("execution_mode") != "host-orchestrator"
            or launch_receipt.get("image_id") != profile["image_id"]
            or launch_receipt.get("selected_gpu_uuids") != assigned
            or start_ref.get("path") != str(start_path)
            or _normalize_sha(start_ref.get("sha256")) != sha256_file(start_path)
            or start_ref.get("receipt_sha256") != start.get("receipt_sha256")):
        raise RuntimeError("canonical launch receipt does not bind this FormalStart/Q08 invocation")
    tracking = launch_receipt.get("tracking") or {}
    mode = tracking.get("effective_mode")
    if mode not in ("disabled", "offline", "online"):
        raise RuntimeError("canonical run-campaign effective tracking mode is invalid")
    replacements = {
        "${RUN_DIR}": str(run_root),
        "${DATA_DIR}": str(Path(profile["workspace"]) / profile["paths"]["data"]),
        "${MODELS_DIR}": str(Path(profile["workspace"]) / profile["paths"]["models"]),
        "${EXPERIMENT_DIR}": str(Path(profile["workspace"]) / profile["source"]["experiment"]["path"]),
        "${OPENEVO_DIR}": str(Path(profile["workspace"]) / profile["source"]["openevo"]["path"]),
        "${GPU_UUID}": selected,
        "${GPU_UUIDS}": selected,
    }
    argv = [str(arg) for arg in profile["command"]]
    for token, value in replacements.items():
        argv = [arg.replace(token, value) for arg in argv]
    expected_command = ["/usr/bin/env", f"OPENEVO_AUTHORIZED_GPU_UUIDS={selected}", f"OPENEVO_ASSIGNED_GPU_UUIDS={selected}", f"OPENEVO_FORMAL_START_RECEIPT={start_path}", f"WANDB_MODE={mode}", f"OPENEVO_WANDB_MODE={mode}", *argv]
    identity = "sha256:" + _canonical_hash({"image_id": profile["image_id"], "argv": expected_command})
    if start.get("command_identity") != identity:
        raise RuntimeError("FormalStart command identity differs from the Q08 host orchestrator profile")
    control = run_root / "control"
    claims = control / "claims"
    _require_uid1001_dir(control, create=True)
    _require_uid1001_dir(claims, create=True)
    if claims.stat().st_mode & 0o022:
        raise RuntimeError("Q08 claim directory must not be group/world writable")
    start_match = re.fullmatch(r"formal-start-attempt-(\d{4})\.json", start_path.name)
    if start_match is None:
        raise RuntimeError("Q08 FormalStart path does not carry a canonical attempt number")
    control_attempt = int(start_match.group(1))
    if recovery is not None:
        recovery_attempt = int(recovery["attempt"])
        continuation = recovery.get("continuation") or {}
        expected_attempt = int(continuation.get("control_attempt", recovery_attempt))
        if control_attempt != expected_attempt:
            raise RuntimeError("Q08 FormalStart attempt differs from the exact recovery continuation")
        recovery_dir = run_root / f"control/engineering-recovery-attempt-{recovery_attempt:04d}"
        _require_uid1001_dir(recovery_dir, create=True)
    context_namespace = "control/cells"
    if recovery is not None:
        context_namespace = str((recovery.get("continuation") or {}).get("context_namespace", context_namespace))
    context_root = (run_root / context_namespace).resolve()
    try:
        context_root.relative_to(run_root.resolve())
    except ValueError as exc:
        raise RuntimeError("Q08 recovery context namespace escapes the fixed run root") from exc

    # Preflight every selected cell path and its exact context identity before
    # consuming the append-only engineering admission claim.
    start_ref = {"path": str(start_path), "sha256": sha256_file(start_path), "receipt_sha256": start["receipt_sha256"]}
    qualification_ref = {"path": str(qualification_path), "sha256": qsha}
    release_ref = {"path": str(bound_path(repo, RELEASE).relative_to(repo)), "sha256": sha256_file(bound_path(repo, RELEASE))}
    profile_ref = {"path": str(bound_path(repo, PROFILE).relative_to(repo)), "sha256": sha256_file(bound_path(repo, PROFILE))}
    router_ref = {"path": str(bound_path(repo, ROUTER).relative_to(repo)), "sha256": sha256_file(bound_path(repo, ROUTER))}
    contexts = {}
    _require_uid1001_dir(context_root, create=True)
    for cell in plan["selected_cells"]:
        cell_parent = run_root / "cells" / cell
        _require_uid1001_dir(cell_parent, create=True)
        output_leaf = cell_parent / "seen-eval"
        if output_leaf.exists() or output_leaf.is_symlink():
            raise RuntimeError(f"Q08 cell output leaf already exists; fixed one-use route forbids resume: {cell}")
        attempt_context_root = context_root / f"attempt-{control_attempt:04d}"
        _require_uid1001_dir(attempt_context_root, create=True)
        context_dir = attempt_context_root / cell
        _require_uid1001_dir(context_dir, create=True)
        context_path = context_dir / "launch-context.json"
        if context_path.exists() or context_path.is_symlink():
            raise RuntimeError(f"Q08 launch context already exists; refusing replay: {cell}")
        context = {
            "schema": "openevo.q08.cell-launch-context.v1", "status": "FORMAL_START_COMMITTED",
            "control_attempt": control_attempt,
            "campaign_id": profile["campaign_id"], "run_id": profile["run_id"], "cell_id": cell,
            "selected_gpu_uuids": assigned, "qualification": qualification_ref,
            "formal_start": start_ref, "release": release_ref, "profile": profile_ref,
            "router": router_ref, "ready_receipt_sha256": (start.get("readiness") or {}).get("ready_receipt", {}).get("sha256"),
            "task_limit": 96, "final_panel_access": 0, "resume_allowed": False,
        }
        if (context["control_attempt"] != control_attempt or context["cell_id"] != cell
                or context["run_id"] != profile["run_id"] or context["campaign_id"] != profile["campaign_id"]
                or context["selected_gpu_uuids"] != assigned or context["task_limit"] != 96
                or context["resume_allowed"] is not False):
            raise RuntimeError(f"Q08 prospective launch context identity mismatch: {cell}")
        contexts[cell] = (context_path, context)
    if recovery is not None:
        admission_path = recovery_dir / "admission-claim.json"
        _atomic_once(admission_path, {
            "schema": "openevo.q08.engineering-recovery-admission.v1",
            "status": f"CONSUMED_FOR_CONTROL_ATTEMPT_{control_attempt:04d}",
            # `recovery` is the authenticated document; the raw committed
            # receipt hash remains in the verified release descriptor.
            "recovery_receipt_sha256": release["engineering_recovery"]["sha256"],
            "formal_start_sha256": sha256_file(start_path),
            "qualification_sha256": qsha,
            "run_id": profile["run_id"], "campaign_id": profile["campaign_id"],
            "scientific_resume": False, "formal_task_consumed_before_claim": 0,
            "cell_claims_consumed_before_claim": 0, "episodes_consumed_before_claim": 0,
            "prepared_context_paths": {cell: str(item[0]) for cell, item in contexts.items()},
        })
    for cell in plan["selected_cells"]:
        cell_parent = run_root / "cells" / cell
        context_path, context = contexts[cell]
        _atomic_once(context_path, context)
        command = _docker_argv(profile, repo, cell, selected, context_path, qualification_path)
        print(json.dumps({"status": "Q08_CELL_LAUNCH", "cell_id": cell, "gpu_uuid": selected, "argv": command}, sort_keys=True), flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print(json.dumps({"status":"Q08_CELL_FAILED_CONTAINER_PRESERVED","cell_id":cell,"container_name":_container_name(profile, cell, control_attempt),"exit_code":result.returncode},sort_keys=True),file=sys.stderr,flush=True)
            return result.returncode
        summary = read_json(cell_parent / "seen-eval" / "summary.json")
        if (summary.get("status") != "PASS" or summary.get("attempted_tasks") != 96
                or summary.get("valid_tasks") != 96 or summary.get("engineering_invalid_tasks") != 0
                or summary.get("training_steps") != 0 or summary.get("external_teacher_calls") != 0
                or summary.get("final_panel_access") != 0):
            print(json.dumps({"status":"Q08_CELL_SUMMARY_INVALID_CONTAINER_PRESERVED","cell_id":cell,"container_name":_container_name(profile, cell, control_attempt),"summary":summary},sort_keys=True),file=sys.stderr,flush=True)
            raise RuntimeError(f"Q08 cell did not seal PASS with 96 valid/0 invalid tasks: {cell}")
        subprocess.run(["docker", "rm", _container_name(profile, cell, control_attempt)], check=True, capture_output=True, text=True)
    print(json.dumps({"status": "Q08_SELECTED_CELLS_SEALED", "cells": plan["selected_cells"], "formal_start_receipt": start_ref, "qualification_receipt": qualification_ref}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--cell", choices=CELL_IDS)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    repo = args.repo_root.expanduser().resolve(strict=True)
    if args.plan_only:
        _, _, _, _, plan = _plan(repo, args.cell)
        print(json.dumps(plan, sort_keys=True, indent=2))
        return 0
    return execute(repo, args.cell, args.qualification)


if __name__ == "__main__":
    raise SystemExit(main())
