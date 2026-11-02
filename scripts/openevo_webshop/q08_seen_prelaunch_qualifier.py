#!/usr/bin/env python3
"""Q08-specific zero-task qualification using the frozen runtime and UUID pool."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from q08_seen_launch_guard import CELL_IDS, MATRIX_SHA, PANEL_SHA, PROFILE, RELEASE, REUSE_SHA, ROUTER, _load_static_chain, bound_path, read_json, sha256_file, validate_engineering_recovery


def _once(path: Path, value: dict) -> None:
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"Q08 qualification evidence is append-only: {path}")
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        # UID1001 consumes these root-produced receipts through the read-only mount.
        os.fchown(fd, -1, 1001)
        os.fchmod(fd, 0o640)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _docker(profile: dict, selected: str, argv: list[str], container_name: str) -> list[str]:
    workspace = Path(profile["workspace"]).resolve()
    runs = workspace / profile["paths"]["runs"]
    cache = workspace / profile["paths"]["cache"]
    git_safe_dirs = [str(Path(profile["frozen_inputs"]["operational_source"]["path"]).resolve())]
    if "--authority-repo" in argv:
        authority_index = argv.index("--authority-repo") + 1
        git_safe_dirs.append(str(Path(argv[authority_index]).resolve()))
    command = [
        "docker", "run", "--name", container_name, "--read-only", "--security-opt", "no-new-privileges",
        "--cap-drop", "ALL", "--pids-limit", "4096", "--shm-size", "8g",
        "--tmpfs", profile["container"]["tmpfs"],
        "--gpus", f"device={selected}", "--user", "1001:1001", "--network", "none",
        "--mount", f"type=bind,src={workspace},dst={workspace},readonly",
        "--mount", f"type=bind,src={runs},dst={runs}",
        "--mount", f"type=bind,src={cache},dst={cache}",
        "--env", f"CUDA_VISIBLE_DEVICES={selected}",
        "--env", f"OPENEVO_ASSIGNED_GPU_UUIDS={selected}",
        "--env", "TMPDIR=/tmp", "--env", "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor",
        "--env", "HF_HOME=/tmp/hf-home", "--env", "TRANSFORMERS_CACHE=/tmp/hf-home/transformers",
        "--env", f"GIT_CONFIG_COUNT={len(git_safe_dirs)}",
        "--entrypoint", argv[0], profile["image_id"], *argv[1:],
    ]
    for index, directory in enumerate(git_safe_dirs):
        command[command.index("--entrypoint"):command.index("--entrypoint")] = [
            "--env", f"GIT_CONFIG_KEY_{index}=safe.directory",
            "--env", f"GIT_CONFIG_VALUE_{index}={directory}",
        ]
    return command


def _run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def _prelaunch(profile: dict, selected: str, attempt_root: Path, marker: Path) -> dict:
    from scripts.openevo_webshop import prelaunch_zero_formal as zf

    if marker.exists() or marker.is_symlink():
        raise RuntimeError(f"Q08 zero-formal write marker already exists: {marker}")
    marker.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    os.chown(marker.parent, 1001, 1001)
    for parent in (marker.parent, *marker.parent.parents):
        if parent == Path(profile["workspace"]).resolve():
            break
    payload = zf.payload_command("python3", str(marker), 1)
    name = f"experiment-host-q08-preflight-{profile['run_id']}-{attempt_root.name}-{selected[-8:]}"
    command = _docker(profile, selected, payload, name)
    started = time.monotonic()
    completed = _run(command)
    parsed = None
    parse_error = None
    try:
        parsed = zf.parse_payload_output(completed.stdout)
    except zf.PrelaunchError as exc:
        parse_error = str(exc)
    receipt = zf.build_prelaunch_receipt(
        run_id=profile["run_id"], attempt=1, selected_gpu_uuids=[selected],
        image_id=profile["image_id"],
        command_identity="sha256:" + hashlib.sha256(json.dumps({"argv": command, "image_id": profile["image_id"]}, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        return_code=completed.returncode, wall_seconds=max(0.0, time.monotonic() - started),
        payload=parsed, payload_error=parse_error, stderr_tail=completed.stderr,
    )
    receipt_path = attempt_root / "prelaunch-zero-formal.json"
    _once(receipt_path, receipt)
    zf.validate_prelaunch_receipt(receipt, expected_gpu_uuids=[selected])
    subprocess.run(["docker", "rm", name], check=True, capture_output=True, text=True)
    return {"path": str(receipt_path), "sha256": sha256_file(receipt_path), "receipt_sha256": receipt["receipt_sha256"], "receipt": receipt, "command": command}


def _validate_wrapper(profile: dict, repo: Path, selected: str, attempt_root: Path) -> dict:
    frozen = profile["frozen_inputs"]
    source_a = Path(frozen["operational_source"]["path"]).resolve()
    lineage = "bounded"
    rounds = 40
    run_root = Path(profile["workspace"]) / profile["paths"]["runs"] / profile["run_id"]
    command = [
        "python3", str(source_a / "scripts/openevo_webshop/q08_seen_checkpoint_probe.py"),
        "--panel-manifest", str(repo / frozen["panel_manifest"]["path"]),
        "--checkpoint-matrix", str(repo / frozen["checkpoint_matrix"]["path"]),
        "--runtime-source-root", frozen["runtime_source_root"], "--seed-root", frozen["seed_root"],
        "--memory", frozen["memory"], "--skill", frozen["skill"], "--agent", frozen["agent"],
        "--output-dir", str(run_root / "cells" / "bounded-r040" / "seen-eval"),
        "--lineage", lineage, "--completed-rounds", str(rounds), "--model", frozen["model"],
        "--revision", frozen["model_revision"], "--gpu-uuid", selected, "--validate-only",
        "--authority-repo", str(repo),
    ]
    name = f"experiment-host-q08-identity-{profile['run_id']}-{attempt_root.name}-{selected[-8:]}"
    completed = _run(_docker(profile, selected, command, name))
    if completed.returncode != 0:
        raise RuntimeError(f"Q08 wrapper validate-only failed: rc={completed.returncode}; stderr={completed.stderr[-3000:]}")
    values = [line for line in completed.stdout.splitlines() if line.strip().startswith("{")]
    result = json.loads(values[-1])
    identity = result.get("launch_guard_identity") or {}
    if (result.get("status") != "IDENTITY_PASS" or result.get("task_count") != 96
            or result.get("model_calls") != 0 or result.get("training_steps") != 0
            or result.get("final_panel_access") != 0 or result.get("launch_guard_identity_validated") is not True
            or identity.get("status") != "IDENTITY_PASS" or identity.get("claim_consumed") is not False):
        raise RuntimeError("Q08 wrapper validate-only returned unexpected identity/cost values")
    subprocess.run(["docker", "rm", name], check=True, capture_output=True, text=True)
    path = attempt_root / "wrapper-identity-validation.json"
    doc = {"schema": "openevo.q08.wrapper-identity-validation.v1", "status": "PASS", "result": result, "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(), "return_code": completed.returncode, "formal_task_consumed": False, "scientific_budget_consumed": 0}
    _once(path, doc)
    return {"path": str(path), "sha256": sha256_file(path), "result": result}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-repo", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--engineering-recovery", type=Path)
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise RuntimeError("Q08 selected-GPU qualification requires the canonical root control plane")
    repo = args.authority_repo.resolve(strict=True)
    profile, release, router, current = _load_static_chain(repo)
    source_a = Path(profile["frozen_inputs"]["operational_source"]["path"]).resolve(strict=True)
    a_sha = profile["frozen_inputs"]["operational_source"]["revision"]
    b_sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if args.gpu_uuid not in profile["gpu_gate"]["authorized_uuids"]:
        raise RuntimeError("selected GPU UUID is outside the frozen standing pool")
    canonical = profile["frozen_inputs"]["canonical_runner"]
    canonical_root = Path(canonical["path"]).resolve(strict=True)
    canonical_head = subprocess.run(["git", "-C", str(canonical_root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    canonical_dirty = subprocess.run(["git", "-C", str(canonical_root), "status", "--porcelain", "--untracked-files=all"], check=True, capture_output=True, text=True).stdout.strip()
    if canonical_head != canonical["revision"] or canonical_dirty:
        raise RuntimeError("canonical runner source identity is not the exact clean pinned checkout")
    if sha256_file(canonical_root / "scripts/oe_impl.py") != canonical["oe_impl_sha256"]:
        raise RuntimeError("canonical run-campaign owner source SHA drift")
    sys.path.insert(0, str(canonical_root))
    from scripts import oe_impl

    oe_impl.validate_sources_and_paths(profile)
    if oe_impl.resolve_image_identity(profile) != profile["image_id"]:
        raise RuntimeError("Q08 pinned image id differs from host Docker image identity")
    storage = profile.get("storage_gate") or {}
    if storage.get("filesystem_path") != profile["workspace"] or int(storage.get("minimum_free_bytes", 0)) <= 0:
        raise RuntimeError("Q08 storage gate must declare the exact workspace and a positive minimum")
    usage = shutil.disk_usage(storage["filesystem_path"])
    if usage.free < int(storage["minimum_free_bytes"]):
        raise RuntimeError(f"Q08 workspace free space below frozen gate: {usage.free} < {storage['minimum_free_bytes']}")
    mount = subprocess.run(["findmnt", "-T", storage["filesystem_path"], "-no", "SOURCE,FSTYPE,TARGET"], check=True, capture_output=True, text=True).stdout.strip().split(maxsplit=2)
    if len(mount) != 3 or mount[1] != "fuse.galaxyfs" or mount[2] != "/workspace":
        raise RuntimeError("Q08 workspace filesystem/mount identity drift")
    selected = [args.gpu_uuid]
    cards = oe_impl.snapshot_gpus()
    oe_impl.select_gpu_uuids(args.gpu_uuid, profile["gpu_gate"]["authorized_uuids"], cards)
    qualification_path = Path(profile["frozen_inputs"]["qualification_path"]).resolve()
    run_root = Path(profile["workspace"]) / profile["paths"]["runs"] / profile["run_id"]
    recovery = validate_engineering_recovery(repo, profile, release)
    recovery_ref = release.get("engineering_recovery")
    if run_root.exists() or run_root.is_symlink():
        if recovery is None or args.engineering_recovery is None:
            raise RuntimeError("Q08 existing run root requires the exact zero-consumption engineering recovery receipt")
        if args.engineering_recovery.resolve(strict=True) != bound_path(repo,recovery_ref["path"]):
            raise RuntimeError("Q08 engineering recovery CLI path differs from the release-bound receipt")
    elif recovery is not None or args.engineering_recovery is not None:
        raise RuntimeError("Q08 engineering recovery cannot be used without the preserved failed run root")
    for cell in CELL_IDS:
        fixed_leaf = run_root / "cells" / cell / "seen-eval"
        fixed_claim = run_root / "control" / "claims" / f"{cell}.claim"
        if fixed_leaf.exists() or fixed_leaf.is_symlink() or fixed_claim.exists() or fixed_claim.is_symlink():
            raise RuntimeError(f"Q08 one-use output/claim already exists for fixed cell {cell}")
    attempt_root = qualification_path.parent / f"attempt-{a_sha[:12]}-{b_sha[:12]}-{args.gpu_uuid[-8:]}"
    attempt_root.mkdir(mode=0o750, parents=True, exist_ok=False)
    os.chown(attempt_root, -1, 1001)
    os.chmod(attempt_root, 0o750)
    lock_root = oe_impl.workspace_path(profile, str(profile.get("lock_root", ".gpu-locks")))
    gate_profile = {**profile, "wait": False, "gpu_gate": {**profile["gpu_gate"], "interval_seconds": 1, "timeout_seconds": 0}}
    with oe_impl.gpu_locks(selected, lock_root):
        gpu_receipt_path = attempt_root / "gpu-owner-gate.json"
        gate = oe_impl.wait_for_idle(gate_profile, selected, attempt_root)
        gpu_owner = {"schema": "openevo.q08.gpu-owner-gate.v1", "status": "PASS", "selected_gpu_uuids": selected, "snapshot": gate, "lock_root": str(lock_root), "locked_at_epoch": time.time(), "formal_task_consumed": False, "scientific_budget_consumed": 0}
        _once(gpu_receipt_path, gpu_owner)
        runs_root = oe_impl.workspace_path(profile, str(profile["paths"]["runs"]))
        marker = runs_root / ".q08-preflight" / profile["run_id"] / attempt_root.name / args.gpu_uuid[-8:] / "write-probe"
        prelaunch = _prelaunch(profile, args.gpu_uuid, attempt_root, marker)
        wrapper = _validate_wrapper(profile, repo, args.gpu_uuid, attempt_root)
    binding = {
        "operational_source_sha": a_sha, "authority_git_sha": b_sha,
        "campaign_id": profile["campaign_id"], "run_id": profile["run_id"],
        "gpu_uuid": args.gpu_uuid, "profile_sha256": sha256_file(bound_path(repo, PROFILE)),
        "release_sha256": sha256_file(bound_path(repo, RELEASE)), "router_sha256": sha256_file(bound_path(repo, ROUTER)),
        "panel_sha256": PANEL_SHA, "matrix_sha256": MATRIX_SHA, "reuse_audit_sha256": REUSE_SHA,
        "image_id": profile["image_id"], "eligible_cells": list(CELL_IDS), "episodes_per_cell": 96,
        "fresh_seen_episodes_max": 960, "reused_historical_episodes": 768,
        "program_total_maximum_including_reuse": 1920,
    }
    if recovery is not None:
        binding["engineering_recovery_sha256"] = sha256_file(bound_path(repo,recovery_ref["path"]))
    receipt = {
        "schema": "openevo.q08.zero-formal-qualification.v1", "status": "ZERO_FORMAL_QUALIFIED",
        "campaign_id": profile["campaign_id"], "execution_sha": a_sha, "authority_git_sha": b_sha,
        "selected_gpu_uuids": selected, "authorized_cells": list(CELL_IDS),
        "formal_task_budget": 0, "scientific_budget_consumed": 0, "task_bearing_calls": 0,
        "model_calls": 0, "training_calls": 0, "training_steps": 0, "final_panel_access": 0,
        "scientific_state_mutated": False, "uid": 1001, "gid": 1001,
        "image_id": profile["image_id"], "filesystem": {"path": storage["filesystem_path"], "fstype": mount[1], "mount_target": mount[2], "available_bytes": usage.free, "minimum_free_bytes": storage["minimum_free_bytes"]},
        "gpu_owner_receipt": {"path": str(gpu_receipt_path), "sha256": sha256_file(gpu_receipt_path)},
        "prelaunch_receipt": {"path": prelaunch["path"], "sha256": prelaunch["sha256"], "receipt_sha256": prelaunch["receipt_sha256"]},
        "wrapper_identity_receipt": wrapper, "q08_binding": binding,
        "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    receipt["receipt_sha256"] = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    receipt_path = qualification_path
    _once(receipt_path, receipt)
    print(json.dumps({"status": receipt["status"], "qualification": str(receipt_path), "sha256": sha256_file(receipt_path), "formal_task_consumed": False, "scientific_budget_consumed": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
