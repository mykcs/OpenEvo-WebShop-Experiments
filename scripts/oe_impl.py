#!/usr/bin/env python3
"""Fail-closed one-command launcher for a profiled WebShop experiment.

The launcher owns experiment orchestration only. Host account policy, Docker
authority and the standing GPU pool remain owned by ``zju-server``. A profile
records exact source revisions, persistent paths and command; this script adds a
double-sampled GPU gate and an explicit UUID-bound Docker invocation.

``OPENEVO_AUTHORIZED_GPU_UUIDS`` is retained as a compatibility variable name.
Its value is the current launch's runtime projection of the standing GPU pool,
not evidence of a fresh per-campaign human approval.

``--gpus auto:N`` is capacity-oriented. It selects N currently live-idle GPUs
from that runtime candidate pool. With ``--wait`` it resamples the whole pool;
a busy card therefore reduces capacity rather than pinning the launch to a stale
UUID choice. Explicit UUID selectors remain exact and fail-closed.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Callable, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.openevo_webshop import gpu_resource_wrapper
from scripts.openevo_webshop import experiment_reliability_runtime as reliability
from scripts.openevo_webshop.gpu_resource_gate import (
    classify_device_handles,
    co_resident_capacity,
    idle as canonical_idle,
    is_trusted_monitor_handle,
    parse_gpu_rows,
    process_identity,
    running_gpu_container_owners,
    visible_device_handles,
)
from scripts.openevo_webshop.path_contract import (
    PathContractError,
    run_cpu_no_gpu_probe,
    validate_contract,
    validate_host_path,
)
from scripts.openevo_webshop import execution_readiness
from scripts.openevo_webshop import prelaunch_zero_formal
from scripts.openevo_webshop import rank_consistency
from scripts.openevo_webshop import runtime_live_gate
from scripts.openevo_webshop import runtime_qualification
from scripts.openevo_webshop import workspace_discovery


GPU_UUID_RE = re.compile(r"^GPU-[A-Za-z0-9-]+$")
IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
HOST_TMPFS_ROOT = Path("/dev/shm")
DEFAULT_AUTH_ENV = "OPENEVO_AUTHORIZED_GPU_UUIDS"
CANONICAL_WORKSPACE_REGISTRY_POINTER = Path("/workspace/openevo-workspace/server-registry.json")


class LaunchError(RuntimeError):
    pass


class NonRetryableLaunchError(LaunchError):
    """A deterministic engineering/authority blocker must not become a wait loop."""


class PrelaunchEngineeringError(NonRetryableLaunchError):
    pass


class AuthorityRevokedError(NonRetryableLaunchError):
    pass


@contextlib.contextmanager
def _exclusive_owned_lock(path: Path):
    """Open an owned regular lock without following links, then take flock."""
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise LaunchError(f"cannot safely open launcher lock {path}: {exc}") from exc
    try:
        current = os.fstat(fd)
        try:
            named = path.lstat()
        except OSError as exc:
            raise LaunchError(f"launcher lock path changed while opening: {path}") from exc
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_uid != os.geteuid()
            or current.st_nlink != 1
            or stat.S_ISLNK(named.st_mode)
            or (named.st_dev, named.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise LaunchError(f"launcher lock must be an owned, unlinked regular file: {path}")
        os.fchmod(fd, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        with os.fdopen(fd, "a+b", closefd=False) as lock_file:
            yield lock_file
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _run(command: list[str], *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, text=True, capture_output=capture)


def load_profile(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchError(f"profile unreadable: {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != "openevo.webshop.run-profile.v1":
        raise LaunchError("profile schema must be openevo.webshop.run-profile.v1")
    if "_runtime_asset_snapshots" in payload:
        raise LaunchError("profile must not declare launcher-private runtime snapshot state")
    required = {"run_id", "image", "image_id", "workspace", "identity", "source", "paths", "command", "tracking"}
    missing = sorted(required - payload.keys())
    if missing:
        raise LaunchError(f"profile missing keys: {', '.join(missing)}")
    if not isinstance(payload["command"], list) or not payload["command"]:
        raise LaunchError("profile command must be a non-empty argv list")
    if not isinstance(payload["image_id"], str) or not IMAGE_ID_RE.fullmatch(payload["image_id"]):
        raise LaunchError("profile image_id must be a Docker immutable local image ID (sha256:<64 lowercase hex>)")
    tracking = payload["tracking"]
    if not isinstance(tracking, dict) or tracking.get("mode") not in {"online", "offline", "disabled"}:
        raise LaunchError("tracking.mode must be online, offline, or disabled")
    if tracking.get("on_error", "fallback_offline") not in {"fallback_offline", "fail"}:
        raise LaunchError("tracking.on_error must be fallback_offline or fail")
    identity = payload["identity"]
    if not isinstance(identity, dict) or not all(isinstance(identity.get(key), int) for key in ("uid", "gid")):
        raise LaunchError("identity.uid and identity.gid must be integers")
    for key in ("experiment", "openevo"):
        source = payload["source"].get(key) if isinstance(payload["source"], dict) else None
        if not isinstance(source, dict) or not isinstance(source.get("path"), str) or not isinstance(source.get("revision"), str):
            raise LaunchError(f"source.{key} requires path and revision")
    return payload


def workspace_path(profile: Mapping[str, Any], relative: str) -> Path:
    root = Path(str(profile["workspace"])).expanduser().resolve()
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise LaunchError(f"path escapes workspace: {relative}")
    return candidate


def _reject_symlink_components(root: Path, path: Path, label: str, *, allow_missing: bool = False) -> None:
    """Reject symlink traversal for content-addressed runtime inputs."""
    root = root.resolve()
    try:
        relative = path.absolute().relative_to(root)
    except ValueError as exc:
        raise LaunchError(f"{label} escapes workspace: {path}") from exc
    if ".." in relative.parts:
        raise LaunchError(f"{label} path must not contain parent traversal: {path}")
    current = root
    for part in relative.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as exc:
            if allow_missing:
                continue
            raise LaunchError(f"{label} path missing: {current}") from exc
        if stat.S_ISLNK(mode):
            raise LaunchError(f"{label} contains symlink: {current}")


def _validate_sealed_tmpfs_ancestry(path: Path) -> None:
    """Require a private root-owned descendant of the system sticky tmpfs."""
    if os.geteuid() != 0:
        raise LaunchError("sealed /dev/shm runtime staging requires the root launcher")
    root = HOST_TMPFS_ROOT.resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise LaunchError("sealed runtime staging must be below /dev/shm") from exc
    if not relative.parts:
        raise LaunchError("sealed runtime staging cannot be the /dev/shm root")
    root_info = root.lstat()
    if (not stat.S_ISDIR(root_info.st_mode) or root_info.st_uid != 0
            or not (root_info.st_mode & stat.S_ISVTX)):
        raise LaunchError("/dev/shm must be a root-owned sticky directory")
    current = resolved
    while current != root:
        info = current.lstat()
        if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or info.st_uid != 0 or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)):
            raise LaunchError(f"sealed runtime staging ancestor is not root-owned and private: {current}")
        current = current.parent


def _validate_sealed_tmpfs_creation_parent(path: Path) -> None:
    """Validate the deepest existing ancestor without creating staging state."""
    if os.geteuid() != 0:
        raise LaunchError("sealed /dev/shm runtime staging requires the root launcher")
    root = HOST_TMPFS_ROOT.resolve(strict=True)
    candidate = path
    while not candidate.exists() and candidate != root:
        candidate = candidate.parent
    if candidate != root:
        _validate_sealed_tmpfs_ancestry(candidate)
        return
    info = root.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != 0
            or not (info.st_mode & stat.S_ISVTX)):
        raise LaunchError("/dev/shm must be a root-owned sticky directory")


def _validate_sealed_disk_ancestry(path: Path, *, allow_missing: bool = False) -> None:
    """Prove a daemon-visible staging path cannot be replaced by the run user."""
    if os.geteuid() != 0:
        raise LaunchError("sealed disk staging requires the root launcher")
    if not path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise LaunchError("sealed disk staging path must be normalized and absolute")
    current = Path("/")
    for part in ("", *path.parts[1:]):
        if part:
            current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            if allow_missing:
                return
            raise LaunchError(f"sealed disk staging ancestor is missing: {current}")
        if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or info.st_uid != 0 or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)):
            raise LaunchError(f"sealed disk staging ancestor is not root-owned and private: {current}")


def _is_account_sealed_staging(workspace: Path, staging: Path) -> bool:
    anchor = workspace.parent / ".openevo-runtime-sealed"
    return staging != anchor and staging.is_relative_to(anchor)


def runtime_asset_digest(path: Path, expected_type: str) -> str:
    """Hash one file or a deterministic tree; reject links and special files."""
    path = Path(path)
    mode = path.lstat().st_mode
    if expected_type == "file":
        if not stat.S_ISREG(mode):
            raise LaunchError(f"runtime asset expected regular file: {path}")
        digest = hashlib.sha256(b"openevo-runtime-file-v1\0")
        digest.update(b"exec\0" if mode & 0o111 else b"data\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if expected_type != "dir" or not stat.S_ISDIR(mode):
        raise LaunchError(f"runtime asset expected directory: {path}")

    digest = hashlib.sha256(b"openevo-runtime-tree-v1\0")
    entries = sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix())
    for entry in entries:
        rel = entry.relative_to(path).as_posix()
        entry_mode = entry.lstat().st_mode
        if stat.S_ISLNK(entry_mode):
            raise LaunchError(f"runtime asset tree contains symlink: {entry}")
        if stat.S_ISDIR(entry_mode):
            digest.update(f"dir\0{rel}\0".encode())
            continue
        if not stat.S_ISREG(entry_mode):
            raise LaunchError(f"runtime asset tree contains special file: {entry}")
        file_digest = hashlib.sha256()
        with entry.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(chunk)
        digest.update(
            f"file\0{rel}\0{entry.stat().st_size}\0{bool(entry_mode & 0o111)}\0"
            f"{file_digest.hexdigest()}\0".encode()
        )
    return digest.hexdigest()


def _stage_runtime_asset_snapshot(
    source: Path, final_parent: Path, target: str, expected_type: str, expected_sha: str
) -> dict[str, str]:
    """Serialize publication of one content-addressed asset for concurrent launchers."""
    lock_path = final_parent / ".snapshot.lock"
    with _exclusive_owned_lock(lock_path):
        final = final_parent / ("asset" if expected_type == "file" else "tree")
        if final.is_symlink():
            raise LaunchError(f"runtime snapshot must not be a symlink: {final}")
        if not final.exists():
            temp_parent = Path(tempfile.mkdtemp(prefix=".staging-", dir=final_parent))
            temp = temp_parent / ("asset" if expected_type == "file" else "tree")
            try:
                if expected_type == "file":
                    shutil.copy2(source, temp)
                else:
                    shutil.copytree(source, temp, symlinks=True)
                if runtime_asset_digest(source, expected_type) != expected_sha:
                    raise LaunchError(f"runtime asset source changed while staging: {source}")
                if runtime_asset_digest(temp, expected_type) != expected_sha:
                    raise LaunchError(f"staged runtime asset identity mismatch: {temp}")
                os.rename(temp, final)
            finally:
                shutil.rmtree(temp_parent, ignore_errors=True)
        if runtime_asset_digest(final, expected_type) != expected_sha:
            raise LaunchError(f"existing runtime snapshot is corrupt: {final}")
        if expected_type == "dir":
            for staged_path in sorted(final.rglob("*"), key=lambda item: len(item.parts), reverse=True):
                if staged_path.is_symlink():
                    raise LaunchError(f"runtime snapshot contains symlink: {staged_path}")
                relative = staged_path.relative_to(final)
                source_mode = (source / relative).lstat().st_mode
                executable = bool(source_mode & 0o111)
                staged_path.chmod(0o555 if staged_path.is_dir() or executable else 0o444)
            final.chmod(0o555)
        else:
            final.chmod(0o555 if source.lstat().st_mode & 0o111 else 0o444)
        if runtime_asset_digest(final, expected_type) != expected_sha:
            raise LaunchError(f"runtime asset snapshot changed after staging: {final}")
        result = {"source": str(final), "target": target, "expected_type": expected_type, "sha256": expected_sha}
        return result


def _validate_runtime_asset_target(target: str, seen: list[str]) -> None:
    parsed = PurePosixPath(target)
    if not parsed.is_absolute() or parsed.as_posix() != target or ".." in parsed.parts:
        raise LaunchError(f"runtime asset target must be a normalized absolute path: {target}")
    if any(not re.fullmatch(r"[A-Za-z0-9._-]+", part) for part in parsed.parts[1:]):
        raise LaunchError(f"runtime asset target contains an unsupported path component: {target}")
    if target == "/workspace" or not target.startswith("/workspace/"):
        raise LaunchError(f"runtime asset target must be below /workspace: {target}")
    fixed = (
        "/workspace/src/openevo-experiment", "/workspace/src/openevo-core",
        "/workspace/models", "/workspace/data", "/workspace/runs", "/workspace/cache",
    )
    for existing in [*fixed, *seen]:
        if target == existing or target.startswith(existing + "/") or existing.startswith(target + "/"):
            raise LaunchError(f"runtime asset target overlaps another mount: {target} and {existing}")


def stage_runtime_assets(
    profile: Mapping[str, Any], *, validate_only: bool = False
) -> list[dict[str, str]]:
    """Verify pinned runtime inputs; optionally publish content-addressed snapshots."""
    assets = profile.get("runtime_assets", [])
    overlays = profile.get("runtime_asset_overlays", [])
    if not isinstance(assets, list) or not isinstance(overlays, list):
        raise LaunchError("profile.runtime_assets must be a list")
    if not assets and not overlays:
        return []
    base_targets: dict[str, str] = {}
    base_sources = {str(item["target"]): str(item["source"]) for item in assets if isinstance(item, dict)}
    base_seen: list[str] = []
    for index, item in enumerate(assets):
        if not isinstance(item, dict) or set(item) != {"source", "target", "expected_type", "sha256"}:
            raise LaunchError(f"runtime_assets[{index}] requires exactly source,target,expected_type,sha256")
        _validate_runtime_asset_target(str(item.get("target", "")), base_seen)
        if item.get("expected_type") == "dir":
            base_targets[str(item["target"])] = "dir"
        base_seen.append(str(item["target"]))
    overlay_targets: list[str] = []
    staged_specs: list[tuple[dict[str, Any], str, bool]] = [(item, f"runtime_assets[{i}]", False) for i, item in enumerate(assets)]
    for index, item in enumerate(overlays):
        label = f"runtime_asset_overlays[{index}]"
        if not isinstance(item, dict) or set(item) != {"source", "target", "expected_type", "sha256", "base_target"}:
            raise LaunchError(f"{label} requires exactly source,target,expected_type,sha256,base_target")
        if item.get("expected_type") != "file":
            raise LaunchError(f"{label}.expected_type must be file")
        base_target, target = item.get("base_target"), item.get("target")
        if not isinstance(base_target, str) or base_targets.get(base_target) != "dir":
            raise LaunchError(f"{label}.base_target must name a declared directory runtime asset")
        parsed = PurePosixPath(target) if isinstance(target, str) else PurePosixPath(".")
        if (not parsed.is_absolute() or parsed.as_posix() != target or not target.startswith(base_target + "/")
                or any(not re.fullmatch(r"[A-Za-z0-9._-]+", part) for part in parsed.parts[1:])):
            raise LaunchError(f"{label}.target must be a normalized file path below its base_target")
        if any(target == existing or target.startswith(existing + "/") or existing.startswith(target + "/")
               for existing in overlay_targets):
            raise LaunchError(f"{label}.target duplicates or overlaps another overlay")
        overlay_targets.append(target)
        staged_specs.append((item, label, True))
    workspace = Path(str(profile["workspace"])).expanduser().resolve()
    staging_relative = profile.get("runtime_asset_staging_root")
    if not isinstance(staging_relative, str) or not staging_relative:
        raise LaunchError("runtime_assets requires runtime_asset_staging_root")
    staging_norm = Path(staging_relative)
    if staging_norm.is_absolute():
        if not (isinstance(profile.get("execution"), Mapping)
                and profile["execution"].get("mode") == "host-orchestrator"
                    and ((staging_norm != HOST_TMPFS_ROOT and staging_norm.is_relative_to(HOST_TMPFS_ROOT))
                         or _is_account_sealed_staging(workspace, staging_norm))
                    and ".." not in staging_norm.parts
                    and staging_norm.as_posix() == staging_relative):
            raise LaunchError("absolute runtime_asset_staging_root requires host-orchestrator inputs under /dev/shm or the sealed account cache")
        staging_root = staging_norm
        _reject_symlink_components(Path("/"), staging_root, "sealed runtime staging root", allow_missing=True)
        if _is_account_sealed_staging(workspace, staging_root):
            _validate_sealed_disk_ancestry(staging_root, allow_missing=True)
    else:
        if staging_norm == Path(".") or ".." in staging_norm.parts:
            raise LaunchError("runtime_asset_staging_root must be normalized without parent traversal")
        staging_root = workspace / staging_relative
        _reject_symlink_components(workspace, staging_root, "runtime asset staging root", allow_missing=True)
    fixed_sources = [
        workspace_path(profile, str(profile["source"][key]["path"]))
        for key in ("experiment", "openevo")
    ] + [workspace_path(profile, str(profile["paths"][key])) for key in ("models", "data", "runs", "cache")]
    staging_resolved = staging_root.resolve(strict=False)
    for fixed in fixed_sources:
        fixed_resolved = fixed.resolve(strict=False)
        if (staging_resolved == fixed_resolved
                or staging_resolved in fixed_resolved.parents
                or fixed_resolved in staging_resolved.parents):
            raise LaunchError("runtime_asset_staging_root must not overlap any fixed Docker bind source")
    if validate_only:
        if staging_root.exists():
            root_stat = staging_root.lstat()
            if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
                raise LaunchError("runtime asset staging root must be a real directory")
            if root_stat.st_uid != os.geteuid():
                raise LaunchError("runtime asset staging root must be owned by the launcher identity")
            if staging_norm.is_absolute():
                if _is_account_sealed_staging(workspace, staging_root):
                    _validate_sealed_disk_ancestry(staging_root)
                else:
                    _validate_sealed_tmpfs_ancestry(staging_root)
        elif staging_norm.is_absolute() and not _is_account_sealed_staging(workspace, staging_root):
            _validate_sealed_tmpfs_creation_parent(staging_root)
    else:
        staging_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root_stat = staging_root.lstat()
        if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
            raise LaunchError("runtime asset staging root must be a real directory")
        if root_stat.st_uid != os.geteuid():
            raise LaunchError("runtime asset staging root must be owned by the launcher identity")
        if staging_norm.is_absolute():
            if _is_account_sealed_staging(workspace, staging_root):
                _validate_sealed_disk_ancestry(staging_root)
            else:
                _validate_sealed_tmpfs_ancestry(staging_root)
        staging_root.chmod(0o700)
        staging_root = staging_root.resolve()

    staged: list[dict[str, str]] = []
    seen_targets: list[str] = []
    for item, label, is_overlay in staged_specs:
        source_name = item["source"]
        target = item["target"]
        expected_type = item["expected_type"]
        expected_sha = item["sha256"]
        source_path = Path(source_name) if isinstance(source_name, str) else Path()
        if (not isinstance(source_name, str) or not source_name or source_path.is_absolute()
                or ".." in source_path.parts):
            raise LaunchError(f"{label}.source must be a workspace-relative path")
        if not isinstance(target, str):
            raise LaunchError(f"{label}.target must be a string")
        if not is_overlay:
            _validate_runtime_asset_target(target, seen_targets)
            seen_targets.append(target)
        if expected_type not in {"file", "dir"}:
            raise LaunchError(f"{label}.expected_type must be file or dir")
        if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise LaunchError(f"{label}.sha256 must be 64 lowercase hex characters")
        source = workspace / source_name
        _reject_symlink_components(workspace, source, f"{label}.source")
        source = source.resolve()
        if "," in str(source):
            raise LaunchError(f"runtime asset source path cannot contain Docker mount delimiter: {source}")
        if is_overlay:
            base_source_name = base_sources[str(item["base_target"])]
            base_source = (workspace / base_source_name).resolve()
            relative_target = PurePosixPath(target).relative_to(PurePosixPath(str(item["base_target"])))
            destination = base_source.joinpath(*relative_target.parts)
            _reject_symlink_components(base_source, destination, f"{label}.base destination", allow_missing=True)
            try:
                parent_stat = destination.parent.lstat().st_mode
            except OSError as exc:
                raise LaunchError(f"{label}.target parent must be an existing directory in its base asset: {target}") from exc
            if not stat.S_ISDIR(parent_stat) or stat.S_ISLNK(parent_stat):
                raise LaunchError(f"{label}.target parent must be a real directory in its base asset: {target}")
            try:
                destination_stat = destination.lstat().st_mode
            except FileNotFoundError:
                destination_stat = None
            if destination_stat is not None and (
                not stat.S_ISREG(destination_stat) or stat.S_ISLNK(destination_stat)
            ):
                raise LaunchError(f"{label}.target must be absent or an existing regular file in its base asset: {target}")
        actual_sha = runtime_asset_digest(source, expected_type)
        if actual_sha != expected_sha:
            raise LaunchError(f"runtime asset SHA-256 mismatch for {source}: expected {expected_sha}, observed {actual_sha}")

        if validate_only:
            staged_item = {
                "source": str(source), "target": target,
                "expected_type": expected_type, "sha256": expected_sha,
            }
            if is_overlay:
                staged_item["base_target"] = str(item["base_target"])
            staged.append(staged_item)
            continue

        final_parent = staging_root / expected_sha
        try:
            final_parent.mkdir(mode=0o700)
        except FileExistsError:
            pass
        parent_stat = final_parent.lstat()
        if not stat.S_ISDIR(parent_stat.st_mode) or stat.S_ISLNK(parent_stat.st_mode):
            raise LaunchError(f"runtime snapshot bucket must be a real directory: {final_parent}")
        if parent_stat.st_uid != os.geteuid():
            raise LaunchError(f"runtime snapshot bucket has unexpected owner: {final_parent}")
        final_parent.chmod(0o700)
        staged_item = _stage_runtime_asset_snapshot(source, final_parent, target, expected_type, expected_sha)
        if is_overlay:
            staged_item["base_target"] = str(item["base_target"])
        staged.append(staged_item)
    return staged


def _publish_host_effective_source(
    temp_tree: Path,
    effective_parent: Path,
    effective_sha: str,
    target: str,
    base_sha: str,
    overlay_rows: list[tuple[Mapping[str, str], Path]],
) -> dict[str, str]:
    """Serialize content-addressed tree publication and its stable receipt."""
    lock_path = effective_parent / f".{effective_sha}.lock"
    with _exclusive_owned_lock(lock_path):
        final = effective_parent / effective_sha
        if final.exists():
            if final.is_symlink() or runtime_asset_digest(final, "dir") != effective_sha:
                raise LaunchError(f"existing host effective source snapshot is corrupt: {final}")
            _remove_host_source_temp_tree(temp_tree)
        else:
            temp_tree.chmod(0o755)
            os.rename(temp_tree, final)
            final.chmod(0o555)
        if runtime_asset_digest(final, "dir") != effective_sha:
            raise LaunchError("host effective source changed during publication")
        entries = [final, *final.rglob("*")]
        if any(path.is_symlink() or path.lstat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) for path in entries):
            raise LaunchError("host effective source must be a read-only regular tree")
        identity = {
            "source_root": str(final),
            "target": target,
            "expected_type": "dir",
            "sha256": effective_sha,
            "base_sha256": base_sha,
            "overlay_targets": ",".join(str(item["target"]) for item, _ in overlay_rows),
        }
        receipt_path = effective_parent / f"{effective_sha}.receipt.json"
        receipt = {
            "schema": "openevo.host-runtime-source.v1",
            **identity,
            "overlays": [{"target": str(item["target"]), "sha256": str(item["sha256"])} for item, _ in overlay_rows],
        }
        atomic_json(receipt_path, receipt)
        identity["receipt_path"] = str(receipt_path)
        identity["receipt_sha256"] = prelaunch_zero_formal.sha256_file(receipt_path)
        return identity


def _grant_host_source_read_access(path: Path, gid: int, mode: int) -> None:
    """Grant only the declared run group traversal/read access to staged source."""
    try:
        info = path.lstat()
    except OSError as exc:
        raise LaunchError(f"host source access path is unavailable: {path}: {exc}") from exc
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_uid != os.geteuid():
        raise LaunchError(f"host source access path must be a launcher-owned real directory: {path}")
    try:
        os.chown(path, info.st_uid, gid)
        path.chmod(mode)
    except OSError as exc:
        raise LaunchError(f"cannot grant declared run-group source access to {path}: {exc}") from exc
    updated = path.lstat()
    if updated.st_gid != gid or stat.S_IMODE(updated.st_mode) != mode:
        raise LaunchError(f"host source access mode/group did not persist: {path}")


def _grant_host_source_tree_read_access(path: Path, gid: int) -> None:
    """Make the immutable effective tree readable only through its declared group."""
    for entry in [path, *path.rglob("*")]:
        info = entry.lstat()
        if stat.S_ISLNK(info.st_mode) or not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
            raise LaunchError(f"host effective source contains an unsupported entry: {entry}")
        mode = 0o550 if stat.S_ISDIR(info.st_mode) or info.st_mode & 0o111 else 0o440
        try:
            os.chown(entry, info.st_uid, gid)
            entry.chmod(mode)
        except OSError as exc:
            raise LaunchError(f"cannot grant declared run-group read access to {entry}: {exc}") from exc
        updated = entry.lstat()
        if updated.st_gid != gid or stat.S_IMODE(updated.st_mode) != mode:
            raise LaunchError(f"host effective source group access did not persist: {entry}")


def _remove_host_source_temp_tree(path: Path) -> None:
    if path.exists() and not path.is_symlink():
        for entry in sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if entry.is_symlink():
                continue
            mode = entry.lstat().st_mode
            if stat.S_ISDIR(mode):
                entry.chmod(stat.S_IRWXU)
            elif stat.S_ISREG(mode):
                entry.chmod(mode | stat.S_IWUSR)
        path.chmod(stat.S_IRWXU)
    shutil.rmtree(path, ignore_errors=True)


def stage_host_runtime_source(profile: Mapping[str, Any]) -> dict[str, str] | None:
    """Build the exact host-importable source tree from staged assets/overlays."""
    execution = profile.get("execution") if isinstance(profile.get("execution"), Mapping) else {}
    if execution.get("mode") != "host-orchestrator":
        return None
    snapshots = profile.get("_runtime_asset_snapshots", [])
    declared = profile.get("runtime_assets", [])
    declared_overlays = profile.get("runtime_asset_overlays", [])
    if not snapshots and not declared and not declared_overlays:
        return None
    target = execution.get("runtime_source_target")
    if not isinstance(target, str) or not target:
        raise LaunchError("host-orchestrator runtime assets require execution.runtime_source_target")
    source_dirs = [asset for asset in snapshots if asset.get("expected_type") == "dir"]
    if len(source_dirs) != 1 or source_dirs[0].get("target") != target:
        raise LaunchError("host-orchestrator requires exactly one staged runtime source directory at runtime_source_target")
    if any(asset.get("expected_type") != "dir" for asset in snapshots if "base_target" not in asset):
        raise LaunchError("host-orchestrator cannot ignore standalone runtime file assets")
    if any(asset.get("base_target") != target for asset in snapshots if "base_target" in asset):
        raise LaunchError("host-orchestrator runtime overlay must target its effective source directory")

    base = source_dirs[0]
    base_root = Path(base["source"])
    staging_root = (Path(str(profile["workspace"])).expanduser().resolve()
                    / str(profile["runtime_asset_staging_root"])).resolve()
    try:
        base_root.resolve(strict=True).relative_to(staging_root)
    except (OSError, ValueError) as exc:
        raise LaunchError("host runtime source snapshot escapes runtime asset staging root") from exc
    if base_root.is_symlink() or runtime_asset_digest(base_root, "dir") != base["sha256"]:
        raise LaunchError("host runtime source snapshot changed before effective-tree staging")

    overlay_rows: list[tuple[Mapping[str, str], Path]] = []
    overlay_targets = {str(item["target"]): item for item in declared_overlays}
    for item in snapshots:
        if "base_target" not in item:
            continue
        raw_target = str(item["target"])
        rel = PurePosixPath(raw_target).relative_to(PurePosixPath(target))
        if not rel.parts or any(part in {"", ".", ".."} for part in rel.parts):
            raise LaunchError("host runtime overlay has an unsafe relative path")
        source = Path(item["source"])
        if source.is_symlink() or runtime_asset_digest(source, "file") != item["sha256"]:
            raise LaunchError(f"host runtime overlay changed before effective-tree staging: {source}")
        overlay_rows.append((item, rel))
    if len(overlay_rows) != len(declared_overlays):
        raise LaunchError("host runtime overlays were not all staged")
    if set(overlay_targets) != {str(item["target"]) for item, _ in overlay_rows}:
        raise LaunchError("host runtime overlay identities do not match the declared profile")

    workspace = Path(str(profile["workspace"])).expanduser().resolve()
    effective_parent = staging_root / "host-effective-sources"
    if staging_root.is_relative_to(HOST_TMPFS_ROOT) or _is_account_sealed_staging(workspace, staging_root):
        _reject_symlink_components(Path("/"), effective_parent, "sealed host effective source staging root", allow_missing=True)
    else:
        _reject_symlink_components(workspace, effective_parent, "host effective source staging root", allow_missing=True)
    if _is_account_sealed_staging(workspace, staging_root):
        _validate_sealed_disk_ancestry(effective_parent, allow_missing=True)
    effective_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent_stat = effective_parent.lstat()
    if not stat.S_ISDIR(parent_stat.st_mode) or stat.S_ISLNK(parent_stat.st_mode) or parent_stat.st_uid != os.geteuid():
        raise LaunchError("host effective source staging root must be a launcher-owned real directory")
    if staging_root.is_relative_to(HOST_TMPFS_ROOT):
        _validate_sealed_tmpfs_ancestry(effective_parent)
    elif _is_account_sealed_staging(workspace, staging_root):
        _validate_sealed_disk_ancestry(effective_parent)
    identity = profile.get("identity") if isinstance(profile.get("identity"), Mapping) else {}
    run_gid = identity.get("gid")
    if not isinstance(run_gid, int) or run_gid <= 0:
        raise LaunchError("host-orchestrator runtime source requires a positive declared identity.gid")
    _grant_host_source_read_access(staging_root, run_gid, 0o710)
    _grant_host_source_read_access(effective_parent, run_gid, 0o710)

    temp_parent = Path(tempfile.mkdtemp(prefix=".host-source-", dir=effective_parent))
    temp_tree = temp_parent / "tree"
    try:
        shutil.copytree(base_root, temp_tree, symlinks=False)
        for path in sorted(temp_tree.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                path.chmod(stat.S_IRWXU)
            elif stat.S_ISREG(mode):
                executable = bool(mode & 0o111)
                path.chmod(stat.S_IRUSR | stat.S_IWUSR | (stat.S_IXUSR if executable else 0))
            else:
                raise LaunchError(f"host effective source contains a special file: {path}")
        for item, rel in overlay_rows:
            destination = temp_tree.joinpath(*rel.parts)
            try:
                destination.relative_to(temp_tree)
            except ValueError as exc:
                raise LaunchError("host runtime overlay escapes its effective source tree") from exc
            parent = destination.parent
            try:
                parent.relative_to(temp_tree)
                parent_stat = parent.lstat()
            except (ValueError, OSError) as exc:
                raise LaunchError(f"host runtime overlay parent is missing or unsafe: {parent}") from exc
            if not stat.S_ISDIR(parent_stat.st_mode) or stat.S_ISLNK(parent_stat.st_mode):
                raise LaunchError(f"host runtime overlay parent is not a real directory: {parent}")
            try:
                destination_stat = destination.lstat()
            except FileNotFoundError:
                destination_stat = None
            if destination_stat is not None and (
                not stat.S_ISREG(destination_stat.st_mode) or stat.S_ISLNK(destination_stat.st_mode)
            ):
                raise LaunchError(f"host runtime overlay target is not a regular source file: {destination}")
            source_path = Path(item["source"])
            flags = os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
            flags |= os.O_CREAT | os.O_EXCL if destination_stat is None else os.O_TRUNC
            try:
                output_fd = os.open(destination, flags, 0o600)
            except OSError as exc:
                raise LaunchError(f"host runtime overlay target could not be safely created: {destination}") from exc
            try:
                if not stat.S_ISREG(os.fstat(output_fd).st_mode):
                    raise LaunchError(f"host runtime overlay target is not a regular source file: {destination}")
                with os.fdopen(output_fd, "wb", closefd=True) as output_file, source_path.open("rb") as input_file:
                    shutil.copyfileobj(input_file, output_file)
                    output_file.flush()
                    os.fsync(output_file.fileno())
            except BaseException:
                try:
                    os.close(output_fd)
                except OSError:
                    pass
                raise
            destination.chmod(stat.S_IRUSR | stat.S_IWUSR | (stat.S_IXUSR if source_path.stat().st_mode & 0o111 else 0))
        for path in sorted(temp_tree.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                path.chmod(0o555)
            elif stat.S_ISREG(mode):
                path.chmod(0o555 if mode & 0o111 else 0o444)
        temp_tree.chmod(0o555)
        effective_sha = runtime_asset_digest(temp_tree, "dir")
        result = _publish_host_effective_source(
            temp_tree,
            effective_parent,
            effective_sha,
            target,
            str(base["sha256"]),
            overlay_rows,
        )
        _grant_host_source_tree_read_access(Path(result["source_root"]), run_gid)
        receipt_path = Path(result["receipt_path"])
        receipt_stat = receipt_path.lstat()
        os.chown(receipt_path, receipt_stat.st_uid, run_gid)
        receipt_path.chmod(0o440)
        result["receipt_sha256"] = prelaunch_zero_formal.sha256_file(receipt_path)
        return result
    finally:
        _remove_host_source_temp_tree(temp_parent)


def validate_host_runtime_source(profile: Mapping[str, Any]) -> None:
    identity = profile.get("_host_runtime_source")
    execution = profile.get("execution") if isinstance(profile.get("execution"), Mapping) else {}
    has_runtime_assets = bool(profile.get("runtime_assets") or profile.get("runtime_asset_overlays"))
    if execution.get("mode") != "host-orchestrator" or not has_runtime_assets:
        if identity is not None:
            raise LaunchError("host effective source is present for a profile that does not consume it")
        return
    if not isinstance(identity, Mapping):
        raise LaunchError("host-orchestrator runtime assets have no effective source snapshot")
    root = Path(str(identity.get("source_root", "")))
    workspace = Path(str(profile["workspace"])).expanduser().resolve()
    staging_root = (workspace / str(profile.get("runtime_asset_staging_root", ""))).resolve()
    disk_sealed = _is_account_sealed_staging(workspace, staging_root)
    sealed_tmpfs = HOST_TMPFS_ROOT.resolve(strict=True) if os.geteuid() == 0 and not disk_sealed else None
    if disk_sealed:
        _validate_sealed_disk_ancestry(root.parent)
    try:
        root.resolve(strict=True).relative_to(staging_root)
        if sealed_tmpfs is not None:
            staging_root.relative_to(sealed_tmpfs)
    except (OSError, ValueError) as exc:
        raise LaunchError("host effective source must stay under its /dev/shm sealed staging root") from exc
    if root.is_symlink() or runtime_asset_digest(root, "dir") != identity.get("sha256"):
        raise LaunchError("host effective source identity changed")
    entries = [root, *root.rglob("*")]
    run_identity = profile.get("identity") if isinstance(profile.get("identity"), Mapping) else {}
    run_gid = run_identity.get("gid")
    if not isinstance(run_gid, int) or run_gid <= 0:
        raise LaunchError("host-orchestrator runtime source requires a positive declared identity.gid")
    if any(
        path.is_symlink()
        or path.lstat().st_uid != os.geteuid()
        or path.lstat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        or path.lstat().st_gid != run_gid
        or not (path.lstat().st_mode & stat.S_IRGRP)
        or (stat.S_ISDIR(path.lstat().st_mode) and not (path.lstat().st_mode & stat.S_IXGRP))
        for path in entries
    ):
        raise LaunchError("host effective source contains a writable or linked entry")
    for parent in (staging_root, root.parent):
        info = parent.lstat()
        if (info.st_uid != os.geteuid() or info.st_gid != run_gid or not (info.st_mode & stat.S_IXGRP)
                or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)):
            raise LaunchError(f"host effective source parent is not traversable read-only by declared run group: {parent}")
    if disk_sealed:
        run_uid = run_identity.get("uid")
        for parent in root.parents:
            info = parent.lstat()
            execute_bit = stat.S_IXUSR if info.st_uid == run_uid else (stat.S_IXGRP if info.st_gid == run_gid else stat.S_IXOTH)
            if not info.st_mode & execute_bit:
                raise LaunchError(f"host runtime staging ancestor is not traversable by declared run identity: {parent}")
    if sealed_tmpfs is not None:
        sealed_info = sealed_tmpfs.lstat()
        if sealed_info.st_uid != 0 or not stat.S_ISDIR(sealed_info.st_mode) or not (sealed_info.st_mode & stat.S_ISVTX):
            raise LaunchError("/dev/shm is not a root-owned sticky directory")
        parent = staging_root
        while parent != sealed_tmpfs:
            info = parent.lstat()
            if info.st_uid != 0 or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
                raise LaunchError(f"host runtime staging ancestor is not root-owned and immutable: {parent}")
            parent = parent.parent
    receipt_path = Path(str(identity.get("receipt_path", "")))
    if receipt_path.is_symlink() or not receipt_path.is_file() or prelaunch_zero_formal.sha256_file(receipt_path) != identity.get("receipt_sha256"):
        raise LaunchError("host effective source receipt is missing or changed")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchError(f"host effective source receipt is unreadable: {exc}") from exc
    if any(receipt.get(key) != identity.get(key) for key in ("source_root", "target", "sha256", "base_sha256")):
        raise LaunchError("host effective source receipt does not match the staged profile identity")


_HOST_IMPORT_PROBE = r'''import hashlib, importlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve(strict=True)
names = json.loads(sys.argv[2])
loaded = {}
for name in names:
    module = importlib.import_module(name)
    origin = getattr(module, "__file__", None)
    if not isinstance(origin, str):
        raise RuntimeError(f"module has no concrete source origin: {name}")
    path = pathlib.Path(origin).resolve(strict=True)
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"module origin escapes effective source tree: {name}: {path}") from exc
    if not path.is_file() or path.suffix not in {".py", ".pyc"}:
        raise RuntimeError(f"module origin is not a Python source/bytecode file: {name}: {path}")
    loaded[name] = {"path": str(path), "relative_path": rel, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
print(json.dumps({"modules": loaded, "python": sys.executable, "python_version": sys.version,
                  "sys_path": sys.path, "no_user_site": sys.flags.no_user_site}, sort_keys=True))
'''


def _module_source_candidates(root: Path, module_name: str) -> tuple[Path, ...]:
    if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module_name):
        raise LaunchError(f"invalid runtime import module name: {module_name!r}")
    relative = Path(*module_name.split("."))
    return (root / relative.with_suffix(".py"), root / relative / "__init__.py")


def _host_runtime_entrypoint_identity(profile: Mapping[str, Any], root: Path) -> dict[str, str]:
    """Require the actual host argv entrypoint to be the frozen source tree."""
    execution = profile.get("execution") if isinstance(profile.get("execution"), Mapping) else {}
    entrypoint = execution.get("runtime_entrypoint")
    command = profile.get("command")
    if not isinstance(command, list) or not command or not isinstance(command[0], str):
        raise LaunchError("host runtime entrypoint requires a Python executable at command[0]")
    if not isinstance(entrypoint, Mapping):
        raise LaunchError("host-orchestrator runtime assets require execution.runtime_entrypoint")
    kind = entrypoint.get("kind")
    if kind == "module":
        module = entrypoint.get("module")
        if not isinstance(module, str):
            raise LaunchError("host runtime module entrypoint must name a module")
        matches = [path for path in _module_source_candidates(root, module) if path.is_file()]
        if len(matches) != 1:
            raise LaunchError("host runtime module entrypoint must resolve to one frozen source file")
        source = matches[0].resolve(strict=True)
        if root not in source.parents or command[1:3] != ["-m", module]:
            raise LaunchError("formal host command must execute its declared module from the effective source tree")
        return {"kind": "module", "module": module, "path": str(source),
                "relative_path": source.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    if kind == "script":
        raw_path = entrypoint.get("path")
        if not isinstance(raw_path, str):
            raise LaunchError("host runtime script entrypoint must name a relative Python file")
        relative = PurePosixPath(raw_path)
        if relative.is_absolute() or relative.as_posix() != raw_path or any(part in {"", ".", ".."} for part in relative.parts):
            raise LaunchError("host runtime script entrypoint path must be normalized and relative")
        source = root.joinpath(*relative.parts)
        if source.is_symlink() or not source.is_file() or root not in source.resolve(strict=True).parents:
            raise LaunchError("host runtime script entrypoint must be a regular file in the effective source tree")
        expected_arg = "${SOURCE_ROOT}/" + relative.as_posix()
        if len(command) < 2 or command[1] != expected_arg:
            raise LaunchError("formal host command must execute the declared script from the effective source tree")
        return {"kind": "script", "path": str(source.resolve(strict=True)),
                "relative_path": relative.as_posix(),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    raise LaunchError("host runtime entrypoint kind must be 'module' or 'script'")


def probe_host_runtime_imports(
    profile: Mapping[str, Any], *, cwd: Path | None = None
) -> dict[str, Any] | None:
    """Import declared host runtime modules on CPU and bind origins to the snapshot."""
    execution = profile.get("execution") if isinstance(profile.get("execution"), Mapping) else {}
    source = profile.get("_host_runtime_source")
    has_assets = bool(profile.get("runtime_assets") or profile.get("runtime_asset_overlays"))
    if execution.get("mode") != "host-orchestrator" or not has_assets:
        return None
    if not isinstance(source, Mapping):
        raise LaunchError("host runtime import probe requires a staged effective source tree")
    modules = execution.get("runtime_import_modules")
    if (
        not isinstance(modules, list)
        or not modules
        or any(not isinstance(name, str) or not name for name in modules)
        or len(set(modules)) != len(modules)
    ):
        raise LaunchError("host-orchestrator runtime assets require unique execution.runtime_import_modules")

    root = Path(str(source["source_root"])).resolve(strict=True)
    entrypoint_identity = _host_runtime_entrypoint_identity(profile, root)
    expected_sources: dict[str, dict[str, str]] = {}
    for name in modules:
        matches = [path for path in _module_source_candidates(root, name) if path.is_file()]
        if len(matches) != 1:
            raise LaunchError(f"runtime import module must resolve to exactly one source file in effective tree: {name}")
        expected = matches[0].resolve(strict=True)
        if root not in expected.parents:
            raise LaunchError(f"runtime import source escapes effective tree: {name}")
        expected_sources[name] = {
            "path": str(expected),
            "relative_path": expected.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(expected.read_bytes()).hexdigest(),
        }

    command = profile.get("command")
    runtime_python = execution.get("runtime_python")
    if not isinstance(command, list) or not command or not isinstance(command[0], str):
        raise LaunchError("host runtime import probe requires a Python executable at command[0]")
    executable = runtime_python if isinstance(runtime_python, str) else command[0]
    if runtime_python is not None and runtime_python != command[0]:
        raise LaunchError("execution.runtime_python must equal the formal host command executable")
    executable_path = Path(executable).expanduser()
    if not executable_path.is_absolute():
        found = shutil.which(executable)
        if not found:
            raise LaunchError(f"host runtime Python executable is unavailable: {executable}")
        executable_path = Path(found)
    try:
        # Keep the requested path for invocation. Virtualenv interpreters are
        # commonly symlinks to the system Python; resolving before exec makes
        # Python ignore the adjacent pyvenv.cfg and silently drops venv packages.
        executable_path = Path(os.path.abspath(executable_path))
        executable_realpath = executable_path.resolve(strict=True)
    except OSError as exc:
        raise LaunchError(f"host runtime Python executable cannot be resolved: {executable}: {exc}") from exc
    if not executable_path.is_file() or not executable_realpath.is_file():
        raise LaunchError(f"host runtime Python executable is not a regular file: {executable_path}")
    executable_sha256 = prelaunch_zero_formal.sha256_file(executable_realpath)

    env = dict(os.environ)
    env.pop("PYTHONHOME", None)
    env["PYTHONPATH"] = str(root)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OPENEVO_EFFECTIVE_SOURCE_ROOT"] = str(root)
    env["OPENEVO_EFFECTIVE_SOURCE_SHA256"] = str(source["sha256"])
    try:
        result = subprocess.run(
            [str(executable_path), "-c", _HOST_IMPORT_PROBE, str(root), json.dumps(modules)],
            cwd=(cwd or Path.cwd()).resolve(strict=True), env=env, check=False,
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LaunchError(f"host runtime CPU import probe could not complete: {exc}") from exc
    try:
        observed_realpath = executable_path.resolve(strict=True)
        observed_sha256 = prelaunch_zero_formal.sha256_file(observed_realpath)
    except OSError as exc:
        raise LaunchError(f"host runtime Python executable changed during CPU import probe: {exc}") from exc
    if observed_realpath != executable_realpath or observed_sha256 != executable_sha256:
        raise LaunchError("host runtime Python executable changed during CPU import probe")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "no output").strip()[-4000:]
        raise LaunchError(f"host runtime CPU import probe failed (exit {result.returncode}): {detail}")
    try:
        observed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise LaunchError("host runtime CPU import probe did not return valid JSON") from exc
    loaded = observed.get("modules") if isinstance(observed, dict) else None
    if not isinstance(loaded, dict) or set(loaded) != set(modules):
        raise LaunchError("host runtime CPU import probe returned an incomplete module set")
    try:
        observed_python = Path(str(observed.get("python", ""))).resolve(strict=True)
    except OSError as exc:
        raise LaunchError("host runtime CPU import probe returned an invalid interpreter path") from exc
    if observed_python != executable_realpath:
        raise LaunchError("host runtime CPU import probe used a different Python executable")
    for name in modules:
        if loaded[name] != expected_sources[name]:
            raise LaunchError(
                f"host runtime imported a different source than the frozen tree: {name}; "
                f"expected={expected_sources[name]!r} observed={loaded[name]!r}"
            )
    identity: dict[str, Any] = {
        "schema": "openevo.host-runtime-import-identity.v1",
        "status": "PASS",
        "source_root": str(root),
        "source_sha256": str(source["sha256"]),
        "python_executable": str(observed.get("python")),
        "python_realpath": str(executable_realpath),
        "python_sha256": executable_sha256,
        "python_version": observed.get("python_version"),
        "python_sys_path": observed.get("sys_path"),
        "python_no_user_site": observed.get("no_user_site"),
        "entrypoint": entrypoint_identity,
        "modules": expected_sources,
        "module_list_sha256": hashlib.sha256(json.dumps(modules, separators=(",", ":")).encode()).hexdigest(),
        "probe_sha256": hashlib.sha256(_HOST_IMPORT_PROBE.encode()).hexdigest(),
    }
    identity["identity_sha256"] = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return identity


def verify_host_runtime_imports(profile: Mapping[str, Any]) -> None:
    expected = profile.get("_host_runtime_import_identity")
    if expected is None:
        execution = profile.get("execution") if isinstance(profile.get("execution"), Mapping) else {}
        if execution.get("mode") == "host-orchestrator" and (profile.get("runtime_assets") or profile.get("runtime_asset_overlays")):
            raise LaunchError("host runtime source is missing its CPU import identity")
        return
    observed = probe_host_runtime_imports(profile)
    if not isinstance(observed, Mapping) or observed.get("identity_sha256") != expected.get("identity_sha256"):
        raise LaunchError("host runtime import identity changed after pre-GPU qualification")
    receipt_path_value = expected.get("receipt_path")
    if receipt_path_value is not None:
        receipt_path = Path(str(receipt_path_value))
        if receipt_path.is_symlink() or not receipt_path.is_file() or prelaunch_zero_formal.sha256_file(receipt_path) != expected.get("receipt_sha256"):
            raise LaunchError("host runtime import identity receipt is missing or changed")


def git_identity(path: Path, expected_revision: str) -> None:
    if not (path / ".git").exists():
        raise LaunchError(f"Git checkout missing: {path}")
    actual = _run(["git", "-C", str(path), "rev-parse", "HEAD"]).stdout.strip()
    if actual != expected_revision:
        raise LaunchError(f"Git revision mismatch at {path}: expected {expected_revision}, observed {actual}")
    dirty = _run(["git", "-C", str(path), "status", "--porcelain"]).stdout.strip()
    if dirty:
        raise LaunchError(f"Git checkout is dirty: {path}")


def validate_sources_and_paths(
    profile: Mapping[str, Any], *, stage_runtime: bool = True
) -> None:
    for key in ("experiment", "openevo"):
        source = profile["source"][key]
        git_identity(workspace_path(profile, str(source["path"])), str(source["revision"]))
    paths = profile["paths"]
    if not isinstance(paths, dict):
        raise LaunchError("profile.paths must be an object")
    for key in ("models", "data", "runs", "cache"):
        if key not in paths:
            raise LaunchError(f"profile.paths missing {key}")
        target = workspace_path(profile, str(paths[key]))
        if not target.exists():
            raise LaunchError(f"persistent path missing: {target}")
    for item in profile.get("required_paths", []):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise LaunchError("required_paths entries need a path")
        target = workspace_path(profile, item["path"])
        if not target.exists():
            raise LaunchError(f"required path missing: {target}")
        expected_type = str(item.get("expected_type") or ("dir" if target.is_dir() else "file"))
        try:
            validate_host_path(
                {
                    "host_path": str(target),
                    "expected_type": expected_type,
                    "readable": True,
                    "writable": bool(item.get("writable", False)),
                },
                f"required_paths.{item['path']}",
            )
        except PathContractError as exc:
            raise LaunchError(f"required path contract failed: {exc}") from exc

    path_specs = {
        "experiment": {
            "host_path": str(workspace_path(profile, str(profile["source"]["experiment"]["path"]))),
            "container_path": "/workspace/src/openevo-experiment",
            "expected_type": "dir",
            "access": "ro",
        },
        "openevo": {
            "host_path": str(workspace_path(profile, str(profile["source"]["openevo"]["path"]))),
            "container_path": "/workspace/src/openevo-core",
            "expected_type": "dir",
            "access": "ro",
        },
        "models": {
            "host_path": str(workspace_path(profile, str(paths["models"]))),
            "container_path": "/workspace/models",
            "expected_type": "dir",
            "access": "ro",
        },
        "data": {
            "host_path": str(workspace_path(profile, str(paths["data"]))),
            "container_path": "/workspace/data",
            "expected_type": "dir",
            "access": "ro",
        },
        "runs": {
            "host_path": str(workspace_path(profile, str(paths["runs"]))),
            "container_path": "/workspace/runs",
            "expected_type": "dir",
            "access": "rw",
            "writable": True,
        },
        "cache": {
            "host_path": str(workspace_path(profile, str(paths["cache"]))),
            "container_path": "/workspace/cache",
            "expected_type": "dir",
            "access": "rw",
            "writable": True,
        },
    }
    try:
        validate_contract({"paths": path_specs}, probe=run_cpu_no_gpu_probe)
    except PathContractError as exc:
        raise LaunchError(f"host/container path contract failed before GPU allocation: {exc}") from exc
    if profile.get("runtime_assets") or profile.get("runtime_asset_overlays"):
        if not isinstance(profile, dict):
            raise LaunchError("runtime asset profiles must be mutable during pre-GPU staging")
        profile["_runtime_asset_snapshots"] = stage_runtime_assets(
            profile, validate_only=not stage_runtime
        )
        if stage_runtime:
            if isinstance(profile.get("execution"), Mapping) and profile["execution"].get("mode") == "host-orchestrator":
                profile["_host_runtime_source"] = stage_host_runtime_source(profile)
                profile["_host_runtime_import_identity"] = probe_host_runtime_imports(profile)


def image_id(image: str) -> str:
    if shutil.which("docker") is None:
        raise LaunchError("docker is not available")
    value = _run(["docker", "image", "inspect", image, "--format", "{{.Id}}"] ).stdout.strip()
    if not IMAGE_ID_RE.fullmatch(value):
        raise LaunchError(f"image has no immutable local ID: {image}")
    return value


def resolve_image_identity(profile: Mapping[str, Any]) -> str:
    """Resolve the provenance reference, or the frozen ID if the tag is absent."""
    expected = str(profile["image_id"])
    try:
        observed = image_id(str(profile["image"]))
    except subprocess.CalledProcessError:
        try:
            observed = image_id(expected)
        except (LaunchError, subprocess.CalledProcessError) as pinned_error:
            raise LaunchError(
                f"image reference unavailable ({profile['image']}) and frozen "
                f"image ID unavailable ({expected})"
            ) from pinned_error
    if observed != expected:
        raise LaunchError(
            "image identity mismatch: "
            f"reference {profile['image']} expected {expected}, observed {observed}"
        )
    return expected


def snapshot_gpus() -> list[dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        raise LaunchError("nvidia-smi is unavailable")
    rows = _run([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]).stdout
    cards = parse_gpu_rows(rows)
    container_owners, owner_query_error = running_gpu_container_owners()
    for card in cards:
        result = _run([
            "nvidia-smi", "-i", card["uuid"],
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ], check=False)
        card["compute_apps"] = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        card["compute_apps_query_error"] = None if result.returncode == 0 else (result.stderr.strip() or "compute-app query failed")
        card["device_handles"] = visible_device_handles(card["index"])
        monitoring, blocking, _ = classify_device_handles(card["device_handles"])
        card["monitoring_handles"] = monitoring
        card["blocking_device_handles"] = blocking
        card["gpu_owner_containers"] = container_owners.get(card["uuid"], [])
        card["gpu_owner_query_error"] = owner_query_error
    return cards


def idle_cards(
    cards: Iterable[Mapping[str, Any]],
    selected: Iterable[str],
    *,
    max_memory_mib: int,
    max_utilization: int,
) -> tuple[bool, list[str]]:
    by_uuid = {str(card["uuid"]): card for card in cards}
    failures: list[str] = []
    for uuid in selected:
        card = by_uuid.get(uuid)
        if card is None:
            failures.append(f"{uuid}:not-visible")
            continue
        if int(card["memory_used_mib"]) > max_memory_mib:
            failures.append(f"{uuid}:memory={card['memory_used_mib']}MiB")
        if int(card["utilization_gpu_percent"]) > max_utilization:
            failures.append(f"{uuid}:utilization={card['utilization_gpu_percent']}%")
        if card.get("compute_apps_query_error"):
            failures.append(f"{uuid}:compute-app-query-failed")
        elif card.get("compute_apps"):
            failures.append(f"{uuid}:compute-apps")
        if card.get("gpu_owner_query_error"):
            failures.append(f"{uuid}:gpu-owner-query-failed")
        elif card.get("gpu_owner_containers"):
            failures.append(f"{uuid}:gpu-owner-container")
        if not canonical_idle(card, max_memory_mib=max_memory_mib, max_util_percent=max_utilization):
            monitoring, blocking, classification_failed = classify_device_handles(card.get("device_handles", []))
        else:
            monitoring, blocking, classification_failed = [], [], False
        if blocking:
            failures.append(
                f"{uuid}:device-handle-classification-failed"
                if classification_failed
                else f"{uuid}:device-handle"
            )
    return not failures, failures


def _capacity_cards_minimum_free(
    cards: Iterable[Mapping[str, Any]],
    selected: Iterable[str],
    *,
    minimum_free_memory_mib: int,
    max_utilization: int = 100,
) -> tuple[bool, list[str]]:
    by_uuid = {str(card["uuid"]): card for card in cards}
    failures: list[str] = []
    for uuid in selected:
        card = by_uuid.get(uuid)
        if card is None:
            failures.append(f"{uuid}:not-visible")
            continue
        eligible, reasons = co_resident_capacity(
            card,
            minimum_free_memory_mib=minimum_free_memory_mib,
            max_util_percent=max_utilization,
        )
        failures.extend(f"{uuid}:{reason}" for reason in reasons)
        if not eligible and not reasons:
            failures.append(f"{uuid}:capacity-ineligible")
    return not failures, failures


def _capacity_cards_strict(
    cards: Iterable[Mapping[str, Any]],
    selected: Iterable[str],
    *,
    qualified_peak_memory_mib: int,
    min_transient_headroom_mib: int,
    co_residency_authorized: bool,
) -> tuple[bool, list[str]]:
    """Capacity-based eligibility under a previously authorized resource contract."""
    if qualified_peak_memory_mib <= 0 or min_transient_headroom_mib < 0:
        raise PrelaunchEngineeringError("capacity GPU gate requires positive peak and non-negative headroom")
    by_uuid = {str(card["uuid"]): card for card in cards}
    failures: list[str] = []
    required_free = qualified_peak_memory_mib + min_transient_headroom_mib
    for uuid in selected:
        card = by_uuid.get(uuid)
        if card is None:
            failures.append(f"{uuid}:not-visible")
            continue
        try:
            used = int(card["memory_used_mib"])
            total = int(card["memory_total_mib"])
        except (KeyError, TypeError, ValueError):
            failures.append(f"{uuid}:memory-capacity-unknown")
            continue
        free = total - used
        if free < required_free:
            failures.append(
                f"{uuid}:free-memory={free}MiB<required={required_free}MiB"
            )
        if card.get("compute_apps_query_error"):
            failures.append(f"{uuid}:compute-app-query-failed")
            continue
        _monitoring, blocking, classification_failed = classify_device_handles(
            card.get("device_handles", [])
        )
        occupied = bool(card.get("compute_apps")) or bool(blocking)
        if classification_failed:
            failures.append(f"{uuid}:device-handle-classification-failed")
        elif occupied and not co_residency_authorized:
            failures.append(f"{uuid}:occupied-co-residency-not-authorized")
    return not failures, failures

def _capacity_gate_settings(profile: Mapping[str, Any]) -> dict[str, Any] | None:
    gate = profile.get("gpu_gate", {})
    if not isinstance(gate, Mapping) or gate.get("mode", "legacy_idle") != "capacity":
        return None
    try:
        peak = int(gate["qualified_peak_memory_mib"])
        headroom = int(gate["min_transient_headroom_mib"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PrelaunchEngineeringError(
            "capacity GPU gate requires qualified_peak_memory_mib and min_transient_headroom_mib"
        ) from exc
    co_residency = gate.get("co_residency_authorized")
    if not isinstance(co_residency, bool):
        raise PrelaunchEngineeringError(
            "capacity GPU gate requires boolean co_residency_authorized from the resource contract"
        )
    if peak <= 0 or headroom < 0:
        raise PrelaunchEngineeringError(
            "capacity GPU gate requires positive peak and non-negative headroom"
        )
    return {
        "qualified_peak_memory_mib": peak,
        "min_transient_headroom_mib": headroom,
        "co_residency_authorized": co_residency,
    }

def capacity_cards(
    cards: Iterable[Mapping[str, Any]],
    selected: Iterable[str],
    *,
    minimum_free_memory_mib: int | None = None,
    max_utilization: int = 100,
    qualified_peak_memory_mib: int | None = None,
    min_transient_headroom_mib: int = 0,
    co_residency_authorized: bool | None = None,
) -> tuple[bool, list[str]]:
    if qualified_peak_memory_mib is not None:
        return _capacity_cards_strict(
            cards,
            selected,
            qualified_peak_memory_mib=qualified_peak_memory_mib,
            min_transient_headroom_mib=min_transient_headroom_mib,
            co_residency_authorized=co_residency_authorized,
        )
    if minimum_free_memory_mib is None:
        raise LaunchError("capacity gate requires a free-memory contract")
    return _capacity_cards_minimum_free(
        cards,
        selected,
        minimum_free_memory_mib=minimum_free_memory_mib,
        max_utilization=max_utilization,
    )


def resource_cards_eligible(
    cards: Iterable[Mapping[str, Any]],
    selected: Iterable[str],
    *,
    resource_mode: str,
    max_idle_memory_mib: int,
    max_utilization: int,
    minimum_free_memory_mib: int,
    capacity_gate: Mapping[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    if capacity_gate is not None:
        return capacity_cards(
            cards,
            selected,
            max_utilization=max_utilization,
            **dict(capacity_gate),
        )
    if resource_mode == "exclusive-live-idle":
        return idle_cards(
            cards, selected,
            max_memory_mib=max_idle_memory_mib,
            max_utilization=max_utilization,
        )
    if resource_mode == "co-resident-capacity":
        return capacity_cards(
            cards, selected,
            minimum_free_memory_mib=minimum_free_memory_mib,
            max_utilization=max_utilization,
        )
    raise LaunchError(f"unsupported GPU resource mode: {resource_mode}")


def _auto_count(spec: str) -> int | None:
    if not spec.startswith("auto:"):
        return None
    try:
        count = int(spec.split(":", 1)[1])
    except ValueError as exc:
        raise LaunchError("GPU selector must be auto:N or a comma-separated UUID list") from exc
    if count < 1:
        raise LaunchError("auto:N requires N >= 1")
    return count


def select_gpu_uuids(
    spec: str,
    runtime_allowed: Iterable[str],
    cards: list[Mapping[str, Any]],
    *,
    max_memory_mib: int = 16,
    max_utilization: int = 0,
    resource_mode: str = "exclusive-live-idle",
    minimum_free_memory_mib: int = 0,
    capacity_gate: Mapping[str, Any] | None = None,
) -> list[str]:
    allowed = {value.strip() for value in runtime_allowed if value.strip()}
    if not allowed:
        raise LaunchError(f"{DEFAULT_AUTH_ENV} must list the current launch runtime candidate GPU UUIDs")
    visible = {str(card["uuid"]) for card in cards}
    count = _auto_count(spec)
    if count is not None:
        eligible: list[str] = []
        for uuid in sorted(allowed & visible):
            ok, _ = resource_cards_eligible(
                cards,
                [uuid],
                resource_mode=resource_mode,
                max_idle_memory_mib=max_memory_mib,
                max_utilization=max_utilization,
                minimum_free_memory_mib=minimum_free_memory_mib,
                capacity_gate=capacity_gate,
            )
            if ok:
                eligible.append(uuid)
        if len(eligible) < count:
            raise LaunchError(
                f"authorized eligible GPU count under {resource_mode} is {len(eligible)}; "
                f"requested {count}"
            )
        return eligible[:count]

    selected = [value.strip() for value in spec.split(",") if value.strip()]
    if not selected or any(not GPU_UUID_RE.fullmatch(value) for value in selected):
        raise LaunchError("GPU selector must use GPU UUIDs, never ordinal indices")
    if len(set(selected)) != len(selected):
        raise LaunchError("GPU UUID selector contains duplicates")
    unauthorized = sorted(set(selected) - allowed)
    missing = sorted(set(selected) - visible)
    if unauthorized:
        raise LaunchError(f"GPU UUIDs are not authorized/runtime-allowed: {unauthorized}")
    if missing:
        raise LaunchError(f"GPU UUIDs are not visible: {missing}")
    return selected


def gpu_selection_kwargs(profile: Mapping[str, Any]) -> dict[str, Any]:
    resource_mode, max_memory, max_util, minimum_free, _, _ = _gate_settings(profile)
    return {
        "max_memory_mib": max_memory,
        "max_utilization": max_util,
        "resource_mode": resource_mode,
        "minimum_free_memory_mib": minimum_free,
        "capacity_gate": _capacity_gate_settings(profile),
    }


@contextlib.contextmanager
def gpu_locks(uuids: Iterable[str], root: Path, *, wait_seconds: float = 0):
    """Keep exact project flock exclusion; optionally wait for a bounded handoff.

    Never remove a lockfile or steal another owner's lock. Wait is pre-formal
    and changes no paid-work or science identity. A timeout remains a refusal.
    """
    if not 0 <= wait_seconds <= 60:
        raise LaunchError("project GPU lock wait must be within 0..60 seconds")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    deadline = time.monotonic() + wait_seconds
    handles = []
    try:
        for uuid in sorted(uuids):
            handle = (root / f"{uuid}.lock").open("a+")
            try:
                while True:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError as exc:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise LaunchError(f"project GPU lock busy: {uuid} (bounded wait exhausted)") from exc
                        time.sleep(min(0.1, remaining))
            except BaseException:
                handle.close()
                raise
            handles.append(handle)
        yield
    finally:
        for handle in reversed(handles):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def replace_tokens(argv: Iterable[str], replacements: Mapping[str, str]) -> list[str]:
    rendered: list[str] = []
    for value in argv:
        result = str(value)
        for token, replacement in replacements.items():
            result = result.replace(token, replacement)
        rendered.append(result)
    return rendered


def docker_bind_mount(source: Path, target: str, *, read_only: bool) -> str:
    """Serialize one Docker ``--mount`` bind using Docker's key/value grammar."""
    mount = f"type=bind,src={source},dst={target}"
    return mount + ",readonly" if read_only else mount


def verify_runtime_assets_visible_to_docker(
    profile: Mapping[str, Any], immutable_image: str
) -> None:
    """Verify daemon-side bind visibility and exact content before GPU selection."""
    assets = profile.get("_runtime_asset_snapshots", [])
    if not assets:
        return
    workspace = Path(str(profile["workspace"])).expanduser().resolve()
    staging_root = workspace / str(profile["runtime_asset_staging_root"])
    hash_script = r"""import hashlib,pathlib,stat,sys
p=pathlib.Path(sys.argv[1]); kind=sys.argv[2]
if kind == 'file':
 m=p.lstat().st_mode
 if not stat.S_ISREG(m): raise SystemExit('expected regular file')
 h=hashlib.sha256(b'openevo-runtime-file-v1\0'); h.update(b'exec\0' if m & 0o111 else b'data\0')
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 print(h.hexdigest())
elif kind == 'dir':
 if not stat.S_ISDIR(p.lstat().st_mode): raise SystemExit('expected directory')
 h=hashlib.sha256(b'openevo-runtime-tree-v1\0')
 for e in sorted(p.rglob('*'),key=lambda x:x.relative_to(p).as_posix()):
  r=e.relative_to(p).as_posix(); m=e.lstat().st_mode
  if stat.S_ISLNK(m): raise SystemExit('symlink in runtime tree')
  if stat.S_ISDIR(m): h.update(f'dir\0{r}\0'.encode()); continue
  if not stat.S_ISREG(m): raise SystemExit('special file in runtime tree')
  q=hashlib.sha256()
  with e.open('rb') as f:
   for b in iter(lambda:f.read(1048576),b''): q.update(b)
  h.update(f'file\0{r}\0{e.stat().st_size}\0{bool(m & 0o111)}\0{q.hexdigest()}\0'.encode())
 print(h.hexdigest())
else: raise SystemExit('unsupported runtime asset type')
"""
    for index, asset in enumerate(assets):
        raw_source = Path(str(asset["source"]))
        try:
            if staging_root.is_symlink():
                raise LaunchError(f"runtime staging root is a symlink: {staging_root}")
            _reject_symlink_components(staging_root, raw_source, "runtime snapshot")
        except LaunchError as exc:
            raise LaunchError(f"runtime asset snapshot path is not link-free before Docker probe: {raw_source}") from exc
        source = raw_source.resolve(strict=True)
        expected_type = str(asset["expected_type"])
        expected_sha = str(asset["sha256"])
        if runtime_asset_digest(source, expected_type) != expected_sha:
            raise LaunchError(f"runtime asset changed before Docker visibility probe: {source}")
        target = f"/__openevo_runtime_asset_probe_{index}"
        command = [
            "docker", "run", "--pull=never", "--rm", "--network", "none",
            "--read-only", "--security-opt", "no-new-privileges", "--cap-drop", "ALL",
            "--pids-limit", "64", "--env", "CUDA_VISIBLE_DEVICES=",
            "--mount", docker_bind_mount(source, target, read_only=True),
            "--entrypoint", "python3", immutable_image, "-c", hash_script, target, expected_type,
        ]
        try:
            result = _run(command, check=False)
        except OSError as exc:
            raise LaunchError(f"Docker daemon runtime asset visibility probe could not run: {exc}") from exc
        observed = result.stdout.strip()
        if (result.returncode != 0 or not re.fullmatch(r"[0-9a-f]{64}", observed)
                or observed != expected_sha):
            detail = result.stderr.strip()[-500:]
            raise LaunchError(
                f"runtime asset is not exactly visible to Docker daemon before GPU selection: "
                f"{source} rc={result.returncode} observed_sha256={observed or 'missing'} {detail}"
            )


def docker_command(
    profile: Mapping[str, Any],
    selected: list[str],
    image: str,
    effective_mode: str,
    *,
    command: list[str] | None = None,
) -> list[str]:
    workspace = Path(str(profile["workspace"])).expanduser().resolve()
    paths = profile["paths"]
    run_dir = f"/workspace/runs/{profile['run_id']}"
    replacements = {
        "${RUN_DIR}": run_dir,
        "${DATA_DIR}": "/workspace/data",
        "${MODELS_DIR}": "/workspace/models",
        "${EXPERIMENT_DIR}": "/workspace/src/openevo-experiment",
        "${OPENEVO_DIR}": "/workspace/src/openevo-core",
        "${GPU_UUID}": selected[0],
    }
    argv = replace_tokens(command or profile["command"], replacements)
    env = {
        "OPENEVO_RUN_ID": str(profile["run_id"]),
        "OPENEVO_ASSIGNED_GPU_UUIDS": ",".join(selected),
        "CUDA_VISIBLE_DEVICES": ",".join(selected),
        "WANDB_MODE": effective_mode,
        "OPENEVO_WANDB_MODE": effective_mode,
        "WANDB_DIR": f"{run_dir}/wandb",
        "WANDB_ARTIFACT_DIR": f"{run_dir}/wandb-artifacts",
        "HF_HOME": "/workspace/cache/huggingface",
        "TRANSFORMERS_CACHE": "/workspace/cache/huggingface/transformers",
        "TMPDIR": "/tmp",
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": "safe.directory",
        "GIT_CONFIG_VALUE_0": "/workspace/src/openevo-experiment",
        "GIT_CONFIG_KEY_1": "safe.directory",
        "GIT_CONFIG_VALUE_1": "/workspace/src/openevo-core",
    }
    gpu_request = f"device={selected[0]}" if len(selected) == 1 else f'"device={",".join(selected)}"'
    command_line = [
        "docker", "run", "--rm", "--read-only", "--security-opt", "no-new-privileges",
        "--cap-drop", "ALL", "--pids-limit", "4096", "--tmpfs", "/tmp:rw,noexec,nosuid,size=8g",
        "--name", f"experiment-host-openevo-webshop-{profile['run_id']}",
        "--gpus", gpu_request, "--user", f"{profile['identity']['uid']}:{profile['identity']['gid']}",
        "--network", str(profile.get("container", {}).get("network", "bridge")),
    ]
    mounts = [
        (workspace_path(profile, str(profile["source"]["experiment"]["path"])), "/workspace/src/openevo-experiment", "ro"),
        (workspace_path(profile, str(profile["source"]["openevo"]["path"])), "/workspace/src/openevo-core", "ro"),
        (workspace_path(profile, str(paths["models"])), "/workspace/models", "ro"),
        (workspace_path(profile, str(paths["data"])), "/workspace/data", "ro"),
        (workspace_path(profile, str(paths["runs"])), "/workspace/runs", "rw"),
        (workspace_path(profile, str(paths["cache"])), "/workspace/cache", "rw"),
    ]
    for source, target, mode in mounts:
        command_line.extend(["--mount", docker_bind_mount(source, target, read_only=mode == "ro")])
    runtime_snapshots = profile.get("_runtime_asset_snapshots", [])
    declared_assets = profile.get("runtime_assets", [])
    declared_overlays = profile.get("runtime_asset_overlays", [])
    if (declared_assets or declared_overlays) and len(runtime_snapshots) != len(declared_assets) + len(declared_overlays):
        raise LaunchError("runtime assets were not staged and validated before GPU allocation")
    if runtime_snapshots and not (declared_assets or declared_overlays):
        raise LaunchError("launcher-private runtime snapshots require declared profile runtime_assets")
    seen_runtime_targets: list[str] = []
    seen_overlay_targets: list[str] = []
    declared_directory_targets = {
        str(item.get("target")) for item in declared_assets if item.get("expected_type") == "dir"
    }
    base_snapshots = [asset for asset in runtime_snapshots if "base_target" not in asset]
    overlay_snapshots = [asset for asset in runtime_snapshots if "base_target" in asset]
    staging_relative = profile.get("runtime_asset_staging_root")
    if runtime_snapshots and not isinstance(staging_relative, str):
        raise LaunchError("runtime snapshot staging root is missing")
    staging_root = (workspace / str(staging_relative)).resolve() if runtime_snapshots else None

    def append_runtime_mount(asset: Mapping[str, Any], *, overlay: bool) -> None:
        source = Path(asset["source"])
        try:
            source.resolve(strict=True).relative_to(staging_root)
        except (OSError, ValueError, TypeError) as exc:
            raise LaunchError(f"runtime asset snapshot source escapes its staging root: {source}") from exc
        if source.is_symlink():
            raise LaunchError(f"runtime asset snapshot must not be a symlink: {source}")
        target = str(asset["target"])
        if overlay:
            base_target = str(asset.get("base_target", ""))
            parsed = PurePosixPath(target)
            if (base_target not in declared_directory_targets or not target.startswith(base_target + "/")
                    or not parsed.is_absolute() or parsed.as_posix() != target
                    or any(not re.fullmatch(r"[A-Za-z0-9._-]+", part) for part in parsed.parts[1:])):
                raise LaunchError(f"runtime asset overlay target is not below a declared directory mount: {target}")
            if any(target == seen or target.startswith(seen + "/") or seen.startswith(target + "/")
                   for seen in seen_overlay_targets):
                raise LaunchError(f"runtime asset overlay overlaps another runtime mount: {target}")
            seen_overlay_targets.append(target)
        else:
            _validate_runtime_asset_target(target, seen_runtime_targets)
            seen_runtime_targets.append(target)
        if asset.get("expected_type") not in {"file", "dir"} or not re.fullmatch(
            r"[0-9a-f]{64}", str(asset.get("sha256", ""))
        ):
            raise LaunchError("runtime asset snapshot has an invalid type or content identity")
        if overlay and asset.get("expected_type") != "file":
            raise LaunchError("runtime asset overlays must be single files")
        if runtime_asset_digest(source, asset["expected_type"]) != asset["sha256"]:
            raise LaunchError(f"staged runtime asset changed before Docker launch: {source}")
        command_line.extend(["--mount", docker_bind_mount(source, target, read_only=True)])

    execution = profile.get("execution") if isinstance(profile.get("execution"), Mapping) else {}
    if execution.get("mode") == "host-orchestrator" and (declared_assets or declared_overlays):
        host_source = profile.get("_host_runtime_source")
        target = execution.get("runtime_source_target")
        if not isinstance(host_source, Mapping) or not isinstance(target, str) or not target:
            raise LaunchError("host-orchestrator runtime assets have no validated effective source tree")
        validate_host_runtime_source(profile)
        if any(asset.get("target") != target for asset in base_snapshots):
            raise LaunchError("host-orchestrator runtime source target changed after staging")
        command_line.extend([
            "--mount",
            docker_bind_mount(str(host_source["source_root"]), target, read_only=True),
        ])
    else:
        for asset in base_snapshots:
            append_runtime_mount(asset, overlay=False)
        for asset in overlay_snapshots:
            append_runtime_mount(asset, overlay=True)
    for key, value in env.items():
        command_line.extend(["--env", f"{key}={value}"])
    if os.environ.get("WANDB_API_KEY") and effective_mode == "online":
        command_line.extend(["--env", "WANDB_API_KEY"])
    entrypoint = argv[0]
    if not entrypoint:
        raise LaunchError("profile command executable must not be empty")
    command_line.extend(["--entrypoint", entrypoint, image])
    command_line.extend(argv[1:])
    return command_line



def host_orchestrator_command(
    profile: Mapping[str, Any],
    selected: list[str],
    run_dir: Path,
    effective_mode: str,
    formal_start_path: Path,
) -> list[str]:
    if os.geteuid() != 0:
        raise LaunchError("host orchestrator requires the root Docker control plane")
    return host_orchestrator_identity_command(profile, selected, run_dir, effective_mode, formal_start_path)


def host_orchestrator_identity_command(
    profile: Mapping[str, Any],
    selected: list[str],
    run_dir: Path,
    effective_mode: str,
    formal_start_path: Path | None,
    *,
    preflight_context: Mapping[str, Any] | None = None,
) -> list[str]:
    execution = profile.get("execution") if isinstance(profile.get("execution"), Mapping) else {}
    if execution.get("mode") != "host-orchestrator":
        raise LaunchError("host orchestrator profile must declare execution.mode=host-orchestrator")
    if preflight_context is None:
        if not isinstance(profile.get("_formal_start_context"), Mapping) or formal_start_path is None:
            raise LaunchError("host orchestrator requires canonical run-campaign readiness")
    elif profile.get("_formal_start_context") is not None or formal_start_path is not None:
        raise LaunchError("zero-formal host preflight must not carry Formal Start authority")
    workspace = Path(str(profile["workspace"])).expanduser().resolve()
    paths = profile["paths"]
    runs_root = workspace_path(profile, str(paths["runs"])).resolve()
    try:
        run_relative = run_dir.resolve(strict=False).relative_to(runs_root)
    except ValueError as exc:
        raise LaunchError("host orchestrator run directory must be under the mounted runs root") from exc
    if not run_relative.parts:
        raise LaunchError("host orchestrator run directory must name a child of the mounted runs root")
    host_source = profile.get("_host_runtime_source")
    replacements = {
        "${RUN_DIR}": f"/workspace/runs/{run_relative.as_posix()}",
        "${DATA_DIR}": str(workspace_path(profile, str(paths["data"]))),
        "${MODELS_DIR}": str(workspace_path(profile, str(paths["models"]))),
        "${EXPERIMENT_DIR}": str(workspace_path(profile, str(profile["source"]["experiment"]["path"]))),
        "${OPENEVO_DIR}": str(workspace_path(profile, str(profile["source"]["openevo"]["path"]))),
        "${GPU_UUID}": selected[0],
        "${GPU_UUIDS}": ",".join(selected),
    }
    if host_source is not None:
        validate_host_runtime_source(profile)
        source_root = str(host_source["source_root"])
        _host_runtime_entrypoint_identity(profile, Path(source_root).resolve(strict=True))
        replacements["${SOURCE_ROOT}"] = source_root
    elif profile.get("runtime_assets") or profile.get("runtime_asset_overlays"):
        raise LaunchError("host-orchestrator cannot ignore declared runtime assets")
    command = list(profile["command"])
    run_dir_indexes = [index for index, value in enumerate(command) if value == "--run-dir"]
    if len(run_dir_indexes) > 1 or (run_dir_indexes and run_dir_indexes[0] + 1 >= len(command)):
        raise LaunchError("host-orchestrator command must contain at most one complete --run-dir argument")
    if run_dir_indexes:
        index = run_dir_indexes[0] + 1
        configured_run_dir = command[index]
        if configured_run_dir != "${RUN_DIR}":
            if not Path(configured_run_dir).is_absolute():
                raise LaunchError("host orchestrator --run-dir must be ${RUN_DIR} or an absolute host path")
            try:
                configured_relative = Path(configured_run_dir).resolve(strict=False).relative_to(runs_root)
            except ValueError as exc:
                raise LaunchError("host orchestrator --run-dir must be under the mounted runs root") from exc
            container_run_dir = "/workspace/runs"
            if configured_relative.parts:
                container_run_dir += f"/{configured_relative.as_posix()}"
            replacements[configured_run_dir] = container_run_dir
    frozen_inputs = profile.get("frozen_inputs")
    if isinstance(frozen_inputs, Mapping) and "qualification_path" in frozen_inputs:
        qualification_path = frozen_inputs.get("qualification_path")
        if not isinstance(qualification_path, str) or not qualification_path:
            raise LaunchError("frozen_inputs.qualification_path must be a non-empty string")
        qualification_indexes = [index for index, value in enumerate(command) if value == "--qualification"]
        if len(qualification_indexes) != 1 or qualification_indexes[0] + 1 >= len(command):
            raise LaunchError("host-orchestrator command must contain exactly one --qualification argument")
        # The frozen path is the single source of truth. The resulting argv is
        # also what FormalStart hashes through _command_identity().
        command[qualification_indexes[0] + 1] = qualification_path
    argv = replace_tokens(command, replacements)
    if not argv:
        raise LaunchError("host orchestrator command is empty")
    command_env = [
        "/usr/bin/env",
        "-u",
        "PYTHONHOME",
        f"OPENEVO_AUTHORIZED_GPU_UUIDS={','.join(selected)}",
        f"OPENEVO_ASSIGNED_GPU_UUIDS={','.join(selected)}",
        f"WANDB_MODE={effective_mode}",
        f"OPENEVO_WANDB_MODE={effective_mode}",
        *([f"PYTHONPATH={host_source['source_root']}",
           "PYTHONNOUSERSITE=1",
           "PYTHONDONTWRITEBYTECODE=1",
           f"OPENEVO_EFFECTIVE_SOURCE_ROOT={host_source['source_root']}",
           f"OPENEVO_EFFECTIVE_SOURCE_SHA256={host_source['sha256']}"] if host_source is not None else []),
    ]
    if preflight_context is None:
        command_env.append(f"OPENEVO_FORMAL_START_RECEIPT={formal_start_path}")
    else:
        command_env.extend([
            f"CUDA_VISIBLE_DEVICES={','.join(selected)}",
            f"OPENEVO_STEP119_PREFLIGHT_CONTEXT_SHA256={preflight_context['context_sha256']}",
            f"OPENEVO_STEP119_CAMPAIGN_ID={preflight_context['campaign_id']}",
            f"OPENEVO_STEP119_EXECUTION_SHA={preflight_context['execution_sha']}",
            f"OPENEVO_STEP119_PROFILE_SHA256={preflight_context['runtime_profile_sha256']}",
            f"OPENEVO_STEP119_IMAGE_ID={preflight_context['image_id']}",
        ])
    return [*command_env, *argv]


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _tracking_probe_identity(profile: Mapping[str, Any], immutable_image: str) -> dict[str, Any]:
    tracking = profile["tracking"]
    epoch = (
        tracking.get("probe_epoch")
        or os.environ.get("OPENEVO_EXECUTION_EPOCH")
        or os.environ.get("OPENEVO_CAMPAIGN_ID")
        or tracking.get("group")
        or profile["run_id"]
    )
    return {
        "epoch": str(epoch),
        "image_id": immutable_image,
        "network": str(profile.get("container", {}).get("network", "bridge")),
        "base_url": os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai"),
        "entity": tracking.get("entity") or os.environ.get("WANDB_ENTITY"),
        "project": tracking.get("project") or os.environ.get("WANDB_PROJECT"),
        "auth_present": bool(os.environ.get("WANDB_API_KEY")),
    }


def _tracking_probe_path(profile: Mapping[str, Any], identity: Mapping[str, Any]) -> Path:
    cache = workspace_path(profile, str(profile["paths"]["cache"]))
    digest = hashlib.sha256(
        json.dumps(dict(identity), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return cache / "openevo-platform" / "wandb-probes" / f"{digest}.json"


def _tracking_result_from_receipt(
    profile: Mapping[str, Any],
    receipt: Mapping[str, Any],
    path: Path,
    *,
    reused: bool,
) -> tuple[str, str | None, str, bool]:
    status = receipt.get("status")
    if status == "pass":
        return "online", None, str(path), reused
    reason = str(receipt.get("reason") or "online W&B preflight failed")
    if profile["tracking"].get("on_error", "fallback_offline") == "fallback_offline":
        return "offline", reason, str(path), reused
    raise LaunchError(reason)


def choose_effective_tracking(
    profile: Mapping[str, Any],
    immutable_image: str,
    *,
    refresh: bool = False,
) -> tuple[str, str | None, str | None, bool]:
    """Choose tracking mode with one online probe per execution epoch."""
    requested = str(profile["tracking"]["mode"])
    if requested != "online":
        return requested, None, None, False

    identity = _tracking_probe_identity(profile, immutable_image)
    receipt_path = _tracking_probe_path(profile, identity)
    receipt_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = receipt_path.with_suffix(".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if receipt_path.is_file() and not refresh:
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise LaunchError(f"W&B probe receipt is unreadable: {receipt_path}: {exc}") from exc
            if receipt.get("identity") == identity:
                return _tracking_result_from_receipt(profile, receipt, receipt_path, reused=True)

        if not identity["auth_present"]:
            receipt = {
                "schema": "openevo.platform-wandb-probe.v1",
                "status": "fail",
                "identity": identity,
                "reason": "WANDB_API_KEY is absent",
                "return_code": None,
                "observed_at_epoch": time.time(),
            }
        else:
            probe = _run([
                "docker", "run", "--rm", "--network", str(identity["network"]), "--read-only", "--tmpfs", "/tmp:rw,size=64m",
                "--env", "WANDB_API_KEY", "--env", "WANDB_MODE=online", "--entrypoint", "python3", immutable_image, "-c",
                "import wandb; api=wandb.Api(timeout=10); api.viewer; print('wandb-preflight-pass')",
            ], check=False)
            passed = probe.returncode == 0
            receipt = {
                "schema": "openevo.platform-wandb-probe.v1",
                "status": "pass" if passed else "fail",
                "identity": identity,
                "reason": None if passed else f"online W&B preflight failed with rc={probe.returncode}",
                "return_code": probe.returncode,
                "observed_at_epoch": time.time(),
            }
        atomic_json(receipt_path, receipt)
        return _tracking_result_from_receipt(profile, receipt, receipt_path, reused=False)


def ensure_run_directory(profile: Mapping[str, Any]) -> Path:
    runs = workspace_path(profile, str(profile["paths"]["runs"]))
    run_dir = runs / str(profile["run_id"])
    if run_dir.exists() and not profile.get("resume", False):
        raise LaunchError(f"run directory already exists; choose a new run_id or set resume: {run_dir}")
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    uid = int(profile["identity"]["uid"])
    gid = int(profile["identity"]["gid"])
    if os.geteuid() == 0:
        os.chown(run_dir, uid, gid)
    stat = run_dir.stat()
    if stat.st_uid != uid or stat.st_gid != gid:
        raise LaunchError(f"run directory owner {stat.st_uid}:{stat.st_gid} != {uid}:{gid}")
    return run_dir



def _gate_settings(profile: Mapping[str, Any]) -> tuple[str, int, int, int, int, int]:
    gate = profile.get("gpu_gate", {})
    if not isinstance(gate, Mapping):
        gate = {}
    capacity_gate = _capacity_gate_settings(profile)
    if capacity_gate is not None:
        resource_mode = "co-resident-capacity"
        minimum_free_memory_mib = (
            int(capacity_gate["qualified_peak_memory_mib"])
            + int(capacity_gate["min_transient_headroom_mib"])
        )
    else:
        resource_mode = str(gate.get("resource_mode", "exclusive-live-idle"))
        minimum_free_memory_mib = int(gate.get("minimum_free_memory_mib", 0))
    if resource_mode == "co-resident-capacity" and minimum_free_memory_mib <= 0:
        raise LaunchError("co-resident-capacity requires a positive minimum_free_memory_mib")
    if resource_mode not in {"exclusive-live-idle", "co-resident-capacity"}:
        raise LaunchError(f"unsupported GPU resource mode: {resource_mode}")
    return (
        resource_mode,
        int(gate.get("max_idle_memory_mib", 16)),
        int(gate.get("max_idle_utilization_percent", 0))
        if resource_mode == "exclusive-live-idle"
        else int(gate.get("max_co_resident_utilization_percent", 100)),
        minimum_free_memory_mib,
        int(gate.get("interval_seconds", 60)),
        int(gate.get("timeout_seconds", 0)),
    )


def wait_for_idle(profile: Mapping[str, Any], selected: list[str], run_dir: Path) -> list[dict[str, Any]]:
    resource_mode, max_memory, max_util, minimum_free, interval, timeout = _gate_settings(profile)
    capacity_gate = _capacity_gate_settings(profile)
    started = time.monotonic()
    attempt = 0
    while True:
        attempt += 1
        first = snapshot_gpus()
        first_ok, first_failures = resource_cards_eligible(
            first, selected, resource_mode=resource_mode,
            max_idle_memory_mib=max_memory, max_utilization=max_util,
            minimum_free_memory_mib=minimum_free,
            capacity_gate=capacity_gate,
        )
        second = first
        second_ok = False
        second_failures = first_failures
        if first_ok:
            time.sleep(min(2, max(1, interval)))
            second = snapshot_gpus()
            second_ok, second_failures = resource_cards_eligible(
                second, selected, resource_mode=resource_mode,
                max_idle_memory_mib=max_memory, max_utilization=max_util,
                minimum_free_memory_mib=minimum_free,
                capacity_gate=capacity_gate,
            )
        receipt = {
            "schema": "openevo.webshop.gpu-gate.v2",
            "resource_mode": resource_mode,
            "minimum_free_memory_mib": minimum_free,
            "gate_mode": "capacity" if capacity_gate is not None else resource_mode,
            "capacity_gate": dict(capacity_gate) if capacity_gate is not None else None,
            "attempt": attempt,
            "selected_gpu_uuids": selected,
            "first": first,
            "second": second,
            "eligible": first_ok and second_ok,
            "first_failures": first_failures,
            "second_failures": second_failures,
            "observed_at_epoch": time.time(),
        }
        atomic_json(run_dir / f"gpu-gate-{attempt:04d}.json", receipt)
        if first_ok and second_ok:
            return [dict(card) for card in second if card["uuid"] in selected]
        if timeout and time.monotonic() - started >= timeout:
            raise LaunchError(f"GPU idle gate timed out after {attempt} attempts")
        if not profile.get("wait", False):
            raise LaunchError(
                f"selected GPUs are not eligible under {resource_mode}: {second_failures}"
            )
        time.sleep(interval)


def _record_auto_selection(
    run_dir: Path,
    attempt: int,
    cards: list[Mapping[str, Any]],
    runtime_allowed: Iterable[str],
    selected: list[str] | None,
    error: str | None,
) -> None:
    atomic_json(
        run_dir / f"gpu-auto-selection-{attempt:04d}.json",
        {
            "schema": "openevo.webshop.gpu-auto-selection.v1",
            "attempt": attempt,
            "runtime_candidate_uuids": sorted({value.strip() for value in runtime_allowed if value.strip()}),
            "selected_gpu_uuids": selected or [],
            "error": error,
            "snapshot": cards,
            "observed_at_epoch": time.time(),
        },
    )


def _command_identity(command: list[str], image_id_value: str) -> str:
    payload = {"image_id": image_id_value, "argv": command}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _prelaunch_runtime_python(profile: Mapping[str, Any]) -> str:
    config = profile.get("prelaunch") if isinstance(profile.get("prelaunch"), Mapping) else {}
    explicit = config.get("runtime_python") if isinstance(config, Mapping) else None
    if explicit is not None:
        if isinstance(explicit, str) and re.fullmatch(
            r"python(?:\d+(?:\.\d+)*)?(?:\.exe)?", Path(explicit).name, re.IGNORECASE
        ):
            return explicit
        raise PrelaunchEngineeringError("prelaunch.runtime_python must name a Python interpreter executable")
    executable = str(profile["command"][0])
    if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?(?:\.exe)?", Path(executable).name, re.IGNORECASE):
        return executable
    raise PrelaunchEngineeringError(
        "profile command is not a Python executable; declare prelaunch.runtime_python "
        "so the selected-GPU kernel smoke uses the exact scientific interpreter"
    )


def _control_evidence_dir(profile: Mapping[str, Any], run_dir: Path) -> Path:
    context = profile.get("_formal_start_context")
    if isinstance(context, Mapping) and isinstance(context.get("control_root"), str):
        root = Path(str(context["control_root"])).expanduser().resolve()
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        return root
    return run_dir


def _runtime_asset_receipt_identity(profile: Mapping[str, Any]) -> list[dict[str, str]]:
    identity = []
    for asset in profile.get("_runtime_asset_snapshots", []):
        item = {
            "target": str(asset["target"]),
            "expected_type": str(asset["expected_type"]),
            "sha256": str(asset["sha256"]),
        }
        if "base_target" in asset:
            item["base_target"] = str(asset["base_target"])
        identity.append(item)
    host_source = profile.get("_host_runtime_source")
    if isinstance(host_source, Mapping):
        identity.append({
            "target": str(host_source["target"]),
            "expected_type": "dir",
            "sha256": str(host_source["sha256"]),
            "runtime_role": "host-effective-source",
            "receipt_sha256": str(host_source["receipt_sha256"]),
        })
    host_import = profile.get("_host_runtime_import_identity")
    if isinstance(host_import, Mapping):
        identity.append({
            "target": str(host_import.get("receipt_path", "")),
            "expected_type": "file",
            "sha256": str(host_import.get("identity_sha256", "")),
            "runtime_role": "host-runtime-import-identity",
            "receipt_sha256": str(host_import.get("receipt_sha256", "")),
        })
    return identity


def write_host_runtime_import_receipt(profile: dict[str, Any], run_dir: Path) -> dict[str, Any] | None:
    identity = profile.get("_host_runtime_import_identity")
    if not isinstance(identity, Mapping):
        return None
    receipt_path = run_dir / "host-runtime-import-identity.json"
    receipt = {**dict(identity), "run_id": str(profile["run_id"]), "observed_at_epoch": time.time()}
    atomic_json(receipt_path, receipt)
    profile["_host_runtime_import_identity"] = {
        **dict(identity),
        "receipt_path": str(receipt_path),
        "receipt_sha256": prelaunch_zero_formal.sha256_file(receipt_path),
    }
    return profile["_host_runtime_import_identity"]


def host_selected_gpu_preflight_command(
    profile: Mapping[str, Any], selected: list[str], payload: list[str]
) -> list[str]:
    """Use the declared host interpreter after dropping the control-plane identity."""
    if os.geteuid() != 0:
        raise PrelaunchEngineeringError("host selected-GPU preflight requires the root control plane")
    identity = profile.get("identity") or {}
    uid, gid = identity.get("uid"), identity.get("gid")
    if type(uid) is not int or type(gid) is not int or uid <= 0 or gid <= 0:
        raise PrelaunchEngineeringError("host selected-GPU preflight requires positive run UID/GID")
    expected_python = str((profile.get("execution") or {}).get("runtime_python") or profile["command"][0])
    if payload[0] != expected_python:
        raise PrelaunchEngineeringError("host selected-GPU preflight must use the declared scientific interpreter")
    validate_host_runtime_source(profile)
    source = profile.get("_host_runtime_source")
    env = ["/usr/bin/env", "-u", "PYTHONHOME", "-u", "OPENEVO_FORMAL_START_RECEIPT", "-u", "PYTHONPATH",
           "PYTHONNOUSERSITE=1", "PYTHONDONTWRITEBYTECODE=1",
           f"CUDA_VISIBLE_DEVICES={','.join(selected)}",
           f"OPENEVO_ASSIGNED_GPU_UUIDS={','.join(selected)}"]
    if isinstance(source, Mapping):
        env.extend([f"PYTHONPATH={source['source_root']}",
                    f"OPENEVO_EFFECTIVE_SOURCE_ROOT={source['source_root']}",
                    f"OPENEVO_EFFECTIVE_SOURCE_SHA256={source['sha256']}"])
    wrapper = (
        "import json,os,sys; "
        "uid,gid=int(sys.argv[1]),int(sys.argv[2]); argv=json.loads(sys.argv[3]); "
        "os.setgroups([]); os.setgid(gid); os.setuid(uid); "
        "assert os.geteuid()==uid and os.getegid()==gid and not os.getgroups(); "
        "os.chdir('/'); os.execvpe(argv[0],argv,os.environ)"
    )
    # The root bootstrap ignores user/site customization; only the UID1001 child
    # imports the qualified scientific environment.
    return [*env, "/usr/bin/python3", "-I", "-S", "-c", wrapper,
            str(uid), str(gid), json.dumps(payload)]


def run_selected_gpu_preflight(
    profile: Mapping[str, Any],
    selected: list[str],
    run_dir: Path,
    immutable_image: str,
    control_dir: Path,
) -> dict[str, Any]:
    """Exercise the exact selected-GPU Docker/runtime path with zero formal work."""
    attempt = prelaunch_zero_formal.next_attempt(control_dir)
    receipt_path = control_dir / f"prelaunch-zero-formal-attempt-{attempt:04d}.json"
    execution = profile.get("execution") if isinstance(profile.get("execution"), Mapping) else {}
    host_mode = execution.get("mode") == "host-orchestrator"
    marker = (str(run_dir / f".prelaunch-write-probe-{attempt:04d}") if host_mode
              else f"/workspace/runs/{profile['run_id']}/.prelaunch-write-probe-{attempt:04d}")
    try:
        payload = prelaunch_zero_formal.payload_command(
            _prelaunch_runtime_python(profile), marker, len(selected)
        )
        command = (host_selected_gpu_preflight_command(profile, selected, payload) if host_mode
                   else docker_command(profile, selected, immutable_image, "disabled", command=payload))
    except LaunchError as exc:
        raise PrelaunchEngineeringError(f"selected-GPU preflight command setup failed: {exc}") from exc
    started = time.monotonic()
    completed = _run(command, check=False)
    wall = max(0.0, time.monotonic() - started)

    parsed: dict[str, Any] | None = None
    parse_error: str | None = None
    try:
        parsed = prelaunch_zero_formal.parse_payload_output(completed.stdout or "")
    except prelaunch_zero_formal.PrelaunchError as exc:
        parse_error = str(exc)

    receipt = prelaunch_zero_formal.build_prelaunch_receipt(
        run_id=str(profile["run_id"]),
        attempt=attempt,
        selected_gpu_uuids=selected,
        image_id=immutable_image,
        command_identity=_command_identity(command, immutable_image),
        return_code=int(completed.returncode),
        wall_seconds=wall,
        payload=parsed,
        payload_error=parse_error,
        stderr_tail=completed.stderr or "",
        runtime_assets=_runtime_asset_receipt_identity(profile),
    )
    atomic_json(receipt_path, receipt)
    try:
        prelaunch_zero_formal.validate_prelaunch_receipt(
            receipt, expected_gpu_uuids=selected
        )
    except prelaunch_zero_formal.PrelaunchError as exc:
        raise PrelaunchEngineeringError(str(exc)) from exc
    return {
        "path": str(receipt_path),
        "sha256": prelaunch_zero_formal.sha256_file(receipt_path),
        "receipt_sha256": receipt["receipt_sha256"],
        "receipt": receipt,
    }


def run_step119_model_load_preflight(
    profile: Mapping[str, Any],
    selected: list[str],
    run_dir: Path,
    immutable_image: str,
    control_dir: Path,
    candidate_identity: Mapping[str, Any],
    profile_sha256: str,
) -> dict[str, Any]:
    """Run the hash-bound Step119 native model-load path with no tasks or environments."""
    execution = profile.get("execution") if isinstance(profile.get("execution"), Mapping) else {}
    entrypoint = execution.get("runtime_entrypoint")
    if (execution.get("mode") != "host-orchestrator" or not isinstance(entrypoint, Mapping)
            or entrypoint.get("kind") != "module"
            or entrypoint.get("module") != "q03_step119_validation_runtime"):
        raise LaunchError("model-load preflight is restricted to the pinned Q03 Step119 host runtime")
    if not isinstance(profile.get("_host_runtime_source"), Mapping):
        raise LaunchError("model-load preflight requires the validated effective host runtime source")
    command = list(profile.get("command") or [])
    run_dir_indexes = [index for index, value in enumerate(command) if value == "--run-dir"]
    if len(run_dir_indexes) != 1 or run_dir_indexes[0] + 1 >= len(command):
        raise LaunchError("Q03 Step119 host command must contain one --run-dir argument")
    if "-m" not in command or command[command.index("-m") + 1] != "q03_step119_validation_runtime":
        raise LaunchError("Q03 Step119 command does not invoke its declared runtime module")
    if "--model-load-preflight-only" in command:
        raise LaunchError("formal task command must not already contain the preflight-only mode")
    command[run_dir_indexes[0] + 1] = str(run_dir)
    host_source = profile["_host_runtime_source"]
    context_projection = {
        "schema": "openevo.q03-step119-launcher-preflight-context.v1",
        "status": "LAUNCHER_AUTHORIZED_ZERO_TASK_PREFLIGHT",
        "campaign_id": candidate_identity["campaign_id"],
        "execution_sha": candidate_identity["execution_sha"],
        "candidate_hashes": {
            "current_campaign": candidate_identity["current_campaign_sha256"],
            "design": candidate_identity["design_sha256"],
            "activation": candidate_identity["activation_sha256"],
            "runtime_profile": profile_sha256,
            "registry_pointer": candidate_identity["registry_pointer_sha256"],
        },
        "runtime_profile_sha256": profile_sha256,
        "selected_gpu_uuids": list(selected),
        "effective_source_sha256": str(host_source["sha256"]),
        "execution_surface": "host-orchestrator",
        "image_id": immutable_image,
        "run_id": run_dir.name,
        "formal_task_count": 0,
        "training_rollouts": 0,
        "generation_calls": 0,
        "optimizer_updates": 0,
        "final_panel_access": 0,
        "scientific_state_mutated": False,
    }
    context_sha256 = prelaunch_zero_formal.canonical_sha256(context_projection)
    context = {**context_projection, "context_sha256": context_sha256}
    context_json = json.dumps(context, sort_keys=True, separators=(",", ":")).encode("utf-8")
    context_b64 = base64.urlsafe_b64encode(context_json).decode("ascii").rstrip("=")
    command.extend(["--model-load-preflight-context", context_b64])
    command.append("--model-load-preflight-only")
    source_root = str(host_source["source_root"])
    validate_host_runtime_source(profile)
    smoke_profile = {
        **profile,
        "run_id": run_dir.name,
        "command": command,
        "_formal_start_context": None,
    }
    host_argv = host_orchestrator_identity_command(
        smoke_profile, selected, run_dir, "disabled", None, preflight_context=context,
    )
    started = time.monotonic()
    result = _run(host_argv, check=False)
    wall_seconds = max(0.0, time.monotonic() - started)
    payload = None
    payload_error = None
    lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    try:
        payload = json.loads(lines[-1]) if lines else None
        if not isinstance(payload, dict):
            raise ValueError("preflight payload must be a JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        payload_error = str(exc)
    passed = (
        result.returncode == 0 and payload is not None
        and payload.get("status") == "MODEL_LOAD_PREFLIGHT_PASS"
        and payload.get("native_init_workers_returned") is True
        and payload.get("actor_rollout_model_initialized") is True
        and payload.get("vllm_runtime_initialized") is True
        and payload.get("formal_task_count") == 0
        and payload.get("environment_creation_calls") == 0
        and payload.get("dataset_item_reads") == 0
        and payload.get("generation_calls") == 0
        and payload.get("optimizer_updates") == 0
        and isinstance(payload.get("preflight_authorization"), dict)
        and payload["preflight_authorization"].get("context_sha256") == context_sha256
    )
    receipt_projection = {
        "schema": "openevo.q03-step119-zero-task-model-load-preflight.v1",
        "status": "PASS" if passed else "FAIL",
        "campaign_id": candidate_identity["campaign_id"],
        "execution_sha": candidate_identity["execution_sha"],
        "current_campaign_sha256": candidate_identity["current_campaign_sha256"],
        "design_sha256": candidate_identity["design_sha256"],
        "activation_sha256": candidate_identity["activation_sha256"],
        "candidate_hashes": {
            "current_campaign": candidate_identity["current_campaign_sha256"],
            "design": candidate_identity["design_sha256"],
            "activation": candidate_identity["activation_sha256"],
            "runtime_profile": profile_sha256,
        },
        "preflight_authorization_context_sha256": context_sha256,
        "execution_surface": "host-orchestrator",
        "selected_gpu_uuids": list(selected),
        "host_router_path": candidate_identity["host_router_path"],
        "host_control_head": candidate_identity["host_control_head"],
        "registry_pointer": {
            "path": candidate_identity["registry_pointer_path"],
            "sha256": candidate_identity["registry_pointer_sha256"],
        },
        "image_id": immutable_image,
        "command_identity": _command_identity(host_argv, immutable_image),
        "runtime_assets": _runtime_asset_receipt_identity(smoke_profile),
        "return_code": int(result.returncode),
        "wall_seconds": wall_seconds,
        "payload": payload,
        "payload_error": payload_error,
        "stderr_tail": (result.stderr or "")[-4000:],
        "formal_task_count": 0,
        "environment_creation_calls": 0,
        "dataset_item_reads": 0,
        "generation_calls": 0,
        "optimizer_updates": 0,
        "scientific_state_mutated": False,
        "observed_at_epoch": time.time(),
    }
    receipt = {
        **receipt_projection,
        "receipt_sha256": prelaunch_zero_formal.canonical_sha256(receipt_projection),
    }
    receipt_path = _next_attempt_path(control_dir, "q03-step119-model-load-preflight")
    atomic_json(receipt_path, receipt)
    if not passed:
        raise PrelaunchEngineeringError(
            f"Q03 Step119 model-load preflight failed; receipt: {receipt_path}; "
            f"rc={result.returncode}; {payload_error or (result.stderr or '')[-1000:]}"
        )
    return {"path": str(receipt_path), "sha256": prelaunch_zero_formal.sha256_file(receipt_path), "receipt": receipt}


def _next_attempt_path(control_dir: Path, stem: str) -> Path:
    index = 1
    while (control_dir / f"{stem}-attempt-{index:04d}.json").exists():
        index += 1
    return control_dir / f"{stem}-attempt-{index:04d}.json"


def _campaign_id(profile: Mapping[str, Any]) -> str:
    tracking = profile.get("tracking", {}) if isinstance(profile.get("tracking"), Mapping) else {}
    return str(profile.get("campaign_id") or tracking.get("group") or profile["run_id"])


def _incident_root(profile: Mapping[str, Any]) -> Path:
    return workspace_path(profile, str(profile["paths"]["cache"])) / "openevo-platform" / "engineering-incidents"


def _record_engineering_incident(
    profile: Mapping[str, Any],
    *,
    layer: str,
    code: str,
    validator_id: str,
    repair_class: str,
    detail: str,
    selected_gpu_uuids: Iterable[str] = (),
) -> None:
    try:
        reliability.record_incident(
            _incident_root(profile),
            category="PRELAUNCH_ENGINEERING",
            layer=layer,
            code=code,
            validator_id=validator_id,
            repair_class=repair_class,
            detail=detail,
            run_id=str(profile["run_id"]),
            selected_gpu_uuids=list(selected_gpu_uuids),
        )
    except Exception:
        pass


def _record_lifecycle(
    profile: Mapping[str, Any],
    control_dir: Path,
    *,
    key: str,
    epoch: float,
    meaning: str,
    source: Mapping[str, Any],
) -> None:
    if not isinstance(profile.get("_formal_start_context"), Mapping):
        return
    reliability.record_lifecycle_event(
        control_dir,
        run_id=str(profile["run_id"]),
        campaign_id=_campaign_id(profile),
        key=key,
        epoch=epoch,
        meaning=meaning,
        source=source,
    )


def run_rank_consistency_preflight(
    profile: Mapping[str, Any],
    control_dir: Path,
) -> dict[str, Any] | None:
    """Validate declared config/controller/state/checkpoint rank closure before GPU use."""
    try:
        receipt = rank_consistency.evaluate_profile(
            profile,
            resolve_path=lambda raw: workspace_path(profile, raw),
        )
    except rank_consistency.RankConsistencyError as exc:
        raise PrelaunchEngineeringError(str(exc)) from exc
    if receipt is None:
        return None

    receipt_path = _next_attempt_path(control_dir, "rank-consistency")
    atomic_json(receipt_path, receipt)
    try:
        rank_consistency.require_pass(receipt)
    except rank_consistency.RankConsistencyError as exc:
        raise PrelaunchEngineeringError(str(exc)) from exc
    return {
        "path": str(receipt_path),
        "sha256": prelaunch_zero_formal.sha256_file(receipt_path),
        "receipt_sha256": receipt["receipt_sha256"],
        "receipt": receipt,
    }


def _next_formal_start_path(control_dir: Path, authorized_basename: str | None = None) -> Path:
    if authorized_basename is None:
        return _next_attempt_path(control_dir, "formal-start")
    match = re.fullmatch(r"formal-start-attempt-(\d{4})\.json", authorized_basename)
    if match is None:
        raise LaunchError("recovery authority has an invalid formal-start basename")
    requested = int(match.group(1))
    existing = [
        int(found.group(1))
        for path in control_dir.glob("formal-start-attempt-*.json")
        if (found := re.fullmatch(r"formal-start-attempt-(\d{4})\.json", path.name))
    ]
    if requested <= max(existing, default=0):
        raise LaunchError(
            "recovery-authorized Formal Start attempt is not newer than existing receipts"
        )
    receipt_path = control_dir / authorized_basename
    if receipt_path.exists():
        raise LaunchError(f"formal start receipt path already exists: {receipt_path}")
    return receipt_path


def _recovery_formal_start_basename(readiness: Mapping[str, Any], repo_root: Path) -> str | None:
    """Resolve the append-only Formal Start name from the exact current release."""
    release_raw = readiness.get("release_receipt_path")
    release_sha = readiness.get("release_receipt_sha256")
    if not release_raw and not release_sha:
        return None
    if not isinstance(release_raw, str) or not isinstance(release_sha, str):
        raise LaunchError("readiness omitted the exact release receipt identity")
    release_path = Path(release_raw).expanduser().resolve()
    root = repo_root.expanduser().resolve()
    if release_path != root and root not in release_path.parents:
        raise LaunchError("release receipt escapes the execution repository")
    if not release_path.is_file() or prelaunch_zero_formal.sha256_file(release_path) != release_sha:
        raise LaunchError("release receipt changed after readiness evaluation")
    try:
        release = json.loads(release_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchError(f"could not read current release receipt: {exc}") from exc
    recovery = release.get("engineering_recovery")
    if recovery is None:
        return None
    if not isinstance(recovery, Mapping):
        raise LaunchError("release engineering_recovery binding is malformed")
    recovery_raw = recovery.get("path")
    recovery_sha = recovery.get("sha256")
    if not isinstance(recovery_raw, str) or not isinstance(recovery_sha, str):
        raise LaunchError("release engineering_recovery lacks path or SHA-256")
    recovery_path = Path(recovery_raw).expanduser()
    if not recovery_path.is_absolute():
        recovery_path = root / recovery_path
    recovery_path = recovery_path.resolve()
    if recovery_path != root and root not in recovery_path.parents:
        raise LaunchError("engineering recovery receipt escapes the execution repository")
    if (not recovery_path.is_file()
            or prelaunch_zero_formal.sha256_file(recovery_path).removeprefix("sha256:")
            != recovery_sha.removeprefix("sha256:")):
        raise LaunchError("release-bound engineering recovery receipt is missing or changed")
    try:
        recovery_receipt = json.loads(recovery_path.read_text(encoding="utf-8"))
        continuation = recovery_receipt["continuation"]
        basename = continuation["formal_start_basename"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise LaunchError(f"recovery receipt has no Formal Start continuation: {exc}") from exc
    if not isinstance(basename, str) or not re.fullmatch(r"formal-start-attempt-\d{4}\.json", basename):
        raise LaunchError("recovery receipt has an invalid Formal Start continuation basename")
    return basename


def write_formal_start_receipt(
    profile: Mapping[str, Any],
    selected: list[str],
    run_dir: Path,
    immutable_image: str,
    command: list[str],
    prelaunch: Mapping[str, Any],
    control_dir: Path,
    receipt_path: Path | None = None,
    on_committed: Callable[[], None] | None = None,
) -> dict[str, Any] | None:
    context = profile.get("_formal_start_context")
    if not isinstance(context, Mapping):
        return None
    receipt_path = receipt_path or _next_formal_start_path(control_dir)
    if receipt_path.exists():
        raise LaunchError(f"formal start receipt path already exists: {receipt_path}")
    tracking = profile.get("tracking", {}) if isinstance(profile.get("tracking"), Mapping) else {}
    campaign_id = str(profile.get("campaign_id") or tracking.get("group") or profile["run_id"])
    receipt = prelaunch_zero_formal.build_formal_start_receipt(
        run_id=str(profile["run_id"]),
        campaign_id=campaign_id,
        selected_gpu_uuids=selected,
        image_id=immutable_image,
        command_identity=_command_identity(command, immutable_image),
        prelaunch_receipt_path=Path(str(prelaunch["path"])),
        readiness=context,
        output_root=str(run_dir),
        runtime_assets=_runtime_asset_receipt_identity(profile),
    )
    atomic_json(receipt_path, receipt)
    if on_committed is not None:
        on_committed()
    return {
        "path": str(receipt_path),
        "sha256": prelaunch_zero_formal.sha256_file(receipt_path),
        "receipt_sha256": receipt["receipt_sha256"],
        "receipt": receipt,
    }


def execute_gpu_command(
    profile: Mapping[str, Any],
    selected: list[str],
    command: list[str],
    run_dir: Path,
    immutable_image: str,
) -> tuple[int, dict[str, Any]]:
    """Run one GPU-owning command behind an atomic exact-accounting receipt."""
    for asset in profile.get("_runtime_asset_snapshots", []):
        snapshot = Path(asset["source"])
        if snapshot.is_symlink() or runtime_asset_digest(snapshot, asset["expected_type"]) != asset["sha256"]:
            raise LaunchError(f"runtime asset snapshot changed immediately before execution: {snapshot}")
        entries = [snapshot, *snapshot.rglob("*")] if asset["expected_type"] == "dir" else [snapshot]
        for entry in entries:
            mode = entry.lstat().st_mode
            if stat.S_ISLNK(mode) or mode & (stat.S_IWGRP | stat.S_IWOTH | stat.S_IWUSR):
                raise LaunchError(f"runtime asset snapshot contains a writable or linked entry before execution: {entry}")
    validate_host_runtime_source(profile)
    verify_host_runtime_imports(profile)
    receipt_path = run_dir / "gpu-resource-receipt.json"
    tracking = profile.get("tracking", {}) if isinstance(profile.get("tracking"), Mapping) else {}
    campaign_id = str(profile.get("campaign_id") or tracking.get("group") or profile["run_id"])
    wrapper_args = argparse.Namespace(
        receipt=str(receipt_path),
        attempt_id=str(profile["run_id"]),
        campaign_id=campaign_id,
        stage="launch",
        command_identity=_command_identity(command, immutable_image),
        gpu_uuids=",".join(selected),
        path_contract=None,
        container_root=None,
        command=command,
    )
    try:
        return_code = gpu_resource_wrapper.run(wrapper_args)
        accounting = gpu_resource_wrapper.classify_receipt(receipt_path)
    except Exception as exc:
        raise NonRetryableLaunchError(
            f"GPU payload invocation or resource receipt evaluation failed; task consumption may have occurred: {exc}"
        ) from exc
    if accounting.get("status") != "exact":
        raise NonRetryableLaunchError(
            f"GPU resource receipt is not exact after payload exit; task consumption may have occurred: {accounting}"
        )
    try:
        for asset in profile.get("_runtime_asset_snapshots", []):
            snapshot = Path(asset["source"])
            if snapshot.is_symlink() or runtime_asset_digest(snapshot, asset["expected_type"]) != asset["sha256"]:
                raise LaunchError(f"runtime asset snapshot changed during execution: {snapshot}")
            entries = [snapshot, *snapshot.rglob("*")] if asset["expected_type"] == "dir" else [snapshot]
            for entry in entries:
                mode = entry.lstat().st_mode
                if stat.S_ISLNK(mode) or mode & (stat.S_IWGRP | stat.S_IWOTH | stat.S_IWUSR):
                    raise LaunchError(f"runtime asset snapshot became writable or linked during execution: {entry}")
        validate_host_runtime_source(profile)
    except (LaunchError, OSError) as exc:
        raise NonRetryableLaunchError(f"post-run runtime asset validation failed: {exc}") from exc
    return return_code, {
        "path": str(receipt_path),
        "sha256": "sha256:" + hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "accounting": accounting,
    }


def command_launch(args: argparse.Namespace, pre_launch_hook=None) -> int:
    profile = load_profile(args.profile)
    gate = dict(profile.get("gpu_gate", {}))
    if args.timeout_seconds:
        gate["timeout_seconds"] = args.timeout_seconds
    profile = {
        **profile,
        "run_id": args.run_id or profile["run_id"],
        "wait": args.wait,
        "resume": args.resume,
        "gpu_gate": gate,
        "_formal_start_context": getattr(args, "formal_start_context", None),
        "_runtime_asset_snapshots": [],
    }
    validate_sources_and_paths(profile)
    validate_host_runtime_source(profile)
    immutable_image = resolve_image_identity(profile)
    verify_runtime_assets_visible_to_docker(profile, immutable_image)
    runtime_allowed = [value for value in os.environ.get(DEFAULT_AUTH_ENV, "").split(",") if value.strip()]
    run_dir = ensure_run_directory(profile)
    write_host_runtime_import_receipt(profile, run_dir)
    control_dir = _control_evidence_dir(profile, run_dir)
    context = profile.get("_formal_start_context")
    if isinstance(context, Mapping) and isinstance(context.get("ready_at_epoch"), (int, float)):
        _record_lifecycle(
            profile,
            control_dir,
            key="T0",
            epoch=float(context["ready_at_epoch"]),
            meaning="current release/readiness authority became machine-ready",
            source=dict(context.get("ready_receipt") or {}),
        )
    try:
        rank_receipt = run_rank_consistency_preflight(profile, control_dir)
    except PrelaunchEngineeringError as exc:
        _record_engineering_incident(
            profile,
            layer="rank_consistency",
            code="RANK_CONSISTENCY_FAILED",
            validator_id="scripts/openevo_webshop/rank_consistency.py",
            repair_class="SEMANTICS_PRESERVING_CONFIG_CLOSURE",
            detail=str(exc),
        )
        if profile.get("_formal_start_context") and not profile.get("resume", False):
            try:
                run_dir.rmdir()
            except OSError:
                pass
        raise
    if rank_receipt is not None and isinstance(profile.get("_formal_start_context"), Mapping):
        profile["_formal_start_context"] = {
            **dict(profile["_formal_start_context"]),
            "rank_consistency": rank_receipt,
        }
    effective_mode, tracking_reason, tracking_probe_receipt, tracking_probe_reused = choose_effective_tracking(
        profile,
        immutable_image,
        refresh=args.refresh_tracking_probe,
    )
    resource_mode, max_memory, max_util, minimum_free, interval, timeout = _gate_settings(profile)
    capacity_gate = _capacity_gate_settings(profile)
    auto = _auto_count(args.gpus) is not None
    selection_started = time.monotonic()
    selection_attempt = 0

    while True:
        selection_attempt += 1
        cards = snapshot_gpus()
        try:
            selected = select_gpu_uuids(
                args.gpus,
                runtime_allowed,
                cards,
                max_memory_mib=max_memory,
                max_utilization=max_util,
                resource_mode=resource_mode,
                minimum_free_memory_mib=minimum_free,
                capacity_gate=capacity_gate,
            )
            if auto:
                _record_auto_selection(control_dir, selection_attempt, cards, runtime_allowed, selected, None)
        except LaunchError as exc:
            if auto:
                _record_auto_selection(control_dir, selection_attempt, cards, runtime_allowed, None, str(exc))
            if not (auto and args.wait):
                raise
            if timeout and time.monotonic() - selection_started >= timeout:
                raise LaunchError(f"auto GPU selection timed out after {selection_attempt} attempts") from exc
            time.sleep(interval)
            continue

        try:
            # --wait must also cover a transient project-lock handoff for an
            # exact authorized UUID; the existing no-replay/authority gates remain.
            lock_wait = 0.0
            if args.wait:
                remaining = timeout - (time.monotonic() - selection_started) if timeout else 60.0
                lock_wait = max(0.0, min(60.0, remaining))
            with gpu_locks(selected, workspace_path(profile, str(profile.get("lock_root", ".gpu-locks"))),
                           wait_seconds=lock_wait):
                gate_profile = profile if not auto else {**profile, "wait": False}
                final_cards = wait_for_idle(gate_profile, selected, control_dir)
                resource_observed_at = time.time()
                _record_lifecycle(
                    profile,
                    control_dir,
                    key="T1",
                    epoch=resource_observed_at,
                    meaning="selected GPU UUIDs passed the live resource gate",
                    source={"selected_gpu_uuids": list(selected)},
                )

                lease_receipt: dict[str, Any] | None = None
                lease_ref: dict[str, Any] | None = None
                lease_handed_off = False
                if isinstance(profile.get("_formal_start_context"), Mapping):
                    prelaunch_cfg = profile.get("prelaunch")
                    ttl = (
                        int(prelaunch_cfg.get("lease_ttl_seconds", 900))
                        if isinstance(prelaunch_cfg, Mapping)
                        else 900
                    )
                    ready_ref = dict(profile["_formal_start_context"].get("ready_receipt") or {})
                    lease_receipt = reliability.build_prelaunch_lease(
                        run_id=str(profile["run_id"]),
                        campaign_id=_campaign_id(profile),
                        selected_gpu_uuids=selected,
                        runtime_candidate_uuids=runtime_allowed,
                        readiness_receipt_sha256=ready_ref.get("sha256"),
                        ttl_seconds=ttl,
                    )
                    lease_path = _next_attempt_path(control_dir, "prelaunch-lease")
                    atomic_json(lease_path, lease_receipt)
                    lease_ref = {
                        "path": str(lease_path),
                        "sha256": prelaunch_zero_formal.sha256_file(lease_path),
                        "receipt_sha256": lease_receipt["receipt_sha256"],
                    }
                    profile["_formal_start_context"] = {
                        **dict(profile["_formal_start_context"]),
                        "prelaunch_coordination_lease": lease_ref,
                    }
                    _record_lifecycle(
                        profile,
                        control_dir,
                        key="T2",
                        epoch=float(lease_receipt["acquired_at_epoch"]),
                        meaning="bounded project-launcher coordination lease acquired",
                        source=lease_ref,
                    )

                try:
                    # Before binding the selected devices to a payload, re-prove
                    # the campaign authority and the zero-formal runtime path.
                    if pre_launch_hook is not None:
                        pre_launch_hook()
                    try:
                        prelaunch = run_selected_gpu_preflight(
                            profile, selected, run_dir, immutable_image, control_dir
                        )
                    except LaunchError as exc:
                        _record_engineering_incident(
                            profile,
                            layer="selected_gpu_preflight",
                            code="SELECTED_GPU_ZERO_FORMAL_FAILED",
                            validator_id="scripts/openevo_webshop/prelaunch_zero_formal.py",
                            repair_class="SEMANTICS_PRESERVING_RUNTIME_REPAIR",
                            detail=str(exc),
                            selected_gpu_uuids=selected,
                        )
                        if profile.get("_formal_start_context") and not profile.get("resume", False):
                            try:
                                run_dir.rmdir()
                            except OSError:
                                pass
                        raise

                    _record_lifecycle(
                        profile,
                        control_dir,
                        key="T3",
                        epoch=float(prelaunch["receipt"]["observed_at_epoch"]),
                        meaning="selected-GPU zero-formal preflight passed",
                        source={
                            "path": prelaunch["path"],
                            "sha256": prelaunch["sha256"],
                            "receipt_sha256": prelaunch["receipt_sha256"],
                        },
                    )
                    manifest = {
                        "schema": "openevo.webshop.launch-receipt.v2",
                        "run_id": profile["run_id"],
                        "profile": str(args.profile),
                        "image_ref": profile["image"],
                        "image_id": immutable_image,
                        "gpu_selector": args.gpus,
                        "runtime_candidate_uuids": sorted(set(runtime_allowed)),
                        "selected_gpu_uuids": selected,
                        "tracking": {
                            "requested_mode": profile["tracking"]["mode"],
                            "effective_mode": effective_mode,
                            "fallback_reason": tracking_reason,
                            "probe_receipt": tracking_probe_receipt,
                            "probe_reused": tracking_probe_reused,
                        },
                        "started_at_epoch": time.time(),
                        "gpu_after_gate": final_cards,
                        "prelaunch_zero_formal": {
                            "path": prelaunch["path"],
                            "sha256": prelaunch["sha256"],
                            "receipt_sha256": prelaunch["receipt_sha256"],
                        },
                        "prelaunch_coordination_lease": lease_ref,
                        "rank_consistency": rank_receipt,
                    }
                    atomic_json(run_dir / "launch-receipt.json", manifest)

                    execution = profile.get("execution") if isinstance(profile.get("execution"), Mapping) else {}
                    execution_mode = str(execution.get("mode") or "container")
                    if execution_mode == "container":
                        command = docker_command(profile, selected, immutable_image, effective_mode)
                        formal_start_path = None
                    elif execution_mode == "host-orchestrator":
                        start_basename = None
                        if isinstance(profile.get("_formal_start_context"), Mapping):
                            start_basename = profile["_formal_start_context"].get(
                                "formal_start_basename"
                            )
                        formal_start_path = _next_formal_start_path(
                            control_dir, start_basename
                        )
                        command = host_orchestrator_command(
                            profile, selected, run_dir, effective_mode, formal_start_path
                        )
                    else:
                        raise LaunchError(f"unsupported execution mode: {execution_mode}")
                    manifest["execution_mode"] = execution_mode
                    manifest["runtime_assets"] = []
                    for asset in profile.get("_runtime_asset_snapshots", []):
                        item = {
                            "container_target": asset["target"],
                            "expected_type": asset["expected_type"],
                            "sha256": asset["sha256"],
                            "staged_source": asset["source"],
                        }
                        if "base_target" in asset:
                            item["overrides_base_target"] = asset["base_target"]
                        manifest["runtime_assets"].append(item)
                    if isinstance(profile.get("_host_runtime_source"), Mapping):
                        manifest["host_effective_source"] = {
                            key: profile["_host_runtime_source"][key]
                            for key in ("source_root", "target", "sha256", "base_sha256", "overlay_targets", "receipt_path", "receipt_sha256")
                        }
                    if isinstance(profile.get("_host_runtime_import_identity"), Mapping):
                        manifest["host_runtime_import_identity"] = profile["_host_runtime_import_identity"]
                    atomic_json(run_dir / "launch-receipt.json", manifest)

                    if lease_receipt is not None:
                        reliability.require_live_lease(
                            lease_receipt,
                            expected_gpu_uuids=selected,
                        )

                    def mark_formal_start_committed() -> None:
                        nonlocal lease_handed_off
                        if lease_receipt is not None:
                            # From the durable Formal Start write onward this is
                            # no longer a prelaunch scope, even if readback or
                            # observability bookkeeping fails afterward.
                            lease_handed_off = True

                    formal_start = write_formal_start_receipt(
                        profile,
                        selected,
                        run_dir,
                        immutable_image,
                        command,
                        prelaunch,
                        control_dir,
                        receipt_path=formal_start_path,
                        on_committed=mark_formal_start_committed,
                    )
                    if formal_start is not None:
                        manifest["formal_start"] = formal_start
                        if lease_receipt is not None:
                            handoff = reliability.build_lease_handoff(
                                lease_receipt,
                                formal_start_receipt_sha256=formal_start["receipt_sha256"],
                            )
                            handoff_path = _next_attempt_path(
                                control_dir, "prelaunch-lease-handoff"
                            )
                            atomic_json(handoff_path, handoff)
                            manifest["prelaunch_coordination_handoff"] = {
                                "path": str(handoff_path),
                                "sha256": prelaunch_zero_formal.sha256_file(handoff_path),
                                "receipt_sha256": handoff["receipt_sha256"],
                            }
                        _record_lifecycle(
                            profile,
                            control_dir,
                            key="T4",
                            epoch=float(formal_start["receipt"]["committed_at_epoch"]),
                            meaning="Formal Start boundary committed before task-bearing payload",
                            source={
                                "path": formal_start["path"],
                                "sha256": formal_start["sha256"],
                                "receipt_sha256": formal_start["receipt_sha256"],
                            },
                        )
                        atomic_json(run_dir / "launch-receipt.json", manifest)

                    print(json.dumps({
                        "run_id": profile["run_id"],
                        "image_id": immutable_image,
                        "gpu_uuids": selected,
                        "tracking_mode": effective_mode,
                        "tracking_probe_reused": tracking_probe_reused,
                    }, sort_keys=True))
                    return_code, resource = execute_gpu_command(
                        profile, selected, command, run_dir, immutable_image
                    )
                    manifest["exit_code"] = return_code
                    manifest["finished_at_epoch"] = time.time()
                    manifest["gpu_resource_receipt"] = resource
                    atomic_json(run_dir / "launch-receipt.json", manifest)
                    return return_code
                finally:
                    if lease_receipt is not None and not lease_handed_off:
                        release = reliability.build_lease_release(
                            lease_receipt,
                            reason="PRELAUNCH_SCOPE_EXIT",
                        )
                        release_path = _next_attempt_path(
                            control_dir, "prelaunch-lease-release"
                        )
                        atomic_json(release_path, release)
        except NonRetryableLaunchError:
            raise
        except LaunchError as exc:
            if not (auto and args.wait):
                raise
            if timeout and time.monotonic() - selection_started >= timeout:
                raise LaunchError(f"auto GPU acquisition timed out after {selection_attempt} attempts") from exc
            time.sleep(interval)


def command_run_campaign(args: argparse.Namespace) -> int:
    """Canonical campaign execution entrypoint: readiness first, then GPU launch."""
    if getattr(args, "model_load_preflight", False) and not getattr(args, "preflight_only", False):
        raise LaunchError("--model-load-preflight requires --preflight-only")
    if getattr(args, "preflight_only", False):
        return command_zero_formal_preflight(args)
    if getattr(args, "fingerprint_file", None) is None or getattr(args, "cache_root", None) is None:
        raise LaunchError("formal run-campaign requires --fingerprint-file and --cache-root")

    readiness_args = argparse.Namespace(
        release=getattr(args, "release", None),
        campaign=getattr(args, "campaign", None),
        current_campaign=getattr(args, "current_campaign", None),
        repo_root=getattr(args, "repo_root", None),
        fingerprint_file=args.fingerprint_file,
        cache_root=args.cache_root,
        output_root=args.output_root,
        run_output_root=getattr(args, "run_output_root", None),
        runtime_python=getattr(args, "runtime_python", None),
        source_root=getattr(args, "source_root", None),
        webshop_data=getattr(args, "webshop_data", None),
        model=getattr(args, "model", None),
        openevo_root=getattr(args, "openevo_root", None),
        index_root=getattr(args, "index_root", None),
        image=getattr(args, "image", None),
        expected_image_digest=getattr(args, "expected_image_digest", None),
        artifact_root=getattr(args, "artifact_root", None),
        storage_root=getattr(args, "storage_root", None),
        experiment_config=getattr(args, "experiment_config", None),
        run_manifest=getattr(args, "run_manifest", None),
        profile=getattr(args, "profile", None),
        runtime_container=getattr(args, "runtime_container", None),
        require_java=getattr(args, "require_java", None),
        require_cuda=getattr(args, "require_cuda", False),
        allow_dirty=getattr(args, "allow_dirty", False),
        resume=args.resume,
        refresh_qualification=getattr(args, "refresh_qualification", False),
    )

    result = execution_readiness.evaluate_readiness(readiness_args)
    verdict = result.get("verdict")
    if verdict != "READY":
        reason = result.get("error", verdict)
        raise LaunchError(f"execution readiness: {verdict}: {reason}")

    execution_repo_root = Path(getattr(args, "repo_root", None) or REPO_ROOT).expanduser().resolve()
    formal_start_basename = _recovery_formal_start_basename(result, execution_repo_root)

    profile_path = result.get("profile_path")
    if profile_path is None:
        raise LaunchError(
            "execution readiness passed but release receipt lacks a hash-bound "
            "execution_readiness.profile {path, sha256}; add one to the release "
            "receipt or launch via 'oe run <profile>'"
        )

    if args.dry_run:
        print(json.dumps({
            "verdict": "READY",
            "profile": profile_path,
            "profile_sha256": (result.get("profile") or {}).get("sha256"),
            "gpu_selector": args.gpus,
            "ready_receipt": result.get("ready_receipt_path"),
        }, sort_keys=True))
        print("GPU_LAUNCH_READY")
        return 0

    def reverify_readiness() -> None:
        """Re-prove readiness after a GPU wait, immediately before binding.

        A READY receipt is not a reservation: the campaign may close, the
        release may be superseded, or the frozen profile/artifacts may drift
        while ``--wait`` resamples the pool. Re-running the chain is cheap
        (runtime qualification is a cache hit) and fails closed.
        """
        again = execution_readiness.evaluate_readiness(readiness_args)
        if again.get("verdict") != "READY":
            raise AuthorityRevokedError(
                "execution readiness revoked while waiting for GPUs: "
                f"{again.get('verdict')}: {again.get('error', 'no reason recorded')}"
            )
        for key in ("release_receipt_path", "release_receipt_sha256"):
            if again.get(key) != result.get(key):
                raise AuthorityRevokedError(
                    f"execution authority changed while waiting for GPUs: {key}"
                )

    ready_receipt_path = Path(str(result["ready_receipt_path"])).resolve()
    ready_receipt_document = json.loads(ready_receipt_path.read_text(encoding="utf-8"))
    readiness_context = {
        "control_root": str(ready_receipt_path.parent),
        "ready_at_epoch": ready_receipt_document.get("ready_at_epoch"),
        "ready_receipt": {
            "path": str(ready_receipt_path),
            "sha256": prelaunch_zero_formal.sha256_file(ready_receipt_path),
        },
        "scientific_identity_sha256": result.get("scientific_identity_sha256"),
        "formal_start_basename": formal_start_basename,
        "release_receipt": {
            "path": result.get("release_receipt_path"),
            "sha256": result.get("release_receipt_sha256"),
        },
        "profile": {
            "path": str(profile_path),
            "sha256": (result.get("profile") or {}).get("sha256"),
        },
        "authority": result.get("authority"),
    }
    launch_args = argparse.Namespace(
        profile=Path(profile_path),
        gpus=args.gpus,
        wait=args.wait,
        timeout_seconds=args.timeout_seconds,
        run_id=args.run_id,
        resume=args.resume,
        refresh_tracking_probe=args.refresh_tracking_probe,
        formal_start_context=readiness_context,
    )
    try:
        return command_launch(launch_args, pre_launch_hook=reverify_readiness)
    except LaunchError as exc:
        message = str(exc)
        gpu_resource_markers = (
            "live-idle GPU",
            "not eligible under",
            "GPU acquisition timed out",
            "GPU selection timed out",
        )
        if any(marker in message for marker in gpu_resource_markers):
            print(f"[PENDING_NO_GPU]         requested={args.gpus} reason={message}")
            print(f"oe: no eligible GPU; campaign remains ready: {message}", file=sys.stderr)
            return 2
        raise


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchError(f"{label} unreadable: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LaunchError(f"{label} must be a JSON object: {path}")
    return value


def _zero_formal_candidate_identity(current_path: Path, repo_root: Path) -> dict[str, Any]:
    """Validate candidate identity against trusted external bootstrap authority.

    Failures here intentionally create no campaign receipt: until the candidate
    and external pointer establish trusted campaign identity, evidence must not
    be written under an authoritative campaign namespace.
    """
    root = repo_root.expanduser().resolve()
    current_path = current_path.expanduser().resolve()
    if not current_path.is_file() or root not in current_path.parents:
        raise LaunchError("candidate current-campaign must be a file inside repo_root")
    head = _run(["git", "-C", str(root), "rev-parse", "HEAD"]).stdout.strip()
    dirty = _run(["git", "-C", str(root), "status", "--porcelain"]).stdout.strip()
    if dirty:
        raise LaunchError("zero-formal qualification requires an exact clean Git checkout")

    candidate_bytes = current_path.read_bytes()
    campaign = _read_json_object(current_path, "candidate current-campaign")
    campaign_id = campaign.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise LaunchError("candidate current-campaign requires campaign_id")
    if campaign_id in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", campaign_id):
        raise LaunchError("candidate campaign_id must be a safe path slug")
    if campaign.get("status") != "owner-authorized-plan-frozen-formal-locked":
        raise LaunchError("candidate current-campaign is not an owner-authorized formal-locked plan")
    if campaign.get("formal_task_consumption_allowed") is not False:
        raise LaunchError("zero-formal route requires formal_task_consumption_allowed=false")
    if campaign.get("task_bearing_train_dev_allowed") is not False:
        raise LaunchError("zero-formal route requires task_bearing_train_dev_allowed=false")
    if campaign.get("gpu_allocation_allowed") is not True:
        raise LaunchError("candidate does not authorize zero-formal GPU qualification")
    if campaign.get("final_test_status") != "locked" or campaign.get("final_panel_access_allowed") is not False:
        raise LaunchError("candidate final panel must remain locked")

    design_rel = campaign.get("design")
    if not isinstance(design_rel, str):
        raise LaunchError("candidate current-campaign requires a design path")
    design_path = (root / design_rel).resolve()
    if root not in design_path.parents or not design_path.is_file():
        raise LaunchError("candidate design path escapes repo_root or is missing")
    design = _read_json_object(design_path, "candidate design")
    if design.get("final_panel_access") != 0:
        raise LaunchError("candidate design must declare final_panel_access=0")
    final_contract = design.get("frozen_scientific_contract")
    if isinstance(final_contract, dict) and final_contract.get("final_panel_unlock") is not False:
        raise LaunchError("candidate design must keep final_panel_unlock=false")

    expected_revision = "".join(char for char in campaign_id.split("-", 2)[:2] if char.isdigit())
    activation_dir = root / "configs" / "experiment" / "activations"
    activations: list[tuple[Path, dict[str, Any]]] = []
    if activation_dir.is_dir():
        for path in activation_dir.glob("*.json"):
            try:
                value = _read_json_object(path, "activation")
            except LaunchError:
                continue
            if value.get("design_path") == design_rel and value.get("revision") == expected_revision:
                activations.append((path, value))
    if len(activations) != 1:
        raise LaunchError("candidate must resolve to exactly one activation by design path and campaign revision")
    activation_path, activation = activations[0]
    if activation.get("status") != "PENDING_ZERO_FORMAL_PREFLIGHT":
        raise LaunchError("candidate activation must be PENDING_ZERO_FORMAL_PREFLIGHT")
    if activation.get("owner_authorized") is not True or activation.get("formal_task_consumption_allowed") is not False:
        raise LaunchError("candidate activation lacks zero-formal owner authorization")
    if activation.get("final_panel_access_allowed") is not False:
        raise LaunchError("candidate activation must keep final panel access disabled")
    if activation.get("execution_sha") not in (None, head):
        raise LaunchError("candidate activation execution_sha does not match the clean checkout")
    gpu_uuids = campaign.get("authorized_gpu_uuids")
    if not isinstance(gpu_uuids, list) or not gpu_uuids or activation.get("gpu_uuids") != gpu_uuids:
        raise LaunchError("campaign and activation GPU UUID allowlists must match")

    pointer_path = CANONICAL_WORKSPACE_REGISTRY_POINTER.expanduser().resolve()
    try:
        pointer_sha_before = prelaunch_zero_formal.sha256_file(pointer_path)
        discovery = workspace_discovery.classify_checkout(
            root, registry_path=pointer_path
        )
        pointer_sha_after = prelaunch_zero_formal.sha256_file(pointer_path)
    except (workspace_discovery.WorkspaceDiscoveryError, OSError, json.JSONDecodeError) as exc:
        raise LaunchError("repo-external canonical workspace pointer is unavailable or invalid") from exc
    if pointer_sha_before != pointer_sha_after:
        raise LaunchError("repo-external canonical workspace pointer changed during discovery")
    canonical_control = discovery.get("canonical_control_checkout")
    if not canonical_control:
        raise LaunchError("server registry has no canonical control checkout")
    host_root = Path(str(canonical_control)).expanduser().resolve()
    try:
        host_head = _run(["git", "-C", str(host_root), "rev-parse", "HEAD"]).stdout.strip()
        host_dirty = _run(["git", "-C", str(host_root), "status", "--porcelain"]).stdout.strip()
        if host_dirty:
            raise LaunchError("canonical host control checkout must be clean")
        _run(["git", "-C", str(host_root), "merge-base", "--is-ancestor", head, host_head])
    except subprocess.CalledProcessError as exc:
        raise LaunchError("candidate execution SHA is not contained in canonical host control HEAD") from exc
    except OSError as exc:
        raise LaunchError("canonical host control checkout Git identity is unavailable") from exc
    # Zero-formal qualification may use a campaign-specific candidate router.
    # Bind the exact relative candidate path on the canonical control checkout;
    # do not replace the repository-wide current-campaign pointer before the
    # candidate has passed its own qualification/release gates.
    candidate_rel = current_path.relative_to(root)
    host_router = Path(str(canonical_control)) / candidate_rel
    try:
        host_bytes = host_router.read_bytes()
    except OSError as exc:
        raise LaunchError(f"canonical host current-campaign is unavailable: {host_router}") from exc
    if hashlib.sha256(host_bytes).digest() != hashlib.sha256(candidate_bytes).digest():
        raise LaunchError("canonical host candidate router does not exactly match the zero-formal candidate")
    host_design = Path(str(canonical_control)).resolve() / design_rel
    host_activation = Path(str(canonical_control)).resolve() / activation_path.relative_to(root)
    try:
        if prelaunch_zero_formal.sha256_file(host_design) != prelaunch_zero_formal.sha256_file(design_path):
            raise LaunchError("canonical host design does not match the zero-formal candidate")
        if prelaunch_zero_formal.sha256_file(host_activation) != prelaunch_zero_formal.sha256_file(activation_path):
            raise LaunchError("canonical host activation does not match the zero-formal candidate")
    except OSError as exc:
        raise LaunchError("canonical host design/activation artifacts are unavailable") from exc

    return {
        "campaign": campaign,
        "campaign_id": campaign_id,
        "execution_sha": head,
        "current_campaign_sha256": prelaunch_zero_formal.sha256_file(current_path),
        "design_sha256": prelaunch_zero_formal.sha256_file(design_path),
        "activation_sha256": prelaunch_zero_formal.sha256_file(activation_path),
        "host_router_path": str(host_router),
        "host_control_head": host_head,
        "registry_pointer_path": str(pointer_path),
        "registry_pointer_sha256": pointer_sha_after,
        "gpu_uuids": gpu_uuids,
    }


@contextlib.contextmanager
def zero_formal_campaign_lock(registry_pointer: Path, campaign_id: str):
    """Serialize preflight attempts for one campaign; never waits or reserves GPU."""
    lock_root = registry_pointer.expanduser().resolve().parent / ".locks"
    lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_stat = lock_root.lstat()
    if not stat.S_ISDIR(lock_stat.st_mode) or lock_stat.st_uid != os.getuid():
        raise LaunchError("shared preflight lock directory must be a user-owned directory")
    if stat.S_IMODE(lock_stat.st_mode) != 0o700:
        lock_root.chmod(0o700)
        lock_stat = lock_root.lstat()
        if stat.S_IMODE(lock_stat.st_mode) != 0o700:
            raise LaunchError("shared preflight lock directory must have mode 0700")
    campaign_key = hashlib.sha256(campaign_id.encode("utf-8")).hexdigest()
    lock_path = lock_root / f"zero-formal-{campaign_key}.lock"
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LaunchError(f"zero-formal preflight already running for campaign {campaign_id}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def command_zero_formal_preflight(args: argparse.Namespace) -> int:
    """Run engineering-only selected-GPU qualification without Formal Start."""
    repo_root = Path(args.repo_root or REPO_ROOT).expanduser().resolve()
    current_path = Path(args.current_campaign).expanduser()
    if not current_path.is_absolute():
        current_path = repo_root / current_path
    identity = _zero_formal_candidate_identity(current_path, repo_root)
    output_root = Path(args.output_root).expanduser().resolve()
    control_dir = (output_root / "control" / identity["campaign_id"]).resolve()
    try:
        control_dir.relative_to(output_root)
    except ValueError as exc:
        raise LaunchError("zero-formal control receipt path escapes output_root") from exc
    identity = {**identity, "repo_root": repo_root, "current_campaign_path": current_path}
    with zero_formal_campaign_lock(Path(identity["registry_pointer_path"]), identity["campaign_id"]):
        return _run_zero_formal_preflight(args, identity, control_dir)


def _run_zero_formal_preflight(
    args: argparse.Namespace, identity: Mapping[str, Any], control_dir: Path
) -> int:
    profile_path_value = getattr(args, "preflight_profile", None)
    profile_path: Path | None = None
    profile_sha256: str | None = None
    profile: dict[str, Any] | None = None
    selected: list[str] = []
    run_dir: Path | None = None
    run_dir_created = False
    prelaunch_result: dict[str, Any] | None = None
    model_load_result: dict[str, Any] | None = None
    prelaunch_attempt_path: Path | None = None
    try:
        if profile_path_value is None:
            raise LaunchError("--preflight-only requires --preflight-profile with an immutable container profile")
        profile_path = Path(profile_path_value).expanduser().resolve()
        profile_sha256 = prelaunch_zero_formal.sha256_file(profile_path)
        profile = load_profile(profile_path)
        if profile["source"]["experiment"]["revision"] != identity["execution_sha"]:
            raise LaunchError("preflight profile experiment revision does not match candidate execution SHA")
        validate_sources_and_paths(profile)
        resolve_image_identity(profile)
        profile_uid = int(profile["identity"]["uid"])
        profile_gid = int(profile["identity"]["gid"])
        if os.geteuid() == 0:
            if (profile_uid, profile_gid) != (1001, 1001):
                raise LaunchError("root Docker control plane requires scientific container UID/GID 1001:1001")
        elif profile_uid != os.getuid() or profile_gid != os.getgid():
            raise LaunchError("preflight profile container UID/GID must match the current host user")
        allowed = set(identity["gpu_uuids"])
        selected_arg = str(args.gpus)
        if _auto_count(selected_arg) is not None:
            raise LaunchError("zero-formal candidate preflight requires exact authorized GPU UUIDs")
        selected = [value.strip() for value in selected_arg.split(",") if value.strip()]
        if not selected or len(selected) != len(set(selected)) or not set(selected) <= allowed:
            raise LaunchError("selected GPU UUIDs must be a unique subset of the candidate allowlist")

        cards = snapshot_gpus()
        select_gpu_uuids(selected_arg, identity["gpu_uuids"], cards, **gpu_selection_kwargs(profile))
        lock_root = workspace_path(profile, str(profile.get("lock_root", ".gpu-locks")))
        with gpu_locks(selected, lock_root):
            gate_profile = {**profile, "wait": False, "gpu_gate": {**profile.get("gpu_gate", {}), "interval_seconds": 1}}
            wait_for_idle(gate_profile, selected, control_dir)
            revalidated = _zero_formal_candidate_identity(
                Path(identity["current_campaign_path"]), Path(identity["repo_root"])
            )
            for key in (
                "execution_sha", "current_campaign_sha256", "design_sha256", "activation_sha256",
                "host_control_head", "registry_pointer_path", "registry_pointer_sha256",
            ):
                if revalidated[key] != identity[key]:
                    raise LaunchError(f"zero-formal candidate authority changed before Docker smoke: {key}")
            run_dir = (
                workspace_path(profile, str(profile["paths"]["runs"]))
                / f".zero-formal-{identity['campaign_id']}-{uuid.uuid4().hex}"
            )
            run_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
            if os.geteuid() == 0:
                os.chown(run_dir, profile_uid, profile_gid)
            run_dir_created = True
            smoke_profile = {**profile, "run_id": run_dir.name}
            prelaunch_attempt = prelaunch_zero_formal.next_attempt(control_dir)
            prelaunch_attempt_path = control_dir / f"prelaunch-zero-formal-attempt-{prelaunch_attempt:04d}.json"
            try:
                prelaunch = run_selected_gpu_preflight(
                    smoke_profile, selected, run_dir, profile["image_id"], control_dir
                )
            except Exception:
                # The helper persists detailed FAIL receipts before raising. Bind
                # that evidence when available without masking the original error.
                try:
                    failed_receipt = json.loads(prelaunch_attempt_path.read_text(encoding="utf-8"))
                    if isinstance(failed_receipt, dict):
                        prelaunch_result = {
                            "receipt": failed_receipt,
                            "sha256": prelaunch_zero_formal.sha256_file(prelaunch_attempt_path),
                        }
                except Exception:
                    pass
                raise
            prelaunch_result = prelaunch
            if getattr(args, "model_load_preflight", False):
                model_load_result = run_step119_model_load_preflight(
                    smoke_profile,
                    selected,
                    run_dir,
                    profile["image_id"],
                    control_dir,
                    revalidated,
                    profile_sha256,
                )
        receipt = prelaunch_zero_formal.build_qualification_receipt(
            campaign_id=identity["campaign_id"],
            execution_sha=identity["execution_sha"],
            current_campaign_sha256=identity["current_campaign_sha256"],
            design_sha256=identity["design_sha256"],
            activation_sha256=identity["activation_sha256"],
            profile_sha256=profile_sha256,
            host_router_path=identity["host_router_path"],
            host_control_head=identity["host_control_head"],
            registry_pointer_path=identity["registry_pointer_path"],
            registry_pointer_sha256=identity["registry_pointer_sha256"],
            selected_gpu_uuids=selected,
            prelaunch_receipt=prelaunch_result["receipt"],
            prelaunch_receipt_file_sha256=prelaunch_result["sha256"],
        )
        receipt_path = _next_attempt_path(control_dir, "zero-formal-qualification")
        atomic_json(receipt_path, receipt)
        print(json.dumps({
            "verdict": receipt["status"],
            "receipt": str(receipt_path),
            "model_load_preflight": model_load_result["path"] if model_load_result else None,
        }, sort_keys=True))
        return 0
    except (LaunchError, OSError, subprocess.CalledProcessError) as exc:
        try:
            blocked = prelaunch_zero_formal.build_qualification_receipt(
                campaign_id=identity["campaign_id"],
                execution_sha=identity["execution_sha"],
                current_campaign_sha256=identity["current_campaign_sha256"],
                design_sha256=identity["design_sha256"],
                activation_sha256=identity["activation_sha256"],
                profile_sha256=profile_sha256,
                host_router_path=identity["host_router_path"],
                host_control_head=identity["host_control_head"],
                registry_pointer_path=identity["registry_pointer_path"],
                registry_pointer_sha256=identity["registry_pointer_sha256"],
                selected_gpu_uuids=selected,
                prelaunch_receipt=prelaunch_result["receipt"] if prelaunch_result else None,
                prelaunch_receipt_file_sha256=prelaunch_result.get("sha256") if prelaunch_result else None,
                status="ZERO_FORMAL_BLOCKED",
                failure_reason=str(exc),
            )
            receipt_path = _next_attempt_path(control_dir, "zero-formal-qualification")
            atomic_json(receipt_path, blocked)
        except Exception as receipt_exc:
            print(f"oe: could not persist zero-formal blocked receipt: {receipt_exc}", file=sys.stderr)
        raise
    finally:
        if run_dir is not None and run_dir_created:
            try:
                run_dir.rmdir()
            except OSError:
                pass


def command_doctor(args: argparse.Namespace) -> int:
    """Canonical fast-start entrypoint: verify workspace, campaign, runtime, live gate.

    When invoked without a profile, this is the fresh-Agent startup check. It
    classifies the current checkout, reports campaign authority, reuses the
    KNOWN_GOOD runtime qualification cache, rechecks volatile launch facts, and
    optionally proves the full ``run-campaign`` chain with ``--dry-run``.

    The legacy profile/image-doctor mode is preserved when a positional profile
    argument is supplied.
    """
    if getattr(args, "profile", None) is not None:
        return _command_doctor_profile(args)

    return _command_doctor_startup(args)


def _command_doctor_profile(args: argparse.Namespace) -> int:
    """Original ``oe doctor <profile>`` behavior: validate profile + image."""
    profile = load_profile(args.profile)
    validate_sources_and_paths(profile, stage_runtime=False)
    immutable_image = resolve_image_identity(profile)
    print(json.dumps({"status": "pass", "profile": str(args.profile), "image_id": immutable_image, "gpu_launch": "not-run"}, sort_keys=True))
    return 0


def _command_doctor_startup(args: argparse.Namespace) -> int:
    """New fresh-Agent startup path."""
    import time

    output_root = Path(args.output_root).expanduser().resolve()
    control_root = output_root / "control"
    control_root.mkdir(mode=0o700, parents=True, exist_ok=True)

    receipt: dict[str, Any] = {
        "schema": "openevo.startup-readiness.v1",
        "observed_at_epoch": time.time(),
        "canonical_checkout": {},
        "campaign": {},
        "runtime_qualification": None,
        "live_gate": None,
        "run_campaign_dry_run": None,
        "next_action": "UNKNOWN",
        "next_command": None,
        "errors": [],
    }

    def fail(action: str, message: str, exit_code: int) -> int:
        receipt["next_action"] = action
        receipt["errors"].append(message)
        atomic_json(control_root / "startup-readiness.json", receipt)
        print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
        print(f"oe: startup readiness: {action}: {message}", file=sys.stderr)
        return exit_code

    # 1. Canonical checkout check (fail fast before expensive work).
    try:
        checkout = workspace_discovery.classify_checkout(
            REPO_ROOT,
            registry_path=args.registry,
            env_override=os.environ.get("OPENEVO_CANONICAL_CONTROL_CHECKOUT"),
        )
    except workspace_discovery.WorkspaceDiscoveryError as exc:
        return fail("WORKSPACE_DISCOVERY_ERROR", str(exc), 3)

    receipt["canonical_checkout"] = checkout
    if not checkout["is_canonical_control"]:
        next_command = checkout.get("next_command") or f"cd {checkout.get('canonical_control_checkout') or '<canonical-control-checkout>'} && ./scripts/oe doctor"
        receipt["next_command"] = next_command
        return fail("NON_CANONICAL_EXECUTION_CHECKOUT", next_command, 3)

    # 2. Campaign authority check.
    current_campaign_path = Path(args.current_campaign).expanduser().resolve()
    try:
        campaign = json.loads(current_campaign_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return fail("CAMPAIGN_LOAD_ERROR", f"cannot load {current_campaign_path}: {exc}", 4)

    receipt["campaign"] = {
        "campaign_id": campaign.get("campaign_id"),
        "status": campaign.get("status"),
        "formal_task_consumption_allowed": bool(campaign.get("formal_task_consumption_allowed", False)),
        "current_campaign_path": str(current_campaign_path),
    }

    status = campaign.get("status", "")
    formal_allowed = bool(campaign.get("formal_task_consumption_allowed", False))
    release_path: Path | None = getattr(args, "release", None)

    if status.startswith("closed") or status.startswith("complete"):
        # A closed campaign is still informative for startup; do not run GPU work.
        if release_path is None or not formal_allowed:
            receipt["next_action"] = "BLOCKED_CAMPAIGN_CLOSED"
            receipt["next_command"] = "no formal GPU execution for a closed campaign"
            atomic_json(control_root / "startup-readiness.json", receipt)
            print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
            print("oe: startup readiness: BLOCKED_CAMPAIGN_CLOSED: current campaign is closed or complete", file=sys.stderr)
            return 4

    if not formal_allowed and release_path is None:
        receipt["next_action"] = "PREPARE_REQUIRED"
        receipt["next_command"] = "create a launch-authorized release receipt and re-run with --release"
        atomic_json(control_root / "startup-readiness.json", receipt)
        print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
        print("oe: startup readiness: PREPARE_REQUIRED: campaign does not allow formal task consumption", file=sys.stderr)
        return 4

    # 3. Runtime qualification via fingerprint defaults.
    fingerprint_file = Path(args.fingerprint_file).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    try:
        qargs = execution_readiness.build_qualification_args(
            release={"upstream_patch": {}} if release_path is None else json.loads(release_path.read_text(encoding="utf-8")),
            fingerprint_file=fingerprint_file,
            cache_root=cache_root,
            repo_root=REPO_ROOT,
            allow_dirty=args.allow_dirty,
            profile=getattr(args, "profile_route", None),
            runtime_python=getattr(args, "runtime_python", None),
            source_root=getattr(args, "source_root", None),
            webshop_data=getattr(args, "webshop_data", None),
            model=getattr(args, "model", None),
            openevo_root=getattr(args, "openevo_root", None),
            index_root=getattr(args, "index_root", None),
            require_java=getattr(args, "require_java", None),
        )
        qualification = runtime_qualification.qualify(qargs)
    except Exception as exc:  # noqa: BLE001
        return fail("RUNTIME_QUALIFICATION_ERROR", str(exc), 5)

    receipt["runtime_qualification"] = {
        "status": qualification.get("status"),
        "reused": qualification.get("reused"),
        "cache_key": qualification.get("cache_key"),
        "fingerprint_sha256": qualification.get("request", {}).get("runtime_fingerprint_sha256"),
        "qualification_decision": qualification.get("qualification_decision"),
    }
    if qualification.get("status") != "PASS":
        return fail("RUNTIME_NOT_PROVISIONED", "; ".join(qualification.get("failures", ["unknown"])), 5)

    qualification_receipt_path = control_root / "runtime-qualification.json"
    atomic_json(qualification_receipt_path, qualification)

    # 4. Volatile live gate.
    try:
        gate_args = argparse.Namespace(
            qualification_receipt=qualification_receipt_path,
            runtime_python=qargs.runtime_python,
            cache_root=cache_root,
            artifact_root=None,
            storage_root=None,
            # The startup receipt directory already exists; check a distinct
            # prospective run directory for writability and collision instead.
            output_root=output_root / "run",
            minimum_free_bytes=runtime_live_gate.envdoc.DEFAULT_MIN_FREE_BYTES,
            require_cuda=False,
            resume=False,
            experiment_config=None,
            run_manifest=None,
            output=control_root / "runtime-live-gate.json",
        )
        live_gate = runtime_live_gate.run_gate(gate_args)
        if gate_args.output:
            atomic_json(gate_args.output, live_gate)
    except Exception as exc:  # noqa: BLE001
        return fail("RUNTIME_LIVE_GATE_ERROR", str(exc), 5)

    receipt["live_gate"] = {
        "status": live_gate.get("status"),
        "receipt_path": str(gate_args.output) if gate_args.output else None,
    }
    if live_gate.get("status") != "PASS":
        return fail("RUNTIME_LIVE_GATE_FAIL", "; ".join(live_gate.get("failures", ["unknown"])), 5)

    # 5. Optional full run-campaign dry-run to prove the chain end-to-end.
    if args.run_campaign_dry_run and release_path is not None and formal_allowed:
        try:
            dry_run_root = output_root / "dry-run"
            dry_run_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            dry_run_result = execution_readiness.evaluate_readiness(
                argparse.Namespace(
                    release=str(release_path),
                    campaign=None,
                    current_campaign=str(current_campaign_path),
                    repo_root=str(REPO_ROOT),
                    fingerprint_file=str(fingerprint_file),
                    cache_root=str(cache_root),
                    output_root=str(dry_run_root),
                    run_output_root=None,
                    runtime_python=qargs.runtime_python,
                    source_root=qargs.source_root,
                    webshop_data=qargs.webshop_data,
                    model=qargs.model,
                    openevo_root=qargs.openevo_root,
                    index_root=qargs.index_root,
                    image=qargs.image,
                    expected_image_digest=qargs.expected_image_digest,
                    artifact_root=None,
                    storage_root=None,
                    experiment_config=None,
                    run_manifest=None,
                    profile=qargs.profile,
                    runtime_container=qargs.runtime_container,
                    require_java=qargs.require_java,
                    require_cuda=False,
                    allow_dirty=qargs.allow_dirty,
                    resume=False,
                    refresh_qualification=False,
                )
            )
            receipt["run_campaign_dry_run"] = {
                "verdict": dry_run_result.get("verdict"),
                "ready_receipt": dry_run_result.get("ready_receipt_path"),
            }
            if dry_run_result.get("verdict") != "READY":
                return fail("RUN_CAMPAIGN_DRY_RUN_NOT_READY", dry_run_result.get("error", "unknown"), 6)
        except Exception as exc:  # noqa: BLE001
            return fail("RUN_CAMPAIGN_DRY_RUN_ERROR", str(exc), 6)

    # 6. Success: ready for next legal action.
    if release_path is not None and formal_allowed and receipt["run_campaign_dry_run"] is not None:
        receipt["next_action"] = "READY_TO_EXECUTE"
        receipt["next_command"] = f"./scripts/oe run-campaign --release {release_path} --fingerprint-file {fingerprint_file} --cache-root {cache_root} --output-root <run-output-root> --gpus auto:N --wait"
    else:
        receipt["next_action"] = "READY_CHECK_ALLOWED"
        receipt["next_command"] = "./scripts/oe doctor --release <release-receipt> to prove full readiness"

    atomic_json(control_root / "startup-readiness.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"STARTUP_READINESS: {receipt['next_action']}")
    return 0


def command_host(args: argparse.Namespace) -> int:
    """Run a host orchestrator from the checkout named by its path contract."""
    from scripts.openevo_webshop.path_contract import PathContractError

    try:
        contract = json.loads(args.path_contract.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise LaunchError(f"host path contract cannot be read: {exc}") from exc
    if not isinstance(contract, dict):
        raise LaunchError("host path contract must be a JSON object")
    paths = contract.get("paths")
    if not isinstance(paths, dict):
        raise LaunchError("host path contract must contain named paths")
    experiment = paths.get("experiment")
    if not isinstance(experiment, dict) or not isinstance(experiment.get("host_path"), str):
        raise LaunchError("host path contract must bind an experiment checkout")
    try:
        cwd = args.cwd.resolve(strict=True)
        bound = Path(experiment["host_path"]).resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise LaunchError(f"host checkout path cannot be resolved: {exc}") from exc
    if cwd != bound:
        raise LaunchError(f"host checkout differs from path contract: cwd={cwd} contract={bound}")
    try:
        validate_contract(contract, probe=run_cpu_no_gpu_probe)
    except PathContractError as exc:
        raise LaunchError(f"host/container path contract failed before delegation: {exc}") from exc
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise LaunchError("host requires a command")
    return subprocess.run(command, cwd=args.cwd, check=False).returncode


def command_recover_plan(args: argparse.Namespace) -> int:
    """Read-only exactly-once recovery projection from sealed round evidence."""
    from scripts.openevo_runtime import round_transaction

    try:
        plan = round_transaction.build_round_recovery_plan(
            args.round_root,
            args.round_index,
            rollouts_per_round=args.rollouts_per_round,
        )
    except round_transaction.RoundPlanError as exc:
        raise LaunchError(f"recover-plan: {exc}") from exc
    print(
        json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False)
        if args.json
        else round_transaction.render(plan)
    )
    return 2 if plan.get("verdict") == "MANUAL_REVIEW_REQUIRED" else 0


def command_mark_first_valid_step(args: argparse.Namespace) -> int:
    """Record producer-validated T5 observability without granting science authority."""
    control_root = args.control_root.expanduser().resolve()
    summary = reliability.lifecycle_summary(control_root)
    source = args.source_receipt.expanduser().resolve()
    if not source.is_file():
        raise LaunchError(f"first-valid-step source receipt is missing: {source}")
    try:
        receipt = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchError(f"first-valid-step source receipt is unreadable: {source}") from exc
    if not isinstance(receipt, Mapping):
        raise LaunchError("first-valid-step source receipt must be a JSON object")
    if receipt.get("status") != "SEALED":
        raise LaunchError("first-valid-step source receipt is not SEALED")
    if receipt.get("run_id") != summary.get("run_id"):
        raise LaunchError("first-valid-step source receipt run_id does not match lifecycle")
    updated = reliability.record_lifecycle_event(
        control_root,
        run_id=str(summary["run_id"]),
        campaign_id=str(summary["campaign_id"]),
        key="T5",
        epoch=float(args.epoch if args.epoch is not None else time.time()),
        meaning="producer-declared first valid scientific step sealed",
        source={"path": str(source), "sha256": reliability.sha256_file(source)},
    )
    print(json.dumps(updated, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def command_reliability_report(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {
        "schema": "openevo.experiment-reliability-report.202609281950",
        "observability_only": True,
        "scientific_authority_granted": False,
    }
    if args.control_root is not None:
        payload["lifecycle"] = reliability.lifecycle_summary(
            args.control_root.expanduser().resolve()
        )
    if args.incident_root is not None:
        payload["incidents"] = reliability.incident_summary(
            args.incident_root.expanduser().resolve()
        )
    if "lifecycle" not in payload and "incidents" not in payload:
        raise LaunchError("reliability-report requires --control-root and/or --incident-root")
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def command_supervise_campaign(args: argparse.Namespace) -> int:
    """Plan or execute receipt-driven unattended orchestration.

    This is an engineering scheduler only. Formal execution is always
    delegated to the canonical run-campaign entrypoint named by the config.
    """
    from scripts.openevo_runtime import unattended_supervisor

    try:
        cfg = unattended_supervisor.config(args.config)
        if args.render_systemd:
            if args.execute:
                raise LaunchError("--render-systemd and --execute are mutually exclusive")
            payload = unattended_supervisor.render_service(cfg, REPO_ROOT / "scripts/oe")
        else:
            if args.execute and args.max_cycles is not None:
                raise LaunchError("--max-cycles is test/planning-only; do not bound a live execute loop")
            payload = unattended_supervisor.run(
                cfg,
                execute=bool(args.execute),
                max_cycles=args.max_cycles,
            )
    except unattended_supervisor.SupervisorError as exc:
        raise LaunchError(f"supervise-campaign: {exc}") from exc
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="subcommand", required=True)
    doctor = sub.add_parser("doctor")
    doctor.add_argument("profile", type=Path, nargs="?", help="legacy profile/image-doctor mode (optional)")
    doctor.add_argument("--registry", type=Path, default=Path("configs/platform/server-registry.json"), help="canonical workspace registry")
    doctor.add_argument("--current-campaign", type=Path, default=Path("configs/experiment/current-campaign.json"))
    doctor.add_argument("--fingerprint-file", type=Path, default=Path("configs/runtime/known-good-runtime-fingerprint-v1.json"))
    doctor.add_argument("--repo-root", type=Path, help="override repository root for all repo-relative defaults")
    doctor.add_argument("--cache-root", type=Path, default=Path("/workspace/runtime/qualification-cache"))
    doctor.add_argument("--output-root", type=Path, default=Path(f"/tmp/oe-startup-{time.strftime('%Y%m%d-%H%M%S')}"))
    doctor.add_argument("--release", type=Path, help="optional release receipt for full readiness dry-run")
    doctor.add_argument("--run-campaign-dry-run", action=argparse.BooleanOptionalAction, default=True, help="prove run-campaign --dry-run when release is ready")
    doctor.add_argument("--runtime-python", type=Path)
    doctor.add_argument("--source-root", type=Path)
    doctor.add_argument("--webshop-data", type=Path)
    doctor.add_argument("--model", type=Path)
    doctor.add_argument("--openevo-root", type=Path)
    doctor.add_argument("--index-root", type=Path)
    doctor.add_argument("--require-java", action=argparse.BooleanOptionalAction, default=None)
    doctor.add_argument("--allow-dirty", action="store_true")
    doctor.add_argument("--profile-route", default=None, help="qualification route profile override")
    for name in ("run",):
        child = sub.add_parser(name)
        child.add_argument("profile", type=Path)
        if name == "run":
            child.add_argument("--gpus", required=True, help="auto:N or comma-separated runtime-allowed GPU UUIDs")
            child.add_argument(
                "--wait",
                action="store_true",
                help="for auto:N, resample the whole runtime candidate pool until N GPUs are live-idle; for explicit UUIDs, wait for those exact UUIDs",
            )
            child.add_argument("--timeout-seconds", type=int, default=0)
            child.add_argument("--run-id")
            child.add_argument("--resume", action="store_true")
            child.add_argument(
                "--refresh-tracking-probe",
                action="store_true",
                help="repeat W&B probe after a deliberate network/auth change",
            )
    host = sub.add_parser("host")
    host.add_argument("--path-contract", type=Path, required=True)
    host.add_argument("--cwd", type=Path, default=Path.cwd())
    host.add_argument("command", nargs=argparse.REMAINDER)

    recover = sub.add_parser(
        "recover-plan",
        help="read-only exactly-once recovery projection; never launches work",
    )
    recover.add_argument("--round-root", type=Path, required=True)
    recover.add_argument("--round-index", type=int, required=True)
    recover.add_argument("--rollouts-per-round", type=int, default=128)
    recover.add_argument("--json", action="store_true")

    mark_step = sub.add_parser(
        "mark-first-valid-step",
        help="record producer-validated T5 lifecycle observability",
    )
    mark_step.add_argument("--control-root", type=Path, required=True)
    mark_step.add_argument("--source-receipt", type=Path, required=True)
    mark_step.add_argument("--epoch", type=float)

    report = sub.add_parser(
        "reliability-report",
        help="print non-authoritative lifecycle and repeated-incident observability",
    )
    report.add_argument("--control-root", type=Path)
    report.add_argument("--incident-root", type=Path)

    supervise = sub.add_parser(
        "supervise-campaign",
        help="receipt-driven unattended orchestration for independent formal arms",
    )
    supervise.add_argument("--config", type=Path, required=True)
    supervise.add_argument(
        "--execute",
        action="store_true",
        help="dispatch only actions that the current canonical oe path re-authorizes",
    )
    supervise.add_argument(
        "--render-systemd",
        action="store_true",
        help="render the existing fail-closed manual-start systemd bridge",
    )
    supervise.add_argument(
        "--max-cycles",
        type=int,
        help="bounded engineering/test loop; omitted for a long-lived supervisor",
    )

    run_campaign = sub.add_parser("run-campaign")
    run_campaign.add_argument("--release", type=Path, help="path to execution release receipt")
    run_campaign.add_argument("--campaign", help="campaign id (resolves via current-campaign.json)")
    run_campaign.add_argument("--current-campaign", type=Path, default=Path("configs/experiment/current-campaign.json"))
    run_campaign.add_argument("--repo-root", type=Path)
    run_campaign.add_argument("--fingerprint-file", type=Path)
    run_campaign.add_argument("--cache-root", type=Path)
    run_campaign.add_argument("--output-root", type=Path, required=True)
    run_campaign.add_argument(
        "--run-output-root",
        type=Path,
        help="formal run output root to collision-check (control receipts never live here)",
    )
    run_campaign.add_argument("--runtime-python", type=Path)
    run_campaign.add_argument("--source-root", type=Path)
    run_campaign.add_argument("--webshop-data", type=Path)
    run_campaign.add_argument("--model", type=Path)
    run_campaign.add_argument("--openevo-root", type=Path)
    run_campaign.add_argument("--index-root", type=Path)
    run_campaign.add_argument("--image")
    run_campaign.add_argument("--expected-image-digest")
    run_campaign.add_argument("--artifact-root", type=Path)
    run_campaign.add_argument("--storage-root", type=Path)
    run_campaign.add_argument("--experiment-config", type=Path)
    run_campaign.add_argument("--run-manifest", type=Path)
    run_campaign.add_argument(
        "--profile",
        default=None,
        help="qualification route profile; defaults to the runtime fingerprint's route.profile (else 'full')",
    )
    run_campaign.add_argument(
        "--runtime-container",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="force container-mode qualification; defaults to the runtime fingerprint execution mode",
    )
    run_campaign.add_argument(
        "--require-java",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="force the java qualification check; defaults to the runtime fingerprint route.requires_java",
    )
    run_campaign.add_argument("--require-cuda", action="store_true")
    run_campaign.add_argument("--allow-dirty", action="store_true")
    run_campaign.add_argument("--gpus", required=True, help="auto:N or comma-separated runtime-allowed GPU UUIDs")
    run_campaign.add_argument("--wait", action="store_true")
    run_campaign.add_argument("--timeout-seconds", type=int, default=0)
    run_campaign.add_argument("--run-id")
    run_campaign.add_argument("--resume", action="store_true")
    run_campaign.add_argument("--refresh-qualification", action="store_true")
    run_campaign.add_argument("--refresh-tracking-probe", action="store_true")
    run_campaign.add_argument("--dry-run", action="store_true")
    run_campaign.add_argument(
        "--preflight-only",
        action="store_true",
        help="run candidate-bound zero-formal engineering qualification without Formal Start or science payload",
    )
    run_campaign.add_argument(
        "--preflight-profile",
        type=Path,
        help="immutable container profile used only for the synthetic zero-formal GPU smoke",
    )
    run_campaign.add_argument(
        "--model-load-preflight",
        action="store_true",
        help="also run the pinned Q03 Step119 native model-load path with zero datasets/tasks",
    )

    args = parser.parse_args(argv)

    # If a repo root is supplied, resolve repo-relative default paths against it.
    repo_root = getattr(args, "repo_root", None)
    if repo_root is not None:
        root = Path(repo_root).expanduser().resolve()
        for attr in ("registry", "current_campaign", "fingerprint_file"):
            value = getattr(args, attr, None)
            if value is not None and not value.is_absolute():
                setattr(args, attr, root / value)
    if args.subcommand == "run-campaign":
        repo_root = getattr(args, "repo_root", None)
        if repo_root is not None:
            root = Path(repo_root).expanduser().resolve()
            for attr in ("current_campaign", "release"):
                value = getattr(args, attr, None)
                if value is not None and isinstance(value, Path) and not value.is_absolute():
                    setattr(args, attr, root / value)

    try:
        if args.subcommand == "doctor":
            return command_doctor(args)
        if args.subcommand == "host":
            return command_host(args)
        if args.subcommand == "recover-plan":
            return command_recover_plan(args)
        if args.subcommand == "mark-first-valid-step":
            return command_mark_first_valid_step(args)
        if args.subcommand == "reliability-report":
            return command_reliability_report(args)
        if args.subcommand == "supervise-campaign":
            return command_supervise_campaign(args)
        if args.subcommand == "run-campaign":
            return command_run_campaign(args)
        return command_launch(args)
    except (LaunchError, reliability.ReliabilityRecordError, subprocess.CalledProcessError) as exc:
        print(f"oe: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
