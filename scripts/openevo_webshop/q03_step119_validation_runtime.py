"""Run the frozen Step119 validation config under the experiment owner UID.

The OpenEVO launcher runs a host-orchestrator with root only for resource
control. This entrypoint validates the frozen inputs, drops to the profile's
unprivileged UID/GID, and only then invokes the pinned SEED main_ppo module.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import json
import os
import pwd
import re
import runpy
import stat
import sys
from pathlib import Path
from typing import Sequence


MODEL_SHA256 = "4b9207ce1d7fdbe9c3acb3e06684ef8d0a1bcdadf34d3fe6c23e8c47ec7f7b1b"
CONFIG_SHA256 = "f2a8580ea78fe4d6b08a6bc18c5cd78dd210113ca0517b832b5def2721c7097e"
GENERATION_CONFIG_SHA256 = "86b9e6a06a36bb731cb91b0f5fc184e28016e6cd9c46c4f11b59a0fd53bff629"
TOKENIZER_SHA256 = "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506"
TOKENIZER_CONFIG_SHA256 = "779410a34a98fb2c6c925a1be6f24d632f1723473161b037f6676eb6c04886c6"
CHAT_TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
PARQUET_SHA256 = "271bf898898887e2d68494080888cc63c7e6c92db5f573366ca21b773af7f71f"
MANIFEST_SHA256 = "4498a0821301043fa6d7b0677763d4e9f7c8165645f4e7dfb12af86670176d2d"
VALIDATION_ROOT_RELATIVE = "q03_step119_validation"
MANIFEST_RELATIVE = f"{VALIDATION_ROOT_RELATIVE}/task_manifest.json"
PARQUET_RELATIVE = f"{VALIDATION_ROOT_RELATIVE}/test.parquet"
CUSTOM_DATASET_RELATIVE = "verl/utils/dataset/q03_step119_validation_dataset.py"
MANIFEST_HELPER_RELATIVE = "verl/utils/dataset/q03_source_faithful_manifest.py"
CUSTOM_DATASET_SHA256 = "3ac6ae9336cb477168be79951b579ecad589869db68dd5d7dd56fa385a9ad954"
MANIFEST_HELPER_SHA256 = "b508f4ccd9e384bfbab537b4c5a447628a519885397bcdf37195d60d643975bc"
SEALED_TMPFS_ROOT = Path("/dev/shm")
EXPECTED_RUN_ROOT = Path("/workspace/runs/q03-step119-val128-20261025")


class RuntimeContractError(RuntimeError):
    pass


def _canonical_sha256(value: dict[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)


def validate_model_load_preflight_authorization(
    encoded_context: str | None,
    *,
    run_dir: Path,
    environ: dict[str, str] | None = None,
) -> dict[str, object]:
    """Require the candidate/profile/GPU context minted by the canonical launcher."""
    env = os.environ if environ is None else environ
    if not encoded_context:
        raise RuntimeContractError("model-load preflight requires launcher authorization context")
    try:
        padded = encoded_context + "=" * (-len(encoded_context) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        context = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeContractError("model-load preflight launcher context is invalid") from exc
    if not isinstance(context, dict):
        raise RuntimeContractError("model-load preflight launcher context must be an object")
    context_sha = context.get("context_sha256")
    projection = {key: value for key, value in context.items() if key != "context_sha256"}
    if not isinstance(context_sha, str) or _canonical_sha256(projection) != context_sha:
        raise RuntimeContractError("model-load preflight launcher context hash mismatch")
    if env.get("OPENEVO_STEP119_PREFLIGHT_CONTEXT_SHA256") != context_sha:
        raise RuntimeContractError("model-load preflight context is not bound to the launcher invocation")
    required = {
        "schema": "openevo.q03-step119-launcher-preflight-context.v1",
        "status": "LAUNCHER_AUTHORIZED_ZERO_TASK_PREFLIGHT",
        "formal_task_count": 0,
        "training_rollouts": 0,
        "generation_calls": 0,
        "optimizer_updates": 0,
        "final_panel_access": 0,
        "scientific_state_mutated": False,
    }
    for key, expected in required.items():
        if type(context.get(key)) is not type(expected) or context.get(key) != expected:
            raise RuntimeContractError(f"model-load preflight launcher context violates {key}")
    bindings = {
        "campaign_id": "OPENEVO_STEP119_CAMPAIGN_ID",
        "execution_sha": "OPENEVO_STEP119_EXECUTION_SHA",
        "runtime_profile_sha256": "OPENEVO_STEP119_PROFILE_SHA256",
        "effective_source_sha256": "OPENEVO_EFFECTIVE_SOURCE_SHA256",
        "image_id": "OPENEVO_STEP119_IMAGE_ID",
    }
    for field, env_name in bindings.items():
        value = context.get(field)
        if not isinstance(value, str) or not value or env.get(env_name) != value:
            raise RuntimeContractError(f"model-load preflight launcher context does not match {field}")
    if context.get("run_id") != run_dir.name:
        raise RuntimeContractError("model-load preflight launcher context does not match run directory")
    selected = context.get("selected_gpu_uuids")
    if not isinstance(selected, list) or not selected or any(not isinstance(item, str) for item in selected):
        raise RuntimeContractError("model-load preflight launcher context has invalid selected GPU UUIDs")
    if selected != [item for item in env.get("OPENEVO_AUTHORIZED_GPU_UUIDS", "").split(",") if item]:
        raise RuntimeContractError("model-load preflight selected GPUs differ from launcher authorization")
    if selected != [item for item in env.get("OPENEVO_ASSIGNED_GPU_UUIDS", "").split(",") if item]:
        raise RuntimeContractError("model-load preflight selected GPUs differ from assigned devices")
    candidate_hashes = context.get("candidate_hashes")
    required_hashes = ("current_campaign", "design", "activation", "runtime_profile", "registry_pointer")
    if not isinstance(candidate_hashes, dict) or any(
        not _is_sha256(candidate_hashes.get(key)) for key in required_hashes
    ):
        raise RuntimeContractError("model-load preflight launcher context lacks candidate hashes")
    if candidate_hashes["runtime_profile"] != context["runtime_profile_sha256"]:
        raise RuntimeContractError("model-load preflight candidate profile hash is inconsistent")
    if not _is_sha256(context.get("effective_source_sha256")):
        raise RuntimeContractError("model-load preflight effective source hash is invalid")
    if not re.fullmatch(r"[0-9a-f]{40}", str(context.get("execution_sha") or "")):
        raise RuntimeContractError("model-load preflight execution SHA is invalid")
    if not _is_sha256(context.get("image_id")):
        raise RuntimeContractError("model-load preflight image digest is invalid")
    if len(selected) != len(set(selected)) or any(not re.fullmatch(r"GPU-[A-Za-z0-9-]+", item) for item in selected):
        raise RuntimeContractError("model-load preflight selected GPU UUIDs are invalid or duplicated")
    return context


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_regular(path: Path, expected_sha256: str, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeContractError(f"{label} must be a regular non-symlink file: {path}")
    observed = _sha256(path)
    if observed != expected_sha256:
        raise RuntimeContractError(f"{label} SHA-256 mismatch: expected {expected_sha256}, observed {observed}")


def _identity_access(path: Path, uid: int, gid: int) -> tuple[bool, bool]:
    info = path.lstat()
    try:
        account = pwd.getpwuid(uid)
        groups = set(os.getgrouplist(account.pw_name, gid))
    except (KeyError, OSError) as exc:
        raise RuntimeContractError(f"cannot resolve run-as identity {uid}:{gid}") from exc
    owner = info.st_uid == uid
    group = info.st_gid in groups
    mode = stat.S_IMODE(info.st_mode)
    readable = bool(mode & (stat.S_IRUSR if owner else stat.S_IRGRP if group else stat.S_IROTH))
    writable = bool(mode & (stat.S_IWUSR if owner else stat.S_IWGRP if group else stat.S_IWOTH))
    if stat.S_ISDIR(info.st_mode):
        executable = bool(mode & (stat.S_IXUSR if owner else stat.S_IXGRP if group else stat.S_IXOTH))
        readable = readable and executable
    return readable, writable


def _check_read_only_for_run_user(path: Path, *, uid: int, gid: int, label: str) -> None:
    if path.is_symlink():
        raise RuntimeContractError(f"{label} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(path.parent.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise RuntimeContractError(f"{label} path or parent is unresolved: {path}") from exc
    readable, writable = _identity_access(resolved, uid, gid)
    if not readable or writable:
        raise RuntimeContractError(f"{label} must be readable and not writable by run-as UID {uid}: {path}")
    parent = resolved.parent.lstat()
    try:
        account = pwd.getpwuid(uid)
        groups = set(os.getgrouplist(account.pw_name, gid))
    except (KeyError, OSError) as exc:
        raise RuntimeContractError(f"cannot resolve run-as identity {uid}:{gid}") from exc
    owner = parent.st_uid == uid
    group = parent.st_gid in groups
    mode = stat.S_IMODE(parent.st_mode)
    searchable = bool(mode & (stat.S_IXUSR if owner else stat.S_IXGRP if group else stat.S_IXOTH))
    writable = bool(mode & (stat.S_IWUSR if owner else stat.S_IWGRP if group else stat.S_IWOTH))
    if not searchable or writable:
        raise RuntimeContractError(f"{label} parent must be searchable and not writable by run-as UID {uid}: {resolved.parent}")


def _check_traversable_for_run_user(path: Path, *, uid: int, gid: int, label: str) -> None:
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISDIR(info.st_mode):
        raise RuntimeContractError(f"{label} must be a real directory: {path}")
    try:
        account = pwd.getpwuid(uid)
        groups = set(os.getgrouplist(account.pw_name, gid))
    except (KeyError, OSError) as exc:
        raise RuntimeContractError(f"cannot resolve run-as identity {uid}:{gid}") from exc
    owner = info.st_uid == uid
    group = info.st_gid in groups
    mode = stat.S_IMODE(info.st_mode)
    searchable = bool(mode & (stat.S_IXUSR if owner else stat.S_IXGRP if group else stat.S_IXOTH))
    writable = bool(mode & (stat.S_IWUSR if owner else stat.S_IWGRP if group else stat.S_IWOTH))
    if not searchable or writable:
        raise RuntimeContractError(f"{label} must be searchable and not writable by run-as UID {uid}: {path}")


def _check_sealed_input_ancestry(path: Path, *, uid: int, gid: int, label: str) -> None:
    """Prevent a same-UID rename from replacing an input after hash validation.

    Inputs must live below the root-owned /dev/shm staging area. Every
    directory from the object through its immediate parent must be
    non-writable by the workload identity. /dev/shm itself is the one
    permitted world-writable ancestor: its root-owned sticky bit prevents a
    lower-UID process from removing or renaming the root-owned sealed child.
    """
    launcher_uid = os.geteuid()
    if launcher_uid == 0 and uid == launcher_uid:
        raise RuntimeContractError("sealed inputs require a run-as identity distinct from the launcher")
    root = SEALED_TMPFS_ROOT.resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeContractError(f"{label} must be inside the sealed tmpfs input root: {path}") from exc
    anchor = root.lstat()
    if not stat.S_ISDIR(anchor.st_mode):
        raise RuntimeContractError("sealed input root must be a real directory")
    if launcher_uid == 0 and (anchor.st_uid != 0 or not (anchor.st_mode & stat.S_ISVTX)):
        raise RuntimeContractError("sealed tmpfs input root must be a root-owned sticky directory")
    expected_owner = 0 if launcher_uid == 0 else launcher_uid
    leaf_info = resolved.lstat()
    if leaf_info.st_uid != expected_owner:
        raise RuntimeContractError(f"{label} must be owned by the trusted launcher: {resolved}")
    readable, writable = _identity_access(resolved, uid, gid)
    if not readable or writable:
        raise RuntimeContractError(f"{label} is not immutable to the run-as identity: {resolved}")
    current = resolved.parent if resolved.is_file() else resolved
    while current != root:
        info = current.lstat()
        if info.st_uid != expected_owner:
            raise RuntimeContractError(f"{label} must be owned by the trusted launcher: {current}")
        _check_traversable_for_run_user(current, uid=uid, gid=gid, label=f"{label} ancestor")
        parent = current.parent
        if parent == current:
            raise RuntimeContractError(f"{label} ancestry did not reach sealed tmpfs input root")
        current = parent


def _check_exact_run_root(run_dir: Path, *, uid: int, gid: int) -> None:
    try:
        resolved = run_dir.resolve(strict=True)
        expected = EXPECTED_RUN_ROOT.resolve(strict=False)
    except OSError as exc:
        raise RuntimeContractError("Step119 run directory cannot be resolved") from exc
    if run_dir.is_symlink() or resolved != expected or not resolved.is_dir():
        raise RuntimeContractError(f"Step119 output must use its exact fresh run root: {EXPECTED_RUN_ROOT}")
    info = resolved.lstat()
    if info.st_uid != uid or info.st_gid != gid:
        raise RuntimeContractError("Step119 output root ownership differs from the declared run-as identity")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise RuntimeContractError("Step119 output root must be private to the declared run-as identity")


def _check_expected_new_run_root(run_dir: Path, *, uid: int, gid: int) -> None:
    try:
        expected = EXPECTED_RUN_ROOT.resolve(strict=False)
        candidate = run_dir.resolve(strict=False)
        parent = EXPECTED_RUN_ROOT.parent.resolve(strict=True)
    except OSError as exc:
        raise RuntimeContractError("Step119 output parent cannot be resolved") from exc
    if candidate != expected or run_dir.is_symlink():
        raise RuntimeContractError(f"Step119 output must use its exact fresh run root: {EXPECTED_RUN_ROOT}")
    if (
        parent != expected.parent
        or Path(os.path.abspath(EXPECTED_RUN_ROOT.parent)) != parent
        or not parent.is_dir()
        or EXPECTED_RUN_ROOT.parent.is_symlink()
    ):
        raise RuntimeContractError("Step119 output parent must be the exact real runs directory")
    parent_info = parent.lstat()
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or (parent_info.st_uid, parent_info.st_gid) != (uid, gid)
        or stat.S_IMODE(parent_info.st_mode) & 0o077
    ):
        raise RuntimeContractError("Step119 output parent must be private to the declared run-as identity")
    if run_dir.exists():
        raise RuntimeContractError("Step119 output root already exists; refusing to reuse it")


def _create_exact_run_root(run_dir: Path, *, uid: int, gid: int) -> None:
    """Atomically create the one frozen, fresh scientific output directory.

    ``oe run-campaign`` collision-checks this separate output root, while its
    launcher receipts live under the profile run ID. The host runtime owns
    creation of the scientific root after Formal Start and before task input
    is consumed. A race or stale directory fails closed.
    """
    parent_fd = -1
    child_fd = -1
    _check_expected_new_run_root(run_dir, uid=uid, gid=gid)
    expected = EXPECTED_RUN_ROOT.resolve(strict=False)
    parent = expected.parent
    try:
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        parent_info = os.fstat(parent_fd)
        current_parent = os.stat(parent, follow_symlinks=False)
        if (parent_info.st_dev, parent_info.st_ino) != (current_parent.st_dev, current_parent.st_ino):
            raise RuntimeContractError("Step119 output parent changed during launch")
        try:
            os.mkdir(expected.name, 0o700, dir_fd=parent_fd)
        except FileExistsError as exc:
            raise RuntimeContractError("Step119 output root appeared during launch; refusing to reuse it") from exc
        child_fd = os.open(
            expected.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        os.fchown(child_fd, uid, gid)
        os.fchmod(child_fd, 0o700)
        created = os.fstat(child_fd)
        named = os.stat(expected.name, dir_fd=parent_fd, follow_symlinks=False)
        current_parent = os.stat(parent, follow_symlinks=False)
        if (
            (created.st_dev, created.st_ino) != (named.st_dev, named.st_ino)
            or (parent_info.st_dev, parent_info.st_ino) != (current_parent.st_dev, current_parent.st_ino)
        ):
            raise RuntimeContractError("Step119 output directory identity changed during creation")
    except OSError as exc:
        raise RuntimeContractError("Step119 output root could not be created safely") from exc
    finally:
        if child_fd >= 0:
            os.close(child_fd)
        if parent_fd >= 0:
            os.close(parent_fd)
    _check_exact_run_root(run_dir, uid=uid, gid=gid)


def _prepare_run_root(
    run_dir: Path, *, uid: int, gid: int, model_load_preflight_only: bool
) -> None:
    if model_load_preflight_only:
        _check_preflight_run_dir(run_dir, uid=uid, gid=gid)
    else:
        _create_exact_run_root(run_dir, uid=uid, gid=gid)


def _validated_run_identity(uid: int, gid: int) -> pwd.struct_passwd:
    try:
        identity = pwd.getpwuid(uid)
    except KeyError as exc:
        raise RuntimeContractError(f"cannot resolve run-as identity {uid}:{gid}") from exc
    if identity.pw_gid != gid:
        raise RuntimeContractError("Step119 run-as GID differs from the UID's primary group")
    return identity


def _check_preflight_run_dir(run_dir: Path, *, uid: int, gid: int) -> None:
    expected_parent = EXPECTED_RUN_ROOT.parent.resolve(strict=True)
    try:
        resolved = run_dir.resolve(strict=True)
        resolved.relative_to(expected_parent)
    except (OSError, ValueError) as exc:
        raise RuntimeContractError("model-load preflight output must be under the Step119 run parent") from exc
    if run_dir.is_symlink() or resolved == EXPECTED_RUN_ROOT.resolve(strict=False):
        raise RuntimeContractError("model-load preflight requires a separate temporary output directory")
    if resolved.parent != expected_parent:
        raise RuntimeContractError("model-load preflight output must be a direct child of the Step119 run parent")
    if not resolved.name.startswith(".zero-formal-") or not resolved.is_dir() or any(resolved.iterdir()):
        raise RuntimeContractError("model-load preflight output must be a fresh .zero-formal-* directory")
    info = resolved.lstat()
    if (info.st_uid, info.st_gid) != (uid, gid) or stat.S_IMODE(info.st_mode) & 0o077:
        raise RuntimeContractError("model-load preflight output must be private to the declared run identity")


def _check_fresh_run_dir(run_dir: Path) -> None:
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise RuntimeContractError("Step119 run directory must be an existing real directory")
    # The launcher creates these two current-attempt receipts before invoking
    # this host runner. Every trainer result, metric, checkpoint, or other file
    # in the output root is evidence of reuse and must stop the one-pass run.
    allowed = {"host-runtime-import-identity.json", "launch-receipt.json"}
    unexpected = sorted(path.name for path in run_dir.iterdir() if path.name not in allowed)
    if unexpected:
        raise RuntimeContractError(
            "Step119 run directory is not fresh; unexpected prior output: "
            + ", ".join(unexpected[:8])
        )
    for name in allowed:
        path = run_dir / name
        if path.exists() and (path.is_symlink() or not path.is_file()):
            raise RuntimeContractError(f"Step119 launcher receipt must be a regular file: {path}")
        if path.exists():
            try:
                receipt = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise RuntimeContractError(f"Step119 launcher receipt is unreadable: {path}") from exc
            if not isinstance(receipt, dict) or receipt.get("run_id") != run_dir.name:
                raise RuntimeContractError(f"Step119 launcher receipt does not match this run root: {path}")


def validate_inputs(
    *, model_path: Path, parquet_path: Path, manifest_path: Path,
    source_root: Path, run_as_uid: int, run_as_gid: int,
    run_dir: Path,
    model_load_preflight_only: bool = False,
    allow_missing_formal_run_root: bool = False,
) -> None:
    if model_path.is_symlink() or not model_path.is_dir():
        raise RuntimeContractError("Step119 model path must be a directory")
    model_path = model_path.resolve(strict=True)
    _check_regular(model_path / "model.safetensors", MODEL_SHA256, "Step119 merged model")
    _check_regular(model_path / "config.json", CONFIG_SHA256, "Step119 merged model config")
    _check_regular(model_path / "generation_config.json", GENERATION_CONFIG_SHA256, "Step119 generation config")
    _check_regular(model_path / "tokenizer.json", TOKENIZER_SHA256, "Step119 tokenizer")
    _check_regular(model_path / "tokenizer_config.json", TOKENIZER_CONFIG_SHA256, "Step119 tokenizer config")
    _check_regular(model_path / "chat_template.jinja", CHAT_TEMPLATE_SHA256, "Step119 chat template")
    if source_root.is_symlink():
        raise RuntimeContractError("Step119 runtime source root must not be a symlink")
    source_root = source_root.resolve(strict=True)
    expected_manifest = source_root / MANIFEST_RELATIVE
    expected_parquet = source_root / PARQUET_RELATIVE
    if manifest_path.resolve(strict=True) != expected_manifest.resolve(strict=True):
        raise RuntimeContractError("Step119 manifest must be the exact manifest staged inside the immutable source root")
    if parquet_path.resolve(strict=True) != expected_parquet.resolve(strict=True):
        raise RuntimeContractError("Step119 validation carrier must be staged inside the immutable source root")
    _check_regular(parquet_path, PARQUET_SHA256, "Step119 public validation carrier")
    _check_regular(manifest_path, MANIFEST_SHA256, "Step119 public task manifest")
    adapter = source_root / CUSTOM_DATASET_RELATIVE
    _check_regular(adapter, CUSTOM_DATASET_SHA256, "Step119 custom validation dataset adapter")
    helper = source_root / MANIFEST_HELPER_RELATIVE
    _check_regular(helper, MANIFEST_HELPER_SHA256, "Step119 source-faithful manifest helper")
    for label, path in (
        ("model directory", model_path),
        ("merged model weights", model_path / "model.safetensors"),
        ("merged model config", model_path / "config.json"),
        ("merged generation config", model_path / "generation_config.json"),
        ("merged tokenizer", model_path / "tokenizer.json"),
        ("merged tokenizer config", model_path / "tokenizer_config.json"),
        ("merged chat template", model_path / "chat_template.jinja"),
        ("validation carrier", parquet_path),
        ("task manifest", manifest_path),
        ("validation source root", source_root),
        ("runtime dataset adapter", adapter),
        ("runtime manifest helper", helper),
    ):
        _check_read_only_for_run_user(path, uid=run_as_uid, gid=run_as_gid, label=label)
        _check_sealed_input_ancestry(path, uid=run_as_uid, gid=run_as_gid, label=label)
    if model_load_preflight_only:
        _check_preflight_run_dir(run_dir, uid=run_as_uid, gid=run_as_gid)
    elif allow_missing_formal_run_root:
        _check_expected_new_run_root(run_dir, uid=run_as_uid, gid=run_as_gid)
    else:
        _check_exact_run_root(run_dir, uid=run_as_uid, gid=run_as_gid)
    if not allow_missing_formal_run_root:
        _check_fresh_run_dir(run_dir)


def build_hydra_overrides(
    *, model_path: Path, parquet_path: Path, manifest_path: Path, source_root: Path, run_dir: Path
) -> list[str]:
    """Build the only allowed config: 128 public validation rows, no train file."""
    return [
        "algorithm.adv_estimator=gigpo",
        "algorithm.use_kl_in_reward=False",
        "data.train_files=q03://step119/non-readable-training-sentinel",
        f"data.val_files={parquet_path}",
        "data.train_batch_size=16",
        "data.val_batch_size=128",
        "data.max_prompt_length=4096",
        "data.max_response_length=512",
        "data.filter_overlong_prompts=True",
        "data.truncation=left",
        "data.return_raw_chat=True",
        "data.shuffle=False",
        "+data.apply_chat_template_kwargs.enable_thinking=False",
        f"data.custom_cls.path={source_root / CUSTOM_DATASET_RELATIVE}",
        "data.custom_cls.name=Q03Step119ValidationDataset",
        f"+data.q03_step119_validation_parquet={parquet_path}",
        f"+data.q03_step119_validation_parquet_sha256={PARQUET_SHA256}",
        f"+data.q03_step119_task_manifest={manifest_path}",
        f"+data.q03_step119_task_manifest_sha256={MANIFEST_SHA256}",
        "+data.q03_step119_validation_base_seed=1000",
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.use_remove_padding=False",
        "actor_rollout_ref.model.override_config._attn_implementation=sdpa",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=sync",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.6",
        "actor_rollout_ref.rollout.prompt_length=4096",
        "actor_rollout_ref.rollout.response_length=512",
        "actor_rollout_ref.rollout.max_model_len=null",
        "actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=TRITON_ATTN",
        "actor_rollout_ref.rollout.val_kwargs.top_k=-1",
        "actor_rollout_ref.rollout.val_kwargs.top_p=1.0",
        "actor_rollout_ref.rollout.val_kwargs.temperature=0.4",
        "actor_rollout_ref.rollout.val_kwargs.n=1",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
        "env.env_name=Webshop",
        "env.seed=0",
        "env.history_length=2",
        "env.max_steps=15",
        "env.rollout.n=1",
        "env.webshop.use_small=True",
        "env.webshop.human_goals=False",
        "algorithm.seed.enable_analysis=False",
        "algorithm.seed.save_analysis=False",
        "reward_model.enable=False",
        "trainer.critic_warmup=0",
        "trainer.logger=[console]",
        "trainer.project_name=agentic_webshop",
        "trainer.experiment_name=202610251359-q03-step119-val128",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.save_freq=-1",
        "trainer.resume_mode=disable",
        "trainer.test_freq=-1",
        "trainer.total_epochs=1",
        "trainer.val_before_train=True",
        "trainer.val_only=True",
        f"trainer.default_local_dir={run_dir}",
        f"trainer.rollout_data_dir={run_dir}",
    ]


def run_zero_task_model_load_preflight(
    *, native_run, native_module, trainer_class, environments_module, config, ray_module
) -> dict[str, object]:
    """Load the frozen actor/vLLM workers through native init_workers, without data or tasks."""
    from unittest.mock import patch

    class EmptyDataset:
        def __len__(self) -> int:
            return 0

        def __getitem__(self, index: int) -> object:
            raise RuntimeContractError("model-load preflight must never read dataset rows")

    empty_dataset = EmptyDataset()
    counts = {"dataset_factory_calls": 0, "dataset_item_reads": 0, "environment_calls": 0, "fit_calls": 0}

    def dataset_factory(*_args, **_kwargs):
        counts["dataset_factory_calls"] += 1
        return empty_dataset

    def no_environments(*_args, **_kwargs):
        counts["environment_calls"] += 1
        return None, None

    def no_fit(trainer):
        counts["fit_calls"] += 1
        if trainer.envs is not None or trainer.val_envs is not None:
            raise RuntimeContractError("model-load preflight unexpectedly created an environment")
        if trainer.actor_rollout_wg is None:
            raise RuntimeContractError("native init_workers returned without an actor/rollout worker group")
        if len(trainer.train_dataset) != 0 or len(trainer.val_dataset) != 0:
            raise RuntimeContractError("model-load preflight dataset factory was not isolated")

    ray_initialized_here = False
    try:
        if ray_module.is_initialized():
            raise RuntimeContractError("model-load preflight requires its own isolated Ray runtime")
        ray_init = config.get("ray_init", {})
        default_runtime_env = native_module.get_ppo_ray_runtime_env()
        runtime_env = native_module.OmegaConf.merge(
            default_runtime_env, ray_init.get("runtime_env", {})
        )
        ray_init_kwargs = native_module.OmegaConf.to_container(
            native_module.OmegaConf.create({**ray_init, "runtime_env": runtime_env}), resolve=True
        )
        ray_module.init(**ray_init_kwargs)
        ray_initialized_here = True
        with (
            patch.object(native_module, "create_rl_dataset", dataset_factory),
            patch.object(native_module, "create_rl_sampler", lambda *_args, **_kwargs: None),
            patch.object(environments_module, "make_envs", no_environments),
            patch.object(trainer_class, "fit", no_fit),
        ):
            native_run(object(), config)
    finally:
        if ray_initialized_here:
            ray_module.shutdown()

    if counts != {"dataset_factory_calls": 2, "dataset_item_reads": 0, "environment_calls": 1, "fit_calls": 1}:
        raise RuntimeContractError(f"model-load preflight isolation accounting mismatch: {counts}")
    return {
        "status": "MODEL_LOAD_PREFLIGHT_PASS",
        "native_init_workers_returned": True,
        "actor_rollout_model_initialized": True,
        "vllm_runtime_initialized": True,
        "formal_task_count": 0,
        "environment_factory_interceptions": counts["environment_calls"],
        "environment_creation_calls": 0,
        "dataset_factory_calls": counts["dataset_factory_calls"],
        "dataset_item_reads": counts["dataset_item_reads"],
        "generation_calls": 0,
        "optimizer_updates": 0,
        "fit_calls": counts["fit_calls"],
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-as-uid", type=int, required=True)
    parser.add_argument("--run-as-gid", type=int, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--parquet-path", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-load-preflight-only", action="store_true")
    parser.add_argument("--model-load-preflight-context")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.run_as_uid <= 0 or args.run_as_gid <= 0:
        raise RuntimeContractError("Step119 workload must run under a non-root UID/GID")
    if args.model_load_preflight_only:
        if os.geteuid() != 0 and (os.geteuid(), os.getegid()) != (args.run_as_uid, args.run_as_gid):
            raise RuntimeContractError("model-load preflight must run as root control plane or declared UID/GID")
    elif os.geteuid() != 0:
        raise RuntimeContractError("Step119 formal host runner requires the guarded root control plane")
    identity = _validated_run_identity(args.run_as_uid, args.run_as_gid)
    preflight_context = None
    if args.model_load_preflight_only:
        preflight_context = validate_model_load_preflight_authorization(
            args.model_load_preflight_context,
            run_dir=args.run_dir,
        )
    elif args.model_load_preflight_context is not None:
        raise RuntimeContractError("launcher authorization context is only valid for model-load preflight")
    validate_inputs(
        model_path=args.model_path,
        parquet_path=args.parquet_path,
        manifest_path=args.manifest_path,
        source_root=args.source_root,
        run_as_uid=args.run_as_uid,
        run_as_gid=args.run_as_gid,
        run_dir=args.run_dir,
        model_load_preflight_only=args.model_load_preflight_only,
        allow_missing_formal_run_root=not args.model_load_preflight_only,
    )
    _prepare_run_root(
        args.run_dir,
        uid=args.run_as_uid,
        gid=args.run_as_gid,
        model_load_preflight_only=args.model_load_preflight_only,
    )
    if not args.model_load_preflight_only:
        _check_exact_run_root(args.run_dir, uid=args.run_as_uid, gid=args.run_as_gid)
        _check_fresh_run_dir(args.run_dir)
    overrides = build_hydra_overrides(
        model_path=args.model_path.resolve(strict=True),
        parquet_path=args.parquet_path.resolve(strict=True),
        manifest_path=args.manifest_path.resolve(strict=True),
        source_root=args.source_root.resolve(strict=True),
        run_dir=args.run_dir.resolve(strict=True),
    )
    if os.geteuid() == 0:
        os.initgroups(identity.pw_name, args.run_as_gid)
        os.setgid(args.run_as_gid)
        os.setuid(args.run_as_uid)
        if os.geteuid() != args.run_as_uid or os.getegid() != args.run_as_gid:
            raise RuntimeContractError("Step119 runtime did not drop to the authorized UID/GID")
    if args.model_load_preflight_only:
        from hydra import compose, initialize_config_dir
        import ray
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        import agent_system.environments as environments

        native_module = importlib.import_module("verl.trainer.main_ppo")
        config_dir = Path(native_module.__file__).resolve().parent / "config"
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            config = compose(config_name="ppo_trainer", overrides=overrides)
        native_run = native_module.TaskRunner.__ray_metadata__.modified_class.run
        receipt = run_zero_task_model_load_preflight(
            native_run=native_run,
            native_module=native_module,
            trainer_class=RayPPOTrainer,
            environments_module=environments,
            config=config,
            ray_module=ray,
        )
        receipt["preflight_authorization"] = {
            "context_sha256": preflight_context["context_sha256"],
            "campaign_id": preflight_context["campaign_id"],
            "execution_sha": preflight_context["execution_sha"],
            "runtime_profile_sha256": preflight_context["runtime_profile_sha256"],
            "selected_gpu_uuids": preflight_context["selected_gpu_uuids"],
            "effective_source_sha256": preflight_context["effective_source_sha256"],
        }
        print(json.dumps(receipt, sort_keys=True))
        return 0

    sys.argv = ["verl.trainer.main_ppo", *overrides]
    runpy.run_module("verl.trainer.main_ppo", run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
