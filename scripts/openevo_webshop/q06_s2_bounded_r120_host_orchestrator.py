#!/usr/bin/env python3
"""Hash-bound Q06 S2 Bounded-R120 cell adapter for canonical Formal Start.

This module is deliberately separate from the consumed S1 GDR adapter. It
accepts one prospective S2 contract, one endpoint and one fixed 64-task pass.
All authority is supplied by the canonical run-campaign release/router chain;
the module refuses to launch while the prospective contract remains blocked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

CONTRACT_SCHEMA = "openevo.q06.s2-prospective-cell-contract.v1"
CONTRACT_PREFIX = "configs/experiment/advisor-questions/q06-s2-bounded-r120-prospective-contract."
CELL_ID = "bounded-r120"
CHECKPOINT_ID = "after_120_rounds"
PROBE_PATH = "scripts/openevo_webshop/q17_fixed_checkpoint_probe.py"
IMAGE_ID = "sha256:d0f8796010899ae41a7af8ae3f44f47b96dea73b297b5af124049b3ae88e2850"
GPU_RE = re.compile(r"^GPU-[0-9a-f-]{36}$")
MAX_WALL_SECONDS = 2 * 60 * 60


class GateError(RuntimeError):
    """A fail-closed Q06 S2 identity or launch-gate rejection."""


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def same_sha256_digest(actual: object, declared: object) -> bool:
    """Compare canonical bytes across bare and sha256:-prefixed digest forms."""
    if not isinstance(actual, str) or not isinstance(declared, str):
        return False
    actual_digest = actual.removeprefix("sha256:")
    declared_digest = declared.removeprefix("sha256:")
    return (re.fullmatch(r"[0-9a-f]{64}", actual_digest) is not None
            and re.fullmatch(r"[0-9a-f]{64}", declared_digest) is not None
            and actual_digest == declared_digest)


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"unreadable authority JSON: {path}: {exc}") from exc
    require(isinstance(value, dict), f"authority JSON must be an object: {path}")
    return value


def repo_file(repo: Path, ref: object, label: str) -> tuple[Path, str]:
    repo = repo.resolve()
    require(isinstance(ref, dict), f"{label} must be a path+sha256 reference")
    path_value, digest = ref.get("path"), ref.get("sha256")
    require(isinstance(path_value, str) and isinstance(digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
            f"{label} reference is malformed")
    path = (repo / path_value).resolve()
    require(path != repo and repo in path.parents, f"{label} escapes authority repository")
    require(path.is_file() and not path.is_symlink(), f"{label} is missing or symlinked: {path}")
    actual = sha256_file(path)
    require(actual == digest, f"{label} byte hash mismatch: {path}")
    return path, actual


def load_contract(repo: Path, ref: object) -> tuple[Path, dict[str, Any], str]:
    repo = repo.resolve()
    path, digest = repo_file(repo, ref, "prospective contract")
    rel = path.relative_to(repo).as_posix()
    require(rel.startswith(CONTRACT_PREFIX) and rel.endswith(".json"),
            "release does not reference the Q06 S2 Bounded-R120 contract family")
    contract = read_object(path)
    require(contract.get("schema") == CONTRACT_SCHEMA
            and contract.get("question_id") == "Q06_TOKEN_CLICK_ENTROPY"
            and contract.get("stage") == "S2"
            and contract.get("cell", {}).get("cell_id") == CELL_ID
            and contract.get("cell", {}).get("mechanism") == "Bounded State"
            and contract.get("cell", {}).get("checkpoint_id") == CHECKPOINT_ID,
            "prospective contract is not the exact single Bounded-R120 cell")
    require(contract.get("append_only") is True,
            "prospective contract must remain append-only")
    require(contract.get("status") == "LAUNCH_AUTHORIZED",
            "prospective contract is not launch-authorized")
    constraints = contract.get("execution_constraints", {})
    require(constraints == {
        "training_rollouts": 0,
        "optimizer_updates": 0,
        "teacher_calls": 0,
        "adaptive_reruns": 0,
        "retry_on_scientific_outcome": False,
        "episodes": 64,
        "passes": 1,
        "max_generation_calls": 960,
        "max_generated_tokens": 491520,
        "max_scorer_forward_calls": 100000,
        "max_scorer_input_tokens": 1000000,
        "max_gpu_hours": 2,
        "max_persistent_output_bytes": 1073741824,
        "final_panel_access_allowed": False,
        "formal_task_consumption_allowed": True,
        "gpu_allocation_allowed": True,
    }, "contract launch constraints differ from the single frozen 64-task cell")
    gates = contract.get("gates", {})
    cost_source_path, _ = repo_file(repo, gates.get("scorer_cost_source"), "Q06 scorer cost source")
    cost_source = read_object(cost_source_path)
    largest_forward_calls = max(
        sum(int(group["estimated_forward_calls"]) for group in arm["groups"].values())
        for arm in cost_source.get("arms", {}).values()
    )
    largest_input_tokens = max(
        sum(int(group["estimated_input_tokens"]) for group in arm["groups"].values())
        for arm in cost_source.get("arms", {}).values()
    )
    declared_cost_source = gates["scorer_cost_source"]
    require(cost_source.get("classification") == "derived_cpu_only_not_observed_forward_calls"
            and largest_forward_calls == declared_cost_source.get("largest_single_arm_forward_calls")
            and largest_input_tokens == declared_cost_source.get("largest_single_arm_input_tokens")
            and constraints["max_scorer_forward_calls"] >= 2 * largest_forward_calls
            and constraints["max_scorer_input_tokens"] >= 2 * largest_input_tokens,
            "scorer ceilings are not at least 2x the verified derived single-arm source costs")
    require(gates.get("checkpoint_identity", {}).get("status") == "PASS_READ_ONLY_IDENTITY"
            and gates.get("scoring_cost_ceiling", {}).get("status") == "FROZEN"
            and gates.get("resource_qualification", {}).get("status") == "PASS"
            and gates.get("task_bearing_activation", {}).get("status") == "PASS",
            "prospective contract has unresolved checkpoint, cost, resource or activation gates")
    return path, contract, digest


def validate_release_router(repo: Path, release_path: Path, router_path: Path) -> tuple[dict, dict]:
    repo = repo.resolve()
    release_path, router_path = release_path.resolve(), router_path.resolve()
    require(repo in release_path.parents and repo in router_path.parents,
            "release/router must be contained in the authority repository")
    require(release_path.is_file() and not release_path.is_symlink()
            and router_path.is_file() and not router_path.is_symlink(),
            "release/router missing or symlinked")
    release, router = read_object(release_path), read_object(router_path)
    execution_source = release.get("execution_source")
    require(isinstance(execution_source, dict)
            and isinstance(execution_source.get("path"), str)
            and isinstance(execution_source.get("repository"), str)
            and re.fullmatch(r"[0-9a-f]{40}", str(execution_source.get("revision", ""))) is not None,
            "S2 release must pin execution-source repository, path and exact 40-character revision")
    require(release.get("schema") == "openevo.experiment-q06-s2-bounded-r120-release.v1"
            and release.get("status") == "launch-authorized"
            and release.get("formal_task_consumption_allowed") is True
            and release.get("formal_rollout_limit") == 64
            and release.get("final_panel_access_limit") == 0,
            "S2 release status, endpoint budget or final-panel lock drift")
    require(router.get("schema") == "openevo.q06.s2-bounded-r120-current-campaign.v1"
            and router.get("status") == "launch-authorized"
            and router.get("formal_task_consumption_allowed") is True
            and router.get("formal_rollout_limit") == 64
            and router.get("final_panel_access_allowed") is False,
            "S2 current router status, endpoint budget or final-panel lock drift")
    release_sha = sha256_file(release_path)
    require(router.get("release_receipt_sha256") == release_sha,
            "S2 router does not bind exact release bytes")
    require(router.get("campaign_id") == release.get("campaign_id")
            and router.get("experiment_id") == release.get("experiment_id")
            and router.get("endpoint") == CHECKPOINT_ID
            and release.get("endpoint") == CHECKPOINT_ID,
            "S2 router/release campaign or endpoint identity drift")
    activation_ref = release.get("canonical_activation")
    require(isinstance(activation_ref, dict)
            and activation_ref.get("path") == router_path.relative_to(repo).as_posix(),
            "S2 release does not name the exact canonical activation/router")
    return release, router


def validate_manifest(repo: Path, contract: dict, release: dict) -> tuple[Path, dict]:
    repo = repo.resolve()
    manifest_path, _ = repo_file(repo, release.get("source_manifest"), "S2 probe manifest")
    manifest = read_object(manifest_path)
    adapter = contract["known_shared_identity"]["adapter"]
    cell = contract["cell"]
    constraints = contract["execution_constraints"]
    require(manifest.get("schema") == "openevo.q06.s2-bounded-r120-probe-manifest.v1"
            and manifest.get("cell_id") == CELL_ID
            and manifest.get("checkpoint_id") == CHECKPOINT_ID
            and manifest.get("runtime_execution_sha") == release.get("execution_source", {}).get("revision"),
            "S2 probe manifest is not the exact Bounded-R120 manifest")
    require(manifest.get("contract") == release.get("prospective_contract")
            and manifest.get("checkpoint_id") == cell["checkpoint_id"]
            and manifest.get("completed_stage2_rounds") == cell["completed_stage2_rounds"]
            and manifest.get("actual_parameter_update_events") == cell["actual_parameter_update_events"]
            and manifest.get("source_summary_sha256") == cell["source_summary_sha256"],
            "S2 probe manifest/source checkpoint binding drift")
    require(manifest.get("checkpoints") == {
        CHECKPOINT_ID: {
            "adapter_sha256": adapter["adapter_sha256"],
            "state_id": manifest.get("checkpoints", {}).get(CHECKPOINT_ID, {}).get("state_id"),
        }
    } and isinstance(manifest["checkpoints"][CHECKPOINT_ID]["state_id"], str)
            and manifest["checkpoints"][CHECKPOINT_ID]["state_id"],
            "probe manifest does not bind the exact Bounded-R120 adapter/state")
    require(manifest.get("panel_sha256") == cell["evaluation_panel_sha256"]
            and manifest.get("panel_size") == 64
            and manifest.get("final_panel_access_allowed") is False
            and manifest.get("budget") == {
                "new_episodes": 64, "fixed_passes": 1, "adaptive_reruns": 0,
                "external_teacher_calls": 0, "training_rollouts": 0,
                "optimizer_updates": 0, "final_panel_access": 0,
                "max_generation_calls": constraints["max_generation_calls"],
                "max_generated_tokens": constraints["max_generated_tokens"],
                "max_scorer_forward_calls": constraints["max_scorer_forward_calls"],
                "max_scorer_input_tokens": constraints["max_scorer_input_tokens"],
            }, "S2 panel or one-pass budget drift")
    known = contract["known_shared_identity"]
    carriers = manifest.get("common_carriers", {})
    require(manifest.get("model", {}).get("base_model") == known["base_model"]
            and manifest.get("model", {}).get("revision") == known["base_model_revision"]
            and carriers.get("agent_system_sha256") == known["common_agent_system_sha256"]
            and carriers.get("skill_sha256") == known["common_skill_sha256"]
            and carriers.get("memory_sha256") == known["common_memory_sha256"]
            and manifest.get("sampling") == known["sampling"],
            "S2 manifest model, common carriers or sampling drift")
    summary_path = Path(str(cell.get("source_summary_path", "")))
    source_summary, adapter_dir, adapter_file = resolve_source_summary_adapter(contract)
    summary_checkpoint = source_summary["checkpoint"]
    require(summary_checkpoint.get("state_id")
            == manifest["checkpoints"][CHECKPOINT_ID].get("state_id"),
            "Q06 source summary and manifest checkpoint state differ")
    require(manifest.get("source_summary_path") == str(summary_path)
            and manifest.get("source_summary_sha256") == cell.get("source_summary_sha256")
            and manifest.get("endpoint", {}).get("adapter_path") == str(adapter_dir)
            and manifest.get("endpoint", {}).get("checkpoint_id") == CHECKPOINT_ID,
            "S2 manifest adapter directory/endpoint mismatch")
    require(release.get("source_manifest", {}).get("path") == manifest_path.relative_to(repo).as_posix(),
            "S2 release manifest path is not repository-relative and exact")
    return manifest_path, manifest


def resolve_source_summary_adapter(contract: dict) -> tuple[dict, Path, Path]:
    """Resolve the R120 adapter from its hash-bound result summary, not a guessed run root."""
    cell = contract.get("cell", {})
    adapter = contract.get("known_shared_identity", {}).get("adapter", {})
    summary_path_value = cell.get("source_summary_path")
    require(isinstance(summary_path_value, str) and Path(summary_path_value).is_absolute(),
            "contract source summary path must be absolute")
    summary_path = Path(summary_path_value)
    require(summary_path.is_file() and not summary_path.is_symlink()
            and sha256_file(summary_path) == cell.get("source_summary_sha256"),
            "Q06 source summary is missing or differs from the frozen checkpoint identity")
    source_summary = read_object(summary_path)
    checkpoint = source_summary.get("checkpoint", {})
    require(source_summary.get("checkpoint_id") == CHECKPOINT_ID
            and checkpoint.get("actual_parameter_update_events")
            == cell.get("actual_parameter_update_events")
            and checkpoint.get("adapter_sha256") == adapter.get("adapter_sha256"),
            "Q06 source summary does not bind the frozen R120 checkpoint")
    adapter_dir = Path(str(checkpoint.get("adapter_path", "")))
    adapter_file = adapter_dir / Path(str(adapter.get("relative_file", "adapter_model.safetensors"))).name
    require(adapter_dir.is_absolute() and adapter_file.is_file() and not adapter_file.is_symlink()
            and adapter_file.stat().st_size == adapter.get("size_bytes")
            and sha256_file(adapter_file) == adapter.get("adapter_sha256"),
            "source-summary adapter payload is missing or differs from the frozen digest")
    return source_summary, adapter_dir, adapter_file


def validate_profile_and_contract_binding(repo: Path, release_path: Path, router_path: Path,
                                          release: dict, router: dict,
                                          contract_path: Path, contract: dict,
                                          contract_sha: str) -> tuple[Path, dict]:
    repo = repo.resolve()
    profile_path, _ = repo_file(repo, release.get("execution_readiness", {}).get("profile"), "S2 runtime profile")
    contract_ref = release.get("prospective_contract")
    require(isinstance(contract_ref, dict)
            and contract_ref.get("path") == contract_path.relative_to(repo).as_posix()
            and contract_ref.get("sha256") == contract_sha
            and router.get("prospective_contract") == contract_ref,
            "S2 release/router do not bind the exact prospective contract bytes")
    activation = release.get("canonical_activation", {})
    require(contract.get("gates", {}).get("task_bearing_activation", {}).get("path")
            == activation.get("path"),
            "prospective contract and release disagree on canonical activation identity")
    adapter_ref = release.get("host_orchestrator")
    adapter_path, _ = repo_file(repo, adapter_ref, "S2 host orchestrator")
    guard_ref = release.get("probe_guard")
    guard_path, _ = repo_file(repo, guard_ref, "post-model-load probe guard")
    profile = read_object(profile_path)
    require(profile.get("execution", {}).get("mode") == "host-orchestrator"
            and profile.get("s2_prospective_contract") == contract_ref
            and profile.get("host_orchestrator") == adapter_ref
            and profile.get("probe_guard") == guard_ref
            and profile.get("execution_source") == release.get("execution_source"),
            "S2 profile does not bind the contract, guard, orchestrator and execution source")
    command = profile.get("command")
    require(isinstance(command, list) and len(command) > 0
            and str(adapter_path) in command,
            f"S2 profile command does not invoke hash-bound host adapter {adapter_path}: {command}")
    for flag, expected in (("--authority-repo", str(repo)),
                           ("--release", str(release_path.resolve())),
                           ("--router", str(router_path.resolve())),
                           ("--execution-source", str(Path(release["execution_source"]["path"]).resolve()))):
        require(flag in command and command[command.index(flag) + 1] == expected,
                f"S2 profile command does not bind {flag} to its exact authority path")
    return profile_path, profile


def verify_execution_source(source: Path, source_ref: object,
                            run=subprocess.run) -> str:
    """Require the exact clean execution checkout named by the release."""
    require(isinstance(source_ref, dict), "release lacks pinned execution-source identity")
    expected_path = source_ref.get("path")
    expected_revision = source_ref.get("revision")
    expected_repository = source_ref.get("repository")
    source = source.resolve()
    require(isinstance(expected_path, str) and Path(expected_path).resolve() == source
            and isinstance(expected_revision, str)
            and re.fullmatch(r"[0-9a-f]{40}", expected_revision) is not None
            and isinstance(expected_repository, str) and expected_repository,
            "execution-source path/repository/revision is not precisely pinned")
    root = run(["git", "-C", str(source), "rev-parse", "--show-toplevel"],
               check=True, capture_output=True, text=True).stdout.strip()
    remote = run(["git", "-C", str(source), "remote", "get-url", "origin"],
                 check=True, capture_output=True, text=True).stdout.strip()
    remote_matches = remote == expected_repository
    if not remote_matches:
        # Some approved server checkouts use a named SSH host alias and a
        # Git url.insteadOf rule for the same canonical repository URL.
        rewrites = run(["git", "-C", str(source), "config", "--get-regexp",
                        r"^url\..*\.insteadof$"],
                       check=False, capture_output=True, text=True).stdout.splitlines()
        for entry in rewrites:
            key, _, prefix = entry.partition(" ")
            if not key.startswith("url.") or not key.endswith(".insteadof"):
                continue
            target = key[len("url."):-len(".insteadof")]
            if expected_repository.startswith(prefix) and (
                    target + expected_repository[len(prefix):]) == remote:
                remote_matches = True
                break
    head = run(["git", "-C", str(source), "rev-parse", "HEAD"],
               check=True, capture_output=True, text=True).stdout.strip()
    dirty = run(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
                check=True, capture_output=True, text=True).stdout.strip()
    require(Path(root).resolve() == source and remote_matches
            and head == expected_revision and not dirty,
            "execution-source checkout is not the exact clean pinned repository/ref")
    return head


def uid_write_rename_gate(output: Path, *, uid: int = 1001,
                          geteuid=os.geteuid) -> None:
    """Prove post-load UID 1001 can write and atomically rename in output."""
    require(geteuid() == uid, "post-model-load output probe must run as UID 1001")
    require(output.is_dir() and not output.is_symlink(), "post-load output directory missing or symlinked")
    output = output.resolve()
    nonce = uuid.uuid4().hex
    temporary = output / f".q06-s2-uid-gate-{nonce}.tmp"
    renamed = output / f".q06-s2-uid-gate-{nonce}.renamed"
    try:
        with temporary.open("xb") as stream:
            stream.write(nonce.encode("ascii"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, renamed)
        require(renamed.read_bytes() == nonce.encode("ascii"),
                "post-model-load write/rename readback mismatch")
    finally:
        temporary.unlink(missing_ok=True)
        renamed.unlink(missing_ok=True)


def output_tree_bytes(output: Path) -> int:
    total = 0
    for parent, dirs, files in os.walk(output, followlinks=False):
        for name in dirs + files:
            path = Path(parent) / name
            require(not path.is_symlink(), f"S2 output contains a symlink: {path}")
            if path.is_file():
                total += path.stat().st_size
    return total


def cleanup_owned_container(name: str, owner_token: str, *, run=subprocess.run) -> None:
    """Remove only a uniquely named container carrying this launch's owner label."""
    label = "com.openevo.q06.s2.owner"
    inspect = run(["docker", "inspect", "--format", f"{{{{index .Config.Labels \"{label}\"}}}}", name],
                  check=False, capture_output=True, text=True)
    if inspect.returncode != 0:
        return
    require(inspect.stdout.strip() == owner_token,
            "refusing cleanup: container does not carry this exact S2 owner token")
    run(["docker", "rm", "--force", name], check=True, capture_output=True, text=True)


def run_with_limits(command: list[str], output: Path, *, max_wall_seconds: int = MAX_WALL_SECONDS,
                    max_output_bytes: int = 1073741824, popen=subprocess.Popen,
                    run=subprocess.run, monotonic=time.monotonic, sleep=time.sleep,
                    owner_token: str | None = None) -> int:
    """Run one named container with wall/output caps and exact process/container cleanup."""
    owner_token = owner_token or uuid.uuid4().hex
    name = "q06-s2-bounded-r120-" + owner_token[:20]
    label = "com.openevo.q06.s2.owner=" + owner_token
    owned_command = [*command[:2], "--name", name, "--label", label, *command[2:]]
    require(output_tree_bytes(output) <= max_output_bytes,
            "S2 persistent output-byte ceiling exceeded before probe launch")
    started = monotonic()
    proc = popen(owned_command, start_new_session=True)
    try:
        while proc.poll() is None:
            elapsed = monotonic() - started
            if elapsed >= max_wall_seconds:
                raise GateError("S2 hard 2 GPU-hour wall cap exceeded")
            if output_tree_bytes(output) > max_output_bytes:
                raise GateError("S2 persistent output-byte ceiling exceeded")
            sleep(min(0.5, max_wall_seconds - elapsed))
        if output_tree_bytes(output) > max_output_bytes:
            raise GateError("S2 persistent output-byte ceiling exceeded")
        return int(proc.returncode)
    except GateError:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
        cleanup_owned_container(name, owner_token, run=run)
        raise


def validate_adapter_payload(contract: dict, manifest: dict) -> Path:
    adapter = contract["known_shared_identity"]["adapter"]
    ref = manifest.get("endpoint", {}).get("source_files", {}).get("adapter")
    adapter_dir = Path(str(manifest.get("endpoint", {}).get("adapter_path", "")))
    expected_path = adapter_dir / Path(str(adapter.get("relative_file", "adapter_model.safetensors"))).name
    require(isinstance(ref, dict)
            and Path(str(ref.get("path", ""))).resolve() == expected_path.resolve()
            and ref.get("bytes") == adapter["size_bytes"]
            and ref.get("sha256") == adapter["adapter_sha256"],
            "contract and manifest do not bind identical adapter payload bytes")
    path = Path(ref["path"])
    require(path.is_file() and not path.is_symlink()
            and path.stat().st_size == adapter["size_bytes"]
            and sha256_file(path) == adapter["adapter_sha256"],
            "Bounded-R120 adapter payload is missing or has different bytes")
    return path


def validate_formal_start(repo: Path, release_path: Path, router_path: Path,
                          profile_path: Path, output: Path, gpu_uuid: str,
                          receipt_path: Path) -> tuple[dict, dict]:
    require(GPU_RE.fullmatch(gpu_uuid) is not None, "invalid selected GPU UUID")
    require(receipt_path.is_file() and not receipt_path.is_symlink(),
            "canonical oe Formal Start receipt is missing")
    from scripts.openevo_webshop import prelaunch_zero_formal

    start = read_object(receipt_path)
    require(start.get("schema") == prelaunch_zero_formal.FORMAL_START_SCHEMA
            and start.get("receipt_sha256") == prelaunch_zero_formal.canonical_sha256(
                {key: value for key, value in start.items() if key != "receipt_sha256"})
            and start.get("status") == "FORMAL_START_COMMITTED"
            and start.get("formal_start_committed") is True
            and start.get("formal_task_consumption_may_begin") is True,
            "canonical Formal Start receipt is not committed and exact")
    require(start.get("selected_gpu_uuids") == [gpu_uuid]
            and os.getenv("OPENEVO_ASSIGNED_GPU_UUIDS") == gpu_uuid
            and os.getenv("OPENEVO_AUTHORIZED_GPU_UUIDS") == gpu_uuid,
            "selected GPU is not bound by canonical Formal Start")
    release, router = validate_release_router(repo, release_path, router_path)
    require(gpu_uuid in router.get("authorized_gpu_uuids", []),
            "selected GPU is outside the S2 canonical router authority")
    require(start.get("campaign_id") == release.get("campaign_id")
            and start.get("image_id") == IMAGE_ID,
            "Formal Start campaign/image differs from S2 release")
    readiness = start.get("readiness", {})
    profile_ref = release["execution_readiness"]["profile"]
    require(readiness.get("profile", {}).get("path") == str(profile_path.resolve())
            and same_sha256_digest(readiness.get("profile", {}).get("sha256"),
                                   profile_ref["sha256"]),
            "Formal Start profile is not the exact S2 READY profile")
    ready_ref = readiness.get("ready_receipt", {})
    ready_path = Path(str(ready_ref.get("path", ""))).resolve()
    require(ready_path.is_file() and not ready_path.is_symlink()
            and same_sha256_digest(sha256_file(ready_path), ready_ref.get("sha256")),
            "canonical READY receipt is missing or hash-mismatched")
    ready = read_object(ready_path)
    ready_release = ready.get("release_receipt", {})
    expected_release_sha = sha256_file(release_path)
    ready_profile = ready.get("profile", {})
    readiness_profile = readiness.get("profile", {})
    require(ready.get("verdict") == "READY"
            and ready_release.get("path") == str(release_path.resolve())
            and same_sha256_digest(ready_release.get("sha256"), expected_release_sha)
            and ready_profile.get("path") == readiness_profile.get("path")
            and same_sha256_digest(ready_profile.get("sha256"), readiness_profile.get("sha256")),
            "READY receipt does not bind this release/profile")
    pre_ref = start.get("prelaunch_receipt", {})
    pre_path = Path(str(pre_ref.get("path", ""))).resolve()
    require(pre_path.is_file() and not pre_path.is_symlink()
            and same_sha256_digest(sha256_file(pre_path), pre_ref.get("sha256")),
            "canonical prelaunch receipt is missing or hash-mismatched")
    prelaunch_zero_formal.validate_prelaunch_receipt(
        read_object(pre_path), expected_gpu_uuids=[gpu_uuid])
    return release, start


def build_probe_command(manifest: dict, source: Path, output: Path, gpu_uuid: str) -> list[str]:
    endpoint = manifest["endpoint"]
    checkpoint = manifest["checkpoints"][CHECKPOINT_ID]
    common = manifest["common_carriers"]
    model = manifest["model"]
    seed_root = Path(manifest["seed_root"]["path"])
    authority_repo = Path(manifest["authority_repo"])
    mounts = [authority_repo, source, seed_root, Path(model["local_model_path"]), Path(endpoint["adapter_path"]),
              Path(common["memory_path"]), Path(common["skill_path"]), Path(common["agent_system_path"])]
    command = ["docker", "run", "--rm", "--network", "none", "--user", "1001:1001",
               "--gpus", "device=" + gpu_uuid]
    for path in dict.fromkeys(mounts):
        require(path.exists() and not path.is_symlink(), f"S2 read-only mount missing or symlinked: {path}")
        command += ["--mount", f"type=bind,src={path},dst={path},readonly"]
    command += ["--mount", f"type=bind,src={output},dst={output}", "--tmpfs", "/tmp:rw,exec,nosuid,size=1g,mode=1777"]
    for value in ["HOME=/tmp", "TMPDIR=/tmp", "HF_HOME=/tmp/hf", "XDG_CACHE_HOME=/tmp/cache",
                  "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_openevo", "TRITON_CACHE_DIR=/tmp/triton",
                  "PYTHONDONTWRITEBYTECODE=1", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1",
                  "CUDA_VISIBLE_DEVICES=" + gpu_uuid,
                  f"PYTHONPATH={authority_repo}:{source}:{source}/vendor/openevo-webshop/src"]:
        command += ["--env", value]
    command += ["--entrypoint", "python3", IMAGE_ID,
                manifest["probe_guard_path"],
                "--authority-repo", str(authority_repo),
                "--probe-script", str(source / PROBE_PATH),
                "--output-dir", str(output),
                "--max-output-bytes", str(manifest["max_output_bytes"]),
                "--max-scorer-forward-calls", str(manifest["budget"]["max_scorer_forward_calls"]),
                "--max-scorer-input-tokens", str(manifest["budget"]["max_scorer_input_tokens"]), "--",
                "--manifest", str(manifest["manifest_path"]),
                "--panel-manifest", str(source / manifest["task_panel_manifest_path"]),
                "--checkpoint-id", CHECKPOINT_ID,
                "--runtime-source-root", str(source),
                "--seed-root", str(seed_root),
                "--model", str(model["local_model_path"]),
                "--revision", model["revision"],
                "--sd-adapter", endpoint["adapter_path"],
                "--memory", common["memory_path"], "--skill", common["skill_path"],
                "--agent", common["agent_system_path"], "--gpu-uuid", gpu_uuid,
                "--output-dir", str(output), "--instrumentation", "on"]
    require(checkpoint["adapter_sha256"] == manifest["endpoint"]["source_files"]["adapter"]["sha256"],
            "q17 probe command endpoint is not the exact Bounded-R120 adapter")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-repo", type=Path, required=True)
    parser.add_argument("--execution-source", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    repo, source, output = args.authority_repo.resolve(), args.execution_source.resolve(), args.output_dir.absolute()
    require(not output.exists() and not output.is_symlink(), "S2 one-pass output already exists")
    require(output.parent.is_dir() and not output.parent.is_symlink(), "S2 output parent must be prepared")
    require(not subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
                               check=True, capture_output=True, text=True).stdout.strip(),
            "S2 authority checkout must be clean")

    release, router = validate_release_router(repo, args.release, args.router)
    contract_path, contract, contract_sha = load_contract(repo, release.get("prospective_contract"))
    manifest_path, manifest = validate_manifest(repo, contract, release)
    manifest["manifest_path"] = str(manifest_path)
    profile_path, profile = validate_profile_and_contract_binding(
        repo, args.release, args.router, release, router,
        contract_path, contract, contract_sha)
    validate_adapter_payload(contract, manifest)
    guard_path, _ = repo_file(repo, release["probe_guard"], "post-model-load probe guard")
    manifest["probe_guard_path"] = str(guard_path)
    receipt = Path(os.environ.get("OPENEVO_FORMAL_START_RECEIPT", "")).resolve()
    release, start = validate_formal_start(
        repo, args.release, args.router, profile_path, output, args.gpu_uuid, receipt)
    require(start.get("image_id") == IMAGE_ID, "qualified image differs from frozen Q06 runtime image")
    from scripts import oe_impl
    profile["_formal_start_context"] = start["readiness"]
    expected_command = oe_impl.host_orchestrator_identity_command(
        profile, [args.gpu_uuid], Path(start["output_root"]), "disabled", receipt)
    require(start.get("command_identity") == oe_impl._command_identity(expected_command, IMAGE_ID),
            "Formal Start command identity differs from the S2 profile command")

    manifest["authority_repo"] = str(repo)
    manifest["max_output_bytes"] = contract["execution_constraints"]["max_persistent_output_bytes"]
    verify_execution_source(source, release["execution_source"])
    command = build_probe_command(manifest, source, output, args.gpu_uuid)
    output.mkdir(mode=0o700, exist_ok=False)
    os.chown(output, 1001, 1001)
    owner_token = uuid.uuid4().hex
    claim = {
        "schema": "openevo.q06.s2-bounded-r120-pass-claim.v1",
        "status": "CONSUMED",
        "cell_id": CELL_ID,
        "checkpoint_id": CHECKPOINT_ID,
        "episode_cap": 64,
        "fixed_passes": 1,
        "selected_gpu_uuid": args.gpu_uuid,
        "owned_container_token": owner_token,
        "prospective_contract": release["prospective_contract"],
        "source_manifest": release["source_manifest"],
        "formal_start": {"path": str(receipt), "sha256": sha256_file(receipt)},
        "command": command,
        "model_calls_before_launch": 0,
    }
    with (output / "s2-pass-claim.json").open("x", encoding="utf-8") as stream:
        json.dump(claim, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return run_with_limits(command, output,
                           max_wall_seconds=MAX_WALL_SECONDS,
                           max_output_bytes=contract["execution_constraints"]["max_persistent_output_bytes"],
                           owner_token=owner_token)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as exc:
        print(f"Q06 S2 BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
