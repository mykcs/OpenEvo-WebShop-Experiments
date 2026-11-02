#!/usr/bin/env python3
"""Execute one previously unconsumed Q06 S2 cell via canonical Formal Start.

Historical single-cell adapters remain frozen. This successor changes only the
checkpoint identity and opt-in observation producer; science runtime, panel,
carriers, data and sampling remain equal to the completed Bounded-R120 cell.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from scripts.openevo_webshop.q06_s2_bounded_r120_host_orchestrator import (
    GateError, GPU_RE, IMAGE_ID, PROBE_PATH, read_object, repo_file, require,
    sha256_file, same_sha256_digest, verify_execution_source,
)

CONTRACT_SCHEMA = "openevo.q06.s2-prospective-cell-contract.v1"
RELEASE_SCHEMA = "openevo.experiment-q06-s2-remaining-release.v1"
ROUTER_SCHEMA = "openevo.q06.s2-remaining-current-campaign.v1"
MANIFEST_SCHEMA = "openevo.q06.s2-remaining-probe-manifest.v1"
READ_ROOT = Path("/workspace")
SHARED_MANIFEST = "docs/evidence/q06-s2-bounded-r120-20261002/probe-manifest.v1.json"
SHARED_MANIFEST_SHA = "668a5bce7187fc0554a430c387b50ac5081a71878b767fcf9865648d0f39d53a"
ALLOWED_CELLS = {
    "bounded-r40": ("Bounded State", 40, "after_040_rounds"),
    "bounded-r80": ("Bounded State", 80, "after_080_rounds"),
    "gdr-r40": ("Effective-State GDR ON", 40, "after_040_rounds"),
    "gdr-r80": ("Effective-State GDR ON", 80, "after_080_rounds"),
    "gdr-r120": ("Effective-State GDR ON", 120, "after_120_rounds"),
    "ordinary-r80": ("Ordinary OpenEVO", 80, "after_080_updates"),
    "ordinary-r160": ("Ordinary OpenEVO", 160, "after_160_updates"),
}
SCIENCE_LIMITS = {
    "training_rollouts": 0, "optimizer_updates": 0, "teacher_calls": 0,
    "adaptive_reruns": 0, "retry_on_scientific_outcome": False,
    "episodes": 64, "passes": 1, "max_generation_calls": 960,
    "max_generated_tokens": 491520, "final_panel_access_allowed": False,
    "formal_task_consumption_allowed": True, "gpu_allocation_allowed": True,
}


def validate_contract(repo: Path, reference: dict) -> tuple[Path, dict]:
    path, _ = repo_file(repo, reference, "remaining-cell contract")
    contract = read_object(path)
    require(contract.get("schema") == CONTRACT_SCHEMA
            and contract.get("status") == "LAUNCH_AUTHORIZED"
            and contract.get("append_only") is True
            and contract.get("question_id") == "Q06_TOKEN_CLICK_ENTROPY"
            and contract.get("stage") == "S2", "remaining-cell contract is not authorized")
    cell = contract.get("cell", {})
    expected = ALLOWED_CELLS.get(cell.get("cell_id"))
    require(expected is not None, "cell is unavailable or already consumed")
    mechanism, rounds, checkpoint_id = expected
    require(cell.get("mechanism") == mechanism
            and cell.get("completed_stage2_rounds") == rounds
            and cell.get("checkpoint_id") == checkpoint_id
            and isinstance(cell.get("lineage_id"), str) and bool(cell["lineage_id"]),
            "remaining-cell checkpoint/lineage identity drift")
    limits = contract.get("execution_constraints", {})
    require(all(limits.get(key) == value for key, value in SCIENCE_LIMITS.items()),
            "remaining-cell single-pass science budget drift")
    for key in ("max_scorer_forward_calls", "max_scorer_input_tokens",
                "max_persistent_output_bytes"):
        require(type(limits.get(key)) is int and limits[key] > 0,
                f"missing finite protective resource limit: {key}")
    provenance = contract.get("resource_limit_provenance")
    require(isinstance(provenance, dict) and bool(provenance.get("source"))
            and bool(provenance.get("classification")), "resource limits lack provenance")
    return path, contract


def verify_external_file(path_value: str, digest: str, label: str) -> Path:
    path = Path(path_value)
    require(path.is_absolute() and not path.is_symlink() and path.is_file()
            and same_sha256_digest(sha256_file(path), digest), f"{label} byte identity drift: {path}")
    return path


def resolve_checkpoint_adapter(contract: dict, manifest: dict) -> tuple[dict, Path]:
    """Bind real summary bytes and, for Ordinary, its separate update receipt."""
    cell = contract["cell"]
    summary_path = verify_external_file(cell["source_summary_path"], cell["source_summary_sha256"],
                                        "checkpoint summary")
    summary = read_object(summary_path)
    checkpoint = summary["checkpoint"]
    updates = checkpoint.get("actual_parameter_update_events", checkpoint.get("update_count"))
    adapter_identity = contract["known_shared_identity"]["adapter"]
    require(summary.get("checkpoint_id") == cell["checkpoint_id"]
            and updates == cell["actual_parameter_update_events"]
            and checkpoint.get("adapter_sha256") == adapter_identity["adapter_sha256"],
            "source summary checkpoint binding drift")
    adapter_dir = manifest["endpoint"]["adapter_path"]
    if "adapter_path" in checkpoint:
        require(checkpoint["adapter_path"] == adapter_dir, "summary adapter path drift")
    else:
        require(cell["cell_id"].startswith("ordinary-"), "source summary lacks adapter path")
        require(all(manifest.get(key) == cell.get(key) and cell.get(key) for key in
                    ("source_update_path", "source_update_sha256")), "Ordinary update identity absent or differs")
        update_path = verify_external_file(cell["source_update_path"], cell["source_update_sha256"],
                                           "Ordinary parameter update")
        update = read_object(update_path)
        require(update.get("adapter_path") == adapter_dir
                and same_sha256_digest(update.get("adapter_digest_after"), checkpoint["adapter_sha256"]),
                "Ordinary update adapter identity drift")
    adapter_path = verify_external_file(str(Path(adapter_dir) / "adapter_model.safetensors"),
                                        adapter_identity["adapter_sha256"], "adapter")
    require(adapter_path.stat().st_size == adapter_identity["size_bytes"], "adapter size drift")
    return checkpoint, adapter_path


def validate_endpoint_source_files(endpoint: dict) -> None:
    files = endpoint.get("source_files", {})
    require(isinstance(files, dict) and bool(files), "checkpoint source-file inventory missing")
    expected_config = Path(endpoint["adapter_path"]) / "adapter_config.json"
    require(any(isinstance(ref, dict) and ref.get("path") == str(expected_config)
                for ref in files.values()), "adapter configuration must be hash-bound")
    for role, reference in files.items():
        require(isinstance(reference, dict) and isinstance(reference.get("path"), str)
                and type(reference.get("bytes")) is int, f"invalid checkpoint source-file reference: {role}")
        path = verify_external_file(reference["path"], reference.get("sha256"), role)
        require(path.stat().st_size == reference["bytes"], f"checkpoint source-file size drift: {role}")


def validate_manifest(repo: Path, release: dict, contract: dict,
                      source: Path, science_source: Path) -> tuple[Path, dict]:
    manifest_path, _ = repo_file(repo, release["source_manifest"], "remaining-cell manifest")
    manifest = read_object(manifest_path)
    shared_path, _ = repo_file(repo, {"path": SHARED_MANIFEST, "sha256": SHARED_MANIFEST_SHA},
                               "frozen shared science identity")
    shared = read_object(shared_path)
    cell = contract["cell"]
    checkpoint_id = cell["checkpoint_id"]
    require(manifest.get("schema") == MANIFEST_SCHEMA
            and manifest.get("cell_id") == cell["cell_id"]
            and manifest.get("checkpoint_id") == checkpoint_id
            and manifest.get("contract") == release["prospective_contract"]
            and manifest.get("runtime_execution_sha") == release["science_runtime_source"]["revision"],
            "remaining-cell manifest identity drift")
    for key in ("model", "sampling", "common_carriers", "seed_root", "webshop_dataset",
                "runtime_files", "task_panel_manifest_path", "task_panel_manifest_sha256",
                "panel_sha256", "panel_size", "final_panel_access_allowed"):
        require(manifest.get(key) == shared.get(key), f"frozen shared science changed: {key}")
    require(cell.get("evaluation_panel_sha256") == manifest["panel_sha256"]
            and manifest["panel_size"] == 64 and manifest["final_panel_access_allowed"] is False,
            "remaining-cell panel identity drift")
    known = contract["known_shared_identity"]
    for key, value in {
        "base_model": manifest["model"]["base_model"],
        "base_model_revision": manifest["model"]["revision"],
        "common_agent_system_sha256": manifest["common_carriers"]["agent_system_sha256"],
        "common_skill_sha256": manifest["common_carriers"]["skill_sha256"],
        "common_memory_sha256": manifest["common_carriers"]["memory_sha256"],
        "sampling": manifest["sampling"],
    }.items():
        require(known.get(key) == value, f"contract shared identity drift: {key}")
    checkpoint, adapter_path = resolve_checkpoint_adapter(contract, manifest)
    adapter_identity = known["adapter"]
    require(manifest.get("checkpoints") == {checkpoint_id: {
        "adapter_sha256": checkpoint["adapter_sha256"], "state_id": checkpoint["state_id"]}},
            "manifest checkpoint/state binding drift")
    for key in ("completed_stage2_rounds", "actual_parameter_update_events",
                "source_summary_path", "source_summary_sha256"):
        require(manifest.get(key) == cell.get(key), f"manifest/source summary drift: {key}")
    endpoint = manifest["endpoint"]
    require(endpoint.get("checkpoint_id") == checkpoint_id, "adapter endpoint drift")
    require(adapter_path.stat().st_size == adapter_identity["size_bytes"]
            and endpoint["source_files"]["adapter"] == {
                "path": str(adapter_path), "bytes": adapter_path.stat().st_size,
                "sha256": adapter_identity["adapter_sha256"]}, "adapter size/manifest drift")
    validate_endpoint_source_files(endpoint)
    verify_external_file(str(source / PROBE_PATH), manifest["producer_sha256"], "observation producer")
    for relative, digest in manifest["runtime_files"].items():
        verify_external_file(str(science_source / relative), digest, "frozen science runtime")
    carriers = manifest["common_carriers"]
    for key in ("agent_system", "skill", "memory"):
        verify_external_file(carriers[key + "_path"], carriers[key + "_sha256"], key)
    seed = manifest["seed_root"]
    for relative, digest in seed["package_files_sha256"].items():
        verify_external_file(str(Path(seed["path"]) / relative), digest, "Webshop seed package")
    dataset = manifest["webshop_dataset"]
    verify_external_file(dataset["manifest_path"], dataset["manifest_sha256"], "Webshop dataset")
    verify_external_file(str(science_source / manifest["task_panel_manifest_path"]),
                         manifest["task_panel_manifest_sha256"], "task panel")
    return manifest_path, manifest


def validate_authority(repo: Path, release_path: Path, router_path: Path,
                       source: Path) -> tuple[dict, dict, dict, Path, dict]:
    repo, release_path, router_path = repo.resolve(), release_path.resolve(), router_path.resolve()
    require(repo in release_path.parents and repo in router_path.parents
            and not release_path.is_symlink() and not router_path.is_symlink(), "authority path escapes repo")
    release, router = read_object(release_path), read_object(router_path)
    require(release.get("schema") == RELEASE_SCHEMA and router.get("schema") == ROUTER_SCHEMA,
            "remaining-cell release/router schema drift")
    for document in (release, router):
        require(document.get("status") == "launch-authorized"
                and document.get("formal_task_consumption_allowed") is True
                and document.get("formal_rollout_limit") == 64,
                "remaining-cell release/router authorization drift")
    require(release.get("final_panel_access_limit") == 0
            and router.get("final_panel_access_allowed") is False
            and router.get("release_receipt_sha256") == sha256_file(release_path)
            and router.get("campaign_id") == release.get("campaign_id")
            and router.get("experiment_id") == release.get("experiment_id")
            and release["canonical_activation"]["path"] == router_path.relative_to(repo).as_posix(),
            "release/router exact binding drift")
    _, contract = validate_contract(repo, release["prospective_contract"])
    cell = contract["cell"]
    require(router.get("prospective_contract") == release["prospective_contract"]
            and release.get("endpoint") == router.get("endpoint") == cell["checkpoint_id"]
            and release.get("budget") == contract["execution_constraints"], "release/contract budget drift")
    verify_execution_source(source, release["execution_source"])
    science_source = Path(release["science_runtime_source"]["path"]).resolve()
    verify_execution_source(science_source, release["science_runtime_source"])
    manifest_path, manifest = validate_manifest(repo, release, contract, source, science_source)
    return release, router, contract, manifest_path, manifest


def validate_formal_start(repo: Path, release: dict, router: dict, release_path: Path,
                          router_path: Path, source: Path, output: Path,
                          gpu_uuid: str, receipt_path: Path) -> dict:
    from scripts import oe_impl
    from scripts.openevo_webshop import prelaunch_zero_formal

    require(GPU_RE.fullmatch(gpu_uuid) is not None
            and gpu_uuid in router.get("authorized_gpu_uuids", [])
            and os.getenv("OPENEVO_ASSIGNED_GPU_UUIDS") == gpu_uuid
            and os.getenv("OPENEVO_AUTHORIZED_GPU_UUIDS") == gpu_uuid, "selected GPU authorization drift")
    start = read_object(receipt_path)
    require(start.get("schema") == prelaunch_zero_formal.FORMAL_START_SCHEMA
            and start.get("receipt_sha256") == prelaunch_zero_formal.canonical_sha256(
                {key: value for key, value in start.items() if key != "receipt_sha256"})
            and start.get("status") == "FORMAL_START_COMMITTED"
            and start.get("formal_start_committed") is True
            and start.get("formal_task_consumption_may_begin") is True
            and start.get("selected_gpu_uuids") == [gpu_uuid]
            and start.get("campaign_id") == release["campaign_id"]
            and start.get("image_id") == IMAGE_ID, "canonical Formal Start identity drift")
    profile_path, _ = repo_file(repo, release["execution_readiness"]["profile"], "runtime profile")
    repo_file(repo, release["host_orchestrator"], "remaining-cell orchestrator")
    repo_file(repo, release["probe_guard"], "post-load probe guard")
    require((repo / release["host_orchestrator"]["path"]).resolve() == Path(__file__).resolve(),
            "release binds a different orchestrator")
    profile = read_object(profile_path)
    require(profile.get("execution", {}).get("mode") == "host-orchestrator"
            and profile.get("execution_source") == release["execution_source"]
            and profile.get("s2_prospective_contract") == release["prospective_contract"]
            and profile.get("host_orchestrator") == release["host_orchestrator"]
            and profile.get("probe_guard") == release["probe_guard"], "runtime profile binding drift")
    readiness = start["readiness"]
    require(readiness["profile"]["path"] == str(profile_path)
            and same_sha256_digest(readiness["profile"]["sha256"], sha256_file(profile_path)),
            "Formal Start profile binding drift")
    ready_ref = readiness["ready_receipt"]
    ready = read_object(verify_external_file(ready_ref["path"], ready_ref["sha256"], "READY receipt"))
    require(ready.get("verdict") == "READY"
            and ready["release_receipt"]["path"] == str(release_path.resolve())
            and same_sha256_digest(ready["release_receipt"]["sha256"], sha256_file(release_path))
            and ready["profile"] == readiness["profile"], "READY receipt authority drift")
    pre_ref = start["prelaunch_receipt"]
    prelaunch_zero_formal.validate_prelaunch_receipt(read_object(verify_external_file(
        pre_ref["path"], pre_ref["sha256"], "prelaunch receipt")), expected_gpu_uuids=[gpu_uuid])
    profile["_formal_start_context"] = readiness
    command = oe_impl.host_orchestrator_identity_command(
        profile, [gpu_uuid], Path(start["output_root"]),
        os.getenv("OPENEVO_WANDB_MODE", "disabled"), receipt_path)
    require(start["command_identity"] == oe_impl._command_identity(command, IMAGE_ID),
            "Formal Start command identity drift")
    for flag, expected in (("--authority-repo", str(repo)), ("--execution-source", str(source)),
                           ("--release", str(release_path.resolve())),
                           ("--router", str(router_path.resolve())), ("--output-dir", str(output)),
                           ("--gpu-uuid", gpu_uuid)):
        require(flag in command and command[command.index(flag) + 1] == expected,
                f"Formal Start actual caller differs from profile: {flag}")
    return start


def build_probe_command(repo: Path, source: Path, science_source: Path, manifest_path: Path,
                        manifest: dict, release: dict, contract: dict,
                        output: Path, gpu_uuid: str) -> list[str]:
    cell, limits = contract["cell"], contract["execution_constraints"]
    model, common, endpoint = manifest["model"], manifest["common_carriers"], manifest["endpoint"]
    mounts = [repo, source, science_source, Path(manifest["seed_root"]["path"]),
              Path(model["local_model_path"]), Path(endpoint["adapter_path"]),
              *(Path(common[key + "_path"]) for key in ("memory", "skill", "agent_system"))]
    command = ["docker", "run", "--rm", "--network", "none", "--user", "1001:1001",
               "--gpus", "device=" + gpu_uuid]
    for path in dict.fromkeys(mounts):
        require(path.exists() and not path.is_symlink() and (path == READ_ROOT or READ_ROOT in path.parents),
                f"read-only scientific input outside qualified root or missing: {path}")
    # GalaxyFS descendant binds have previously hidden nested paths. Reuse the
    # root-bind recipe qualified with real CPU+CUDA calls; only this fresh exact
    # output is writable. No existing workload receives a mutation.
    command += ["--mount", f"type=bind,src={READ_ROOT},dst={READ_ROOT},readonly"]
    command += ["--mount", f"type=bind,src={output},dst={output}",
                "--tmpfs", "/tmp:rw,exec,nosuid,size=1g,mode=1777"]
    for value in ("HOME=/tmp", "TMPDIR=/tmp", "HF_HOME=/tmp/hf", "XDG_CACHE_HOME=/tmp/cache",
                  "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_openevo", "TRITON_CACHE_DIR=/tmp/triton",
                  "PYTHONDONTWRITEBYTECODE=1", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1",
                  "CUDA_VISIBLE_DEVICES=" + gpu_uuid,
                  f"PYTHONPATH={science_source}:{repo}:{source}:{science_source}/vendor/openevo-webshop/src"):
        command += ["--env", value]
    command += ["--entrypoint", "python3", IMAGE_ID, str(repo / release["probe_guard"]["path"]),
                "--authority-repo", str(repo), "--probe-script", str(source / PROBE_PATH),
                "--output-dir", str(output), "--max-output-bytes", str(limits["max_persistent_output_bytes"]),
                "--max-scorer-forward-calls", str(limits["max_scorer_forward_calls"]),
                "--max-scorer-input-tokens", str(limits["max_scorer_input_tokens"]), "--",
                "--manifest", str(manifest_path), "--panel-manifest", str(science_source / manifest["task_panel_manifest_path"]),
                "--checkpoint-id", cell["checkpoint_id"], "--runtime-source-root", str(science_source),
                "--seed-root", manifest["seed_root"]["path"], "--model", model["local_model_path"],
                "--revision", model["revision"], "--sd-adapter", endpoint["adapter_path"],
                "--memory", common["memory_path"], "--skill", common["skill_path"],
                "--agent", common["agent_system_path"], "--gpu-uuid", gpu_uuid,
                "--output-dir", str(output), "--instrumentation", "on",
                "--q06-pre-action-diagnostics", "--q06-lineage-id", cell["lineage_id"]]
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("authority-repo", "execution-source", "release", "router", "output-dir"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args(argv)
    repo, source, output = args.authority_repo.resolve(), args.execution_source.resolve(), args.output_dir.absolute()
    require(not output.exists() and not output.is_symlink() and output.parent.is_dir()
            and not output.parent.is_symlink(), "cell output consumed or output parent unavailable")
    require(not subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
                               check=True, capture_output=True, text=True).stdout.strip(), "authority checkout dirty")
    release, router, contract, manifest_path, manifest = validate_authority(repo, args.release, args.router, source)
    receipt = Path(os.environ.get("OPENEVO_FORMAL_START_RECEIPT", "")).resolve()
    validate_formal_start(repo, release, router, args.release, args.router, source,
                          output, args.gpu_uuid, receipt)
    science_source = Path(release["science_runtime_source"]["path"]).resolve()
    command = build_probe_command(repo, source, science_source, manifest_path, manifest, release,
                                  contract, output, args.gpu_uuid)
    output.mkdir(mode=0o700, exist_ok=False)
    os.chown(output, 1001, 1001)
    token = uuid.uuid4().hex
    command = [*command[:2], "--name", f"q06-s2-{contract['cell']['cell_id']}-{token[:12]}",
               "--label", "com.openevo.q06.s2.owner=" + token, *command[2:]]
    claim = {"schema": "openevo.q06.s2-remaining-pass-claim.v1", "status": "CONSUMED",
             "cell": contract["cell"], "episode_cap": 64, "fixed_passes": 1,
             "prospective_contract": release["prospective_contract"], "source_manifest": release["source_manifest"],
             "formal_start": {"path": str(receipt), "sha256": sha256_file(receipt)},
             "selected_gpu_uuid": args.gpu_uuid, "owned_container_token": token, "command": command,
             "resource_limit_provenance": contract["resource_limit_provenance"], "model_calls_before_launch": 0}
    with (output / "s2-pass-claim.json").open("x", encoding="utf-8") as stream:
        json.dump(claim, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    # No agent-made wall-time budget: canonical wrapper accounts actual GPU time.
    # The in-container guard enforces frozen forward/token/storage safety limits.
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as exc:
        print(f"Q06 S2 remaining BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
