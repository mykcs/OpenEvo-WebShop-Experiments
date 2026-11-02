#!/usr/bin/env python3
"""Canonical single-GPU host adapter for the frozen fresh-checkpoint probe.

Uses the same host-orchestrator/RO-workspace Docker boundary as Q05 sensitivity.
There is deliberately no scheduler or automatic replay policy here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

REPO = Path(__file__).resolve().parents[2]
WORKER = "scripts/analysis/advisor_fresh_checkpoint_validation_20261020.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2) + "\n")


def checked_manifest(path: Path, expected: str, ids: list[str]) -> dict:
    if len(expected) != 64 or sha(path) != expected:
        raise ValueError("Frozen manifest SHA drift")
    manifest = read(path)
    if manifest.get("final_panel_access_allowed") is not False or manifest.get("optimizer_updates") != 0:
        raise ValueError("Fresh evaluation must retain zero final access and optimizer updates")
    if len(manifest.get("tasks", [])) != 64 or not ids or len(set(ids)) != len(ids):
        raise ValueError("Exactly64 frozen tasks and unique checkpoint IDs required")
    panel_sha = hashlib.sha256(json.dumps(manifest["tasks"], ensure_ascii=False,
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if manifest.get("panel_sha256") != panel_sha:
        raise ValueError("Frozen task panel SHA drift")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", item) for item in ids):
        raise ValueError("Checkpoint IDs must be safe single output path components")
    if any(item not in manifest["checkpoints"] for item in ids):
        raise ValueError("Lane references an unknown frozen checkpoint")
    return manifest


def validate_start(args: argparse.Namespace) -> tuple[dict, dict]:
    manifest = checked_manifest(args.manifest, args.manifest_sha256, args.checkpoint_id)
    formal_path = Path(os.environ["OPENEVO_FORMAL_START_RECEIPT"])
    formal = read(formal_path)
    gpu = args.gpu_uuid
    if (manifest.get("image_id") != args.image
            or formal.get("formal_start_committed") is not True
            or formal.get("formal_task_consumption_may_begin") is not True
            or formal.get("run_id") != args.run_id
            or formal.get("selected_gpu_uuids") != [gpu]
            or formal.get("image_id") != args.image
            or os.environ.get("OPENEVO_ASSIGNED_GPU_UUIDS") != gpu
            or args.output.resolve() != Path(formal["output_root"]).resolve() / "evaluation"):
        raise ValueError("Canonical Formal Start, GPU, image or output binding differs")
    pref = formal["readiness"]["profile"]
    profile_path = Path(pref["path"])
    if not profile_path.is_absolute() or profile_path.is_symlink():
        raise ValueError("Formal profile path must be an absolute regular authority file")
    if sha(profile_path) != pref["sha256"].removeprefix("sha256:"):
        raise ValueError("Formal profile SHA drift")
    profile = read(profile_path)
    # Compare the entire command, rather than only the manifest path: the lane
    # population, GPU and output are part of the prospective execution binding.
    expected = ["python3", "${EXPERIMENT_DIR}/" + str(Path(__file__).relative_to(REPO)),
                "--manifest", str(args.manifest), "--manifest-sha256", args.manifest_sha256,
                "--run-id", args.run_id, "--gpu-uuid", gpu, "--image", args.image,
                "--output", "${RUN_DIR}/evaluation"]
    for item in args.checkpoint_id:
        expected.extend(["--checkpoint-id", item])
    if profile["command"] != expected:
        raise ValueError("Formal profile command does not bind this exact checkpoint lane")
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain",
                                     "--untracked-files=all"], text=True).strip()
    if dirty or head != profile["source"]["experiment"]["revision"]:
        raise ValueError("Fresh execution checkout must be clean and exact")
    observed = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", args.image], text=True).strip()
    if observed != args.image:
        raise ValueError("Immutable Docker image identity differs")
    workspace = Path("/workspace")
    if (not REPO.is_relative_to(workspace)
            or not args.manifest.resolve().is_relative_to(workspace)
            or not args.output.resolve().is_relative_to(workspace)):
        raise ValueError("Frozen inputs and output must reside under the canonical workspace")
    return formal, manifest


def host(args: argparse.Namespace) -> None:
    formal, manifest = validate_start(args)
    # A second start must never repeat even a completed cell. A partial lane
    # requires a separately reconciled successor, as in the existing Q05 adapter.
    args.output.mkdir(exist_ok=False, parents=True, mode=0o755)
    write_once(args.output / "START.json", {
        "status": "STARTED_NO_AUTOMATIC_REPLAY", "formal_start": formal,
        "manifest_sha256": args.manifest_sha256, "checkpoint_ids": args.checkpoint_id,
        "gpu_uuid": args.gpu_uuid, "started_at_epoch": time.time(),
    })
    workspace = Path("/workspace")
    for index, checkpoint_id in enumerate(args.checkpoint_id):
        output = args.output / checkpoint_id
        output.mkdir(mode=0o700)
        os.chown(output, 1001, 1001)
        cmd = ["docker", "run", "--rm", "--pull=never", "--read-only", "--network", "none",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "--name", f"{args.run_id}-{index}", "--user", "1001:1001",
               "--gpus", "device=" + args.gpu_uuid,
               "--mount", f"type=bind,src={workspace},dst={workspace},readonly",
               "--mount", f"type=bind,src={output},dst={output}",
               "--tmpfs", "/tmp:rw,exec,nosuid,nodev,size=4g,mode=1777"]
        for value in ("HOME=/tmp", "TMPDIR=/tmp", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1",
                      "PYTHONDONTWRITEBYTECODE=1", "WANDB_MODE=disabled",
                      "CUBLAS_WORKSPACE_CONFIG=:4096:8", "CUDA_VISIBLE_DEVICES=" + args.gpu_uuid,
                      "OPENEVO_RUNTIME_IMAGE_ID=" + args.image):
            cmd += ["--env", value]
        runtime_root = Path(manifest["runtime_source_root"])
        core_root = Path(manifest["core_root"])
        pythonpath = [core_root / "src", runtime_root, runtime_root / "vendor/openevo-webshop/src",
                      REPO, REPO / "vendor/openevo-webshop/src"]
        cmd += ["--env", "PYTHONPATH=" + ":".join(map(str, pythonpath))]
        cmd += ["--entrypoint", "python3", args.image, str(REPO / WORKER), "evaluate",
                "--manifest", str(args.manifest), "--checkpoint-id", checkpoint_id,
                "--gpu-uuid", args.gpu_uuid, "--output", str(output)]
        write_once(args.output / f"{checkpoint_id}-launch.json", {"command": cmd})
        started = time.time()
        with (args.output / f"{checkpoint_id}.log").open("xb") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        write_once(args.output / f"{checkpoint_id}-exit.json", {
            "returncode": result.returncode, "started_at_epoch": started,
            "finished_at_epoch": time.time(), "allocated_gpu_count": 1,
        })
        if result.returncode:
            raise RuntimeError("Fresh cell failed; preserve evidence and reconcile before any successor")
        summary = read(output / "COMPLETE.json")
        if (summary.get("status") != "COMPLETE" or summary.get("tasks") != 64
                or summary.get("checkpoint_id") != checkpoint_id
                or summary.get("gpu_uuid") != args.gpu_uuid
                or summary.get("manifest_sha256") != args.manifest_sha256
                or summary.get("optimizer_updates") != 0 or summary.get("final_panel_access") != 0):
            raise ValueError("Fresh checkpoint did not complete all64 frozen tasks")
    write_once(args.output / "EVALUATION_COMPLETE.json", {
        "status": "EVALUATION_COMPLETE_ANALYSIS_REQUIRED", "evaluation_episodes": 64 * len(args.checkpoint_id),
        "manifest_sha256": args.manifest_sha256, "checkpoint_ids": args.checkpoint_id,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    host(parser.parse_args())


if __name__ == "__main__":
    main()
