#!/usr/bin/env python3
"""Train the two preregistered H1.41 continual-update arms.

This stage consumes only the sealed H1.40 E2 records. It creates no new
WebShop harvest and does not evaluate any formal panel; the later evaluation
launcher is the only stage that consumes the 768 preregistered attempts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.openevo_webshop.gpu_resource_wrapper import account_receipt
from scripts.openevo_webshop.launch_h140_r2_e2_harvest import JAVA_HOME, PYTHON
from scripts.openevo_webshop.prepare_h140_g2_training_inputs import prepare
from scripts.openevo_webshop.validate_h141_preregistration import validate as validate_prereg


CAMPAIGN_ID = "20260821-0141-h141-magnitude-screen"
CONFIG = Path("configs/experiment/h1.41-magnitude-screen-v2.json")
PATH_CONTRACT = Path("configs/experiment/h1.41-path-contract-v1.json")
RUNNER = Path("scripts/openevo_webshop/run_parametric_smoke_tracked.py")
WRAPPER = Path("scripts/openevo_webshop/gpu_resource_wrapper.py")
STARTUP_GATE = Path("scripts/openevo_webshop/startup_gate_5090.sh")
TRAINER_PYTHON_CONTAINER = "/workspace/src/openevo-experiment/scripts/openevo_webshop/server_trainer_python.sh"
IMAGE = "experiment-host/openevo-scientific:qwen25-3b-webshop-20260814"
IMAGE_DIGEST = "sha256:ce92e587e7c934827025c001fd9b70ec4cb6de2deacfdf9f0cc0777d3ab928f1"
PAYLOAD_UID_GID = "1001:1001"
MODEL_REVISION = "0c32faf81d8f2f83cf79860d2a48111f166348d6dd38cd348576d8b3a0141c1e"
OPEN_EVO_SHA = "796e6248a61112a61c880885844d3b0420c0b6ae"
RUNTIME_FINGERPRINT = "495e06379dbc4eadfd6a1941182d1ec8158691febd190e7741763eeed192c672"
G1_ADAPTER_FILE_SHA = "47e63d417f85cb2defed5ff1ee934128b96e6a4f78a8f410fa052957d919190c"


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def training_artifact_id(payload: dict[str, Any]) -> str:
    artifact = payload.get("artifact") or {}
    manifest = payload.get("artifact_manifest") or {}
    value = artifact.get("artifact_id") or manifest.get("adapter_id")
    if not isinstance(value, str) or not value:
        raise ValueError("training evidence lacks a stable artifact identity")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_payload_owned(path: Path) -> None:
    subprocess.run(["chown", "-R", PAYLOAD_UID_GID, str(path)], check=True)


def paths(root: Path) -> dict[str, str]:
    contract = read(root / PATH_CONTRACT)
    return {name: str(spec["host_path"]) for name, spec in contract["paths"].items()}


def container_path(host_path: Path, host_run_root: Path) -> str:
    relative = host_path.resolve().relative_to(host_run_root.resolve())
    return f"/workspace/runs/phase-h141-magnitude-screen-20260821/{relative.as_posix()}"


def validate(root: Path, e2_run_root: Path, patched_receipt: Path) -> tuple[dict[str, Any], dict[str, str]]:
    validate_prereg(root, require_router=True)
    config = read(root / CONFIG)
    current = read(root / "configs/experiment/current-campaign.json")
    if current.get("campaign_id") != CAMPAIGN_ID or current.get("status") != "launch-authorized" or current.get("formal_task_consumption_allowed") is not True:
        raise ValueError("H1.41 router is not launch-authorized")
    if config.get("status") != "preregistered-pre-execution" or config.get("formal_task_consumption_allowed") is not False:
        raise ValueError("H1.41 scientific contract was edited or opened")
    guard = config.get("formal_execution_guard") or {}
    if guard.get("tasks_consumed") != 0 or guard.get("gpu_hours_consumed") != 0.0 or guard.get("arms_executed"):
        raise ValueError("H1.41 formal execution guard is not zero")
    if not patched_receipt.is_file() or read(patched_receipt).get("patched_core_base_sha") != OPEN_EVO_SHA:
        raise ValueError("H1.41 patched-core receipt is missing or not pinned")
    p = paths(root)
    for key in ("experiment", "openevo_clean", "openevo_reset_patch", "model", "dataset", "cache", "run_root", "adapter"):
        if not Path(p[key]).exists():
            raise ValueError(f"H1.41 path contract target is missing: {key}")
    if subprocess.run(["git", "diff", "--quiet"], cwd=root, check=False).returncode != 0:
        raise ValueError("experiment checkout is dirty")
    if sha256(Path(p["adapter"]) / "adapter_model.safetensors") != G1_ADAPTER_FILE_SHA:
        raise ValueError("sealed G1 adapter identity changed")
    if not e2_run_root.is_dir():
        raise ValueError("sealed H1.40 E2 run root is missing")
    return config, p


def run_gate(root: Path, uuid: str, scratch: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update({"OPENEVO_AUTHORIZED_GPU_UUIDS": uuid, "OPENEVO_SELECTED_GPU_UUIDS": uuid, "OPENEVO_SCRATCH_ROOT": str(scratch), "JAVA_HOME": JAVA_HOME})
    return subprocess.run(["bash", str(root / STARTUP_GATE), "--", "true"], cwd=root, env=env, capture_output=True, text=True, check=False)


def docker_command(p: dict[str, str], run_root: Path, manifest: Path, artifact_root: Path, evidence: Path, run_id: str, increment: int, prior_adapter: str, prior_id: str, arm: str, uuid: str, experiment_sha: str) -> list[str]:
    core = p["openevo_clean"] if arm == "C0_exact_h140" else p["openevo_reset_patch"]
    prior_mount = container_path(artifact_root / "prior-g1", run_root)
    command = [
        "docker", "run", "--rm", "--init", "--read-only", "--tmpfs", "/tmp:rw,nosuid,size=4g", "--shm-size=64g", "--pids-limit", "4096", "--gpus", f"device={uuid}", "--user", "1001:1001",
        "-v", f"{p['experiment']}:/workspace/src/openevo-experiment:ro", "-v", f"{core}:/workspace/src/openevo-core:ro", "-v", f"{core}:/workspace/openevo-core:ro",
        "-v", f"{p['model']}:/workspace/models/qwen2.5-7b-instruct:ro", "-v", f"{p['model']}:/workspace/hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/{MODEL_REVISION}:ro", "-v", f"{p['cache']}:/workspace/cache:rw", "-v", f"{p['run_root']}:/workspace/runs/phase-h141-magnitude-screen-20260821:rw",
        "-v", "/workspace/openevo-workspace:/workspace/openevo-workspace:ro", "-v", f"{p['adapter']}:{prior_mount}:ro", "-w", "/workspace/src/openevo-experiment",
        "-e", f"CUDA_VISIBLE_DEVICES={uuid}", "-e", "TOKENIZERS_PARALLELISM=false", "-e", "TRANSFORMERS_OFFLINE=1", "-e", "HF_HUB_OFFLINE=1", "-e", f"JAVA_HOME={JAVA_HOME}", "-e", "OPENEVO_TRAINER_CORE=/workspace/src/openevo-core", "-e", "HF_HOME=/workspace/hf-cache", "-e", "HF_HUB_CACHE=/workspace/hf-cache/hub", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "HOME=/tmp", "-e", "TMPDIR=/tmp", "-e", "XDG_CACHE_HOME=/workspace/cache",
        "-e", "PYTHONPATH=/workspace/src/openevo-core/src:/workspace/src/openevo-experiment:/workspace/src/openevo-experiment/scripts/openevo_webshop",
    ]
    if arm == "C1_reset_magnitude":
        command.extend(["-e", "OPENEVO_SD_LORA_RESET_ALL_MAGNITUDES=1"])
    command.extend([
        "--entrypoint", PYTHON, IMAGE, "-m", "scripts.openevo_webshop.run_parametric_smoke_tracked",
        "--dataset-manifest", container_path(manifest, run_root), "--dataset-artifact-id", f"h141-{arm}-increment-{increment:02d}", "--artifact-root", container_path(artifact_root, run_root), "--evidence", container_path(evidence, run_root), "--run-id", run_id,
        "--benchmark", "webshop", "--python", TRAINER_PYTHON_CONTAINER, "--base-model", "Qwen/Qwen2.5-7B-Instruct", "--model-revision", MODEL_REVISION, "--max-steps", "16", "--max-records", "2", "--minimum-reward", "0", "--experiment-git-sha", experiment_sha, "--upstream-openevo-sha", OPEN_EVO_SHA, "--runtime-fingerprint", RUNTIME_FINGERPRINT, "--image-digest", IMAGE_DIGEST, "--gpu-uuid", uuid, "--wandb-mode", "offline", "--wandb-project", "openevo-webshop-experiments", "--wandb-entity", "example-entity", "--campaign-id", CAMPAIGN_ID, "--wandb-group", f"{CAMPAIGN_ID}-{arm}", "--wandb-dir", f"{container_path(artifact_root, run_root)}/wandb",
        "--prior-adapter", prior_adapter, "--prior-artifact-id", prior_id,
    ])
    return command


def train(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    config, p = validate(root, args.e2_run_root.resolve(), args.patched_receipt.resolve())
    run_root = Path(p["run_root"]).resolve()
    if run_root.exists():
        allowed_preflight_entries = {"receipts", "scratch", "inputs", "training", "scheduler-receipt.json"}
        unexpected_entries = sorted(path.name for path in run_root.iterdir() if path.name not in allowed_preflight_entries and not path.name.startswith("failed-engineering-"))
        if unexpected_entries:
            raise ValueError(f"H1.41 run root contains unexpected entries: {unexpected_entries}")
    run_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    input_receipt = run_root / "inputs" / "training-inputs.json"
    if input_receipt.is_file():
        inputs = read(input_receipt)
        if inputs.get("schema") != "openevo.webshop-h1.40-g2-training-inputs.v1" or inputs.get("record_count") != 16 or inputs.get("increment_count") != 8:
            raise ValueError("existing H1.41 training inputs are not the sealed 8x2 selection")
    else:
        inputs = prepare(args.e2_run_root / "e2-harvest-attempts-reconciled.jsonl", args.e2_run_root / "e2-selection-reconciled.json", run_root / "inputs", records_uri_resolver=lambda path: Path(container_path(path, run_root)).as_uri())
    make_payload_owned(run_root / "inputs")
    experiment_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    scheduler_path = run_root / "scheduler-receipt.json"
    if scheduler_path.is_file():
        scheduler = read(scheduler_path)
        if scheduler.get("schema") != "openevo.webshop-h1.41-magnitude-training-scheduler.v1" or scheduler.get("campaign_id") != CAMPAIGN_ID or scheduler.get("status") != "running":
            raise ValueError("existing H1.41 scheduler is not a resumable running receipt")
    else:
        scheduler = {"schema": "openevo.webshop-h1.41-magnitude-training-scheduler.v1", "status": "running", "campaign_id": CAMPAIGN_ID, "experiment_sha": experiment_sha, "open_evo_sha": OPEN_EVO_SHA, "patched_core_receipt": str(args.patched_receipt.resolve()), "training_inputs": inputs, "arms": {}}
        atomic_write(scheduler_path, scheduler)
    for arm in ("C0_exact_h140", "C1_reset_magnitude"):
        arm_root = run_root / "training" / arm
        artifact_root = arm_root / "artifacts" / "sd-lora"
        for name in ("receipts", "logs", "evidence"):
            (arm_root / name).mkdir(mode=0o700, parents=True, exist_ok=True)
        make_payload_owned(arm_root)
        artifact_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        (artifact_root / "prior-g1").mkdir(mode=0o700, exist_ok=True)
        make_payload_owned(artifact_root)
        prior_adapter = container_path(artifact_root / "prior-g1", run_root)
        prior_id = "h1.38b-g1-sd-lora-final"
        scheduler.setdefault("arms", {})
        arm_state = scheduler["arms"].setdefault(arm, {"status": "running", "attempts": []})
        arm_state["status"] = "running"
        arm_state.setdefault("attempts", [])
        atomic_write(scheduler_path, scheduler)
        for item in inputs["increments"]:
            increment = int(item["increment_index"])
            evidence = arm_root / "evidence" / f"increment-{increment:02d}.json"
            receipt = arm_root / "receipts" / f"increment-{increment:02d}.json"
            prior_attempt = next((row for row in arm_state["attempts"] if row.get("increment") == increment), None)
            if prior_attempt is not None:
                prior_adapter = str(prior_attempt["adapter_uri"]).removeprefix("file://")
                prior_id = str(prior_attempt["artifact_id"])
                continue
            if evidence.is_file() and receipt.is_file():
                existing_receipt = read(receipt)
                existing_accounting = account_receipt(existing_receipt)
                existing_payload = read(evidence)
                existing_manifest = existing_payload.get("artifact_manifest") or {}
                if existing_receipt.get("return_code") == 0 and existing_accounting.get("status") == "exact" and existing_payload.get("training_record_count") == 2 and existing_manifest.get("component_count") == 9 + increment:
                    prior_adapter = str(existing_payload["adapter_uri"]).removeprefix("file://")
                    prior_id = training_artifact_id(existing_payload)
                    arm_state["attempts"].append({"increment": increment, "gpu_uuid": existing_receipt["gpu_uuids"][0], "return_code": 0, "receipt": str(receipt), "receipt_sha256": sha256(receipt), "gpu_hours": existing_accounting["gpu_hours"], "evidence": str(evidence), "evidence_sha256": sha256(evidence), "artifact_id": prior_id, "adapter_uri": existing_payload["adapter_uri"]})
                    atomic_write(scheduler_path, scheduler)
                    continue
            log = (arm_root / "logs" / f"increment-{increment:02d}.log").open("w", encoding="utf-8")
            uuid = next((candidate for candidate in read(root / "configs/experiment/current-campaign.json")["authorized_gpu_uuids"] if run_gate(root, candidate, arm_root / "scratch" / candidate).returncode == 0), None)
            if uuid is None:
                raise RuntimeError(f"no authorized live-idle GPU for {arm}")
            command = docker_command(p, run_root, Path(item["dataset_manifest"]), artifact_root, evidence, f"h141-{arm}-{increment:02d}", increment, prior_adapter, prior_id, arm, uuid, experiment_sha)
            wrapper = [PYTHON, str(root / WRAPPER), "--receipt", str(receipt), "--attempt-id", f"{arm}-{increment:02d}", "--campaign-id", CAMPAIGN_ID, "--stage", "training", "--command-identity", f"h141-{arm}-training-v1", "--gpu-uuids", uuid, "--path-contract", str(root / PATH_CONTRACT), "--", *command]
            env = os.environ.copy()
            env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            completed = subprocess.run(wrapper, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=False, text=True)
            log.close()
            value = read(receipt) if receipt.is_file() else {}
            accounting = account_receipt(value)
            if accounting.get("status") != "exact":
                raise RuntimeError(f"{arm} increment {increment} has no exact resource receipt")
            if completed.returncode != 0 or not evidence.is_file():
                raise RuntimeError(f"{arm} increment {increment} failed")
            payload = read(evidence)
            manifest = payload.get("artifact_manifest") or {}
            if payload.get("training_record_count") != 2 or manifest.get("component_count") != 9 + increment:
                raise RuntimeError(f"{arm} increment {increment} artifact contract mismatch")
            prior_adapter = str(payload["adapter_uri"]).removeprefix("file://")
            prior_id = training_artifact_id(payload)
            arm_state["attempts"].append({"increment": increment, "gpu_uuid": uuid, "return_code": completed.returncode, "receipt": str(receipt), "receipt_sha256": sha256(receipt), "gpu_hours": accounting["gpu_hours"], "evidence": str(evidence), "evidence_sha256": sha256(evidence), "artifact_id": prior_id, "adapter_uri": payload["adapter_uri"]})
            atomic_write(scheduler_path, scheduler)
        arm_state["status"] = "pass"
        atomic_write(scheduler_path, scheduler)
    scheduler["status"] = "pass"
    scheduler["formal_task_consumption"] = 0
    scheduler["t2_panel_evaluated"] = False
    scheduler["training_gpu_hours"] = sum(row["gpu_hours"] for arm in scheduler["arms"].values() for row in arm["attempts"])
    atomic_write(run_root / "scheduler-receipt.json", scheduler)
    return {"status": "pass", "campaign_id": CAMPAIGN_ID, "training_gpu_hours": scheduler["training_gpu_hours"], "formal_task_consumption": 0, "t2_panel_evaluated": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--e2-run-root", type=Path, default=Path("/workspace/runs/phase-h140-g2-r2-20260820/formal-e2-harvest-retry1"))
    parser.add_argument("--patched-receipt", type=Path, default=Path("/workspace/runs/phase-h141-magnitude-screen-20260821/receipts/patched-core.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        config, p = validate(args.root.resolve(), args.e2_run_root.resolve(), args.patched_receipt.resolve())
        print(json.dumps({"status": "dry-run", "campaign_id": config["campaign_id"], "run_root": p["run_root"], "formal_task_consumption": 0}, sort_keys=True))
        return 0
    print(json.dumps(train(args), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
