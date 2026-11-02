#!/usr/bin/env python3
"""Run the frozen H1.40-G2 eight-increment SD-LoRA update sequentially."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.openevo_webshop.gpu_resource_wrapper import account_receipt
from scripts.openevo_webshop.launch_h140_r2_e2_harvest import JAVA_HOME, PYTHON, contract_paths, load_json, sha256
from scripts.openevo_webshop.prepare_h140_g2_training_inputs import prepare


CAMPAIGN_ID = "20260820-0140r2-h140-g2-multigeneration"
CONFIG = Path("configs/experiment/h1.40-r2-g2-multigeneration-v1.json")
PATH_CONTRACT = Path("configs/experiment/h1.40-r2-path-contract-v1.json")
E2_RECONCILIATION = Path("docs/evidence/remote-runs/2026-08-20/h1.40-g2-r2/e2-harvest-reconciliation-v1.json")
RUNNER = Path("scripts/openevo_webshop/h1_closed_loop_tracked.py")
WRAPPER = Path("scripts/openevo_webshop/gpu_resource_wrapper.py")
STARTUP_GATE = Path("scripts/openevo_webshop/startup_gate_5090.sh")
IMAGE = "experiment-host/openevo-scientific:qwen25-3b-webshop-20260814"
IMAGE_DIGEST = "sha256:ce92e587e7c934827025c001fd9b70ec4cb6de2deacfdf9f0cc0777d3ab928f1"
MODEL_REVISION = "0c32faf81d8f2f83cf79860d2a48111f166348d6dd38cd348576d8b3a0141c1e"
OPEN_EVO_SHA = "796e6248a61112a61c880885844d3b0420c0b6ae"
G1_ADAPTER_FILE_SHA = "47e63d417f85cb2defed5ff1ee934128b96e6a4f78a8f410fa052957d919190c"
G1_ADAPTER_CONTAINER = "/workspace/runs/phase-h1.38b-method-control-v1-20260820-correction-02/artifacts/sd-lora/workers/sd-lora-b2e7e1aae2fd0f37a79073e9148da390/adapter"
CONTAINER_RUN_ROOT = "/workspace/runs/phase-h140-g2-r2-20260820"
TRAINER_PYTHON_CONTAINER = "/workspace/src/openevo-experiment/scripts/openevo_webshop/server_trainer_python.sh"


def atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def container_path(host_path: Path, host_run_root: Path) -> str:
    try:
        relative = host_path.resolve().relative_to(host_run_root.resolve())
    except ValueError as exc:
        raise ValueError(f"G2 path escapes the path-contract run root: {host_path}") from exc
    return f"{CONTAINER_RUN_ROOT}/{relative.as_posix()}"


def bind_gpu_uuid(command: list[str], uuid: str) -> list[str]:
    return [value.replace("GPU_PLACEHOLDER", uuid) for value in command]


def run_gate(root: Path, uuid: str, scratch: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update({
        "OPENEVO_AUTHORIZED_GPU_UUIDS": uuid,
        "OPENEVO_SELECTED_GPU_UUIDS": uuid,
        "OPENEVO_SCRATCH_ROOT": str(scratch),
        "JAVA_HOME": JAVA_HOME,
    })
    return subprocess.run(["bash", str(root / STARTUP_GATE), "--", "true"], cwd=root, env=env, capture_output=True, text=True, check=False)


def validate_contract(root: Path, e2_root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    config = load_json(root / CONFIG)
    current = load_json(root / "configs/experiment/current-campaign.json")
    latest = current.get("latest_event") or {}
    if current.get("campaign_id") != CAMPAIGN_ID or current.get("status") != "launch-authorized" or current.get("formal_task_consumption_allowed") is not True:
        raise ValueError("R2 router is not open for the next formal stage")
    if latest.get("classification") != "r2-e2-harvest-selection-passed" or latest.get("g2_adapter_created") or latest.get("t2_panel_evaluated"):
        raise ValueError("G2 training requires the passing E2 selection handoff and no existing G2 adapter")
    if latest.get("formal_task_consumption") != 256 or latest.get("formal_gpu_hours_consumed") != 2.185317164430954:
        raise ValueError("current R2 denominator does not match the reconciled E2 harvest")
    if sha256(root / E2_RECONCILIATION) != latest.get("e2_harvest_reconciliation_sha256"):
        raise ValueError("E2 reconciliation hash changed after handoff")
    reconciliation = load_json(root / E2_RECONCILIATION)
    if reconciliation.get("status") != "pass" or reconciliation.get("selection", {}).get("record_count") != 16 or reconciliation.get("resource_accounting", {}).get("resource_unknown_count") != 0:
        raise ValueError("E2 reconciliation is not a complete passing selection")
    training = config.get("frozen_scientific_contract", {}).get("training") or {}
    expected = {"increments": 8, "records_per_increment": 2, "total_records": 16, "learning_rate": 0.0002, "coefficient_learning_rate": 0.01, "rank": 4, "max_steps_per_increment": 16, "replay_capacity": 64, "batch_size": 1, "gradient_accumulation_steps": 1, "max_length": 512, "max_grad_norm": 1.0, "seed": 1993, "dtype": "bfloat16"}
    if any(training.get(key) != value for key, value in expected.items()):
        raise ValueError("frozen G2 training contract mismatch")
    if config.get("frozen_scientific_contract", {}).get("g2", {}).get("status") != "not-created":
        raise ValueError("G2 adapter is already recorded as created")
    expected_runner_sha = config["frozen_scientific_contract"]["success_qualification"]["source_code_sha256"]
    if sha256(root / RUNNER) != expected_runner_sha:
        raise ValueError("closed-loop runner source hash differs from the frozen success-qualification contract")
    if subprocess.run(["git", "diff", "--quiet"], cwd=root, check=False).returncode != 0:
        raise ValueError("experiment checkout is dirty")
    paths = contract_paths(root)
    adapter_model = Path(paths["adapter"]) / "adapter_model.safetensors"
    if sha256(adapter_model) != G1_ADAPTER_FILE_SHA:
        raise ValueError("G1 adapter weights differ from the frozen adapter file")
    return config, paths


def docker_command(paths: dict[str, str], run_root: Path, input_manifest: Path, artifact_root: Path, evidence: Path, run_id: str, increment: int, prior_adapter: str, prior_artifact_id: str, wandb_mode: str, root: Path, experiment_sha: str) -> list[str]:
    repo_host = paths["experiment"]
    core_host = paths["openevo"]
    model_host = paths["model"]
    cache_host = paths["cache"]
    run_host = paths["run_root"]
    command = [
        "docker", "run", "--rm", "--init", "--name", f"h140-r2-g2-train-{increment:02d}",
        "--read-only", "--tmpfs", "/tmp:rw,nosuid,size=4g", "--shm-size=64g", "--pids-limit", "4096",
        "--gpus", "device=GPU_PLACEHOLDER", "--user", "1001:1001",
        "-v", f"{repo_host}:/workspace/src/openevo-experiment:ro",
        "-v", f"{core_host}:/workspace/src/openevo-core:ro",
        "-v", f"{core_host}:/workspace/openevo-core:ro",
        "-v", f"{model_host}:/workspace/models/qwen2.5-7b-instruct:ro",
        "-v", f"{model_host}:/workspace/hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/{MODEL_REVISION}:ro",
        "-v", f"{cache_host}:/workspace/cache:rw",
        "-v", f"{run_host}:/workspace/runs/phase-h140-g2-r2-20260820:rw",
        "-v", "/workspace/openevo-workspace:/workspace/openevo-workspace:ro",
        "-v", f"{paths['adapter']}:{container_path(artifact_root / 'prior-g1', run_root)}:ro",
        "-w", "/workspace/src/openevo-experiment",
        "-e", "CUDA_VISIBLE_DEVICES=GPU_PLACEHOLDER", "-e", "TOKENIZERS_PARALLELISM=false", "-e", "TRANSFORMERS_OFFLINE=1", "-e", "HF_HUB_OFFLINE=1",
        "-e", f"JAVA_HOME={JAVA_HOME}", "-e", "OPENEVO_TRAINER_CORE=/workspace/src/openevo-core", "-e", "HF_HOME=/workspace/hf-cache", "-e", "HF_HUB_CACHE=/workspace/hf-cache/hub", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "HOME=/tmp", "-e", "TMPDIR=/tmp", "-e", "XDG_CACHE_HOME=/workspace/cache",
        "-e", "PYTHONPATH=/workspace/src/openevo-core/src:/workspace/src/openevo-experiment:/workspace/src/openevo-experiment/scripts/openevo_webshop",
        "--entrypoint", PYTHON, IMAGE, "-m", "scripts.openevo_webshop.run_parametric_smoke_tracked",
        "--dataset-manifest", container_path(input_manifest, run_root), "--dataset-artifact-id", f"h140-r2-g2-increment-{increment:02d}",
        "--artifact-root", container_path(artifact_root, run_root), "--evidence", container_path(evidence, run_root), "--run-id", run_id,
        "--benchmark", "webshop", "--python", TRAINER_PYTHON_CONTAINER, "--base-model", "Qwen/Qwen2.5-7B-Instruct", "--model-revision", MODEL_REVISION,
        "--max-steps", "16", "--max-records", "2", "--minimum-reward", "0", "--experiment-git-sha", experiment_sha,
        "--upstream-openevo-sha", OPEN_EVO_SHA, "--runtime-fingerprint", "495e06379dbc4eadfd6a1941182d1ec8158691febd190e7741763eeed192c672", "--image-digest", IMAGE_DIGEST,
        "--gpu-uuid", "GPU_PLACEHOLDER", "--wandb-mode", wandb_mode, "--wandb-project", "openevo-webshop-experiments", "--wandb-entity", "example-entity",
        "--campaign-id", CAMPAIGN_ID, "--wandb-group", CAMPAIGN_ID, "--wandb-dir", f"{container_path(artifact_root, run_root)}/wandb",
    ]
    if prior_adapter:
        command.extend(["--prior-adapter", prior_adapter, "--prior-artifact-id", prior_artifact_id])
    return command


def train(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    _, paths = validate_contract(root, args.e2_run_root.resolve())
    training_root = args.run_root.resolve()
    if training_root.exists() and any(training_root.iterdir()):
        raise ValueError(f"G2 training run root is not empty: {training_root}")
    training_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    input_root = training_root / "inputs"
    selection_json = args.e2_run_root / "e2-selection-reconciled.json"
    harvest_jsonl = args.e2_run_root / "e2-harvest-attempts-reconciled.jsonl"
    run_mount_root = Path(paths["run_root"])
    inputs = prepare(
        harvest_jsonl,
        selection_json,
        input_root,
        records_uri_resolver=lambda path: Path(container_path(path, run_mount_root)).as_uri(),
    )
    for name in ("receipts", "logs", "evidence", "artifacts"):
        (training_root / name).mkdir(mode=0o700, exist_ok=True)
    output_root = training_root / "artifacts" / "sd-lora"
    prior_mount = output_root / "prior-g1"
    prior_mount.mkdir(mode=0o700, parents=True, exist_ok=True)
    subprocess.run(["chown", "-R", "1001:1001", str(training_root)], check=True)
    experiment_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    scheduler: dict[str, Any] = {"schema": "openevo.webshop-h1.40-g2-training-scheduler-receipt.v1", "campaign_id": CAMPAIGN_ID, "experiment_sha": experiment_sha, "open_evo_sha": OPEN_EVO_SHA, "training_contract": inputs, "attempts": []}
    atomic_write(training_root / "scheduler-receipt.json", scheduler)
    prior_adapter = container_path(prior_mount, run_mount_root)
    prior_artifact_id = "h1.38b-g1-sd-lora-final"
    for increment in range(8):
        item = inputs["increments"][increment]
        evidence = training_root / "evidence" / f"g2-increment-{increment:02d}.json"
        receipt = training_root / "receipts" / f"g2-increment-{increment:02d}.json"
        log_path = training_root / "logs" / f"g2-increment-{increment:02d}.log"
        uuid = next((value for value in load_json(root / "configs/experiment/current-campaign.json")["authorized_gpu_uuids"] if run_gate(root, value, training_root / "scratch" / value).returncode == 0), None)
        if uuid is None:
            raise RuntimeError("no authorized GPU passed the immediate live-idle gate for G2 training")
        command = docker_command(paths, Path(paths["run_root"]), Path(item["dataset_manifest"]), output_root, evidence, f"h140-r2-g2-increment-{increment:02d}", increment, prior_adapter, prior_artifact_id, "online" if increment == 0 else "offline", root, experiment_sha)
        command = bind_gpu_uuid(command, uuid)
        wrapper = [PYTHON, str(root / WRAPPER), "--receipt", str(receipt), "--attempt-id", f"g2-increment-{increment:02d}", "--campaign-id", CAMPAIGN_ID, "--stage", "training", "--command-identity", "h140-r2-g2-training-v1", "--gpu-uuids", uuid, "--path-contract", str(root / PATH_CONTRACT), "--", *command]
        env = os.environ.copy(); env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(wrapper, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=False, text=True)
        if not receipt.is_file():
            raise RuntimeError(f"G2 increment {increment} exited without a resource receipt")
        receipt_value = json.loads(receipt.read_text(encoding="utf-8"))
        accounting = account_receipt(receipt_value)
        if accounting.get("status") != "exact":
            raise RuntimeError(f"G2 increment {increment} resource receipt is not exact")
        if completed.returncode != 0 or not evidence.is_file():
            raise RuntimeError(f"G2 increment {increment} failed before a passing training evidence")
        payload = json.loads(evidence.read_text(encoding="utf-8"))
        manifest = payload.get("artifact_manifest") or {}
        if payload.get("schema") != "seed3090.parametric_smoke.v1" or payload.get("training_record_count") != 2 or not isinstance(payload.get("adapter_uri"), str) or manifest.get("effective_rank") != 4 * (increment + 9):
            raise RuntimeError(f"G2 increment {increment} artifact identity/record contract mismatch")
        artifact = payload.get("artifact") or {}
        artifact_id = str(artifact.get("artifact_id") or artifact.get("name") or "")
        if not artifact_id:
            raise RuntimeError(f"G2 increment {increment} did not return an immutable artifact id")
        row = {"increment_index": increment, "gpu_uuid": uuid, "return_code": completed.returncode, "receipt": str(receipt), "receipt_sha256": sha256(receipt), "resource_accounting": accounting, "evidence": str(evidence), "evidence_sha256": sha256(evidence), "adapter_uri": payload["adapter_uri"], "artifact_id": artifact_id, "adapter_manifest": {"component_count": manifest.get("component_count"), "effective_rank": manifest.get("effective_rank"), "state_weights_sha256": manifest.get("state_weights_sha256"), "peft_weights_sha256": manifest.get("peft_weights_sha256")}}
        scheduler["attempts"].append(row); atomic_write(training_root / "scheduler-receipt.json", scheduler)
        prior_adapter = str(payload["adapter_uri"]).removeprefix("file://")
        prior_artifact_id = artifact_id
    total = sum(float(row["resource_accounting"]["gpu_hours"]) for row in scheduler["attempts"])
    result = {"schema": "openevo.webshop-h1.40-g2-training-reconciliation.v1", "status": "pass", "campaign_id": CAMPAIGN_ID, "increment_count": 8, "training_record_count": 16, "training_gpu_hours": total, "resource_unknown_count": 0, "scheduler_receipt": str(training_root / "scheduler-receipt.json"), "scheduler_receipt_sha256": sha256(training_root / "scheduler-receipt.json"), "final_evidence": scheduler["attempts"][-1]["evidence"], "final_adapter_uri": scheduler["attempts"][-1]["adapter_uri"], "g2_adapter_created": True, "t2_panel_evaluated": False, "claim_boundary": "G2 adapter training passed structurally; acquisition, retention, preservation, T2, and multi-generation claims remain unevaluated."}
    atomic_write(training_root / "g2-training-reconciliation.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--e2-run-root", type=Path, default=Path("/workspace/runs/phase-h140-g2-r2-20260820/formal-e2-harvest-retry1"))
    parser.add_argument("--run-root", type=Path, default=Path("/workspace/runs/phase-h140-g2-r2-20260820/formal-g2-training"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        config, _ = validate_contract(args.root.resolve(), args.e2_run_root.resolve())
        print(json.dumps({"status": "dry-run", "increment_count": config["frozen_scientific_contract"]["training"]["increments"], "record_count": config["frozen_scientific_contract"]["training"]["total_records"]}, sort_keys=True))
        return 0
    print(json.dumps(train(args), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
