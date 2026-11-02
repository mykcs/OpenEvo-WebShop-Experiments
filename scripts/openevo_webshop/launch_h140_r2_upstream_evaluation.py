#!/usr/bin/env python3
"""Run the frozen R2 G1-vs-G2 acquisition/retention/preservation gates.

This launcher consumes no T2 tasks. Each one-task/four-seed cell is wrapped
outside Docker so a failed payload still has an exact resource receipt.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.openevo_webshop.gpu_resource_wrapper import account_receipt
from scripts.openevo_webshop.path_contract import load_and_validate
from scripts.openevo_webshop.validate_h140_r2_preregistration import validate as validate_prereg


CAMPAIGN_ID = "20260820-0140r2-h140-g2-multigeneration"
CONFIG = Path("configs/experiment/h1.40-r2-g2-multigeneration-v1.json")
PATH_CONTRACT = Path("configs/experiment/h1.40-r2-path-contract-v1.json")
E2_RECONCILIATION = Path("docs/evidence/remote-runs/2026-08-20/h1.40-g2-r2/e2-harvest-reconciliation-v1.json")
WRAPPER = Path("scripts/openevo_webshop/gpu_resource_wrapper.py")
RUNNER = Path("scripts/openevo_webshop/h1_closed_loop_tracked.py")
STARTUP_GATE = Path("scripts/openevo_webshop/startup_gate_5090.sh")
IMAGE = "experiment-host/openevo-scientific:qwen25-3b-webshop-20260814"
IMAGE_DIGEST = "sha256:ce92e587e7c934827025c001fd9b70ec4cb6de2deacfdf9f0cc0777d3ab928f1"
MODEL_REVISION = "0c32faf81d8f2f83cf79860d2a48111f166348d6dd38cd348576d8b3a0141c1e"
OPEN_EVO_SHA = "796e6248a61112a61c880885844d3b0420c0b6ae"
G1_FILE_SHA = "47e63d417f85cb2defed5ff1ee934128b96e6a4f78a8f410fa052957d919190c"
G1_TENSOR_DIGEST = "eedc36232f04a8a7d6e8fc6a38b935758db65206f4945a6cfa35c4ab666a48f9"
PYTHON = "/workspace/openevo-workspace/runtime/phase-g-live-20260814/env/bin/python"
JAVA_HOME = "/workspace/openevo-workspace/runtime/phase-g-live-20260814/tools/jdk-11.0.20.1"
RUNTIME_FINGERPRINT = "495e06379dbc4eadfd6a1941182d1ec8158691febd190e7741763eeed192c672"
SEED_ROOT = "/workspace/runs/phase-h1.20-action-admissibility-diagnostic-v1-20260815/seed-roots/base-P1"
GPUS = (
    "GPU-00000000-0000-0000-0000-000000000000",
    "GPU-00000000-0000-0000-0000-000000000000",
    "GPU-00000000-0000-0000-0000-000000000000",
    "GPU-00000000-0000-0000-0000-000000000000",
    "GPU-00000000-0000-0000-0000-000000000000",
    "GPU-00000000-0000-0000-0000-000000000000",
)
STAGE_MANIFESTS = {
    "acquisition": Path("configs/experiment/manifests/h1.40-g2-e2-reservoir-v1.json"),
    "retention": Path("configs/experiment/manifests/h1.40-g2-e1-retention-set-v1.json"),
    "preservation": Path("configs/experiment/manifests/h1.40-g2-h139-preservation-anchor-v1.json"),
}
STAGE_COUNTS = {"acquisition": 8, "retention": 8, "preservation": 32}
RUNNER_EVALUATION_STAGES = {"acquisition": "acquisition", "retention": "retention", "preservation": "fresh"}
SEEDS = (2026104100, 2026104101, 2026104102, 2026104103)
PAYLOAD_UID_GID = "1001:1001"
LIVE_GATE_RETRIES = 12
LIVE_GATE_DELAY_SECONDS = 5.0


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_payload_directory(path: Path) -> None:
    """Create a host directory writable by the isolated Docker payload UID."""
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    subprocess.run(["chown", "-R", PAYLOAD_UID_GID, str(path)], check=True)


def container_path(host: Path, run_root: Path, campaign_run_root: Path) -> str:
    run_root_relative = run_root.resolve().relative_to(campaign_run_root.resolve())
    relative = host.resolve().relative_to(run_root.resolve())
    parts = ["/workspace/runs/phase-h140-g2-r2-20260820", *run_root_relative.parts, *relative.parts]
    return "/" + "/".join(part.strip("/") for part in parts if part not in ("", "."))


def selected_e2_rows(root: Path) -> list[dict[str, Any]]:
    selected = read(root / E2_RECONCILIATION)["selection"]["selected_task_ids"]
    rows = read(root / STAGE_MANIFESTS["acquisition"])["tasks"]
    by_id = {row["task_id"]: row for row in rows}
    if [task_id for task_id in selected if task_id not in by_id] != []:
        raise ValueError("E2 selection references an unknown task identity")
    result = [by_id[task_id] for task_id in selected]
    if len(result) != 8:
        raise ValueError("E2 acquisition must use exactly eight selected identities")
    return result


def stage_rows(root: Path, stage: str) -> list[dict[str, Any]]:
    if stage == "acquisition":
        rows = selected_e2_rows(root)
    else:
        rows = read(root / STAGE_MANIFESTS[stage])["tasks"]
    if len(rows) != STAGE_COUNTS[stage] or len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError(f"{stage} task identity count/order mismatch")
    return rows


def validate(root: Path, g2_adapter: Path, g2_tensor_digest: str) -> dict[str, str]:
    validate_prereg(root)
    current = read(root / "configs/experiment/current-campaign.json")
    latest = current.get("latest_event") or {}
    if latest.get("classification") != "r2-g2-training-passed" or current.get("status") != "launch-authorized":
        raise ValueError("R2 router is not at the upstream G1-vs-G2 evaluation gate")
    if latest.get("formal_task_consumption") != 256 or not latest.get("g2_adapter_created") or latest.get("t2_panel_evaluated"):
        raise ValueError("R2 router state does not match the frozen upstream gate")
    if len(g2_tensor_digest) != 64 or any(char not in "0123456789abcdef" for char in g2_tensor_digest):
        raise ValueError("G2 tensor digest must be a lowercase SHA-256 hex string")
    paths = read(root / PATH_CONTRACT)["paths"]
    g1 = Path(paths["adapter"]["host_path"])
    if sha(g1 / "adapter_model.safetensors") != G1_FILE_SHA:
        raise ValueError("G1 adapter file identity mismatch")
    if not g2_adapter.is_dir() or sha(g2_adapter / "adapter_model.safetensors") == "":
        raise ValueError("G2 adapter directory is missing or unreadable")
    if not (root / RUNNER).is_file() or not (root / CONFIG).is_file():
        raise ValueError("frozen R2 runtime files are missing")
    return {"experiment": str(paths["experiment"]["host_path"]), "core": str(paths["openevo"]["host_path"]), "model": str(paths["model"]["host_path"]), "dataset": str(paths["dataset"]["host_path"]), "cache": str(paths["cache"]["host_path"]), "run_root": str(paths["run_root"]["host_path"]), "g1": str(g1)}


def run_gate(root: Path, uuid: str, scratch: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update({"OPENEVO_AUTHORIZED_GPU_UUIDS": uuid, "OPENEVO_SELECTED_GPU_UUIDS": uuid, "OPENEVO_SCRATCH_ROOT": str(scratch), "JAVA_HOME": JAVA_HOME})
    return subprocess.run(["bash", str(root / STARTUP_GATE), "--", "true"], cwd=root, env=env, capture_output=True, text=True, check=False)


def wait_for_live_gate(root: Path, uuid: str, scratch: Path) -> subprocess.CompletedProcess[str]:
    last = run_gate(root, uuid, scratch)
    for _ in range(LIVE_GATE_RETRIES - 1):
        if last.returncode == 0:
            return last
        detail = f"{last.stdout}\n{last.stderr}"
        if not any(token in detail for token in ("not genuinely idle/usable", "memory.used=", "utilization=", "NVML")):
            return last
        time.sleep(LIVE_GATE_DELAY_SECONDS)
        last = run_gate(root, uuid, scratch)
    return last


def docker_command(paths: dict[str, str], run_root: Path, row: dict[str, Any], stage: str, arm: str, uuid: str, output: Path, g2_adapter: Path, g2_tensor_digest: str, experiment_sha: str) -> list[str]:
    adapter_host = Path(paths["g1"]) if arm == "g1" else g2_adapter
    adapter_container = "/workspace/runs/phase-h1.38b-method-control-v1-20260820-correction-02/artifacts/sd-lora/workers/sd-lora-b2e7e1aae2fd0f37a79073e9148da390/adapter" if arm == "g1" else "/workspace/g2-final-adapter"
    run_host = Path(paths["run_root"])
    try:
        output_container = container_path(output, run_root, run_host)
    except ValueError as exc:
        raise ValueError(f"evaluation output is outside the run root: {output}") from exc
    manifest_container = f"/workspace/src/openevo-experiment/{STAGE_MANIFESTS[stage].as_posix()}"
    return [
        "docker", "run", "--rm", "--init", "--name", f"h140-r2-{stage}-{arm}-{row['task_id']}",
        "--read-only", "--tmpfs", "/tmp:rw,nosuid,size=4g", "--mount", "type=tmpfs,destination=/root,tmpfs-mode=0755", "--shm-size=64g", "--pids-limit", "4096", "--gpus", f"device={uuid}", "--user", "1001:1001",
        "-v", f"{paths['experiment']}:/workspace/src/openevo-experiment:ro", "-v", f"{paths['core']}:/workspace/src/openevo-core:ro", "-v", f"{paths['model']}:/workspace/models/qwen2.5-7b-instruct:ro", "-v", f"{paths['dataset']}:/workspace/data/webshop-1000:ro", "-v", f"{paths['cache']}:/workspace/cache:rw", "-v", f"{run_host}:/workspace/runs/phase-h140-g2-r2-20260820:rw", "-v", "/workspace/openevo-workspace:/workspace/openevo-workspace:ro", "-v", f"{adapter_host}:{adapter_container}:ro", "-v", f"{SEED_ROOT}:/workspace/runs/phase-h1.20-action-admissibility-diagnostic-v1-20260815/seed-roots/base-P1:ro",
        "-w", "/workspace/src/openevo-experiment", "-e", f"CUDA_VISIBLE_DEVICES={uuid}", "-e", "TOKENIZERS_PARALLELISM=false", "-e", "TRANSFORMERS_OFFLINE=1", "-e", "HF_HUB_OFFLINE=1", "-e", "WANDB_ENTITY=example-entity", "-e", "WANDB_PROJECT=openevo-webshop-experiments", "-e", "JAVA_HOME=" + JAVA_HOME, "-e", "PYTHONPATH=/workspace/src/openevo-core/src:/workspace/src/openevo-experiment:/workspace/src/openevo-experiment/scripts/openevo_webshop:/workspace/src/openevo-experiment/vendor/openevo-webshop/src", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "HOME=/tmp", "-e", "TMPDIR=/tmp", "-e", "XDG_CACHE_HOME=/workspace/cache",
        "--entrypoint", PYTHON, IMAGE, "-m", "scripts.openevo_webshop.h1_closed_loop_tracked", "--task-manifest", manifest_container, "--seed-root", "/workspace/runs/phase-h1.20-action-admissibility-diagnostic-v1-20260815/seed-roots/base-P1", "--model", "/workspace/models/qwen2.5-7b-instruct", "--model-revision", MODEL_REVISION, "--adapter", adapter_container, "--expected-adapter-file-sha256", G1_FILE_SHA if arm == "g1" else sha(g2_adapter / "adapter_model.safetensors"), "--expected-adapter-tensor-digest", G1_TENSOR_DIGEST if arm == "g1" else g2_tensor_digest, "--arm", "sd-lora", "--tasks", str(row["goal_idx"]), "--rollout-seeds", *(str(seed) for seed in SEEDS), "--output", output_container, "--gpu-uuid", uuid, "--experiment-git-sha", experiment_sha, "--upstream-openevo-sha", OPEN_EVO_SHA, "--runtime-fingerprint", RUNTIME_FINGERPRINT, "--image-digest", IMAGE_DIGEST, "--temperature", "0.0", "--max-steps", "30", "--max-new", "512", "--prompt-contract", "seed3090.webshop-compact-action-prompt.v2", "--action-contract", "seed3090.webshop-action-parser.v3-action-command", "--evaluation-stage", RUNNER_EVALUATION_STAGES[stage], "--wandb-mode", "offline", "--wandb-project", "openevo-webshop-experiments", "--wandb-entity", "example-entity", "--campaign-id", CAMPAIGN_ID, "--wandb-group", CAMPAIGN_ID, "--wandb-dir", f"{output_container}/wandb",
    ]


def run_cell(root: Path, paths: dict[str, str], run_root: Path, row: dict[str, Any], stage: str, arm: str, uuid: str, g2_adapter: Path, g2_tensor_digest: str, experiment_sha: str, scheduler: dict[str, Any], lock: threading.Lock, stop: threading.Event) -> None:
    cell_id = f"{stage}-{arm}-{row['task_id']}"
    output = run_root / "cells" / stage / arm / row["task_id"]
    receipt = run_root / "receipts" / f"{cell_id}.json"
    log = run_root / "logs" / f"{cell_id}.log"
    prepare_payload_directory(output)
    receipt.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    log.parent.mkdir(mode=0o700, exist_ok=True)
    gate = wait_for_live_gate(root, uuid, run_root / "scratch" / uuid)
    if gate.returncode != 0:
        stop.set()
        raise RuntimeError(f"{cell_id}: live-idle gate failed: {gate.stderr[-500:]}")
    command = docker_command(paths, run_root, row, stage, arm, uuid, output, g2_adapter, g2_tensor_digest, experiment_sha)
    wrapper = [PYTHON, str(root / WRAPPER), "--receipt", str(receipt), "--attempt-id", cell_id, "--campaign-id", CAMPAIGN_ID, "--stage", "evaluation", "--command-identity", f"h140-r2-{stage}-g1-vs-g2-v1", "--gpu-uuids", uuid, "--path-contract", str(root / PATH_CONTRACT), "--", *command]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    completed = subprocess.run(wrapper, cwd=root, env=env, stdout=log.open("w", encoding="utf-8"), stderr=subprocess.STDOUT, check=False, text=True)
    row_result = {"cell_id": cell_id, "stage": stage, "arm": arm, "task_id": row["task_id"], "goal_idx": row["goal_idx"], "gpu_uuid": uuid, "return_code": completed.returncode, "receipt": str(receipt), "resource_accounting": account_receipt(json.loads(receipt.read_text(encoding="utf-8")) if receipt.is_file() else {})}
    with lock:
        scheduler["cells"].append(row_result)
        atomic_write(run_root / "scheduler-receipt.json", scheduler)
    if not receipt.is_file():
        stop.set()
        raise RuntimeError(f"{cell_id}: wrapper did not produce a receipt")


def execute(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    g2_adapter = args.g2_adapter.resolve()
    paths = validate(root, g2_adapter, args.g2_tensor_digest)
    load_and_validate(root / PATH_CONTRACT)
    run_root = args.run_root.resolve()
    if run_root.exists() and any(run_root.iterdir()):
        raise ValueError(f"upstream evaluation run root is not empty: {run_root}")
    prepare_payload_directory(run_root)
    stages = [args.stage] if args.stage != "all" else ["acquisition", "retention", "preservation"]
    cells = [(stage, arm, row) for stage in stages for arm in ("g1", "g2") for row in stage_rows(root, stage)]
    experiment_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    scheduler: dict[str, Any] = {"schema": "openevo.webshop-h1.40-r2-upstream-evaluation-scheduler.v1", "status": "running", "campaign_id": CAMPAIGN_ID, "experiment_git_sha": experiment_sha, "g2_adapter": str(g2_adapter), "adapter_identities": {"g1": {"adapter_file_sha256": G1_FILE_SHA, "tensor_digest": G1_TENSOR_DIGEST}, "g2": {"adapter_file_sha256": sha(g2_adapter / "adapter_model.safetensors"), "tensor_digest": args.g2_tensor_digest}}, "stages": stages, "expected_cell_count": len(cells), "cells": []}
    atomic_write(run_root / "scheduler-receipt.json", scheduler)
    lock = threading.Lock()
    stop = threading.Event()
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        for batch_start in range(0, len(cells), len(GPUS)):
            if stop.is_set():
                break
            batch = cells[batch_start : batch_start + len(GPUS)]
            futures = [
                pool.submit(
                    run_cell,
                    root,
                    paths,
                    run_root,
                    row,
                    stage,
                    arm,
                    GPUS[offset],
                    g2_adapter,
                    args.g2_tensor_digest,
                    experiment_sha,
                    scheduler,
                    lock,
                    stop,
                )
                for offset, (stage, arm, row) in enumerate(batch)
            ]
            for future in futures:
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001
                    failures.append(str(exc))
                    stop.set()
                    break
    scheduler["status"] = "pass" if not failures and len(scheduler["cells"]) == len(cells) else "incomplete"
    scheduler["failures"] = failures
    scheduler["completed_cell_count"] = len(scheduler["cells"])
    atomic_write(run_root / "scheduler-receipt.json", scheduler)
    if failures:
        raise RuntimeError("; ".join(failures))
    return {"status": "pass", "stage_count": len(stages), "cell_count": len(cells), "attempt_count": len(cells) * 4, "run_root": str(run_root)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--stage", choices=("acquisition", "retention", "preservation", "all"), default="all")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--g2-adapter", type=Path, required=True)
    parser.add_argument("--g2-tensor-digest", required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
