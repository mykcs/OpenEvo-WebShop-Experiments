#!/usr/bin/env python3
"""Execute the frozen H1.40-G2-R2 endogenous E2 harvest.

This is the formal-stage transport/orchestration entry point. Scientific
semantics remain in the frozen R2 config and ``h1_closed_loop_tracked.py``;
every GPU-bearing task/seed attempt is wrapped outside Docker so its resource
receipt is created before allocation and finalized even when the payload fails.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import queue
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


CAMPAIGN_ID = "20260820-0140r2-h140-g2-multigeneration"
SUCCESSOR_CAMPAIGN_ID = "20260821-0141-h141-magnitude-screen"
CONFIG = Path("configs/experiment/h1.40-r2-g2-multigeneration-v1.json")
E2_MANIFEST = Path("configs/experiment/manifests/h1.40-g2-e2-reservoir-v1.json")
PATH_CONTRACT = Path("configs/experiment/h1.40-r2-path-contract-v1.json")
RELEASE = Path("configs/experiment/activations/h1.40-r2-g2-formal-execution-release-v1.json")
WRAPPER = Path("scripts/openevo_webshop/gpu_resource_wrapper.py")
RUNNER = Path("scripts/openevo_webshop/h1_closed_loop_tracked.py")
STARTUP_GATE = Path("scripts/openevo_webshop/startup_gate_5090.sh")
SELECTOR = Path("scripts/openevo_webshop/select_h140_g2_e2.py")
IMAGE = "experiment-host/openevo-scientific:qwen25-3b-webshop-20260814"
IMAGE_DIGEST = "sha256:ce92e587e7c934827025c001fd9b70ec4cb6de2deacfdf9f0cc0777d3ab928f1"
MODEL_REVISION = "0c32faf81d8f2f83cf79860d2a48111f166348d6dd38cd348576d8b3a0141c1e"
OPEN_EVO_SHA = "796e6248a61112a61c880885844d3b0420c0b6ae"
ADAPTER_FILE_SHA = "47e63d417f85cb2defed5ff1ee934128b96e6a4f78a8f410fa052957d919190c"
ADAPTER_TENSOR_DIGEST = "eedc36232f04a8a7d6e8fc6a38b935758db65206f4945a6cfa35c4ab666a48f9"
PYTHON = "/workspace/openevo-workspace/runtime/phase-g-live-20260814/env/bin/python"
JAVA_HOME = "/workspace/openevo-workspace/runtime/phase-g-live-20260814/tools/jdk-11.0.20.1"
RUNTIME_FINGERPRINT = "495e06379dbc4eadfd6a1941182d1ec8158691febd190e7741763eeed192c672"
SEED_ROOT = "/workspace/runs/phase-h1.20-action-admissibility-diagnostic-v1-20260815/seed-roots/base-P1"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
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


def contract_paths(root: Path) -> dict[str, str]:
    paths = load_json(root / PATH_CONTRACT).get("paths")
    if not isinstance(paths, dict):
        raise ValueError("path contract paths are missing")
    return {name: str(spec["host_path"]) for name, spec in paths.items()}


def build_attempts(root: Path) -> list[dict[str, Any]]:
    manifest = load_json(root / E2_MANIFEST)
    tasks = manifest.get("tasks")
    seeds = manifest.get("rollout_seeds")
    if not isinstance(tasks, list) or len(tasks) != 64 or not isinstance(seeds, list) or len(seeds) != 4:
        raise ValueError("E2 manifest must contain exactly 64 tasks and 4 frozen seeds")
    attempts: list[dict[str, Any]] = []
    for task_position, task in enumerate(tasks):
        if not isinstance(task, dict) or not isinstance(task.get("goal_idx"), int) or not isinstance(task.get("task_id"), str):
            raise ValueError(f"invalid E2 task row at position {task_position}")
        for seed_position, seed in enumerate(seeds):
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError("E2 rollout seeds must be integers")
            attempt_id = f"e2-{len(attempts):03d}-{task['task_id']}-s{seed}"
            attempts.append({
                "attempt_index": len(attempts),
                "attempt_id": attempt_id,
                "task_position": task_position,
                "seed_position": seed_position,
                "task_id": task["task_id"],
                "goal_idx": int(task["goal_idx"]),
                "rollout_seed": seed,
            })
    if len(attempts) != 256:
        raise ValueError(f"E2 attempt matrix must contain 256 attempts, found {len(attempts)}")
    return attempts


def validate_release(root: Path, *, allow_completed_router: bool = False) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    config = load_json(root / CONFIG)
    release = load_json(root / RELEASE)
    current = load_json(root / "configs/experiment/current-campaign.json")
    config_sha = sha256(root / CONFIG)
    terminal_router = current.get("campaign_id") == CAMPAIGN_ID and current.get("status") == "closed-upstream-gates-failed"
    successor_router = current.get("campaign_id") == SUCCESSOR_CAMPAIGN_ID and current.get("replacement_of") == CAMPAIGN_ID
    if current.get("campaign_id") not in {CAMPAIGN_ID, SUCCESSOR_CAMPAIGN_ID} or (current.get("status") != "launch-authorized" and not (allow_completed_router and (terminal_router or successor_router))):
        raise ValueError("current router is not launch-authorized R2 or an explicitly allowed completed router")
    current_classification = (current.get("latest_event") or {}).get("classification")
    if current_classification != "r2-formal-execution-release-published" and not (allow_completed_router and (current_classification in {"r2-e2-harvest-selection-passed", "r2-g2-training-passed", "r2-upstream-gates-failed"} or successor_router)):
        raise ValueError("formal-execution release is not the current router event")
    if release.get("status") != "pass" or release.get("formal_task_consumption_allowed") is not True:
        raise ValueError("formal-execution release is not a passing task-consumption receipt")
    if release.get("contract_sha256") != f"sha256:{config_sha}" or release.get("scientific_contract_changed") is not False:
        raise ValueError("formal release does not bind the frozen R2 contract")
    if config.get("campaign_id") != CAMPAIGN_ID or config.get("formal_task_consumption_allowed") is not False:
        raise ValueError("R2 preregistration was edited/opened")
    expected = release.get("manifest_sha256")
    paths = contract_paths(root)
    manifest_paths = load_json(root / CONFIG)["frozen_scientific_contract"]["selection"]["manifests"]
    actual: dict[str, str] = {}
    for name, relative in manifest_paths.items():
        path = root / relative
        actual[name] = sha256(path)
    if expected != actual:
        raise ValueError("formal release manifest hashes do not match the checkout")
    qualification = release.get("qualification") or {}
    if qualification.get("qualification_gpu_hours") != 0.007486578837253748:
        raise ValueError("qualification GPU-hours are not the exact frozen total")
    if release.get("formal_state_before_release") != {"formal_task_consumption": 0, "formal_gpu_hours_consumed": 0.0, "g2_adapter_created": False, "t2_panel_evaluated": False}:
        raise ValueError("formal release did not preserve zero state")
    return config, release, paths


def run_gate(root: Path, uuid: str, scratch: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update({
        "OPENEVO_AUTHORIZED_GPU_UUIDS": uuid,
        "OPENEVO_SELECTED_GPU_UUIDS": uuid,
        "OPENEVO_SCRATCH_ROOT": str(scratch),
        "JAVA_HOME": JAVA_HOME,
    })
    return subprocess.run(["bash", str(root / STARTUP_GATE), "--", "true"], cwd=root, env=env, capture_output=True, text=True, check=False)


def docker_command(paths: dict[str, str], attempt: dict[str, Any], uuid: str, output: Path, run_root: Path, root: Path, experiment_sha: str) -> list[str]:
    repo_host = paths["experiment"]
    core_host = paths["openevo"]
    model_host = paths["model"]
    dataset_host = paths["dataset"]
    cache_host = paths["cache"]
    run_host = paths["run_root"]
    adapter_host = paths["adapter"]
    container_run_root = "/workspace/runs/phase-h140-g2-r2-20260820"
    host_run_root = Path(run_host).resolve()
    try:
        output_relative = output.resolve().relative_to(host_run_root)
    except ValueError as exc:
        raise ValueError(f"formal output is outside the path-contract run root: {output}") from exc
    container_output = f"{container_run_root}/{output_relative.as_posix()}"
    seed_container = "/workspace/runs/phase-h1.20-action-admissibility-diagnostic-v1-20260815/seed-roots/base-P1"
    adapter_container = "/workspace/runs/phase-h1.38b-method-control-v1-20260820-correction-02/artifacts/sd-lora/workers/sd-lora-b2e7e1aae2fd0f37a79073e9148da390/adapter"
    return [
        "docker", "run", "--rm", "--name", f"h140-r2-e2-{attempt['attempt_index']:03d}",
        "--read-only", "--tmpfs", "/tmp:rw,nosuid,size=4g", "--mount", "type=tmpfs,destination=/root,tmpfs-mode=0755",
        "--shm-size=64g", "--pids-limit", "4096", "--gpus", f"device={uuid}", "--user", "1001:1001",
        "-v", f"{repo_host}:/workspace/src/openevo-experiment:ro",
        "-v", f"{core_host}:/workspace/src/openevo-core:ro",
        "-v", f"{model_host}:/workspace/models/qwen2.5-7b-instruct:ro",
        "-v", f"{dataset_host}:/workspace/data/webshop-1000:ro",
        "-v", f"{cache_host}:/workspace/cache:rw",
        "-v", f"{run_host}:/workspace/runs/phase-h140-g2-r2-20260820:rw",
        "-v", "/workspace/openevo-workspace:/workspace/openevo-workspace:ro",
        "-v", f"{adapter_host}:{adapter_container}:ro",
        "-v", f"{SEED_ROOT}:{seed_container}:ro",
        "-w", "/workspace/src/openevo-experiment",
        "-e", f"CUDA_VISIBLE_DEVICES={uuid}", "-e", "TOKENIZERS_PARALLELISM=false", "-e", "TRANSFORMERS_OFFLINE=1", "-e", "HF_HUB_OFFLINE=1",
        "-e", f"WANDB_MODE={attempt['wandb_mode']}", "-e", "WANDB_ENTITY=example-entity", "-e", "WANDB_PROJECT=openevo-webshop-experiments",
        "-e", f"WANDB_RUN_GROUP={CAMPAIGN_ID}", "-e", f"OPENEVO_CAMPAIGN_ID={CAMPAIGN_ID}", "-e", f"OPENEVO_WANDB_MODE={attempt['wandb_mode']}",
        "-e", f"JAVA_HOME={JAVA_HOME}",
        "-e", "PYTHONPATH=/workspace/src/openevo-core/src:/workspace/src/openevo-experiment:/workspace/src/openevo-experiment/scripts/openevo_webshop:/workspace/src/openevo-experiment/vendor/openevo-webshop/src",
        "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "HOME=/tmp", "-e", "TMPDIR=/tmp", "-e", "XDG_CACHE_HOME=/workspace/cache",
        "--entrypoint", PYTHON, IMAGE,
        "-m", "scripts.openevo_webshop.h1_closed_loop_tracked",
        "--task-manifest", "/workspace/src/openevo-experiment/configs/experiment/manifests/h1.40-g2-e2-reservoir-v1.json",
        "--seed-root", seed_container, "--model", "/workspace/models/qwen2.5-7b-instruct", "--model-revision", MODEL_REVISION,
        "--adapter", adapter_container, "--expected-adapter-file-sha256", ADAPTER_FILE_SHA, "--expected-adapter-tensor-digest", ADAPTER_TENSOR_DIGEST,
        "--arm", "self-generated-harvest", "--tasks", str(attempt["goal_idx"]), "--rollout-seeds", str(attempt["rollout_seed"]),
        "--output", container_output, "--gpu-uuid", uuid, "--experiment-git-sha", experiment_sha,
        "--upstream-openevo-sha", OPEN_EVO_SHA, "--runtime-fingerprint", RUNTIME_FINGERPRINT, "--image-digest", IMAGE_DIGEST,
        "--temperature", "0.0", "--max-steps", "30", "--max-new", "512",
        "--prompt-contract", "seed3090.webshop-compact-action-prompt.v2", "--action-contract", "seed3090.webshop-action-parser.v3-action-command",
        "--wandb-mode", attempt["wandb_mode"], "--wandb-project", "openevo-webshop-experiments", "--wandb-entity", "example-entity",
        "--campaign-id", CAMPAIGN_ID, "--wandb-group", CAMPAIGN_ID, "--wandb-dir", f"{container_output}/wandb",
    ]


def fallback_row(attempt: dict[str, Any], receipt: Path, reason: str) -> dict[str, Any]:
    return {
        "schema": "openevo.webshop-h1.40-e2-attempt-fallback.v1",
        "arm": "self-generated-harvest", "task_id": attempt["task_id"], "goal_idx": attempt["goal_idx"],
        "rollout_seed": attempt["rollout_seed"], "qualified_positive": False, "identity_ok": False,
        "trace_complete": False, "parser_or_fallback_failure": False, "error": reason,
        "resource_receipt": str(receipt), "invalid_taxonomy": "runtime",
    }


def run_attempt(root: Path, paths: dict[str, str], run_root: Path, attempt: dict[str, Any], uuid: str, experiment_sha: str, lock: threading.Lock, scheduler: dict[str, Any], gate_timeout: float, stop_event: threading.Event) -> None:
    if stop_event.is_set():
        return
    output = run_root / "attempts" / attempt["attempt_id"]
    receipt = run_root / "receipts" / f"{attempt['attempt_id']}.json"
    log_path = run_root / "logs" / f"{attempt['attempt_id']}.log"
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    receipt.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    log_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    attempt = dict(attempt, gpu_uuid=uuid, wandb_mode="online" if attempt["attempt_index"] == 0 else "offline")
    gate_started = time.monotonic()
    while True:
        gate = run_gate(root, uuid, run_root / "scratch" / uuid)
        if gate.returncode == 0:
            break
        if gate_timeout and time.monotonic() - gate_started >= gate_timeout:
            raise RuntimeError(f"GPU {uuid} did not become idle before the resource-gate timeout")
        time.sleep(30)
    command = docker_command(paths, attempt, uuid, output, run_root, root, experiment_sha)
    wrapper = [
        PYTHON, str(root / WRAPPER), "--receipt", str(receipt), "--attempt-id", attempt["attempt_id"],
        "--campaign-id", CAMPAIGN_ID, "--stage", "harvest", "--command-identity", "h140-r2-e2-harvest-v1",
        "--gpu-uuids", uuid, "--path-contract", str(root / PATH_CONTRACT), "--", *command,
    ]
    wrapper_env = os.environ.copy()
    wrapper_env["PYTHONPATH"] = str(root) + (os.pathsep + wrapper_env["PYTHONPATH"] if wrapper_env.get("PYTHONPATH") else "")
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(wrapper, cwd=root, env=wrapper_env, stdout=log, stderr=subprocess.STDOUT, check=False, text=True)
    if not receipt.is_file():
        row = {
            **attempt,
            "return_code": completed.returncode,
            "receipt": str(receipt),
            "resource_accounting": account_receipt({}),
            "launcher_invocation_status": "failed-before-wrapper-receipt",
            "launcher_error": "gpu_resource_wrapper exited without creating its pre-allocation receipt",
        }
        with lock:
            scheduler["attempts"].append(row)
            scheduler["fatal_error"] = row["launcher_error"]
            atomic_write(run_root / "scheduler-receipt.json", scheduler)
        stop_event.set()
        raise RuntimeError(f"{attempt['attempt_id']}: wrapper exited without creating a receipt")
    resource = json.loads(receipt.read_text(encoding="utf-8"))
    row = {**attempt, "return_code": completed.returncode, "receipt": str(receipt), "resource_accounting": account_receipt(resource)}
    with lock:
        scheduler["attempts"].append(row)
        atomic_write(run_root / "scheduler-receipt.json", scheduler)


def aggregate_and_select(root: Path, run_root: Path, attempts: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(item["task_id"], item["rollout_seed"]): item for item in attempts}
    rows: list[dict[str, Any]] = []
    for spec in sorted(attempts, key=lambda item: item["attempt_index"]):
        attempt_file = run_root / "attempts" / spec["attempt_id"] / "attempts.jsonl"
        parsed: list[dict[str, Any]] = []
        if attempt_file.is_file():
            parsed = [json.loads(line) for line in attempt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        matching = [row for row in parsed if row.get("task_id") == spec["task_id"] and row.get("rollout_seed") == spec["rollout_seed"]]
        rows.append(matching[0] if len(matching) == 1 else fallback_row(spec, Path(spec["receipt"]), "payload attempt row missing or malformed"))
    combined = run_root / "e2-harvest-attempts.jsonl"
    combined.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    selection = run_root / "e2-selection.json"
    subprocess.run([PYTHON, str(root / SELECTOR), "--manifest", str(root / E2_MANIFEST), "--harvest-jsonl", str(combined), "--output", str(selection)], cwd=root, check=True)
    result = load_json(selection)
    resource_unknown = [item for item in attempts if item.get("resource_accounting", {}).get("status") != "exact"]
    reconciliation = {
        "schema": "openevo.webshop-h1.40-g2-e2-harvest-reconciliation.v1",
        "campaign_id": CAMPAIGN_ID, "status": result.get("status"), "attempt_count": len(rows),
        "qualified_positive_count": sum(bool(row.get("qualified_positive")) for row in rows),
        "formal_task_consumption": len(rows), "manifest": str(root / E2_MANIFEST),
        "manifest_sha256": sha256(root / E2_MANIFEST), "combined_attempts": str(combined),
        "combined_attempts_sha256": sha256(combined), "resource_receipt_count": len(attempts),
        "resource_unknown_count": len(resource_unknown), "resource_unknown_attempt_ids": [item["attempt_id"] for item in resource_unknown],
        "selection": str(selection), "selection_sha256": sha256(selection),
        "g2_training_allowed": result.get("status") == "pass", "t2_evaluation_allowed": False,
    }
    atomic_write(run_root / "e2-harvest-reconciliation.json", reconciliation)
    return reconciliation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--run-root", type=Path, default=Path("/workspace/runs/phase-h140-g2-r2-20260820/formal-e2-harvest"))
    parser.add_argument("--gate-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config, release, paths = validate_release(root)
    attempts = build_attempts(root)
    if sha256(root / RUNNER) != config["frozen_scientific_contract"]["success_qualification"]["source_code_sha256"]:
        raise ValueError("closed-loop runner source hash differs from the frozen success-qualification contract")
    if subprocess.run(["git", "diff", "--quiet"], cwd=root, check=False).returncode != 0:
        raise ValueError("experiment checkout is dirty")
    if args.dry_run:
        print(json.dumps({"status": "dry-run", "attempt_count": len(attempts), "qualification_gpu_hours": release["qualification"]["qualification_gpu_hours"], "run_root": str(args.run_root), "contract": str(root / PATH_CONTRACT)}, sort_keys=True))
        return 0
    run_root = args.run_root.resolve()
    if run_root.exists() and any(run_root.iterdir()):
        raise ValueError(f"formal E2 run root is not empty: {run_root}")
    run_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    for name in ("attempts", "receipts", "logs", "scratch"):
        (run_root / name).mkdir(mode=0o700, exist_ok=True)
    subprocess.run(["chown", "-R", "1001:1001", str(run_root)], check=True)
    scheduler: dict[str, Any] = {"schema": "openevo.webshop-h1.40-e2-scheduler-receipt.v1", "campaign_id": CAMPAIGN_ID, "manifest_sha256": sha256(root / E2_MANIFEST), "experiment_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(), "open_evo_sha": OPEN_EVO_SHA, "attempts": []}
    atomic_write(run_root / "scheduler-receipt.json", scheduler)
    authorized = list(load_json(root / "configs/experiment/current-campaign.json")["authorized_gpu_uuids"])
    idle = [uuid for uuid in authorized if run_gate(root, uuid, run_root / "scratch" / uuid).returncode == 0]
    if not idle:
        raise RuntimeError("no authorized GPU passed the immediate live-idle gate")
    scheduler["selected_live_idle_gpu_uuids"] = idle
    atomic_write(run_root / "scheduler-receipt.json", scheduler)
    work = queue.Queue()
    for attempt in attempts:
        work.put(attempt)
    lock = threading.Lock()
    stop_event = threading.Event()
    def worker(uuid: str) -> None:
        while True:
            if stop_event.is_set():
                return
            try:
                attempt = work.get_nowait()
            except queue.Empty:
                return
            try:
                run_attempt(root, paths, run_root, attempt, uuid, scheduler["experiment_sha"], lock, scheduler, args.gate_timeout_seconds, stop_event)
            finally:
                work.task_done()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(idle)) as pool:
        list(pool.map(worker, idle))
    reconciliation = aggregate_and_select(root, run_root, scheduler["attempts"])
    print(json.dumps(reconciliation, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
