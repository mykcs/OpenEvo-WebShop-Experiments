#!/usr/bin/env python3
"""Launch the preregistered H1.41 C0/C1 upstream evaluation panels.

The two trained arms are evaluated on the sealed E2, E1, and H1.39 panels.
The historical G1 reference is carried as an immutable input to reconciliation;
it is not silently re-run or mixed into the H1.41 formal denominator.
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
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.openevo_webshop.gpu_resource_wrapper import account_receipt
from scripts.openevo_webshop.path_contract import load_and_validate
from scripts.openevo_webshop.validate_h141_preregistration import validate as validate_prereg


CAMPAIGN_ID = "20260821-0141-h141-magnitude-screen"
CONFIG = Path("configs/experiment/h1.41-magnitude-screen-v2.json")
PATH_CONTRACT = Path("configs/experiment/h1.41-path-contract-v1.json")
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
PAYLOAD_UID_GID = "1001:1001"
TRAINING_SCHEDULER = "scheduler-receipt.json"
STAGES = ("acquisition", "retention", "preservation")
STAGE_TO_PANEL = {"acquisition": "acquisition_e2", "retention": "retention_e1", "preservation": "preservation_h139"}
STAGE_TO_RUNNER = {"acquisition": "acquisition", "retention": "retention", "preservation": "fresh"}
MANIFESTS = {
    "acquisition": Path("configs/experiment/manifests/h1.40-g2-e2-reservoir-v1.json"),
    "retention": Path("configs/experiment/manifests/h1.40-g2-e1-retention-set-v1.json"),
    "preservation": Path("configs/experiment/manifests/h1.40-g2-h139-preservation-anchor-v1.json"),
}
LIVE_GATE_RETRIES = 12
LIVE_GATE_DELAY_SECONDS = 5.0


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adapter_identity(adapter: Path) -> dict[str, str | int]:
    state_path = adapter / "openevo_sd_lora_state.json"
    state_weights = adapter / "openevo_sd_lora_state.safetensors"
    if not state_path.is_file() or not state_weights.is_file():
        raise ValueError(f"H1.41 adapter state manifest/weights are incomplete: {adapter}")
    state = load(state_path)
    state_sha = sha(state_weights)
    if state.get("state_weights_sha256") != state_sha:
        raise ValueError(f"H1.41 adapter state-weights digest mismatch: {adapter}")
    completed = subprocess.run(
        [PYTHON, str(ROOT / "scripts/openevo_webshop/compute_adapter_tensor_digest.py"), "--adapter", str(adapter)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"H1.41 adapter tensor digest failed for {adapter}: {completed.stderr.strip()}")
    digest = json.loads(completed.stdout)
    tensor_digest = str(digest.get("loaded_tensor_digest") or "")
    if len(tensor_digest) != 64:
        raise ValueError(f"H1.41 adapter tensor digest is incomplete: {adapter}")
    return {
        "adapter": str(adapter),
        "adapter_file_sha256": sha(adapter / "adapter_model.safetensors"),
        "state_weights_sha256": state_sha,
        "tensor_digest": tensor_digest,
        "tensor_count": int(digest["tensor_count"]),
    }


def host_adapter_path(uri: str, run_root: Path) -> Path:
    raw = uri.removeprefix("file://")
    container_root = PurePosixPath("/workspace/runs/phase-h141-magnitude-screen-20260821")
    candidate = PurePosixPath(raw)
    if candidate == container_root or container_root not in candidate.parents:
        raise ValueError(f"training adapter URI is outside the H1.41 run root: {uri}")
    return run_root / Path(*candidate.relative_to(container_root).parts)


def atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_run_root(run_root: Path) -> None:
    """Create host-side output parents writable by the isolated payload UID."""

    run_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    subprocess.run(["chown", PAYLOAD_UID_GID, str(run_root)], check=True)
    for name in ("cells", "receipts", "logs", "scratch"):
        directory = run_root / name
        directory.mkdir(mode=0o700, exist_ok=True)
        subprocess.run(["chown", PAYLOAD_UID_GID, str(directory)], check=True)


def contract_paths(root: Path) -> dict[str, str]:
    contract = load(root / PATH_CONTRACT)
    return {name: str(spec["host_path"]) for name, spec in contract["paths"].items()}


def container_path(host: Path, campaign_root: Path) -> str:
    relative = host.resolve().relative_to(campaign_root.resolve())
    return f"/workspace/runs/phase-h141-magnitude-screen-20260821/{relative.as_posix()}"


def panel_rows(root: Path, stage: str) -> list[dict[str, Any]]:
    config = load(root / CONFIG)
    panel = config["frozen_scientific_contract"]["evaluation_panels"][STAGE_TO_PANEL[stage]]
    rows = {row["task_id"]: row for row in load(root / MANIFESTS[stage])["tasks"]}
    task_ids = panel["task_ids"]
    if any(task_id not in rows for task_id in task_ids) or len(task_ids) != len(set(task_ids)):
        raise ValueError(f"{stage} panel identity mismatch")
    return [rows[task_id] for task_id in task_ids]


def validate(root: Path, training_receipt: Path) -> tuple[dict[str, str], dict[str, Any]]:
    validate_prereg(root, require_router=True)
    config = load(root / CONFIG)
    current = load(root / "configs/experiment/current-campaign.json")
    if current.get("status") != "launch-authorized" or current.get("formal_task_consumption_allowed") is not True:
        raise ValueError("H1.41 router is not launch-authorized")
    if not training_receipt.is_file():
        raise ValueError("H1.41 training scheduler receipt is missing")
    training = load(training_receipt)
    if training.get("schema") != "openevo.webshop-h1.41-magnitude-training-scheduler.v1" or training.get("status") != "pass":
        raise ValueError("H1.41 training is not a complete passing handoff")
    if set(training.get("arms", {})) != {"C0_exact_h140", "C1_reset_magnitude"}:
        raise ValueError("H1.41 training arms are incomplete")
    for arm, payload in training["arms"].items():
        if payload.get("status") != "pass" or len(payload.get("attempts", [])) != 8:
            raise ValueError(f"{arm} does not contain eight passing increments")
        if not payload["attempts"][-1].get("adapter_uri"):
            raise ValueError(f"{arm} final adapter URI is missing")
    paths = contract_paths(root)
    for name in ("experiment", "openevo_clean", "openevo_reset_patch", "model", "dataset", "cache", "run_root", "adapter"):
        if not Path(paths[name]).exists():
            raise ValueError(f"H1.41 path-contract target is missing: {name}")
    if sha(Path(paths["adapter"]) / "adapter_model.safetensors") != G1_FILE_SHA:
        raise ValueError("sealed G1 adapter identity changed")
    load_and_validate(root / PATH_CONTRACT)
    return paths, config


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


def live_idle_authorized(root: Path, authorized: list[str], scratch: Path) -> list[str]:
    """Return the authorized UUIDs that pass the immediate live-idle gate."""

    available = []
    for uuid in authorized:
        gate = run_gate(root, uuid, scratch / uuid)
        if gate.returncode == 0:
            available.append(uuid)
    return available


def docker_command(paths: dict[str, str], run_root: Path, row: dict[str, Any], stage: str, arm: str, uuid: str, output: Path, adapter: Path, adapter_sha: str, tensor_digest: str, experiment_sha: str) -> list[str]:
    campaign_root = Path(paths["run_root"])
    core = Path(paths["openevo_clean"] if arm == "C0_exact_h140" else paths["openevo_reset_patch"])
    adapter_container = "/workspace/prior-g1" if arm == "g1" else f"/workspace/{arm}-final-adapter"
    manifest_container = f"/workspace/src/openevo-experiment/{MANIFESTS[stage].as_posix()}"
    output_container = container_path(output, campaign_root)
    return [
        "docker", "run", "--rm", "--init", "--read-only", "--tmpfs", "/tmp:rw,nosuid,size=4g", "--mount", "type=tmpfs,destination=/root,tmpfs-mode=0755", "--shm-size=64g", "--pids-limit", "4096", "--gpus", f"device={uuid}", "--user", PAYLOAD_UID_GID,
        "-v", f"{paths['experiment']}:/workspace/src/openevo-experiment:ro", "-v", f"{core}:/workspace/src/openevo-core:ro", "-v", f"{Path(paths['model'])}:/workspace/models/qwen2.5-7b-instruct:ro", "-v", f"{Path(paths['dataset'])}:/workspace/data/webshop-1000:ro", "-v", f"{Path(paths['cache'])}:/workspace/cache:rw", "-v", f"{campaign_root}:/workspace/runs/phase-h141-magnitude-screen-20260821:rw", "-v", "/workspace/openevo-workspace:/workspace/openevo-workspace:ro", "-v", f"{adapter}:{adapter_container}:ro", "-v", f"{SEED_ROOT}:/workspace/runs/phase-h1.20-action-admissibility-diagnostic-v1-20260815/seed-roots/base-P1:ro", "-w", "/workspace/src/openevo-experiment",
        "-e", f"CUDA_VISIBLE_DEVICES={uuid}", "-e", "TOKENIZERS_PARALLELISM=false", "-e", "TRANSFORMERS_OFFLINE=1", "-e", "HF_HUB_OFFLINE=1", "-e", f"JAVA_HOME={JAVA_HOME}", "-e", "PYTHONPATH=/workspace/src/openevo-core/src:/workspace/src/openevo-experiment:/workspace/src/openevo-experiment/scripts/openevo_webshop", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "HOME=/tmp", "-e", "TMPDIR=/tmp", "-e", "XDG_CACHE_HOME=/workspace/cache",
        "--entrypoint", PYTHON, IMAGE, "-m", "scripts.openevo_webshop.h1_closed_loop_tracked", "--task-manifest", manifest_container, "--seed-root", "/workspace/runs/phase-h1.20-action-admissibility-diagnostic-v1-20260815/seed-roots/base-P1", "--model", "/workspace/models/qwen2.5-7b-instruct", "--model-revision", MODEL_REVISION, "--adapter", adapter_container, "--expected-adapter-file-sha256", adapter_sha, "--expected-adapter-tensor-digest", tensor_digest, "--arm", "sd-lora", "--tasks", str(row["goal_idx"]), "--rollout-seeds", *(str(seed) for seed in load(ROOT / CONFIG)["frozen_scientific_contract"]["evaluation_panels"][STAGE_TO_PANEL[stage]]["rollout_seeds"]), "--output", output_container, "--gpu-uuid", uuid, "--experiment-git-sha", experiment_sha, "--upstream-openevo-sha", OPEN_EVO_SHA, "--runtime-fingerprint", RUNTIME_FINGERPRINT, "--image-digest", IMAGE_DIGEST, "--temperature", "0.0", "--max-steps", "30", "--max-new", "512", "--prompt-contract", "seed3090.webshop-compact-action-prompt.v2", "--action-contract", "seed3090.webshop-action-parser.v3-action-command", "--evaluation-stage", STAGE_TO_RUNNER[stage], "--wandb-mode", "offline", "--wandb-project", "openevo-webshop-experiments", "--wandb-entity", "example-entity", "--campaign-id", CAMPAIGN_ID, "--wandb-group", CAMPAIGN_ID, "--wandb-dir", f"{output_container}/wandb",
    ]


def run_cell(root: Path, paths: dict[str, str], run_root: Path, row: dict[str, Any], stage: str, arm: str, uuid: str, adapter: Path, adapter_sha: str, tensor_digest: str, experiment_sha: str, scheduler: dict[str, Any], lock: threading.Lock, stop: threading.Event) -> None:
    cell_id = f"{stage}-{arm}-{row['task_id']}"
    output = run_root / "cells" / stage / arm / row["task_id"]
    receipt = run_root / "receipts" / f"{cell_id}.json"
    log = run_root / "logs" / f"{cell_id}.log"
    for directory in (
        run_root / "cells",
        run_root / "cells" / stage,
        run_root / "cells" / stage / arm,
        output,
    ):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        subprocess.run(["chown", PAYLOAD_UID_GID, str(directory)], check=True)
    receipt.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    log.parent.mkdir(mode=0o700, exist_ok=True)
    gate = wait_for_live_gate(root, uuid, run_root / "scratch" / uuid)
    if gate.returncode != 0:
        stop.set()
        raise RuntimeError(f"{cell_id}: immediate live-idle gate failed")
    command = docker_command(paths, run_root, row, stage, arm, uuid, output, adapter, adapter_sha, tensor_digest, experiment_sha)
    wrapper = [PYTHON, str(root / WRAPPER), "--receipt", str(receipt), "--attempt-id", cell_id, "--campaign-id", CAMPAIGN_ID, "--stage", "evaluation", "--command-identity", "h141-magnitude-screen-evaluation-v1", "--gpu-uuids", uuid, "--path-contract", str(root / PATH_CONTRACT), "--", *command]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(wrapper, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False, text=True)
    row_result = {"cell_id": cell_id, "stage": stage, "arm": arm, "task_id": row["task_id"], "goal_idx": row["goal_idx"], "gpu_uuid": uuid, "return_code": completed.returncode, "receipt": str(receipt), "resource_accounting": account_receipt(load(receipt) if receipt.is_file() else {})}
    with lock:
        scheduler["cells"].append(row_result)
        atomic_write(run_root / "scheduler-receipt.json", scheduler)
    if not receipt.is_file():
        stop.set()
        raise RuntimeError(f"{cell_id}: wrapper did not produce an exact receipt")


def execute(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    training_receipt = args.training_receipt.resolve()
    paths, config = validate(root, training_receipt)
    run_root = args.run_root.resolve()
    if run_root.exists() and any(run_root.iterdir()):
        raise ValueError(f"H1.41 evaluation run root is not empty: {run_root}")
    prepare_run_root(run_root)
    experiment_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    arms: dict[str, dict[str, str]] = {}
    for arm, payload in load(training_receipt)["arms"].items():
        final = payload["attempts"][-1]
        adapter = host_adapter_path(str(final["adapter_uri"]), Path(paths["run_root"]))
        evidence = load(Path(final["evidence"]))
        if not adapter.is_dir():
            raise ValueError(f"{arm} final adapter directory is missing")
        arms[arm] = adapter_identity(adapter)
    cells = [(stage, arm, row) for stage in STAGES for arm in arms for row in panel_rows(root, stage)]
    reference_path = ROOT / "docs/evidence/remote-runs/2026-08-21/h1.41/h1.41-g1-deterministic-reference-v1.json"
    scheduler: dict[str, Any] = {"schema": "openevo.webshop-h1.41-upstream-evaluation-scheduler.v1", "status": "running", "campaign_id": CAMPAIGN_ID, "experiment_git_sha": experiment_sha, "training_scheduler": str(training_receipt), "training_scheduler_sha256": sha(training_receipt), "deterministic_g1_reference": str(reference_path), "deterministic_g1_reference_sha256": sha(reference_path), "adapter_identities": arms, "stages": list(STAGES), "expected_cell_count": len(cells), "formal_task_attempts": len(cells) * 4, "cells": []}
    atomic_write(run_root / "scheduler-receipt.json", scheduler)
    lock = threading.Lock()
    stop = threading.Event()
    failures: list[str] = []
    authorized = load(root / "configs/experiment/current-campaign.json")["authorized_gpu_uuids"]
    available = live_idle_authorized(root, authorized, run_root / "scratch" / "batch-000")
    if not available:
        raise RuntimeError("no authorized GPU passed the immediate live-idle gate for H1.41 evaluation")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(authorized)) as pool:
        offset = 0
        batch_index = 0
        while offset < len(cells):
            if stop.is_set():
                break
            futures = []
            available = live_idle_authorized(root, authorized, run_root / "scratch" / f"batch-{batch_index:03d}")
            if not available:
                raise RuntimeError("no authorized GPU passed the immediate live-idle gate for H1.41 evaluation batch")
            for index, (stage, arm, row) in enumerate(cells[offset : offset + len(available)]):
                uuid = available[index]
                identity = arms[arm]
                futures.append(pool.submit(run_cell, root, paths, run_root, row, stage, arm, uuid, Path(identity["adapter"]), identity["adapter_file_sha256"], identity["tensor_digest"], experiment_sha, scheduler, lock, stop))
            for future in futures:
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001
                    failures.append(str(exc))
                    stop.set()
                    break
            offset += len(futures)
            batch_index += 1
    scheduler["status"] = "pass" if not failures and len(scheduler["cells"]) == len(cells) else "incomplete"
    scheduler["failures"] = failures
    scheduler["completed_cell_count"] = len(scheduler["cells"])
    atomic_write(run_root / "scheduler-receipt.json", scheduler)
    if failures:
        raise RuntimeError("; ".join(failures))
    return {"status": "pass", "campaign_id": CAMPAIGN_ID, "cell_count": len(cells), "formal_task_attempts": len(cells) * 4, "run_root": str(run_root)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--training-receipt", type=Path, default=Path("/workspace/runs/phase-h141-magnitude-screen-20260821/scheduler-receipt.json"))
    parser.add_argument("--run-root", type=Path, default=Path("/workspace/runs/phase-h141-magnitude-screen-20260821/evaluation"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        paths, config = validate(args.root.resolve(), args.training_receipt.resolve())
        cells = sum(len(panel_rows(args.root.resolve(), stage)) for stage in STAGES) * len(config["frozen_scientific_contract"]["arms"])
        print(json.dumps({"status": "dry-run", "campaign_id": CAMPAIGN_ID, "cell_count": cells, "formal_task_attempts": cells * 4, "run_root": paths["run_root"]}, sort_keys=True))
        return 0
    print(json.dumps(execute(args), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
