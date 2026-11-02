#!/usr/bin/env python3
"""Run the frozen Phase G WebShop base-vs-adapter diagnostic campaign."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_SRC = REPO_ROOT / "vendor/openevo-webshop/src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BENCHMARK_SRC) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_SRC))

from scripts.openevo_webshop.webshop_comparison_contract import (  # noqa: E402
    ACTION_CONTRACT,
    PROMPT_CONTRACT,
)
from scripts.openevo_webshop.webshop_task_manifest import (  # noqa: E402
    advance_access_ledger,
    load_access_ledger,
    load_manifest,
    write_access_ledger,
)
from scripts.validate_run_manifest import validate as validate_run_manifest  # noqa: E402


TASKS = (6680, 2590)
GENERATION_SEEDS = (202652, 202653)
ENVIRONMENT_SEED = 2026
MAX_STEPS = 15
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.4
HISTORY_LENGTH = 2
ARMS = ("base", "adapter_1x", "adapter_cumulative")
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
OPENEVO_STABLE = "796e6248a61112a61c880885844d3b0420c0b6ae"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_json(value: Any) -> str:
    return digest_bytes(canonical_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.chmod(0o600)
    temporary.replace(path)


def gpu_snapshot(expected_uuid: str) -> dict[str, Any]:
    def run(arguments: list[str]) -> str:
        completed = subprocess.run(arguments, check=True, capture_output=True, text=True)
        return completed.stdout.strip()

    rows = run([
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    if expected_uuid not in rows:
        raise RuntimeError(f"expected GPU UUID is not visible: {expected_uuid}")
    compute_apps = run([
        "nvidia-smi",
        "-i",
        expected_uuid,
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    return {"gpu_rows": rows.splitlines(), "compute_apps": compute_apps.splitlines()}


def resolve_adapter(path: Path, *, expected_components: int, model_revision: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    required = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "openevo_sd_lora_state.json",
        "openevo_sd_lora_state.safetensors",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(f"adapter {path} is incomplete: {', '.join(missing)}")

    state_path = path / "openevo_sd_lora_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("component_count") != expected_components:
        raise ValueError(f"adapter component count mismatch for {path}")
    if state.get("base_model") != "Qwen/Qwen2.5-3B-Instruct":
        raise ValueError(f"adapter base model mismatch for {path}")
    if state.get("model_revision") != model_revision:
        raise ValueError(f"adapter model revision mismatch for {path}")
    state_weights = path / "openevo_sd_lora_state.safetensors"
    state_weights_sha256 = sha256_file(state_weights)
    if state.get("state_weights_sha256") != state_weights_sha256:
        raise ValueError(f"adapter state-weights digest mismatch for {path}")

    evidence_path = path.parents[3] / "parameter-update-evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("status") != "pass":
        raise ValueError(f"adapter reload evidence is not PASS: {evidence_path}")
    after = evidence.get("after", {}).get("digest")
    reloaded = evidence.get("reloaded", {}).get("digest")
    if not after or after != reloaded:
        raise ValueError(f"adapter fresh-reload digest mismatch: {evidence_path}")
    if Path(str(evidence.get("state_path", ""))).name != state_weights.name:
        raise ValueError(f"adapter evidence points at a different state file: {evidence_path}")

    adapter_config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "adapter_id": state.get("adapter_id"),
        "component_count": state.get("component_count"),
        "effective_rank": state.get("effective_rank"),
        "base_model": state.get("base_model"),
        "model_revision": state.get("model_revision"),
        "state_manifest_sha256": sha256_file(state_path),
        "state_weights_sha256": state_weights_sha256,
        "peft_weights_sha256": sha256_file(path / "adapter_model.safetensors"),
        "fresh_reload_digest": reloaded,
        "fresh_reload_evidence": str(evidence_path),
        "peft_rank": adapter_config.get("r"),
        "identity_digest": digest_json({
            "adapter_id": state.get("adapter_id"),
            "component_count": state.get("component_count"),
            "effective_rank": state.get("effective_rank"),
            "model_revision": state.get("model_revision"),
            "state_manifest_sha256": sha256_file(state_path),
            "state_weights_sha256": state_weights_sha256,
            "peft_weights_sha256": sha256_file(path / "adapter_model.safetensors"),
        }),
    }


def task_rows(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = [*manifest["memory_tasks"], *manifest["promotion_tasks"], *manifest["tasks"]]
    selected = {int(row["goal_idx"]): row for row in rows}
    missing = [task for task in TASKS if task not in selected]
    if missing:
        raise ValueError(f"Phase G tasks are absent from the frozen manifest: {missing}")
    for task in TASKS:
        if selected[task]["split"] != "promotion-dev":
            raise ValueError(f"Phase G task {task} is not promotion-dev")
    return selected


def protocol_config(manifest: dict[str, Any], identities: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "openevo.phase-g.webshop-eval.v1",
        "tasks": list(TASKS),
        "generation_seeds": list(GENERATION_SEEDS),
        "environment_seed": ENVIRONMENT_SEED,
        "arms": list(ARMS),
        "max_steps": MAX_STEPS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "history_length": HISTORY_LENGTH,
        "prompt_contract": PROMPT_CONTRACT,
        "action_contract": ACTION_CONTRACT,
        "task_manifest_sha256": manifest["manifest_sha256"],
        "adapter_identities": identities,
    }


def validate_existing_episode(path: Path, expected: dict[str, Any], protocol_hash: str) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"existing episode identity mismatch at {path}: {key}")
    if record.get("protocol_hash") != protocol_hash:
        raise ValueError(f"existing episode protocol mismatch at {path}")
    if record.get("status") != "complete":
        raise ValueError(f"existing episode is not complete at {path}")
    return record


def model_backend(model: str, revision: str, dtype: str, adapter_path: str | None):
    from openevo_webshop.inference import TransformersBackend

    return TransformersBackend(
        model,
        device="cuda",
        revision=revision,
        dtype=dtype,
        adapter_path=adapter_path,
        local_files_only=True,
    )


def episode_path(root: Path, task_id: str, seed: int, arm: str) -> Path:
    return root / "episodes" / f"{task_id}-seed{seed}-{arm}.json"


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for record in records:
        by_arm[record["arm"]].append(record)

    metrics: dict[str, Any] = {}
    for arm, rows in by_arm.items():
        steps = sum(len(row["episode"].get("trace", [])) for row in rows)
        fallback = sum(
            sum(bool(step.get("used_fallback")) for step in row["episode"].get("trace", []))
            for row in rows
        )
        metrics[arm] = {
            "episode_count": len(rows),
            "average_reward": sum(float(row["episode"].get("task_score", 0.0)) for row in rows) / max(1, len(rows)),
            "success_rate": sum(bool(row["episode"].get("won")) for row in rows) / max(1, len(rows)),
            "parser_fallback_rate": fallback / max(1, steps),
            "average_episode_length": steps / max(1, len(rows)),
        }

    indexed = {(row["goal_idx"], row["generation_seed"], row["arm"]): row for row in records}
    paired: dict[str, list[dict[str, Any]]] = {"base_to_adapter_1x": [], "base_to_adapter_cumulative": [], "adapter_1x_to_adapter_cumulative": []}
    for task in TASKS:
        for seed in GENERATION_SEEDS:
            base = indexed[(task, seed, "base")]["episode"]["task_score"]
            one = indexed[(task, seed, "adapter_1x")]["episode"]["task_score"]
            cumulative = indexed[(task, seed, "adapter_cumulative")]["episode"]["task_score"]
            paired["base_to_adapter_1x"].append({"goal_idx": task, "generation_seed": seed, "delta": one - base})
            paired["base_to_adapter_cumulative"].append({"goal_idx": task, "generation_seed": seed, "delta": cumulative - base})
            paired["adapter_1x_to_adapter_cumulative"].append({"goal_idx": task, "generation_seed": seed, "delta": cumulative - one})
    return {"by_arm": metrics, "paired_deltas": paired, "task_seed_details": records}


def build_run_manifest(args: argparse.Namespace, summary: dict[str, Any], manifest: dict[str, Any], identities: dict[str, Any], protocol_hash: str, started: float, output_paths: list[Path]) -> dict[str, Any]:
    code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    peak_memory = max(
        (float(row["episode"].get("peak_gpu_memory_mb") or 0.0) for row in summary["task_seed_details"]),
        default=0.0,
    )
    return {
        "run": {"run_id": args.output.name, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "phase": "phase_g", "status": "pass"},
        "server": {"server_identity": "experiment-host", "control_plane": "user-space fallback", "experiment_container": "non-root fallback runtime", "image_tag": None, "image_digest": None},
        "hardware": {"gpu_indices": [args.gpu_index], "gpu_uuids": [args.gpu_uuid], "gpu_model": "NVIDIA GeForce RTX 5090"},
        "code": {"openevo_experiment_sha": code_sha, "upstream_openevo_repo": "https://github.com/CompLifeLab-ZJU/OpenEvo.git", "upstream_openevo_sha": OPENEVO_STABLE},
        "model": {"model_id": args.model, "model_revision": args.model_revision, "tokenizer_revision": args.model_revision, "dtype": args.dtype, "adapter_identity": "base+adapter_1x+adapter_cumulative"},
        "data": {"benchmark": "webshop", "split": "promotion-dev", "dataset_revision": "pinned task manifest", "dataset_hash": "sha256:" + manifest["manifest_sha256"], "index_hash": "sha256:" + manifest["index"]["materialized_tree"]["tree_sha256"], "task_manifest_identity": "seed3090.webshop-task-manifest.v1:" + manifest["manifest_sha256"]},
        "experiment": {"experiment_id": None, "passport_sha256": None, "method": "phase_g_base_vs_sd_lora", "seed": 2026, "generation": None, "config_hash": "sha256:" + protocol_hash, "gate_evidence": {"real_model_execution": True, "real_generation": True, "environment_generation_separated": True, "clean_model_per_arm": True, "adapter_identities_verified": identities}},
        "command": {"argv": sys.argv, "environment": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "")}},
        "outputs": {"artifacts": [{"logical_name": path.name, "sha256": "sha256:" + sha256_file(path), "locations": [str(path)]} for path in output_paths], "metrics": {"wall_time_s": time.time() - started, "peak_gpu_memory_mb": peak_memory}},
        "resources": {"peak_gpu_memory_mb": peak_memory, "wall_time_s": time.time() - started},
        "tracking": {"mode": "disabled", "wandb": {"enabled": False, "entity": None, "project": None, "id": None}, "weave": {"enabled": False, "entity": None, "project": None, "id": None}},
    }


def run(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.task_manifest)
    rows = task_rows(manifest)
    identities = {
        "base": None,
        "adapter_1x": resolve_adapter(args.adapter_one, expected_components=1, model_revision=args.model_revision),
        "adapter_cumulative": resolve_adapter(args.adapter_cumulative, expected_components=2, model_revision=args.model_revision),
    }
    protocol = protocol_config(manifest, identities)
    protocol_hash = digest_json(protocol)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=True)
    campaign_path = args.output / "campaign-manifest.json"
    campaign = {**protocol, "protocol_hash": protocol_hash, "status": "planned"}
    if campaign_path.exists():
        existing = json.loads(campaign_path.read_text(encoding="utf-8"))
        if existing.get("protocol_hash") != protocol_hash:
            raise ValueError("existing Phase G campaign manifest does not match the frozen protocol")
    else:
        atomic_json(campaign_path, campaign)

    lock_path = Path(args.lock or f"/tmp/openevo-webshop-gpu-locks/GPU-{args.gpu_uuid}.lock")
    lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"GPU lock busy: {lock_path}") from exc
        before_gpu = gpu_snapshot(args.gpu_uuid)
        if before_gpu["compute_apps"]:
            raise RuntimeError("compute process exists before Phase G model load")
        started = time.time()
        records: list[dict[str, Any]] = []
        for task in TASKS:
            row = rows[task]
            for seed in GENERATION_SEEDS:
                for arm in ARMS:
                    expected = {"task_id": row["task_id"], "goal_idx": task, "split": "promotion-dev", "arm": arm, "generation_seed": seed, "environment_seed": ENVIRONMENT_SEED, "model_revision": args.model_revision, "adapter_identity_digest": identities[arm]["identity_digest"] if identities[arm] else None}
                    path = episode_path(args.output, row["task_id"], seed, arm)
                    if path.exists():
                        records.append(validate_existing_episode(path, expected, protocol_hash))
                        continue
                    backend = model_backend(args.model, args.model_revision, args.dtype, identities[arm]["path"] if identities[arm] else None)
                    from openevo_webshop.env_compat import WebshopEnv, WebshopEnvConfig
                    from openevo_webshop.runner import run_episode

                    env = WebshopEnv(WebshopEnvConfig(seed_root=args.seed_root, num_products=1000, seed=ENVIRONMENT_SEED))
                    try:
                        episode = run_episode(env, backend, task, max_steps=MAX_STEPS, max_new=MAX_NEW_TOKENS, temperature=TEMPERATURE, generation_seed=seed, environment_seed=ENVIRONMENT_SEED)
                        peak_memory = 0.0
                        try:
                            import torch
                            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
                        except Exception:  # noqa: BLE001 - evidence field may remain zero on unsupported runtime
                            pass
                        record = {**expected, "schema": "openevo.phase-g.webshop-episode.v1", "status": "complete", "protocol_hash": protocol_hash, "adapter_identity": identities[arm], "episode": {**episode.__dict__, "peak_gpu_memory_mb": peak_memory}}
                        atomic_json(path, record)
                        records.append(record)
                    except Exception as exc:
                        atomic_json(path, {**expected, "schema": "openevo.phase-g.webshop-episode.v1", "status": "error", "protocol_hash": protocol_hash, "error": f"{type(exc).__name__}: {exc}"})
                        raise
                    finally:
                        env.close()
                        backend.close()
        after_gpu = gpu_snapshot(args.gpu_uuid)
        if after_gpu["compute_apps"]:
            raise RuntimeError("compute process remained after Phase G episode")

    summary = {"schema": "openevo.phase-g.webshop-summary.v1", "status": "pass", "protocol_hash": protocol_hash, "gpu_uuid": args.gpu_uuid, "before_gpu": before_gpu, "after_gpu": after_gpu, "episode_count": len(records), **summarize(records)}
    summary_path = args.output / "summary.json"
    atomic_json(summary_path, summary)
    campaign["status"] = "complete"
    atomic_json(campaign_path, campaign)
    output_paths = [campaign_path, summary_path, *sorted((args.output / "episodes").glob("*.json"))]
    run_manifest = build_run_manifest(args, summary, manifest, identities, protocol_hash, started, output_paths)
    run_manifest_path = args.output / "run-manifest.json"
    atomic_json(run_manifest_path, run_manifest)
    errors = validate_run_manifest(run_manifest_path)
    if errors:
        raise RuntimeError("Phase G Run Manifest validation failed: " + "; ".join(errors))
    if args.access_ledger is not None:
        ledger = load_access_ledger(args.access_ledger, task_manifest_sha256=manifest["manifest_sha256"])
        ledger = advance_access_ledger(ledger, {rows[task]["task_id"] for task in TASKS}, "PROMOTION_CONSUMED", run_uuid=args.output.name)
        write_access_ledger(args.access_ledger, ledger)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--benchmark-src", type=Path, default=BENCHMARK_SRC)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--access-ledger", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--adapter-one", type=Path, required=True)
    parser.add_argument("--adapter-cumulative", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
