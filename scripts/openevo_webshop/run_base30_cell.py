#!/usr/bin/env python3
"""Run one frozen BASE-30 WebShop task cell on one explicit GPU UUID."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest_json(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()



def configure_scratch_environment(out, wandb_mode):
    """Keep W&B, caches, and temporary files off a potentially full root disk."""
    roots = {
        "wandb": out / "wandb",
        "wandb-cache": out / "wandb-cache",
        "wandb-config": out / "wandb-config",
        "wandb-artifacts": out / "wandb-artifacts",
        "tmp": out / "tmp",
        "xdg-cache": out / "xdg-cache",
    }
    for path in roots.values():
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    for key, path in (
        ("TMPDIR", roots["tmp"]),
        ("XDG_CACHE_HOME", roots["xdg-cache"]),
        ("WANDB_DIR", roots["wandb"]),
        ("WANDB_CACHE_DIR", roots["wandb-cache"]),
        ("WANDB_CONFIG_DIR", roots["wandb-config"]),
        ("WANDB_ARTIFACT_DIR", roots["wandb-artifacts"]),
    ):
        os.environ[key] = str(path)
    os.environ["WANDB_MODE"] = wandb_mode
    os.environ["OPENEVO_WANDB_MODE"] = wandb_mode


def gpu_snapshot(expected_uuid):
    def run(args):
        return subprocess.check_output(args, text=True).strip()

    rows = run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"])
    if expected_uuid not in rows:
        raise RuntimeError(f"expected GPU UUID is not visible: {expected_uuid}")
    apps = run(["nvidia-smi", "-i", expected_uuid, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    return {"gpu_rows": rows.splitlines(), "compute_apps": apps.splitlines(), "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def load_manifest(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = doc.get("manifest_sha256")
    payload = dict(doc)
    payload.pop("manifest_sha256", None)
    if digest != hashlib.sha256(canonical_bytes(payload)).hexdigest():
        raise ValueError("task manifest digest mismatch")
    return doc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-root", required=True)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--task-manifest", required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--phase", default="base30")
    parser.add_argument("--rollout-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--prompt-contract", default=None)
    parser.add_argument(
        "--arm",
        default="BASE_CONTROL",
        choices=("BASE_CONTROL", "OPEN_EVO", "SEED"),
        help="Method arm label recorded in every episode and W&B config.",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Optional local PEFT/SD-LoRA adapter. Omit for BASE_CONTROL or merged SEED HF weights.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline"),
        default=os.environ.get("OPENEVO_WANDB_MODE", "online"),
        help="Online by default; use offline only when authentication/network is unavailable.",
    )
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    manifest = load_manifest(args.task_manifest)
    tasks = {int(row["goal_idx"]): row for row in manifest["tasks"]}
    if args.task not in tasks:
        raise ValueError(f"task {args.task} missing from manifest")
    task = tasks[args.task]
    expected_uuid = args.gpu_uuid
    java_home = os.environ.get("JAVA_HOME")
    javac = Path(java_home) / "bin" / "javac" if java_home else None
    if javac is None or not javac.is_file():
        raise RuntimeError("JDK contract failed: set JAVA_HOME to a JDK containing bin/javac before launch")
    out = Path(args.output_root)
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    configure_scratch_environment(out, args.wandb_mode)
    free_bytes = shutil.disk_usage(out).free
    if free_bytes < 512 * 1024 * 1024:
        raise RuntimeError(f"output filesystem has too little free space: {free_bytes} bytes")
    (out / "gpu-before.json").write_text(json.dumps(gpu_snapshot(expected_uuid), indent=2) + "\n", encoding="utf-8")
    if gpu_snapshot(expected_uuid)["compute_apps"]:
        raise RuntimeError(f"GPU {expected_uuid} has compute apps before launch")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, expected_uuid):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES is already bound to {visible!r}, expected {expected_uuid}")
    os.environ["CUDA_VISIBLE_DEVICES"] = expected_uuid

    repo_root = Path(args.benchmark_root).resolve()
    vendor_src = repo_root / "vendor/openevo-webshop/src"
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(vendor_src))
    from openevo_webshop.env_compat import WebshopEnv, WebshopEnvConfig
    from openevo_webshop.inference import TransformersBackend
    from openevo_webshop.runner import run_episode
    from scripts.openevo_webshop.webshop_comparison_contract import COMPACT_PROMPT_CONTRACT

    import wandb

    seeds = list(args.rollout_seeds)
    prompt_contract = args.prompt_contract or COMPACT_PROMPT_CONTRACT
    run = wandb.init(
        entity="example-entity",
        project="openevo-webshop-experiments",
        group=args.group,
        name=f"{args.arm.lower()}-task-{args.task}",
        job_type=f"{args.phase}-eval",
        config={"phase": args.phase, "arm": args.arm, "task": args.task, "rollout_seeds": seeds, "gpu_uuid": expected_uuid, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "model": args.model, "model_revision": args.model_revision, "adapter_path": args.adapter_path, "wandb_mode": args.wandb_mode, "manifest_sha256": manifest["manifest_sha256"], "runtime_storage_root": str(out), "wandb_storage_root": str(out / "wandb")},
        mode=args.wandb_mode,
    )
    env = WebshopEnv(WebshopEnvConfig(seed_root=args.seed_root, num_products=1000, seed=2026))
    backend = TransformersBackend(args.model, device="cuda", revision=args.model_revision, dtype="float16", adapter_path=args.adapter_path, local_files_only=True)
    attempts_path = out / "attempts.jsonl"
    summaries = []
    try:
        for attempt_index, rollout_seed in enumerate(seeds, 1):
            error = None
            identity_ok = False
            parser_failure = False
            trace_complete = False
            episode = None
            started = time.time()
            try:
                result = run_episode(env, backend, args.task, max_steps=30, max_new=512, temperature=0.0, generation_seed=rollout_seed, environment_seed=2026, prompt_contract=prompt_contract)
                episode = result.__dict__
                first_prompt = episode.get("trace", [{}])[0].get("prompt", "") if episode.get("trace") else ""
                identity_ok = task["task_description"] in first_prompt
                parser_failure = any(bool(step.get("parser_failure") or step.get("used_fallback")) for step in episode.get("trace", []))
                trace_complete = bool(episode.get("trace")) and not bool(episode.get("error"))
            except Exception as exc:  # record operational failures, never hide them
                error = f"{type(exc).__name__}: {exc}"
            score = float((episode or {}).get("task_score", 0.0) or 0.0)
            row = {
                "schema": "seed3090.webshop-h1.15-carrier-episode.v1",
                "arm": args.arm,
                "task_id": task["task_id"],
                "goal_idx": args.task,
                "rollout_seed": rollout_seed,
                "episode": episode,
                "error": error,
                "identity_ok": identity_ok,
                "trace_complete": trace_complete,
                "parser_or_fallback_failure": parser_failure,
                "final_task_score": round(score, 3),
                "won": bool((episode or {}).get("won", False)),
                "qualified_positive": bool(error is None and identity_ok and trace_complete and not parser_failure and score > 0.0),
                "protocol_fingerprint": "3c430fa52850598325202af4e27241e6a9ac44c907e3e1f9b589212d0ba4d154",
                "elapsed_s": round(time.time() - started, 3),
            }
            with attempts_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            summaries.append(row)
            run.log({
                "eval/attempt_index": attempt_index,
                "eval/final_task_score": row["final_task_score"],
                "eval/qualified_positive": int(row["qualified_positive"]),
                "eval/parser_failure": int(row["parser_or_fallback_failure"]),
                "eval/trace_complete": int(row["trace_complete"]),
                "eval/win": int(row["won"]),
                "eval/episode_steps": len((episode or {}).get("trace", [])),
                "eval/rollout_seed": rollout_seed,
            }, step=attempt_index)
    finally:
        backend.close()
        env.close()
        (out / "gpu-after.json").write_text(json.dumps(gpu_snapshot(expected_uuid), indent=2) + "\n", encoding="utf-8")
        run.finish()
    valid = [r for r in summaries if r["error"] is None and r["identity_ok"] and r["trace_complete"] and not r["parser_or_fallback_failure"]]
    summary = {
        "schema": "seed3090.webshop-base30-cell-summary.v1",
        "phase": args.phase,
        "arm": args.arm,
        "model": args.model,
        "model_revision": args.model_revision,
        "adapter_path": args.adapter_path,
        "wandb_mode": args.wandb_mode,
        "task_id": task["task_id"],
        "goal_idx": args.task,
        "gpu_uuid": expected_uuid,
        "scheduled": len(summaries),
        "valid": len(valid),
        "invalid": len(summaries) - len(valid),
        "qualified_positive": sum(bool(r["qualified_positive"]) for r in valid),
        "mean_task_score_valid": round(sum(float(r["final_task_score"]) for r in valid) / max(1, len(valid)), 6),
        "wandb_run_id": run.id,
        "wandb_url": f"https://wandb.ai/example-entity/openevo-webshop-experiments/runs/{run.id}",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
