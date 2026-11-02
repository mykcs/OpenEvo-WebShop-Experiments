#!/usr/bin/env python3
"""Tracked H1 closed-loop WebShop evaluation for one frozen adapter arm."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

try:
    from scripts.openevo_webshop.parametric_rnd import (
        ENV_SEED,
        FixedEnvironmentSeed,
        ParserSafeRetryBackend,
        gpu_snapshot,
        normalize_episode_contract,
    )
    from scripts.openevo_webshop.wandb_tracking import add_wandb_arguments, start_wandb_run
    from scripts.openevo_webshop.webshop_comparison_contract import ACTION_COMMAND_CONTRACT, ACTION_CONTRACT, TOLERANT_ACTION_CONTRACT
except ModuleNotFoundError:  # pragma: no cover - direct script execution in a container
    from parametric_rnd import (
        ENV_SEED,
        FixedEnvironmentSeed,
        ParserSafeRetryBackend,
        gpu_snapshot,
        normalize_episode_contract,
    )
    from wandb_tracking import add_wandb_arguments, start_wandb_run
    from webshop_comparison_contract import ACTION_COMMAND_CONTRACT, ACTION_CONTRACT, TOLERANT_ACTION_CONTRACT
try:
    from openevo_webshop.prompts import COMPACT_PROMPT_CONTRACT, PROMPT_CONTRACT, STATE_AWARE_PROMPT_CONTRACT
except ModuleNotFoundError:  # pragma: no cover - repository checkout import
    from scripts.openevo_webshop.webshop_comparison_contract import (
        COMPACT_PROMPT_CONTRACT,
        PROMPT_CONTRACT,
        STATE_AWARE_PROMPT_CONTRACT,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def _selected_tasks(manifest_path: Path, task_ids: list[int]) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("tasks") or manifest.get("train") or manifest.get("train_tasks") or []
    selected = [row for row in rows if int(row["goal_idx"]) in task_ids]
    if len(selected) != len(task_ids) or len({row["goal_idx"] for row in selected}) != len(task_ids):
        raise ValueError("H1 task selection is missing or duplicated in the frozen manifest")
    for row in selected:
        description = row.get("task_description")
        if not isinstance(description, str) or hashlib.sha256(description.encode()).hexdigest() != row.get("task_description_sha256"):
            raise ValueError(f"task description identity mismatch for {row.get('goal_idx')}")
    return selected


def _require_uuid_cuda_binding(expected_uuid: str) -> str:
    """Require one UUID-bound CUDA namespace before loading a model.

    The runner uses ``device='cuda'`` intentionally; CUDA_VISIBLE_DEVICES must
    therefore map logical ``cuda:0`` to the UUID recorded in the GPU ledger.
    Failing closed prevents several supposedly UUID-bound processes from
    silently loading onto physical GPU 0.
    """

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    if tokens != [expected_uuid]:
        raise SystemExit(
            "H1 requires CUDA_VISIBLE_DEVICES to contain exactly the selected "
            f"GPU UUID {expected_uuid}; observed {visible!r}"
        )
    return visible


def _action_counts(episode: dict[str, Any]) -> dict[str, int]:
    counts = {name: 0 for name in ("search", "click", "select", "buy", "other")}
    for step in episode.get("trace", []) or []:
        action = str(step.get("action") or "").lower()
        name = next((candidate for candidate in counts if candidate != "other" and action.startswith(candidate + "[")), "other")
        counts[name] += 1
    return counts


def run(args: argparse.Namespace) -> int:
    tasks = _selected_tasks(args.task_manifest.resolve(), args.tasks)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = args.lock or args.output / f"GPU-{args.gpu_uuid}.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"GPU lock busy: {lock_path}") from exc
        cuda_visible_devices = _require_uuid_cuda_binding(args.gpu_uuid)
        before = gpu_snapshot(args.gpu_uuid)
        _atomic_json(args.output / "gpu-before.json", before)
        if before["compute_apps"]:
            raise SystemExit("compute process exists before H1 model load")

        from openevo_webshop.env_compat import WebshopEnv, WebshopEnvConfig
        from openevo_webshop.inference import make_backend
        from openevo_webshop.runner import run_episode

        backend = make_backend(
            "transformers",
            args.model,
            device="cuda",
            revision=args.model_revision,
            adapter_path=args.adapter if args.adapter != "none" else None,
            local_files_only=True,
        )
        adapter_load_receipt = backend.load_receipt
        expected_adapter_loaded = args.adapter != "none"
        if not bool(adapter_load_receipt.get("verified")) or bool(
            adapter_load_receipt.get("adapter_loaded")
        ) != expected_adapter_loaded:
            raise SystemExit("adapter load receipt does not match requested evaluation arm")
        if expected_adapter_loaded:
            if not args.expected_adapter_file_sha256 or not args.expected_adapter_tensor_digest:
                raise SystemExit("adapted evaluation requires frozen adapter file and tensor digests")
            if (
                adapter_load_receipt.get("adapter_file_sha256") != args.expected_adapter_file_sha256
                or adapter_load_receipt.get("loaded_tensor_digest") != args.expected_adapter_tensor_digest
            ):
                raise SystemExit("loaded adapter identity differs from the frozen evaluation arm")
        elif args.expected_adapter_file_sha256 or args.expected_adapter_tensor_digest:
            raise SystemExit("BASE evaluation must not declare adapter digests")
        protocol = {
            "phase": "H1",
            "arm": args.arm,
            "evaluation_stage": args.evaluation_stage,
            "increment_index": args.increment_index,
            "model": args.model,
            "model_revision": args.model_revision,
            "task_manifest_sha256": hashlib.sha256(args.task_manifest.read_bytes()).hexdigest(),
            "environment_seed": ENV_SEED,
            "generation_seeds": args.rollout_seeds,
            "temperature": args.temperature,
            "max_steps": args.max_steps,
            "max_new_tokens": args.max_new,
            "prompt_contract": args.prompt_contract,
            "action_contract": args.action_contract,
            "evaluator": "WebShop task_score/won",
            "parser_retry_seed_offset": 900000,
            "cuda_visible_devices": cuda_visible_devices,
            "gpu_binding_contract": "single UUID via CUDA_VISIBLE_DEVICES; logical cuda:0",
            "adapter_loaded": adapter_load_receipt["adapter_loaded"],
            "adapter_file_sha256": adapter_load_receipt["adapter_file_sha256"],
            "adapter_tensor_digest": adapter_load_receipt["loaded_tensor_digest"],
            "latency_contract": "attempt_wall_time_seconds measured around one closed-loop episode; invalid attempts remain outside speed denominators",
        }
        protocol_fingerprint = _digest(protocol)
        tracker = start_wandb_run(
            mode=args.wandb_mode,
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.arm}-h1-closed-loop",
            job_type="evaluation",
            group=args.wandb_group or "H1-successful-supervision-train-fit-v1",
            config={**protocol, "campaign_id": args.campaign_id, "experiment_git_sha": args.experiment_git_sha, "upstream_openevo_sha": args.upstream_openevo_sha, "runtime_fingerprint": args.runtime_fingerprint, "image_digest": args.image_digest, "gpu_uuid": args.gpu_uuid, "adapter": args.adapter},
            tags=("benchmark:webshop", "phase:h1", f"method:{args.arm if args.arm in {'base', 'ordinary-lora', 'sd-lora'} else 'sd-lora'}", f"arm:{args.arm}", "class:closed-loop", f"prompt:{args.prompt_contract}"),
            notes="H1 closed-loop evaluation; only generation seed varies from the frozen evaluation protocol.",
            directory=args.wandb_dir or args.output / "wandb",
        )
        attempts_path = args.output / "attempts.jsonl"
        rows: list[dict[str, Any]] = []
        started = time.time()
        try:
            with attempts_path.open("w", encoding="utf-8") as attempts:
                for task in tasks:
                    for seed in args.rollout_seeds:
                        attempt_started = time.perf_counter()
                        raw_env = WebshopEnv(WebshopEnvConfig(seed_root=str(args.seed_root), num_products=1000, seed=ENV_SEED))
                        env = FixedEnvironmentSeed(raw_env, ENV_SEED)
                        safe_backend = ParserSafeRetryBackend(backend, action_contract=args.action_contract)
                        try:
                            result = run_episode(env, safe_backend, int(task["goal_idx"]), max_steps=args.max_steps, max_new=args.max_new, temperature=args.temperature, generation_seed=int(seed), prompt_contract=args.prompt_contract)
                            episode = result.__dict__
                            normalize_episode_contract(episode)
                            if safe_backend.events:
                                episode["parser_events"] = safe_backend.events
                                episode["parser_retry_events"] = [event for event in safe_backend.events if event.get("event_type") == "parser-retry"]
                                episode["parser_repair_events"] = [event for event in safe_backend.events if event.get("event_type") == "missing-closing-action-tag"]
                            error = None
                            attempt_wall_time_seconds = time.perf_counter() - attempt_started
                        except BaseException as exc:
                            episode = {"episode": int(task["goal_idx"]), "rollout_seed": seed, "trace": [], "task_score": 0.0, "won": False, "error": True}
                            error = f"{type(exc).__name__}: {exc}"
                            attempt_wall_time_seconds = time.perf_counter() - attempt_started
                        finally:
                            raw_env.close()
                        trace = episode.get("trace") if isinstance(episode.get("trace"), list) else []
                        first_prompt = trace[0].get("prompt", "") if trace else ""
                        score = float(episode.get("task_score", 0.0) or 0.0)
                        parser_failure = bool(episode.get("error")) or any(bool(step.get("used_fallback")) or not bool(step.get("action_valid")) for step in trace if isinstance(step, dict))
                        identity_ok = task["task_description"] in first_prompt
                        complete = bool(trace) and len(trace) <= args.max_steps and len(trace) == int(episode.get("steps", len(trace)))
                        repair_events = episode.get("parser_repair_events") if isinstance(episode.get("parser_repair_events"), list) else []
                        retry_events = episode.get("parser_retry_events") if isinstance(episode.get("parser_retry_events"), list) else []
                        row = {"arm": args.arm, "evaluation_stage": args.evaluation_stage, "increment_index": args.increment_index, "task_id": task["task_id"], "goal_idx": task["goal_idx"], "rollout_seed": seed, "final_task_score": score, "won": bool(episode.get("won")), "identity_ok": identity_ok, "trace_complete": complete, "parser_or_fallback_failure": parser_failure, "parser_repair_used": bool(repair_events), "parser_retry_used": bool(retry_events), "qualified_positive": bool(identity_ok and complete and not parser_failure and score > 0 and not error), "error": error, "episode": episode, "protocol_fingerprint": protocol_fingerprint, "adapter_load_receipt": adapter_load_receipt}
                        row["attempt_wall_time_seconds"] = attempt_wall_time_seconds
                        _atomic_json(args.output / "episodes" / f"{task['task_id']}-{seed}.json", row)
                        attempts.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                        attempts.flush()
                        rows.append(row)
                        tracker.log({"eval/attempt_index": len(rows), "eval/final_task_score": score, "eval/win": int(bool(episode.get("won"))), "eval/qualified_positive": int(row["qualified_positive"]), "eval/parser_failure": int(parser_failure), "eval/parser_repair_used": int(row["parser_repair_used"]), "eval/trace_complete": int(complete), "eval/episode_steps": len(trace), "eval/attempt_wall_time_seconds": attempt_wall_time_seconds, "eval/generation_seed": seed, "eval/action_counts": _action_counts(episode)})
            after = gpu_snapshot(args.gpu_uuid)
            _atomic_json(args.output / "gpu-after.json", after)
            summary = {"schema": "seed3090.webshop-h1-closed-loop.v1", "arm": args.arm, "attempt_count": len(rows), "valid_count": sum(r["identity_ok"] and r["trace_complete"] and not r["parser_or_fallback_failure"] and not r["error"] for r in rows), "qualified_positive_count": sum(r["qualified_positive"] for r in rows), "mean_task_score": sum(r["final_task_score"] for r in rows) / len(rows) if rows else 0.0, "protocol_fingerprint": protocol_fingerprint, "adapter_load_receipt": adapter_load_receipt, "wall_time_seconds": time.time() - started, "tracking": tracker.metadata()}
            _atomic_json(args.output / "summary.json", summary)
            tracker.set_summary({"run/status": "pass", **{key: value for key, value in summary.items() if isinstance(value, (int, float))}})
            return 0
        finally:
            backend.close()
            tracker.finish()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapter", required=True, help="none or an immutable adapter directory")
    parser.add_argument("--expected-adapter-file-sha256")
    parser.add_argument("--expected-adapter-tensor-digest")
    parser.add_argument(
        "--arm",
        choices=(
            "base",
            "failure",
            "success",
            "success-state",
            "student-recovery",
            "teacher-action",
            "oracle-action",
            "self-evolution",
            "diversity-4",
            "diversity-8",
            "diversity-16",
            "ordinary-lora",
            "sd-lora",
            "base-r",
            "openevo-transfer",
            "self-generated-harvest",
        ),
        required=True,
    )
    parser.add_argument("--tasks", type=int, nargs="+", required=True)
    parser.add_argument("--rollout-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--evaluation-stage", choices=("acquisition", "retention", "fresh"))
    parser.add_argument("--increment-index", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--experiment-git-sha", required=True)
    parser.add_argument("--upstream-openevo-sha", required=True)
    parser.add_argument("--runtime-fingerprint", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--prompt-contract", choices=(PROMPT_CONTRACT, COMPACT_PROMPT_CONTRACT, STATE_AWARE_PROMPT_CONTRACT), default=PROMPT_CONTRACT)
    parser.add_argument("--action-contract", choices=(ACTION_CONTRACT, TOLERANT_ACTION_CONTRACT, ACTION_COMMAND_CONTRACT), default=ACTION_CONTRACT)
    add_wandb_arguments(parser, default_project="openevo-webshop-experiments")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
