#!/usr/bin/env python3
"""SEED official held-out WebShop evaluation driver (one frozen arm, one contract).

Runs a list of held-out (or, for qualification, train-pool) WebShop tasks under a
single frozen evaluation contract and a single frozen arm, writing per-episode
JSON records and an attempts.jsonl stream. Designed for one GPU, sequential
execution, with fail-closed GPU-UUID binding and frozen adapter identity
enforcement.

Two contracts:
- ``seed-eval`` (PRIMARY): byte-exact SEED public-code inference contract via
  openevo_webshop.seed_eval.run_seed_eval_episode (pass-through projection, SEED
  prompt/history, temperature sampling with a frozen per-episode seed).
- ``openevo-native`` (DIAGNOSTIC): the historical OpenEvo-native contract
  (compact-v2 prompt + v3-action-command parser, temperature 0) via
  openevo_webshop.runner.run_episode. Does not enter the comparison table.

The driver never consumes held-out tasks unless given the frozen panel manifest;
qualification uses train-pool goal_idx (>=500) or synthetic bounded generation.
"""

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
        FixedEnvironmentSeed,
        ParserSafeRetryBackend,
        gpu_snapshot,
        normalize_episode_contract,
    )
    from scripts.openevo_webshop.wandb_tracking import add_wandb_arguments, start_wandb_run
except ModuleNotFoundError:  # pragma: no cover - direct script execution in a container
    from parametric_rnd import (
        FixedEnvironmentSeed,
        ParserSafeRetryBackend,
        gpu_snapshot,
        normalize_episode_contract,
    )
    from wandb_tracking import add_wandb_arguments, start_wandb_run

SEED_EVAL_CONTRACT = "seed-eval"
OPENEVO_NATIVE_CONTRACT = "openevo-native"
COMPACT_PROMPT_CONTRACT = "seed3090.webshop-compact-action-prompt.v2"
ACTION_COMMAND_CONTRACT = "seed3090.webshop-action-parser.v3-action-command"


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def _require_uuid_cuda_binding(expected_uuid: str) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    if tokens != [expected_uuid]:
        raise SystemExit(
            "evaluation requires CUDA_VISIBLE_DEVICES to contain exactly the selected "
            f"GPU UUID {expected_uuid}; observed {visible!r}"
        )
    return visible


def _load_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Return task rows with goal_idx, instruction hash, and generation seed."""
    if args.task_manifest is not None:
        manifest = json.loads(args.task_manifest.read_text(encoding="utf-8"))
        rows = manifest.get("tasks") or []
        selected = [r for r in rows if int(r["goal_idx"]) in set(args.tasks)] if args.tasks else list(rows)
        if not selected:
            raise SystemExit("no tasks selected from the frozen manifest")
        for row in selected:
            instruction = row.get("instruction")
            if not isinstance(instruction, str) or hashlib.sha256(instruction.encode()).hexdigest() != row.get("instruction_sha256"):
                raise SystemExit(f"manifest task identity mismatch for goal_idx {row.get('goal_idx')}")
        return [
            {
                "goal_idx": int(r["goal_idx"]),
                "task_id": r.get("task_id", f"webshop_seed0_goal{int(r['goal_idx']):05d}"),
                "instruction_sha256": r["instruction_sha256"],
                "generation_seed": int(r["generation_seed"]),
                "panel_ordinal": r.get("panel_ordinal"),
            }
            for r in selected
        ]
    if not args.tasks:
        raise SystemExit("either --task-manifest or --tasks is required")
    if args.generation_seed_base is None:
        raise SystemExit("--generation-seed-base is required when --tasks is used without a manifest")
    return [
        {
            "goal_idx": int(goal_idx),
            "task_id": f"webshop_qual_goal{int(goal_idx):05d}",
            "instruction_sha256": None,
            "generation_seed": int(args.generation_seed_base) + offset,
            "panel_ordinal": None,
        }
        for offset, goal_idx in enumerate(args.tasks)
    ]


def run(args: argparse.Namespace) -> int:
    tasks = _load_tasks(args)
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
            raise SystemExit("compute process exists before model load")

        from openevo_webshop.env_compat import WebshopEnv, WebshopEnvConfig
        from openevo_webshop.inference import make_backend

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

        if args.contract == SEED_EVAL_CONTRACT:
            temperature, max_steps, do_sample = 0.4, 15, True
        else:
            temperature, max_steps, do_sample = 0.0, 30, False
        temperature = args.temperature if args.temperature is not None else temperature
        max_steps = args.max_steps if args.max_steps is not None else max_steps

        protocol = {
            "campaign_id": args.campaign_id,
            "arm": args.arm,
            "contract": args.contract,
            "model": args.model,
            "model_revision": args.model_revision,
            "environment_seed": args.environment_seed,
            "temperature": temperature,
            "do_sample": do_sample,
            "top_p": 1.0,
            "top_k": "disabled" if do_sample else None,
            "max_steps": max_steps,
            "max_new_tokens": args.max_new,
            "evaluator": "WebShop raw task_score + exact success (won)",
            "task_manifest_sha256": hashlib.sha256(args.task_manifest.read_bytes()).hexdigest() if args.task_manifest else None,
            "cuda_visible_devices": cuda_visible_devices,
            "gpu_binding_contract": "single UUID via CUDA_VISIBLE_DEVICES; logical cuda:0",
            "adapter_loaded": adapter_load_receipt["adapter_loaded"],
            "adapter_file_sha256": adapter_load_receipt["adapter_file_sha256"],
            "adapter_tensor_digest": adapter_load_receipt["loaded_tensor_digest"],
        }
        protocol_fingerprint = _digest(protocol)
        tracker = start_wandb_run(
            mode=args.wandb_mode,
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.arm}-{args.contract}-seed-heldout",
            job_type="evaluation",
            group=args.wandb_group or args.campaign_id,
            config={**protocol, "experiment_git_sha": args.experiment_git_sha, "image_digest": args.image_digest, "gpu_uuid": args.gpu_uuid, "adapter": args.adapter},
            tags=("benchmark:webshop", "campaign:seed-heldout", f"arm:{args.arm}", f"contract:{args.contract}"),
            notes="SEED official held-out WebShop comparison; frozen arm, frozen contract.",
            directory=args.wandb_dir or args.output / "wandb",
        )

        attempts_path = args.output / "attempts.jsonl"
        rows: list[dict[str, Any]] = []
        resumed_episodes = 0
        episodes_dir = args.output / "episodes"
        started = time.time()
        try:
            with attempts_path.open("w", encoding="utf-8") as attempts:
                for task in tasks:
                    # Resume-skip: an episode is deterministic (frozen per-task
                    # generation_seed, environment_seed, model, adapter), so a
                    # previously COMPLETED episode (non-empty trace) recorded for this
                    # exact task identity is reused verbatim instead of re-consuming
                    # GPU. A stub with an empty trace (a runtime/infra error, not a
                    # policy outcome) is NOT reused; it is re-run to obtain the first
                    # real measurement. This is completion of an incomplete
                    # measurement, not task replacement.
                    episode_file = episodes_dir / f"{task['task_id']}-{task['generation_seed']}.json"
                    if episode_file.is_file():
                        try:
                            existing = json.loads(episode_file.read_text(encoding="utf-8"))
                        except Exception:  # noqa: BLE001 - unparseable -> re-run fresh
                            existing = None
                        if (
                            isinstance(existing, dict)
                            and existing.get("task_id") == task["task_id"]
                            and int(existing.get("generation_seed", -1)) == int(task["generation_seed"])
                            and int(existing.get("goal_idx", -2)) == int(task["goal_idx"])
                            and existing.get("arm") == args.arm
                            and existing.get("contract") == args.contract
                            and isinstance(existing.get("episode"), dict)
                            and len(existing["episode"].get("trace") or []) > 0
                        ):
                            rows.append(existing)
                            resumed_episodes += 1
                            continue
                    attempt_started = time.perf_counter()
                    raw_env = WebshopEnv(WebshopEnvConfig(seed_root=str(args.seed_root), num_products=1000, seed=args.environment_seed))
                    env = FixedEnvironmentSeed(raw_env, args.environment_seed)
                    error = None
                    try:
                        if args.contract == SEED_EVAL_CONTRACT:
                            from openevo_webshop.seed_eval import run_seed_eval_episode

                            episode = run_seed_eval_episode(
                                env,
                                backend,
                                int(task["goal_idx"]),
                                environment_seed=args.environment_seed,
                                generation_seed=int(task["generation_seed"]),
                                max_steps=max_steps,
                                max_new=args.max_new,
                                temperature=temperature,
                                expected_task_description_sha256=task["instruction_sha256"],
                            )
                        else:
                            from openevo_webshop.runner import run_episode

                            safe_backend = ParserSafeRetryBackend(backend, action_contract=ACTION_COMMAND_CONTRACT)
                            result = run_episode(
                                env,
                                safe_backend,
                                int(task["goal_idx"]),
                                max_steps=max_steps,
                                max_new=args.max_new,
                                temperature=temperature,
                                generation_seed=int(task["generation_seed"]),
                                environment_seed=args.environment_seed,
                                prompt_contract=COMPACT_PROMPT_CONTRACT,
                                expected_task_description_sha256=task["instruction_sha256"],
                            )
                            episode = result.__dict__
                            normalize_episode_contract(episode)
                            if safe_backend.events:
                                episode["parser_events"] = safe_backend.events
                        attempt_wall_time_seconds = time.perf_counter() - attempt_started
                    except BaseException as exc:  # noqa: BLE001 - record and continue
                        episode = {"episode": int(task["goal_idx"]), "task_score": 0.0, "won": False, "trace": [], "error": True}
                        error = f"{type(exc).__name__}: {exc}"
                        attempt_wall_time_seconds = time.perf_counter() - attempt_started
                    finally:
                        raw_env.close()

                    trace = episode.get("trace") if isinstance(episode.get("trace"), list) else []
                    score = float(episode.get("task_score", 0.0) or 0.0)
                    if args.contract == SEED_EVAL_CONTRACT:
                        invalid_steps = int(episode.get("invalid_action_steps", 0) or 0)
                        runtime_invalid = bool(error)
                        valid_episode = (not runtime_invalid) and bool(trace)
                    else:
                        invalid_steps = sum(1 for s in trace if isinstance(s, dict) and (bool(s.get("used_fallback")) or not bool(s.get("action_valid"))))
                        runtime_invalid = bool(error)
                        valid_episode = (not runtime_invalid) and bool(trace)
                    row = {
                        "arm": args.arm,
                        "contract": args.contract,
                        "campaign_id": args.campaign_id,
                        "task_id": task["task_id"],
                        "goal_idx": int(task["goal_idx"]),
                        "panel_ordinal": task.get("panel_ordinal"),
                        "generation_seed": int(task["generation_seed"]),
                        "environment_seed": args.environment_seed,
                        "final_task_score": score,
                        "task_score_x100": score * 100.0,
                        "won": bool(episode.get("won")),
                        "done": bool(episode.get("done")),
                        "steps": int(episode.get("steps", len(trace)) or 0),
                        "terminal_reason": episode.get("terminal_reason"),
                        "invalid_action_steps": invalid_steps,
                        "valid_episode": valid_episode,
                        "runtime_invalid": runtime_invalid,
                        "error": error,
                        "gpu_uuid": args.gpu_uuid,
                        "attempt_wall_time_seconds": attempt_wall_time_seconds,
                        "episode": episode,
                        "protocol_fingerprint": protocol_fingerprint,
                        "adapter_load_receipt": adapter_load_receipt,
                    }
                    _atomic_json(args.output / "episodes" / f"{task['task_id']}-{task['generation_seed']}.json", row)
                    attempts.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    attempts.flush()
                    rows.append(row)
                    tracker.log({
                        "eval/attempt_index": len(rows),
                        "eval/final_task_score": score,
                        "eval/win": int(bool(episode.get("won"))),
                        "eval/invalid_action_steps": invalid_steps,
                        "eval/valid_episode": int(valid_episode),
                        "eval/episode_steps": len(trace),
                        "eval/attempt_wall_time_seconds": attempt_wall_time_seconds,
                        "eval/generation_seed": int(task["generation_seed"]),
                    })
            after = gpu_snapshot(args.gpu_uuid)
            _atomic_json(args.output / "gpu-after.json", after)
            valid_rows = [r for r in rows if r["valid_episode"]]
            summary = {
                "schema": "openevo.webshop-seed-heldout-eval.v1",
                "arm": args.arm,
                "contract": args.contract,
                "campaign_id": args.campaign_id,
                "attempt_count": len(rows),
                "resumed_episode_count": resumed_episodes,
                "fresh_episode_count": len(rows) - resumed_episodes,
                "valid_count": len(valid_rows),
                "invalid_count": len(rows) - len(valid_rows),
                "mean_task_score": sum(r["final_task_score"] for r in rows) / len(rows) if rows else 0.0,
                "mean_task_score_valid_only": sum(r["final_task_score"] for r in valid_rows) / len(valid_rows) if valid_rows else 0.0,
                "exact_success_rate": sum(1 for r in rows if r["won"]) / len(rows) if rows else 0.0,
                "exact_success_rate_valid_only": sum(1 for r in valid_rows if r["won"]) / len(valid_rows) if valid_rows else 0.0,
                "protocol_fingerprint": protocol_fingerprint,
                "adapter_load_receipt": adapter_load_receipt,
                "wall_time_seconds": time.time() - started,
                "gpu_uuid": args.gpu_uuid,
                "tracking": tracker.metadata(),
            }
            _atomic_json(args.output / "summary.json", summary)
            tracker.set_summary({"run/status": "pass", **{k: v for k, v in summary.items() if isinstance(v, (int, float))}})
            print(json.dumps({
                "status": "pass",
                "arm": args.arm,
                "contract": args.contract,
                "total_episodes": len(rows),
                "resumed_episodes": resumed_episodes,
                "fresh_episodes": len(rows) - resumed_episodes,
                "valid_episodes": len(valid_rows),
            }))
            return 0
        finally:
            backend.close()
            tracker.finish()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", choices=(SEED_EVAL_CONTRACT, OPENEVO_NATIVE_CONTRACT), required=True)
    parser.add_argument("--arm", choices=("base", "sd-lora"), required=True)
    parser.add_argument("--task-manifest", type=Path)
    parser.add_argument("--tasks", type=int, nargs="+")
    parser.add_argument("--generation-seed-base", type=int)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--environment-seed", type=int, default=0)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapter", required=True, help="none or an immutable adapter directory")
    parser.add_argument("--expected-adapter-file-sha256")
    parser.add_argument("--expected-adapter-tensor-digest")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-new", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--experiment-git-sha", required=True)
    parser.add_argument("--image-digest", required=True)
    add_wandb_arguments(parser, default_project="openevo-webshop-experiments")
    args = parser.parse_args()
    # --campaign-id is registered by add_wandb_arguments (shared helper); the driver
    # always passes it. Require it here so a missing value fails fast and loud.
    if not (args.campaign_id or "").strip():
        parser.error("--campaign-id is required")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
