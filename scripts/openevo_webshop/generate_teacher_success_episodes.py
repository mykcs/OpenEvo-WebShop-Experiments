#!/usr/bin/env python3
"""Generate train-only successful demonstrations with a frozen strong teacher."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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
        raise ValueError("teacher task selection is missing or duplicated")
    return selected


def qualified_episode(episode: dict[str, Any]) -> bool:
    trace = episode.get("trace") or []
    return bool(trace) and float(episode.get("task_score") or 0.0) == 1.0 and not episode.get("error") and all(
        bool(step.get("action_valid")) and not step.get("used_fallback") and not step.get("parser_failure") for step in trace
    )


def configure_runtime_cache(args: argparse.Namespace) -> None:
    if args.runtime_cache_dir is None:
        return
    cache_root = args.runtime_cache_dir.expanduser().resolve()
    cache_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.environ.setdefault("TRITON_HOME", str(cache_root / "triton"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(cache_root / "triton"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_root / "torchinductor"))


def _teacher_usage(backend: Any, args: argparse.Namespace) -> dict[str, Any]:
    usage = getattr(backend, "usage_totals", None)
    totals = dict(usage) if isinstance(usage, dict) else {}
    result: dict[str, Any] = {"token_usage": totals}
    if args.input_cost_per_1k is not None or args.output_cost_per_1k is not None:
        input_tokens = float(totals.get("prompt_tokens", 0))
        output_tokens = float(totals.get("completion_tokens", 0))
        input_rate = float(args.input_cost_per_1k or 0.0)
        output_rate = float(args.output_cost_per_1k or 0.0)
        result["estimated_cost"] = {
            "currency": args.cost_currency,
            "input_cost_per_1k_tokens": args.input_cost_per_1k,
            "output_cost_per_1k_tokens": args.output_cost_per_1k,
            "amount": input_tokens / 1000.0 * input_rate + output_tokens / 1000.0 * output_rate,
        }
    return result


def _close_backend(backend: Any) -> None:
    """Close local backends when supported; gateway backends are stateless."""
    close = getattr(backend, "close", None)
    if callable(close):
        close()


def run_teacher_rollouts(args: argparse.Namespace) -> dict[str, Any]:
    configure_runtime_cache(args)
    if not args.teacher_model:
        args.teacher_model = str(args.model_path)
    from openevo_webshop.env_compat import WebshopEnv, WebshopEnvConfig
    from openevo_webshop.inference import make_backend
    from openevo_webshop.runner import run_episode
    from scripts.openevo_webshop.parametric_rnd import (
        FixedEnvironmentSeed,
        ParserSafeRetryBackend,
        gpu_snapshot,
    )

    gpu_workload_launched = args.backend == "transformers"
    if gpu_workload_launched:
        visible = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
        if visible != [args.gpu_uuid]:
            raise ValueError(f"CUDA_VISIBLE_DEVICES must contain exactly {args.gpu_uuid}; observed {visible!r}")
        before = gpu_snapshot(args.gpu_uuid)
        if before["compute_apps"]:
            raise ValueError(f"compute app exists before teacher load: {before['compute_apps']}")
    else:
        before = {"status": "not-applicable", "gpu_workload_launched": False}
    tasks = _selected_tasks(args.task_manifest.resolve(), args.tasks)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=True)
    _atomic_json(args.output / "gpu-before.json", before)
    if args.backend == "gateway":
        raw_backend = make_backend("gateway", args.teacher_model, base_url=args.teacher_base_url)
    else:
        raw_backend = make_backend("transformers", str(args.model_path), device="cuda", revision=None, adapter_path=None, local_files_only=True)
    backend = ParserSafeRetryBackend(raw_backend, action_contract=args.action_contract)
    rows: list[dict[str, Any]] = []
    started = time.time()
    try:
        for task in tasks:
            for rollout_seed in args.rollout_seeds:
                raw_env = WebshopEnv(WebshopEnvConfig(seed_root=str(args.seed_root), num_products=1000, seed=2026))
                env = FixedEnvironmentSeed(raw_env, 2026)
                error = None
                try:
                    result = run_episode(env, backend, int(task["goal_idx"]), max_steps=args.max_steps, max_new=args.max_new, temperature=0.0, generation_seed=int(rollout_seed), prompt_contract=args.prompt_contract)
                    result.action_contract = args.action_contract
                    episode = dataclasses.asdict(result)
                except BaseException as exc:  # noqa: BLE001 - preserve runtime failures separately
                    episode = {"episode": int(task["goal_idx"]), "trace": [], "task_score": 0.0, "won": False, "error": True}
                    error = f"{type(exc).__name__}: {exc}"
                finally:
                    raw_env.close()
                row = {"schema": "seed3090.webshop-teacher-episode.v2", "arm": "teacher-success", "task_id": task["task_id"], "goal_idx": task["goal_idx"], "rollout_seed": int(rollout_seed), "qualified_positive": qualified_episode(episode), "error": error, "teacher_backend": args.backend, "teacher_identity": args.teacher_identity, "teacher_model": args.teacher_model, "teacher_contract_sha256": args.teacher_contract_sha256, "teacher_prompt_sha256": args.teacher_prompt_sha256, "teacher_payload_sha256": args.teacher_payload_sha256, "teacher_evidence_sha256": args.teacher_evidence_sha256, "parser_events": list(backend.events), "episode": episode, "protocol_fingerprint": _digest({"backend": args.backend, "teacher_model": args.teacher_model, "teacher_contract_sha256": args.teacher_contract_sha256, "teacher_prompt_sha256": args.teacher_prompt_sha256, "task_manifest_sha256": hashlib.sha256(args.task_manifest.read_bytes()).hexdigest(), "prompt_contract": args.prompt_contract, "action_contract": args.action_contract, "temperature": 0.0, "max_steps": args.max_steps, "max_new_tokens": args.max_new, "rollout_seed": int(rollout_seed)})}
                backend.events.clear()
                path = args.output / "episodes" / f"{task['task_id']}-{rollout_seed}.json"
                _atomic_json(path, row)
                rows.append(row)
        usage = _teacher_usage(raw_backend, args)
        _close_backend(raw_backend)
        after = gpu_snapshot(args.gpu_uuid) if gpu_workload_launched else {"status": "not-applicable", "gpu_workload_launched": False}
        _atomic_json(args.output / "gpu-after.json", after)
        summary = {"schema": "seed3090.webshop-teacher-rollout.v2", "teacher_backend": args.backend, "teacher_identity": args.teacher_identity, "teacher_model": args.teacher_model, "teacher_contract_sha256": args.teacher_contract_sha256, "teacher_prompt_sha256": args.teacher_prompt_sha256, "teacher_payload_sha256": args.teacher_payload_sha256, "teacher_evidence_sha256": args.teacher_evidence_sha256, "prompt_contract": args.prompt_contract, "action_contract": args.action_contract, "task_ids": args.tasks, "rollout_seeds": args.rollout_seeds, "attempt_count": len(rows), "valid_count": sum(not row["error"] for row in rows), "qualified_positive_count": sum(row["qualified_positive"] for row in rows), "runtime_error_count": sum(bool(row["error"]) for row in rows), "parser_repair_count": sum(sum(event.get("event_type") == "missing-closing-action-tag" for event in row.get("parser_events", [])) for row in rows), "parser_retry_count": sum(sum(event.get("event_type") == "parser-retry" for event in row.get("parser_events", [])) for row in rows), "wall_time_seconds": time.time() - started, "gpu_uuid": args.gpu_uuid, "gpu_workload_launched": gpu_workload_launched, "runtime_cache_dir": str(args.runtime_cache_dir) if args.runtime_cache_dir else None, **usage}
        _atomic_json(args.output / "summary.json", summary)
        return summary
    finally:
        _close_backend(raw_backend)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--backend", choices=("transformers", "gateway"), default="transformers")
    parser.add_argument("--teacher-model", default="")
    parser.add_argument("--teacher-base-url")
    parser.add_argument("--prompt-contract", default="seed3090.webshop-compact-action-prompt.v2")
    parser.add_argument("--action-contract", default="seed3090.webshop-action-parser.v1")
    parser.add_argument("--teacher-identity", required=True)
    parser.add_argument("--teacher-contract-sha256", default="")
    parser.add_argument("--teacher-prompt-sha256", default="")
    parser.add_argument("--teacher-payload-sha256", required=True)
    parser.add_argument("--teacher-evidence-sha256", required=True)
    parser.add_argument("--tasks", type=int, nargs="+", required=True)
    parser.add_argument("--rollout-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", default="not-applicable")
    parser.add_argument("--experiment-git-sha", required=True)
    parser.add_argument("--upstream-openevo-sha", required=True)
    parser.add_argument("--runtime-fingerprint", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--runtime-cache-dir", type=Path)
    parser.add_argument("--input-cost-per-1k", type=float)
    parser.add_argument("--output-cost-per-1k", type=float)
    parser.add_argument("--cost-currency", default="USD")
    args = parser.parse_args()
    if args.backend == "transformers" and args.model_path is None:
        parser.error("--model-path is required for the transformers backend")
    if args.backend == "gateway":
        if not args.teacher_model:
            parser.error("--teacher-model is required for the gateway backend")
        if not args.teacher_contract_sha256 or not args.teacher_prompt_sha256:
            parser.error("gateway backend requires --teacher-contract-sha256 and --teacher-prompt-sha256")
    print(json.dumps(run_teacher_rollouts(args), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
