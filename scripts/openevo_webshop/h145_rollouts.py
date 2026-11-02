#!/usr/bin/env python3
"""H1.45 rollout runner with fail-closed invalid-action semantics.

Malformed or inadmissible model output never becomes an environment action. A
unique deterministic wrapper repair may proceed, but repaired episodes are
never eligible for SD-LoRA training. The gateway backend is used only by the
explicit Kimi teacher arm; Stage 2 always runs the student locally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from openevo_webshop.comparison_contract import format_webshop_actions, render_compact_webshop_prompt, extract_goal
from scripts.openevo_webshop.webshop_comparison_contract import parse_action_command_contract, repair_action_command
from openevo_webshop.env_compat import WebshopEnv, WebshopEnvConfig
from openevo_webshop.inference import make_backend
from scripts.openevo_webshop.parametric_rnd import FixedEnvironmentSeed, gpu_snapshot

SCHEDULE_SCHEMA = "openevo.h144-on-training-schedule.v1"
RUN_SCHEMA = "openevo.h145-rollout-segment.v1"
ACTION_RE = re.compile(r"(?:search|click|select|buy)\[[^\]\n]+\]", re.IGNORECASE)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_rows(path: Path, start: int, end: int) -> list[dict[str, Any]]:
    schedule = json.loads(path.read_text(encoding="utf-8"))
    if schedule.get("schema") != SCHEDULE_SCHEMA:
        raise ValueError(f"schedule schema mismatch: {schedule.get('schema')!r}")
    details: dict[tuple[int, int, str], dict[str, Any]] = {}
    for task in schedule.get("stage1", {}).get("tasks", []):
        details[(int(task["worker_seed"]), int(task["source_session_index"]), str(task["instruction_sha256"]))] = task
    for group in schedule.get("stage2", {}).get("rounds", []):
        for task in group.get("groups", []):
            details[(int(task["worker_seed"]), int(task["source_session_index"]), str(task["instruction_sha256"]))] = task
    rows = []
    for row in schedule.get("rollouts", []):
        position = int(row["position"])
        if start <= position <= end:
            key = (int(row["worker_seed"]), int(row["source_session_index"]), str(row["instruction_sha256"]))
            task = details[key]
            rows.append({**row, "instruction": task["instruction"]})
    if [int(row["position"]) for row in rows] != list(range(start, end + 1)):
        raise ValueError("schedule does not cover the requested contiguous range")
    return rows


def admissible(action: str, available: dict[str, Any]) -> bool:
    candidates = format_webshop_actions(available)
    if action.startswith("search["):
        return bool(available.get("has_search_bar"))
    return action in candidates


def run_episode(env: Any, backend: Any, task: dict[str, Any], *, seed: int,
                temperature: float, max_steps: int, max_new: int,
                top_p: float | None = None, top_k: int | None = None,
                repetition_penalty: float | None = None) -> dict[str, Any]:
    observation, info = env.reset(int(task["source_session_index"]), seed=int(task["worker_seed"]))
    goal = extract_goal(observation)
    history: list[str] = []
    trace: list[dict[str, Any]] = []
    repaired_any = False
    for step_index in range(max_steps):
        available = env.get_available_actions()
        prompt = render_compact_webshop_prompt(
            task_description=goal, current_observation=observation,
            available_actions=available, step_count=step_index,
            history_length=2, action_history=" | ".join(history[-2:]) or "(none)",
            max_prompt_chars=20_000,
        )
        raw = backend.generate(
            prompt, max_new=max_new, temperature=temperature,
            generation_seed=int(seed) + step_index,
            top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty,
        )
        prepared, action, repaired = repair_action_command(raw)
        parsed_action, parsed_valid = parse_action_command_contract(prepared, require_think=True)
        valid = bool(parsed_valid and action == parsed_action and admissible(action, available))
        step = {
            "step": step_index, "prompt": prompt, "raw_completion": raw,
            "repaired_completion": prepared if repaired else None,
            "action": action if valid else None, "action_valid": valid,
            "used_fallback": False, "parser_failure": not valid,
            "parser_repair_used": bool(repaired), "generation_seed": int(seed) + step_index,
            "environment_seed": int(task["worker_seed"]),
        }
        trace.append(step)
        if not valid:
            return {"episode": int(task["source_session_index"]), "steps": step_index + 1,
                    "task_score": 0.0, "won": False, "trace": trace,
                    "trace_complete": False, "terminated_on_invalid_action": True,
                    "parser_repair_used": repaired_any or bool(repaired), "trainable": False}
        repaired_any = repaired_any or bool(repaired)
        observation, reward, done, info = env.step(action)
        info = info or {}
        history.append(f"obs:{str(observation)[:60]} -> act:{action}")
        step["task_score"] = float(info.get("task_score", reward) or 0.0)
        if done:
            won = bool(info.get("won", reward == 1.0))
            return {"episode": int(task["source_session_index"]), "steps": step_index + 1,
                    "task_score": max(float(step["task_score"]), 0.0), "won": won,
                    "trace": trace, "trace_complete": True,
                    "terminated_on_invalid_action": False,
                    "parser_repair_used": repaired_any,
                    "trainable": bool(won and step["task_score"] == 1.0 and not repaired_any)}
    return {"episode": int(task["source_session_index"]), "steps": max_steps,
            "task_score": max((float(item.get("task_score", 0.0)) for item in trace), default=0.0),
            "won": False, "trace": trace, "trace_complete": True,
            "terminated_on_invalid_action": False, "parser_repair_used": repaired_any,
            "trainable": False}


def backend_transport_stats(backend: Any) -> dict[str, int]:
    value = getattr(backend, "transport_stats", None)
    if isinstance(value, dict):
        return {"attempts": int(value.get("attempts", 0)), "retries": int(value.get("retries", 0))}
    return {"attempts": 0, "retries": 0}


def main(args: argparse.Namespace) -> int:
    tasks = load_rows(args.schedule.resolve(), args.start_position, args.end_position)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=True)
    (args.output / "episodes").mkdir(mode=0o700, exist_ok=True)
    if args.backend == "transformers":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible != args.gpu_uuid:
            raise ValueError(f"CUDA_VISIBLE_DEVICES must equal selected UUID {args.gpu_uuid!r}")
        write_json(args.output / "gpu-before.json", gpu_snapshot(args.gpu_uuid))
        backend = make_backend("transformers", args.model, device="cuda", revision=args.model_revision,
                               adapter_path=None if args.adapter == "none" else args.adapter, local_files_only=True)
    else:
        write_json(args.output / "gpu-before.json", {"status": "not-applicable", "gpu_workload_launched": False})
        if not os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENEVO_TEACHER_API_KEY"):
            os.environ["OPENAI_API_KEY"] = os.environ["OPENEVO_TEACHER_API_KEY"]
        backend = make_backend(
            "gateway", args.teacher_model, base_url=args.teacher_base_url,
            omit_sampling_parameters=True, transport_attempts=2, transport_retry_delay=1.0,
        )
    rows = []
    started = time.time()
    try:
        for task in tasks:
            path = args.output / "episodes" / f"{int(task['position']):06d}-{task['task_id']}.json"
            if path.is_file() and args.resume:
                row = json.loads(path.read_text(encoding="utf-8"))
                rows.append(row)
                continue
            transport_before = backend_transport_stats(backend)
            raw_env = WebshopEnv(WebshopEnvConfig(seed_root=str(args.seed_root), num_products=1000, seed=int(task["worker_seed"])))
            try:
                episode = run_episode(
                    FixedEnvironmentSeed(raw_env, int(task["worker_seed"])), backend, task,
                    seed=args.generation_seed_base + int(task["position"]),
                    temperature=args.temperature, max_steps=args.max_steps, max_new=args.max_new,
                    top_p=args.top_p, top_k=args.top_k, repetition_penalty=args.repetition_penalty,
                )
                error = None
            except BaseException as exc:  # preserve runtime failures separately
                episode = {"episode": int(task["source_session_index"]), "steps": 0, "task_score": 0.0,
                           "won": False, "trace": [], "trace_complete": False, "trainable": False}
                error = f"{type(exc).__name__}: {exc}"
            finally:
                raw_env.close()
            trace = episode["trace"]
            identity_ok = bool(trace and task["instruction"] in trace[0].get("prompt", ""))
            parser_failure = bool(error) or any(item.get("parser_failure") for item in trace)
            transport_after = backend_transport_stats(backend)
            transport_attempts = transport_after["attempts"] - transport_before["attempts"]
            transport_retries = transport_after["retries"] - transport_before["retries"]
            row = {
                "schema": RUN_SCHEMA, "schedule_position": int(task["position"]), "stage": task["stage"],
                "block_index": task.get("block_index"), "round_index": task.get("round_index"),
                "task_id": task["task_id"], "worker_seed": int(task["worker_seed"]),
                "source_session_index": int(task["source_session_index"]),
                "instruction_sha256": task["instruction_sha256"],
                "rollout_seed": args.generation_seed_base + int(task["position"]),
                "final_task_score": float(episode.get("task_score", 0.0)), "won": bool(episode.get("won")),
                "identity_ok": identity_ok, "trace_complete": bool(episode.get("trace_complete")),
                "parser_or_fallback_failure": parser_failure, "parser_repair_used": bool(episode.get("parser_repair_used")),
                "terminated_on_invalid_action": bool(episode.get("terminated_on_invalid_action")),
                "task_bearing_budget_consumed": not bool(error), "engineering_invalid": bool(error),
                "gateway_http_attempts": transport_attempts, "gateway_transport_retries": transport_retries,
                "qualified_positive": bool(identity_ok and episode.get("trace_complete") and episode.get("trainable")
                                            and not parser_failure and not error),
                "error": error, "episode": episode,
                "teacher_mode": args.backend, "teacher_identity": args.teacher_identity,
            }
            write_json(path, row)
            rows.append(row)
        rows.sort(key=lambda row: row["schedule_position"])
        (args.output / "attempts.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        write_json(args.output / "gpu-after.json", gpu_snapshot(args.gpu_uuid) if args.backend == "transformers" else {"status": "not-applicable", "gpu_workload_launched": False})
        write_json(args.output / "summary.json", {
            "schema": RUN_SCHEMA, "status": "pass", "stage": args.stage,
            "start_position": args.start_position, "end_position": args.end_position,
            "attempt_count": len(rows), "qualified_positive_count": sum(int(r["qualified_positive"]) for r in rows),
            "valid_count": sum(int(r["identity_ok"] and r["trace_complete"] and not r["parser_or_fallback_failure"] and not r["error"]) for r in rows),
            "task_bearing_attempt_count": sum(int(r["task_bearing_budget_consumed"]) for r in rows),
            "engineering_invalid_count": sum(int(r["engineering_invalid"]) for r in rows),
            "invalid_action_termination_count": sum(int(r["terminated_on_invalid_action"]) for r in rows),
            "parser_repair_count": sum(int(r["parser_repair_used"]) for r in rows),
            "teacher_mode": args.backend, "teacher_identity": args.teacher_identity,
            "gateway_transport": backend_transport_stats(backend),
            "wall_time_seconds": time.time() - started,
        })
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, required=True); parser.add_argument("--start-position", type=int, required=True)
    parser.add_argument("--end-position", type=int, required=True); parser.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    parser.add_argument("--seed-root", type=Path, required=True); parser.add_argument("--model", required=True); parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapter", required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--gpu-uuid", default="not-applicable")
    parser.add_argument("--backend", choices=("transformers", "gateway"), default="transformers"); parser.add_argument("--teacher-model", default=""); parser.add_argument("--teacher-base-url")
    parser.add_argument("--teacher-identity", required=True); parser.add_argument("--generation-seed-base", type=int, default=145000000)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=15); parser.add_argument("--max-new", type=int, default=512); parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
