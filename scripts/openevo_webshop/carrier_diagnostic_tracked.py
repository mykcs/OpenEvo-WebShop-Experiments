#!/usr/bin/env python3
"""Run one frozen WebShop carrier-diagnostic cell.

The benchmark-side runner compares no experience, deterministic text memory,
and a reloaded official SD-LoRA artifact.  The memory bank is train-only and
is never mutated during evaluation.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from openevo_webshop.prompts import COMPACT_PROMPT_CONTRACT
from parametric_rnd import (
    ENV_SEED,
    FixedEnvironmentSeed,
    ParserSafeRetryBackend,
    gpu_snapshot,
    normalize_episode_contract,
)
from wandb_tracking import add_wandb_arguments, start_wandb_run
from scripts.openevo_webshop.webshop_comparison_contract import (
    ACTION_CONTRACT,
    extract_goal,
    fallback_webshop_action,
    parse_webshop_action,
    render_compact_webshop_prompt,
)


TOKEN_RE = re.compile(r"[a-z0-9]+")
ALLOWED_ACTIONS_RE = re.compile(r"Allowed actions:\s*\n\[(.*?)\]\s*\n", re.DOTALL)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_digest(path: Path) -> str | None:
    if path.name == "none":
        return None
    required = ("adapter_config.json", "adapter_model.safetensors", "openevo_sd_lora_state.json", "openevo_sd_lora_state.safetensors")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(f"adapter is incomplete: {', '.join(missing)}")
    return _digest({name: _sha256_file(path / name) for name in required})


def _adapter_path(value: str | Path) -> Path | None:
    """Normalize the CLI sentinel/path before filesystem operations."""

    return None if str(value) == "none" else Path(value)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _selected_tasks(manifest_path: Path, task_ids: list[int]) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("tasks") or manifest.get("train") or []
    selected = [row for row in rows if int(row["goal_idx"]) in task_ids]
    if len(selected) != len(task_ids) or len({int(row["goal_idx"]) for row in selected}) != len(task_ids):
        raise ValueError("carrier task selection is missing or duplicated")
    for row in selected:
        description = row.get("task_description")
        if not isinstance(description, str) or hashlib.sha256(description.encode()).hexdigest() != row.get("task_description_sha256"):
            raise ValueError(f"task description identity mismatch for {row.get('goal_idx')}")
    return selected


def _tokens(text: str) -> set[str]:
    return {token for token in TOKEN_RE.findall(text.lower()) if len(token) > 1}


def _allowed_actions(prompt: str) -> set[str]:
    match = ALLOWED_ACTIONS_RE.search(prompt)
    if match is None:
        return set()
    return set(re.findall(r"'([^']+)'", match.group(1)))


@dataclasses.dataclass(frozen=True)
class MemoryEntry:
    event_id: str
    prompt: str
    action: str
    tokens: frozenset[str]


class RecoveryMemory:
    """Frozen deterministic top-1 retrieval over the shared recovery corpus."""

    def __init__(self, records_path: Path, *, max_prompt_chars: int) -> None:
        self.records_path = records_path.resolve()
        self.records_sha256 = _sha256_file(self.records_path)
        self.max_prompt_chars = int(max_prompt_chars)
        entries: list[MemoryEntry] = []
        for line_number, line in enumerate(self.records_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            traces = record.get("traces")
            if not isinstance(traces, list) or len(traces) != 1:
                raise ValueError(f"{self.records_path}:{line_number}: expected one trace")
            trace = traces[0]
            prompt_messages = trace.get("prompt_messages")
            response_messages = trace.get("response_messages")
            if not isinstance(prompt_messages, list) or not isinstance(response_messages, list):
                raise ValueError(f"{self.records_path}:{line_number}: malformed trace messages")
            prompt = str(prompt_messages[0].get("content", ""))
            action = str(response_messages[0].get("content", ""))
            _, valid = parse_webshop_action(f"<think>memory</think><action>{action}</action>")
            if not prompt.strip() or not valid:
                raise ValueError(f"{self.records_path}:{line_number}: non-canonical memory record")
            event_id = str(record.get("event_id", ""))
            if not event_id:
                raise ValueError(f"{self.records_path}:{line_number}: missing event_id")
            entries.append(MemoryEntry(event_id, prompt, action, frozenset(_tokens(prompt))))
        if not entries:
            raise ValueError("recovery memory bank is empty")
        if len({entry.event_id for entry in entries}) != len(entries):
            raise ValueError("recovery memory bank contains duplicate event_id")
        self.entries = tuple(sorted(entries, key=lambda entry: entry.event_id))

    def retrieve(self, prompt: str) -> tuple[MemoryEntry, float]:
        query = _tokens(prompt)
        if not query:
            return self.entries[0], 0.0
        ranked: list[tuple[float, str, MemoryEntry]] = []
        for entry in self.entries:
            denominator = math.sqrt(len(query) * len(entry.tokens)) or 1.0
            score = len(query & entry.tokens) / denominator
            ranked.append((score, entry.event_id, entry))
        score, _, entry = min(ranked, key=lambda row: (-row[0], row[1]))
        return entry, float(score)

    def injection(self, prompt: str, *, action_policy: str = "raw") -> tuple[str, dict[str, Any]]:
        if action_policy not in {"raw", "admissible-only"}:
            raise ValueError(f"unknown memory action policy: {action_policy}")
        entry, score = self.retrieve(prompt)
        source_prompt = entry.prompt[: self.max_prompt_chars]
        action_admissible = entry.action in _allowed_actions(prompt)
        if action_policy == "admissible-only" and not action_admissible:
            action_text = "Recorded recovery action: (not admissible in the current state; use the state only)"
        else:
            action_text = f"Recorded recovery action: {entry.action}"
        text = (
            "[experience]\n"
            "A similar train-only recovery state and its recorded recovery action are below. "
            "Use it only as experience; obey the current task and available actions.\n"
            f"Similar state:\n{source_prompt}\n"
            f"{action_text}\n"
            "[/experience]"
        )
        return text, {
            "event_id": entry.event_id,
            "similarity": score,
            "action": entry.action,
            "action_admissible": action_admissible,
            "action_injected": action_policy == "raw" or action_admissible,
            "action_policy": action_policy,
            "source_prompt_sha256": hashlib.sha256(entry.prompt.encode()).hexdigest(),
            "memory_records_sha256": self.records_sha256,
        }


def _action_counts(episode: dict[str, Any]) -> dict[str, int]:
    counts = {name: 0 for name in ("search", "click", "select", "buy", "other")}
    for step in episode.get("trace", []) or []:
        action = str(step.get("action") or "").lower()
        name = next((candidate for candidate in counts if candidate != "other" and action.startswith(candidate + "[")), "other")
        counts[name] += 1
    return counts


def _run_episode(
    env: Any,
    backend: Any,
    idx: int,
    *,
    carrier: str,
    memory: RecoveryMemory | None,
    action_policy: str,
    max_steps: int,
    max_new: int,
    temperature: float,
    generation_seed: int,
) -> dict[str, Any]:
    obs, info = env.reset(idx, seed=ENV_SEED)
    del info
    goal = extract_goal(obs)
    history: list[str] = []
    trace: list[dict[str, Any]] = []
    won = False
    task_score = 0.0
    total_reward = 0.0
    for step_index in range(max_steps):
        available_actions = env.get_available_actions()
        prompt = render_compact_webshop_prompt(
            task_description=goal or "buy the required product",
            current_observation=obs,
            available_actions=available_actions,
            step_count=step_index,
            history_length=2,
            action_history=" | ".join(history[-2:]) or "(none)",
            max_prompt_chars=20_000,
        )
        retrieval = None
        if carrier == "text-memory":
            if memory is None:
                raise ValueError("TEXT-MEMORY requires a memory bank")
            evolution_text, retrieval = memory.injection(prompt, action_policy=action_policy)
            prompt = f"{prompt}\n\n{evolution_text}\n"
        step_seed = int(generation_seed) + step_index
        raw = backend.generate(prompt, max_new=max_new, temperature=temperature, generation_seed=step_seed)
        action, action_valid = parse_webshop_action(raw, require_think=True)
        used_fallback = not action_valid
        if used_fallback:
            action = fallback_webshop_action(goal)
        obs, reward, done, info = env.step(action)
        info = info or {}
        score = float(info.get("task_score", reward) or 0.0)
        task_score = max(task_score, score)
        won = bool(info.get("won", reward == 1.0)) if done else won
        if done:
            total_reward = 10.0 if won else 0.0
        history.append(f"obs:{str(obs)[:60]} -> act:{action}")
        trace.append({
            "step": step_index,
            "prompt": prompt,
            "raw_completion": raw,
            "action": action,
            "action_valid": bool(action_valid),
            "used_fallback": used_fallback,
            "parser_failure": not action_valid,
            "generation_seed": step_seed,
            "environment_seed": ENV_SEED,
            "task_score": score,
            "carrier": carrier,
            "retrieval": retrieval,
        })
        if done:
            break
    return {
        "episode": idx,
        "goal": goal[:80],
        "steps": len(trace),
        "task_score": round(task_score, 3),
        "won": won,
        "reward": total_reward,
        "error": False,
        "prompt_contract": COMPACT_PROMPT_CONTRACT,
        "action_contract": ACTION_CONTRACT,
        "trace": trace,
    }


def run(args: argparse.Namespace) -> int:
    tasks = _selected_tasks(args.task_manifest.resolve(), args.tasks)
    memory = RecoveryMemory(args.memory_records, max_prompt_chars=args.max_memory_prompt_chars)
    adapter_path = _adapter_path(args.adapter)
    adapter_digest = _tree_digest(adapter_path) if adapter_path is not None else None
    if args.arm == "parametric" and adapter_path is None:
        raise ValueError("PARAMETRIC requires a non-none adapter")
    if args.arm != "parametric" and adapter_path is not None:
        raise ValueError("BASE and TEXT-MEMORY must not load an adapter")
    if args.memory_action_policy != "raw" and args.arm != "text-memory":
        raise ValueError("non-raw memory action policy requires TEXT-MEMORY")
    args.output.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = args.lock or args.output / f"GPU-{args.gpu_uuid}.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"GPU lock busy: {lock_path}") from exc
        visible = [token.strip() for token in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if token.strip()]
        if visible != [args.gpu_uuid]:
            raise SystemExit(f"CUDA_VISIBLE_DEVICES must contain exactly {args.gpu_uuid}; observed {visible!r}")
        before = gpu_snapshot(args.gpu_uuid)
        _atomic_json(args.output / "gpu-before.json", before)
        if before["compute_apps"]:
            raise SystemExit("compute process exists before carrier model load")
        from openevo_webshop.env_compat import WebshopEnv, WebshopEnvConfig
        from openevo_webshop.inference import make_backend

        backend = make_backend(
            "transformers",
            args.model,
            device="cuda",
            revision=args.model_revision,
            adapter_path=adapter_path,
            local_files_only=True,
        )
        protocol = {
            "schema": "seed3090.webshop-h1.20-carrier-cell.v1",
            "arm": args.arm,
            "model": args.model,
            "model_revision": args.model_revision,
            "task_manifest_sha256": _sha256_file(args.task_manifest),
            "memory_records_sha256": memory.records_sha256,
            "memory_record_count": len(memory.entries),
            "adapter_digest": adapter_digest,
            "tasks": args.tasks,
            "rollout_seeds": args.rollout_seeds,
            "environment_seed": ENV_SEED,
            "temperature": args.temperature,
            "max_steps": args.max_steps,
            "max_new_tokens": args.max_new,
            "prompt_contract": COMPACT_PROMPT_CONTRACT,
            "action_contract": ACTION_CONTRACT,
            "retrieval": "top-1 token-set cosine; event_id tie-break",
            "memory_action_policy": args.memory_action_policy,
            "gpu_uuid": args.gpu_uuid,
        }
        protocol_fingerprint = _digest(protocol)
        tracker = start_wandb_run(
            mode=args.wandb_mode,
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.arm}-{args.memory_action_policy}-carrier",
            job_type="evaluation",
            group=args.wandb_group or "H1.15-recovery-corpus-carrier-diagnostic-v1",
            config={**protocol, "experiment_git_sha": args.experiment_git_sha, "upstream_openevo_sha": args.upstream_openevo_sha, "runtime_fingerprint": args.runtime_fingerprint, "image_digest": args.image_digest},
            tags=("benchmark:webshop", "phase:h1.20", "class:carrier-diagnostic", f"arm:{args.arm}", f"memory-policy:{args.memory_action_policy}"),
            notes="Frozen carrier diagnostic; evaluation is teacher-free and held out.",
            directory=args.wandb_dir or args.output / "wandb",
        )
        attempts_path = args.output / "attempts.jsonl"
        rows: list[dict[str, Any]] = []
        started = time.time()
        try:
            from openevo_webshop.env_compat import WebshopEnv, WebshopEnvConfig

            with attempts_path.open("w", encoding="utf-8") as attempts:
                for task in tasks:
                    for seed in args.rollout_seeds:
                        raw_env = WebshopEnv(WebshopEnvConfig(seed_root=str(args.seed_root), num_products=1000, seed=ENV_SEED))
                        env = FixedEnvironmentSeed(raw_env, ENV_SEED)
                        safe_backend = ParserSafeRetryBackend(backend)
                        error = None
                        try:
                            episode = _run_episode(env, safe_backend, int(task["goal_idx"]), carrier=args.arm, memory=memory if args.arm == "text-memory" else None, action_policy=args.memory_action_policy, max_steps=args.max_steps, max_new=args.max_new, temperature=args.temperature, generation_seed=int(seed))
                            normalize_episode_contract(episode)
                            if safe_backend.events:
                                episode["parser_retry_events"] = safe_backend.events
                        except BaseException as exc:  # noqa: BLE001 - preserve runtime failures
                            episode = {"episode": int(task["goal_idx"]), "trace": [], "task_score": 0.0, "won": False, "error": True}
                            error = f"{type(exc).__name__}: {exc}"
                        finally:
                            raw_env.close()
                        trace = episode.get("trace") if isinstance(episode.get("trace"), list) else []
                        first_prompt = trace[0].get("prompt", "") if trace else ""
                        score = float(episode.get("task_score", 0.0) or 0.0)
                        parser_failure = bool(episode.get("error")) or any(bool(step.get("used_fallback")) or not bool(step.get("action_valid")) for step in trace if isinstance(step, dict))
                        identity_ok = task["task_description"] in first_prompt
                        complete = bool(trace) and len(trace) <= args.max_steps and len(trace) == int(episode.get("steps", len(trace)))
                        row = {
                            "schema": "seed3090.webshop-h1.15-carrier-episode.v1",
                            "arm": args.arm,
                            "task_id": task["task_id"],
                            "goal_idx": task["goal_idx"],
                            "rollout_seed": seed,
                            "final_task_score": score,
                            "won": bool(episode.get("won")),
                            "identity_ok": identity_ok,
                            "trace_complete": complete,
                            "parser_or_fallback_failure": parser_failure,
                            "qualified_positive": bool(identity_ok and complete and not parser_failure and score > 0 and not error),
                            "error": error,
                            "episode": episode,
                            "protocol_fingerprint": protocol_fingerprint,
                        }
                        _atomic_json(args.output / "episodes" / f"{task['task_id']}-{seed}.json", row)
                        attempts.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                        attempts.flush()
                        rows.append(row)
                        retrievals = [step.get("retrieval") for step in trace if step.get("retrieval")]
                        tracker.log({
                            "eval/attempt_index": len(rows),
                            "eval/final_task_score": score,
                            "eval/win": int(bool(episode.get("won"))),
                            "eval/qualified_positive": int(row["qualified_positive"]),
                            "eval/parser_failure": int(parser_failure),
                            "eval/trace_complete": int(complete),
                            "eval/episode_steps": len(trace),
                            "eval/generation_seed": seed,
                            "eval/retrieval_steps": len(retrievals),
                            "eval/retrieval_mean_similarity": sum(float(item["similarity"]) for item in retrievals) / len(retrievals) if retrievals else 0.0,
                            "eval/action_counts": _action_counts(episode),
                        })
            after = gpu_snapshot(args.gpu_uuid)
            _atomic_json(args.output / "gpu-after.json", after)
            summary = {
                "schema": "seed3090.webshop-h1.15-carrier-summary.v1",
                "arm": args.arm,
                "attempt_count": len(rows),
                "valid_count": sum(bool(r["identity_ok"] and r["trace_complete"] and not r["parser_or_fallback_failure"] and not r["error"]) for r in rows),
                "qualified_positive_count": sum(bool(r["qualified_positive"]) for r in rows),
                "mean_task_score": sum(float(r["final_task_score"]) for r in rows) / len(rows) if rows else 0.0,
                "protocol_fingerprint": protocol_fingerprint,
                "memory_records_sha256": memory.records_sha256,
                "adapter_digest": adapter_digest,
                "gpu_uuid": args.gpu_uuid,
                "wall_time_seconds": time.time() - started,
                "tracking": tracker.metadata(),
            }
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
    parser.add_argument("--memory-records", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--arm", choices=("base", "base-30", "text-memory", "parametric"), required=True)
    parser.add_argument("--tasks", type=int, nargs="+", required=True)
    parser.add_argument("--rollout-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--experiment-git-sha", required=True)
    parser.add_argument("--upstream-openevo-sha", required=True)
    parser.add_argument("--runtime-fingerprint", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--max-new", type=int, default=512)
    parser.add_argument("--max-memory-prompt-chars", type=int, default=1400)
    parser.add_argument("--memory-action-policy", choices=("raw", "admissible-only"), default="raw")
    add_wandb_arguments(parser, default_project="openevo-webshop-experiments")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
