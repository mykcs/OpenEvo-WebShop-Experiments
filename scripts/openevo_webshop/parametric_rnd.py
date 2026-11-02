#!/usr/bin/env python3
"""Bounded, seed-paired WebShop evidence for the parametric R&D gate.

This is benchmark-side code.  It never changes OpenEvo Core and records every
attempt, including zero/partial/parser failures, before classifying episodes.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


CONSUMED = [612, 566, 1657, 2655, 3420, 3768, 4513, 5216, 5457, 6624]
TRAIN = [612, 566, 1657, 2655, 3420, 3768, 4513, 5216, 5457, 6624]
REGRESSION = [5457, 612]
HARVEST_SEEDS = [202600, 202601, 202602]
DEV_COUNT = 4
ENV_SEED = 2026
MAX_STEPS = 15
MAX_NEW = 512
TEMPERATURE = 0.4
PREFIX_CANDIDATES = 3
RETRY_SEED_OFFSET = 900000


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_json(value: Any) -> str:
    return digest_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def all_goals(seed_root: Path) -> list[dict[str, Any]]:
    root = seed_root.resolve()
    webshop = root / "agent_system/environments/env_package/webshop/webshop"
    sys.path[:0] = [str(root), str(webshop)]
    os.chdir(webshop)
    import gym  # type: ignore
    from web_agent_site.envs import WebAgentTextEnv  # type: ignore

    env = gym.make(
        "WebAgentTextEnv-v0",
        observation_mode="text",
        file_path=str(webshop / "data/items_shuffle_1000.json"),
        attr_path=str(webshop / "data/items_ins_v2_1000.json"),
        num_products=1000,
        seed=ENV_SEED,
        disable_env_checker=True,
    )
    try:
        return list(env.server.goals)
    finally:
        env.close()


def choose_dev_indices(goal_count: int) -> list[int]:
    excluded = set(CONSUMED) | set(range(500))
    pool = [idx for idx in range(500, goal_count) if idx not in excluded]
    random.Random(20260813).shuffle(pool)
    return sorted(pool[:DEV_COUNT])


def build_manifest(seed_root: Path, output: Path) -> dict[str, Any]:
    goals = all_goals(seed_root)
    dev = choose_dev_indices(len(goals))
    rows: dict[str, list[dict[str, Any]]] = {}
    for split, indices in (("train", TRAIN), ("dev", dev), ("regression", REGRESSION)):
        rows[split] = []
        for i, goal_idx in enumerate(indices):
            goal = goals[goal_idx]
            text = str(goal["instruction_text"])
            rows[split].append({
                "sample_index": i,
                "task_id": f"webshop_{goal_idx:06d}",
                "goal_idx": goal_idx,
                "split": split,
                "task_description": text,
                "task_description_sha256": hashlib.sha256(text.encode()).hexdigest(),
            })
    payload: dict[str, Any] = {
        "schema": "seed3090.webshop-parametric-rnd-manifest.v1",
        "run_kind": "parametric_rnd_diagnostic",
        "source": {
            "seed3090_main": "e772294fef017fa921d697fee7a7cbc5c707fd0d",
            "official_open_evo_stable": "796e6248a61112a61c880885844d3b0420c0b6ae",
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "model_revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        },
        "benchmark": {
            "name": "webshop",
            "catalog": "indexes_1k",
            "num_products": 1000,
            "environment_seed": ENV_SEED,
            "max_steps": MAX_STEPS,
            "max_new_tokens": MAX_NEW,
            "temperature": TEMPERATURE,
            "prompt_contract": "seed3090.webshop-shared-prompt.v2",
            "action_contract": "seed3090.webshop-action-parser.v1",
            "runner": "openevo_webshop.runner.run_episode",
            "paired_order": "even before-after, odd after-before",
            "algorithm_variant": "parser_safe_retry_v1",
            "parser_retry": "one deterministic alternate seed only when the original completion is malformed",
            "parser_retry_seed_offset": RETRY_SEED_OFFSET,
            "environment_seed_source": "fixed_environment_seed_wrapper",
        },
        "harvest": {
            "tasks": TRAIN,
            "rollout_seeds": HARVEST_SEEDS,
            "max_attempts_per_task": 3,
            "qualification": "task identity + final trace score > 0 + complete trace + no fallback/parser/API error",
        },
        "train": rows["train"],
        "dev": rows["dev"],
        "regression": rows["regression"],
        "formal_held_out": {
            "state": "UNSEEN",
            "task_ids": "withheld",
            "selection_rule": "future immutable extraction from remaining pool; no text access in this run",
        },
    }
    payload["manifest_sha256"] = digest_json(payload)
    atomic_json(output, payload)
    return payload


def gpu_snapshot(expected_uuid: str | None = None) -> dict[str, Any]:
    def run(args: list[str]) -> str:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()

    rows = run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"])
    app_cmd = ["nvidia-smi"]
    if expected_uuid:
        app_cmd += ["-i", expected_uuid]
    app_cmd += ["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]
    apps = run(app_cmd)
    return {"gpu_rows": rows.splitlines(), "compute_apps": apps.splitlines(), "expected_uuid": expected_uuid}


class FixedEnvironmentSeed:
    def __init__(self, env: Any, environment_seed: int) -> None:
        self._env = env
        self._environment_seed = int(environment_seed)

    def reset(self, idx: int, seed: int | None = None):
        return self._env.reset(idx, seed=self._environment_seed)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


class FirstStepPrefixSelector:
    def __init__(self, backend: Any, candidate_count: int = PREFIX_CANDIDATES) -> None:
        self._backend = backend
        self._candidate_count = int(candidate_count)
        self._first_call = True
        self.candidates: list[dict[str, Any]] = []

    @property
    def model_name(self) -> str:
        return self._backend.model_name

    @staticmethod
    def _goal_terms(prompt: str) -> set[str]:
        import re
        task = re.search(r"Your task is: (.+?)\.", prompt, flags=re.IGNORECASE | re.DOTALL)
        text = task.group(1) if task else prompt
        return {token for token in re.findall(r"[a-z0-9]{3,}", text.lower()) if token not in {"the", "and", "with", "for", "price", "less", "than"}}

    @classmethod
    def _rank(cls, raw: str, prompt: str, offset: int) -> tuple[int, int, int, int]:
        import re
        from scripts.openevo_webshop.webshop_comparison_contract import parse_webshop_action
        action, valid = parse_webshop_action(raw, require_think=True)
        query = action[len("search["):-1] if action.startswith("search[") and action.endswith("]") else action
        overlap = len(cls._goal_terms(prompt) & set(re.findall(r"[a-z0-9]{3,}", query.lower())))
        return (int(valid), int(action.startswith("search[")), overlap, -offset)

    def generate(self, prompt: str, max_new: int = 128, temperature: float = 0.4, generation_seed: int | None = None) -> str:
        if not self._first_call or self._candidate_count <= 1:
            self._first_call = False
            return self._backend.generate(prompt, max_new=max_new, temperature=temperature, generation_seed=generation_seed)
        self._first_call = False
        candidates: list[tuple[tuple[int, int, int, int], str, dict[str, Any]]] = []
        for offset in range(self._candidate_count):
            seed = None if generation_seed is None else int(generation_seed) * 1000 + offset
            raw = self._backend.generate(prompt, max_new=max_new, temperature=temperature, generation_seed=seed)
            rank = self._rank(raw, prompt, offset)
            action, valid = __import__("scripts.openevo_webshop.webshop_comparison_contract", fromlist=["parse_webshop_action"]).parse_webshop_action(raw, require_think=True)
            candidates.append((rank, raw, {"offset": offset, "generation_seed": seed, "action": action, "action_valid": bool(valid), "rank": rank, "raw_completion": raw}))
        candidates.sort(key=lambda item: item[0], reverse=True)
        self.candidates = [item[2] for item in candidates]
        return candidates[0][1]


class ParserSafeRetryBackend:
    def __init__(self, backend: Any, retry_seed_offset: int = RETRY_SEED_OFFSET, action_contract: str = "seed3090.webshop-action-parser.v1") -> None:
        self._backend = backend
        self._retry_seed_offset = int(retry_seed_offset)
        self._action_contract = action_contract
        self.events: list[dict[str, Any]] = []

    @property
    def model_name(self) -> str:
        return self._backend.model_name

    def generate(self, prompt: str, max_new: int = 128, temperature: float = 0.4, generation_seed: int | None = None) -> str:
        from scripts.openevo_webshop.webshop_comparison_contract import (
            ACTION_COMMAND_CONTRACT,
            TOLERANT_ACTION_CONTRACT,
            parse_webshop_action,
            parse_action_command_contract,
            repair_action_command,
            repair_missing_action_close,
        )
        raw = self._backend.generate(prompt, max_new=max_new, temperature=temperature, generation_seed=generation_seed)
        prepared = raw
        repair_applied = False
        repair_event_type = "missing-closing-action-tag"
        if self._action_contract == TOLERANT_ACTION_CONTRACT:
            prepared, action, repair_applied = repair_missing_action_close(raw)
            valid = bool(action and parse_action_command_contract(prepared, require_think=True)[1])
        elif self._action_contract == ACTION_COMMAND_CONTRACT:
            repair_event_type = "action-command-recovery"
            prepared, action, repair_applied = repair_action_command(raw)
            valid = bool(action and parse_action_command_contract(prepared, require_think=True)[1])
        else:
            action, valid = parse_webshop_action(raw, require_think=True)
        if repair_applied:
            self.events.append({
                "event_type": repair_event_type,
                "generation_seed": generation_seed,
                "action": action,
                "original_valid": False,
                "repaired_valid": True,
                "original_raw_completion": raw,
                "repaired_raw_completion": prepared,
            })
        if valid:
            return prepared
        retry_seed = None if generation_seed is None else int(generation_seed) + self._retry_seed_offset
        retry_raw = self._backend.generate(prompt, max_new=max_new, temperature=temperature, generation_seed=retry_seed)
        if self._action_contract == ACTION_COMMAND_CONTRACT:
            retry_action, retry_valid = parse_action_command_contract(retry_raw, require_think=True)
        else:
            retry_action, retry_valid = parse_webshop_action(retry_raw, require_think=True)
        self.events.append({
            "event_type": "parser-retry",
            "generation_seed": generation_seed,
            "retry_seed": retry_seed,
            "action": action,
            "retry_action": retry_action,
            "original_valid": bool(valid),
            "retry_valid": bool(retry_valid),
            "original_raw_completion": raw,
            "retry_raw_completion": retry_raw,
        })
        return retry_raw if retry_valid else raw


def normalize_episode_contract(episode: dict[str, Any]) -> None:
    for step in episode.get("trace", []) or []:
        if isinstance(step, dict):
            step["environment_seed"] = ENV_SEED


def run_harvest(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    task_rows = manifest.get("train") or manifest.get("tasks") or []
    selected = [row for row in task_rows if int(row["goal_idx"]) in args.tasks]
    if not selected:
        raise SystemExit("no selected train tasks")
    rollout_seeds = tuple(
        int(seed) for seed in (getattr(args, "rollout_seeds", None) or HARVEST_SEEDS)
    )
    max_steps = int(getattr(args, "max_steps", MAX_STEPS))
    max_new = int(getattr(args, "max_new", MAX_NEW))
    temperature = float(getattr(args, "temperature", TEMPERATURE))
    if not rollout_seeds:
        raise SystemExit("at least one rollout seed is required")
    args.output.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_root = Path(os.environ.get("OPENEVO_WEBSHOP_GPU_LOCK_ROOT", "/tmp/openevo-webshop-gpu-locks"))
    lock_path = args.lock or lock_root / f"GPU-{args.gpu_uuid}.lock"
    lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"GPU lock busy: {lock_path}") from exc
        before = gpu_snapshot(args.gpu_uuid)
        atomic_json(args.output / "gpu-before.json", before)
        if before["compute_apps"]:
            raise SystemExit("compute process exists before model load")
        from openevo_webshop.env_compat import WebshopEnv, WebshopEnvConfig
        from openevo_webshop.inference import make_backend
        from openevo_webshop.runner import run_episode
        backend = make_backend(
            "transformers",
            args.model,
            device="cuda",
            revision=getattr(args, "model_revision", None),
            local_files_only=True,
        )
        started = time.time()
        attempts_path = args.output / "attempts.jsonl"
        with attempts_path.open("w", encoding="utf-8") as attempts:
            for row in selected:
                for seed in rollout_seeds:
                    raw_env = WebshopEnv(WebshopEnvConfig(seed_root=str(args.seed_root), num_products=1000, seed=ENV_SEED))
                    env = FixedEnvironmentSeed(raw_env, ENV_SEED)
                    safe_backend = ParserSafeRetryBackend(backend)
                    try:
                        try:
                            result = run_episode(env, safe_backend, int(row["goal_idx"]), max_steps=max_steps, max_new=max_new, temperature=temperature, generation_seed=seed)
                            episode = result.__dict__
                            normalize_episode_contract(episode)
                            if safe_backend.events:
                                episode["parser_retry_events"] = safe_backend.events
                                for event in safe_backend.events:
                                    for step in episode.get("trace", []) or []:
                                        if step.get("generation_seed") == event.get("generation_seed"):
                                            step["generation_seed"] = event.get("retry_seed")
                                            step["parser_retry"] = True
                            error = None
                        except BaseException as exc:
                            episode = {"episode": int(row["goal_idx"]), "rollout_seed": seed, "trace": [], "task_score": 0.0, "won": False, "error": True}
                            error = f"{type(exc).__name__}: {exc}"
                    finally:
                        raw_env.close()
                    trace = episode.get("trace") if isinstance(episode.get("trace"), list) else []
                    first_prompt = trace[0].get("prompt", "") if trace else ""
                    final_score = float(trace[-1].get("task_score", 0.0)) if trace else 0.0
                    max_score = float(episode.get("task_score", 0.0) or 0.0)
                    parser_failure = bool(episode.get("error")) or any(bool(s.get("used_fallback")) or not bool(s.get("action_valid")) for s in trace if isinstance(s, dict))
                    identity_ok = bool(task_rows) and row["task_description"] in first_prompt and hashlib.sha256(row["task_description"].encode()).hexdigest() == row["task_description_sha256"]
                    complete = bool(trace) and len(trace) <= max_steps and len(trace) == int(episode.get("steps", len(trace)))
                    qualified = bool(identity_ok and complete and not parser_failure and final_score > 0.0 and not error)
                    record = {
                        "attempt_id": f"harvest:{row['task_id']}:{seed}",
                        "task_id": row["task_id"], "goal_idx": row["goal_idx"], "rollout_seed": seed,
                        "source_reward": max_score, "final_task_score": final_score,
                        "partial_positive": max_score > 0.0 and final_score <= 0.0,
                        "identity_ok": bool(identity_ok), "trace_complete": complete,
                        "parser_or_fallback_failure": parser_failure,
                        "parser_retry_used": bool(safe_backend.events),
                        "parser_retry_events": len(safe_backend.events),
                        "qualified_positive": qualified,
                        "error": error, "episode": episode,
                    }
                    ep_path = args.output / "episodes" / f"{row['task_id']}-{seed}.json"
                    atomic_json(ep_path, record)
                    attempts.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    attempts.flush()
            after = gpu_snapshot(args.gpu_uuid)
            atomic_json(args.output / "gpu-after.json", after)
        backend._model = None
        backend._tok = None
        summary = {
            "schema": "seed3090.webshop-parametric-harvest.v1",
            "manifest_sha256": manifest["manifest_sha256"],
            "gpu_uuid": args.gpu_uuid, "tasks": [row["task_id"] for row in selected],
            "rollout_seeds": list(rollout_seeds),
            "max_steps": max_steps, "max_new_tokens": max_new, "temperature": temperature,
            "attempt_count": len(selected) * len(rollout_seeds),
            "wall_time_seconds": time.time() - started,
        }
        atomic_json(args.output / "summary.json", summary)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--seed-root", type=Path, required=True)
    m.add_argument("--output", type=Path, required=True)
    h = sub.add_parser("harvest")
    h.add_argument("--manifest", type=Path, required=True)
    h.add_argument("--seed-root", type=Path, required=True)
    h.add_argument("--model", type=str, required=True)
    h.add_argument("--model-revision")
    h.add_argument("--output", type=Path, required=True)
    h.add_argument("--gpu-uuid", required=True)
    h.add_argument("--lock", type=Path)
    h.add_argument("--tasks", type=int, nargs="+", required=True)
    h.add_argument("--rollout-seeds", type=int, nargs="+")
    h.add_argument("--max-steps", type=int, default=MAX_STEPS)
    h.add_argument("--max-new", type=int, default=MAX_NEW)
    h.add_argument("--temperature", type=float, default=TEMPERATURE)
    args = parser.parse_args()
    if args.command == "manifest":
        payload = build_manifest(args.seed_root, args.output)
        print(json.dumps({"manifest": str(args.output), "manifest_sha256": payload["manifest_sha256"], "dev_task_count": len(payload["dev"])}, sort_keys=True))
        return 0
    return run_harvest(args)


if __name__ == "__main__":
    raise SystemExit(main())
