#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

MAX_STEPS = 15
STEP_SEED_STRIDE = 100_000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    tmp.replace(path)


def pctl(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(x) for x in values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


class GenerateObserver:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.original = model.generate
        self.calls: list[dict[str, Any]] = []

    def install(self) -> None:
        self.model.generate = self._wrapped

    def close(self) -> None:
        self.model.generate = self.original

    def mark(self) -> int:
        return len(self.calls)

    def since(self, mark: int) -> list[dict[str, Any]]:
        return self.calls[mark:]

    def _wrapped(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        call_kwargs = dict(kwargs)
        input_ids = call_kwargs.get("input_ids")
        if input_ids is None:
            raise RuntimeError("diagnostic observer requires explicit input_ids")
        call_kwargs["return_dict_in_generate"] = True
        call_kwargs["output_scores"] = True
        output = self.original(*args, **call_kwargs)
        sequences = output.sequences
        scores = list(output.scores or [])
        n_input = int(input_ids.shape[1])
        generated_ids = sequences[0, n_input:n_input + len(scores)]
        entropies: list[float] = []
        nlls: list[float] = []
        for score, token_id in zip(scores, generated_ids.tolist()):
            logits = score[0].float()
            entropy = torch.distributions.Categorical(logits=logits).entropy()
            log_probs = torch.log_softmax(logits, dim=-1)
            entropies.append(float(entropy.item()))
            nlls.append(float((-log_probs[int(token_id)]).item()))
        self.calls.append({
            "generated_token_count": len(scores),
            "entropy_sum_nats": float(sum(entropies)),
            "entropy_mean_nats": statistics.mean(entropies) if entropies else None,
            "entropy_p90_nats": pctl(entropies, 0.90),
            "chosen_nll_sum_nats": float(sum(nlls)),
            "chosen_nll_mean_nats": statistics.mean(nlls) if nlls else None,
            "generated_ids_sha256": hashlib.sha256(
                json.dumps([int(x) for x in generated_ids.tolist()], separators=(",", ":")).encode()
            ).hexdigest(),
        })
        return sequences


def phase_summary(calls: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = sum(int(x["generated_token_count"]) for x in calls)
    entropy_sum = sum(float(x["entropy_sum_nats"]) for x in calls)
    nll_sum = sum(float(x["chosen_nll_sum_nats"]) for x in calls)
    return {
        "generated_token_count": tokens,
        "entropy_mean_nats": entropy_sum / tokens if tokens else None,
        "chosen_nll_mean_nats": nll_sum / tokens if tokens else None,
    }


def load_science_modules(runtime_source_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(runtime_source_root))
    from openevo_webshop.comparison_contract import extract_goal
    from openevo_webshop.env_compat import WebshopEnv, WebshopEnvConfig
    from scripts.openevo_webshop.ceiling1_harness2011_prompt import (
        PROMPT_CONTRACT_SHA256,
        render_harness2011_compact_webshop_prompt,
    )
    from scripts.openevo_webshop.ceiling1_stage2_format_hard_runtime import (
        Qwen3Stage2FormatHardRuntime,
    )
    from scripts.openevo_webshop.parametric_rnd import FixedEnvironmentSeed
    return {
        "extract_goal": extract_goal,
        "WebshopEnv": WebshopEnv,
        "WebshopEnvConfig": WebshopEnvConfig,
        "PROMPT_CONTRACT_SHA256": PROMPT_CONTRACT_SHA256,
        "render_prompt": render_harness2011_compact_webshop_prompt,
        "Runtime": Qwen3Stage2FormatHardRuntime,
        "FixedEnvironmentSeed": FixedEnvironmentSeed,
    }


def validate_runtime_files(manifest: dict[str, Any], runtime_source_root: Path) -> None:
    for rel, expected in manifest["runtime_files"].items():
        actual = sha256_file(runtime_source_root / rel)
        if actual != expected:
            raise RuntimeError(f"runtime file hash mismatch: {rel}: {actual} != {expected}")


def validate_payload(path: Path, expected_sha: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha:
        raise RuntimeError(f"{label} hash mismatch: {actual} != {expected_sha}")


def run_episode(
    modules: dict[str, Any], env: Any, runtime: Any, observer: GenerateObserver | None,
    task: dict[str, Any], base_seed: int,
) -> dict[str, Any]:
    obs, _ = env.reset(int(task["source_session_index"]), seed=int(task["worker_seed"]))
    goal = modules["extract_goal"](obs)
    observed_instruction_sha = hashlib.sha256(goal.encode()).hexdigest()
    if observed_instruction_sha != task["instruction_sha256"]:
        raise RuntimeError(
            f"instruction identity mismatch for {task['task_identity']}: "
            f"{observed_instruction_sha} != {task['instruction_sha256']}"
        )
    history: list[str] = []
    trace: list[dict[str, Any]] = []
    env_steps = 0
    for step_idx in range(MAX_STEPS):
        available = env.get_available_actions()
        prompt = modules["render_prompt"](
            task_description=goal, current_observation=obs, available_actions=available,
            step_count=step_idx, history_length=2,
            action_history=" | ".join(history[-2:]) or "(none)", max_prompt_chars=20_000,
        )
        generation_seed = int(base_seed) + step_idx * STEP_SEED_STRIDE
        mark = observer.mark() if observer else 0
        try:
            raw, receipt = runtime.generate_h2(
                prompt, available_actions=available, generation_seed=generation_seed,
                max_new_tokens=512, do_sample=True, temperature=1.0, top_p=1.0,
                top_k=0, repetition_penalty=1.0,
            )
        except RuntimeError as exc:
            if "action prefix escaped" in str(exc) or "no compilable action schema" in str(exc):
                return {"trace": trace, "steps": step_idx + 1, "score": 0.0, "won": False,
                        "invalid_terminated": True, "constraint_escape": True,
                        "environment_action_schema_violation": False, "webshop_env_step_calls": env_steps}
            raise
        calls = observer.since(mark) if observer else []
        if observer and len(calls) != 2:
            raise RuntimeError(f"expected exactly two generate calls per step, got {len(calls)}")
        action_channel = receipt.get("action_channel") or {}
        action = action_channel.get("action")
        valid = bool(action_channel.get("admissible"))
        tok = runtime.base._tok
        raw_token_count = len(tok(raw, add_special_tokens=False)["input_ids"])
        diag = None
        if observer:
            diag = {
                "reasoning": calls[0],
                "action": calls[1],
                "combined": phase_summary(calls),
                "raw_completion_token_count": raw_token_count,
                "observed_pre_action_model_tokens": action_channel.get("observed_pre_action_model_tokens"),
            }
        step = {
            "step": step_idx,
            "generation_seed": generation_seed,
            "raw_completion": raw,
            "raw_completion_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "raw_completion_token_count": raw_token_count,
            "action": action if valid else None,
            "action_valid": valid,
            "runtime_input_tokens": receipt.get("input_tokens"),
            "action_channel": action_channel,
            "generation_diagnostics": diag,
        }
        trace.append(step)
        if not valid:
            return {"trace": trace, "steps": step_idx + 1, "score": 0.0, "won": False,
                    "invalid_terminated": True, "constraint_escape": False,
                    "environment_action_schema_violation": False, "webshop_env_step_calls": env_steps}
        try:
            obs, reward, done, info = env.step(action)
            env_steps += 1
        except BaseException as exc:
            step["environment_error"] = f"{type(exc).__name__}: {exc}"
            return {"trace": trace, "steps": step_idx + 1, "score": 0.0, "won": False,
                    "invalid_terminated": True, "constraint_escape": False,
                    "environment_action_schema_violation": True, "webshop_env_step_calls": env_steps}
        info = info or {}
        score = float(info.get("task_score", reward) or 0.0)
        step["score"] = score
        history.append(f"act:{action} -> obs:{str(obs)[:80]}")
        if done:
            return {"trace": trace, "steps": step_idx + 1, "score": score,
                    "won": bool(info.get("won", reward == 1.0)), "invalid_terminated": False,
                    "constraint_escape": False, "environment_action_schema_violation": False,
                    "webshop_env_step_calls": env_steps}
    return {"trace": trace, "steps": MAX_STEPS,
            "score": max([float(x.get("score", 0.0)) for x in trace] or [0.0]),
            "won": False, "invalid_terminated": False, "constraint_escape": False,
            "environment_action_schema_violation": False, "webshop_env_step_calls": env_steps}


def summarize_rows(rows: list[dict[str, Any]], instrumented: bool) -> dict[str, Any]:
    scores = [float(r["episode"]["score"]) for r in rows]
    steps = [int(r["episode"]["steps"]) for r in rows]
    raw_tokens = [
        int(step["raw_completion_token_count"])
        for row in rows for step in row["episode"]["trace"]
    ]
    reasoning_calls = [
        step["generation_diagnostics"]["reasoning"]
        for row in rows for step in row["episode"]["trace"]
        if step.get("generation_diagnostics")
    ]
    action_calls = [
        step["generation_diagnostics"]["action"]
        for row in rows for step in row["episode"]["trace"]
        if step.get("generation_diagnostics")
    ]
    return {
        "attempt_count": len(rows),
        "mean_score": statistics.mean(scores) if scores else None,
        "success_rate": sum(bool(r["episode"]["won"]) for r in rows) / len(rows) if rows else None,
        "positive_reward_rate": sum(float(r["episode"]["score"]) > 0 for r in rows) / len(rows) if rows else None,
        "mean_episode_steps": statistics.mean(steps) if steps else None,
        "invalid_terminated_count": sum(bool(r["episode"]["invalid_terminated"]) for r in rows),
        "constraint_escape_count": sum(bool(r["episode"]["constraint_escape"]) for r in rows),
        "environment_action_schema_violation_count": sum(bool(r["episode"]["environment_action_schema_violation"]) for r in rows),
        "mean_raw_completion_tokens_per_step": statistics.mean(raw_tokens) if raw_tokens else None,
        "p90_raw_completion_tokens_per_step": pctl([float(x) for x in raw_tokens], 0.90),
        "reasoning_generation": phase_summary(reasoning_calls) if instrumented else None,
        "action_generation": phase_summary(action_calls) if instrumented else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--runtime-source-root", type=Path, required=True)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--sd-adapter", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instrumentation", choices=("off", "on"), default="on")
    parser.add_argument("--task-limit", type=int)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != args.gpu_uuid:
        raise SystemExit("CUDA_VISIBLE_DEVICES mismatch")
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("final_panel_access_allowed") is not False:
        raise RuntimeError("diagnostic manifest must explicitly forbid final-panel access")
    checkpoint = manifest["checkpoints"][args.checkpoint_id]
    validate_runtime_files(manifest, args.runtime_source_root)
    validate_payload(args.sd_adapter / "adapter_model.safetensors", checkpoint["adapter_sha256"], "adapter")
    common = manifest["common_carriers"]
    validate_payload(args.memory, common["memory_sha256"], "memory")
    validate_payload(args.skill, common["skill_sha256"], "skill")
    validate_payload(args.agent, common["agent_system_sha256"], "agent_system")
    if args.model != manifest["model"]["local_model_path"] or args.revision != manifest["model"]["revision"]:
        raise RuntimeError("model identity mismatch")
    modules = load_science_modules(args.runtime_source_root)
    runtime = modules["Runtime"](
        args.model, args.revision, sd_adapter=args.sd_adapter,
        memory=args.memory.read_text(), skill_bundle=args.skill.read_text(),
        agent_system=args.agent.read_text(),
    )
    observer: GenerateObserver | None = None
    if args.instrumentation == "on":
        observer = GenerateObserver(runtime.base._model)
        observer.install()
    rows: list[dict[str, Any]] = []
    tasks = list(manifest["tasks"])
    if args.task_limit is not None:
        tasks = tasks[: int(args.task_limit)]
    started = time.time()
    try:
        for task in tasks:
            raw_env = modules["WebshopEnv"](
                modules["WebshopEnvConfig"](
                    seed_root=str(args.seed_root), num_products=1000,
                    seed=int(task["worker_seed"]),
                )
            )
            error = None
            try:
                episode = run_episode(
                    modules,
                    modules["FixedEnvironmentSeed"](raw_env, int(task["worker_seed"])),
                    runtime,
                    observer,
                    task,
                    int(task["paired_generation_seed"]),
                )
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
                episode = {"trace": [], "steps": 0, "score": 0.0, "won": False,
                           "invalid_terminated": False, "constraint_escape": False,
                           "environment_action_schema_violation": False,
                           "webshop_env_step_calls": 0}
            finally:
                raw_env.close()
            rows.append({
                "task_identity": task["task_identity"],
                "probe_ordinal": int(task["probe_ordinal"]),
                "source_session_index": int(task["source_session_index"]),
                "worker_seed": int(task["worker_seed"]),
                "paired_generation_seed": int(task["paired_generation_seed"]),
                "checkpoint_id": args.checkpoint_id,
                "state_id": checkpoint["state_id"],
                "adapter_sha256": checkpoint["adapter_sha256"],
                "instrumentation": args.instrumentation,
                "engineering_invalid": bool(error),
                "error": error,
                "episode": episode,
            })
    finally:
        if observer:
            observer.close()
        runtime.close()
    summary = {
        "schema": "openevo.q17.advisor-fixed-checkpoint-probe.202609111535",
        "status": "PASS" if rows and not any(r["engineering_invalid"] for r in rows) else "FAIL",
        "checkpoint_id": args.checkpoint_id,
        "checkpoint": checkpoint,
        "instrumentation": args.instrumentation,
        "task_count": len(rows),
        "task_limit": args.task_limit,
        "formal_task_consumption": 0,
        "final_panel_access": 0,
        "external_teacher_calls": 0,
        "runtime_execution_sha": manifest["runtime_execution_sha"],
        "prompt_contract_sha256": modules["PROMPT_CONTRACT_SHA256"],
        "panel_sha256": manifest["panel_sha256"],
        "metrics": summarize_rows(rows, args.instrumentation == "on"),
        "wall_time_seconds": time.time() - started,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "summary.json", summary)
    with (args.output_dir / "attempts.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "status": summary["status"],
        "checkpoint_id": args.checkpoint_id,
        "instrumentation": args.instrumentation,
        **summary["metrics"],
    }, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
