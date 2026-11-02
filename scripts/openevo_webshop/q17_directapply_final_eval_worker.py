#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from openevo_webshop.comparison_contract import extract_goal
from openevo_webshop.env_compat import WebshopEnv, WebshopEnvConfig
from scripts.openevo_webshop.ceiling1_harness2011_runtime import Harness2011Runtime
from scripts.openevo_webshop.ceiling1_stage2_format_hard_runtime import Qwen3Stage2FormatHardRuntime
from scripts.openevo_webshop.ceiling1_harness2011_prompt import (
    PROMPT_CONTRACT_SHA256 as H2011_PROMPT_SHA256,
    PROMPT_PROFILE as H2011_PROMPT_PROFILE,
    render_harness2011_compact_webshop_prompt,
)
from scripts.openevo_webshop.parametric_rnd import FixedEnvironmentSeed, gpu_snapshot

EXPECTED_PANEL_FILE_SHA = "aea97afdd5b176126cf67f7ed315189c95fbf37b07263c84b947903cca69aa4d"
EXPECTED_PANEL_CONTENT_SHA = "3529b94f97491434da01f9b2adc2faf1d0571545603a788d367561ae2b529d48"
EXPECTED_PANEL_DIGEST = "cd531caff21d13d72df5c6fb013fadc10457393134d8bc3bcdc63fadf76bc453"
MAX_STEPS = 15
MAX_NEW = 512


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def load_panel(path: Path, ordinals: list[int]) -> list[dict[str, Any]]:
    if sha(path) != EXPECTED_PANEL_FILE_SHA:
        raise ValueError("final panel file SHA mismatch")
    document = json.loads(path.read_text())
    if (
        document.get("manifest_content_sha256") != EXPECTED_PANEL_CONTENT_SHA
        or document.get("selection", {}).get("selected_panel_digest") != EXPECTED_PANEL_DIGEST
    ):
        raise ValueError("final panel content identity mismatch")
    rows = document["tasks"]
    by_ordinal = {int(row["panel_ordinal"]): row for row in rows}
    if len(rows) != 128 or len(by_ordinal) != 128:
        raise ValueError("final panel count/identity mismatch")
    selected = []
    for ordinal in ordinals:
        row = by_ordinal[ordinal]
        if (
            int(row["environment_seed"]) != 0
            or hashlib.sha256(row["instruction"].encode()).hexdigest() != row["instruction_sha256"]
        ):
            raise ValueError(f"panel row identity mismatch {ordinal}")
        selected.append(row)
    return selected



def episode(env: Any, runtime: Harness2011Runtime, row: dict[str, Any]) -> dict[str, Any]:
    observation, _ = env.reset(int(row["goal_idx"]), seed=0)
    goal = extract_goal(observation)
    history: list[str] = []
    trace: list[dict[str, Any]] = []
    env_steps = 0
    for step in range(MAX_STEPS):
        available = env.get_available_actions()
        prompt = render_harness2011_compact_webshop_prompt(
            task_description=goal,
            current_observation=observation,
            available_actions=available,
            step_count=step,
            history_length=2,
            action_history=" | ".join(history[-2:]) or "(none)",
            max_prompt_chars=20_000,
        )
        generation_seed = int(row["generation_seed"]) + step
        try:
            raw, receipt = runtime.generate_h2(
                prompt,
                available_actions=available,
                generation_seed=generation_seed,
                max_new_tokens=MAX_NEW,
                do_sample=False,
                repetition_penalty=1.0,
            )
        except RuntimeError as exc:
            if "action prefix escaped" in str(exc) or "no compilable action schema" in str(exc):
                trace.append({
                    "step": step,
                    "base_prompt": prompt,
                    "generation_seed": generation_seed,
                    "constraint_escape": True,
                    "parser_repair_used": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                return {
                    "task_score": 0.0,
                    "won": False,
                    "trace": trace,
                    "trace_complete": False,
                    "terminated_on_invalid_action": True,
                    "parser_repair_used": False,
                    "interface_failure_kind": "constraint_escape",
                    "webshop_env_step_calls": env_steps,
                }
            raise
        channel = receipt.get("action_channel") or {}
        action = channel.get("action")
        valid = bool(channel.get("admissible") and action)
        step_row = {
            "step": step,
            "base_prompt": prompt,
            "prompt": receipt.get("decoded_input") or prompt,
            "runtime_messages": receipt.get("messages"),
            "runtime_sampling": receipt.get("sampling"),
            "runtime_active_adapters": receipt.get("active_adapters"),
            "runtime_adapters": receipt.get("adapters"),
            "runtime_carrier_readback": receipt.get("carrier_readback"),
            "runtime_harness": receipt.get("harness"),
            "runtime_context_receipt": receipt,
            "runtime_input_tokens": receipt.get("input_tokens"),
            "raw_completion": raw,
            "repaired_completion": None,
            "action_channel": channel,
            "action": action if valid else None,
            "action_valid": valid,
            "parser_failure": not valid,
            "parser_repair_used": False,
            "generation_seed": generation_seed,
            "constraint_escape": False,
        }
        trace.append(step_row)
        if not valid:
            return {
                "task_score": 0.0,
                "won": False,
                "trace": trace,
                "trace_complete": False,
                "terminated_on_invalid_action": True,
                "parser_repair_used": False,
                "interface_failure_kind": "hard_action_channel_invalid",
                "webshop_env_step_calls": env_steps,
            }
        observation, reward, done, info = env.step(action)
        env_steps += 1
        info = info or {}
        history.append(f"obs:{str(observation)[:60]} -> act:{action}")
        step_row["task_score"] = float(info.get("task_score", reward) or 0.0)
        step_row["environment_accepted"] = True
        if done:
            won = bool(info.get("won", reward == 1.0))
            return {
                "task_score": max(float(step_row["task_score"]), 0.0),
                "won": won,
                "trace": trace,
                "trace_complete": True,
                "terminated_on_invalid_action": False,
                "parser_repair_used": False,
                "interface_failure_kind": None,
                "webshop_env_step_calls": env_steps,
            }
    return {
        "task_score": max((float(item.get("task_score", 0.0)) for item in trace), default=0.0),
        "won": False,
        "trace": trace,
        "trace_complete": True,
        "terminated_on_invalid_action": False,
        "parser_repair_used": False,
        "interface_failure_kind": None,
        "webshop_env_step_calls": env_steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--ordinals", type=int, nargs="+", required=True)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--sd-adapter", type=Path)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--agent-system", type=Path, required=True)
    parser.add_argument("--arm", choices=("qwen25-3b", "qwen3-1p7b"), required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-state-id", required=True)
    parser.add_argument("--activation-id", required=True)
    parser.add_argument("--experiment-git-sha", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != args.gpu_uuid:
        raise SystemExit("CUDA_VISIBLE_DEVICES mismatch")

    rows = load_panel(args.panel, args.ordinals)
    memory = args.memory.read_text()
    skill = args.skill.read_text()
    agent = args.agent_system.read_text()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "episodes").mkdir(exist_ok=True)
    atomic_json(args.output / "gpu-before.json", gpu_snapshot(args.gpu_uuid))
    runtime_cls = Qwen3Stage2FormatHardRuntime if args.arm == "qwen3-1p7b" else Harness2011Runtime
    runtime = runtime_cls(
        args.model,
        args.model_revision,
        sd_adapter=args.sd_adapter,
        memory=memory,
        skill_bundle=skill,
        agent_system=agent,
    )
    started = time.time()
    results = []
    try:
        for row in rows:
            episode_path = args.output / "episodes" / f"{int(row['panel_ordinal']):03d}-{row['task_id']}.json"
            if args.resume and episode_path.is_file():
                prior = json.loads(episode_path.read_text())
                identity = (
                    prior.get("final_state_id") == args.final_state_id
                    and prior.get("activation_id") == args.activation_id
                    and prior.get("experiment_git_sha") == args.experiment_git_sha
                    and prior.get("image_digest") == args.image_digest
                )
                if not identity:
                    raise RuntimeError("final-eval resume identity drift")
                if not prior.get("engineering_invalid"):
                    results.append(prior)
                    continue
            env = WebshopEnv(WebshopEnvConfig(seed_root=str(args.seed_root), num_products=1000, seed=0))
            error = None
            try:
                result = episode(FixedEnvironmentSeed(env, 0), runtime, row)
            except BaseException as exc:
                result = {
                    "task_score": 0.0,
                    "won": False,
                    "trace": [],
                    "trace_complete": False,
                    "parser_repair_used": False,
                }
                error = f"{type(exc).__name__}: {exc}"
            finally:
                env.close()
            trace = result["trace"]
            identity_ok = bool(trace and row["instruction"] in trace[0].get("prompt", ""))
            parser_failure = bool(error) or any(not item.get("action_valid") for item in trace)
            score = float(result.get("task_score", 0.0))
            exact = bool(result.get("won") and score == 1.0)
            record = {
                "schema": "openevo.ceiling1-final-eval-row.v1",  # legacy-revision-compat: historical serialized schema identity
                "arm": args.arm,
                "panel_ordinal": int(row["panel_ordinal"]),
                "task_id": row["task_id"],
                "goal_idx": int(row["goal_idx"]),
                "environment_seed": 0,
                "generation_seed": int(row["generation_seed"]),
                "final_task_score": score,
                "exact_success": exact,
                "identity_ok": identity_ok,
                "trace_complete": bool(result.get("trace_complete")),
                "parser_or_action_failure": parser_failure,
                "parser_repair_used": bool(result.get("parser_repair_used")),
                "engineering_invalid": bool(error),
                "error": error,
                "final_state_id": args.final_state_id,
                "activation_id": args.activation_id,
                "experiment_git_sha": args.experiment_git_sha,
                "image_digest": args.image_digest,
                "chat_template_enable_thinking": False if args.arm == "qwen3-1p7b" else None,
                "harness_profile": H2011_PROMPT_PROFILE,
                "prompt_contract_sha256": H2011_PROMPT_SHA256,
                "episode": result,
            }
            atomic_json(episode_path, record)
            results.append(record)
    finally:
        runtime.close()
        atomic_json(args.output / "gpu-after.json", gpu_snapshot(args.gpu_uuid))

    results.sort(key=lambda item: int(item["panel_ordinal"]))
    invalid = sum(int(item["engineering_invalid"]) for item in results)
    valid_count = sum(int(item["identity_ok"] and item["trace_complete"] and not item["parser_or_action_failure"] and not item["engineering_invalid"]) for item in results)
    repairs = sum(int(item["parser_repair_used"]) for item in results)
    interface_failures = len(results) - valid_count - invalid
    status = "PASS" if invalid == 0 and valid_count == len(results) and repairs == 0 else ("ENGINEERING_RECOVERY_REQUIRED" if invalid else "INTERFACE_CONTRACT_FAILURE")
    summary = {
        "schema": "openevo.ceiling1-final-eval-shard.v1",  # legacy-revision-compat: historical serialized schema identity
        "status": status,
        "arm": args.arm,
        "attempt_count": len(results),
        "engineering_invalid_count": invalid,
        "exact_success_count": sum(int(item["exact_success"]) for item in results),
        "mean_task_score": sum(item["final_task_score"] for item in results) / len(results),
        "valid_count": valid_count,
        "interface_failure_count": interface_failures,
        "parser_repair_count": repairs,
        "gpu_uuid": args.gpu_uuid,
        "final_state_id": args.final_state_id,
        "activation_id": args.activation_id,
        "experiment_git_sha": args.experiment_git_sha,
        "image_digest": args.image_digest,
        "sampling": {
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "repetition_penalty": 1.0,
            "max_steps": MAX_STEPS,
            "max_new_tokens": MAX_NEW,
        },
        "panel_file_sha256": EXPECTED_PANEL_FILE_SHA,
        "chat_template_enable_thinking": False if args.arm == "qwen3-1p7b" else None,
        "harness_profile": H2011_PROMPT_PROFILE,
        "prompt_contract_sha256": H2011_PROMPT_SHA256,
        "wall_time_seconds": time.time() - started,
    }
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if status == "PASS" else (2 if invalid else 3)


if __name__ == "__main__":
    raise SystemExit(main())
