#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def split(items: list[int], count: int) -> list[list[int]]:
    quotient, remainder = divmod(len(items), count)
    result = []
    cursor = 0
    for index in range(count):
        size = quotient + (1 if index < remainder else 0)
        result.append(items[cursor:cursor + size])
        cursor += size
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--sd-adapter", type=Path)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--agent-system", type=Path, required=True)
    parser.add_argument("--arm", choices=("qwen25-3b", "qwen3-1p7b"), required=True)
    parser.add_argument("--gpu-uuid", dest="gpus", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-state-id", required=True)
    parser.add_argument("--activation-id", required=True)
    parser.add_argument("--experiment-git-sha", required=True)
    parser.add_argument("--image-digest", required=True)
    args = parser.parse_args()
    if len(args.gpus) != len(set(args.gpus)) or len(args.gpus) > 128:
        raise ValueError("final-eval GPU UUIDs must be unique and no more than panel rows")

    args.output.mkdir(parents=True, exist_ok=True)
    ordinal_groups = split(list(range(128)), len(args.gpus))
    worker = Path(__file__).resolve().with_name("q17_directapply_final_eval_worker.py")
    started = time.time()
    histories = []
    processes: list[tuple[subprocess.Popen[Any], Path]] = []
    for attempt in range(3):
        processes = []
        for index, (gpu, ordinals) in enumerate(zip(args.gpus, ordinal_groups, strict=True)):
            output = args.output / f"shard-{index:02d}"
            command = [
                sys.executable,
                str(worker),
                "--panel", str(args.panel),
                "--ordinals", *map(str, ordinals),
                "--seed-root", str(args.seed_root),
                "--model", args.model,
                "--model-revision", args.model_revision,
                "--memory", str(args.memory),
                "--skill", str(args.skill),
                "--agent-system", str(args.agent_system),
                "--arm", args.arm,
                "--gpu-uuid", gpu,
                "--output", str(output),
                "--final-state-id", args.final_state_id,
                "--activation-id", args.activation_id,
                "--experiment-git-sha", args.experiment_git_sha,
                "--image-digest", args.image_digest,
                "--resume",
            ]
            if args.sd_adapter:
                command += ["--sd-adapter", str(args.sd_adapter)]
            processes.append((subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}), output))
        workers = [{"rc": process.wait(), "output": str(output)} for process, output in processes]
        histories.append({"attempt": attempt, "workers": workers})
        invalid = 0
        interface_failure = False
        for _, output in processes:
            summary_path = output / "summary.json"
            if not summary_path.is_file():
                invalid += 999
                continue
            shard = json.loads(summary_path.read_text())
            invalid += int(shard.get("engineering_invalid_count", 999))
            if shard.get("status") == "INTERFACE_CONTRACT_FAILURE" or int(shard.get("valid_count", -1)) != int(shard.get("attempt_count", -2)):
                interface_failure = True
        if interface_failure:
            raise RuntimeError("final eval interface contract failed; do not retry or publish")
        if invalid == 0:
            break
    else:
        raise RuntimeError("final eval engineering recovery exhausted")

    rows = []
    summaries = []
    for _, output in processes:
        summaries.append(json.loads((output / "summary.json").read_text()))
        for episode in (output / "episodes").glob("*.json"):
            rows.append(json.loads(episode.read_text()))
    rows.sort(key=lambda item: int(item["panel_ordinal"]))
    if [int(item["panel_ordinal"]) for item in rows] != list(range(128)):
        raise RuntimeError("final panel exact ordinal coverage failed")

    expected_identity = {
        "activation_id": args.activation_id,
        "final_state_id": args.final_state_id,
        "experiment_git_sha": args.experiment_git_sha,
        "image_digest": args.image_digest,
    }
    for key, value in expected_identity.items():
        if {item.get(key) for item in rows} != {value} or {item.get(key) for item in summaries} != {value}:
            raise RuntimeError(f"final-eval {key} drift across panel")
    if [summary.get("gpu_uuid") for summary in summaries] != args.gpus:
        raise RuntimeError("final-eval GPU UUID drift across shards")
    expected_sampling = {
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "repetition_penalty": 1.0,
        "max_steps": 15,
        "max_new_tokens": 512,
    }
    if any(summary.get("sampling") != expected_sampling for summary in summaries):
        raise RuntimeError("final-eval sampling semantics drift")
    if args.arm == "qwen3-1p7b" and any(summary.get("chat_template_enable_thinking") is not False for summary in summaries):
        raise RuntimeError("Qwen3 final-eval thinking policy drift")
    if any(int(summary.get("valid_count", -1)) != int(summary.get("attempt_count", -2)) for summary in summaries):
        raise RuntimeError("final-eval valid coverage drift")
    if any(int(summary.get("interface_failure_count", -1)) != 0 for summary in summaries):
        raise RuntimeError("final-eval interface failure remained")
    if any(int(summary.get("parser_repair_count", -1)) != 0 for summary in summaries):
        raise RuntimeError("final-eval parser repair is forbidden under Harness201.1")
    invalid = sum(int(item["engineering_invalid"]) for item in rows)
    valid_count = sum(int(item["identity_ok"] and item["trace_complete"] and not item["parser_or_action_failure"] and not item["engineering_invalid"]) for item in rows)
    if invalid:
        raise RuntimeError("engineering-invalid row remained after recovery")
    if valid_count != 128:
        raise RuntimeError(f"final-eval valid row coverage must be 128/128, got {valid_count}")

    summary = {
        "schema": "openevo.ceiling1-final-eval.v1",  # legacy-revision-compat: historical serialized schema identity
        "status": "PASS",
        "arm": args.arm,
        "attempt_count": 128,
        "valid_count": valid_count,
        "exact_success_count": sum(int(item["exact_success"]) for item in rows),
        "exact_success_rate": sum(int(item["exact_success"]) for item in rows) / 128,
        "mean_task_score": sum(float(item["final_task_score"]) for item in rows) / 128,
        "parser_repair_count": sum(int(item["parser_repair_used"]) for item in rows),
        "engineering_invalid_count": 0,
        "interface_failure_count": 0,
        "final_state_id": args.final_state_id,
        "activation_id": args.activation_id,
        "experiment_git_sha": args.experiment_git_sha,
        "image_digest": args.image_digest,
        "gpu_uuids": args.gpus,
        "panel_file_sha256": "aea97afdd5b176126cf67f7ed315189c95fbf37b07263c84b947903cca69aa4d",
        "panel_content_sha256": "3529b94f97491434da01f9b2adc2faf1d0571545603a788d367561ae2b529d48",
        "selected_panel_digest": "cd531caff21d13d72df5c6fb013fadc10457393134d8bc3bcdc63fadf76bc453",
        "sampling": expected_sampling,
        "recovery_history": histories,
        "wall_time_seconds": time.time() - started,
    }
    atomic_json(args.output / "attempts.json", rows)
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
