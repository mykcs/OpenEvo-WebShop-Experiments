#!/usr/bin/env python3
"""Measure teacher-forced completion log-probability for one H1 arm."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from parametric_rnd import gpu_snapshot
from wandb_tracking import add_wandb_arguments, start_wandb_run


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def _episode_payload(path: Path) -> dict[str, Any]:
    source = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(source.get("episode"), dict):
        raw = dict(source["episode"])
        raw["task_id"] = source.get("task_id", raw.get("task_id"))
        raw["rollout_seed"] = source.get("rollout_seed", raw.get("rollout_seed"))
        return raw
    return source


def _score_episode(model: Any, tokenizer: Any, path: Path, device: Any) -> dict[str, Any]:
    import torch

    episode = _episode_payload(path)
    rows: list[dict[str, Any]] = []
    for index, step in enumerate(episode.get("trace") or []):
        prompt = step.get("prompt")
        completion = step.get("raw_completion")
        if not isinstance(prompt, str) or not isinstance(completion, str) or not completion.strip():
            continue
        user_messages = [{"role": "user", "content": prompt}]
        full_messages = [*user_messages, {"role": "assistant", "content": completion}]
        prompt_ids = tokenizer.apply_chat_template(user_messages, add_generation_prompt=True, return_tensors="pt").to(device)
        full_ids = tokenizer.apply_chat_template(full_messages, add_generation_prompt=False, return_tensors="pt").to(device)
        prompt_length = int(prompt_ids.shape[1])
        if int(full_ids.shape[1]) <= prompt_length:
            continue
        with torch.no_grad():
            logits = model(input_ids=full_ids).logits.float()
        target_ids = full_ids[:, prompt_length:]
        target_logits = logits[:, prompt_length - 1 : -1, :]
        token_logprob = torch.log_softmax(target_logits, dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        values = token_logprob[0].detach().cpu().tolist()
        rows.append({"step": step.get("step", index), "generation_seed": step.get("generation_seed"), "token_count": len(values), "mean_logprob": float(sum(values) / len(values)), "sum_logprob": float(sum(values))})
    token_count = sum(row["token_count"] for row in rows)
    total_logprob = sum(row["sum_logprob"] for row in rows)
    return {"task_id": episode.get("task_id"), "rollout_seed": episode.get("rollout_seed"), "source_episode": str(path), "source_episode_sha256": _sha256(path), "trace_steps_scored": len(rows), "token_count": token_count, "sum_logprob": total_logprob, "mean_logprob": total_logprob / token_count if token_count else None, "steps": rows}


def run(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    args.output.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = args.lock or args.output / f"GPU-{args.gpu_uuid}.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"GPU lock busy: {lock_path}") from exc
        before = gpu_snapshot(args.gpu_uuid)
        _atomic_json(args.output / "gpu-before.json", before)
        if before["compute_apps"]:
            raise SystemExit("compute process exists before H1 teacher-forced load")
        device = torch.device("cuda")
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.model_revision, torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval()
        if args.adapter != "none":
            model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False, local_files_only=True).eval()
        protocol = {"phase": "H1", "arm": args.arm, "model": args.model, "model_revision": args.model_revision, "episode_paths": [str(path) for path in args.episode], "adapter": args.adapter}
        protocol_fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        tracker = start_wandb_run(mode=args.wandb_mode, project=args.wandb_project, entity=args.wandb_entity, name=f"{args.arm}-h1-teacher-forced", job_type="diagnostic", group=args.wandb_group or "H1-successful-supervision-train-fit-v1", config={**protocol, "campaign_id": args.campaign_id, "experiment_git_sha": args.experiment_git_sha, "upstream_openevo_sha": args.upstream_openevo_sha, "runtime_fingerprint": args.runtime_fingerprint, "image_digest": args.image_digest, "gpu_uuid": args.gpu_uuid}, tags=("benchmark:webshop", "phase:h1", "method:sd-lora", f"arm:{args.arm}", "class:teacher-forced"), notes="Teacher-forced target log-probability diagnostic; not closed-loop evidence.", directory=args.wandb_dir or args.output / "wandb")
        try:
            rows = [_score_episode(model, tokenizer, path.resolve(), device) for path in args.episode]
            token_count = sum(int(row["token_count"]) for row in rows)
            total_logprob = sum(float(row["sum_logprob"]) for row in rows)
            summary = {"schema": "seed3090.webshop-h1-teacher-forced.v1", "arm": args.arm, "episode_count": len(rows), "token_count": token_count, "mean_logprob": total_logprob / token_count if token_count else None, "protocol_fingerprint": protocol_fingerprint, "rows": rows, "tracking": tracker.metadata()}
            _atomic_json(args.output / "summary.json", summary)
            tracker.log({"teacher_forced/episode_count": len(rows), "teacher_forced/token_count": token_count, "teacher_forced/mean_logprob": summary["mean_logprob"]})
            tracker.set_summary({"run/status": "pass", "teacher_forced/mean_logprob": summary["mean_logprob"], "teacher_forced/token_count": token_count})
            _atomic_json(args.output / "gpu-after.json", gpu_snapshot(args.gpu_uuid))
            return 0
        finally:
            del model, tokenizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            tracker.finish()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--arm", choices=("base", "failure", "success"), required=True)
    parser.add_argument("--episode", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--experiment-git-sha", required=True)
    parser.add_argument("--upstream-openevo-sha", required=True)
    parser.add_argument("--runtime-fingerprint", required=True)
    parser.add_argument("--image-digest", required=True)
    add_wandb_arguments(parser, default_project="openevo-webshop-experiments")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
