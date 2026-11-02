#!/usr/bin/env python3
"""Reload one official SD-LoRA artifact and run paired WebShop episodes."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_task_binding(manifest_path: Path, *, episode: int, env_seed: int) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("environment_seed") != env_seed:
        raise ValueError(
            f"task manifest environment_seed={manifest.get('environment_seed')} "
            f"does not match --env-seed={env_seed}"
        )
    matches = []
    for field in ("memory_tasks", "promotion_tasks", "tasks"):
        for row in manifest.get(field, []):
            if isinstance(row, dict) and row.get("goal_idx") == episode:
                matches.append(row)
    if len(matches) != 1:
        raise ValueError(f"task manifest must contain exactly one binding for episode {episode}")
    row = matches[0]
    description = row.get("task_description")
    if not isinstance(description, str) or not description:
        raise ValueError(f"task manifest episode {episode} has no task_description")
    description_sha256 = hashlib.sha256(description.encode("utf-8")).hexdigest()
    if row.get("task_description_sha256") != description_sha256:
        raise ValueError(f"task manifest description hash mismatch for episode {episode}")
    return {
        "manifest_sha256": _sha256_file(manifest_path),
        "task_id": row.get("task_id"),
        "goal_idx": episode,
        "split": row.get("split"),
        "task_description": description,
        "task_description_sha256": description_sha256,
    }


def _validate_task_binding(binding: dict[str, object], episode: dict[str, object]) -> None:
    trace = episode.get("trace")
    first_prompt = trace[0].get("prompt") if isinstance(trace, list) and trace else None
    expected = binding["task_description"]
    if not isinstance(first_prompt, str) or not isinstance(expected, str) or expected not in first_prompt:
        raise ValueError(
            f"WebShop task identity mismatch for goal_idx={binding['goal_idx']}: "
            "episode prompt does not contain the manifest task description"
        )


def _backend(model_path: str, adapter_path: str | None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    class Backend:
        model_name = model_path

        def __init__(self):
            self._tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype=torch.float16,
            ).to("cuda")
            if adapter_path:
                self._model = PeftModel.from_pretrained(
                    self._model,
                    adapter_path,
                    is_trainable=False,
                    local_files_only=True,
                )
            self._model.eval()

        def generate(self, prompt: str, max_new: int = 128) -> str:
            encoded = self._tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            encoded = {key: value.to(self._model.device) for key, value in encoded.items()}
            n_input = encoded["input_ids"].shape[1]
            with torch.no_grad():
                output = self._model.generate(
                    **encoded,
                    max_new_tokens=max_new,
                    do_sample=False,
                    pad_token_id=self._tok.eos_token_id,
                )
            return self._tok.decode(output[0][n_input:], skip_special_tokens=True)

        def close(self):
            del self._model
            del self._tok
            gc.collect()
            torch.cuda.empty_cache()

    return Backend()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-root", required=True)
    parser.add_argument("--benchmark-src", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument(
        "--task-manifest",
        type=Path,
        help="Optional frozen WebShop task manifest used to fail closed on goal identity.",
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--env-seed",
        type=int,
        required=True,
        help="Seed used when shuffling the WebShop goal catalog.",
    )
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--max-new", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    task_binding = None
    if args.task_manifest is not None:
        task_binding = _load_task_binding(
            args.task_manifest.resolve(), episode=args.episode, env_seed=args.env_seed
        )

    import sys

    sys.path.insert(0, args.benchmark_src)
    from openevo_webshop.env_compat import WebshopEnv, WebshopEnvConfig
    from openevo_webshop.runner import run_episode

    env_config = WebshopEnvConfig(
        seed_root=args.seed_root,
        num_products=1000,
        extra_env_kwargs={"seed": args.env_seed},
    )
    env = WebshopEnv(env_config)
    rows: dict[str, object] = {
        "schema": "seed3090.parametric_rollout.v2",
        "model": args.model,
        "adapter": args.adapter,
        "episode": args.episode,
        "environment_seed": args.env_seed,
        "max_steps": args.max_steps,
    }
    base = _backend(args.model, None)
    try:
        before = run_episode(env, base, args.episode, max_steps=args.max_steps, max_new=args.max_new)
        if task_binding is not None:
            _validate_task_binding(task_binding, before.__dict__)
        rows["before"] = before.__dict__
    finally:
        base.close()
    evolved = _backend(args.model, args.adapter)
    try:
        after = run_episode(env, evolved, args.episode, max_steps=args.max_steps, max_new=args.max_new)
        rows["after"] = after.__dict__
    finally:
        evolved.close()
        env.close()
    rows["adapter_reload"] = True
    if task_binding is not None:
        rows["task_binding"] = task_binding
    rows["after_rollout_completed"] = not bool(rows["after"].get("error"))
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.chmod(0o600)
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0 if rows["after_rollout_completed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
