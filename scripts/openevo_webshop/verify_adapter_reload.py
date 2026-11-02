#!/usr/bin/env python3
"""Load an OpenEvo adapter twice and emit a no-task reload receipt."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def loaded_adapter_snapshot(model: Any, torch: Any) -> dict[str, Any]:
    digest = hashlib.sha256()
    lora_tensor_count = 0
    lora_parameter_count = 0
    base_trainable_parameter_count = 0
    for name, parameter in sorted(model.named_parameters()):
        if "lora_" not in name:
            if bool(parameter.requires_grad):
                base_trainable_parameter_count += int(parameter.numel())
            continue
        tensor = parameter.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        lora_tensor_count += 1
        lora_parameter_count += int(tensor.numel())
    if lora_tensor_count < 1 or lora_parameter_count < 1:
        raise RuntimeError("requested adapter reload exposed no active LoRA tensors")
    if base_trainable_parameter_count != 0:
        raise RuntimeError("requested inference adapter left base parameters trainable")
    return {
        "tensor_digest": digest.hexdigest(),
        "loaded_lora_tensor_count": lora_tensor_count,
        "loaded_lora_parameter_count": lora_parameter_count,
        "base_trainable_parameter_count": base_trainable_parameter_count,
    }


def load_once(model_path: Path, adapter_path: Path, revision: str, torch: Any) -> dict[str, Any]:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        revision=revision,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    try:
        adapted = PeftModel.from_pretrained(
            base,
            str(adapter_path),
            is_trainable=False,
            local_files_only=True,
            autocast_adapter_dtype=False,
        ).eval()
        return loaded_adapter_snapshot(adapted, torch)
    finally:
        del base
        if "adapted" in locals():
            del adapted
        gc.collect()
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--increment-index", type=int)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    model_path = args.model.resolve()
    adapter_path = args.adapter.resolve()
    sd_state_path = adapter_path / "openevo_sd_lora_state.json"
    ordinary_state_path = adapter_path / "openevo_ordinary_lora_state.json"
    state_path = sd_state_path if sd_state_path.is_file() else ordinary_state_path
    peft_path = adapter_path / "adapter_model.safetensors"
    config_path = adapter_path / "adapter_config.json"
    for path in (model_path / "config.json", state_path, peft_path, config_path):
        if not path.is_file():
            raise SystemExit(f"required reload asset is missing: {path}")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("model_revision") != args.model_revision:
        raise SystemExit("adapter model revision does not match the requested revision")

    import torch

    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA is required but unavailable")
    if not torch.cuda.is_available():
        raise SystemExit("adapter reload requires CUDA for an independent model load")

    started = time.monotonic()
    before = load_once(model_path, adapter_path, args.model_revision, torch)
    after = load_once(model_path, adapter_path, args.model_revision, torch)
    state_weights = adapter_path / "openevo_sd_lora_state.safetensors"
    state_matches = before == after
    result = {
        "schema": "openevo.webshop-adapter-reload.v1",
        "status": "pass" if state_matches else "fail",
        "task_consumption": "none",
        "model_revision": args.model_revision,
        "method": "sd-lora" if state_path == sd_state_path else "ordinary-lora",
        "adapter_id": state.get("adapter_id"),
        "increment_index": args.increment_index
        if args.increment_index is not None
        else state.get("increment_index", state.get("continual_task_index", state.get("task_index"))),
        "component_count": state.get("component_count"),
        "state_weights_sha256": sha256_file(state_weights) if state_weights.is_file() else None,
        "peft_weights_sha256": sha256_file(peft_path),
        "first_load_digest": before["tensor_digest"],
        "second_load_digest": after["tensor_digest"],
        "loaded_parameter_digest_match": state_matches,
        "loaded_lora_tensor_count": before["loaded_lora_tensor_count"],
        "loaded_lora_parameter_count": before["loaded_lora_parameter_count"],
        "base_trainable_parameter_count": before["base_trainable_parameter_count"],
        "wall_time_seconds": time.monotonic() - started,
        "gpu": {
            "count": torch.cuda.device_count(),
            "device": torch.cuda.get_device_name(0),
        },
    }
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.chmod(0o600)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
