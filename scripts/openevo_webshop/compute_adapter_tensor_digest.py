#!/usr/bin/env python3
"""Compute the frozen inference-side digest for a PEFT adapter.

The closed-loop evaluator hashes active ``PeftModel.named_parameters()``
entries.  The serialized safetensors keys omit the PEFT adapter-name segment,
so this small CPU-only helper applies that deterministic naming projection
without loading the base model or consuming a WebShop task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _loaded_parameter_name(key: str) -> str:
    for suffix in (".lora_A.weight", ".lora_B.weight"):
        if key.endswith(suffix):
            return key[: -len(".weight")] + ".default.weight"
    return key


def compute(adapter: Path) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    weights = adapter / "adapter_model.safetensors"
    if not weights.is_file():
        raise ValueError(f"adapter weights are missing: {weights}")
    state = load_file(str(weights), device="cpu")
    digest = hashlib.sha256()
    lora_keys = []
    for key, tensor in sorted(state.items()):
        if "lora_" not in key:
            continue
        name = _loaded_parameter_name(key)
        # vendor/openevo-webshop loads the base model and active LoRA tensors
        # as bfloat16 before hashing them in TransformersBackend._tensor_digest.
        tensor = tensor.detach().to(dtype=torch.bfloat16).cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        lora_keys.append(name)
    if not lora_keys:
        raise ValueError("adapter weights expose no LoRA tensors")
    return {
        "schema": "openevo.webshop-adapter-tensor-digest.v1",
        "adapter": str(adapter.resolve()),
        "source": "adapter_model.safetensors",
        "naming_projection": "safetensors.lora_A/B.weight-to-PeftModel.default.weight",
        "inference_dtype": "torch.bfloat16",
        "tensor_count": len(lora_keys),
        "loaded_tensor_digest": digest.hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compute(args.adapter.resolve()), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
