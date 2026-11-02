#!/usr/bin/env python3
"""No-task H1.39 structural/load qualification on one UUID-bound GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapter", required=True, help="none or an immutable adapter directory")
    parser.add_argument("--arm", choices=("base", "ordinary-lora", "sd-lora"), required=True)
    parser.add_argument("--expected-adapter-file-sha256")
    parser.add_argument("--expected-adapter-tensor-digest")
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    visible = [token.strip() for token in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if token.strip()]
    if visible != [args.gpu_uuid]:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES is not exactly the selected UUID: {visible!r}")
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for H1.39 structural load qualification")

    from openevo_webshop.inference import make_backend

    adapter_path = None if args.adapter == "none" else args.adapter
    backend = make_backend("transformers", args.model, device="cuda", revision=args.model_revision, adapter_path=adapter_path, local_files_only=True)
    receipt = backend.load_receipt
    expected_loaded = args.arm != "base"
    if receipt.get("verified") is not True or bool(receipt.get("adapter_loaded")) != expected_loaded:
        raise SystemExit("adapter load receipt does not match structural arm contract")
    if expected_loaded:
        if receipt.get("adapter_file_sha256") != args.expected_adapter_file_sha256 or receipt.get("loaded_tensor_digest") != args.expected_adapter_tensor_digest:
            raise SystemExit("structural adapter identity does not match frozen H1.38B identity")
    else:
        if receipt.get("adapter_file_sha256") is not None or receipt.get("loaded_tensor_digest") is not None:
            raise SystemExit("BASE structural gate exposed adapter identity")

    prompt = "Return exactly one short WebShop action command and no explanation."
    first = backend.generate(prompt, max_new=16, temperature=0.0, generation_seed=2026082039)
    second = backend.generate(prompt, max_new=16, temperature=0.0, generation_seed=2026082039)
    if not isinstance(first, str) or not first or first != second:
        raise SystemExit("deterministic bounded generation/replay identity gate failed")
    result = {
        "schema": "openevo.webshop-h1.39-mr-load-qualification.v1",
        "status": "pass",
        "task_consumption": "none",
        "arm": args.arm,
        "gpu_uuid": args.gpu_uuid,
        "model_revision": args.model_revision,
        "adapter_load_receipt": receipt,
        "bounded_generation": {"max_new_tokens": 16, "temperature": 0.0, "replay_match": True, "output_sha256": hashlib.sha256(first.encode()).hexdigest()},
        "numeric_json_serialization": _digest({"score": 0.0, "valid": True, "gpu_hours": 0.0}),
        "formal_panel_consumed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "arm": args.arm, "task_consumption": "none", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
