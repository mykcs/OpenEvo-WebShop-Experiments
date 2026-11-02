#!/usr/bin/env python3
"""Fail closed unless both eight-increment H1.38B training chains qualify."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts.openevo_webshop.ordinary_lora_contract import (
    PINNED_DATASET_HELPER_SHA256,
    PINNED_OPENEVO_SHA,
    PINNED_TRAINER_HELPERS_SHA256,
    MatchedOrdinaryLoraConfig,
    digest_json,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _adapter_path(uri: str) -> Path:
    if not uri.startswith("file:///workspace/runs/"):
        raise ValueError("SD-LoRA adapter URI escaped the formal run root")
    return Path(uri.removeprefix("file://"))


def _inventory(path: Path) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    digest = hashlib.sha256()
    numel = 0
    nonzero = 0
    tensors = 0
    with safe_open(path, framework="pt", device="cpu") as handle:
        for name in sorted(handle.keys()):
            if "lora_" not in name:
                continue
            tensor = handle.get_tensor(name).contiguous()
            digest.update(name.encode() + b"\0")
            digest.update(str(tuple(tensor.shape)).encode() + b"\0")
            digest.update(str(tensor.dtype).encode() + b"\0")
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
            numel += int(tensor.numel())
            nonzero += int(torch.count_nonzero(tensor))
            tensors += 1
    if tensors < 1 or numel < 1 or nonzero < 1:
        raise ValueError(f"adapter contains no changed LoRA weights: {path}")
    return {"tensor_count": tensors, "parameter_count": numel, "nonzero_count": nonzero, "tensor_digest": digest.hexdigest()}


def verify(
    plans_path: Path,
    ordinary_paths: list[Path],
    sd_paths: list[Path],
    ordinary_reload_paths: list[Path],
    sd_reload_paths: list[Path],
    output: Path,
) -> dict[str, Any]:
    plans = _read(plans_path)
    unsigned = dict(plans)
    observed_bundle = str(unsigned.pop("bundle_sha256", ""))
    if observed_bundle != digest_json(unsigned):
        raise ValueError("frozen sample-plan bundle digest mismatch")
    if plans.get("official_open_evo_sha") != PINNED_OPENEVO_SHA or plans.get(
        "official_helper_contract"
    ) != {
        "dataset_helper_sha256": PINNED_DATASET_HELPER_SHA256,
        "trainer_helpers_sha256": PINNED_TRAINER_HELPERS_SHA256,
    }:
        raise ValueError("frozen official helper contract mismatch")
    if plans.get("ordinary_config") != MatchedOrdinaryLoraConfig().as_dict():
        raise ValueError("frozen ordinary-LoRA config mismatch")
    frozen = plans.get("increments")
    if not isinstance(frozen, list) or len(frozen) != 8:
        raise ValueError("qualification requires eight frozen sample plans")
    ordinary = sorted((_read(path) for path in ordinary_paths), key=lambda row: int(row.get("increment_index", -1)))
    sd = sorted(
        (_read(path) for path in sd_paths),
        key=lambda row: int((row.get("artifact_manifest") or {}).get("continual_task_index", -1)),
    )
    if len(ordinary) != 8 or len(sd) != 8:
        raise ValueError("qualification requires eight evidence rows per method")
    ordinary_reloads = sorted(
        (_read(path) for path in ordinary_reload_paths),
        key=lambda row: int(row.get("increment_index", -1)),
    )
    sd_reloads = sorted(
        (_read(path) for path in sd_reload_paths),
        key=lambda row: int(row.get("increment_index", -1)),
    )
    if len(ordinary_reloads) != 8 or len(sd_reloads) != 8:
        raise ValueError("qualification requires eight save/reload receipts per method")
    sd_adapter_paths: list[Path] = []
    training_receipts: list[dict[str, Any]] = []
    previous_ordinary_tensor: str | None = None
    previous_ordinary_file: str | None = None
    previous_sd_inventory: str | None = None
    previous_sd_file: str | None = None
    previous_sd_artifact_id: str | None = None
    for increment in range(8):
        frozen_item = frozen[increment]
        training_plan = frozen_item["training_plan"]
        if int(frozen_item.get("increment_index", -1)) != increment:
            raise ValueError("frozen sample-plan order drifted")
        o = ordinary[increment]
        if o.get("schema") != "openevo.webshop-h1.38b-ordinary-lora-training-evidence.v1" or o.get("status") != "pass":
            raise ValueError(f"ordinary-LoRA increment {increment} did not pass")
        if int(o.get("increment_index", -1)) != increment or o.get("sample_plan") != training_plan:
            raise ValueError(f"ordinary-LoRA sample plan mismatch at increment {increment}")
        if o.get("provenance", {}).get("dataset_records_sha256") != frozen_item["current_dataset_records_sha256"]:
            raise ValueError(f"ordinary-LoRA dataset mismatch at increment {increment}")
        proof = o.get("state_proof") or {}
        fallback = o.get("no_base_fallback") or {}
        metrics = o.get("metrics") or {}
        if (
            proof.get("before_trainable_tensor_sha256") == proof.get("after_trainable_tensor_sha256")
            or proof.get("after_trainable_tensor_sha256") != proof.get("reloaded_trainable_tensor_sha256")
            or not bool(fallback.get("adapter_loaded"))
            or int(fallback.get("base_trainable_parameter_count", -1)) != 0
            or int(metrics.get("optimizer_steps", -1)) != int(training_plan["planned_optimizer_steps"])
        ):
            raise ValueError(f"ordinary-LoRA state/reload proof failed at increment {increment}")
        if increment and proof.get("before_trainable_tensor_sha256") != previous_ordinary_tensor:
            raise ValueError(f"ordinary-LoRA continuation chain failed at increment {increment}")
        if o.get("adapter", {}).get("before_adapter_model_sha256") != previous_ordinary_file:
            raise ValueError(f"ordinary-LoRA adapter-file chain failed at increment {increment}")
        s = sd[increment]
        manifest = s.get("artifact_manifest")
        if not isinstance(manifest, dict):
            raise ValueError(f"SD-LoRA manifest missing at increment {increment}")
        config = manifest.get("training_config") or {}
        target_names = manifest.get("target_module_names") or []
        expected_replay = len(training_plan["replay_example_ids"])
        expected_optimizer = len(training_plan["optimizer_example_ids"])
        if (
            s.get("dataset_records_sha256") != frozen_item["current_dataset_records_sha256"]
            or (s.get("artifact") or {}).get("lineage", {}).get("dataset_records_sha256")
            != frozen_item["current_dataset_records_sha256"]
            or s.get("official_open_evo_stable") != PINNED_OPENEVO_SHA
            or int(manifest.get("continual_task_index", -1)) != increment
            or int(manifest.get("component_count", -1)) != increment + 1
            or int(manifest.get("effective_rank", -1)) != 4 * (increment + 1)
            or int(manifest.get("training_record_count", -1)) != len(training_plan["current_example_ids"])
            or int(manifest.get("replay_training_record_count", -1)) != expected_replay
            or int(manifest.get("optimizer_training_record_count", -1)) != expected_optimizer
            or int(manifest.get("steps_completed", -1)) != int(training_plan["planned_optimizer_steps"])
            or not math.isfinite(float(manifest.get("training_loss", float("nan"))))
            or config.get("rank") != 4
            or config.get("base_model") != "Qwen/Qwen2.5-7B-Instruct"
            or config.get("model_revision")
            != "0c32faf81d8f2f83cf79860d2a48111f166348d6dd38cd348576d8b3a0141c1e"
            or config.get("target_modules") != ["q_proj", "v_proj"]
            or len(target_names) != 56
            or any(not str(name).endswith((".q_proj", ".v_proj")) for name in target_names)
            or config.get("learning_rate") != 2.0e-4
            or config.get("weight_decay") != 0.0
            or config.get("epochs") != 1
            or config.get("max_steps") != 16
            or config.get("max_length") != 512
            or config.get("max_records") != 16
            or config.get("per_device_train_batch_size") != 1
            or config.get("gradient_accumulation_steps") != 1
            or config.get("max_grad_norm") != 1.0
            or config.get("replay_capacity") != 64
            or config.get("seed") != 1993
            or config.get("dtype") != "bfloat16"
            or config.get("coefficient_init") != 0.8
            or config.get("coefficient_learning_rate") != 0.01
            or config.get("minimum_reward") != 0.0
        ):
            raise ValueError(f"SD-LoRA matched contract failed at increment {increment}")
        adapter = _adapter_path(str(s.get("adapter_uri") or ""))
        state = _read(adapter / "openevo_sd_lora_state.json")
        if state.get("replay_data_sha256") != frozen_item["next_replay_buffer_file_sha256"]:
            raise ValueError(f"SD-LoRA replay bytes mismatch at increment {increment}")
        sd_inventory = _inventory(adapter / "adapter_model.safetensors")
        if previous_sd_inventory is not None and sd_inventory["tensor_digest"] == previous_sd_inventory:
            raise ValueError(f"SD-LoRA weights did not change at increment {increment}")
        if s.get("prior_parametric_memory_artifact_id") != previous_sd_artifact_id:
            raise ValueError(f"SD-LoRA adapter chain failed at increment {increment}")
        ordinary_reload = ordinary_reloads[increment]
        sd_reload = sd_reloads[increment]
        for method, receipt in (("ordinary-lora", ordinary_reload), ("sd-lora", sd_reload)):
            if (
                receipt.get("status") != "pass"
                or receipt.get("method") != method
                or int(receipt.get("increment_index", -1)) != increment
                or receipt.get("task_consumption") != "none"
                or receipt.get("first_load_digest") != receipt.get("second_load_digest")
                or int(receipt.get("loaded_lora_tensor_count", 0)) < 1
                or int(receipt.get("loaded_lora_parameter_count", 0)) < 1
                or int(receipt.get("base_trainable_parameter_count", -1)) != 0
            ):
                raise ValueError(f"{method} save/reload receipt failed at increment {increment}")
        if ordinary_reload.get("peft_weights_sha256") != o.get("adapter", {}).get("after_adapter_model_sha256"):
            raise ValueError(f"ordinary-LoRA reload file mismatch at increment {increment}")
        if sd_reload.get("peft_weights_sha256") != hashlib.sha256(
            (adapter / "adapter_model.safetensors").read_bytes()
        ).hexdigest():
            raise ValueError(f"SD-LoRA reload file mismatch at increment {increment}")
        common = {
            "schema": "openevo.webshop-h1.38b-training-receipt.v1",
            "status": "pass",
            "increment_index": increment,
            "task_id": o.get("task_id"),
            "goal_idx": o.get("goal_idx"),
            "parent_records_sha256": o.get("provenance", {}).get("parent_records_sha256"),
            "parent_selection_sha256": o.get("provenance", {}).get("parent_selection_sha256"),
            "sample_plan_sha256": training_plan["plan_sha256"],
        }
        training_receipts.append({
            **common,
            "arm": "ordinary-lora",
            "prior_adapter_file_sha256": previous_ordinary_file,
            "adapter_file_sha256": ordinary_reload["peft_weights_sha256"],
            "evaluation_tensor_digest": ordinary_reload["first_load_digest"],
            "metrics": {
                **o["metrics"],
                "training_time_seconds": float(o["metrics"].get("training_time_seconds", 0.0))
                + float(ordinary_reload.get("wall_time_seconds", 0.0)),
            },
            "state_proof": proof,
            "no_base_fallback": fallback,
        })
        training_receipts.append({
            **common,
            "arm": "sd-lora",
            "prior_adapter_file_sha256": previous_sd_file,
            "adapter_file_sha256": sd_reload["peft_weights_sha256"],
            "evaluation_tensor_digest": sd_reload["first_load_digest"],
            "metrics": {
                "optimizer_steps": manifest["steps_completed"],
                "training_loss": manifest["training_loss"],
                "training_time_seconds": float(manifest.get("training_time_seconds", 0.0))
                + float(sd_reload.get("wall_time_seconds", 0.0)),
            },
            "state_proof": {
                "serialized_tensor_sha256": sd_inventory["tensor_digest"],
                "changed_from_prior": previous_sd_inventory is None or sd_inventory["tensor_digest"] != previous_sd_inventory,
                "reloaded_trainable_tensor_sha256": sd_reload["first_load_digest"],
            },
            "no_base_fallback": {
                "adapter_requested": True,
                "adapter_loaded": True,
                "loaded_lora_tensor_count": sd_reload["loaded_lora_tensor_count"],
                "base_trainable_parameter_count": sd_reload["base_trainable_parameter_count"],
                "loaded_adapter_tensor_sha256": sd_reload["first_load_digest"],
            },
        })
        previous_ordinary_tensor = str(proof["after_trainable_tensor_sha256"])
        previous_ordinary_file = str(ordinary_reload["peft_weights_sha256"])
        previous_sd_inventory = str(sd_inventory["tensor_digest"])
        previous_sd_file = str(sd_reload["peft_weights_sha256"])
        previous_sd_artifact_id = str(
            (s.get("artifact") or {}).get("artifact_id")
            or (s.get("artifact") or {}).get("name")
            or ""
        )
        sd_adapter_paths.append(adapter)
    ordinary_final = Path(str(ordinary[-1]["adapter"]["path"]))
    sd_final = sd_adapter_paths[-1]
    ordinary_inventory = _inventory(ordinary_final / "adapter_model.safetensors")
    sd_inventory = _inventory(sd_final / "adapter_model.safetensors")
    if ordinary_inventory["parameter_count"] != sd_inventory["parameter_count"]:
        raise ValueError("final ordinary/SD serialized LoRA capacity does not match")
    ordinary_reload, sd_reload = ordinary_reloads[-1], sd_reloads[-1]
    result = {
        "schema": "openevo.webshop-h1.38b-qualification.v1",
        "status": "pass",
        "increment_count": 8,
        "records_and_token_plans_matched": True,
        "ordinary_state_change_and_reload": True,
        "sd_state_growth_and_reload": True,
        "no_base_fallback": True,
        "final_capacity": {
            "ordinary_effective_rank": 32,
            "sd_effective_rank": 32,
            "serialized_parameter_count": ordinary_inventory["parameter_count"],
            "ordinary_inventory": ordinary_inventory,
            "sd_inventory": sd_inventory,
            "stagewise_allocated_capacity_matched": False,
            "unavoidable_difference": "ordinary rank-32 is active from increment 0; SD-LoRA grows by one frozen rank-4 component per increment",
        },
        "sample_plan_bundle_sha256": observed_bundle,
        "training_receipts": training_receipts,
        "final_adapters": {
            "ordinary-lora": {
                "path": str(ordinary_final),
                "adapter_file_sha256": ordinary_reload["peft_weights_sha256"],
                "evaluation_tensor_digest": ordinary_reload["first_load_digest"],
            },
            "sd-lora": {
                "path": str(sd_final),
                "adapter_file_sha256": sd_reload["peft_weights_sha256"],
                "evaluation_tensor_digest": sd_reload["first_load_digest"],
            },
        },
        "formal_fresh_task_consumption_allowed": True,
    }
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--ordinary-evidence", type=Path, action="append", required=True)
    parser.add_argument("--sd-evidence", type=Path, action="append", required=True)
    parser.add_argument("--ordinary-reload", type=Path, action="append", required=True)
    parser.add_argument("--sd-reload", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.plans, args.ordinary_evidence, args.sd_evidence, args.ordinary_reload, args.sd_reload, args.output)
    print(json.dumps({"status": result["status"], "formal_fresh_task_consumption_allowed": result["formal_fresh_task_consumption_allowed"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
