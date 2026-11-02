#!/usr/bin/env python3
"""Train one fail-closed increment of the H1.38B ordinary LoRA control."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

try:
    from scripts.openevo_webshop.ordinary_lora_contract import (
        ADAPTER_CONFIG_FILE,
        ADAPTER_WEIGHTS_FILE,
        ORDINARY_REPLAY_FILE,
        ORDINARY_STATE_FILE,
        PINNED_OPENEVO_SHA,
        MAX_TRAINING_EXAMPLES_PER_INCREMENT,
        MatchedOrdinaryLoraConfig,
        build_sample_plan,
        canonical_bytes,
        digest_json,
        example_identity,
        load_prior_state,
        sha256_file,
        validate_official_helpers,
        validate_increment_record_identities,
        validate_plan_digest,
        validate_training_evidence,
    )
except ModuleNotFoundError:  # pragma: no cover - direct container invocation
    from ordinary_lora_contract import (  # type: ignore
        ADAPTER_CONFIG_FILE,
        ADAPTER_WEIGHTS_FILE,
        ORDINARY_REPLAY_FILE,
        ORDINARY_STATE_FILE,
        PINNED_OPENEVO_SHA,
        MAX_TRAINING_EXAMPLES_PER_INCREMENT,
        MatchedOrdinaryLoraConfig,
        build_sample_plan,
        canonical_bytes,
        digest_json,
        example_identity,
        load_prior_state,
        sha256_file,
        validate_official_helpers,
        validate_increment_record_identities,
        validate_plan_digest,
        validate_training_evidence,
    )


def _write_all(fd: int, payload: bytes) -> None:
    pending = memoryview(payload)
    while pending:
        written = os.write(fd, pending)
        if written <= 0:
            raise OSError("ordinary-LoRA private write made no progress")
        pending = pending[written:]


def _write_private_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    try:
        _write_all(fd, payload.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_private_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("ordinary-LoRA replay buffer must not be empty")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    try:
        for row in rows:
            _write_all(fd, canonical_bytes(row) + b"\n")
        os.fsync(fd)
    finally:
        os.close(fd)


def _load_runtime() -> dict[str, Any]:
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from openevo.evolution.framework.contracts import canonical_digest
        from openevo.evolution.models import WorkerClaimInputArtifact
        from openevo.evolution.parametric import sd_lora, sd_lora_trainer
    except ImportError as exc:  # pragma: no cover - exercised only in GPU runtime
        raise RuntimeError(
            "ordinary-LoRA requires the pinned OpenEvo trainer runtime "
            "(torch, transformers, peft, safetensors, accelerate)"
        ) from exc
    helper_receipt = validate_official_helpers(sd_lora_trainer, sd_lora)
    return {
        "torch": torch,
        "LoraConfig": LoraConfig,
        "PeftModel": PeftModel,
        "get_peft_model": get_peft_model,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "canonical_digest": canonical_digest,
        "WorkerClaimInputArtifact": WorkerClaimInputArtifact,
        "sd_lora": sd_lora,
        "official": sd_lora_trainer,
        "helper_receipt": helper_receipt,
    }


def _dataset_examples(path: Path, artifact_id: str, runtime: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records_name = str(manifest.get("records_path") or "records.jsonl")
    if Path(records_name).is_absolute():
        raise ValueError("ordinary-LoRA dataset records_path must be relative")
    records = (path.parent / records_name).resolve()
    if not records.is_relative_to(path.parent.resolve()):
        raise ValueError("ordinary-LoRA dataset records_path escaped its manifest directory")
    if manifest.get("schema") != "openevo.dataset.v1" or manifest.get("record_count") != 2:
        raise ValueError("ordinary-LoRA increment requires exactly two dataset records")
    if not records.is_file() or records.stat().st_size != manifest.get("records_byte_size") or sha256_file(records) != manifest.get("records_sha256"):
        raise ValueError("ordinary-LoRA dataset records receipt mismatch")
    rows = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_increment_record_identities(rows)
    artifact = runtime["WorkerClaimInputArtifact"](
        artifact_id=artifact_id,
        type="dataset",
        uri=path.as_uri(),
        name=str(manifest.get("name") or artifact_id),
        manifest_sha256=runtime["canonical_digest"](manifest),
        records_byte_size=int(manifest["records_byte_size"]),
        records_sha256=str(manifest["records_sha256"]),
    )
    examples = runtime["sd_lora"]._training_examples(
        (artifact,), maximum=MAX_TRAINING_EXAMPLES_PER_INCREMENT, minimum_reward=0.0
    )
    if not examples or len(examples) > MAX_TRAINING_EXAMPLES_PER_INCREMENT:
        raise ValueError("ordinary-LoRA current examples are empty or exceed the matched ceiling")
    return manifest, examples


def _adapter_snapshot(model: Any, torch: Any) -> tuple[str, dict[str, Any], int, tuple[str, ...]]:
    digest = hashlib.sha256()
    tensors: dict[str, Any] = {}
    count = 0
    dtypes: set[str] = set()
    for name, parameter in sorted(model.named_parameters()):
        if "lora_" not in name:
            continue
        tensor = parameter.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        # NumPy does not expose bfloat16.  Viewing the contiguous tensor as raw
        # bytes preserves the declared dtype while remaining deterministic.
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        tensors[name] = tensor.float().clone()
        count += int(tensor.numel())
        dtypes.add(str(tensor.dtype))
    if not tensors:
        raise RuntimeError("ordinary-LoRA model exposes no LoRA tensors")
    return digest.hexdigest(), tensors, count, tuple(sorted(dtypes))


def _update_norm(before: Mapping[str, Any], after: Mapping[str, Any], torch: Any) -> float:
    if set(before) != set(after):
        raise RuntimeError("ordinary-LoRA tensor inventory changed within one increment")
    squared = torch.zeros((), dtype=torch.float64)
    for name in sorted(before):
        if tuple(before[name].shape) != tuple(after[name].shape):
            raise RuntimeError("ordinary-LoRA tensor shape changed within one increment")
        delta = after[name].double() - before[name].double()
        squared += (delta * delta).sum()
    value = math.sqrt(float(squared))
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError("ordinary-LoRA trainable tensors did not change")
    return value


def _load_base(
    runtime: Mapping[str, Any], config: MatchedOrdinaryLoraConfig, device: Any, model_source: str
) -> Any:
    torch = runtime["torch"]
    return runtime["AutoModelForCausalLM"].from_pretrained(
        model_source,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)


def _prepare_model(base: Any, prior: Path | None, runtime: Mapping[str, Any], config: MatchedOrdinaryLoraConfig) -> Any:
    if prior is None:
        peft_config = runtime["LoraConfig"](
            r=config.rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(config.target_modules),
        )
        model = runtime["get_peft_model"](base, peft_config)
    else:
        model = runtime["PeftModel"].from_pretrained(
            base,
            str(prior),
            is_trainable=True,
            local_files_only=True,
            autocast_adapter_dtype=False,
        )
    torch = runtime["torch"]
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            parameter.data = parameter.data.to(dtype=torch.bfloat16)
            parameter.requires_grad = True
        else:
            parameter.requires_grad = False
    if set(getattr(model, "peft_config", {})) != {"default"}:
        raise RuntimeError("ordinary-LoRA requires exactly one active default adapter")
    return model


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.upstream_openevo_sha != PINNED_OPENEVO_SHA:
        raise ValueError("ordinary-LoRA refuses an unpinned OpenEvo runtime")
    config = MatchedOrdinaryLoraConfig().validate()
    if not 0 <= args.increment_index < 8:
        raise ValueError("increment index must be in [0, 8)")
    if (args.increment_index == 0) != (args.prior_adapter is None):
        raise ValueError("increment zero must have no prior; later increments require one prior")
    runtime = _load_runtime()
    torch = runtime["torch"]
    device = torch.device("cpu") if args.qualification else torch.device("cuda", 0)
    model_root = getattr(args, "model_root", None)
    if not args.qualification and model_root is None:
        raise ValueError("formal ordinary-LoRA requires an explicit local model root")
    model_source = str(model_root.resolve()) if model_root is not None else config.base_model
    if not args.qualification:
        runtime["official"]._initialize_cuda_runtime(
            torch, component_name="H1.38B ordinary-LoRA training"
        )
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)

    dataset_path = args.dataset_manifest.resolve()
    manifest, current_examples = _dataset_examples(dataset_path, args.dataset_artifact_id, runtime)
    observed_examples = digest_json([example_identity(item) for item in current_examples])
    if args.expected_current_examples_sha256 and observed_examples != args.expected_current_examples_sha256:
        raise ValueError("ordinary-LoRA current example digest mismatch")

    prior = args.prior_adapter.resolve() if args.prior_adapter else None
    prior_state = load_prior_state(prior, config, args.increment_index) if prior else None
    prior_replay = (
        runtime["official"]._read_examples(
            prior / ORDINARY_REPLAY_FILE,
            expected_count=int(prior_state["replay_buffer_record_count"]),
        )
        if prior and prior_state
        else []
    )
    tokenizer = runtime["AutoTokenizer"].from_pretrained(
        model_source,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("ordinary-LoRA tokenizer requires an EOS or padding token")
        tokenizer.pad_token = tokenizer.eos_token
    original_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    try:
        plan, next_replay, encoded = build_sample_plan(
            current_examples=current_examples,
            prior_replay_buffer=prior_replay,
            increment_index=args.increment_index,
            tokenizer=tokenizer,
            official=runtime["official"],
            config=config,
        )
    finally:
        tokenizer.truncation_side = original_truncation_side
    validate_plan_digest(plan, args.expected_plan_sha256, qualification=args.qualification)

    started = time.monotonic()
    base = _load_base(runtime, config, device, model_source)
    target_names = runtime["official"]._target_module_names(base, config.target_modules)
    if len(target_names) != config.expected_target_module_count:
        raise RuntimeError("ordinary-LoRA target module inventory mismatch")
    expected_capacity = sum(
        config.rank
        * (
            int(base.get_submodule(name).in_features)
            + int(base.get_submodule(name).out_features)
        )
        for name in target_names
    )
    model = _prepare_model(base, prior, runtime, config)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    base_trainable = sum(
        int(parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    )
    if not trainable or base_trainable != 0 or any(parameter.dtype != torch.bfloat16 for parameter in trainable):
        raise RuntimeError("ordinary-LoRA trainable parameter/dtype contract mismatch")
    before_digest, before_tensors, trainable_count, trainable_dtypes = _adapter_snapshot(model, torch)
    if prior_state and before_digest != prior_state["trainable_tensor_sha256"]:
        raise RuntimeError("ordinary-LoRA prior reload differs from the previous saved tensor digest")
    if trainable_count != expected_capacity:
        raise RuntimeError("ordinary-LoRA serialized rank-32 capacity accounting mismatch")
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    losses: list[float] = []
    gradient_norms: list[float] = []
    steps = 0
    micro_steps = 0
    pending = False
    observed_batch_digests: list[str] = []
    for batch in runtime["official"]._batches(
        encoded,
        batch_size=config.per_device_train_batch_size,
        seed=config.seed,
        epochs=config.epochs,
    ):
        observed_batch_digests.append(digest_json(batch))
        tensors = {
            key: value.to(device, non_blocking=device.type == "cuda")
            for key, value in runtime["official"]._collate(torch, batch, int(tokenizer.pad_token_id)).items()
        }
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            loss = model(**tensors).loss
            scaled_loss = loss / config.gradient_accumulation_steps
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("ordinary-LoRA training produced non-finite loss")
        scaled_loss.backward()
        losses.append(float(loss.detach().float().cpu()))
        micro_steps += 1
        pending = True
        if micro_steps % config.gradient_accumulation_steps:
            continue
        norm = torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
        norm_value = float(norm.detach().float().cpu())
        if not math.isfinite(norm_value):
            raise RuntimeError("ordinary-LoRA training produced non-finite gradient norm")
        gradient_norms.append(norm_value)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        steps += 1
        pending = False
        if steps >= config.max_steps:
            break
    if pending and steps < config.max_steps:
        norm = torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
        norm_value = float(norm.detach().float().cpu())
        if not math.isfinite(norm_value):
            raise RuntimeError("ordinary-LoRA training produced non-finite gradient norm")
        gradient_norms.append(norm_value)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        steps += 1
    if steps != plan["planned_optimizer_steps"] or not losses or not gradient_norms:
        raise RuntimeError("ordinary-LoRA observed optimizer plan drifted")
    if observed_batch_digests != plan["batch_order_item_sha256"]:
        raise RuntimeError("ordinary-LoRA observed shuffle order drifted")
    after_digest, after_tensors, after_count, after_dtypes = _adapter_snapshot(model, torch)
    update_norm = _update_norm(before_tensors, after_tensors, torch)
    if after_count != trainable_count or after_dtypes != trainable_dtypes:
        raise RuntimeError("ordinary-LoRA trainable capacity changed within one increment")

    output = args.output_adapter.resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("ordinary-LoRA output adapter already exists")
    model.save_pretrained(output, safe_serialization=True)
    _write_private_jsonl(output / ORDINARY_REPLAY_FILE, next_replay)
    state = {
        "schema": "openevo.webshop-ordinary-sequential-lora-state.v1",
        "algorithm_family": "ordinary sequential LoRA with matched bounded replay",
        "base_model": config.base_model,
        "model_revision": config.model_revision,
        "increment_index": args.increment_index,
        "rank": config.rank,
        "effective_rank": config.rank,
        "target_modules": list(config.target_modules),
        "target_module_names": list(target_names),
        "trainable_parameter_count": trainable_count,
        "trainable_dtypes": list(trainable_dtypes),
        "dataset_manifest_sha256": runtime["canonical_digest"](manifest),
        "dataset_records_sha256": manifest["records_sha256"],
        "current_examples_sha256": observed_examples,
        "sample_plan_sha256": plan["plan_sha256"],
        "replay_buffer_record_count": len(next_replay),
        "replay_sha256": sha256_file(output / ORDINARY_REPLAY_FILE),
        "prior_adapter_model_sha256": sha256_file(prior / ADAPTER_WEIGHTS_FILE) if prior else None,
        "adapter_model_sha256": sha256_file(output / ADAPTER_WEIGHTS_FILE),
        "trainable_tensor_sha256": after_digest,
        "upstream_openevo_sha": args.upstream_openevo_sha,
        "official_helper_contract": runtime["helper_receipt"],
        "training_config": config.as_dict(),
        "capacity_accounting": {
            "ordinary_rank_all_increments": 32,
            "sd_component_rank": 4,
            "sd_final_component_count": 8,
            "sd_final_effective_rank": 32,
            "final_effective_rank_matched": True,
            "stagewise_allocated_capacity_matched": False,
        },
    }
    _write_private_json(output / ORDINARY_STATE_FILE, state)

    del optimizer, model, base
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    reload_base = _load_base(runtime, config, device, model_source)
    reloaded = runtime["PeftModel"].from_pretrained(
        reload_base,
        str(output),
        is_trainable=False,
        local_files_only=True,
        autocast_adapter_dtype=False,
    ).eval()
    reload_digest, _, reload_count, reload_dtypes = _adapter_snapshot(reloaded, torch)
    if reload_count != trainable_count or reload_dtypes != trainable_dtypes:
        raise RuntimeError("ordinary-LoRA fresh reload capacity/dtype mismatch")
    elapsed = time.monotonic() - started
    evidence = {
        "schema": "openevo.webshop-h1.38b-ordinary-lora-training-evidence.v1",
        "status": "pass",
        "arm": "ordinary-lora",
        "increment_index": args.increment_index,
        "task_id": manifest.get("task_id"),
        "goal_idx": manifest.get("goal_idx"),
        "adapter": {
            "path": str(output),
            "before_adapter_model_sha256": state["prior_adapter_model_sha256"],
            "after_adapter_model_sha256": state["adapter_model_sha256"],
            "rank": config.rank,
            "lora_alpha": config.lora_alpha,
            "target_module_names": list(target_names),
            "trainable_parameter_count": trainable_count,
            "trainable_dtypes": list(trainable_dtypes),
        },
        "sample_plan": plan,
        "metrics": {
            "optimizer_steps": steps,
            "training_loss": sum(losses) / len(losses),
            "maximum_gradient_norm": max(gradient_norms),
            "training_time_seconds": elapsed,
            "gpu_peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
            ),
            "qualification_device": device.type,
        },
        "state_proof": {
            "digest_kind": "trainable_tensor_values_v1",
            "before_trainable_tensor_sha256": before_digest,
            "after_trainable_tensor_sha256": after_digest,
            "reloaded_trainable_tensor_sha256": reload_digest,
            "update_l2_norm": update_norm,
        },
        "no_base_fallback": {
            "adapter_requested": True,
            "adapter_loaded": bool(getattr(reloaded, "peft_config", None)),
            "loaded_lora_tensor_count": reload_count,
            "loaded_lora_tensor_dtypes": list(reload_dtypes),
            "base_trainable_parameter_count": base_trainable,
            "loaded_adapter_tensor_sha256": reload_digest,
            "adapter_model_sha256": state["adapter_model_sha256"],
        },
        "provenance": {
            "upstream_openevo_sha": args.upstream_openevo_sha,
            "official_helper_contract": runtime["helper_receipt"],
            "dataset_manifest": str(dataset_path),
            "dataset_manifest_sha256": runtime["canonical_digest"](manifest),
            "dataset_records_sha256": manifest["records_sha256"],
            "parent_records_sha256": manifest.get("parent_records_sha256"),
            "parent_selection_sha256": manifest.get("parent_selection_sha256"),
            "prior_state_sha256": sha256_file(prior / ORDINARY_STATE_FILE) if prior else None,
            "output_state_sha256": sha256_file(output / ORDINARY_STATE_FILE),
        },
    }
    validate_training_evidence(evidence)
    args.evidence.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _write_private_json(args.evidence.resolve(), evidence)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--dataset-artifact-id", required=True)
    parser.add_argument("--increment-index", type=int, required=True)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--prior-adapter", type=Path)
    parser.add_argument("--output-adapter", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--upstream-openevo-sha", required=True)
    parser.add_argument("--expected-current-examples-sha256")
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--qualification", action="store_true")
    payload = train(parser.parse_args())
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
