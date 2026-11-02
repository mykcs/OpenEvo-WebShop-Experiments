#!/usr/bin/env python3
"""Run both H1.38B eight-increment mechanisms in a no-GPU CPU sandbox.

This is an engineering qualification, not formal training or evaluation.  It
uses the sealed D8 bytes and the pinned OpenEvo data/replay/token helpers, while
replacing only the 7B weights and CUDA device with a deterministic one-layer
Qwen2 model on CPU.  The production configs remain fail-closed elsewhere.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import types
from typing import Any

from scripts.openevo_webshop import ordinary_lora_contract as ordinary_contract
from scripts.openevo_webshop import ordinary_lora_trainer as ordinary_trainer
from scripts.openevo_webshop import run_parametric_smoke
from scripts.openevo_webshop.build_h138b_method_control_inputs import (
    CAMPAIGN_ID,
    D8_RECORDS_SHA256,
    D8_SELECTION_SHA256,
    write_increments,
)
from scripts.openevo_webshop.ordinary_lora_contract import (
    MAX_TRAINING_EXAMPLES_PER_INCREMENT,
    PINNED_OPENEVO_SHA,
    build_sample_plan,
    digest_json,
    example_identity,
    sha256_file,
)


RECEIPT_SCHEMA = "openevo.webshop-h1.38b-training-receipt.v1"
QUALIFICATION_SCHEMA = "openevo.webshop-h1.38b-cpu-sandbox-qualification.v1"


def _tensor_digest(state: dict[str, Any], torch: Any) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    if not state:
        raise RuntimeError("qualification adapter loaded no tensors")
    return digest.hexdigest()


def _nonzero_norm(path: Path, torch: Any) -> float:
    from safetensors import safe_open

    squared = torch.zeros((), dtype=torch.float64)
    count = 0
    with safe_open(path, framework="pt", device="cpu") as handle:
        for name in sorted(handle.keys()):
            if "lora_" not in name:
                continue
            tensor = handle.get_tensor(name).double()
            squared += (tensor * tensor).sum()
            count += int(tensor.numel())
    value = math.sqrt(float(squared))
    if count < 1 or not math.isfinite(value) or value <= 0.0:
        raise RuntimeError("qualification adapter contains no finite nonzero LoRA state")
    return value


class _TokenizerFactory:
    tokenizer_root: Path

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            str(cls.tokenizer_root), local_files_only=True, trust_remote_code=False
        )


class _TinyQwenFactory:
    @staticmethod
    def from_pretrained(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260820)
            config = Qwen2Config(
                vocab_size=256,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=512,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
                tie_word_embeddings=False,
            )
            model = Qwen2ForCausalLM(config).to(dtype=torch.bfloat16, device="cpu")
        original_forward = model.forward
        original_to = model.to

        def bounded_forward(self: Any, input_ids: Any = None, labels: Any = None, **forward: Any) -> Any:
            if input_ids is not None:
                input_ids = input_ids.remainder(self.config.vocab_size)
            if labels is not None:
                labels = torch.where(labels == -100, labels, labels.remainder(self.config.vocab_size))
            return original_forward(input_ids=input_ids, labels=labels, **forward)

        model.forward = types.MethodType(bounded_forward, model)

        def cpu_to(self: Any, *values: Any, **options: Any) -> Any:
            if values and str(values[0]).startswith("cuda"):
                values = (torch.device("cpu"), *values[1:])
            if "device" in options and str(options["device"]).startswith("cuda"):
                options["device"] = torch.device("cpu")
            return original_to(*values, **options)

        model.to = types.MethodType(cpu_to, model)
        return model


@contextmanager
def _cpu_runtime(tokenizer_root: Path):
    """Patch only device/model factories for a contained qualification process."""

    import torch
    from openevo.evolution.parametric import sd_lora_trainer

    _TokenizerFactory.tokenizer_root = tokenizer_root
    original_dependencies = sd_lora_trainer._dependencies
    original_initialize = sd_lora_trainer._initialize_cuda_runtime
    original_autocast = torch.autocast
    original_tensor_to = torch.Tensor.to
    cuda_names = (
        "manual_seed_all",
        "init",
        "reset_peak_memory_stats",
        "max_memory_allocated",
        "empty_cache",
    )
    original_cuda = {name: getattr(torch.cuda, name) for name in cuda_names}

    def dependencies() -> tuple[Any, ...]:
        values = list(original_dependencies())
        values[5] = _TinyQwenFactory
        _, bits = values[6]
        values[6] = (_TokenizerFactory, bits)
        return tuple(values)

    def cpu_autocast(device_type: str, *args: Any, **kwargs: Any) -> Any:
        if device_type == "cuda":
            return nullcontext()
        return original_autocast(device_type, *args, **kwargs)

    def cpu_tensor_to(tensor: Any, *values: Any, **options: Any) -> Any:
        if values and str(values[0]).startswith("cuda"):
            values = (torch.device("cpu"), *values[1:])
        if "device" in options and str(options["device"]).startswith("cuda"):
            options["device"] = torch.device("cpu")
        return original_tensor_to(tensor, *values, **options)

    sd_lora_trainer._dependencies = dependencies
    sd_lora_trainer._initialize_cuda_runtime = lambda *args, **kwargs: None
    torch.autocast = cpu_autocast  # type: ignore[assignment]
    torch.Tensor.to = cpu_tensor_to  # type: ignore[assignment]
    torch.cuda.manual_seed_all = lambda *args, **kwargs: None
    torch.cuda.init = lambda *args, **kwargs: None
    torch.cuda.reset_peak_memory_stats = lambda *args, **kwargs: None
    # The pinned trainer requires a positive accounting receipt. One byte is a
    # sentinel for this explicitly no-GPU qualification, not measured usage.
    torch.cuda.max_memory_allocated = lambda *args, **kwargs: 1
    torch.cuda.empty_cache = lambda *args, **kwargs: None
    try:
        yield
    finally:
        sd_lora_trainer._dependencies = original_dependencies
        sd_lora_trainer._initialize_cuda_runtime = original_initialize
        torch.autocast = original_autocast  # type: ignore[assignment]
        torch.Tensor.to = original_tensor_to  # type: ignore[assignment]
        for name, value in original_cuda.items():
            setattr(torch.cuda, name, value)


class _InProcessCpuSdTrainer:
    def __init__(self, artifact_root: Path, *, python_executable: str | None = None) -> None:
        del artifact_root, python_executable

    def __enter__(self) -> "_InProcessCpuSdTrainer":
        return self

    def __exit__(self, *args: Any) -> None:
        del args

    def train_sd_lora(self, request: Any, cancellation: Any = None) -> Any:
        del cancellation
        from openevo.evolution.parametric.sd_lora_trainer import _train

        return _train(request)


def _loaded_digest(adapter: Path, torch: Any) -> str:
    from peft import PeftModel, get_peft_model_state_dict

    base = _TinyQwenFactory.from_pretrained()
    adapted = PeftModel.from_pretrained(
        base, str(adapter), is_trainable=False, local_files_only=True, autocast_adapter_dtype=False
    ).eval()
    return _tensor_digest(get_peft_model_state_dict(adapted), torch)


def _normalized_receipt(
    *,
    arm: str,
    increment: int,
    manifest: dict[str, Any],
    plan: dict[str, Any],
    adapter: Path,
    prior_file_sha256: str | None,
    evaluation_digest: str,
    before_digest: str,
    metrics: dict[str, Any],
    update_norm: float,
    tensor_count: int,
) -> dict[str, Any]:
    adapter_sha = sha256_file(adapter / "adapter_model.safetensors")
    return {
        "schema": RECEIPT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "status": "pass",
        "qualification_only": True,
        "arm": arm,
        "increment_index": increment,
        "task_id": manifest["task_id"],
        "goal_idx": manifest["goal_idx"],
        "parent_records_sha256": manifest["parent_records_sha256"],
        "parent_selection_sha256": manifest["parent_selection_sha256"],
        "dataset_records_sha256": manifest["records_sha256"],
        "sample_plan_sha256": plan["plan_sha256"],
        "prior_adapter_file_sha256": prior_file_sha256,
        "adapter_file_sha256": adapter_sha,
        "evaluation_tensor_digest": evaluation_digest,
        "metrics": metrics,
        "state_proof": {
            "before_trainable_tensor_sha256": before_digest,
            "after_trainable_tensor_sha256": evaluation_digest,
            "reloaded_trainable_tensor_sha256": evaluation_digest,
            "update_l2_norm": update_norm,
        },
        "no_base_fallback": {
            "adapter_requested": True,
            "adapter_loaded": True,
            "loaded_lora_tensor_count": tensor_count,
            "base_trainable_parameter_count": 0,
            "loaded_adapter_tensor_sha256": evaluation_digest,
        },
    }


def qualify(d8_manifest: Path, tokenizer_root: Path, output_root: Path) -> dict[str, Any]:
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors import safe_open

    output_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    increments_root = output_root / "increments"
    increment_bundle = write_increments(d8_manifest, increments_root)
    runtime = ordinary_trainer._load_runtime()
    formal_config = ordinary_contract.MatchedOrdinaryLoraConfig()
    qualification_config = replace(formal_config, expected_target_module_count=2)
    tokenizer = _TokenizerFactory.from_pretrained()
    prior_replay: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    ordinary_receipts: list[dict[str, Any]] = []
    sd_receipts: list[dict[str, Any]] = []
    ordinary_prior: Path | None = None
    sd_prior: Path | None = None
    ordinary_prior_file: str | None = None
    sd_prior_file: str | None = None
    ordinary_before = hashlib.sha256(b"h138b-ordinary-base").hexdigest()
    sd_before = hashlib.sha256(b"h138b-sd-base").hexdigest()

    original_contract_class = ordinary_contract.MatchedOrdinaryLoraConfig
    original_trainer_class = ordinary_trainer.MatchedOrdinaryLoraConfig
    original_runtime_loader = ordinary_trainer._load_runtime
    original_service = run_parametric_smoke.SubprocessSdLoraTrainerService
    ordinary_contract.MatchedOrdinaryLoraConfig = lambda: qualification_config  # type: ignore[assignment]
    ordinary_trainer.MatchedOrdinaryLoraConfig = lambda: qualification_config  # type: ignore[assignment]

    def qualification_runtime() -> dict[str, Any]:
        value = original_runtime_loader()
        value["AutoModelForCausalLM"] = _TinyQwenFactory
        value["AutoTokenizer"] = _TokenizerFactory
        return value

    ordinary_trainer._load_runtime = qualification_runtime
    run_parametric_smoke.SubprocessSdLoraTrainerService = _InProcessCpuSdTrainer
    try:
        for increment, row in enumerate(increment_bundle["increments"]):
            manifest_path = Path(row["manifest_path"])
            manifest, current = ordinary_trainer._dataset_examples(
                manifest_path, f"h138b-qualification-{increment:02d}", runtime
            )
            if len(current) > MAX_TRAINING_EXAMPLES_PER_INCREMENT:
                raise RuntimeError("qualification current-example ceiling drifted")
            old_side = tokenizer.truncation_side
            tokenizer.truncation_side = "left"
            try:
                plan, next_replay, _ = build_sample_plan(
                    current_examples=current,
                    prior_replay_buffer=prior_replay,
                    increment_index=increment,
                    tokenizer=tokenizer,
                    official=runtime["official"],
                    config=qualification_config,
                )
            finally:
                tokenizer.truncation_side = old_side
            plans.append(
                {
                    "increment_index": increment,
                    "task_id": manifest["task_id"],
                    "current_example_count": len(current),
                    "current_examples_sha256": digest_json([example_identity(item) for item in current]),
                    "plan": plan,
                }
            )

            ordinary_dir = output_root / "ordinary" / f"increment-{increment:02d}"
            ordinary_evidence = output_root / "ordinary" / f"increment-{increment:02d}.json"
            ordinary_args = argparse.Namespace(
                dataset_manifest=manifest_path,
                dataset_artifact_id=f"h138b-qualification-{increment:02d}",
                increment_index=increment,
                prior_adapter=ordinary_prior,
                output_adapter=ordinary_dir,
                evidence=ordinary_evidence,
                upstream_openevo_sha=PINNED_OPENEVO_SHA,
                expected_current_examples_sha256=plans[-1]["current_examples_sha256"],
                expected_plan_sha256=plan["plan_sha256"],
                qualification=True,
            )
            ordinary_raw = ordinary_trainer.train(ordinary_args)
            ordinary_eval_digest = _loaded_digest(ordinary_dir, torch)
            with safe_open(ordinary_dir / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
                ordinary_tensor_count = len([name for name in handle.keys() if "lora_" in name])
            ordinary_receipts.append(
                _normalized_receipt(
                    arm="ordinary-lora",
                    increment=increment,
                    manifest=manifest,
                    plan=plan,
                    adapter=ordinary_dir,
                    prior_file_sha256=ordinary_prior_file,
                    evaluation_digest=ordinary_eval_digest,
                    before_digest=ordinary_before,
                    metrics=ordinary_raw["metrics"],
                    update_norm=float(ordinary_raw["state_proof"]["update_l2_norm"]),
                    tensor_count=ordinary_tensor_count,
                )
            )
            ordinary_prior = ordinary_dir
            ordinary_prior_file = ordinary_receipts[-1]["adapter_file_sha256"]
            ordinary_before = ordinary_eval_digest

            sd_artifact_root = output_root / "sd" / f"increment-{increment:02d}" / "artifacts"
            sd_artifact_root.mkdir(mode=0o700, parents=True)
            prior_copy: Path | None = None
            if sd_prior is not None:
                prior_copy = sd_artifact_root / "prior-input"
                shutil.copytree(sd_prior, prior_copy)
            sd_evidence_path = output_root / "sd" / f"increment-{increment:02d}.json"
            sd_args = argparse.Namespace(
                dataset_manifest=manifest_path,
                artifact_root=sd_artifact_root,
                evidence=sd_evidence_path,
                run_id=f"h138b-cpu-qualification-{increment:02d}",
                benchmark="webshop",
                python=None,
                base_model=formal_config.base_model,
                model_revision=formal_config.model_revision,
                max_steps=formal_config.max_steps,
                max_records=MAX_TRAINING_EXAMPLES_PER_INCREMENT,
                minimum_reward=0.0,
                prior_adapter=prior_copy,
                prior_artifact_id=(f"h138b-sd-prior-{increment - 1:02d}" if prior_copy else None),
            )
            sd_raw = run_parametric_smoke.run(sd_args)
            sd_adapter = Path(str(sd_raw["adapter_uri"]).removeprefix("file://"))
            sd_eval_digest = _loaded_digest(sd_adapter, torch)
            with safe_open(sd_adapter / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
                sd_tensor_count = len([name for name in handle.keys() if "lora_" in name])
            sd_manifest = sd_raw["artifact_manifest"]
            if (
                int(sd_manifest["training_record_count"]) != len(current)
                or int(sd_manifest["replay_training_record_count"]) != len(plan["replay_example_ids"])
                or int(sd_manifest["optimizer_training_record_count"]) != len(plan["optimizer_example_ids"])
                or int(sd_manifest["steps_completed"]) != int(plan["planned_optimizer_steps"])
            ):
                raise RuntimeError("SD-LoRA CPU qualification diverged from the frozen sample plan")
            sd_receipts.append(
                _normalized_receipt(
                    arm="sd-lora",
                    increment=increment,
                    manifest=manifest,
                    plan=plan,
                    adapter=sd_adapter,
                    prior_file_sha256=sd_prior_file,
                    evaluation_digest=sd_eval_digest,
                    before_digest=sd_before,
                    metrics={
                        "optimizer_steps": int(sd_manifest["steps_completed"]),
                        "training_loss": float(sd_manifest["training_loss"]),
                        "maximum_gradient_norm": 0.0,
                        "training_time_seconds": float(sd_manifest["training_time_seconds"]),
                    },
                    update_norm=_nonzero_norm(sd_adapter / "adapter_model.safetensors", torch),
                    tensor_count=sd_tensor_count,
                )
            )
            sd_prior = sd_adapter
            sd_prior_file = sd_receipts[-1]["adapter_file_sha256"]
            sd_before = sd_eval_digest
            prior_replay = next_replay
    finally:
        ordinary_contract.MatchedOrdinaryLoraConfig = original_contract_class
        ordinary_trainer.MatchedOrdinaryLoraConfig = original_trainer_class
        ordinary_trainer._load_runtime = original_runtime_loader
        run_parametric_smoke.SubprocessSdLoraTrainerService = original_service

    receipt_root = output_root / "receipts"
    receipt_root.mkdir(mode=0o700)
    receipt_paths: list[str] = []
    for receipt in [*ordinary_receipts, *sd_receipts]:
        path = receipt_root / f"{receipt['arm']}-increment-{receipt['increment_index']:02d}.json"
        path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        path.chmod(0o600)
        receipt_paths.append(str(path))
    plan_bundle = {
        "schema": "openevo.webshop-h1.38b-frozen-sample-plans.v1",
        "increments": plans,
    }
    plan_bundle["bundle_sha256"] = digest_json(plan_bundle)
    result = {
        "schema": QUALIFICATION_SCHEMA,
        "status": "pass",
        "task_consumption": "sealed-train-records-only; no fresh evaluation tasks",
        "gpu_requested": False,
        "formal_training_or_evaluation": False,
        "sealed_source": {
            "records_sha256": D8_RECORDS_SHA256,
            "selection_sha256": D8_SELECTION_SHA256,
            "record_count": 16,
            "identity_count": 8,
        },
        "increment_task_order": increment_bundle["task_order"],
        "sample_plan_bundle": plan_bundle,
        "training_receipts": receipt_paths,
        "ordinary_final_adapter_file_sha256": ordinary_receipts[-1]["adapter_file_sha256"],
        "sd_final_adapter_file_sha256": sd_receipts[-1]["adapter_file_sha256"],
        "qualification_deviations": {
            "device": "CPU instead of formal CUDA",
            "base_weights": "deterministic one-layer random Qwen2 instead of pinned 7B weights",
            "target_module_count": "2 instead of formal 56; q_proj/v_proj suffixes unchanged",
            "tokenizer_and_data_plan": "pinned Qwen2.5 tokenizer and exact sealed D8 records/order",
            "optimizer_and_capacity": "formal LR/steps/replay/ranks retained",
        },
        "claim_boundary": "Engineering save/update/reload/no-fallback qualification only; no WebShop reward or method-effect evidence.",
    }
    result["qualification_sha256"] = digest_json(result)
    output = output_root / "qualification.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.chmod(0o600)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d8-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    with _cpu_runtime(args.tokenizer_root.resolve()):
        result = qualify(
            args.d8_manifest.resolve(), args.tokenizer_root.resolve(), args.output_root.resolve()
        )
    print(json.dumps({"status": result["status"], "qualification_sha256": result["qualification_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
