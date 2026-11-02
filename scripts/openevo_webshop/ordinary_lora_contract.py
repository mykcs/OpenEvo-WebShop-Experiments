"""Fail-closed contracts shared by the H1.38B ordinary-LoRA trainer.

The runtime trainer imports the data, replay, encoding, batching, and optimizer-
step helpers from the pinned OpenEvo SD-LoRA implementation.  This module keeps
the method-independent validation and evidence rules importable on a bare CPU
test environment without torch/transformers/peft.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
import textwrap
from typing import Any, Callable, Mapping, Sequence


PINNED_OPENEVO_SHA = "796e6248a61112a61c880885844d3b0420c0b6ae"
PINNED_TRAINER_HELPERS_SHA256 = "6c39d199424c5d0e1ab6b065e63e3653640e4454f3759a422a0b5a24ef163295"
PINNED_DATASET_HELPER_SHA256 = "ca54c4aeb6ba618dfce44dffafe9be4498cab574d19289d57abc8471da94499a"

ORDINARY_STATE_FILE = "openevo_ordinary_lora_state.json"
ORDINARY_REPLAY_FILE = "openevo_ordinary_lora_replay.jsonl"
ADAPTER_CONFIG_FILE = "adapter_config.json"
ADAPTER_WEIGHTS_FILE = "adapter_model.safetensors"
MAX_TRAINING_EXAMPLES_PER_INCREMENT = 16

_TRAINER_HELPERS = (
    "_batches",
    "_bounded_replay_buffer",
    "_collate",
    "_encode_example",
    "_planned_optimizer_steps",
    "_select_replay_examples",
    "_target_module_names",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class MatchedOrdinaryLoraConfig:
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    model_revision: str = "0c32faf81d8f2f83cf79860d2a48111f166348d6dd38cd348576d8b3a0141c1e"
    rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    optimizer: str = "AdamW"
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.0
    epochs: int = 1
    max_steps: int = 16
    max_length: int = 512
    replay_capacity: int = 64
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    dtype: str = "bfloat16"
    seed: int = 1993
    expected_target_module_count: int = 56

    def validate(self) -> "MatchedOrdinaryLoraConfig":
        expected = MatchedOrdinaryLoraConfig()
        if self != expected:
            changed = [
                name
                for name, expected_value in asdict(expected).items()
                if asdict(self)[name] != expected_value
            ]
            raise ValueError("ordinary-LoRA matched config drifted: " + ", ".join(changed))
        if self.rank != 4 * 8 or self.lora_alpha != self.rank:
            raise ValueError("ordinary-LoRA final capacity must match SD-LoRA rank 4 x 8")
        return self

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_modules"] = list(self.target_modules)
        return payload


def _source_digest(functions: Mapping[str, Callable[..., Any]]) -> str:
    digest = hashlib.sha256()
    for name in sorted(functions):
        source = textwrap.dedent(inspect.getsource(functions[name])).strip()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def validate_official_helpers(trainer_module: Any, dataset_module: Any) -> dict[str, str]:
    try:
        trainer = {name: getattr(trainer_module, name) for name in _TRAINER_HELPERS}
        dataset = {"_training_examples": getattr(dataset_module, "_training_examples")}
    except AttributeError as exc:
        raise RuntimeError("pinned OpenEvo helper is unavailable") from exc
    trainer_digest = _source_digest(trainer)
    dataset_digest = _source_digest(dataset)
    if trainer_digest != PINNED_TRAINER_HELPERS_SHA256:
        raise RuntimeError("pinned OpenEvo trainer helper contract drifted")
    if dataset_digest != PINNED_DATASET_HELPER_SHA256:
        raise RuntimeError("pinned OpenEvo dataset helper contract drifted")
    return {"trainer_helpers_sha256": trainer_digest, "dataset_helper_sha256": dataset_digest}


def example_identity(example: Mapping[str, Any]) -> str:
    value = {
        key: example[key]
        for key in ("messages", "target_message_start", "tools")
        if key in example
    }
    return digest_json(value)


def sequence_digest(examples: Sequence[Mapping[str, Any]]) -> str:
    return digest_json([example_identity(example) for example in examples])


def validate_increment_record_identities(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 2:
        raise ValueError("ordinary-LoRA increment requires exactly two records")
    task_ids: list[str] = []
    rollout_seeds: list[int] = []
    source_digests: list[str] = []
    for row in rows:
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("ordinary-LoRA D8 record metadata is missing")
        task_id = str(row.get("task_id") or "")
        rollout_seed = metadata.get("rollout_seed")
        source_digest = str(metadata.get("source_record_sha256") or "")
        if not task_id or isinstance(rollout_seed, bool) or not isinstance(rollout_seed, int):
            raise ValueError("ordinary-LoRA D8 record identity is malformed")
        if len(source_digest) != 64:
            raise ValueError("ordinary-LoRA D8 source record digest is malformed")
        task_ids.append(task_id)
        rollout_seeds.append(rollout_seed)
        source_digests.append(source_digest)
    if len(set(task_ids)) != 1 or len(set(rollout_seeds)) != 2 or len(set(source_digests)) != 2:
        raise ValueError("ordinary-LoRA increment must contain one identity and two independent rollouts")
    return {
        "task_id": task_ids[0],
        "rollout_seeds": rollout_seeds,
        "source_record_sha256": source_digests,
    }


def build_sample_plan(
    *,
    current_examples: Sequence[dict[str, Any]],
    prior_replay_buffer: Sequence[dict[str, Any]],
    increment_index: int,
    tokenizer: Any,
    official: Any,
    config: MatchedOrdinaryLoraConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, list[int]]]]:
    """Build the optimizer/replay plan through the pinned SD-LoRA helpers."""

    config.validate()
    if not 0 <= increment_index < 8:
        raise ValueError("increment index must be in [0, 8)")
    if not current_examples:
        raise ValueError("ordinary-LoRA increment has no current examples")
    replay_examples = official._select_replay_examples(
        prior_replay_buffer,
        count=len(current_examples),
        seed=config.seed + increment_index,
    )
    optimizer_examples = [*current_examples, *replay_examples]
    replay_buffer = official._bounded_replay_buffer(
        prior_replay_buffer,
        current_examples,
        capacity=config.replay_capacity,
    )
    encoded = [
        official._encode_example(tokenizer, example, config.max_length)
        for example in optimizer_examples
    ]
    if any(not item.get("input_ids") or not item.get("labels") for item in encoded):
        raise ValueError("official encoding produced an empty sequence")
    if any(len(item["input_ids"]) != len(item["labels"]) for item in encoded):
        raise ValueError("official encoding input/label lengths differ")
    loss_tokens = [sum(label != -100 for label in item["labels"]) for item in encoded]
    if any(count < 1 for count in loss_tokens):
        raise ValueError("official encoding produced no assistant loss tokens")
    planned_steps = official._planned_optimizer_steps(
        record_count=len(encoded),
        batch_size=config.per_device_train_batch_size,
        epochs=config.epochs,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_steps=config.max_steps,
    )
    batch_order = list(
        official._batches(
            encoded,
            batch_size=config.per_device_train_batch_size,
            seed=config.seed,
            epochs=config.epochs,
        )
    )[:planned_steps]
    if len(batch_order) != planned_steps:
        raise ValueError("official batch order does not match the planned optimizer steps")
    plan: dict[str, Any] = {
        "schema": "openevo.webshop-h1.38b-matched-sample-plan.v1",
        "increment_index": increment_index,
        "replay_seed": config.seed + increment_index,
        "batch_seed": config.seed,
        "current_example_ids": [example_identity(item) for item in current_examples],
        "replay_example_ids": [example_identity(item) for item in replay_examples],
        "optimizer_example_ids": [example_identity(item) for item in optimizer_examples],
        "next_replay_buffer_ids": [example_identity(item) for item in replay_buffer],
        "input_token_counts": [len(item["input_ids"]) for item in encoded],
        "loss_token_counts": loss_tokens,
        "encoded_examples_sha256": digest_json(encoded),
        "batch_order_item_sha256": [digest_json(batch) for batch in batch_order],
        "batch_order_sha256": digest_json(batch_order),
        "consumed_example_count": sum(len(batch) for batch in batch_order),
        "consumed_input_token_total": sum(
            len(item["input_ids"]) for batch in batch_order for item in batch
        ),
        "consumed_loss_token_total": sum(
            sum(label != -100 for label in item["labels"])
            for batch in batch_order
            for item in batch
        ),
        "input_token_total": sum(len(item["input_ids"]) for item in encoded),
        "loss_token_total": sum(loss_tokens),
        "planned_optimizer_steps": planned_steps,
        "batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "epochs": config.epochs,
        "max_steps": config.max_steps,
    }
    plan["plan_sha256"] = digest_json(plan)
    return plan, list(replay_buffer), encoded


def validate_plan_digest(plan: Mapping[str, Any], expected: str | None, *, qualification: bool) -> str:
    observed = str(plan.get("plan_sha256") or "")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256", None)
    if observed != digest_json(unsigned):
        raise ValueError("sample plan self-digest mismatch")
    if expected is None:
        if not qualification:
            raise ValueError("formal ordinary-LoRA training requires a frozen sample-plan digest")
    elif observed != expected:
        raise ValueError("ordinary-LoRA sample plan differs from the frozen SD-matched plan")
    return observed


def validate_training_evidence(evidence: Mapping[str, Any]) -> None:
    if evidence.get("status") != "pass":
        raise ValueError("ordinary-LoRA evidence status is not pass")
    metrics = evidence.get("metrics")
    proof = evidence.get("state_proof")
    fallback = evidence.get("no_base_fallback")
    if not isinstance(metrics, Mapping) or not isinstance(proof, Mapping) or not isinstance(fallback, Mapping):
        raise ValueError("ordinary-LoRA evidence sections are missing")
    numeric = (
        metrics.get("training_loss"),
        metrics.get("maximum_gradient_norm"),
        proof.get("update_l2_norm"),
    )
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in numeric):
        raise ValueError("ordinary-LoRA evidence has non-finite metrics")
    if int(metrics.get("optimizer_steps") or 0) < 1 or float(proof["update_l2_norm"]) <= 0.0:
        raise ValueError("ordinary-LoRA optimizer/update proof is empty")
    before = proof.get("before_trainable_tensor_sha256")
    after = proof.get("after_trainable_tensor_sha256")
    reloaded = proof.get("reloaded_trainable_tensor_sha256")
    if not all(isinstance(value, str) and len(value) == 64 for value in (before, after, reloaded)):
        raise ValueError("ordinary-LoRA tensor digests are malformed")
    if before == after or after != reloaded:
        raise ValueError("ordinary-LoRA state change/reload equality failed")
    if not (
        fallback.get("adapter_requested") is True
        and fallback.get("adapter_loaded") is True
        and fallback.get("loaded_lora_tensor_count", 0) > 0
        and fallback.get("base_trainable_parameter_count") == 0
        and fallback.get("loaded_adapter_tensor_sha256") == reloaded
    ):
        raise ValueError("ordinary-LoRA no-BASE-fallback proof failed")


def load_prior_state(adapter: Path, config: MatchedOrdinaryLoraConfig, increment_index: int) -> dict[str, Any]:
    required = (ADAPTER_CONFIG_FILE, ADAPTER_WEIGHTS_FILE, ORDINARY_STATE_FILE, ORDINARY_REPLAY_FILE)
    missing = [name for name in required if not (adapter / name).is_file()]
    if missing:
        raise ValueError("prior ordinary-LoRA adapter is incomplete: " + ", ".join(missing))
    state = json.loads((adapter / ORDINARY_STATE_FILE).read_text(encoding="utf-8"))
    adapter_config = json.loads((adapter / ADAPTER_CONFIG_FILE).read_text(encoding="utf-8"))
    expected = {
        "schema": "openevo.webshop-ordinary-sequential-lora-state.v1",
        "base_model": config.base_model,
        "model_revision": config.model_revision,
        "increment_index": increment_index - 1,
        "rank": config.rank,
        "target_modules": list(config.target_modules),
    }
    if any(state.get(key) != value for key, value in expected.items()):
        raise ValueError("prior ordinary-LoRA state identity mismatch")
    tensor_sha = state.get("trainable_tensor_sha256")
    if not isinstance(tensor_sha, str) or len(tensor_sha) != 64:
        raise ValueError("prior ordinary-LoRA tensor digest is missing")
    if adapter_config.get("r") != config.rank or adapter_config.get("lora_alpha") != config.lora_alpha:
        raise ValueError("prior ordinary-LoRA PEFT capacity mismatch")
    if sorted(adapter_config.get("target_modules") or []) != sorted(config.target_modules):
        raise ValueError("prior ordinary-LoRA PEFT target modules mismatch")
    if (
        float(adapter_config.get("lora_dropout", -1.0)) != config.lora_dropout
        or adapter_config.get("bias") != "none"
        or adapter_config.get("task_type") != "CAUSAL_LM"
    ):
        raise ValueError("prior ordinary-LoRA PEFT method contract mismatch")
    weights_sha = sha256_file(adapter / ADAPTER_WEIGHTS_FILE)
    replay_sha = sha256_file(adapter / ORDINARY_REPLAY_FILE)
    if state.get("adapter_model_sha256") != weights_sha or state.get("replay_sha256") != replay_sha:
        raise ValueError("prior ordinary-LoRA content digest mismatch")
    return state
