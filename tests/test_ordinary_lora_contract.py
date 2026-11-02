from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from scripts.openevo_webshop.ordinary_lora_contract import (
    ADAPTER_CONFIG_FILE,
    ADAPTER_WEIGHTS_FILE,
    ORDINARY_REPLAY_FILE,
    ORDINARY_STATE_FILE,
    PINNED_OPENEVO_SHA,
    MatchedOrdinaryLoraConfig,
    build_sample_plan,
    digest_json,
    load_prior_state,
    sha256_file,
    validate_plan_digest,
    validate_increment_record_identities,
    validate_training_evidence,
)


def example(name: str, task: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": f"prompt-{name}"},
            {"role": "assistant", "content": f"answer-{name}"},
        ],
        "target_message_start": 1,
        "metadata": {"task_id": task},
    }


class FakeOfficial:
    @staticmethod
    def _select_replay_examples(replay, *, count, seed):
        assert seed in range(1993, 2001)
        return list(replay[:count])

    @staticmethod
    def _bounded_replay_buffer(prior, current, *, capacity):
        assert capacity == 64
        return [*prior, *current][:capacity]

    @staticmethod
    def _encode_example(tokenizer, row, maximum):
        assert maximum == 512
        value = len(row["messages"][0]["content"])
        return {"input_ids": [value, value + 1], "labels": [-100, value + 1]}

    @staticmethod
    def _planned_optimizer_steps(*, record_count, batch_size, epochs, gradient_accumulation_steps, max_steps):
        assert (batch_size, epochs, gradient_accumulation_steps, max_steps) == (1, 1, 1, 16)
        return min(record_count, max_steps)

    @staticmethod
    def _batches(rows, *, batch_size, seed, epochs):
        assert batch_size == epochs == 1 and seed == 1993
        return [[row] for row in reversed(rows)]


def passing_evidence() -> dict:
    return {
        "status": "pass",
        "metrics": {"optimizer_steps": 4, "training_loss": 1.0, "maximum_gradient_norm": 0.5},
        "state_proof": {
            "before_trainable_tensor_sha256": "1" * 64,
            "after_trainable_tensor_sha256": "2" * 64,
            "reloaded_trainable_tensor_sha256": "2" * 64,
            "update_l2_norm": 0.25,
        },
        "no_base_fallback": {
            "adapter_requested": True,
            "adapter_loaded": True,
            "loaded_lora_tensor_count": 56 * 2,
            "base_trainable_parameter_count": 0,
            "loaded_adapter_tensor_sha256": "2" * 64,
        },
    }


def test_matched_config_is_final_rank32_and_fully_frozen():
    config = MatchedOrdinaryLoraConfig().validate()
    assert config.rank == config.lora_alpha == 32 == 4 * 8
    assert config.optimizer == "AdamW"
    assert config.learning_rate == 2.0e-4
    assert config.target_modules == ("q_proj", "v_proj")
    assert config.dtype == "bfloat16"
    assert config.per_device_train_batch_size == 1
    assert config.seed == 1993
    assert PINNED_OPENEVO_SHA == "796e6248a61112a61c880885844d3b0420c0b6ae"


def test_matched_config_rejects_any_control_drift():
    with pytest.raises(ValueError, match="learning_rate"):
        MatchedOrdinaryLoraConfig(learning_rate=1.0e-4).validate()


def test_increment_records_use_real_d8_rollout_identity_contract():
    rows = [
        {
            "task_id": "webshop_000500",
            "metadata": {"rollout_seed": seed, "source_record_sha256": digest * 64},
        }
        for seed, digest in ((11, "a"), (12, "b"))
    ]
    receipt = validate_increment_record_identities(rows)
    assert receipt["task_id"] == "webshop_000500"
    assert receipt["rollout_seeds"] == [11, 12]
    rows[1]["metadata"]["rollout_seed"] = 11
    with pytest.raises(ValueError, match="independent rollouts"):
        validate_increment_record_identities(rows)


def test_increment_plan_uses_current_plus_equal_replay_and_exact_tokens():
    current = [example("c0", "task-new"), example("c1", "task-new")]
    prior = [example("p0", "task-old"), example("p1", "task-old")]
    plan, replay, encoded = build_sample_plan(
        current_examples=current,
        prior_replay_buffer=prior,
        increment_index=1,
        tokenizer=object(),
        official=FakeOfficial,
        config=MatchedOrdinaryLoraConfig(),
    )
    assert len(plan["current_example_ids"]) == 2
    assert len(plan["replay_example_ids"]) == 2
    assert len(plan["optimizer_example_ids"]) == 4
    assert plan["input_token_total"] == 8
    assert plan["loss_token_total"] == 4
    assert plan["planned_optimizer_steps"] == 4
    assert len(plan["batch_order_item_sha256"]) == 4
    assert len(plan["encoded_examples_sha256"]) == 64
    assert len(plan["batch_order_sha256"]) == 64
    assert plan["consumed_example_count"] == 4
    assert plan["consumed_input_token_total"] == 8
    assert plan["consumed_loss_token_total"] == 4
    assert len(encoded) == 4 and replay == [*prior, *current]
    assert validate_plan_digest(plan, plan["plan_sha256"], qualification=False) == plan["plan_sha256"]


def test_first_increment_has_no_replay_and_formal_requires_frozen_plan():
    current = [example("c0", "task-new"), example("c1", "task-new")]
    plan, _, _ = build_sample_plan(
        current_examples=current,
        prior_replay_buffer=[],
        increment_index=0,
        tokenizer=object(),
        official=FakeOfficial,
        config=MatchedOrdinaryLoraConfig(),
    )
    assert plan["replay_example_ids"] == []
    assert plan["planned_optimizer_steps"] == 2
    with pytest.raises(ValueError, match="requires a frozen"):
        validate_plan_digest(plan, None, qualification=False)
    validate_plan_digest(plan, None, qualification=True)
    with pytest.raises(ValueError, match="differs"):
        validate_plan_digest(plan, "0" * 64, qualification=False)


def test_plan_freezes_only_batches_consumed_before_max_steps_cap():
    current = [example(f"c{index}", "task-new") for index in range(10)]
    prior = [example(f"p{index}", "task-old") for index in range(10)]
    plan, _, _ = build_sample_plan(
        current_examples=current,
        prior_replay_buffer=prior,
        increment_index=4,
        tokenizer=object(),
        official=FakeOfficial,
        config=MatchedOrdinaryLoraConfig(),
    )
    assert len(plan["optimizer_example_ids"]) == 20
    assert plan["planned_optimizer_steps"] == 16
    assert plan["consumed_example_count"] == 16
    assert len(plan["batch_order_item_sha256"]) == 16


def test_prior_state_is_one_continuing_rank32_adapter_with_content_receipts():
    config = MatchedOrdinaryLoraConfig()
    with TemporaryDirectory() as directory:
        adapter = Path(directory)
        (adapter / ADAPTER_WEIGHTS_FILE).write_bytes(b"weights")
        (adapter / ORDINARY_REPLAY_FILE).write_text("{}\n", encoding="utf-8")
        (adapter / ADAPTER_CONFIG_FILE).write_text(json.dumps({
            "r": 32,
            "lora_alpha": 32,
            "lora_dropout": 0.0,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": ["q_proj", "v_proj"],
        }), encoding="utf-8")
        state = {
            "schema": "openevo.webshop-ordinary-sequential-lora-state.v1",
            "base_model": config.base_model,
            "model_revision": config.model_revision,
            "increment_index": 2,
            "rank": 32,
            "target_modules": ["q_proj", "v_proj"],
            "adapter_model_sha256": sha256_file(adapter / ADAPTER_WEIGHTS_FILE),
            "trainable_tensor_sha256": "a" * 64,
            "replay_sha256": sha256_file(adapter / ORDINARY_REPLAY_FILE),
        }
        (adapter / ORDINARY_STATE_FILE).write_text(json.dumps(state), encoding="utf-8")
        assert load_prior_state(adapter, config, 3)["increment_index"] == 2
        state["increment_index"] = 1
        (adapter / ORDINARY_STATE_FILE).write_text(json.dumps(state), encoding="utf-8")
        with pytest.raises(ValueError, match="identity mismatch"):
            load_prior_state(adapter, config, 3)


def test_evidence_requires_real_update_reload_and_no_base_fallback():
    evidence = passing_evidence()
    validate_training_evidence(evidence)
    evidence["state_proof"]["after_trainable_tensor_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="change/reload"):
        validate_training_evidence(evidence)
    evidence = passing_evidence()
    evidence["no_base_fallback"]["adapter_loaded"] = False
    with pytest.raises(ValueError, match="no-BASE"):
        validate_training_evidence(evidence)


def test_evidence_rejects_nonfinite_loss_or_update_norm():
    evidence = passing_evidence()
    evidence["metrics"]["training_loss"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate_training_evidence(evidence)
    evidence = passing_evidence()
    evidence["state_proof"]["update_l2_norm"] = 0.0
    with pytest.raises(ValueError, match="optimizer/update"):
        validate_training_evidence(evidence)


def test_plan_digest_covers_replay_order_and_encoded_token_counts():
    current = [example("c0", "task-new"), example("c1", "task-new")]
    plan, _, _ = build_sample_plan(
        current_examples=current,
        prior_replay_buffer=[example("p0", "task-old")],
        increment_index=4,
        tokenizer=object(),
        official=FakeOfficial,
        config=MatchedOrdinaryLoraConfig(),
    )
    original = plan["plan_sha256"]
    changed = dict(plan)
    changed["input_token_counts"] = [999, *changed["input_token_counts"][1:]]
    changed.pop("plan_sha256")
    assert digest_json(changed) != original
