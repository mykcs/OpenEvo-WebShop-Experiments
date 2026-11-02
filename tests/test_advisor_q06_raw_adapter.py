import hashlib
import json

import pytest

from scripts.analysis.advisor_q06_raw_adapter import adapt_attempts_bytes


def attempt(*, with_context=True, phase="pre_action", panel="development"):
    context = {
        "schema": "openevo.q06.pre-action-state.v1",
        "phase": phase,
        "step_index": 0,
        "feature_event_index": 10,
        "decision_event_index": 11,
        "action_event_index": 12,
        "completed_rounds": 40,
        "prompt_tokens": 100,
        "history_tokens": 20,
        "state_type": "product_results",
        "observation_sha256": "1" * 64,
        "prompt_sha256": "2" * 64,
        "history_sha256": "3" * 64,
        "available_actions_sha256": "4" * 64,
    }
    trace = {
        "step": 0,
        "generation_diagnostics": {
            "click_choice": {"candidate_count": 2, "entropy_nats": 0.5, "phase": "pre_action", "source_event_index": 10},
            "reasoning": {"generated_token_count": 2, "entropy_sum_nats": 0.3, "phase": "pre_action", "source_event_index": 10},
            "action": {"generated_token_count": 1, "entropy_sum_nats": 0.1, "phase": "pre_action", "source_event_index": 10},
        },
    }
    if with_context:
        trace["q06_pre_action"] = context
    return {
        "task_identity": "dev-task-1",
        "lineage_id": "lineage-1",
        "checkpoint_id": "checkpoint-40",
        "panel_role": panel,
        "engineering_invalid": False,
        "episode": {"score": 0.75, "won": True, "steps": 1, "trace": [trace]},
    }


def payload(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def test_adapter_hashes_exact_source_and_emits_only_pre_action_registered_features():
    raw = payload(attempt())
    result = adapt_attempts_bytes(raw, lineage_id="lineage-1", checkpoint_id="checkpoint-40")

    assert result["status"] == "CPU_ADAPTER_PASS"
    row = result["rows"][0]
    assert result["source_file_sha256"] == hashlib.sha256(raw).hexdigest()
    assert row["feature_provenance_sha256"] == hashlib.sha256(raw).hexdigest()
    assert row["feature_phase"] == "pre_action"
    assert row["target_phase"] == "post_action_or_later"
    assert row["features"] == {
        "round": 40,
        "choice_count": 2,
        "step_index": 0,
        "prompt_tokens": 100,
        "history_tokens": 20,
        "click_entropy_normalized": pytest.approx(0.5 / __import__("math").log(2)),
        "reasoning_entropy_mean": pytest.approx(0.15),
        "action_entropy_mean": pytest.approx(0.1),
    }
    assert row["target"] == 0.75
    assert row["identity_binding"] == "caller_asserted_against_raw_attempt_fields"


def test_current_attempts_shape_without_snapshot_is_typed_partial_not_imputed():
    current_shape = attempt(with_context=False)
    result = adapt_attempts_bytes(payload(current_shape), lineage_id="lineage-1", checkpoint_id="checkpoint-40")

    assert result["status"] == "PARTIAL"
    assert result["normalized_row_count"] == 0
    assert "q06_pre_action snapshot" in result["rejected_records"][0]["reason"]


@pytest.mark.parametrize("mutation,reason", [
    ({"phase": "post_action"}, "phase"),
    ({"feature_event_index": 11}, "strictly before"),
])
def test_adapter_rejects_future_or_not_pre_action_provenance(mutation, reason):
    raw_attempt = attempt()
    raw_attempt["episode"]["trace"][0]["q06_pre_action"].update(mutation)
    result = adapt_attempts_bytes(payload(raw_attempt), lineage_id="lineage-1", checkpoint_id="checkpoint-40")

    assert result["status"] == "PARTIAL"
    assert result["normalized_row_count"] == 0
    assert reason in result["rejected_records"][0]["reason"]


def test_adapter_rejects_final_panel_and_unbound_identity():
    final_attempt = attempt(panel="final")
    final_attempt["episode"]["trace"][0]["generation_diagnostics"]["click_choice"]["candidate_count"] = 1
    final_row = adapt_attempts_bytes(payload(final_attempt), lineage_id="lineage-1", checkpoint_id="checkpoint-40")
    wrong_identity = adapt_attempts_bytes(payload(attempt()), lineage_id="other-lineage", checkpoint_id="checkpoint-40")

    assert final_row["normalized_row_count"] == 0
    assert "development panel" in final_row["rejected_records"][0]["reason"]
    assert final_row["excluded_no_destination_states"] == 0
    assert wrong_identity["normalized_row_count"] == 0
    assert "bound lineage/checkpoint" in wrong_identity["rejected_records"][0]["reason"]


@pytest.mark.parametrize("channel", ["click_choice", "reasoning", "action"])
@pytest.mark.parametrize("event_index", [11, 12])
def test_adapter_rejects_entropy_features_without_pre_action_event_binding(channel, event_index):
    raw_attempt = attempt()
    raw_attempt["episode"]["trace"][0]["generation_diagnostics"][channel]["source_event_index"] = event_index
    result = adapt_attempts_bytes(payload(raw_attempt), lineage_id="lineage-1", checkpoint_id="checkpoint-40")

    assert result["status"] == "PARTIAL"
    assert result["normalized_row_count"] == 0
    assert "precede the decision" in result["rejected_records"][0]["reason"]


def test_adapter_reports_no_destination_exclusions_by_record_and_step():
    raw_attempt = attempt()
    raw_attempt["episode"]["trace"][0]["generation_diagnostics"]["click_choice"]["candidate_count"] = 1
    result = adapt_attempts_bytes(payload(raw_attempt), lineage_id="lineage-1", checkpoint_id="checkpoint-40")

    assert result["excluded_no_destination_states"] == 1
    assert result["excluded_decisions"][0]["source_record_number"] == 1
    assert result["excluded_decisions"][0]["decision_step"] == 0


def test_adapter_rejects_negative_candidate_count_as_invalid():
    raw_attempt = attempt()
    raw_attempt["episode"]["trace"][0]["generation_diagnostics"]["click_choice"]["candidate_count"] = -1
    result = adapt_attempts_bytes(payload(raw_attempt), lineage_id="lineage-1", checkpoint_id="checkpoint-40")

    assert result["rejected_records"]
    assert "nonnegative" in result["rejected_records"][0]["reason"]
    assert result["excluded_no_destination_states"] == 0
