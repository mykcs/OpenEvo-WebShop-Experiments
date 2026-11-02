import copy
import hashlib
import json

import pytest

from scripts.analysis.advisor_q06_prediction_inputs import (
    S3_REFERENCE_CHECKPOINT, S3_REFERENCE_LINEAGE, audit_candidate_cost,
    select_s3_bank, split_rows,
)


def rows():
    return [dict(task_id=f"t{i % 2}", lineage_id=f"l{i // 2}", checkpoint_id=f"r{i}",
                 completed_rounds=40 + 40 * i, decision_step=1, feature_observed_step=1,
                 target_observed_step=2, panel_role="development", feature_phase="pre_action",
                 target_phase="post_action_or_later", feature_provenance_sha256="a" * 64,
                 features={"choice_count": 8, "click_entropy_normalized": .3}, target=.5)
            for i in range(4)]


@pytest.mark.parametrize("mode,heldout,key", [("task", ["t0"], "task_id"), ("lineage", ["l1"], "lineage_id")])
def test_group_leaveout_preserves_whole_group(mode, heldout, key):
    data = rows()
    result = split_rows(data, {"mode": mode, "heldout_ids": heldout})
    assert not ({data[i][key] for i in result["train_indices"]} &
                {data[i][key] for i in result["heldout_indices"]})


def test_time_split_holds_out_future_only():
    result = split_rows(rows(), {"mode": "time", "train_max_round": 80})
    assert result["train_indices"] == [0, 1]
    assert result["heldout_indices"] == [2, 3]


@pytest.mark.parametrize("change", [
    {"feature_observed_step": 2}, {"feature_phase": "post_action"},
    {"features": {"final_score": .5}}, {"features": {"episode_total_steps": 9}},
    {"panel_role": "final"}, {"features": {"choice_count": float("nan")}},
])
def test_leakage_and_final_rows_fail_closed(change):
    data = rows()
    data[0].update(change)
    with pytest.raises(ValueError):
        split_rows(data, {"mode": "time", "train_max_round": 80})


def test_derived_forward_cost_not_free_and_totals_must_match():
    receipt = {"classification": "derived_cpu_only_not_observed_forward_calls", "model_calls": 0,
               "environment_calls": 0, "arms": {"x": {"states_scored": 1,
               "state_receipts": [{"estimated_forward_calls": 12, "estimated_input_tokens": 90}],
               "groups": {"real": {"states": 1, "estimated_forward_calls": 12, "estimated_input_tokens": 90}}}}}
    assert audit_candidate_cost(receipt)["total_estimated_forward_calls"] == 12
    wrong = copy.deepcopy(receipt)
    wrong["arms"]["x"]["groups"]["real"]["estimated_forward_calls"] = 0
    with pytest.raises(ValueError, match="totals"):
        audit_candidate_cost(wrong)


def _s3_context(step):
    from scripts.analysis.advisor_q06_prediction_inputs import Q06_STATE_TYPE_DEFINITION

    def sha(value):
        return hashlib.sha256(value).hexdigest()

    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

    context = {
        "schema": "openevo.q06.pre-action-state.v1", "phase": "pre_action",
        "step_index": step, "completed_rounds": 40, "lineage_id": S3_REFERENCE_LINEAGE,
        "panel_role": "development", "state_type": "results",
        "state_type_definition": Q06_STATE_TYPE_DEFINITION,
        "feature_event_index": 0, "decision_event_index": 2, "action_event_index": 3,
        "event_order_definition": (
            "0 exact state snapshot; 1 generated uncertainty and click scorer diagnostics; "
            "2 action commit to environment; 3 environment transition call"
        ),
        "observation": "obs", "prompt": "prompt", "prompt_tokens": 3,
        "history": ["prior"], "rendered_history": "prior", "history_tokens": 1,
        "available_actions": {"clickables": ["a", "b"]}, "action_schema": {"type": "object"},
        "action_context_token_ids": [11, 12], "action_context_token_count": 2,
        "action_context_phase": "pre_action_action_channel_prefix_before_action_tokens",
        "token_count_definition": (
            "prompt_tokens and history_tokens are independently tokenized exact UTF-8 source strings "
            "using the pinned runtime tokenizer with add_special_tokens=False"
        ),
    }
    for key, value in (
        ("observation_sha256", "obs".encode()), ("prompt_sha256", "prompt".encode()),
        ("history_sha256", canonical(context["history"])),
        ("rendered_history_sha256", "prior".encode()),
        ("available_actions_sha256", canonical(context["available_actions"])),
        ("action_schema_sha256", canonical(context["action_schema"])),
        ("action_context_token_ids_sha256", canonical(context["action_context_token_ids"])),
    ):
        context[key] = sha(value)
    return context


def _s3_attempt(task, step, count, *, reward=.1):
    return {
        "engineering_invalid": False, "panel_role": "development",
        "lineage_id": S3_REFERENCE_LINEAGE, "checkpoint_id": S3_REFERENCE_CHECKPOINT,
        "task_identity": task,
        "episode": {"score": reward, "trace": [{
            "step": step, "q06_pre_action": _s3_context(step),
            "generation_diagnostics": {"click_choice": {
                "phase": "pre_action", "candidate_count": count, "source_event_index": 1,
                "entropy_nats": 0.2,
            }},
        }]},
    }


def _s3_source_spec(raw):
    return {"cell_id": "bounded-r40", "lineage_id": S3_REFERENCE_LINEAGE,
            "checkpoint_id": S3_REFERENCE_CHECKPOINT,
            "attempts_sha256": hashlib.sha256(raw).hexdigest()}


def test_s3_selector_is_hash_ordered_and_does_not_use_labels():
    records = []
    for low, high in ((5, 8), (9, 16), (17, 32), (33, 40)):
        for i in range(65):
            records.append(_s3_attempt(f"t{low}-{i}", i, low, reward=0 if i % 2 else 1))
    raw = b"".join((json.dumps(row, separators=(",", ":")) + "\n").encode() for row in records)
    result = select_s3_bank(_s3_source_spec(raw), raw)
    assert result["status"] == "S3_BANK_SELECTED"
    assert result["selected_count"] == 256
    assert [item["selected"] for item in result["coverage"]] == [64] * 4
    for item in result["selected"]:
        expected = hashlib.sha256(
            f"q06-bank-v1|bounded-r40|{item['task_identity']}|{item['step']}".encode()
        ).hexdigest()
        assert item["selection_sha256"] == expected
    relabeled = copy.deepcopy(records)
    for row in relabeled:
        row["episode"]["score"] = 0 if row["episode"]["score"] else 1
    relabeled_raw = b"".join((json.dumps(row, separators=(",", ":")) + "\n").encode() for row in relabeled)
    relabeled_result = select_s3_bank(_s3_source_spec(relabeled_raw), relabeled_raw)
    assert [(r["task_identity"], r["step"]) for r in result["selected"]] == [
        (r["task_identity"], r["step"]) for r in relabeled_result["selected"]
    ]


def test_s3_short_stratum_stays_partial_without_refill():
    records = [_s3_attempt(f"task-5-{i}", i, 5) for i in range(63)]
    records += [_s3_attempt(f"task-9-{i}", i, 9) for i in range(64)]
    records += [_s3_attempt(f"task-17-{i}", i, 17) for i in range(64)]
    records += [_s3_attempt(f"task-33-{i}", i, 33) for i in range(64)]
    raw = b"".join((json.dumps(row, separators=(",", ":")) + "\n").encode() for row in records)
    result = select_s3_bank(_s3_source_spec(raw), raw)
    assert result["status"] == "PARTIAL"
    assert result["selected_count"] == 255
    assert result["coverage"][0]["selected"] == 63


@pytest.mark.parametrize("mutation", ["digest", "state_type", "action_tokens"])
def test_s3_rejects_context_tampering(mutation):
    attempt = _s3_attempt("task-a", 0, 5)
    context = attempt["episode"]["trace"][0]["q06_pre_action"]
    if mutation == "digest":
        context["prompt"] = "changed"
    elif mutation == "state_type":
        context["state_type_definition"] = "post-outcome-derived"
    else:
        context["action_context_token_ids"][0] = 99
    raw = (json.dumps(attempt, separators=(",", ":")) + "\n").encode()
    with pytest.raises(ValueError):
        select_s3_bank(_s3_source_spec(raw), raw)


def test_s3_requires_frozen_reference_actor_and_source_hash():
    raw = (json.dumps(_s3_attempt("task-a", 0, 5)) + "\n").encode()
    spec = _s3_source_spec(raw)
    wrong_actor = {**spec, "cell_id": "gdr-r40"}
    with pytest.raises(ValueError, match="bounded-r40"):
        select_s3_bank(wrong_actor, raw)
    wrong_hash = {**spec, "attempts_sha256": "0" * 64}
    with pytest.raises(ValueError, match="exact digest"):
        select_s3_bank(wrong_hash, raw)
