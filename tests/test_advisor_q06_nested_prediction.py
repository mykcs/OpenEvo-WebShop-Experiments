import copy

import pytest

from scripts.analysis import advisor_q06_nested_prediction as q06
from scripts.analysis.advisor_q06_raw_adapter import SCHEMA


def rows():
    result = []
    for i in range(24):
        task = f"task-{i:02d}"
        round_no = i // 6
        click = (i % 7) / 7
        result.append({
            "schema": SCHEMA,
            "task_id": task,
            "lineage_id": f"lineage-{i % 6}",
            "checkpoint_id": f"checkpoint-{round_no}",
            "completed_rounds": round_no,
            "decision_step": 0,
            "feature_observed_step": 0,
            "target_observed_step": 1,
            "panel_role": "development",
            "feature_phase": "pre_action",
            "target_phase": "post_action_or_later",
            "feature_provenance_sha256": "a" * 64,
            "source_file_sha256": "b" * 64,
            "source_record_number": i + 1,
            "feature_event_index": 10,
            "decision_event_index": 11,
            "action_event_index": 12,
            "pre_action_context_sha256": "c" * 64,
            "state_type": "product_results" if i % 2 else "search_results",
            "features": {
                "round": round_no,
                "choice_count": 5 + i % 8,
                "step_index": 0,
                "prompt_tokens": 100 + i * 3,
                "history_tokens": 10 + i,
                "click_entropy_normalized": click,
                "reasoning_entropy_mean": 0.1 + click / 10,
                "action_entropy_mean": 0.05 + click / 20,
            },
            "target": min(1.0, max(0.0, 0.2 + 0.5 * click + (i % 3) * 0.03)),
            "secondary_targets": {"exact_success": int(i % 4 == 0)},
        })
    return result


def splits():
    return {
        "task": {"mode": "task", "heldout_ids": ["task-00", "task-01", "task-02", "task-03"]},
        "time": {"mode": "time", "train_max_round": 1},
        "lineage": {"mode": "lineage", "heldout_ids": ["lineage-4", "lineage-5"]},
    }


def test_outer_task_time_lineage_protocols_are_grouped_and_partial_when_coverage_is_thin(monkeypatch):
    monkeypatch.setattr(q06, "BOOTSTRAPS", 200)
    data = rows()
    for spec in splits().values():
        train, test = q06.outer_partition(data, spec)
        if spec["mode"] == "task":
            assert not ({data[i]["task_id"] for i in train} & {data[i]["task_id"] for i in test})
        elif spec["mode"] == "lineage":
            assert not ({data[i]["lineage_id"] for i in train} & {data[i]["lineage_id"] for i in test})
        else:
            assert not ({data[i]["task_id"] for i in train} & {data[i]["task_id"] for i in test})
    result = q06.run_nested_prediction(data, splits(), inner_folds=3)
    assert result["status"] == "COMPLETE_DIAGNOSTIC"
    assert result["final_panel_access"] == 0
    assert result["model_calls"] == result["environment_calls"] == 0
    assert set(result["protocols"]) == {"task", "time", "lineage"}
    assert all("simultaneous_ci95" in v for v in result["protocols"].values())


def test_outer_labels_do_not_enter_inner_tuning(monkeypatch):
    monkeypatch.setattr(q06, "BOOTSTRAPS", 100)
    data = rows()
    held = splits()["task"]["heldout_ids"]
    baseline = q06.run_nested_prediction(data, {"task": splits()["task"]}, inner_folds=3)
    changed = copy.deepcopy(data)
    for row in changed:
        if row["task_id"] in held:
            row["target"] = 1.0 - row["target"]
    rerun = q06.run_nested_prediction(changed, {"task": splits()["task"]}, inner_folds=3)
    assert baseline["protocols"]["task"]["tuning"] == rerun["protocols"]["task"]["tuning"]


@pytest.mark.parametrize("change", [
    {"feature_phase": "post_action"},
    {"panel_role": "final"},
    {"features": {"round": 0, "choice_count": 5, "step_index": 0, "prompt_tokens": 10,
                   "history_tokens": 1, "click_entropy_normalized": 0.2, "reasoning_entropy_mean": 0.1,
                   "action_entropy_mean": 0.05, "final_reward": 1}},
    {"feature_provenance_sha256": "not-a-hash"},
    {"target_observed_step": 0},
    {"feature_event_index": 11},
])
def test_leakage_and_unbound_sources_fail_closed(change):
    data = rows()
    data[0].update(change)
    with pytest.raises(ValueError):
        q06.validate_normalized_rows(data)


def test_holm_adjustment_is_monotone_and_bounded():
    adjusted = q06.holm_adjust({"task": 0.01, "time": 0.04, "lineage": 0.03})
    assert adjusted == {"task": pytest.approx(0.03), "lineage": pytest.approx(0.06), "time": pytest.approx(0.06)}
    assert q06.holm_adjust({"task": 0.02}, family_size=3)["task"] == pytest.approx(0.06)


def test_unestimable_lineage_protocol_cannot_emit_stop_or_continue_decision(monkeypatch):
    monkeypatch.setattr(q06, "BOOTSTRAPS", 100)
    spec = {"lineage": {"mode": "lineage", "heldout_ids": ["lineage-5"]}}
    result = q06.run_nested_prediction(rows(), spec, inner_folds=3)["protocols"]["lineage"]

    assert result["status"] == "PARTIAL"
    assert result["decision"] == "PARTIAL"
    assert result["secondary_exact_success_brier"]["decision"] == "PARTIAL"


def test_task_balanced_loss_gives_each_task_equal_weight():
    sample = [
        {"task_id": "short", "target": 0.0},
        {"task_id": "long", "target": 1.0},
        {"task_id": "long", "target": 1.0},
        {"task_id": "long", "target": 1.0},
    ]
    assert q06._task_balanced_loss(sample, [0.0, 0.0, 0.0, 0.0]) == pytest.approx(0.5)


def test_repeating_one_task_does_not_change_task_weighted_fit_or_tuning():
    data = rows()
    train = data[:18]
    predict = data[18:]
    original_alpha, _ = q06._tune_alpha(train, "uncertainty", inner_folds=3)
    original_predictions = q06._fit_predict(train, predict, "uncertainty", original_alpha)

    repeated = train + [copy.deepcopy(row) for row in train if row["task_id"] == "task-00"] * 4
    repeated_alpha, _ = q06._tune_alpha(repeated, "uncertainty", inner_folds=3)
    repeated_predictions = q06._fit_predict(repeated, predict, "uncertainty", repeated_alpha)

    assert repeated_alpha == original_alpha
    assert repeated_predictions == pytest.approx(original_predictions)
