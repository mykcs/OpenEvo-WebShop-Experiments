import copy

import pytest

from scripts.analysis.advisor_q08_retention_analysis import analyse, frozen_tasks


def payload():
    tasks = frozen_tasks()
    return {"panel_role": "development", "lineage_id": "historical-one", "planned_checkpoints": [40, 80, 120, 160],
            "audit_status": "COMPLETE",
            "late_checkpoint": 160, "checkpoint_status": {str(n): "AVAILABLE" for n in (40, 80, 120, 160)},
            "tasks": tasks,
            "events": [{"task_id": t["task_id"], "round": 50, "update_id": "u50", "rollout_consumed": True,
                        "training_included": True, "accepted_update": True, "state_before": "a", "state_after": "b",
                        "adapter_digest_before": "b" * 64, "adapter_digest_after": "c" * 64,
                        "update_receipt_sha256": "a" * 64} for t in tasks],
            "scores": [{"task_id": t["task_id"], "completed_rounds": cp, "score": score, "engineering_valid": True,
                        "evaluation_seed": 7} for t in tasks for cp, score in ((40, .1), (80, .8), (120, .7), (160, .5))]}


def test_acquisition_and_retention_are_separate_and_fixed_cohort():
    result = analyse(payload(), resamples=1000)
    row = result["task_rows"][0]
    assert row["first_effective_exposure"] == 50
    assert row["acquisition_delta"] == pytest.approx(.7)
    assert row["retention_delta"] == pytest.approx(-.3)
    assert result["cohorts"]["MID_SEEN"]["retention"]["training_replicates"] == 1


def test_rollout_and_noop_never_become_learning():
    data = payload()
    data["events"][0].update(training_included=False, accepted_update=False)
    row = analyse(data, resamples=1000)["task_rows"][0]
    assert row["first_rollout_exposure"] == 50
    assert row["status"] == "NO_EFFECTIVE_EXPOSURE"
    assert row["retention_delta"] is None


def test_missing_first_post_is_typed_partial_not_later_substitute():
    data = payload()
    data["checkpoint_status"]["80"] = "MISSING_PAYLOAD"
    data["scores"] = [r for r in data["scores"] if r["completed_rounds"] != 80]
    row = analyse(data, resamples=1000)["task_rows"][0]
    assert row["first_post_checkpoint"] == 80
    assert row["status"] == "PARTIAL_MISSING_FIRST_POST_CHECKPOINT"
    assert row["retention_delta"] is None


@pytest.mark.parametrize("change", [{"accepted_update": True, "training_included": False}, {"state_after": "a"},
                                     {"update_receipt_sha256": "missing"}, {"adapter_digest_after": "b" * 64}])
def test_unproven_accepted_update_fails_closed(change):
    data = payload()
    data["events"][0].update(change)
    with pytest.raises(ValueError):
        analyse(data, resamples=1000)


def test_changed_eval_seed_or_duplicate_scores_rejected():
    data = payload()
    data["scores"][1]["evaluation_seed"] = 8
    with pytest.raises(ValueError, match="seed changed"):
        analyse(data, resamples=1000)
    data = copy.deepcopy(payload())
    data["scores"].append(data["scores"][0])
    with pytest.raises(ValueError, match="duplicate"):
        analyse(data, resamples=1000)


def test_engineering_invalid_evaluation_is_retained_without_score_imputation():
    data = payload()
    data["scores"][0].update(score=None, engineering_valid=False)
    result = analyse(data, resamples=1000)
    first = next(row for row in result["task_rows"] if row["task_id"] == data["scores"][0]["task_id"])
    assert first["acquisition_delta"] is None
    assert first["retention_delta"] == pytest.approx(-.3)
    assert first["status"] == "RETENTION_MEASURED_ACQUISITION_UNVERIFIED"


def test_initial_baseline_allows_acquisition_without_replacing_missing_post():
    data = payload()
    data["planned_checkpoints"].insert(0, 0)
    data["checkpoint_status"]["0"] = "AVAILABLE"
    data["scores"].append({"task_id": data["tasks"][0]["task_id"], "completed_rounds": 0, "score": .1,
                           "engineering_valid": True, "evaluation_seed": 7})
    data["events"][0]["round"] = 10
    row = analyse(data, resamples=1000)["task_rows"][0]
    assert row["pre_exposure_checkpoint"] == 0
    assert row["acquisition_delta"] == pytest.approx(0)


def test_exposure_after_last_checkpoint_is_invalid():
    data = payload()
    data["events"][0]["round"] = 160
    with pytest.raises(ValueError, match="round"):
        analyse(data, resamples=1000)


def test_deleted_missing_checkpoint_cannot_shift_first_post():
    data = payload()
    data["planned_checkpoints"].remove(80)
    del data["checkpoint_status"]["80"]
    data["scores"] = [r for r in data["scores"] if r["completed_rounds"] != 80]
    with pytest.raises(ValueError, match="frozen checkpoints"):
        analyse(data, resamples=1000)


def test_trimmed_or_relabelled_cohort_cannot_look_complete():
    data = payload()
    data["tasks"].pop()
    with pytest.raises(ValueError, match="original96"):
        analyse(data, resamples=1000)
    data = payload()
    data["tasks"][0]["task_id"] = "substituted"
    with pytest.raises(ValueError, match="exact frozen"):
        analyse(data, resamples=1000)


@pytest.mark.parametrize("audit_status", [None, "UNKNOWN", "PARTIAL"])
def test_missing_or_noncomplete_audit_status_fails_closed(audit_status):
    data = payload()
    if audit_status is None:
        del data["audit_status"]
    else:
        data["audit_status"] = audit_status
    result = analyse(data, resamples=1000)
    assert result["status"] == "PARTIAL_AUDIT_INPUT"
    assert "task_rows" not in result


def test_complete_status_with_audit_gaps_fails_closed():
    data = payload()
    data["audit_gaps"] = [{"round": 4, "reason": "unproven"}]
    result = analyse(data, resamples=1000)
    assert result["status"] == "PARTIAL_AUDIT_INPUT"
    assert "audit_gaps is nonempty" in result["audit_reason"]


def test_partial_training_audit_preserves_descriptive_scores_only():
    data = payload()
    data["audit_status"] = "PARTIAL"
    data["audit_gaps"] = [{"round": 4, "reason": "TRAINER_INCLUSION_PREDICATE_UNBOUND"}]
    data["scores"] = [r for r in data["scores"] if r["completed_rounds"] == 80]
    result = analyse(data, resamples=1000)
    assert result["status"] == "PARTIAL_AUDIT_INPUT"
    assert "task_rows" not in result and "cohorts" not in result
    observations = result["evaluation_observations"]
    assert observations["status"] == "DESCRIPTIVE_EVALUATION_ONLY"
    for cohort in observations["checkpoints"]["80"]["cohorts"].values():
        assert cohort["coverage_complete"] is True
        assert cohort["mean_score"] == pytest.approx(.8)
    assert observations["checkpoints"]["40"]["cohorts"]["EARLY_SEEN"]["mean_score"] is None


def test_partial_audit_still_rejects_invalid_evaluation_facts():
    data = payload()
    data["audit_status"] = "PARTIAL"
    data["scores"][0]["score"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        analyse(data, resamples=1000)
    data = payload()
    data["audit_status"] = "PARTIAL"
    data["scores"][1]["evaluation_seed"] = 8
    with pytest.raises(ValueError, match="seed changed"):
        analyse(data, resamples=1000)
