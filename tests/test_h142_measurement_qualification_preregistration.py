import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "configs" / "experiment" / "designs" / "h1.42-measurement-qualification-v1.json"
ACTIVE_PROTOCOL = ROOT / "configs" / "experiment" / "h1.42-measurement-qualification-v1.json"


def load_prereg():
    return json.loads(PREREG.read_text())


def test_h142_is_design_only_and_execution_closed():
    cfg = load_prereg()
    assert PREREG.is_file()
    assert not ACTIVE_PROTOCOL.exists(), "design-only H1.42 must not enter the active h*.json protocol registry before a separate execution release"
    assert cfg["phase"] == "H1.42-MQ"
    assert cfg["status"] == "preregistered-design-only-execution-closed"
    assert cfg["formal_task_consumption_allowed"] is False
    guard = cfg["formal_execution_guard"]
    assert guard["tasks_consumed"] == 0
    assert guard["gpu_hours_consumed"] == 0
    assert guard["arms_executed"] == []
    assert guard["t2_panel_evaluated"] is False
    assert guard["launcher_must_fail_closed_while_status_is_design_only"] is True


def test_h142_never_opens_training_harvest_or_t2():
    cfg = load_prereg()
    resources = cfg["resource_contract"]
    assert resources["new_training_attempts"] == 0
    assert resources["new_harvest_attempts"] == 0
    assert resources["t2_attempts"] == 0
    assert resources["planned_formal_attempts_if_later_activated"] == 656
    assert resources["gpu_hours_cap"] <= 4.5
    forbidden = "\n".join(cfg["forbidden"])
    assert "H1.40 or H1.41 T2" in forbidden
    assert "new training or adapter creation" in forbidden


def test_h142_positive_control_is_exact_h139_sd_vs_base():
    cfg = load_prereg()
    stage = cfg["measurement_stages"]["M1_positive_control"]
    assert stage["panel"] == "configs/experiment/manifests/h1.39-mr-independent-panel-v1.json"
    assert stage["panel_manifest_file_sha256"] == "cf2dc0726c34ed94640c36509d5a5018d1261d888b9a75a97fdaddbf54662e00"
    assert stage["tasks"] == 64
    assert stage["attempts"] == 512
    assert stage["arms"]["BASE"]["adapter"] == "none"
    assert stage["arms"]["H1.38B_SD_POSITIVE_CONTROL"]["adapter_sha256"] == "47e63d417f85cb2defed5ff1ee934128b96e6a4f78a8f410fa052957d919190c"
    gate = stage["measurement_validity_gate"]
    assert gate["minimum_jointly_usable_task_clusters"] == 58
    assert gate["maximum_invalid_attempt_rate_per_arm"] == 0.05
    assert gate["maximum_absolute_invalid_rate_asymmetry"] == 0.03
    assert "CI lower bound > 0" in gate["positive_control_requirement"]


def test_h142_null_control_uses_identical_artifact_and_outcome_blind_subset():
    cfg = load_prereg()
    stage = cfg["measurement_stages"]["M2_null_repeatability_control"]
    assert stage["attempts"] == 128
    assert len(stage["task_ids"]) == 16
    assert stage["task_ids"][0] == "webshop_000658"
    assert stage["task_ids"][-1] == "webshop_002131"
    a = stage["arms"]["SD_NULL_A"]["adapter_sha256"]
    b = stage["arms"]["SD_NULL_B"]["adapter_sha256"]
    assert a == b == "47e63d417f85cb2defed5ff1ee934128b96e6a4f78a8f410fa052957d919190c"
    assert "outcome-blind" in stage["task_selection"]
    gate = stage["measurement_validity_gate"]
    assert gate["maximum_absolute_point_estimate"] == 0.03
    assert "include zero" in gate["null_control_requirement"]


def test_h142_carries_forward_h141_invalidity_sentinels_without_mechanism_claim():
    cfg = load_prereg()
    sentinels = cfg["parent_h141"]["sentinel_tasks"]
    assert sentinels == ["webshop_003560", "webshop_004357"]
    stage = cfg["measurement_stages"]["M3_h141_sentinel_diagnosis"]
    assert stage["task_ids"] == sentinels
    assert stage["attempts"] == 16
    assert stage["classification_only"] is True
    assert "placeholders are forbidden" in stage["artifact_identity_rule"]
    assert cfg["analysis"]["mechanism_inference_forbidden"] is True


def test_h142_requires_separate_release_and_runtime_identity_freeze():
    cfg = load_prereg()
    req = cfg["activation_requirements"]
    assert req["separate_execution_release_required"] is True
    assert req["materialize_M0_cpu_autopsy"] is True
    assert req["materialize_exact_h141_C0_C1_artifact_hashes"] is True
    assert req["freeze_exact_openevo_experiment_sha"] is True
    assert req["freeze_exact_openevo_core_sha"] is True
    assert req["freeze_exact_run_profile_image_id"] is True
    assert req["reuse_existing_runtime_qualification_only_if_fingerprint_matches"] is True
    assert req["fresh_live_idle_gpu_gate"] is True
    assert req["no_open_t2"] is True
    assert "separate H1.42 execution release" in cfg["next_gate"]
