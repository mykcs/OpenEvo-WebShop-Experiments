import hashlib
import json
from pathlib import Path

from scripts.openevo_webshop.validate_h141_preregistration import validate


ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "configs/experiment/h1.41-magnitude-screen-v2.json"


def _config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_h141_is_preregistered_and_t2_closed():
    config = _config()
    assert config["status"] == "preregistered-pre-execution"
    assert config["formal_task_consumption_allowed"] is False
    assert config["resource_contract"]["t2_attempts"] == 0
    assert config["formal_execution_guard"]["t2_panel_evaluated"] is False
    assert config["parent_closeout"]["t2_panel_reuse_forbidden"] is True


def test_h141_arms_change_only_magnitude_initialization():
    config = _config()
    control = config["frozen_scientific_contract"]["arms"]["C0_exact_h140"]
    treatment = config["frozen_scientific_contract"]["arms"]["C1_reset_magnitude"]
    assert control["magnitude_policy"] == "warm_start_prior_magnitudes_v1"
    assert treatment["magnitude_policy"] == "reset_all_to_coefficient_init_v1"
    assert control["reset_all_to_coefficient_init"] is False
    assert treatment["reset_all_to_coefficient_init"] is True
    assert config["frozen_scientific_contract"]["rank"] == 4
    assert config["frozen_scientific_contract"]["replay_capacity"] == 64
    assert config["frozen_scientific_contract"]["source_selection"]["total_records"] == 16


def test_h141_patch_digest_is_pinned():
    config = _config()
    patch = ROOT / config["frozen_scientific_contract"]["arms"]["C1_reset_magnitude"]["upstream_patch"]
    digest = hashlib.sha256(patch.read_bytes()).hexdigest()
    assert digest == config["frozen_scientific_contract"]["arms"]["C1_reset_magnitude"]["upstream_patch_sha256"]


def test_h141_selector_has_explicit_invalidity_and_no_rescue_arm():
    config = _config()
    selector = config["analysis"]["mechanism_selector"]
    assert "MEASUREMENT_INVALID" in selector
    assert "C1_MECHANISM_SUPPORTED" in selector
    assert "C0_ONLY_PASS" in selector
    forbidden = set(config["forbidden"])
    assert "rank reduction" in forbidden
    assert "replay policy change" in forbidden
    assert "extra seeds/tasks/retries after a stopping rule" in forbidden


def test_h141_resource_denominator_is_derived_from_frozen_panels():
    config = _config()
    panels = config["frozen_scientific_contract"]["evaluation_panels"]
    arms = config["frozen_scientific_contract"]["arms"]
    expected = sum(len(panel["task_ids"]) * len(panel["rollout_seeds"]) for panel in panels.values()) * len(arms)
    assert config["resource_contract"]["formal_task_attempts"] == expected == 384


def test_h141_preregistration_is_machine_valid_before_successor_activation():
    result = validate(ROOT)
    assert result["status"] == "pass"
    current = json.loads((ROOT / "configs/experiment/current-campaign.json").read_text())
    assert result["router_campaign"] == current["campaign_id"]


def test_h141_deterministic_reference_is_task_level_seed_invariant():
    config = _config()
    reference_path = ROOT / config["source_artifacts"]["g1_deterministic_reference"]
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    seeds = reference["source"]["historical_rollout_seeds"]
    assert reference["source"]["temperature"] == 0.0
    assert len(seeds) == 4
    assert sum(len(rows) for rows in reference["panels"].values()) == 48
    for rows in reference["panels"].values():
        for row in rows:
            assert row["valid_attempt_count"] == 4
            assert set(row["scores_by_seed"]) == {str(seed) for seed in seeds}
            assert len(set(row["scores_by_seed"].values())) == 1
