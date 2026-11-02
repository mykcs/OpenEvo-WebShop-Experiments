import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.analysis.post_advisor_bounded_rank_sweep_train_202609282338 import (
    supports_state_rank,
)
from scripts.analysis import post_advisor_bounded_rank_sweep_successor_202609282338 as rank_sweep_successor
from scripts.analysis import post_advisor_bounded_rank32_successor_trace_202609282338 as rank32_successor
from scripts.analysis import post_advisor_bounded_rank_sweep_train_202609282338 as rank_sweep_engine
from scripts.analysis import post_advisor_bounded_rank32_train_trace_202609282338 as rank32_engine
from scripts.openevo_webshop import post_advisor_bounded_rank_sweep_transition_driver_202609282338 as driver
from scripts.openevo_webshop import post_advisor_bounded_rank32_transition_driver_202609282338 as rank32_driver
from scripts.openevo_webshop import post_advisor_bounded_rank_sweep_transition_driver_202609282338 as rank_sweep_driver
from scripts.openevo_webshop.post_advisor_short_fork_runner_202609282338 import (
    supports_persistent_state_rank,
)
from scripts.openevo_webshop import bounded_effective_state_launch_guard_202609282338 as launch_guard


def test_short_fork_r152_uses_exact_external_r151_health_without_local_history(tmp_path, monkeypatch):
    from scripts.openevo_webshop import ceiling1_stage2_arm_controller as base_controller
    from scripts.openevo_webshop.post_advisor_bounded_short_fork_202609282338 import BoundedEffectiveStateController

    state_path = tmp_path / "historical-state.json"
    health_path = tmp_path / "historical-health.json"
    state_path.write_text(json.dumps({
        "state_id": "sealed-r151", "next_round": 152,
        "rollouts_consumed": 152 * 128, "final_panel_access_count": 0,
    }))
    health_path.write_text(json.dumps({
        "schema": "openevo.carrier-health-boundary-decision.v1",  # legacy-revision-compat
        "round_index": 151, "action": "CONTINUE", "ledger_status": "PASS",
        "block_next_round": False, "policy_identity_sha256": "sealed-policy",
    }))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    controller = object.__new__(BoundedEffectiveStateController)
    controller.post_advisor_short_fork = True
    controller.post_advisor_short_fork_authority = {
        "historical_r151_state_path": str(state_path),
        "historical_r151_state_sha256": digest(state_path),
        "historical_r151_health_path": str(health_path),
        "historical_r151_health_sha256": digest(health_path),
        "historical_r151_state_id": "sealed-r151",
    }
    controller.state = {
        "state_id": "sealed-r151", "post_advisor_source_state_id": "sealed-r151",
        "next_round": 152, "post_advisor_fork_origin": "R151_AFTER",
    }
    controller.carrier_health_policy_identity = "sealed-policy"
    # This fixture intentionally bypasses __init__; production initialization
    # binds the optional A9 resolution receipt or leaves it unset.
    controller.a9_resolution_sha = None
    controller.out = tmp_path / "new-arm"
    delegated = []
    monkeypatch.setattr(base_controller.Controller, "ensure_previous_carrier_health_boundary",
                        lambda self, r: delegated.append(r))
    controller.ensure_previous_carrier_health_boundary(152)
    assert delegated == []
    assert not controller.out.exists()
    controller.ensure_previous_carrier_health_boundary(153)
    assert delegated == [153]

    health_path.write_text(health_path.read_text().replace('"CONTINUE"', '"PAUSE_FOR_DIAGNOSTIC"'))
    with pytest.raises(RuntimeError, match="SHA drift"):
        controller.ensure_previous_carrier_health_boundary(152)
    controller.post_advisor_short_fork_authority["historical_r151_health_sha256"] = digest(health_path)
    with pytest.raises(RuntimeError, match="blocks R152"):
        controller.ensure_previous_carrier_health_boundary(152)


def test_rank8_is_supported_and_unregistered_rank_is_rejected():
    assert supports_state_rank(8)
    assert supports_persistent_state_rank(8)
    assert not supports_state_rank(12)
    assert not supports_persistent_state_rank(12)


def test_transition_driver_accepts_rank8_and_rejects_unregistered_rank(monkeypatch):
    argv = [
        "transition-driver",
        "--dataset-manifest", "manifest.json",
        "--dataset-artifact-id", "dataset",
        "--output-dir", "output",
        "--model-path", "model",
        "--round-index", "152",
        "--transition-id", "transition",
        "--treatment-mode", driver.OFF_MODE,
        "--empty-history-manifest", "empty.json",
        "--adapter-config-template", "adapter.json",
        "--state-rank", "8",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert driver.parse_args().state_rank == 8

    monkeypatch.setattr(sys, "argv", [*argv[:-1], "12"])
    with pytest.raises(SystemExit):
        driver.parse_args()


def test_post_advisor_activation_requires_authority_and_rejects_unpassed_status(tmp_path, monkeypatch):
    execution_repo = tmp_path / "execution"
    execution_repo.mkdir()
    design_rel = Path(launch_guard.POST_ADVISOR_DESIGN)
    activation_rel = Path(launch_guard.POST_ADVISOR_ACTIVATION)
    campaign_rel = Path(launch_guard.CURRENT_CAMPAIGN)
    (execution_repo / design_rel).parent.mkdir(parents=True)
    (execution_repo / activation_rel).parent.mkdir(parents=True)
    design = {"schema": "design"}
    (execution_repo / design_rel).write_text(json.dumps(design))
    design_sha = hashlib.sha256((execution_repo / design_rel).read_bytes()).hexdigest()
    campaign = {
        "campaign_id": launch_guard.POST_ADVISOR_CAMPAIGN_ID,
        "design": launch_guard.POST_ADVISOR_DESIGN,
        "formal_task_consumption_allowed": False,
        "task_bearing_train_dev_allowed": False,
        "final_test_status": "locked",
        "final_panel_access_allowed": False,
        "final_unlock_allowed": False,
    }
    (execution_repo / campaign_rel).parent.mkdir(parents=True, exist_ok=True)
    (execution_repo / campaign_rel).write_text(json.dumps(campaign))
    run_git = lambda repo, *args: subprocess.run(
        ["git", "-C", str(repo), *args], check=True, text=True, capture_output=True
    ).stdout.strip()
    run_git(execution_repo, "init", "-q")
    run_git(execution_repo, "config", "user.name", "Post-advisor activation test")
    run_git(execution_repo, "config", "user.email", "activation-test@example.invalid")
    activation = {
        "schema": "openevo.post-advisor-unanswered-activation.202609282338",
        "revision": "202609282338",
        "status": "PENDING_ZERO_FORMAL_PREFLIGHT",
        "execution_sha": None,
        "design_path": launch_guard.POST_ADVISOR_DESIGN,
        "design_sha256": design_sha,
        "campaign_id": launch_guard.POST_ADVISOR_CAMPAIGN_ID,
        "formal_task_consumption_allowed": False,
        "final_panel_access_allowed": False,
    }
    activation_path = execution_repo / activation_rel
    activation_path.write_text(json.dumps(activation))
    run_git(execution_repo, "add", str(design_rel), str(activation_rel), str(campaign_rel))
    run_git(execution_repo, "commit", "-qm", "qualified execution A")
    execution_sha = run_git(execution_repo, "rev-parse", "HEAD")

    authority_repo = tmp_path / "authority"
    subprocess.run(["git", "clone", "-q", str(execution_repo), str(authority_repo)], check=True)
    run_git(authority_repo, "config", "user.name", "Post-advisor authority test")
    run_git(authority_repo, "config", "user.email", "authority-test@example.invalid")
    authority_material = authority_repo / "configs/experiment/activations/post-advisor-test-authority.json"
    authority_material.write_text('{"schema":"test-authority"}\n')
    run_git(authority_repo, "add", str(authority_material.relative_to(authority_repo)))
    run_git(authority_repo, "commit", "-qm", "release B authority material")
    activation["execution_sha"] = execution_sha
    activation_path = authority_repo / activation_rel
    activation_path.write_text(json.dumps(activation))
    authority_campaign = authority_repo / campaign_rel
    authority_campaign.write_text(json.dumps({**campaign, "authority_stage": "C"}))
    run_git(authority_repo, "add", str(activation_rel), str(campaign_rel))
    run_git(authority_repo, "commit", "-qm", "activation C pending")

    with monkeypatch.context() as patch:
        patch.delenv(launch_guard.AUTHORITY_REPO_ENV, raising=False)
        with pytest.raises(RuntimeError, match=launch_guard.AUTHORITY_REPO_ENV):
            launch_guard.validate_post_advisor_activation(execution_repo, execution_sha)
    monkeypatch.setenv(launch_guard.AUTHORITY_REPO_ENV, str(authority_repo))
    with pytest.raises(RuntimeError, match="not PASS"):
        launch_guard.validate_post_advisor_activation(execution_repo, execution_sha)
    with pytest.raises(RuntimeError, match="execution SHA"):
        launch_guard.validate_post_advisor_activation(execution_repo, "0" * 40)

    (execution_repo / design_rel).write_text('{"schema":"changed"}')
    with pytest.raises(RuntimeError, match="clean committed checkout"):
        launch_guard.validate_post_advisor_activation(execution_repo, execution_sha)


@pytest.mark.parametrize(
    ("module", "args"),
    [
        (rank_sweep_driver, {"round_index": 152, "prior_adapter": Path("prior"), "prior_replay": Path("replay"), "treatment_mode": rank_sweep_driver.OFF_MODE, "gdr_runtime_path": None, "beta_policy_path": None}),
        (rank32_driver, {"round_index": 152, "prior_adapter": Path("prior"), "prior_replay": Path("replay"), "treatment_mode": rank32_driver.OFF_MODE, "gdr_runtime_path": None, "beta_policy_path": None}),
    ],
)
def test_standalone_transition_drivers_require_activation_before_training(module, args, monkeypatch):
    monkeypatch.setattr(module, "parse_args", lambda: type("Args", (), args)())
    monkeypatch.setattr(module, "validate_post_advisor_activation", lambda *_: (_ for _ in ()).throw(RuntimeError("activation is not PASS")))
    with pytest.raises(SystemExit, match="activation is not PASS"):
        module.main()


@pytest.mark.parametrize("module", [rank_sweep_successor, rank32_successor])
def test_direct_successor_entrypoints_require_activation_before_training(module, monkeypatch):
    monkeypatch.setattr(module, "parse_args", lambda: object())
    monkeypatch.setattr(module, "validate_post_advisor_activation", lambda *_: (_ for _ in ()).throw(RuntimeError("activation is not PASS")))
    with pytest.raises(SystemExit, match="activation is not PASS"):
        module.main()


@pytest.mark.parametrize("module", [rank_sweep_engine, rank32_engine])
def test_direct_training_engines_require_activation(module, monkeypatch):
    monkeypatch.setattr(module, "validate_post_advisor_activation", lambda *_: (_ for _ in ()).throw(RuntimeError("activation is not PASS")))
    with pytest.raises(RuntimeError, match="activation is not PASS"):
        module.train(object())


@pytest.mark.parametrize("module", [rank_sweep_successor, rank32_successor])
def test_empty_state_successor_helpers_require_activation(module, monkeypatch):
    monkeypatch.setattr(module, "validate_post_advisor_activation", lambda *_: (_ for _ in ()).throw(RuntimeError("activation is not PASS")))
    with pytest.raises(RuntimeError, match="activation is not PASS"):
        module.train_empty(object(), object())

from scripts.analysis.post_advisor_bounded_rank_sweep_train_202609282338 import (
    scalar_gate_coefficients,
    should_select_magnitude_beta,
)
import math


def test_zero_effective_update_skips_beta_search():
    assert not should_select_magnitude_beta({"raw_update_frobenius_norm": 0.0})


def test_nonzero_effective_update_runs_beta_search():
    assert should_select_magnitude_beta({"raw_update_frobenius_norm": 1.0})


def test_beta_interpolates_the_prior_coefficient_from_previous_state():
    base_prior = 1.0
    raw_prior, raw_current = 0.83, -0.61
    beta = 0.5
    gated_prior, gated_current = scalar_gate_coefficients(base_prior, raw_prior, raw_current, beta)
    assert gated_prior == pytest.approx(base_prior + beta * (raw_prior - base_prior))
    assert gated_current == pytest.approx(beta * raw_current)
    assert (gated_prior - base_prior, gated_current) == pytest.approx(
        (beta * (raw_prior - base_prior), beta * raw_current)
    )


def test_beta_requires_a_finite_positive_weight_at_most_one():
    for beta in (0.0, -0.25, 1.01, math.nan, math.inf):
        with pytest.raises(ValueError):
            scalar_gate_coefficients(1.0, 0.9, 0.5, beta)


def test_nonfinite_effective_update_norm_is_rejected():
    with pytest.raises(ValueError, match="must be finite"):
        should_select_magnitude_beta({"raw_update_frobenius_norm": float("nan")})
