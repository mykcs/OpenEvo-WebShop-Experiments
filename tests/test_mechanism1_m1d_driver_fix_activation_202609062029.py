from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "configs/experiment/activations/openevo-mechanism1-m1d-stage1-execution-release-202609062029.json"
PREVIOUS = ROOT / "configs/experiment/activations/openevo-mechanism1-m1d-stage1-execution-release-202609061825.json"
ROUTER = ROOT / "configs/experiment/campaigns/current-mechanism1-autonomous.json"
M1D_CLOSEOUT = ROOT / "docs/evidence/mechanism1/m1d-minimax-closeout-202609071201.json"
DRIVER = ROOT / "scripts/openevo_webshop/run_mechanism1_m1d_seed_stage1_202609060100.py"
MINIMAX = ROOT / "configs/experiment/designs/openevo-mechanism1-m1d-minimax-posthoc-202609061925.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_successor_release_is_parser_only_and_preconsumption():
    release = load(RELEASE)
    prior = release["authority"]["supersedes_activation_release"]
    assert prior["path"] == str(PREVIOUS.relative_to(ROOT))
    assert prior["sha256"] == sha(PREVIOUS)
    assert prior["scope"] == "driver argument parsing only"
    assert release["scientific_contract_changed"] is False
    assert release["resource_authority_changed"] is False
    assert release["driver_runtime_authority_changed"] is True
    assert release["formal_state_before_release"]["baseline_rollout_records"] == 0
    assert release["formal_state_before_release"]["m1d_formal_trajectories_consumed"] == 0
    assert release["driver_runtime_fix"]["formal_trajectory_consumption_before_repair"] == 0
    assert release["driver_runtime_fix"]["scientific_semantics_changed"] is False


def test_release_binds_fixed_driver_and_frozen_minimax_contract():
    release = load(RELEASE)
    assert release["frozen_bindings"]["driver"]["sha256"] == sha(DRIVER)
    assert release["frozen_bindings"]["minimax_posthoc_binding"]["sha256"] == sha(MINIMAX)
    assert "ArgumentParser(allow_abbrev=False)" in DRIVER.read_text(encoding="utf-8")


def test_router_points_to_parser_fix_successor():
    router = load(ROUTER)
    m1d = router["m1d_readiness"]
    assert m1d["activation_release"] == str(RELEASE.relative_to(ROOT))
    assert m1d["activation_release_sha256"] == sha(RELEASE)
    assert m1d["driver_sha256"] == sha(DRIVER)
    assert m1d["minimax_posthoc_binding"] == str(MINIMAX.relative_to(ROOT))
    assert m1d["minimax_posthoc_binding_sha256"] == sha(MINIMAX)
    if m1d["formal_task_consumption_allowed"] is False:
        closeout = load(M1D_CLOSEOUT)
        assert closeout["status"] == "PASS"
        assert closeout["parse_ok"] == 1440 and closeout["parse_failures"] == 0
