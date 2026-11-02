import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "configs/experiment/openevo-mechanism1-m1d-shared-gpu0123-ray-202609061825.json"
REGISTRY = ROOT / "configs/experiment/openevo-mechanism1-m1d-shared-gpu0123-ray-profile-registry-202609061825.json"
RELEASE = ROOT / "configs/experiment/activations/openevo-mechanism1-m1d-stage1-execution-release-202609061825.json"
HISTORICAL_RELEASE = ROOT / "configs/experiment/activations/openevo-mechanism1-m1d-stage1-execution-release-202609061500.json"
ROUTER = ROOT / "configs/experiment/campaigns/current-mechanism1-autonomous.json"
M1D_CLOSEOUT = ROOT / "docs/evidence/mechanism1/m1d-minimax-closeout-202609071201.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_shared_pool_snapshot_and_registry_are_exactly_bound():
    pool = load(POOL)
    registry = load(REGISTRY)
    assert sha(POOL) == "e85c39a592b904e3253818e2a37bdfc2e021efb2cbefbbdeb11b45098b9ba34b"
    assert pool["managed_gpus"] == [0, 1, 2, 3]
    assert pool["managed_gpu_uuids"] == {
        "0": "GPU-00000000-0000-0000-0000-000000000000",
        "1": "GPU-00000000-0000-0000-0000-000000000000",
        "2": "GPU-00000000-0000-0000-0000-000000000000",
        "3": "GPU-00000000-0000-0000-0000-000000000000",
    }
    assert pool["ray_version"] == "2.57.0"
    assert pool["scientific_invariants"]["preemption"] is False
    assert registry["compatible_base_config_sha256"] == [sha(POOL)]
    assert set(registry["profiles"]) == {"mechanism1_seed_stage1_3b"}
    profile = registry["profiles"]["mechanism1_seed_stage1_3b"]
    assert profile["arm"] == "mechanism1-seed-stage1-3b"
    assert (profile["min_gpus"], profile["target_gpus"], profile["max_gpus"]) == (1, 1, 1)
    assert profile["vram_request_mib"] == 9000
    assert profile["slot_class"] == "secondary_spare"


def test_successor_release_changes_resource_authority_only_and_keeps_zero_state():
    release = load(RELEASE)
    assert release["status"] == "PASS"
    assert release["activation_authority"] is True
    assert release["formal_task_consumption_allowed"] is True
    assert release["scientific_contract_changed"] is False
    assert release["resource_authority_changed"] is True
    prior = release["authority"]["supersedes_activation_release"]
    assert prior["path"] == str(HISTORICAL_RELEASE.relative_to(ROOT))
    assert prior["sha256"] == sha(HISTORICAL_RELEASE)
    assert prior["scope"] == "resource authority only"
    assert release["formal_state_before_release"]["baseline_rollout_records"] == 0
    assert release["formal_state_before_release"]["m1d_formal_trajectories_consumed"] == 0
    assert release["formal_budget"]["trajectory_count"] == 1440
    assert release["formal_budget"]["minimax_parse_ok_target"] == 1440
    assert release["formal_budget"]["seed_stage2"] == 0
    assert release["formal_budget"]["final_panel_access"] == 0
    assert release["other_mechanism_phases"] == {"M1-A": "LOCKED", "M1-B": "LOCKED", "M1-C": "LOCKED"}


def test_successor_release_binds_shared_ray_identities_and_one_gpu_policy():
    release = load(RELEASE)
    frozen = release["frozen_bindings"]
    assert frozen["shared_ray_pool_config"]["sha256"] == sha(POOL)
    assert frozen["shared_ray_profile_registry"]["sha256"] == sha(REGISTRY)
    resource = release["resource_authority"]
    assert resource["managed_gpus"] == [0, 1, 2, 3]
    assert resource["selected_gpu_count"] == 1
    assert resource["vram_request_mib"] == 9000
    assert resource["workload"] == "mechanism1_seed_stage1_3b"
    assert resource["slot_class"] == "secondary_spare"
    assert resource["no_preemption"] is True
    assert resource["no_pool_restart_for_m1d"] is True
    assert resource["pool_config_sha256"] == sha(POOL)
    assert resource["profile_registry_sha256"] == sha(REGISTRY)


def test_router_preserves_shared_resource_authority_and_minimax_closeout_gate():
    router = load(ROUTER)
    m1d = router["m1d_readiness"]
    current_release = load(ROOT / m1d["activation_release"])
    assert router["execution_activation_authority"] is False
    assert m1d["activation_release_sha256"] == sha(ROOT / m1d["activation_release"])
    assert m1d["shared_ray_pool_config"] == str(POOL.relative_to(ROOT))
    assert m1d["shared_ray_pool_config_sha256"] == sha(POOL)
    assert m1d["shared_ray_profile_registry"] == str(REGISTRY.relative_to(ROOT))
    assert m1d["shared_ray_profile_registry_sha256"] == sha(REGISTRY)
    assert current_release["resource_authority"]["pool_config_sha256"] == sha(POOL)
    assert current_release["resource_authority"]["profile_registry_sha256"] == sha(REGISTRY)
    if m1d["formal_task_consumption_allowed"] is False:
        closeout = load(M1D_CLOSEOUT)
        assert closeout["status"] == "PASS"
        assert closeout["records"] == closeout["parse_ok"] == closeout["unique_trajectory_ids"] == 1440
        assert closeout["parse_failures"] == 0
        assert "CLOSED" in m1d["remaining_gate"]
