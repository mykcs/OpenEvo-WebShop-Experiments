import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/experiment/designs/openevo-mechanism1-m1d-policy-endpoint-qualification-202609061100.json"
AMEND = ROOT / "configs/experiment/amendments/openevo-mechanism1-m1d-endpoint-gpu3-resource-202609061300.json"
AUX = ROOT / "configs/experiment/openevo-mechanism1-m1d-gpu3-ray-qualification-202609061300.json"
REG = ROOT / "configs/experiment/openevo-mechanism1-m1d-gpu3-ray-profile-registry-202609061300.json"
ROUTER = ROOT / "configs/experiment/campaigns/current-mechanism1-autonomous.json"
M1D_CLOSEOUT = ROOT / "docs/evidence/mechanism1/m1d-minimax-closeout-202609071201.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def current_activation_path() -> Path:
    return ROOT / load(ROUTER)["m1d_readiness"]["activation_release"]


def test_gpu3_amendment_changes_only_resource_binding_and_keeps_activation_locked():
    a = load(AMEND)
    assert a["parent_qualification_plan_sha256"] == sha(PLAN)
    assert a["override_scope"] == "physical Ray pool binding only"
    assert a["scientific_semantics_unchanged"] is True
    assert a["model_identity_inherited_exactly_from_parent"] is True
    assert a["synthetic_request_inherited_exactly_from_parent"] is True
    assert a["activation_authority"] is False
    assert a["formal_task_consumption_allowed"] is False


def test_auxiliary_ray_config_is_gpu3_only_and_cannot_schedule_base_stage2_profiles():
    c = load(AUX)
    assert c["managed_gpus"] == [3]
    assert c["managed_gpu_uuids"] == {"3": "GPU-00000000-0000-0000-0000-000000000000"}
    assert c["profiles"] == {}
    assert c["status"] == "QUALIFICATION_ONLY"
    assert c["ray_tmp_dir"].endswith("m1d-gpu3-202609061300")


def test_gpu3_registry_is_one_gpu_9g_qualification_only():
    c = load(AUX)
    r = load(REG)
    assert r["compatible_base_config_sha256"] == [sha(AUX)]
    assert set(r["profiles"]) == {"mechanism1_seed_stage1_3b"}
    p = r["profiles"]["mechanism1_seed_stage1_3b"]
    assert (p["min_gpus"], p["target_gpus"], p["max_gpus"]) == (1, 1, 1)
    assert p["vram_request_mib"] == 9000
    assert p["slot_class"] == "secondary_spare"


def test_amendment_binds_exact_auxiliary_ray_identities_and_requires_release():
    a = load(AMEND)["resource_contract"]
    assert a["gpu_index"] == 3
    assert a["gpu_uuid"] == "GPU-00000000-0000-0000-0000-000000000000"
    assert a["auxiliary_config_sha256"] == sha(AUX)
    assert a["auxiliary_profile_registry_sha256"] == sha(REG)
    assert a["selection_count"] == 1
    assert a["vram_request_mib_per_gpu"] == 9000
    assert a["main_gpu4567_pool_restart_allowed"] is False
    assert a["main_gpu4567_pool_state_mutation_allowed"] is False
    assert a["release_required_after_smoke"] is True


def test_router_preserves_historical_gpu3_qualification_and_uses_current_activation_separately():
    r = load(ROUTER)
    m = r["m1d_readiness"]
    activation = current_activation_path()
    release = load(activation)
    assert r["execution_activation_authority"] is False
    assert load(AMEND)["activation_authority"] is False
    assert load(AMEND)["formal_task_consumption_allowed"] is False
    assert m["policy_endpoint_resource_amendment"] == str(AMEND.relative_to(ROOT))
    assert m["policy_endpoint_resource_amendment_sha256"] == sha(AMEND)
    assert m["activation_release_sha256"] == sha(activation)
    assert release["resource_authority"]["vram_request_mib"] == 9000
    if m["activation_authority"] is False:
        closeout = load(M1D_CLOSEOUT)
        assert closeout["status"] == "PASS" and closeout["parse_failures"] == 0
        assert m["formal_task_consumption_allowed"] is False
