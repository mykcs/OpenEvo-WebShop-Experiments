import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/experiment/designs/openevo-mechanism1-m1d-policy-endpoint-qualification-202609061100.json"
ROUTER = ROOT / "configs/experiment/campaigns/current-mechanism1-autonomous.json"
M1D_CLOSEOUT = ROOT / "docs/evidence/mechanism1/m1d-minimax-closeout-202609071201.json"
ASSET = ROOT / "configs/assets/models/qwen2.5-3b-instruct.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def current_activation_path() -> Path:
    return ROOT / load(ROUTER)["m1d_readiness"]["activation_release"]


def test_endpoint_plan_is_fail_closed_and_benchmark_free():
    d = load(PLAN)
    assert d["activation_authority"] is False
    assert d["formal_task_consumption_allowed"] is False
    req = d["synthetic_request"]
    assert req["benchmark_task_identity"] is None
    assert req["webshop_reset_count"] == 0
    assert req["formal_task_consumption"] == 0
    assert req["final_panel_access"] == 0
    assert req["temperature"] == 0.4
    assert req["request_level_sampling_seed"] is None


def test_original_base_identity_is_exact_and_merged_base_is_forbidden():
    d = load(PLAN)
    asset = load(ASSET)
    ident = d["model_identity"]
    assert ident["asset_registry_sha256"] == sha(ASSET)
    assert ident["revision"] == asset["revision"]
    assert ident["content_pin"] == asset["content_pin"]
    assert ident["required_files_sha256"] == asset["required_files_sha256"]
    assert any("Stage2 merged" in item for item in d["forbidden_model_inputs"])
    assert d["endpoint_runtime"]["adapter_mounts_allowed"] is False


def test_ray_contract_uses_one_secondary_spare_and_requires_release():
    d = load(PLAN)["ray_contract"]
    assert d["workload"] == "mechanism1_seed_stage1_3b"
    assert d["slot_class"] == "secondary_spare"
    assert d["selection_count"] == 1
    assert d["vram_request_mib_per_gpu"] == 9000
    assert d["outer_ray_is_only_gpu_authority"] is True
    assert d["selection_forbidden_until_original_base_hash_pass"] is True
    assert d["release_required_after_smoke"] is True


def test_current_router_binds_plan_and_only_activates_via_separate_release():
    r = load(ROUTER)
    m = r["m1d_readiness"]
    activation = current_activation_path()
    release = load(activation)
    assert r["execution_activation_authority"] is False
    assert load(PLAN)["activation_authority"] is False
    assert load(PLAN)["formal_task_consumption_allowed"] is False
    assert m["policy_endpoint_qualification_plan"] == str(PLAN.relative_to(ROOT))
    assert m["policy_endpoint_qualification_plan_sha256"] == sha(PLAN)
    assert m["activation_release_sha256"] == sha(activation)
    assert release["activation_authority"] is True
    assert release["formal_task_consumption_allowed"] is True
    if m["activation_authority"] is False:
        closeout = load(M1D_CLOSEOUT)
        assert closeout["status"] == "PASS" and closeout["parse_ok"] == 1440
        assert m["formal_task_consumption_allowed"] is False
