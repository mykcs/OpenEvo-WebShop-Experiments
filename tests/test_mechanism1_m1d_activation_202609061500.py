import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_RELEASE = ROOT / "configs/experiment/activations/openevo-mechanism1-m1d-stage1-execution-release-202609061500.json"
ROUTER = ROOT / "configs/experiment/campaigns/current-mechanism1-autonomous.json"
M1D_CLOSEOUT = ROOT / "docs/evidence/mechanism1/m1d-minimax-closeout-202609071201.json"
MANIFEST = ROOT / "configs/experiment/manifests/openevo-mechanism1-m1d-seed-stage1-matched-202609060000.json"
DRIVER = ROOT / "scripts/openevo_webshop/run_mechanism1_m1d_seed_stage1_202609060100.py"
HISTORICAL_DRIVER = ROOT / "docs/evidence/mechanism1/m1d-driver-runtime-202609061500/run_mechanism1_m1d_seed_stage1_202609060100.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def current_release_path() -> Path:
    router = load(ROUTER)
    return ROOT / router["m1d_readiness"]["activation_release"]


def test_historical_m1d_stage1_release_remains_narrow_and_hash_bound():
    release = load(HISTORICAL_RELEASE)
    assert release["status"] == "PASS"
    assert release["activation_authority"] is True
    assert release["formal_task_consumption_allowed"] is True
    assert release["scientific_contract_changed"] is False
    assert release["formal_budget"] == {
        "final_panel_access": 0,
        "history_length": 2,
        "max_steps": 15,
        "rollouts_per_task": 8,
        "seed_skill_generation": 0,
        "seed_stage2": 0,
        "seed_training": 0,
        "task_count": 180,
        "trajectory_count": 1440,
    }
    assert release["other_mechanism_phases"] == {"M1-A": "LOCKED", "M1-B": "LOCKED", "M1-C": "LOCKED"}
    for name, bound in release["frozen_bindings"].items():
        if name == "driver":
            assert sha256(HISTORICAL_DRIVER) == bound["sha256"]
        else:
            assert sha256(ROOT / bound["path"]) == bound["sha256"]


def test_router_activates_only_m1d_stage1_via_current_hash_bound_release():
    router = load(ROUTER)
    m1d = router["m1d_readiness"]
    release_path = current_release_path()
    release = load(release_path)
    assert router["execution_activation_authority"] is False
    assert m1d["activation_release_sha256"] == sha256(release_path)
    assert release["activation_authority"] is True
    assert release["formal_task_consumption_allowed"] is True
    assert release["other_mechanism_phases"] == {"M1-A": "LOCKED", "M1-B": "LOCKED", "M1-C": "LOCKED"}
    assert m1d["qualification_server_receipt_sha256"] == "67d3e8f3295b99366e50cbba8461fdd5895905f04ae3065493c726d9fbc46d54"
    if m1d["activation_authority"] is False:
        closeout = load(M1D_CLOSEOUT)
        assert closeout["status"] == "PASS"
        assert closeout["parse_ok"] == 1440 and closeout["parse_failures"] == 0
        assert m1d["formal_task_consumption_allowed"] is False
        assert m1d["minimax_closeout_evidence_sha256"] == sha256(M1D_CLOSEOUT)

def test_frozen_manifest_remains_locked_and_driver_hard_stops():
    manifest = load(MANIFEST)
    assert manifest["activation_authority"] is False
    assert manifest["formal_task_consumption_allowed"] is False
    text = DRIVER.read_text(encoding="utf-8")
    assert "args.stop_after_baseline_rollouts = True" in text
    assert '"seed_stage2_started": False' in text
    assert '"trajectory_count": TRAJECTORY_COUNT' in text
