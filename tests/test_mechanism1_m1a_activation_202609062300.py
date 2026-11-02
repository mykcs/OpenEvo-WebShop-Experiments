import hashlib
import json
from pathlib import Path

from scripts.openevo_webshop import mechanism1_m1a_activation_gate_202609062300 as gate

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "configs/experiment/activations/openevo-mechanism1-m1a-adjacent-execution-release-202609062300.json"
ROUTER = ROOT / "configs/experiment/campaigns/current-mechanism1-autonomous.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_activation_release_hash_binds_the_exact_preparation_chain():
    release = json.loads(RELEASE.read_text())
    assert release["status"] == "READY_CONDITIONAL_CURRENT_MAIN"
    assert release["activation_authority"] is True
    assert release["formal_task_consumption_allowed"] is True
    assert release["formal_episode_cap"] == 576
    assert release["scientific_contract"]["task_count"] == 64
    assert release["scientific_contract"]["cell_count"] == 9
    assert release["scientific_contract"]["episodes_per_cell"] == 64
    assert release["scientific_contract"]["r27_to_r49_update_task_overlap_count"] == 0
    assert release["scientific_contract"]["ceiling_final_panel_goal_overlap_count"] == 0
    assert release["scientific_contract"]["ceiling_final_panel_instruction_hash_overlap_count"] == 0
    assert release["scientific_contract"]["random_direction_seeds"] == [2026090501, 2026090502, 2026090503]
    for row in release["frozen_preparation"].values():
        assert sha(ROOT / row["path"]) == row["sha256"]


def test_router_binds_release_and_gate_but_authority_is_conditional():
    release = json.loads(RELEASE.read_text())
    router = json.loads(ROUTER.read_text())
    m1a = router["m1a_lineage"]
    assert m1a["activation_authority"] is True
    assert m1a["formal_task_consumption_allowed"] is True
    assert m1a["authority_effective_only_when_current_main_gate_passes"] is True
    assert m1a["authority_requires_clean_worktree"] is True
    assert m1a["activation_release_sha256"] == sha(RELEASE)
    gate_path = ROOT / release["authority_gate"]["path"]
    assert release["authority_gate"]["sha256"] == sha(gate_path)
    assert m1a["authority_gate_sha256"] == sha(gate_path)


def test_open_branch_identity_is_not_authorized_even_with_valid_static_chain():
    result = gate.evaluate_authority(ROOT, "activation-branch-head", "current-main-head", working_tree_clean=True)
    assert result["authorized"] is False
    assert result["status"] == "NOT_AUTHORIZED_NOT_CURRENT_MAIN"


def test_dirty_current_main_identity_is_not_authorized():
    result = gate.evaluate_authority(ROOT, "same-head", "same-head", working_tree_clean=False)
    assert result["authorized"] is False
    assert result["status"] == "NOT_AUTHORIZED_DIRTY_WORKTREE"


def test_exact_clean_current_main_identity_is_the_only_authorized_case():
    result = gate.evaluate_authority(ROOT, "same-head", "same-head", working_tree_clean=True)
    assert result["authorized"] is True
    assert result["status"] == "AUTHORIZED_CURRENT_MAIN"


def test_activation_release_does_not_rewrite_preparation_or_qualification_accounting():
    release = json.loads(RELEASE.read_text())
    q = release["qualification_contract"]
    z = release["zero_state_before_activation"]
    assert q["formal_webshop_tasks_consumed"] == 0
    assert q["formal_webshop_episodes_consumed"] == 0
    assert q["final_panel_access"] == 0
    assert q["external_teacher_calls"] == 0
    assert q["execution_authority_during_qualification"] is False
    assert z["formal_webshop_tasks_consumed"] == 0
    assert z["formal_webshop_episodes_consumed"] == 0
    assert z["formal_results_observed"] is False
