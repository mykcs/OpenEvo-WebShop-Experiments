import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "configs/experiment/designs/openevo-mechanism1-taskvector-causal-transplant-202609052200.json"
AMENDMENT = ROOT / "configs/experiment/amendments/openevo-mechanism1-m1a-identifiability-202609062100.json"
SUCCESSOR = ROOT / "configs/experiment/designs/openevo-mechanism1-taskvector-causal-transplant-202609062100.json"
M1C_GATE = ROOT / "configs/experiment/amendments/openevo-mechanism1-m1c-m1a-successor-gate-202609062100.json"
ROUTER = ROOT / "configs/experiment/campaigns/current-mechanism1-autonomous.json"
M1D_CLOSEOUT = ROOT / "docs/evidence/mechanism1/m1d-minimax-closeout-202609071201.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_original_preregistration_is_preserved_and_blocked():
    original = load(ORIGINAL)
    amendment = load(AMENDMENT)
    assert original["source_selection"]["source_vector"] == "theta_100pct - theta_75pct"
    assert original["source_selection"]["transplant_target"] == "theta_50pct"
    assert amendment["discovery_source"]["result"] == "ZERO_VECTOR_NOT_DIRECTION_IDENTIFIABLE"
    assert amendment["predecessor"]["disposition"] == "PRESERVED_HISTORICAL_PREREGISTRATION_BLOCKED_NOT_IDENTIFIABLE"
    assert amendment["predecessor"]["may_execute"] is False
    assert amendment["predecessor"]["may_supply_m1c_signal"] is False


def test_successor_selects_adjacent_distinct_states_without_reward():
    successor = load(SUCCESSOR)
    source = successor["source_selection"]
    assert source["previous_state"].startswith("theta_prev = S_(K-1)")
    assert source["transplant_target"].startswith("theta_target = S_(K-2)")
    assert source["source_vector"] == "tau_adjacent_late = theta_final - theta_prev"
    assert source["minimum_distinct_states"] == 3
    assert source["selection_uses_reward"] is False
    assert successor["execution_authority"] is False
    assert successor["formal_task_consumption_allowed"] is False


def test_identifiability_gate_fails_closed_before_task_consumption():
    successor = load(SUCCESSOR)
    gate = successor["identifiability_gate"]
    requirements = " ".join(gate["required"])
    assert gate["evaluated_before_probe_manifest_materialization"] is True
    assert "pairwise distinct" in requirements
    assert "consecutive" in requirements
    assert "strictly greater than zero" in requirements
    assert gate["failure_status"] == "NOT_IDENTIFIABLE"
    assert "zero WebShop episodes" in gate["on_failure"]


def test_budget_controls_and_ceiling_boundary_are_unchanged():
    original = load(ORIGINAL)
    successor = load(SUCCESSOR)
    assert successor["budget"] == original["budget"]
    assert successor["controls"] == original["controls"]
    assert successor["external_teacher_calls"] == original["external_teacher_calls"] == 0
    assert successor["final_panel_access"] == original["final_panel_access"] == 0
    assert successor["preserves"]["ceiling_evidence"] == "IMMUTABLE"


def test_m1c_can_only_consume_the_successor_signal():
    gate = load(M1C_GATE)
    successor = load(SUCCESSOR)
    assert gate["priority_1"]["source_passport"] == str(SUCCESSOR.relative_to(ROOT))
    assert gate["priority_1"]["required_source_status"] == "SEALED_IDENTIFIABLE_COMPLETE"
    assert gate["priority_1"]["direction"] == "tau_adjacent_late"
    assert successor["m1c_signal_contract"]["predecessor_signal_forbidden"] is True
    assert any("predecessor zero-vector" in rule for rule in gate["forbidden"])


def test_current_router_preserves_m1d_authority_and_m1a_lineage_identity():
    router = load(ROUTER)
    lineage = router["m1a_lineage"]
    m1d = router["m1d_readiness"]
    assert lineage["predecessor"]["disposition"] == "BLOCKED_NOT_IDENTIFIABLE"
    assert lineage["successor_passport"] == str(SUCCESSOR.relative_to(ROOT))
    if m1d["activation_authority"] is False:
        closeout = load(M1D_CLOSEOUT)
        assert closeout["status"] == "PASS"
        assert closeout["parse_ok"] == closeout["unique_trajectory_ids"] == 1440
        assert closeout["parse_failures"] == 0
        assert m1d["formal_task_consumption_allowed"] is False
        assert m1d["minimax_closeout_evidence"] == str(M1D_CLOSEOUT.relative_to(ROOT))
