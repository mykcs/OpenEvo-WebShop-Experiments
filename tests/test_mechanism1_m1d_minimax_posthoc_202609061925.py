import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/experiment/designs/openevo-mechanism1-m1d-minimax-posthoc-202609061925.json"
ANALYZER = ROOT / "docs/evidence/mechanism1/m1d-minimax-runtime-202609061925/h146_analyze_trajectories.py"

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def test_m1d_minimax_binding_is_frozen_before_formal_outcomes():
    d = load(PLAN)
    assert d["status"] == "PREREGISTERED_ZERO_FORMAL_TRAJECTORIES"
    assert d["scientific_contract_changed"] is False
    assert d["formal_state_at_binding"] == {
        "m1d_baseline_rollout_records": 0,
        "m1d_minimax_records": 0,
        "final_panel_access": 0,
        "seed_stage2_rollouts": 0,
    }
    assert d["source_bindings"]["m1d_design"]["sha256"] == sha(ROOT / d["source_bindings"]["m1d_design"]["path"])
    assert d["source_bindings"]["m1d_activation"]["sha256"] == sha(ROOT / d["source_bindings"]["m1d_activation"]["path"])

def test_analyzer_identity_and_generation_contract_are_exact():
    d = load(PLAN)["analyzer"]
    assert sha(ANALYZER) == d["payload_sha256"] == "46f9ecda2b912baf17261541df277ca87289f7aaaeb27ef946d089d36938b940"
    assert d["provider"] == "MiniMax" and d["model"] == "MiniMax-M3"
    assert d["analysis_prompt_version"] == "seed"
    assert d["skill_mode"] == "episode_only"
    assert d["thinking_type"] == "adaptive" and d["reasoning_split"] is True
    assert d["temperature"] == 0.0
    assert d["max_completion_tokens"] == 1024
    assert d["workers"] == 4
    assert (d["transport_attempts"], d["parse_attempts"], d["timeout_seconds"]) == (2, 2, 120)
    assert d["resume_success_definition"] == "parse_ok=true only"

def test_new_seed_close_and_existing_openevo_comparator_are_symmetric():
    p = load(PLAN)
    new = p["new_seed_corpus"]
    old = p["existing_openevo_3b_comparator"]
    assert (new["required_trajectory_count"], new["required_parse_ok"], new["required_parse_failures"]) == (1440, 1440, 0)
    assert new["new_webshop_calls_during_analysis"] == 0
    assert old["reuse_allowed_only_if_identity_matches"] is True
    assert old["minimax_parse_ok"] == 1440 and old["minimax_parse_failures"] == 0
    assert old["analyzer_sha256"] == p["analyzer"]["payload_sha256"]
    assert old["model"] == p["analyzer"]["model"]
    assert old["thinking_type"] == p["analyzer"]["thinking_type"]
    assert old["reasoning_split"] == p["analyzer"]["reasoning_split"]
    assert "SEED Stage2" in p["forbidden"]
    assert "MiniMax 1440/1440 parse_ok" in p["hard_stop"]
