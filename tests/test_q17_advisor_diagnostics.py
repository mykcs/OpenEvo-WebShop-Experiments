from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "docs/evidence/q17-advisor-diagnostics-20260911"
MANIFEST = ROOT / "configs/experiment/manifests/q17-advisor-fixed-checkpoint-panel-20260911.json"


def load(path: Path):
    return json.loads(path.read_text())


def test_fixed_panel_and_final_panel_boundary():
    manifest = load(MANIFEST)
    assert manifest["panel_size"] == 32
    assert len(manifest["tasks"]) == 32
    assert len({row["task_identity"] for row in manifest["tasks"]}) == 32
    assert manifest["final_panel_access_allowed"] is False
    assert manifest["ceiling_final_panel_exclusion"]["goal_overlap_count"] == 0
    assert manifest["ceiling_final_panel_exclusion"]["instruction_hash_overlap_count"] == 0


def test_instrumentation_equivalence_is_exact():
    q = load(PACK / "instrumentation-equivalence-q4.json")
    assert q["status"] == "PASS"
    assert q["task_count"] == 4
    assert q["exact_behavior_equivalent"] is True
    assert q["final_panel_access"] == 0
    assert q["formal_task_consumption"] == 0


def test_r127_r128_claim_boundary():
    q = load(PACK / "r127-vs-r128-comparison.json")
    assert q["status"] == "PASS"
    assert q["task_count"] == 32
    assert q["transitions"] == {"both_loss": 21, "both_win": 7, "loss_to_win": 1, "win_to_loss": 3}
    lo, hi = q["mean_score_diff_bootstrap_95ci"]
    assert lo < 0 < hi
    slo, shi = q["success_rate_diff_bootstrap_95ci"]
    assert slo < 0 < shi
    assert q["final_panel_access"] == 0
    assert q["formal_task_consumption"] == 0


def test_existing_log_pack_is_sealed_only_and_records_fallback():
    q = load(PACK / "existing-log-metrics-R000-R141.json")
    assert q["round_end"] == 141
    assert q["sealed_round_count"] == 142
    assert q["active_unsealed_rounds"] == [142]
    assert q["episode_fallback_rounds"] == [24]
    assert q["final_panel_access"] == 0
    assert q["model_calls"] == 0
