import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(rel: str):
    return json.loads((ROOT / rel).read_text())


def test_cross_campaign_budget_scope_registry():
    guard = load("configs/experiment/contracts/webshop-budget-scope-v1.json")
    ceiling = guard["scopes"]["openevo-ceiling-1.0"]
    wb1_guard = guard["scopes"]["wb1-seed-paper-faithful"]
    assert ceiling["rounds"] == 160
    assert ceiling["rollouts_per_round"] == 128
    assert ceiling["total_rollouts"] == 20_480
    assert ceiling["rounds"] * ceiling["rollouts_per_round"] == ceiling["total_rollouts"]
    assert ceiling["completion_state"] == "next_round == 160"
    assert wb1_guard["updates"] == 150
    assert wb1_guard["total_valid_episodes"] == 19_200
    assert ceiling["rounds"] != wb1_guard["updates"]


def test_wb1_protocol_remains_paper_faithful_150_updates():
    guard = load("configs/experiment/contracts/webshop-budget-scope-v1.json")
    expected = guard["scopes"]["wb1-seed-paper-faithful"]
    wb1 = load("configs/experiment/wb1/protocol.json")
    matched = wb1["seed_webshop_extraction"]["stage_counts"]["wb1_matched_stage2"]
    assert matched["updates"] == expected["updates"] == 150
    assert matched["valid_episodes"] == expected["total_valid_episodes"] == 19_200
    assert matched["updates"] * expected["tasks_per_update"] * expected["rollouts_per_task"] == matched["valid_episodes"]
