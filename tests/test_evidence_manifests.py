import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    return json.loads((ROOT / "docs" / "evidence" / name).read_text(encoding="utf-8"))


def test_initial_gate_stayed_closed():
    data = load("PARAMETRIC_RND_MANIFEST_2026-08-13.json")
    assert data["status"] == "EXPERIENCE_BOTTLENECK"
    assert data["qualified_positive"] == 0
    assert data["training_started"] is False


def test_parser_retry_improved_format_not_reward():
    data = load("PARAMETRIC_RND_PARSER_RETRY_V1_MANIFEST_2026-08-13.json")
    assert data["qualified_positive"] == 0
    assert data["complete_zero_reward"] == 29
    assert data["parser_or_fallback_error"] == 1
    assert data["retry_recovered_valid_action"] == 3
    assert data["training_started"] is False
