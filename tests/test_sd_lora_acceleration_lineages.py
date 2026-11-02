from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "docs/science/webshop/program/SD_LORA_ACCELERATION_LINEAGES_202609141053.json"
DOC_PATH = ROOT / "docs/science/webshop/SD_LORA_ACCELERATION_LINES.md"
ROUTER_PATH = ROOT / "docs/science/webshop/README.md"


def _load() -> dict:
    return json.loads(MAP_PATH.read_text())


def test_two_acceleration_lines_have_distinct_scientific_identity() -> None:
    data = _load()
    lines = {row["line_id"]: row for row in data["lines"]}
    assert set(lines) == {"stable-reduction", "bounded-online-recurrence"}
    assert lines["bounded-online-recurrence"]["research_branch"] != lines["bounded-online-recurrence"]["preservation_branch"]
    assert lines["stable-reduction"]["sealed_head"] != lines["bounded-online-recurrence"]["sealed_head"]
    assert lines["stable-reduction"]["reader_name"] == "SD-LoRA Stable Reduction"
    assert lines["bounded-online-recurrence"]["reader_name"] == "SD-LoRA Bounded Online Recurrence"
    assert lines["bounded-online-recurrence"]["preservation_branch"] == "research/sd-lora-bounded-online-recurrence-pass-20260914"
    assert lines["stable-reduction"]["history_representation"] != lines["bounded-online-recurrence"]["history_representation"]

def test_speedups_are_bound_to_their_own_line() -> None:
    lines = {row["line_id"]: row for row in _load()["lines"]}
    stable = lines["stable-reduction"]["sealed_results"]
    bounded = lines["bounded-online-recurrence"]["sealed_results"]
    assert stable["matched_eight_step_trainer_seconds"]["speedup"] == 2.01
    assert bounded["same_round_formal_vanilla_speedup"]["mean"] == 37.04
    guards = "\n".join(_load()["conflation_guards"])
    assert "Do not compare 2.01x and 37.04x" in guards
    assert "Neither line is GDR" in guards


def test_router_and_human_doc_make_vr_ambiguity_explicit() -> None:
    router = ROUTER_PATH.read_text()
    doc = DOC_PATH.read_text()
    assert "SD_LORA_ACCELERATION_LINES.md" in router
    assert "VR" in doc
    assert "不能把第二条线的 37× 写成“v2 后来又从 2× 提升到 37×”" in doc  # legacy-revision-compat: exact frozen historical literal
    assert "GDR/event-write 是第三种独立 scientific treatment" in doc
    assert "a4d33bada8e8cc1bc4f4a24f1470650beca7a493" in doc
    assert "不是第二套 current authority" in doc

def test_startup_rule_requires_lineage_resolution_before_claims() -> None:
    agents = (ROOT / "AGENTS.md").read_text()
    assert "Shared shorthand is zero scientific-lineage identity" in agents
    assert "A later commit or larger speedup is not proof of a successor relationship" in agents
    assert "prospectively matched comparison" in agents
    assert "docs/science/webshop/SD_LORA_ACCELERATION_LINES.md" in agents
