"""Offline science-policy guard for the normal inactive-run audit entry."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOP = ROOT / "docs/infrastructure/server/SERVER_STORAGE_PRESSURE_AND_RECLAIM_SOP.md"
DEV = ROOT / "docs/dev/README.md"


def test_inactive_audit_is_read_only_and_still_protects_scientific_state():
    text = SOP.read_text(encoding="utf-8")
    for marker in (
        "server-artifact-governance/SKILL.md",
        "references/inactive-experiment-reclaim.md",
        "server-artifact-governance/README.md",
        "execution-state",
        "P1",
        "Hugging Face",
        "轨迹",
        "可申请删除／保留／待核实",
        "不删除/移动/隔离",
        "不启动 GPU",
    ):
        assert marker in text, marker


def test_dev_uses_one_shared_prompt_and_parent_host_truth():
    text = DEV.read_text(encoding="utf-8")
    assert "server-artifact-governance/README.md" in text
    assert "mykcs/zju-server" in text
    assert "mykcs/.agents/docs/agents/DEV_PROTOCOL.md" in text
