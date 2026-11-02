from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _normalized(text: str) -> str:
    return " ".join(text.lower().split())


def test_recovery_projection_closeout_is_routed_and_consistent() -> None:
    closeout = _normalized(
        _read("docs/operations/governance/CONVERSATION_LESSONS_CLOSEOUT.md")
    )
    router = _read("docs/troubleshooting/README.md")
    case = _normalized(
        _read(
            "docs/troubleshooting/publication/"
            "CASE-RECOVERY-DISCLOSURE-GOVERNANCE-CLOSEOUT-202609211702.md"
        )
    )
    plan = _normalized(
        _read(
            "docs/operations/publication/"
            "FULL_SERVER_DISASTER_RECOVERY_V1_PLAN_20260919.md"  # legacy-revision-compat
        )
    )

    assert "close current projections after the underlying work closes" in closeout
    assert "CASE-RECOVERY-DISCLOSURE-GOVERNANCE-CLOSEOUT-202609211702.md" in router
    assert "disaster recovery and public disclosure are independent decisions" in case
    assert "status: **final / recovery acceptance pass / merged**" in plan

    evidence_dir = ROOT / "docs/evidence/full-server-disaster-recovery-20260919"
    historical = json.loads((evidence_dir / "CURRENT_RUN_RECOVERY_MATRIX_20260920.json").read_text(encoding="utf-8"))
    assert historical["final_acceptance"] is True
    assert historical["status"] == "PASS_RECOVERY_VERIFIED_ALL_CURRENT_ROOTS"

    live_audit = "docs/evidence/recovery-disclosure-audit-20260926/LIVE_PROVIDER_READBACK_20260926.json"
    for name in ("CURRENT_RUN_RECOVERY_MATRIX.json", "FULL_SERVER_DISASTER_RECOVERY_LEDGER.json"):
        projection = json.loads((evidence_dir / name).read_text(encoding="utf-8"))
        assert projection["status"] == "UNVERIFIED_PARTIAL_CURRENT_SERVER_PAYLOAD"
        assert projection["final_acceptance"] is False
        assert projection["current_disaster_recovery_authority"] == live_audit
        assert projection["historical_20260920_baseline"]["final_acceptance"] is True
