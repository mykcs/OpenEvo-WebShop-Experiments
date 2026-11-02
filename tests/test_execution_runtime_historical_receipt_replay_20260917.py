from __future__ import annotations

import json
from pathlib import Path

from scripts.openevo_runtime.historical_receipt_replay import replay_historical_round


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _round_fixture(root: Path, *, include_complete: bool = True, bad_after: bool = False) -> Path:
    round_root = root / "round-007"
    before = "state-before-7"
    after = "state-after-8"
    _write(round_root / "state-before.json", {"round_index": 7, "state_id": before})
    _write(round_root / "ROLLOUT_PHASE_COMPLETE.json", {"round_index": 7, "status": "PASS", "state_id_before": before})
    _write(round_root / "evidence/evidence.json", {"round_index": 7, "status": "PASS"})
    for rel in ("text-memory", "sd-lora", "skill-bundle", "agent-system"):
        _write(round_root / rel / "terminal-receipt.json", {"round_index": 7, "status": "NOOP"})
    _write(round_root / "state-after.json", {
        "round_index": 7, "state_id": after,
        "next_round": 99 if bad_after else 8,
        "rollouts_consumed": 1024,
    })
    if include_complete:
        _write(round_root / "ROUND_COMPLETE.json", {
            "round_index": 7, "status": "PASS",
            "state_id_before": before, "state_id_after": after,
        })
    return round_root


def test_historical_complete_round_replays_as_sealed_read_only(tmp_path: Path) -> None:
    root = _round_fixture(tmp_path)
    replay = replay_historical_round(root, 7)
    assert replay["decision_class"] == "SEALED"
    assert replay["reference_verdict"] == "COMPLETE_NO_REPLAY"
    assert replay["mutation"] is False
    assert replay["execution_action"] == "NONE"
    assert replay["scientific_retry_authorized"] is False
    assert len(replay["must_not_replay"]) == 9
    assert len(replay["receipt_witnesses"]) == 9


def test_historical_missing_final_ack_replays_as_exact_resume_only(tmp_path: Path) -> None:
    root = _round_fixture(tmp_path, include_complete=False)
    replay = replay_historical_round(root, 7)
    assert replay["decision_class"] == "EXACT_RESUME"
    assert replay["reference_verdict"] == "SAFE_TO_PREPARE_ACK_OR_COMPOSE_ONLY"
    assert replay["target_phase"] == "round_complete"
    assert "state_after" in replay["must_not_replay"]
    assert replay["scientific_retry_authorized"] is False


def test_historical_integrity_drift_replays_blocked(tmp_path: Path) -> None:
    root = _round_fixture(tmp_path, bad_after=True)
    replay = replay_historical_round(root, 7)
    assert replay["decision_class"] == "BLOCKED_INTEGRITY"
    assert replay["reference_verdict"] == "MANUAL_REVIEW_REQUIRED"
    assert replay["integrity_errors"]
    assert replay["launch_authority"] is False
