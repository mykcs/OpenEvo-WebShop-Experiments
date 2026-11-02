from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import tempfile
import unittest

from scripts.openevo_runtime import round_transaction as tx


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_complete_prefix(root: Path, through: str, round_index: int = 1) -> None:
    state_before = "state-before"
    state_after = "state-after"
    docs = {
        "state_before": ("state-before.json", {"schema": "state", "status": "RUNNING", "state_id": state_before}),
        "rollout": ("ROLLOUT_PHASE_COMPLETE.json", {"schema": "rollout", "status": "PASS", "round_index": round_index, "state_id_before": state_before}),
        "evidence": ("evidence/evidence.json", {"schema": "evidence", "status": "PASS", "round_index": round_index}),
        "text_memory": ("text-memory/terminal-receipt.json", {"schema": "terminal", "status": "NOOP", "kind": "text_memory", "round_index": round_index}),
        "sd_lora": ("sd-lora/terminal-receipt.json", {"schema": "terminal", "status": "UPDATE", "round_index": round_index}),
        "skill_bundle": ("skill-bundle/terminal-receipt.json", {"schema": "terminal", "status": "NOOP", "kind": "skill_bundle", "round_index": round_index}),
        "agent_system": ("agent-system/terminal-receipt.json", {"schema": "terminal", "status": "NOOP", "kind": "agent_system", "round_index": round_index}),
        "state_after": ("state-after.json", {"schema": "state", "status": "RUNNING", "state_id": state_after, "next_round": round_index + 1, "rollouts_consumed": (round_index + 1) * 128}),
        "round_complete": ("ROUND_COMPLETE.json", {"schema": "complete", "status": "PASS", "round_index": round_index, "state_id_before": state_before, "state_id_after": state_after}),
    }
    for spec in tx.PHASES:
        rel, doc = docs[spec.name]
        write_json(root / rel, doc)
        if spec.name == through:
            break



class FormalRoundTransactionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_complete_round_forbids_all_replay(self) -> None:
        tmp_path = self.root
        round_root = tmp_path / "round-001"
        write_complete_prefix(round_root, "round_complete")
        plan = tx.build_round_recovery_plan(round_root, 1)
        assert plan["verdict"] == "COMPLETE_NO_REPLAY"
        assert plan["target_phase"] is None
        assert plan["must_not_replay"] == [spec.name for spec in tx.PHASES]
        assert plan["execution_action"] == "NONE"
        assert plan["scientific_retry_authorized"] is False


    def test_partial_round_preserves_completed_prefix(self) -> None:
        tmp_path = self.root
        round_root = tmp_path / "round-001"
        write_complete_prefix(round_root, "rollout")
        plan = tx.build_round_recovery_plan(round_root, 1)
        assert plan["verdict"] == "SAFE_TO_PREPARE_NEXT_PHASE"
        assert plan["target_phase"] == "evidence"
        assert plan["must_not_replay"] == ["state_before", "rollout"]


    def test_successful_container_without_ack_is_never_replayed_blindly(self) -> None:
        tmp_path = self.root
        round_root = tmp_path / "round-001"
        write_complete_prefix(round_root, "evidence")
        write_json(round_root / "container-attempts" / "reflector-text_memory" / "a000.json", {
            "schema": "attempt", "status": "PASS", "container_exit_code": 0,
        })
        plan = tx.build_round_recovery_plan(round_root, 1)
        assert plan["target_phase"] == "text_memory"
        assert plan["recovery_class"] == "SIDE_EFFECT_STATUS_UNKNOWN"
        assert plan["verdict"] == "MANUAL_REVIEW_REQUIRED"
        assert "text_memory" not in plan["must_not_replay"]


    def test_failed_attempt_requires_explicit_no_side_effect_proof(self) -> None:
        tmp_path = self.root
        round_root = tmp_path / "round-001"
        write_complete_prefix(round_root, "evidence")
        write_json(round_root / "container-attempts" / "reflector-text_memory" / "a000.json", {
            "schema": "attempt", "status": "ENGINEERING_FAILURE", "container_exit_code": 1,
        })
        plan = tx.build_round_recovery_plan(round_root, 1)
        assert plan["verdict"] == "MANUAL_REVIEW_REQUIRED"
        assert plan["phases"][3]["status"] == tx.FAILED_SIDE_EFFECT_UNKNOWN


    def test_explicit_no_side_effect_failure_is_retry_candidate_not_authority(self) -> None:
        tmp_path = self.root
        round_root = tmp_path / "round-001"
        write_complete_prefix(round_root, "evidence")
        write_json(round_root / "container-attempts" / "reflector-text_memory" / "a000.json", {
            "schema": "attempt", "status": "ENGINEERING_FAILURE", "container_exit_code": 1,
            "side_effects_started": False,
        })
        plan = tx.build_round_recovery_plan(round_root, 1)
        assert plan["verdict"] == "SAFE_TO_PREPARE_SAME_INPUT_RETRY"
        assert plan["recovery_class"] == "ENGINEERING_RETRYABLE"
        assert plan["authority_required_before_execution"] is True
        assert plan["scientific_retry_authorized"] is False


    def test_closeout_gap_never_replays_scientific_phases(self) -> None:
        tmp_path = self.root
        round_root = tmp_path / "round-001"
        write_complete_prefix(round_root, "agent_system")
        plan = tx.build_round_recovery_plan(round_root, 1)
        assert plan["verdict"] == "SAFE_TO_PREPARE_ACK_OR_COMPOSE_ONLY"
        assert plan["target_phase"] == "state_after"
        assert set(plan["must_not_replay"]) >= {"rollout", "text_memory", "sd_lora", "skill_bundle", "agent_system"}


    def test_state_chain_mismatch_fails_closed(self) -> None:
        tmp_path = self.root
        round_root = tmp_path / "round-001"
        write_complete_prefix(round_root, "round_complete")
        complete = json.loads((round_root / "ROUND_COMPLETE.json").read_text())
        complete["state_id_after"] = "wrong"
        write_json(round_root / "ROUND_COMPLETE.json", complete)
        plan = tx.build_round_recovery_plan(round_root, 1)
        assert plan["verdict"] == "MANUAL_REVIEW_REQUIRED"
        assert plan["recovery_class"] == "INTEGRITY_UNKNOWN"
        assert any("state_id_after" in err for err in plan["integrity_errors"])


    def test_completion_gap_fails_closed(self) -> None:
        tmp_path = self.root
        round_root = tmp_path / "round-001"
        write_complete_prefix(round_root, "rollout")
        write_json(round_root / "skill-bundle" / "terminal-receipt.json", {
            "schema": "terminal", "status": "NOOP", "kind": "skill_bundle", "round_index": 1,
        })
        plan = tx.build_round_recovery_plan(round_root, 1)
        assert plan["verdict"] == "MANUAL_REVIEW_REQUIRED"
        assert any("phase completion gap" in err for err in plan["integrity_errors"])


    def test_symlink_receipt_is_integrity_unknown(self) -> None:
        tmp_path = self.root
        round_root = tmp_path / "round-001"
        round_root.mkdir(parents=True)
        real = tmp_path / "outside.json"
        write_json(real, {"status": "RUNNING", "state_id": "x"})
        (round_root / "state-before.json").symlink_to(real)
        plan = tx.build_round_recovery_plan(round_root, 1)
        assert plan["verdict"] == "MANUAL_REVIEW_REQUIRED"
        assert plan["phases"][0]["status"] == tx.INTEGRITY_UNKNOWN


    def test_missing_round_root_is_error(self) -> None:
        tmp_path = self.root
        with self.assertRaisesRegex(tx.RoundPlanError, "not a directory"):
            tx.build_round_recovery_plan(tmp_path / "missing", 0)

    def test_process_death_after_launch_intent_is_not_retry_authority(self) -> None:
        round_root = self.root / "round-001"
        write_complete_prefix(round_root, "evidence")
        intent = round_root / "container-attempts" / "reflector-text_memory" / "a000-intent.json"
        write_json(intent, {"schema": "intent", "status": "LAUNCHED"})
        plan = tx.build_round_recovery_plan(round_root, 1)
        assert plan["target_phase"] == "text_memory"
        assert plan["verdict"] == "MANUAL_REVIEW_REQUIRED"
        assert plan["scientific_retry_authorized"] is False

    def test_process_death_after_materialized_effect_requires_readback_not_replay(self) -> None:
        round_root = self.root / "round-001"
        write_complete_prefix(round_root, "evidence")
        write_json(round_root / "container-attempts" / "reflector-text_memory" / "a000.json", {
            "schema": "attempt", "status": "PASS", "container_exit_code": 137,
            "side_effects_started": True,
        })
        plan = tx.build_round_recovery_plan(round_root, 1)
        assert plan["recovery_class"] == "SIDE_EFFECT_STATUS_UNKNOWN"
        assert plan["verdict"] == "MANUAL_REVIEW_REQUIRED"
        assert "text_memory" not in plan["must_not_replay"]
        assert plan["scientific_retry_authorized"] is False


if __name__ == "__main__":
    unittest.main()
