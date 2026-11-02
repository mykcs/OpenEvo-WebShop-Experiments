from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from scripts.openevo_runtime.retirement_readiness import (
    PROOF_SCHEMA,
    RetirementReadinessError,
    evaluate_retirement_readiness,
)

ROOT = Path(__file__).resolve().parents[1]
RELEASE = json.loads(
    (ROOT / "configs/experiment/runtime-adoption/execution-runtime-next-campaign-adoption-202609171752.json").read_text()
)
CANDIDATES = [
    {
        "path": "scripts/openevo_webshop/run_ceiling1_stage2_ray_supervised_202609042025.py",
        "category": "SUPERVISOR_POLLING_AND_PROCESS_KEEPER",
        "post_gate_action": "RETIRE_CONTROL_PATH",
    },
    {
        "path": "scripts/openevo_webshop/bounded_effective_state_pair_watcher_20260916.py",
        "category": "HISTORICAL_PAIR_POLLING",
        "post_gate_action": "KEEP_HISTORY_ONLY",
        "source_ref": "fbdb21b2739ffeeca90e17eac83c14adb6087be2",
    },
]


class RetirementReadinessTests(unittest.TestCase):
    def test_current_release_blocks_all_phase_h_mutation(self) -> None:
        result = evaluate_retirement_readiness(RELEASE, CANDIDATES)
        self.assertEqual(result["status"], "BLOCKED_PHASE_H_RETIREMENT")
        self.assertEqual(result["blockers"], ["PRODUCTION_CAMPAIGN_PROOF_MISSING"])
        self.assertFalse(result["phase_h_mutation_authorized"])
        self.assertTrue(all(not row["retirement_authorized"] for row in result["candidates"]))
        self.assertTrue(all(not row["history_only_mark_authorized"] for row in result["candidates"]))

    def test_missing_canary_and_consumer_validation_still_block_retirement(self) -> None:
        release = copy.deepcopy(RELEASE)
        release["resource_canary_receipt_identity"] = None
        release["exact_consumer_validation_receipt_identity"] = None
        result = evaluate_retirement_readiness(release, CANDIDATES)
        self.assertIn("ZERO_FORMAL_RESOURCE_CANARY_PENDING", result["blockers"])
        self.assertIn("EXACT_CONSUMER_VALIDATION_PENDING", result["blockers"])
        self.assertIn("PRODUCTION_CAMPAIGN_PROOF_MISSING", result["blockers"])
        self.assertFalse(result["phase_h_mutation_authorized"])


    def test_full_synthetic_gate_allows_retirement_but_keeps_history_reader(self) -> None:
        release = copy.deepcopy(RELEASE)
        release["resource_canary_receipt_identity"] = "1" * 64
        release["exact_consumer_validation_receipt_identity"] = "2" * 64
        proof = {
            "schema": PROOF_SCHEMA,
            "runtime_modernization_proved": True,
            "closure_gates_passed": True,
            "final_panel_boundary_respected": True,
            "proof_mutates_scientific_semantics": False,
            "execution_generation_id": release["execution_generation_id"],
            "resource_canary_receipt_identity": release["resource_canary_receipt_identity"],
            "exact_consumer_validation_receipt_identity": release["exact_consumer_validation_receipt_identity"],
            "formal_rows_consumed": 1,
        }
        result = evaluate_retirement_readiness(release, CANDIDATES, production_proof=proof)
        self.assertEqual(result["status"], "READY_FOR_PHASE_H_RETIREMENT")
        active, historical = result["candidates"]
        self.assertTrue(active["retirement_authorized"])
        self.assertFalse(historical["retirement_authorized"])
        self.assertTrue(historical["history_only_mark_authorized"])

    def test_malformed_proof_fails_closed_once_prerequisites_exist(self) -> None:
        release = copy.deepcopy(RELEASE)
        release["resource_canary_receipt_identity"] = "1" * 64
        release["exact_consumer_validation_receipt_identity"] = "2" * 64
        with self.assertRaisesRegex(RetirementReadinessError, "production proof schema drift"):
            evaluate_retirement_readiness(release, CANDIDATES, production_proof={"schema": "wrong"})


if __name__ == "__main__":
    unittest.main()
