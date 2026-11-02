import json
import tempfile
import unittest
from pathlib import Path

from scripts.openevo_webshop.reconcile_h140_r2_upstream_evaluation import (
    BOOTSTRAP_ROUNDS,
    SEEDS,
    reconcile,
)


class H140R2UpstreamReconciliationTests(unittest.TestCase):
    def test_complete_paired_upstream_matrix_opens_t2_only_as_a_next_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "run"
            cells = []
            identities = {
                "g1": {"adapter_file_sha256": "a" * 64, "tensor_digest": "b" * 64},
                "g2": {"adapter_file_sha256": "c" * 64, "tensor_digest": "d" * 64},
            }
            for stage, count in (("acquisition", 8), ("retention", 8), ("preservation", 32)):
                for arm in ("g1", "g2"):
                    for index in range(count):
                        task_id = f"webshop_{stage[:2]}_{index:03d}"
                        goal_idx = index
                        cell = {"stage": stage, "arm": arm, "task_id": task_id, "goal_idx": goal_idx, "return_code": 0, "resource_accounting": {"status": "exact", "gpu_hours": 0.001}}
                        cells.append(cell)
                        output = run_root / "cells" / stage / arm / task_id
                        output.mkdir(parents=True)
                        rows = []
                        for seed in SEEDS:
                            rows.append({"arm": "sd-lora", "evaluation_stage": "fresh" if stage == "preservation" else stage, "task_id": task_id, "goal_idx": goal_idx, "rollout_seed": seed, "final_task_score": 0.2 if arm == "g2" else 0.1, "qualified_positive": True, "identity_ok": True, "trace_complete": True, "parser_or_fallback_failure": False, "error": None, "adapter_load_receipt": {"verified": True, "adapter_loaded": True, "adapter_file_sha256": identities[arm]["adapter_file_sha256"], "loaded_tensor_digest": identities[arm]["tensor_digest"]}})
                        (output / "attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            scheduler = {"schema": "openevo.webshop-h1.40-r2-upstream-evaluation-scheduler.v1", "status": "pass", "campaign_id": "20260820-0140r2-h140-g2-multigeneration", "experiment_git_sha": "x", "expected_cell_count": 96, "completed_cell_count": 96, "adapter_identities": identities, "cells": cells}
            (run_root / "scheduler-receipt.json").parent.mkdir(parents=True, exist_ok=True)
            (run_root / "scheduler-receipt.json").write_text(json.dumps(scheduler), encoding="utf-8")
            result = reconcile(root, run_root, root / "reconciliation.json")
            self.assertEqual(result["status"], "pass")
            self.assertTrue(result["t2_evaluation_allowed"])
            self.assertEqual(result["formal_task_consumption"], 384)
            self.assertEqual(result["resource_unknown_count"], 0)
            self.assertEqual(result["stages"]["acquisition"]["g2_minus_g1"]["rounds"], BOOTSTRAP_ROUNDS)


if __name__ == "__main__":
    unittest.main()
