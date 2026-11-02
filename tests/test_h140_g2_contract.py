import json
import tempfile
import unittest
from pathlib import Path

from scripts.openevo_webshop.account_h140_g2_gpu_hours import account
from scripts.openevo_webshop.analyze_h140_g2 import analyze
from scripts.openevo_webshop.plan_h140_g2 import plan
from scripts.openevo_webshop.select_h140_g2_e2 import select
from scripts.openevo_webshop.validate_h140_g2_preregistration import validate
from scripts.openevo_webshop.validate_h140_g2_activation import validate as validate_activation


ROOT = Path(__file__).parents[1]


class H140G2ContractTests(unittest.TestCase):
    def test_materialized_preregistration_is_closed_and_partitioned(self):
        result = validate(ROOT)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["partition_counts"], {"e1": 8, "e2": 64, "anchor": 32, "t2": 64})
        self.assertFalse(result["formal_task_consumption_allowed"])
        self.assertFalse(result["g2_adapter_created"])
        self.assertFalse(result["t2_panel_evaluated"])
        activation = validate_activation(ROOT)
        self.assertTrue(activation["formal_task_consumption_allowed"])
        self.assertEqual(activation["formal_task_consumption"], 0)

    def test_selector_is_outcome_blind_and_freezes_earliest_two_seeds(self):
        manifest = {"rollout_seeds": [1, 2, 3, 4], "tasks": [{"task_id": "webshop_000001"}, {"task_id": "webshop_000002"}]}
        rows = []
        for task_id in ("webshop_000001", "webshop_000002"):
            for seed in (1, 2, 3, 4):
                rows.append({"task_id": task_id, "rollout_seed": seed, "qualified_positive": seed in (2, 4), "reward": 999 if seed == 4 else 0})
        result = select(manifest, rows)
        self.assertEqual(result["status"], "MGH HARVEST_INSUFFICIENT")

        manifest["tasks"].extend({"task_id": f"webshop_{index:06d}"} for index in range(3, 11))
        for task_id in (f"webshop_{index:06d}" for index in range(3, 11)):
            for seed in (1, 2, 3, 4):
                rows.append({"task_id": task_id, "rollout_seed": seed, "qualified_positive": seed in (1, 3)})
        result = select(manifest, rows)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["selected_task_ids"], [f"webshop_{index:06d}" for index in range(1, 9)])
        self.assertTrue(all(record["rollout_seed"] in ((2, 4) if record["task_id"] in ("webshop_000001", "webshop_000002") else (1, 3)) for record in result["selected_records"]))
        self.assertTrue(result["selection_outcome_blind"])

    def test_analyzer_keeps_structural_and_scientific_outcomes_separate(self):
        self.assertTrue(analyze({"status": "bad"})["classification"].startswith("MG0"))
        self.assertTrue(analyze({"harvest": {"status": "pass", "qualified_identity_count": 7}})["classification"].startswith("MGH"))
        base = {"harvest": {"status": "pass", "qualified_identity_count": 8}, "acquisition": {"delta": 0.1}, "t2": {"delta": 0.1, "bootstrap_ci95": [0.01, 0.2]}, "retention": [{"bootstrap_ci95": [0.0, 0.1]}]}
        self.assertTrue(analyze({**base, "acquisition": {"delta": 0}})["classification"].startswith("MG1"))
        self.assertTrue(analyze({**base, "t2": {"delta": 0.1, "bootstrap_ci95": [-0.1, 0.2]}})["classification"].startswith("MG2"))
        self.assertTrue(analyze({**base, "retention": [{"bootstrap_ci95": [-0.06, 0.1]}]})["classification"].startswith("MG3"))
        self.assertTrue(analyze(base)["classification"].startswith("MG4"))

    def test_resource_accounting_does_not_infer_missing_receipts_as_zero(self):
        unknown = account({"qualification_gpu_hours": 0.0})
        self.assertEqual(unknown["status"], "unknown")
        self.assertIsNone(unknown["campaign_total_gpu_hours"])
        self.assertEqual(account({stage: 1.0 for stage in ("qualification_gpu_hours", "harvest_gpu_hours", "engineering_invalid_gpu_hours", "training_gpu_hours", "evaluation_gpu_hours")})["status"], "pass")
        self.assertEqual(account({"qualification_gpu_hours": -1})["status"], "invalid")

    def test_plan_is_cpu_only_and_launcher_refuses_closed_execution(self):
        result = plan(ROOT)
        self.assertEqual(result["status"], "plan-only")
        self.assertFalse(result["gpu_workload_launched"])
        self.assertTrue(all(stage["executes_in_preparation"] is False for stage in result["stages"]))
        launcher = (ROOT / "scripts/openevo_webshop/launch_h140_g2.sh").read_text(encoding="utf-8")
        self.assertIn("router is still closed", launcher)
        self.assertNotIn("docker run", launcher)


if __name__ == "__main__":
    unittest.main()
