import unittest
import json
import tempfile
from pathlib import Path

from scripts.openevo_webshop.campaign_router import is_external_successor_router
from scripts.openevo_webshop import execution_readiness
from scripts.openevo_webshop.validate_h138b_preregistration import validate as validate_h138b
from scripts.openevo_webshop.validate_h140_g2_preregistration import validate as validate_h140
from scripts.openevo_webshop.validate_h140_r2_preregistration import validate as validate_h140_r2
from scripts.openevo_webshop.validate_h141_preregistration import validate as validate_h141


ROOT = Path(__file__).parents[1]


class CampaignRouterTests(unittest.TestCase):
    def test_independent_router_does_not_inherit_historical_gpu_fields(self):
        current = {
            "campaign_id": "future-campaign",
            "phase": "future-phase",
            "plan": "docs/future-plan.md",
            "status": "blocked-pre-execution",
            "formal_task_consumption_allowed": False,
        }
        self.assertTrue(is_external_successor_router(current, {"historical-campaign"}))

    def test_all_historical_preregistrations_pass_against_current_wb1_router(self):
        self.assertEqual(validate_h138b(ROOT)["status"], "pass")
        self.assertEqual(validate_h140(ROOT)["status"], "pass")
        self.assertEqual(validate_h140_r2(ROOT)["status"], "pass")
        self.assertEqual(validate_h141(ROOT)["status"], "pass")

    def test_synthetic_future_router_needs_no_historical_validator_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            import shutil

            shutil.copytree(
                ROOT,
                root,
                ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "*.pyc"),
            )
            self.assertFalse((root / ".git").exists())
            current_path = root / "configs/experiment/current-campaign.json"
            current = json.loads(current_path.read_text(encoding="utf-8"))
            current.update(
                {
                    "campaign_id": "20991231-2359-future-independent-campaign",
                    "phase": "FUTURE-TRACK",
                    "status": "blocked-pre-execution",
                    "protocol_status": "future-preregistration-pending",
                    "formal_task_consumption_allowed": False,
                    "plan": "docs/future-campaign-plan.md",
                }
            )
            current_path.write_text(json.dumps(current), encoding="utf-8")
            (root / "docs/future-campaign-plan.md").write_text("future plan\n", encoding="utf-8")

            self.assertEqual(validate_h138b(root)["status"], "pass")
            self.assertEqual(validate_h140(root)["status"], "pass")
            self.assertEqual(validate_h140_r2(root)["status"], "pass")
            self.assertEqual(validate_h141(root)["status"], "pass")

    def test_current_router_cannot_be_used_as_execution_authority(self):
        with self.assertRaises(execution_readiness.ExecutionReadinessError) as excinfo:
            execution_readiness.verify_current_authority(
                {},
                ROOT / "configs/experiment/current-campaign.json",
                current_campaign_path=ROOT / "configs/experiment/current-campaign.json",
                repo_root=ROOT,
            )
        self.assertEqual(excinfo.exception.verdict, execution_readiness.VERDICT_BLOCKED)


if __name__ == "__main__":
    unittest.main()
