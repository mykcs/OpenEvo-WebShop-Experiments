import unittest
import hashlib
import json
from pathlib import Path

from scripts.openevo_webshop.validate_h140_r2_activation import validate


class H140R2ActivationTests(unittest.TestCase):
    def test_activation_and_current_e2_state_bind_original_contract(self):
        result = validate(Path(__file__).parents[1])
        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["formal_task_consumption_allowed"])
        self.assertEqual(result["formal_task_consumption"], 640)
        self.assertEqual(result["formal_gpu_hours_consumed"], 3.7551676636612837)
        self.assertEqual(result["authorized_gpu_count"], 6)

    def test_formal_release_binds_zero_task_qualification_and_frozen_contract(self):
        root = Path(__file__).parents[1]
        release = json.loads((root / "configs/experiment/activations/h1.40-r2-g2-formal-execution-release-v1.json").read_text())
        prereg = root / "configs/experiment/h1.40-r2-g2-multigeneration-v1.json"
        activation = json.loads((root / "configs/experiment/activations/h1.40-r2-g2-execution-activation-v1.json").read_text())
        self.assertEqual(release["status"], "pass")
        self.assertTrue(release["formal_task_consumption_allowed"])
        self.assertFalse(release["scientific_contract_changed"])
        self.assertEqual(release["contract_sha256"], "sha256:" + hashlib.sha256(prereg.read_bytes()).hexdigest())
        self.assertEqual(release["manifest_sha256"], activation["manifest_sha256"])
        self.assertEqual(release["qualification"]["qualification_gpu_hours"], 0.007486578837253748)
        self.assertEqual(release["resource_accounting"]["campaign_total_gpu_hours"], 0.007486578837253748)
        self.assertEqual(release["formal_state_before_release"], {"formal_task_consumption": 0, "formal_gpu_hours_consumed": 0.0, "g2_adapter_created": False, "t2_panel_evaluated": False})


if __name__ == "__main__":
    unittest.main()
