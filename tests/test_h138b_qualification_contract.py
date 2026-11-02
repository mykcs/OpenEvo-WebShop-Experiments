from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]
VERIFY = ROOT / "scripts/openevo_webshop/verify_h138b_qualification.py"


class H138BQualificationContractTests(unittest.TestCase):
    def test_qualification_gates_all_eight_stages_and_formal_fresh_panel(self):
        text = VERIFY.read_text(encoding="utf-8")
        self.assertIn("len(ordinary) != 8 or len(sd) != 8", text)
        self.assertIn('o.get("sample_plan") != training_plan', text)
        self.assertIn('state.get("replay_data_sha256")', text)
        self.assertIn('s.get("dataset_records_sha256")', text)
        self.assertIn('receipt.get("loaded_lora_tensor_count"', text)
        self.assertIn('config.get("gradient_accumulation_steps")', text)
        self.assertIn('ordinary_inventory["parameter_count"] != sd_inventory["parameter_count"]', text)
        self.assertIn('"formal_fresh_task_consumption_allowed": True', text)
        self.assertIn('"stagewise_allocated_capacity_matched": False', text)


if __name__ == "__main__":
    unittest.main()
