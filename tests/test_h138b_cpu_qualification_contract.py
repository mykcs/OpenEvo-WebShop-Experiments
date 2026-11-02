from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]
QUALIFIER = ROOT / "scripts/openevo_webshop/qualify_h138b_cpu_sandbox.py"


class H138BCpuQualificationContractTests(unittest.TestCase):
    def test_cpu_sandbox_is_nonformal_and_uses_only_the_sealed_source(self):
        text = QUALIFIER.read_text(encoding="utf-8")
        self.assertIn("write_increments(d8_manifest, increments_root)", text)
        self.assertIn('"gpu_requested": False', text)
        self.assertIn('"formal_training_or_evaluation": False', text)
        self.assertIn('"record_count": 16', text)
        self.assertIn('"identity_count": 8', text)
        self.assertIn("MAX_TRAINING_EXAMPLES_PER_INCREMENT", text)

    def test_both_real_method_entrypoints_are_exercised_with_save_reload_receipts(self):
        text = QUALIFIER.read_text(encoding="utf-8")
        self.assertIn("ordinary_trainer.train(ordinary_args)", text)
        self.assertIn("run_parametric_smoke.run(sd_args)", text)
        self.assertGreaterEqual(text.count("_loaded_digest("), 3)
        self.assertIn('"ordinary_final_adapter_file_sha256"', text)
        self.assertIn('"sd_final_adapter_file_sha256"', text)


if __name__ == "__main__":
    unittest.main()
