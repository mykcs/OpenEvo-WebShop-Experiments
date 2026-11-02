from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]
FREEZER = ROOT / "scripts/openevo_webshop/freeze_h138b_sample_plans.py"


class H138BSamplePlanFreezerTests(unittest.TestCase):
    def test_freezer_uses_pinned_shared_helpers_and_eight_increments(self):
        text = FREEZER.read_text(encoding="utf-8")
        self.assertIn('len(manifests) != 8', text)
        self.assertIn('tokenizer.truncation_side = "left"', text)
        self.assertIn('parser.add_argument("--tokenizer-root"', text)
        self.assertIn('local_files_only=True', text)
        self.assertIn('prior_replay_buffer=prior_replay', text)
        self.assertIn('next_replay_buffer_file_sha256', text)
        self.assertIn('current_examples_sha256', text)
        self.assertIn('bundle_sha256', text)


if __name__ == "__main__":
    unittest.main()
