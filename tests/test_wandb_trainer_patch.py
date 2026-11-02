import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "scripts" / "openevo_webshop" / "patches" / "openevo-wandb-trainer-logging.patch"
APPLY = ROOT / "scripts" / "openevo_webshop" / "apply_wandb_trainer_patch.sh"
GOVERNANCE_PATCH = ROOT / "scripts" / "openevo_webshop" / "patches" / "openevo-wandb-governance.patch"


class WandbTrainerPatchTest(unittest.TestCase):
    def test_patch_is_parseable_and_has_no_trailing_whitespace(self):
        text = PATCH.read_text(encoding="utf-8")
        self.assertIn("sd_lora_trainer.py", text)
        self.assertIn("trainer_service.py", text)
        for number, line in enumerate(text.splitlines(), start=1):
            # Unified diff content lines carry a mandatory one-character
            # prefix: space for context, '+' for additions, '-' for removals.
            # Strip that syntax prefix before checking the payload itself.
            payload = line
            if line.startswith((" ", "+", "-")) and not line.startswith(("+++", "---")):
                payload = line[1:]
            self.assertEqual(
                payload,
                payload.rstrip(" \t"),
                f"trailing whitespace in patch payload at line {number}",
            )
        completed = subprocess.run(
            ["git", "apply", "--numstat", str(PATCH)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_patch_is_single_stage_and_preflight_is_fail_closed(self):
        self.assertFalse(GOVERNANCE_PATCH.exists(), "fragile second-stage governance patch must stay removed")
        script = APPLY.read_text(encoding="utf-8")
        self.assertIn("git -C \"$open_evo_root\" apply --check --whitespace=error-all", script)
        self.assertIn("OPENEVO_WANDB_PATCH_CHECK_ONLY", script)
        self.assertIn("796e6248a61112a61c880885844d3b0420c0b6ae", script)
        self.assertIn("30d0c3df140b7d1e36a7b550ad7cf4bb77633b08", script)
        self.assertIn("d60b105a9d9e833d6fb6a14d6afa587e024936b4", script)


if __name__ == "__main__":
    unittest.main()
