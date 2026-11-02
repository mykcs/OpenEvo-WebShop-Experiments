import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.openevo_webshop import launch_h140_r2_e2_harvest as launcher
from scripts.openevo_webshop.launch_h140_r2_e2_harvest import build_attempts, docker_command, validate_release


class H140R2E2LauncherTests(unittest.TestCase):
    def test_release_and_frozen_attempt_matrix(self):
        root = Path(__file__).parents[1]
        original_load_json = launcher.load_json

        def load_historical_router(path: Path):
            if path == root / "configs/experiment/current-campaign.json":
                return {
                    "campaign_id": "20260821-0141-h141-magnitude-screen",
                    "replacement_of": "20260820-0140r2-h140-g2-multigeneration",
                    "status": "closed-measurement-invalid",
                    "latest_event": {"classification": "h1.41-v2-reconciled-measurement-invalid"},
                }
            return original_load_json(path)

        with patch.object(launcher, "load_json", side_effect=load_historical_router):
            config, release, paths = validate_release(root, allow_completed_router=True)
        attempts = build_attempts(root)
        self.assertEqual(config["campaign_id"], "20260820-0140r2-h140-g2-multigeneration")
        self.assertEqual(release["qualification"]["qualification_gpu_hours"], 0.007486578837253748)
        self.assertEqual(len(paths), 8)
        self.assertEqual(len(attempts), 256)
        self.assertEqual(attempts[0]["task_position"], 0)
        self.assertEqual(attempts[0]["seed_position"], 0)
        self.assertEqual(attempts[-1]["task_position"], 63)
        self.assertEqual(attempts[-1]["seed_position"], 3)
        self.assertEqual(len({(row["task_id"], row["rollout_seed"]) for row in attempts}), 256)

    def test_current_wb1_router_cannot_authorize_historical_release(self):
        root = Path(__file__).parents[1]
        with self.assertRaisesRegex(ValueError, "current router"):
            validate_release(root, allow_completed_router=True)

    def test_payload_output_is_bound_to_requested_run_root(self):
        root = Path(__file__).parents[1]
        base = "/workspace/runs/phase-h140-g2-r2-20260820"
        run_root = Path(base) / "formal-e2-harvest-retry1"
        attempt = {"attempt_index": 0, "goal_idx": 5127, "rollout_seed": 2026104000, "wandb_mode": "online"}
        command = docker_command(
            {"run_root": base, "experiment": "/repo", "openevo": "/core", "model": "/model", "dataset": "/data", "cache": "/cache", "adapter": "/adapter"},
            attempt,
            "GPU-test",
            run_root / "attempts" / "e2-000-webshop_005127-s2026104000",
            run_root,
            root,
            "sha",
        )
        output = command[command.index("--output") + 1]
        self.assertEqual(output, "/workspace/runs/phase-h140-g2-r2-20260820/formal-e2-harvest-retry1/attempts/e2-000-webshop_005127-s2026104000")


if __name__ == "__main__":
    unittest.main()
