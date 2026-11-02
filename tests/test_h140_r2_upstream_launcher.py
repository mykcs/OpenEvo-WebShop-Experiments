import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.openevo_webshop import launch_h140_r2_upstream_evaluation as launcher


class H140R2UpstreamLauncherTests(unittest.TestCase):
    def test_container_path_preserves_run_root_subdirectory(self):
        campaign_root = Path("/workspace/runs/phase-h140-g2-r2-20260820")
        run_root = campaign_root / "formal-upstream-evaluation-v2"
        output = run_root / "cells" / "acquisition" / "g1" / "webshop_005386"
        self.assertEqual(
            launcher.container_path(output, run_root, campaign_root),
            "/workspace/runs/phase-h140-g2-r2-20260820/formal-upstream-evaluation-v2/cells/acquisition/g1/webshop_005386",
        )

    def test_preservation_uses_frozen_runner_fresh_label(self):
        self.assertEqual(launcher.RUNNER_EVALUATION_STAGES["preservation"], "fresh")

    def test_gate_retry_policy_is_bounded(self):
        self.assertGreater(launcher.LIVE_GATE_RETRIES, 1)
        self.assertGreater(launcher.LIVE_GATE_DELAY_SECONDS, 0)

    def test_payload_directory_is_handed_to_isolated_uid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run-root"
            with patch.object(launcher.subprocess, "run") as run:
                launcher.prepare_payload_directory(path)
            run.assert_called_once_with(
                ["chown", "-R", "1001:1001", str(path)],
                check=True,
            )
            self.assertTrue(path.is_dir())
            self.assertEqual(path.stat().st_mode & 0o777, 0o700)


if __name__ == "__main__":
    unittest.main()
