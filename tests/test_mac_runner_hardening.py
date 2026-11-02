from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.ci import preflight

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / ".github/runner/openevo-mac"


class MacRunnerHardeningTests(unittest.TestCase):
    def test_current_bundle_satisfies_preflight_contract(self) -> None:
        self.assertEqual(preflight.check_mac_runner_hardening(ROOT), [])

    def test_start_script_keeps_busy_guard_and_isolation(self) -> None:
        text = (RUNNER / "start.sh").read_text(encoding="utf-8")
        for fragment in (
            "require_runner_idle",
            "restore_previous_container",
            "--cap-drop ALL",
            "--security-opt no-new-privileges",
            "--restart no",
            "wait_runner_online",
        ):
            self.assertIn(fragment, text)

    def test_runner_release_is_checksum_verified(self) -> None:
        text = (RUNNER / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("9b1dc70626422526e3c94767cf024896beb15da5342a3f4819bf2feac13e0393", text)
        self.assertIn("sha256sum -c -", text)
        self.assertIn("--retry-all-errors", text)

    def test_workspace_cleanup_refuses_paths_outside_work_root(self) -> None:
        text = (RUNNER / "job-completed.sh").read_text(encoding="utf-8")
        self.assertIn("GITHUB_WORKSPACE is unset; refusing cleanup.", text)
        self.assertIn("Refusing cleanup outside a repository workspace", text)

    def test_power_loss_never_stops_an_unproven_idle_runner(self) -> None:
        text = (RUNNER / "reconcile.sh").read_text(encoding="utf-8")
        self.assertIn("runner_busy_state", text)
        self.assertIn("refusing power-driven stop until idle is proven", text)
        self.assertIn("busy_state", text)



    def test_housekeeping_keeps_one_backup_and_caps_shared_build_cache(self) -> None:
        text = (RUNNER / "housekeeping.sh").read_text(encoding="utf-8")
        self.assertIn('container_backup_prefix="${container}-backup-"', text)
        self.assertIn('bundle_backup_prefix="libexec.backup-"', text)
        self.assertIn('docker container rm -- "$name"', text)
        self.assertIn('docker builder prune --force --max-used-space', text)
        self.assertIn('run_with_timeout', text)
        self.assertIn('CI_HOST_BUILD_CACHE_MAX_USED_SPACE:-6GB', text)
        self.assertIn('basemodel-ci', text)
        self.assertIn('openevo-mac-ci', text)
        self.assertIn('select(.status ==', text)
        self.assertIn('elif any then', text)
        self.assertNotIn('docker system prune', text)

    def test_reconcile_invokes_housekeeping_without_making_cleanup_availability_critical(self) -> None:
        text = (RUNNER / "reconcile.sh").read_text(encoding="utf-8")
        self.assertIn('housekeeping="$bundle_dir/housekeeping.sh"', text)
        self.assertIn('housekeeping failed', text)

    def test_config_hash_is_independent_of_bundle_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hashes = []
            for name in ("one", "two/different/path"):
                bundle = root / name
                bundle.mkdir(parents=True)
                for filename in ("Dockerfile", "config-hash.sh", "entrypoint.sh", "job-completed.sh"):
                    shutil.copy2(RUNNER / filename, bundle / filename)
                result = subprocess.run(
                    [str(bundle / "config-hash.sh")],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                hashes.append(result.stdout.strip())
            self.assertEqual(hashes[0], hashes[1])
            self.assertRegex(hashes[0], r"^[0-9a-f]{64}$")


    def test_reconcile_requires_running_and_healthy(self) -> None:
        text = (RUNNER / "reconcile.sh").read_text(encoding="utf-8")
        self.assertIn('running=false', text)
        self.assertIn("[[ \"$running\" == true && \"$health\" == 'healthy' ]]", text)

    def test_doctor_counts_exact_listener_and_quiets_version_probe(self) -> None:
        text = (RUNNER / "doctor.sh").read_text(encoding="utf-8")
        self.assertIn("pgrep -x Runner.Listener", text)
        self.assertIn("ACTIONS_RUNNER_PRINT_LOG_TO_STDOUT=0", text)
        self.assertIn("Expected exactly one Runner.Listener", text)

    def test_missing_cap_drop_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / ".github/runner/openevo-mac"
            target.parent.mkdir(parents=True)
            shutil.copytree(RUNNER, target)
            start = target / "start.sh"
            start.write_text(start.read_text(encoding="utf-8").replace("    --cap-drop ALL \\\n", ""), encoding="utf-8")
            errors = preflight.check_mac_runner_hardening(root)
            self.assertTrue(any("--cap-drop ALL" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
