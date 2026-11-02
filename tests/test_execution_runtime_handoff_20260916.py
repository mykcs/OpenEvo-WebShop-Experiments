from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.openevo_runtime import runtime_handoff as handoff


class RuntimeHandoffTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_uid1001_rw_then_ro_and_host_identity(self) -> None:
        calls: list[list[str]] = []

        def fake_run(cmd, text=True, capture_output=True):
            calls.append([str(value) for value in cmd])
            mount = next(arg for arg in cmd if ":/handoff:" in str(arg))
            host = Path(str(mount).split(":/handoff:", 1)[0])
            if mount.endswith(":rw"):
                (host / "payload.bin").write_bytes(b"openevo-formal-handoff-smoke\n")
                return SimpleNamespace(
                    returncode=0,
                    stdout="FORMAL_HANDOFF_WRITE_PASS\n",
                    stderr="",
                )
            return SimpleNamespace(
                returncode=0,
                stdout="FORMAL_HANDOFF_READ_PASS\n",
                stderr="",
            )

        identity = {
            "uid": 1001,
            "gid": 1001,
            "mode": 0o600,
            "is_regular_file": True,
            "sha256": hashlib.sha256(b"openevo-formal-handoff-smoke\n").hexdigest(),
        }
        with patch.object(handoff.subprocess, "run", side_effect=fake_run), patch.object(
            handoff, "_handoff_file_identity", return_value=identity
        ):
            result = handoff.production_handoff_smoke(self.root, "science:image")

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["host_readback"]["uid"], 1001)
        self.assertEqual(len(calls), 2)
        self.assertIn("1001:1001", calls[0])
        self.assertTrue(any(str(value).endswith(":/handoff:rw") for value in calls[0]))
        self.assertTrue(any(str(value).endswith(":/handoff:ro") for value in calls[1]))
        self.assertFalse(any("--gpus" in str(value) for call in calls for value in call))

    def test_root_owned_payload_fails_closed(self) -> None:
        def fake_run(cmd, text=True, capture_output=True):
            mount = next(arg for arg in cmd if ":/handoff:" in str(arg))
            host = Path(str(mount).split(":/handoff:", 1)[0])
            if mount.endswith(":rw"):
                (host / "payload.bin").write_bytes(b"openevo-formal-handoff-smoke\n")
            return SimpleNamespace(
                returncode=0,
                stdout="FORMAL_HANDOFF_WRITE_PASS\n",
                stderr="",
            )

        identity = {
            "uid": 0,
            "gid": 0,
            "mode": 0o600,
            "is_regular_file": True,
            "sha256": hashlib.sha256(b"openevo-formal-handoff-smoke\n").hexdigest(),
        }
        with patch.object(handoff.subprocess, "run", side_effect=fake_run), patch.object(
            handoff, "_handoff_file_identity", return_value=identity
        ):
            with self.assertRaisesRegex(RuntimeError, "ownership mismatch"):
                handoff.production_handoff_smoke(self.root, "science:image")

    def test_missing_root_fails_before_subprocess(self) -> None:
        missing = self.root / "missing"
        with patch.object(handoff.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "existing regular directory"):
                handoff.production_handoff_smoke(missing, "science:image")
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
