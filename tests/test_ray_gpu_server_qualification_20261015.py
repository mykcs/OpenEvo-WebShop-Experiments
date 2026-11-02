"""Read-only/CPU contract checks for the opt-in experiment-host GPU Ray qualification."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from scripts.openevo_runtime import qualify_ray_placement_gpu_server as qualification

SCRIPT = Path(qualification.__file__)


class RayServerQualificationContractTests(unittest.TestCase):
    def test_default_is_plan_only_without_importing_ray(self):
        outcome = subprocess.run(
            [sys.executable, str(SCRIPT)], capture_output=True, text=True, check=True
        )
        doc = json.loads(outcome.stdout)
        self.assertEqual(doc["status"], "PLAN_ONLY")
        self.assertEqual(doc["gpu_indices"], [5, 7])
        self.assertFalse(doc["gpu_handoff_verified"])

    def test_execute_must_supply_explicit_nonsensitive_authorization_ref(self):
        for value in ("", "?", "https://example.test/?token=abc", "short"):
            proc = subprocess.run(
                [sys.executable, str(SCRIPT), "--execute", "--authorization-ref", value],
                capture_output=True, text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("--execute requires a non-secret authorization reference", proc.stderr)

    def test_idle_gate_accepts_only_two_verified_rows(self):
        rows = "GPU | owner | workload\nGPU5 | idle | -\nGPU7 | idle | -\n"
        with patch.object(qualification.subprocess, "check_output", return_value=rows):
            qualification.assert_idle()

    def test_idle_gate_rejects_busy_or_unresolved_resource(self):
        for line in ("GPU7 | another-user | training", "GPU7 | owner-unresolved | ?", ""):
            rows = "GPU5 | idle | -\n" + line + "\n"
            with patch.object(qualification.subprocess, "check_output", return_value=rows):
                with self.assertRaisesRegex(RuntimeError, "GPU7 not independently owner-idle"):
                    qualification.assert_idle()

    def test_script_never_imports_ray_or_touches_gpu_on_module_import(self):
        self.assertEqual(qualification.GPU[5], "GPU-00000000-0000-0000-0000-000000000000")
        self.assertEqual(qualification.GPU[7], "GPU-00000000-0000-0000-0000-000000000000")
        self.assertEqual(qualification.SCHEMA, "openevo.ray.gpu-server-qualification.v1")


if __name__ == "__main__":
    unittest.main()
