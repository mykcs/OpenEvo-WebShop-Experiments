from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = ROOT / "scripts" / "validate_run_manifest.py"
EXAMPLE = ROOT / "templates" / "run-manifest.example.json"


class RunManifestReleaseGuardTests(unittest.TestCase):
    def run_validator(self, manifest: dict) -> subprocess.CompletedProcess[str]:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8") as handle:
            json.dump(manifest, handle)
            handle.flush()
            return subprocess.run(
                [sys.executable, str(VALIDATOR), handle.name],
                text=True,
                capture_output=True,
                check=False,
            )

    def example_manifest(self) -> dict:
        return json.loads(EXAMPLE.read_text(encoding="utf-8"))

    def phase_f_manifest(self) -> dict:
        manifest = self.example_manifest()
        manifest["run"].update({"phase": "phase_f", "status": "pass"})
        manifest["server"]["image_digest"] = "sha256:" + "1" * 64
        manifest["hardware"]["gpu_uuids"] = ["GPU-5090-example"]
        manifest["model"].update(
            {
                "model_revision": "example-model-revision",
                "tokenizer_revision": "example-tokenizer-revision",
                "adapter_identity": "adapter-example",
            }
        )
        manifest["code"].update(
            {
                "upstream_openevo_repo": "https://github.com/mykcs/OpenEvo.git",
                "upstream_openevo_sha": "3" * 40,
            }
        )
        manifest["outputs"]["artifacts"] = [
            {
                "logical_name": "adapter",
                "sha256": "sha256:" + "4" * 64,
                "locations": ["results/adapter.safetensors"],
            }
        ]
        manifest["resources"] = {"peak_gpu_memory_mb": 4096, "wall_time_s": 12.5}
        manifest["experiment"]["gate_evidence"] = {
            "real_model_execution": True,
            "finite_loss": True,
            "finite_nonzero_gradient": True,
            "optimizer_step": True,
            "parameter_changed": True,
            "serialized": True,
            "reload_verified": True,
            "state_digest_kind": "trainable_tensor_values_v1",
            "state_before_sha256": "sha256:" + "2" * 64,
            "state_after_sha256": "sha256:" + "5" * 64,
            "state_reloaded_sha256": "sha256:" + "5" * 64,
        }
        return manifest

    def test_phase_c_pass_can_be_cpu_only(self) -> None:
        manifest = self.example_manifest()
        manifest["run"].update({"phase": "phase_c", "status": "pass"})
        manifest["server"]["image_digest"] = None
        manifest["hardware"].update({"gpu_indices": [], "gpu_uuids": [], "gpu_model": None})
        manifest["model"].update(
            {"model_id": None, "model_revision": None, "tokenizer_revision": None, "dtype": None}
        )
        manifest["outputs"]["artifacts"] = [
            {
                "logical_name": "phase-c-audit",
                "sha256": "sha256:" + "7" * 64,
                "locations": ["results/phase-c-audit.json"],
            }
        ]
        manifest["resources"] = {"peak_gpu_memory_mb": None, "wall_time_s": 3.0}
        result = self.run_validator(manifest)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_phase_f_valid_roundtrip_is_accepted(self) -> None:
        result = self.run_validator(self.phase_f_manifest())
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_phase_f_rejects_noop_trainable_state(self) -> None:
        manifest = self.phase_f_manifest()
        manifest["experiment"]["gate_evidence"]["state_after_sha256"] = "sha256:" + "2" * 64
        manifest["experiment"]["gate_evidence"]["state_reloaded_sha256"] = "sha256:" + "2" * 64
        result = self.run_validator(manifest)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be distinct", result.stderr)

    def test_phase_f_reload_must_equal_updated_state(self) -> None:
        manifest = self.phase_f_manifest()
        manifest["experiment"]["gate_evidence"]["state_reloaded_sha256"] = "sha256:" + "6" * 64
        result = self.run_validator(manifest)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must equal state_after_sha256", result.stderr)

    def test_phase_f_rejects_archive_hash_as_state_proof(self) -> None:
        manifest = self.phase_f_manifest()
        manifest["experiment"]["gate_evidence"]["state_digest_kind"] = "archive_file_sha256"
        result = self.run_validator(manifest)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("trainable_tensor_values_v1", result.stderr)


if __name__ == "__main__":
    unittest.main()
