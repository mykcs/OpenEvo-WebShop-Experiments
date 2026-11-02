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


class RunManifestValidationTests(unittest.TestCase):
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

    def test_example_is_valid(self) -> None:
        manifest = self.example_manifest()
        result = self.run_validator(manifest)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_credentials_are_rejected(self) -> None:
        manifest = self.example_manifest()
        manifest["command"]["environment"]["WANDB_API_KEY"] = "not-a-real-key"
        result = self.run_validator(manifest)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("credential-like", result.stderr)

    def test_disabled_tracking_is_valid(self) -> None:
        manifest = self.example_manifest()
        manifest["tracking"]["mode"] = "disabled"
        result = self.run_validator(manifest)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_phase_d_pass_receipt_is_valid(self) -> None:
        manifest = self.example_manifest()
        manifest["run"]["status"] = "pass"
        manifest["server"]["image_digest"] = "sha256:" + "1" * 64
        manifest["hardware"]["gpu_uuids"] = ["GPU-5090-example"]
        manifest["model"].update({"model_revision": "example-model-revision", "tokenizer_revision": "example-tokenizer-revision"})
        manifest["experiment"]["gate_evidence"] = {"real_model_execution": True, "real_generation": True}
        manifest["outputs"]["artifacts"] = [{"logical_name": "generation", "sha256": "sha256:" + "2" * 64, "locations": ["results/generation.json"]}]
        manifest["resources"] = {"peak_gpu_memory_mb": 4096, "wall_time_s": 12.5}
        result = self.run_validator(manifest)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_placeholder_git_sha_is_rejected(self) -> None:
        manifest = self.example_manifest()
        manifest["code"]["openevo_experiment_sha"] = "0" * 40
        result = self.run_validator(manifest)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("placeholder", result.stderr)

    def test_unknown_fields_are_rejected(self) -> None:
        manifest = self.example_manifest()
        manifest["outputs"]["untracked_payload"] = "not in schema"
        result = self.run_validator(manifest)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown field", result.stderr)

    def test_json_schema_constraints_are_enforced(self) -> None:
        cases = (
            ("timestamp timezone", ("run", "timestamp"), "2026-08-14T12:00:00"),
            ("server string type", ("server", "server_identity"), 5090),
            ("dataset hash", ("data", "dataset_hash"), "not-a-digest"),
            ("duplicate GPU indices", ("hardware", "gpu_indices"), [0, 0]),
            ("boolean GPU index", ("hardware", "gpu_indices"), [True]),
            ("duplicate GPU UUIDs", ("hardware", "gpu_uuids"), ["GPU-0", "GPU-0"]),
            ("model string type", ("model", "model_id"), 3),
            ("negative generation", ("experiment", "generation"), -1),
            ("invalid config hash", ("experiment", "config_hash"), ""),
            ("empty argv", ("command", "argv"), []),
            ("nested environment value", ("command", "environment"), {"X": []}),
            ("negative wall time", ("resources", "wall_time_s"), -1),
            ("boolean GPU memory", ("resources", "peak_gpu_memory_mb"), True),
            ("tracking string type", ("tracking", "wandb", "entity"), 7),
        )
        for label, path, invalid_value in cases:
            with self.subTest(label=label):
                manifest = self.example_manifest()
                parent = manifest
                for key in path[:-1]:
                    parent = parent[key]
                parent[path[-1]] = invalid_value
                result = self.run_validator(manifest)
                self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_artifact_schema_constraints_are_enforced(self) -> None:
        manifest = self.example_manifest()
        manifest["outputs"]["artifacts"] = [
            {
                "logical_name": "",
                "sha256": "sha256:" + "1" * 64,
                "locations": [""],
            }
        ]
        result = self.run_validator(manifest)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("logical_name", result.stderr)
        self.assertIn("locations", result.stderr)

    def test_phase_g_can_represent_multiple_independent_slots(self) -> None:
        manifest = self.example_manifest()
        manifest["run"].update({"phase": "phase_g", "status": "pass"})
        manifest["server"]["image_digest"] = "sha256:" + "1" * 64
        manifest["hardware"].update({"gpu_indices": [0, 1], "gpu_uuids": ["GPU-0", "GPU-1"]})
        manifest["model"].update({"model_revision": "example-model-revision", "tokenizer_revision": "example-tokenizer-revision"})
        manifest["outputs"]["artifacts"] = [{"logical_name": "campaign-summary", "sha256": "sha256:" + "2" * 64, "locations": ["results/summary.json"]}]
        manifest["resources"] = {"peak_gpu_memory_mb": 4096, "wall_time_s": 12.5}
        result = self.run_validator(manifest)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_phase_f_cannot_pass_with_a_noop_update(self) -> None:
        manifest = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        manifest["run"].update({"phase": "phase_f", "status": "pass"})
        manifest["server"]["image_digest"] = "sha256:" + "1" * 64
        manifest["hardware"]["gpu_uuids"] = ["GPU-5090-example"]
        manifest["experiment"]["gate_evidence"] = {
            "real_model_execution": True,
            "finite_loss": True,
            "finite_nonzero_gradient": True,
            "optimizer_step": True,
            "parameter_changed": True,
            "serialized": True,
            "reload_verified": True,
            "state_before_sha256": "sha256:" + "2" * 64,
            "state_after_sha256": "sha256:" + "2" * 64,
        }
        manifest["model"].update({"model_revision": "example-model-revision", "tokenizer_revision": "example-tokenizer-revision", "adapter_identity": "adapter-example"})
        manifest["code"].update({"upstream_openevo_repo": "https://github.com/mykcs/OpenEvo.git", "upstream_openevo_sha": "3" * 40})
        manifest["outputs"]["artifacts"] = [{"logical_name": "adapter", "sha256": "sha256:" + "4" * 64, "locations": ["results/adapter.safetensors"]}]
        manifest["resources"] = {"peak_gpu_memory_mb": 4096, "wall_time_s": 12.5}
        result = self.run_validator(manifest)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be distinct", result.stderr)

    def test_passport_crosslink_slots_are_required_and_pairwise(self) -> None:
        manifest = self.example_manifest()
        manifest["experiment"].pop("experiment_id")
        result = self.run_validator(manifest)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("experiment_id", result.stderr)

        manifest = self.example_manifest()
        manifest["experiment"]["experiment_id"] = "202609082022-q17-directapply-no-gdr"
        result = self.run_validator(manifest)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("both be null or both be bound", result.stderr)

        manifest["experiment"]["passport_sha256"] = "not-a-sha"
        result = self.run_validator(manifest)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("bare SHA-256", result.stderr)

    def test_passport_crosslink_bound_pair_is_valid(self) -> None:
        manifest = self.example_manifest()
        manifest["experiment"]["experiment_id"] = "202609082022-q17-directapply-no-gdr"
        manifest["experiment"]["passport_sha256"] = "1" * 64
        result = self.run_validator(manifest)
        self.assertEqual(result.returncode, 0, result.stderr)



if __name__ == "__main__":
    unittest.main()
