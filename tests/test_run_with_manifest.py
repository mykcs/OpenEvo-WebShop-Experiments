import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_with_manifest.py"


class RunWithManifestTests(unittest.TestCase):
    def test_wrapper_writes_and_validates_phase_c_receipt(self):
        manifest = {
            "run": {"run_id": "cpu-wrapper-test", "timestamp": "2026-08-17T00:00:00Z", "phase": "phase_c", "status": "planned"},
            "server": {"server_identity": None, "control_plane": None, "experiment_container": None, "image_tag": None, "image_digest": None},
            "hardware": {"gpu_indices": [], "gpu_uuids": [], "gpu_model": None},
            "code": {"openevo_experiment_sha": "0" * 40, "upstream_openevo_repo": None, "upstream_openevo_sha": None},
            "model": {"model_id": None, "model_revision": None, "tokenizer_revision": None, "dtype": None, "adapter_identity": None},
            "data": {"benchmark": None, "split": None, "dataset_revision": None, "dataset_hash": None, "index_hash": None, "task_manifest_identity": None},
            "experiment": {"experiment_id": None, "passport_sha256": None, "method": "wrapper-test", "seed": 0, "generation": None, "config_hash": None, "gate_evidence": {}},
            "command": {"argv": ["planned"], "environment": {}},
            "outputs": {"artifacts": [], "metrics": {}},
            "resources": {"peak_gpu_memory_mb": None, "wall_time_s": None},
            "tracking": {"mode": "disabled", "wandb": {"enabled": False, "entity": None, "project": None, "id": None}, "weave": {"enabled": False, "entity": None, "project": None, "id": None}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "run-manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--manifest", str(manifest_path),
                    "--output-root", str(root / "outputs"),
                    "--",
                    sys.executable,
                    "-c",
                    "print('manifest-wrapper-ok')",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(saved["run"]["status"], "pass")
        self.assertEqual(saved["outputs"]["metrics"]["exit_status"], 0)
        self.assertGreaterEqual(len(saved["outputs"]["artifacts"]), 2)
        self.assertTrue(all(item["sha256"].startswith("sha256:") for item in saved["outputs"]["artifacts"]))


if __name__ == "__main__":
    unittest.main()
