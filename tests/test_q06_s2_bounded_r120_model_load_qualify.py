from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.openevo_webshop import q06_s2_bounded_r120_host_orchestrator as host
from scripts.openevo_webshop import q06_s2_bounded_r120_model_load_qualify as qualifier


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Q06S2ModelLoadQualificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        subprocess.run(["git", "init", "-q", str(self.source)], check=True)
        subprocess.run(["git", "-C", str(self.source), "config", "user.email", "test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(self.source), "config", "user.name", "Test"], check=True)
        (self.source / "runtime.py").write_text("frozen runtime fixture\n")
        probe = self.source / host.PROBE_PATH
        probe.parent.mkdir(parents=True)
        probe.write_text(
            "def validate_runtime_files(manifest, runtime_source_root):\n"
            "    assert manifest['runtime_files'] == {}\n"
        )
        subprocess.run(["git", "-C", str(self.source), "add", "runtime.py"], check=True)
        subprocess.run(["git", "-C", str(self.source), "add", host.PROBE_PATH], check=True)
        subprocess.run(["git", "-C", str(self.source), "commit", "-qm", "fixture"], check=True)
        self.source_sha = subprocess.run(
            ["git", "-C", str(self.source), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True).stdout.strip()
        self.repository = "https://example.invalid/q06-runtime.git"
        subprocess.run(["git", "-C", str(self.source), "remote", "add", "origin", self.repository], check=True)
        self.model = self.root / "qwen"
        self.model.mkdir()
        self.adapter = self.root / "adapter" / "adapter_model.safetensors"
        self.adapter.parent.mkdir()
        self.adapter.write_bytes(b"exact-r120-adapter")
        self.output = self.root / "output"
        self.output.mkdir()
        self.manifest = self.root / "manifest.json"
        self.adapter_sha = digest(self.adapter.read_bytes())
        self.manifest.write_text(json.dumps({
            "schema": "openevo.q06.s2-bounded-r120-probe-manifest.v1",
            "cell_id": host.CELL_ID,
            "checkpoint_id": host.CHECKPOINT_ID,
            "checkpoints": {host.CHECKPOINT_ID: {"adapter_sha256": self.adapter_sha}},
            "endpoint": {
                "adapter_path": str(self.adapter.parent),
                "source_files": {"adapter": {
                    "path": str(self.adapter), "bytes": self.adapter.stat().st_size,
                    "sha256": self.adapter_sha,
                }},
            },
            "model": {"local_model_path": str(self.model), "revision": "model-revision-abc"},
            "runtime_files": {},
        }))

    def tearDown(self):
        self.temp.cleanup()

    def identity(self, **overrides):
        args = {
            "manifest_path": self.manifest,
            "checkpoint_id": host.CHECKPOINT_ID,
            "model_path": self.model,
            "model_revision": "model-revision-abc",
            "adapter_path": self.adapter.parent,
            "source": self.source,
            "source_sha": self.source_sha,
            "source_repository": self.repository,
            "image_id": host.IMAGE_ID,
            "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
            "output_dir": self.output,
            "environ": {
                "OPENEVO_RUNTIME_IMAGE_ID": host.IMAGE_ID,
                "CUDA_VISIBLE_DEVICES": "GPU-00000000-0000-0000-0000-000000000000",
            },
        }
        args.update(overrides)
        return qualifier.validate_load_identity(**args)

    def test_receipt_binds_exact_load_identity_and_zero_science_calls(self):
        identity = self.identity()
        receipt = qualifier.make_receipt(
            identity, status="PASS", model_forward_calls=0, generate_calls=0,
            optimizer_updates=0, output_write_gate="PASS")
        self.assertEqual(receipt["identity"]["manifest_sha256"], qualifier.sha256_file(self.manifest))
        self.assertEqual(receipt["identity"]["adapter_sha256"], self.adapter_sha)
        self.assertEqual(receipt["identity"]["source_sha"], self.source_sha)
        self.assertEqual(receipt["identity"]["image_id"], host.IMAGE_ID)
        self.assertTrue(receipt["identity"]["gpu_uuid"].startswith("GPU-"))
        self.assertEqual(receipt["output_write_gate"], "PASS")
        self.assertEqual(receipt["task_bearing_calls"], 0)
        self.assertEqual(receipt["model_forward_calls"], 0)
        self.assertEqual(receipt["generate_calls"], 0)
        self.assertEqual(receipt["optimizer_updates"], 0)
        self.assertFalse(receipt["formal_task_consumed"])

    def test_rejects_gpu_image_model_or_adapter_identity_drift(self):
        with self.assertRaisesRegex(qualifier.QualificationError, "selected GPU UUID"):
            self.identity(environ={"OPENEVO_RUNTIME_IMAGE_ID": host.IMAGE_ID,
                                   "CUDA_VISIBLE_DEVICES": "GPU-" + "0" * 36})
        with self.assertRaisesRegex(qualifier.QualificationError, "runtime image"):
            self.identity(image_id="sha256:" + "0" * 64)
        with self.assertRaisesRegex(qualifier.QualificationError, "model path or revision"):
            self.identity(model_revision="wrong-revision")
        with self.assertRaisesRegex(qualifier.QualificationError, "adapter path/size/hash"):
            self.identity(adapter_path=self.root / "other-adapter")

    def test_rejects_dirty_or_wrong_source_checkout(self):
        (self.source / "untracked.py").write_text("dirty\n")
        with self.assertRaisesRegex(qualifier.QualificationError, "exact clean pinned checkout"):
            self.identity()


if __name__ == "__main__":
    unittest.main()
