import json
import unittest
from pathlib import Path

from scripts.openevo_webshop.seed_runtime.prepare_h1_30_source_overlay import load_manifest


MANIFEST = Path("docs/evidence/remote-runs/2026-08-17/h1.30/manifests/h1.30-training-manifest-draft.json")


class H130SeedOverlayTests(unittest.TestCase):
    def test_training_manifest_is_accepted_only_as_draft(self):
        document = json.loads(MANIFEST.read_text())
        loaded = load_manifest(MANIFEST, document["manifest_sha256"])
        self.assertIs(loaded["formal_task_consumption_allowed"], False)
        self.assertEqual(len(loaded["train_tasks"]), 16)
        self.assertEqual(len(loaded["validation_tasks"]), 8)

    def test_overlay_loader_rejects_wrong_manifest_digest(self):
        document = json.loads(MANIFEST.read_text())
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            load_manifest(MANIFEST, "0" * 64)


if __name__ == "__main__":
    unittest.main()
