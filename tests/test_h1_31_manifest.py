import json
import unittest
from pathlib import Path

from scripts.openevo_webshop.build_h1_31_manifest import build_manifest, validate_manifest


ROOT = Path(__file__).resolve().parents[1]
TRAINING = json.loads((ROOT / "docs/evidence/remote-runs/2026-08-17/h1.30/manifests/h1.30-training-manifest-draft.json").read_text())
EVALUATION = json.loads((ROOT / "docs/evidence/remote-runs/2026-08-17/h1.30/manifests/h1.30-task-manifest-v1.json").read_text())


class H131ManifestTests(unittest.TestCase):
    def test_manifest_is_fail_closed_and_disjoint(self):
        document = build_manifest(TRAINING, EVALUATION, experiment_git_sha="test", seed_source_commit="seed")
        self.assertEqual(document["status"], "draft-not-authorized")
        self.assertIs(document["formal_task_consumption_allowed"], False)
        self.assertTrue({row["goal_idx"] for row in document["train_tasks"]}.isdisjoint({row["goal_idx"] for row in document["evaluation_tasks"]}))
        self.assertEqual(document["teacher_contract"]["evaluation_access"], False)
        self.assertEqual(validate_manifest(document), document)

    def test_manifest_rejects_teacher_evaluation_access(self):
        document = build_manifest(TRAINING, EVALUATION, experiment_git_sha="test", seed_source_commit="seed")
        document["teacher_contract"]["evaluation_access"] = True
        with self.assertRaisesRegex(ValueError, "teacher boundary"):
            validate_manifest(document)


if __name__ == "__main__":
    unittest.main()
