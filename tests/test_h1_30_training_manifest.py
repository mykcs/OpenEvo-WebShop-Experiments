import copy
import json
import unittest
from pathlib import Path

from scripts.openevo_webshop.build_h1_30_manifest import build_manifest as build_evaluation_manifest
from scripts.openevo_webshop.build_h1_30_training_manifest import (
    DEV_COUNT,
    TRAIN_COUNT,
    build_manifest,
    validate_manifest,
)
from scripts.openevo_webshop.h1_closed_loop_tracked import _selected_tasks


def fake_goals(count=1400):
    return [{"instruction_text": f"goal {index}"} for index in range(count)]


class H130TrainingManifestTests(unittest.TestCase):
    def _build(self, prior=None):
        evaluation = build_evaluation_manifest(
            fake_goals(),
            index_identity={"file_count": 19, "logical_bytes": 123, "tree_sha256": "a" * 64},
            prior_indices=set(prior or ()),
            experiment_git_sha="exp",
            seed_source_commit="seed",
        )
        return build_manifest(
            fake_goals(),
            index_identity={"file_count": 19, "logical_bytes": 123, "tree_sha256": "a" * 64},
            prior_indices=set(prior or ()),
            evaluation_manifest=evaluation,
            experiment_git_sha="exp",
            seed_source_commit="seed",
        ), evaluation

    def test_build_is_deterministic_and_fail_closed(self):
        first, evaluation = self._build(prior={500, 501, 502})
        second, _ = self._build(prior={500, 501, 502})
        self.assertEqual(first, second)
        self.assertEqual(len(first["train_tasks"]), TRAIN_COUNT)
        self.assertEqual(len(first["validation_tasks"]), DEV_COUNT)
        self.assertFalse(first["formal_task_consumption_allowed"])
        self.assertEqual(first["status"], "draft-not-authorized")
        self.assertEqual(
            first["controls"]["prompt_contract"],
            "seed3090.webshop-compact-action-prompt.v2",
        )
        eval_indices = {row["goal_idx"] for row in evaluation["tasks"]}
        selected = {row["goal_idx"] for row in first["train_tasks"] + first["validation_tasks"]}
        self.assertTrue(selected.isdisjoint(eval_indices))
        validate_manifest(first)

    def test_validation_rejects_status_promotion(self):
        document, _ = self._build()
        tampered = copy.deepcopy(document)
        tampered["status"] = "launch-authorized"
        with self.assertRaisesRegex(ValueError, "draft-not-authorized"):
            validate_manifest(tampered)

    def test_validation_rejects_task_overlap(self):
        document, _ = self._build()
        tampered = copy.deepcopy(document)
        tampered["validation_tasks"][0] = copy.deepcopy(tampered["train_tasks"][0])
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_manifest(tampered)

    def test_closed_loop_runner_accepts_h130_train_tasks(self):
        document, _ = self._build()
        manifest = {
            "schema": "openevo.webshop-h1.30-training-manifest.v1",
            "train_tasks": document["train_tasks"],
        }
        with self.subTest("manifest-shape"):
            from tempfile import TemporaryDirectory
            from pathlib import Path

            with TemporaryDirectory() as directory:
                path = Path(directory) / "manifest.json"
                path.write_text(json.dumps(manifest), encoding="utf-8")
                selected = _selected_tasks(path, [row["goal_idx"] for row in document["train_tasks"][:2]])
        self.assertEqual([row["goal_idx"] for row in selected], [row["goal_idx"] for row in document["train_tasks"][:2]])

    def test_capability_launcher_keeps_wave_cell_roots_disjoint(self):
        launcher = Path(__file__).parents[1] / "scripts/openevo_webshop/launch_h130_open_evo_capability.sh"
        source = launcher.read_text(encoding="utf-8")
        self.assertIn('"${TRAIN_TASKS[@]:0:8}"', source)
        self.assertIn('"${TRAIN_TASKS[@]:8}"', source)
        self.assertIn("OPENEVO_H130_CAMPAIGN_ID", source)
        self.assertIn("OPENEVO_H130_TEMPERATURE", source)
        self.assertNotIn('for task in "${TRAIN_TASKS[@]}"; do train_args+=(--train-task-id', source)

    def test_comparison_config_points_to_fresh_v3_and_stays_fail_closed(self):
        path = Path(__file__).parents[1] / "configs/experiment/h1.30-seed-comparison.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        self.assertFalse(document["formal_task_consumption_allowed"])
        self.assertEqual(document["fresh_v3_panel"]["status"], "draft-not-authorized")
        self.assertEqual(document["fresh_v3_panel"]["evaluation_tasks"], 8)
        self.assertEqual(document["fresh_v3_panel"]["training_tasks"], 16)
        self.assertEqual(document["fresh_v3_panel"]["validation_tasks"], 8)
        self.assertTrue(document["fresh_v3_panel"]["pairwise_disjoint"])
        gate = document["open_evo_capability_gate"]
        self.assertEqual(gate["status"], "failed-v3")
        self.assertEqual(gate["qualified_positive_task_count"], 1)
        self.assertEqual(gate["qualified_positive_episode_count"], 4)
        self.assertFalse(gate["formal_task_consumption_allowed"])


if __name__ == "__main__":
    unittest.main()
