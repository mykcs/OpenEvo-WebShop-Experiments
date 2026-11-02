import copy
import unittest

from scripts.openevo_webshop.build_h1_30_manifest import (
    ACTION_CONTRACT,
    ARMS,
    DATA_FILES,
    DATA_SOURCE,
    EVALUATOR,
    HISTORY_LENGTH,
    MAX_NEW_TOKENS,
    MAX_STEPS,
    NUM_PRODUCTS,
    PROMPT_CONTRACT,
    ROLLOUT_SEEDS,
    TEMPERATURE,
    build_manifest,
    validate_manifest,
)


def fake_goals(count=1200):
    return [{"instruction_text": f"goal {index}"} for index in range(count)]


class H130ManifestTests(unittest.TestCase):
    def test_build_freezes_h130_controls_and_is_deterministic(self):
        kwargs = {
            "index_identity": {"file_count": 19, "logical_bytes": 123, "tree_sha256": "a" * 64},
            "prior_indices": {500, 501, 502},
            "experiment_git_sha": "exp-sha",
            "seed_source_commit": "seed-sha",
        }
        first = build_manifest(fake_goals(), **kwargs)
        second = build_manifest(fake_goals(), **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first["arms"], ARMS)
        self.assertEqual(first["controls"]["environment_seed"], 2026)
        self.assertEqual(first["controls"]["max_steps"], MAX_STEPS)
        self.assertEqual(first["controls"]["max_new_tokens"], MAX_NEW_TOKENS)
        self.assertEqual(first["controls"]["temperature"], TEMPERATURE)
        self.assertEqual(first["controls"]["history_length"], HISTORY_LENGTH)
        self.assertEqual(first["controls"]["prompt_contract"], PROMPT_CONTRACT)
        self.assertEqual(first["controls"]["action_contract"], ACTION_CONTRACT)
        self.assertEqual(first["controls"]["rollout_seeds"], ROLLOUT_SEEDS)
        self.assertEqual(first["source"]["catalog"]["source"], DATA_SOURCE)
        self.assertEqual(len(first["source"]["catalog"]["files"]), len(DATA_FILES))

    def test_selection_excludes_prior_indices(self):
        document = build_manifest(
            fake_goals(),
            index_identity={"file_count": 1, "logical_bytes": 1, "tree_sha256": "b" * 64},
            prior_indices=set(range(500, 1000)),
            experiment_git_sha="exp",
            seed_source_commit="seed",
        )
        selected = {task["goal_idx"] for task in document["tasks"]}
        self.assertTrue(selected.isdisjoint(range(500, 1000)))

    def test_validation_rejects_description_drift(self):
        document = build_manifest(
            fake_goals(),
            index_identity={"file_count": 1, "logical_bytes": 1, "tree_sha256": "c" * 64},
            prior_indices=set(),
            experiment_git_sha="exp",
            seed_source_commit="seed",
        )
        tampered = copy.deepcopy(document)
        tampered["tasks"][0]["task_description"] = "changed"
        with self.assertRaisesRegex(ValueError, "description hash"):
            validate_manifest(tampered)

    def test_validation_rejects_wrong_qualification_controls(self):
        document = build_manifest(
            fake_goals(),
            index_identity={"file_count": 1, "logical_bytes": 1, "tree_sha256": "d" * 64},
            prior_indices=set(),
            experiment_git_sha="exp",
            seed_source_commit="seed",
        )
        tampered = copy.deepcopy(document)
        tampered["controls"]["temperature"] = 0.4
        with self.assertRaisesRegex(ValueError, "controls mismatch"):
            validate_manifest(tampered)


if __name__ == "__main__":
    unittest.main()
