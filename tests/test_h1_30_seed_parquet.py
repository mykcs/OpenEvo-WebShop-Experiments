import copy
import unittest

from scripts.openevo_webshop.build_h1_30_seed_parquet import build_rows


class H130SeedParquetTests(unittest.TestCase):
    def _manifest(self):
        return {
            "train_tasks": [
                {
                    "goal_idx": 1663,
                    "sample_index": 0,
                    "split": "h1.30-training",
                    "task_id": "webshop_001663",
                    "task_description": "Find a green jacket",
                    "task_description_sha256": "e5c1c4c64e4d5e5b3c8e9b1f7bc8a4f7c5f9f4c1a6e7d9c0b0a8a0b4c4d0a1e2",
                }
            ],
            "validation_tasks": [],
        }

    def test_rows_bind_prompt_and_environment_identity(self):
        manifest = self._manifest()
        import hashlib

        manifest["train_tasks"][0]["task_description_sha256"] = hashlib.sha256(b"Find a green jacket").hexdigest()
        row = build_rows(manifest, "train")[0]
        self.assertEqual(row["data_source"], "webshop")
        self.assertEqual(row["goal_idx"], 1663)
        self.assertEqual(row["prompt"][0]["content"], "Find a green jacket")
        self.assertEqual(row["extra_info"]["task_id"], "webshop_001663")

    def test_rows_reject_description_digest_drift(self):
        manifest = self._manifest()
        with self.assertRaisesRegex(ValueError, "description digest"):
            build_rows(manifest, "train")

    def test_rows_reject_duplicate_goal(self):
        manifest = self._manifest()
        import hashlib

        digest = hashlib.sha256(b"Find a green jacket").hexdigest()
        manifest["train_tasks"][0]["task_description_sha256"] = digest
        manifest["train_tasks"].append(copy.deepcopy(manifest["train_tasks"][0]))
        manifest["train_tasks"][1]["sample_index"] = 1
        with self.assertRaisesRegex(ValueError, "duplicate goal_idx"):
            build_rows(manifest, "train")


if __name__ == "__main__":
    unittest.main()
