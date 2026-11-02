import json
import tempfile
import unittest
from pathlib import Path

from scripts.openevo_webshop.build_recovery_dataset import build_dataset


def _episode(path, *, task_id="webshop_train", seed=1):
    payload = {
        "task_id": task_id,
        "rollout_seed": seed,
        "episode": {
            "episode": 3420,
            "trace": [
                {
                    "step": 0,
                    "prompt": "Current observation: Search\nAllowed actions:\n['search[<your query>]']",
                    "action": "search[shorts]",
                    "action_valid": True,
                    "used_fallback": False,
                    "parser_failure": False,
                }
            ],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _labels(path, *, task_id="webshop_train", seed=1):
    path.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "rollout_seed": seed,
                "source_step": 0,
                "recovery_action": "search[shorts]",
                "teacher_identity": "oracle/test-v1",
            }
        )
        + "\n",
        encoding="utf-8",
    )


class RecoveryDatasetTest(unittest.TestCase):
    def test_recovery_dataset_preserves_state_prompt_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            episode = tmp_path / "episode.json"
            labels = tmp_path / "labels.jsonl"
            _episode(episode)
            _labels(labels)

            result = build_dataset(
                [episode],
                [labels],
                tmp_path / "dataset",
                train_task_ids={"webshop_train"},
                evaluation_task_ids={"webshop_eval"},
            )

            record = json.loads((tmp_path / "dataset" / "records.jsonl").read_text().splitlines()[0])
            self.assertEqual(result["record_count"], 1)
            self.assertTrue(result["records_uri"].endswith("/records.jsonl"))
            self.assertTrue(record["traces"][0]["prompt_messages"][0]["content"].startswith("Current observation"))
            self.assertEqual(record["traces"][0]["response_messages"][0]["content"], "search[shorts]")
            self.assertEqual(record["metadata"]["teacher_identity"], "oracle/test-v1")
            records_bytes = (tmp_path / "dataset" / "records.jsonl").read_bytes()
            canonical = b"".join(
                json.dumps(json.loads(line), sort_keys=True, allow_nan=False).encode() + b"\n"
                for line in records_bytes.splitlines()
            )
            self.assertEqual(records_bytes, canonical)

    def test_recovery_dataset_rejects_evaluation_task_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            episode = tmp_path / "episode.json"
            labels = tmp_path / "labels.jsonl"
            _episode(episode, task_id="webshop_eval")
            _labels(labels, task_id="webshop_eval")

            with self.assertRaisesRegex(ValueError, "evaluation task"):
                build_dataset(
                    [episode],
                    [labels],
                    tmp_path / "dataset",
                    train_task_ids={"webshop_train"},
                    evaluation_task_ids={"webshop_eval"},
                )


if __name__ == "__main__":
    unittest.main()
