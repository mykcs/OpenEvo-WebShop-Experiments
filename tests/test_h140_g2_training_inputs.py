import json
import tempfile
import unittest
from pathlib import Path

from scripts.openevo_webshop.prepare_h140_g2_training_inputs import digest, prepare


class H140G2TrainingInputTests(unittest.TestCase):
    def test_materializes_exactly_eight_two_record_increments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harvest = root / "harvest.jsonl"
            rows = []
            selected = []
            for index in range(8):
                task_id = f"webshop_{index:06d}"
                for seed in (2026104000, 2026104001):
                    row = {
                        "task_id": task_id, "rollout_seed": seed, "qualified_positive": True,
                        "identity_ok": True, "trace_complete": True, "parser_or_fallback_failure": False,
                        "error": None, "final_task_score": 1.0,
                        "episode": {"trace": [{"prompt": "p", "raw_completion": "c", "action_valid": True, "used_fallback": False, "parser_failure": False}]},
                    }
                    rows.append(row)
                    selected.append({"task_id": task_id, "rollout_seed": seed, "record_sha256": digest(row)})
            harvest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
            selection = root / "selection.json"
            selection.write_text(json.dumps({"status": "pass", "record_count": 16, "selected_task_ids": [f"webshop_{i:06d}" for i in range(8)], "selected_records": selected}))
            result = prepare(harvest, selection, root / "inputs", records_uri_resolver=lambda path: f"file:///workspace/{path.name}")
            self.assertEqual(result["increment_count"], 8)
            self.assertEqual(result["record_count"], 16)
            for item in result["increments"]:
                self.assertEqual(item["record_count"], 2)
                manifest_path = Path(item["dataset_manifest"])
                self.assertTrue(manifest_path.is_file())
                self.assertEqual(json.loads(manifest_path.read_text())["records_uri"], "file:///workspace/records.jsonl")
                self.assertFalse(manifest_path.read_bytes().endswith(b"\n"))


if __name__ == "__main__":
    unittest.main()
