import json
import tempfile
import unittest
from pathlib import Path

from scripts.openevo_webshop.export_h1_37_compact_analysis_pack import compact_row, normalize_action


class CompactAnalysisPackTests(unittest.TestCase):
    def test_compact_row_uses_receipt_fields_and_only_action_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "evaluation" / "base-task1" / "episodes" / "episode.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "arm": "base",
                        "attempt_wall_time_seconds": 2.5,
                        "error": None,
                        "final_task_score": 1.0,
                        "identity_ok": True,
                        "parser_or_fallback_failure": False,
                        "protocol_fingerprint": "sha256:protocol",
                        "rollout_seed": 7,
                        "task_id": "webshop_000001",
                        "trace_complete": True,
                        "episode": {
                            "steps": 3,
                            "won": True,
                            "trace": [
                                {"action": "search[private product text]", "parser_failure": False},
                                {"action": "click[next page]", "parser_failure": False},
                                {"action": "click[buy now]", "parser_failure": False},
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            row = compact_row(path, root)

            self.assertEqual(row["seed"], 7)
            self.assertEqual(row["score"], 1.0)
            self.assertTrue(row["valid"])
            self.assertEqual(row["normalized_action_sequence"], "search>next_page>buy_now")
            self.assertNotIn("private product text", row["normalized_action_sequence"])
            self.assertEqual(row["terminal_state"], "won")

    def test_normalize_action_drops_action_arguments(self):
        self.assertEqual(normalize_action("search[never publish this]"), "search")
        self.assertEqual(normalize_action("click[buy now]"), "buy_now")


if __name__ == "__main__":
    unittest.main()
