import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.openevo_webshop.summarize_h130_failure_taxonomy import classify, summarize


def _row(action_trace, **fields):
    return {
        "task_id": "webshop_000001",
        "episode": {"trace": [{"action": action} for action in action_trace]},
        **fields,
    }


class H130FailureTaxonomyTests(unittest.TestCase):
    def test_priority_rules_keep_invalid_separate(self):
        self.assertEqual(classify(_row(["click[next >]"] * 6)), "navigation-next-loop")
        self.assertEqual(classify(_row(["click[x]", "click[x]"])), "repeated-action-loop")
        self.assertEqual(classify(_row(["click[back to search]"])), "backtrack-without-success")
        self.assertEqual(classify(_row(["click[x]"], parser_or_fallback_failure=True)), "parser-or-fallback")
        self.assertEqual(classify(_row(["click[x]"], error="boom")), "runtime-error")
        self.assertEqual(classify(_row(["click[buy now]"] , qualified_positive=True)), "qualified-positive")

    def test_summary_is_diagnostic_and_hashes_sources(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "attempts.jsonl"
            source.write_text("{\"task_id\": \"webshop_000001\", \"episode\": {\"trace\": [{\"action\": \"click[next >]\"}]}}\n", encoding="utf-8")
            result = summarize([source], root / "taxonomy.json")
            self.assertTrue(result["diagnostic_only"])
            self.assertEqual(result["attempt_count"], 1)
            self.assertEqual(result["categories"], {"zero-reward-other": 1})
            self.assertEqual(len(result["source_files"]), 1)


if __name__ == "__main__":
    unittest.main()
