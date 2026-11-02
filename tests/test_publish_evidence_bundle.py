import tempfile
import unittest
from pathlib import Path

from scripts.publish_evidence_bundle import build_index


class PublishEvidenceBundleTests(unittest.TestCase):
    def test_indexes_small_reviewable_files_and_lists_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ok.json").write_text('{"status":"pass"}\n', encoding="utf-8")
            (root / "raw.jsonl").write_text('{"private":"trace"}\n', encoding="utf-8")
            (root / "large.txt").write_text("x" * 64, encoding="utf-8")
            (root / "credential.txt").write_text("api_key=super-secret-value\n", encoding="utf-8")

            document = build_index(root, root / "index.json", max_bytes=32)

            self.assertEqual(document["file_count"], 1)
            self.assertEqual(document["files"][0]["path"], "ok.json")
            skipped = {item["path"]: item["reason"] for item in document["skipped"]}
            self.assertEqual(skipped["raw.jsonl"], "unsupported-suffix")
            self.assertEqual(skipped["large.txt"], "larger-than-max-bytes")
            self.assertEqual(skipped["credential.txt"], "likely-credential-bearing")


if __name__ == "__main__":
    unittest.main()
