"""CPU-only regression for the immutable R000-R121 diagnostic snapshot."""
import csv
import hashlib
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "analysis/q17_directapply_frontier/20260910T1640Z"

class FrontierPackIntegrity(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((PACK / "q17_frontier_metrics_R000_R121.json").read_text())
        self.entries = [line.split(None, 1) for line in (PACK / "q17_frontier_manifest_R000_R121.sha256").read_text().splitlines() if line.strip()]

    def test_output_manifest(self):
        outputs = [(sha, name[8:]) for sha, name in self.entries if name.startswith("OUTPUT::")]
        self.assertEqual(len(outputs), 3)
        for expected, name in outputs:
            self.assertEqual(Path(name).name, name)
            self.assertEqual(hashlib.sha256((PACK / name).read_bytes()).hexdigest(), expected, name)

    def test_sealed_budget(self):
        self.assertEqual([r["round"] for r in self.payload["per_round"]], list(range(122)))
        self.assertEqual(self.payload["sealed_rounds"], 122)
        self.assertEqual(self.payload["formal_rollouts"], 15616)
        self.assertTrue(all(r["attempt_count"] == 128 for r in self.payload["per_round"]))
        self.assertEqual(self.payload["active_unsealed_rounds"], [122])

    def test_csv_json_agree(self):
        with (PACK / "q17_frontier_metrics_R000_R121.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        expected = [{k: "" if v is None else str(v) for k, v in row.items()} for row in self.payload["per_round"]]
        self.assertEqual(rows, expected)

    def test_input_scope_and_uniqueness(self):
        inputs = [name for _, name in self.entries if not name.startswith("OUTPUT::")]
        self.assertEqual(len(inputs), len(set(inputs)))
        self.assertEqual(len(inputs), 608)
        for name in inputs:
            self.assertRegex(name, r"^round-\d{3}/(ROUND_COMPLETE.json|attempts.jsonl|task-vector-geometry.json|sd-lora/dataset/selection.json|sd-lora/update.json)$")
            self.assertLessEqual(int(name[6:9]), 121)

if __name__ == "__main__":
    unittest.main()
