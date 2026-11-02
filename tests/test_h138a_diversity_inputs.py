import json
from pathlib import Path
import tempfile
import unittest

from scripts.openevo_webshop import build_h138a_diversity_inputs as h138a


ANALYSIS = Path(__file__).parents[1] / "docs/evidence/remote-runs/2026-08-18/h1.38-reservoir/evidence/h1.38-endogenous-reservoir-analysis.json"


class H138ADiversityInputsTests(unittest.TestCase):
    def test_sealed_reservoir_selection_is_fixed_4_8_16(self):
        analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
        selections = h138a.select_records(analysis)
        expected = {
            "diversity-4": (4, {"webshop_000607", "webshop_002192", "webshop_004533", "webshop_006751"}),
            "diversity-8": (8, {"webshop_000607", "webshop_002192", "webshop_004238", "webshop_004477", "webshop_004533", "webshop_005753", "webshop_005819", "webshop_006751"}),
        }
        for arm, (identity_count, task_ids) in expected.items():
            self.assertEqual(len(selections[arm]), 16)
            self.assertEqual({row["task_id"] for row in selections[arm]}, task_ids)
            self.assertEqual(len({row["task_id"] for row in selections[arm]}), identity_count)
        self.assertEqual(len(selections["diversity-16"]), 16)
        self.assertEqual(len({row["task_id"] for row in selections["diversity-16"]}), 16)

    def test_selection_rejects_insufficient_success_supply(self):
        with self.assertRaisesRegex(ValueError, "lacks"):
            h138a.select_records({"identity_ledger": {"webshop_000500": {"qualified_successes": 1}}})

    def test_raw_loader_scans_one_task_file_once_for_multiple_seeds(self):
        rows = []
        selected = []
        for seed in (1, 2):
            row = {"task_id": "webshop_000607", "goal_idx": 607, "rollout_seed": seed, "identity_ok": True, "trace_complete": True, "parser_or_fallback_failure": False, "error": None, "final_task_score": 1.0}
            rows.append(row)
            selected.append({"task_id": row["task_id"], "goal_idx": 607, "rollout_seed": seed, "record_sha256": h138a._record_hash(row)})
        with tempfile.TemporaryDirectory() as directory:
            attempts = Path(directory) / "harvest/task607/attempts.jsonl"
            attempts.parent.mkdir(parents=True)
            attempts.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            observed = h138a._load_attempt_rows(Path(directory), selected)
        self.assertEqual([row["rollout_seed"] for row in observed], [1, 2])


if __name__ == "__main__":
    unittest.main()
