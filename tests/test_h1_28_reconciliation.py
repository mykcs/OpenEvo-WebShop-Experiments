import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.openevo_webshop.reconcile_h1_28 import ARMS, reconcile


class H128ReconciliationTest(unittest.TestCase):
    def test_h1_28_reconciliation_requires_two_complete_positive_tasks(self):
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            tasks = list(range(1, 9))
            seeds = [10, 11, 12, 13]
            manifest = tmp_path / "manifest.json"
            manifest.write_text(json.dumps({
                "arms": list(ARMS),
                "tasks": [{"goal_idx": task} for task in tasks],
                "rollout": {"seeds": seeds},
            }))
            cells = []
            for arm in ARMS:
                for task in tasks:
                    root = tmp_path / f"{arm}-{task}"
                    root.mkdir()
                    rows = []
                    for seed in seeds:
                        score = 1.0 if arm == "oracle-action" and task in {1, 2} else 0.0
                        rows.append({
                            "arm": arm,
                            "goal_idx": task,
                            "task_id": f"webshop_{task:06d}",
                            "rollout_seed": seed,
                            "identity_ok": True,
                            "trace_complete": True,
                            "parser_or_fallback_failure": False,
                            "error": None,
                            "qualified_positive": score > 0,
                            "final_task_score": score,
                            "protocol_fingerprint": str(task % 10) * 64,
                        })
                    (root / "attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
                    cells.append(f"{arm}:{task}:{root}")

            result = reconcile(manifest, cells, tmp_path / "result.json")

            self.assertEqual(result["attempt_count"], 96)
            self.assertEqual(result["scientific_valid_count"], 96)
            self.assertEqual(result["positive_task_ids"], [1, 2])
            self.assertEqual(result["classification"], "label-fidelity-confirmation-candidate")
            self.assertEqual(result["paired_delta_sums"], {"oracle_minus_base": 8.0, "oracle_minus_teacher": 8.0})


if __name__ == "__main__":
    unittest.main()
