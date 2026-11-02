import json
import tempfile
import unittest
from pathlib import Path

from scripts.openevo_webshop.reconcile_h132_transfer import reconcile


class H132TransferReconciliationTests(unittest.TestCase):
    def _inputs(self, root: Path):
        tasks = list(range(6603, 6611))
        seeds = [2026091100, 2026091101, 2026091102, 2026091103]
        manifest = {"manifest_sha256": "panel-hash", "tasks": [{"goal_idx": task} for task in tasks]}
        config = {
            "panel": {"source_manifest_sha256": "panel-hash", "task_ids": tasks, "source_manifest": "missing.json"},
            "frozen_contract": {"rollout_seeds": seeds},
        }
        cells = []
        for arm in ("base", "openevo-transfer"):
            for task in tasks:
                cell = root / f"{arm}-{task}"
                (cell / "episodes").mkdir(parents=True)
                for seed in seeds:
                    row = {
                        "arm": arm,
                        "goal_idx": task,
                        "rollout_seed": seed,
                        "final_task_score": 0.0 if arm == "base" else 0.25,
                        "won": False,
                        "identity_ok": True,
                        "trace_complete": True,
                        "parser_or_fallback_failure": False,
                        "error": None,
                    }
                    (cell / "episodes" / f"{task}-{seed}.json").write_text(json.dumps(row), encoding="utf-8")
                cells.append((arm, task, cell))
        return config, manifest, cells

    def test_reconciles_complete_paired_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            config, manifest, cells = self._inputs(Path(directory))
            result = reconcile(config, manifest, cells)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["paired"]["pair_count"], 32)
        self.assertEqual(result["paired"]["valid_pair_count"], 32)
        self.assertAlmostEqual(result["paired"]["mean_adapter_minus_base_task_score"], 0.25)

    def test_blocks_missing_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            config, manifest, cells = self._inputs(Path(directory))
            cells[0][2].joinpath("episodes", "6603-2026091100.json").unlink()
            result = reconcile(config, manifest, cells)
        self.assertEqual(result["status"], "blocked")
        self.assertTrue(any("missing episodes" in item for item in result["failures"]))


if __name__ == "__main__":
    unittest.main()
