from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.openevo_webshop import build_h139_mr_manifest as builder


class H139MrContractTests(unittest.TestCase):
    def test_exclusion_ledger_collects_only_explicit_task_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "configs/experiment").mkdir(parents=True)
            (root / "docs/evidence").mkdir(parents=True)
            (root / "configs/experiment/old.json").write_text(json.dumps({"tasks": [{"goal_idx": 612}], "number": 9999}))
            (root / "docs/evidence/old.json").write_text(json.dumps({"evaluation_tasks": [2192], "metric": 9998}))
            ledger = builder.build_exclusion_ledger(root, experiment_sha="a" * 40)
            self.assertEqual(ledger["excluded_goal_indices"], [612, 2192])
            self.assertNotIn(9998, ledger["excluded_goal_indices"])

    def test_panel_selection_is_deterministic_and_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "configs/experiment").mkdir(parents=True)
            (root / "docs/evidence").mkdir(parents=True)
            (root / "configs/experiment/old.json").write_text(json.dumps({"tasks": [{"goal_idx": 500 + i} for i in range(40)]}))
            fake_goals = [{"instruction_text": f"task {i}"} for i in range(650)]
            original = builder.all_goals
            builder.all_goals = lambda _seed_root: fake_goals  # type: ignore[assignment]
            try:
                first = builder.build_manifest(root, repo_root=root, experiment_sha="b" * 40)
                second = builder.build_manifest(root, repo_root=root, experiment_sha="b" * 40)
            finally:
                builder.all_goals = original  # type: ignore[assignment]
            self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
            self.assertEqual(len(first["evaluation_tasks"]), 64)
            self.assertTrue(set(first["evaluation_tasks"]).isdisjoint(range(500, 540)))
            self.assertEqual(first["evaluation"]["attempts"], 768)
            self.assertEqual(set(first["arm_identities"]), {"base", "ordinary-lora", "sd-lora"})


if __name__ == "__main__":
    unittest.main()
