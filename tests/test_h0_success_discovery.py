import json
import tempfile
import unittest
from pathlib import Path

from scripts.openevo_webshop.build_h0_success_discovery_manifest import (
    ROLLOUT_SEEDS,
    build_manifest,
    validate_manifest,
)
from scripts.openevo_webshop.reconcile_h0_success_discovery import reconcile


class H0SuccessDiscoveryContractTests(unittest.TestCase):
    def _manifest(self):
        goals = [{"instruction_text": f"goal {index}"} for index in range(700)]
        return build_manifest(
            goals,
            index_identity={"file_count": 1, "logical_bytes": 1, "tree_sha256": "index"},
            prior_indices=set(range(500, 520)),
            experiment_git_sha="code-sha",
            seed_source_commit="seed-sha",
        )

    def test_manifest_is_disjoint_and_fixed_budget(self):
        manifest = self._manifest()
        validate_manifest(manifest)
        self.assertEqual(manifest["schema"], "openevo.webshop-h0.1-success-discovery.v2")
        self.assertEqual(manifest["task_count"], 16)
        self.assertEqual(manifest["controls"]["rollout_seeds"], ROLLOUT_SEEDS)
        self.assertEqual(manifest["budget"]["attempt_count"], 128)
        self.assertTrue(set(row["goal_idx"] for row in manifest["tasks"]).isdisjoint(range(500, 520)))

    def test_reconciliation_requires_every_task_seed_pair(self):
        manifest = self._manifest()
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            for wave in range(2):
                for slot in range(8):
                    task = manifest["tasks"][wave * 8 + slot]
                    cell = run_root / "harvest" / f"wave{wave}" / f"task{task['goal_idx']}"
                    cell.mkdir(parents=True)
                    rows = []
                    for seed in ROLLOUT_SEEDS:
                        rows.append(
                            {
                                "task_id": task["task_id"],
                                "rollout_seed": seed,
                                "identity_ok": True,
                                "trace_complete": True,
                                "parser_or_fallback_failure": False,
                                "qualified_positive": seed == ROLLOUT_SEEDS[0],
                                "partial_positive": False,
                                "final_task_score": 1.0 if seed == ROLLOUT_SEEDS[0] else 0.0,
                            }
                        )
                    (cell / "attempts.jsonl").write_text(
                        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                    )
            receipt = reconcile(manifest, run_root)
            self.assertEqual(receipt["status"], "pass")
            self.assertEqual(receipt["expected_attempts"], 128)
            self.assertEqual(receipt["observed_attempts"], 128)
            self.assertEqual(receipt["qualified_positive_count"], 16)
            self.assertEqual(receipt["qualified_positive_task_count"], 16)

    def test_launcher_keeps_seed_checkout_out_of_pythonpath(self):
        launcher = Path(__file__).parents[1] / "scripts/openevo_webshop/launch_h0_success_discovery.sh"
        text = launcher.read_text(encoding="utf-8")
        pythonpath_lines = [line for line in text.splitlines() if "PYTHONPATH=" in line]
        self.assertTrue(pythonpath_lines)
        self.assertTrue(all("/workspace/seed-source" not in line for line in pythonpath_lines))
        self.assertIn("--seed-root /workspace/seed-source", text)


if __name__ == "__main__":
    unittest.main()
