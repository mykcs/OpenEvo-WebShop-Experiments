import hashlib
import json
import unittest
from pathlib import Path

from scripts.openevo_webshop.build_h130_task_identity_diagnostic import build
from scripts.openevo_webshop.reconcile_h130_task_identity_diagnostic import reconcile


ROOT = Path(__file__).parents[1]


def load(relative: str):
    path = ROOT / relative
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


class H130TaskIdentityDiagnosticTests(unittest.TestCase):
    def _build(self):
        h0 = "docs/evidence/remote-runs/2026-08-17/h0.1/manifests/h0.1-success-discovery-v2.json"
        training = "docs/evidence/remote-runs/2026-08-17/h1.30/manifests/h1.30-training-manifest-v3.json"
        evaluation = "docs/evidence/remote-runs/2026-08-17/h1.30/manifests/h1.30-task-manifest-v3.json"
        taxonomy = "docs/evidence/remote-runs/2026-08-17/h1.30/evidence/h1.30-failure-taxonomy-v3.json"
        sources = {
            "h0_manifest": load(h0),
            "v3_training_manifest": load(training),
            "v3_evaluation_manifest": load(evaluation),
            "v3_taxonomy": load(taxonomy),
            "h0_manifest_file_sha256": sha256_file(h0),
            "v3_training_manifest_file_sha256": sha256_file(training),
            "v3_evaluation_manifest_file_sha256": sha256_file(evaluation),
            "v3_taxonomy_file_sha256": sha256_file(taxonomy),
        }
        return build(load("configs/experiment/h1.30-task-identity-diagnostic.json"), sources)

    def test_manifest_is_fresh_grouped_and_fail_closed(self):
        document = self._build()
        self.assertEqual(document["schema"], "openevo.webshop-h1.30-training-manifest.v1")
        self.assertFalse(document["formal_task_consumption_allowed"])
        self.assertEqual(len(document["train_tasks"]), 16)
        self.assertEqual(len(document["validation_tasks"]), 8)
        self.assertEqual(
            document["diagnostic_groups"]["prior_positive"][:2],
            ["webshop_001166", "webshop_001880"],
        )
        selected = {row["task_id"] for row in document["train_tasks"]}
        evaluation = {row["task_id"] for row in load(
            "docs/evidence/remote-runs/2026-08-17/h1.30/manifests/h1.30-task-manifest-v3.json"
        )["tasks"]}
        self.assertTrue(selected.isdisjoint(evaluation))

    def test_manifest_digest_is_stable(self):
        first = self._build()
        second = self._build()
        self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
        self.assertEqual(first, second)

    def test_reconciliation_requires_four_attempts_per_task_and_preserves_groups(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "run"
            grouped_ids = {
                "prior_positive": [f"a{index}" for index in range(8)],
                "v3_nonpositive": [f"b{index}" for index in range(8)],
            }
            for group, task_ids in grouped_ids.items():
                for task_id in task_ids:
                    path = run_root / "harvest" / group / task_id / "attempts.jsonl"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    rows = [
                        {
                            "task_id": task_id,
                            "qualified_positive": group == "prior_positive" and seed == 0,
                            "parser_or_fallback_failure": False,
                            "final_task_score": 1 if seed == 0 else 0,
                        }
                        for seed in range(4)
                    ]
                    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"diagnostic_groups": grouped_ids}), encoding="utf-8")
            output = root / "reconciliation.json"
            result = reconcile(manifest, run_root, output)
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["overall"]["attempts"], 64)
            self.assertEqual(result["groups"]["prior_positive"]["qualified_positive_count"], 8)
            self.assertEqual(result["groups"]["v3_nonpositive"]["qualified_positive_count"], 0)


if __name__ == "__main__":
    unittest.main()
