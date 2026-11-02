import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.openevo_webshop import build_h138b_method_control_inputs as h138b


class H138BMethodControlInputsTests(unittest.TestCase):
    def _sealed_fixture(self, root: Path) -> Path:
        selection = []
        rows = []
        for task in range(8):
            task_id = f"webshop_{task + 500:06d}"
            for offset in range(2):
                source_hash = hashlib.sha256(f"source-{task}-{offset}".encode()).hexdigest()
                selection.append(
                    {
                        "task_id": task_id,
                        "goal_idx": task + 500,
                        "rollout_seed": 1000 + task * 2 + offset,
                        "record_sha256": source_hash,
                    }
                )
                rows.append(
                    {
                        "task_id": task_id,
                        "traces": [{"prompt_messages": [], "response_messages": []}],
                        "metadata": {
                            "rollout_seed": 1000 + task * 2 + offset,
                            "source_record_sha256": source_hash,
                        },
                    }
                )
        records = b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)
        records_path = root / "records.jsonl"
        records_path.write_bytes(records)
        manifest = {
            "record_count": 16,
            "identity_count": 8,
            "records_path": "records.jsonl",
            "records_sha256": hashlib.sha256(records).hexdigest(),
            "selection": selection,
            "selection_sha256": h138b._canonical_hash(selection),
            "source": "fixture",
            "source_route": "self-generated only",
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_sealed_constants_match_published_d8_manifest(self):
        manifest = Path(__file__).parents[1] / "docs/evidence/remote-runs/2026-08-19/h1.38a/manifests/diversity-8-dataset-manifest.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(payload["records_sha256"], h138b.D8_RECORDS_SHA256)
        self.assertEqual(payload["selection_sha256"], h138b.D8_SELECTION_SHA256)
        self.assertEqual(payload["record_count"], 16)
        self.assertEqual(payload["identity_count"], 8)

    def test_eight_two_record_increments_reconstruct_parent_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._sealed_fixture(root)
            parent = json.loads(manifest.read_text(encoding="utf-8"))
            old_records, old_selection = h138b.D8_RECORDS_SHA256, h138b.D8_SELECTION_SHA256
            try:
                h138b.D8_RECORDS_SHA256 = parent["records_sha256"]
                h138b.D8_SELECTION_SHA256 = parent["selection_sha256"]
                result = h138b.write_increments(manifest, root / "increments")
            finally:
                h138b.D8_RECORDS_SHA256, h138b.D8_SELECTION_SHA256 = old_records, old_selection
            self.assertEqual(len(result["increments"]), 8)
            reconstructed = b"".join(
                (Path(item["manifest_path"]).with_name("records.jsonl")).read_bytes()
                for item in result["increments"]
            )
            self.assertEqual(reconstructed, (root / "records.jsonl").read_bytes())
            self.assertTrue(all(len(item["rollout_seeds"]) == 2 for item in result["increments"]))
            for item in result["increments"]:
                manifest_path = Path(item["manifest_path"])
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    manifest_path.read_bytes(), h138b.openevo_dataset_manifest_bytes(payload)
                )

    def test_fresh_manifest_is_task_disjoint_and_has_exact_704_attempts(self):
        goals = [{"instruction_text": f"goal {index}"} for index in range(900)]
        selection = []
        for index in range(8):
            for seed in range(2):
                selection.append({"task_id": f"webshop_{500 + index:06d}", "goal_idx": 500 + index, "rollout_seed": seed})
        prior = set(range(500, 700))
        old_count = h138b.EXPECTED_PRIOR_IDENTITY_COUNT
        old_digest = h138b.EXPECTED_PRIOR_IDENTITY_DIGEST
        try:
            h138b.EXPECTED_PRIOR_IDENTITY_COUNT = len(prior)
            h138b.EXPECTED_PRIOR_IDENTITY_DIGEST = h138b._index_digest(prior)
            manifest = h138b.build_evaluation_manifest(
                goals,
                prior_indices=prior,
                d8_selection=selection,
                experiment_git_sha="a" * 40,
            )
        finally:
            h138b.EXPECTED_PRIOR_IDENTITY_COUNT = old_count
            h138b.EXPECTED_PRIOR_IDENTITY_DIGEST = old_digest
        fresh = {item["goal_idx"] for item in manifest["fresh_tasks"]}
        self.assertFalse(fresh & prior)
        self.assertEqual(len(fresh), 48)
        self.assertEqual(manifest["planned_attempts"], 704)
        self.assertEqual(manifest["acquisition"]["attempts"], 32)
        self.assertEqual(manifest["retention"]["attempts"], 96)
        self.assertEqual(manifest["fresh"]["attempts"], 576)
        self.assertEqual(manifest["task_selection"]["prior_identity_indices"], sorted(prior))

    def test_fresh_manifest_rejects_prior_ledger_drift(self):
        goals = [{"instruction_text": f"goal {index}"} for index in range(900)]
        selection = [
            {"task_id": f"webshop_{500 + index:06d}", "goal_idx": 500 + index, "rollout_seed": seed}
            for index in range(8)
            for seed in range(2)
        ]
        with self.assertRaisesRegex(ValueError, "prior-identity count mismatch"):
            h138b.build_evaluation_manifest(
                goals,
                prior_indices=set(range(500, 700)),
                d8_selection=selection,
                experiment_git_sha="a" * 40,
            )


if __name__ == "__main__":
    unittest.main()
