from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "e1_evidence", ROOT / "scripts/analysis/validate_post_advisor_e1_evidence_202610010045.py"
)
assert SPEC and SPEC.loader
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)


class EvidenceBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="e1-evidence-test-")
        self.addCleanup(self.temp.cleanup)
        self.bundle = Path(self.temp.name) / "bundle"
        shutil.copytree(ROOT / evidence.BUNDLE, self.bundle)

    def index(self) -> dict:
        return json.loads((self.bundle / "index.json").read_text())

    def save_index(self, index: dict) -> None:
        (self.bundle / "index.json").write_text(json.dumps(index))

    def test_original_bundle_passes_with_narrow_claim(self) -> None:
        result = evidence.validate(self.bundle)
        self.assertEqual(result["snapshots"], 11)
        self.assertEqual(result["episode_hash_references"], 1024)
        self.assertFalse(result["server_leaves_revalidated"])
        self.assertFalse(result["scientific_efficacy_assessed"])
        self.assertFalse(result["execution_authority"])

    def test_historical_failure_and_producers_are_preserved(self) -> None:
        receipt = json.loads((self.bundle / "reconciliation.json").read_text())
        self.assertEqual(receipt["r159_health"]["status"], "FAIL")
        self.assertEqual(receipt["seal"]["terminal_carrier_health_status"], "FAIL_DIAGNOSTIC_RESOLVED")
        r152 = receipt["rounds"]["152"]
        self.assertNotEqual(r152["rollout_producer_execution_sha"], r152["round_complete_producer_execution_sha"])
        self.assertEqual(receipt["budget"]["e1_formal_rollouts_remaining"], 0)

    def test_even_whitespace_change_is_rejected(self) -> None:
        with (self.bundle / "reconciliation.json").open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaisesRegex(ValueError, "digest drift"):
            evidence.validate(self.bundle)

    def test_missing_snapshot_fails_closed(self) -> None:
        row = next(row for row in self.index()["entries"] if row["role"] == "a11-detail")
        (self.bundle / row["archive_path"]).unlink()
        with self.assertRaisesRegex(ValueError, "missing archive"):
            evidence.validate(self.bundle)

    def test_historical_host_path_is_never_opened(self) -> None:
        index = self.index()
        index["entries"][0]["archive_path"] = "/workspace/live-authority.json"
        self.save_index(index)
        with self.assertRaisesRegex(ValueError, "content-addressed"):
            evidence.validate(self.bundle)

    def test_symlink_escape_is_rejected(self) -> None:
        receipt = self.bundle / "reconciliation.json"
        outside = self.bundle.parent / "outside.json"
        receipt.replace(outside)
        receipt.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escaped bundle"):
            evidence.validate(self.bundle)

    def test_missing_role_is_rejected(self) -> None:
        index = self.index()
        index["entries"][0]["role"] = index["entries"][1]["role"]
        self.save_index(index)
        with self.assertRaisesRegex(ValueError, "roles drift"):
            evidence.validate(self.bundle)

    def test_execution_authority_cannot_be_promoted(self) -> None:
        index = self.index()
        index["execution_authority"] = True
        self.save_index(index)
        with self.assertRaisesRegex(ValueError, "cannot authorize"):
            evidence.validate(self.bundle)

    def test_snapshot_authority_cannot_be_promoted(self) -> None:
        index = self.index()
        index["entries"][0]["execution_authority"] = True
        self.save_index(index)
        with self.assertRaisesRegex(ValueError, "cannot authorize"):
            evidence.validate(self.bundle)

    def test_scientific_efficacy_claim_is_rejected(self) -> None:
        index = self.index()
        index["scientific_efficacy_assessed"] = True
        self.save_index(index)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            evidence.validate(self.bundle)

    def test_source_head_drift_is_rejected(self) -> None:
        index = self.index()
        index["source_git_sha"] = "a" * 40
        self.save_index(index)
        with self.assertRaisesRegex(ValueError, "source head drift"):
            evidence.validate(self.bundle)

    def test_consuming_binding_drift_is_rejected(self) -> None:
        index = self.index()
        index["entries"][0]["receipt_binding"] = []
        self.save_index(index)
        with self.assertRaisesRegex(ValueError, "binding drift"):
            evidence.validate(self.bundle)

    def test_shortened_provenance_paths_are_rejected(self) -> None:
        for role in ("reconciliation", "health-policy"):
            with self.subTest(role=role):
                index = self.index()
                row = next(row for row in index["entries"] if row["role"] == role)
                original = row.copy()
                row["source_path"] = Path(row["source_path"]).name
                row["source_url"] = f"https://github.com/mykcs/openevo-experiment/blob/{evidence.SOURCE_SHA}/{row['source_path']}"
                self.save_index(index)
                with self.assertRaisesRegex(ValueError, "historical source path drift"):
                    evidence.validate(self.bundle)
                row.update(original)
                self.save_index(index)


if __name__ == "__main__":
    unittest.main()
