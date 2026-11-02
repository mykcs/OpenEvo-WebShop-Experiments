import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_webshop_program_ledger import (
    CURRENT_CAMPAIGN_PATH,
    LEDGER_PATH,
    REPORT_PATH,
    verify,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class WebShopProgramLedgerTests(unittest.TestCase):
    def test_repository_ledger_covers_every_protocol_and_current_gate(self):
        result = verify(REPOSITORY_ROOT)

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["config_count"], 59)
        self.assertEqual(result["experiment_entry_count"], 60)
        self.assertFalse(result["h1_38b_launch_authorized"])
        self.assertTrue(result["h1_38b_standing_uuid_authorization"])

    def test_material_claims_bind_direct_supporting_artifacts(self):
        ledger = json.loads((REPOSITORY_ROOT / LEDGER_PATH).read_text(encoding="utf-8"))
        entries = {entry["id"]: entry for entry in ledger["experiments"]}
        self.assertEqual(entries["phase-g"]["record_status"], "diagnostic-only")
        self.assertIn("no standalone machine reconciliation", entries["phase-g"]["denominator"])
        self.assertIn("docs/evidence/remote-runs/2026-08-18/h1.36/REPORT.md", entries["h1.36"]["evidence"])
        self.assertIn("docs/evidence/remote-runs/2026-08-18/h1.37/evidence/h1.37-analysis.json", entries["h1.37"]["evidence"])
        self.assertIn("docs/evidence/remote-runs/2026-08-18/h1.37/REPORT.md", entries["h1.37"]["evidence"])
        self.assertIn("docs/evidence/remote-runs/2026-08-19/h1.38a/evidence/h1.38a-closeout-summary.json", entries["h1.38a-original"]["evidence"])
        self.assertEqual(entries["h1.38b-method-control"]["record_status"], "sealed-result")
        self.assertIn("184 cells", entries["h1.38b-method-control"]["denominator"])
        self.assertEqual(entries["h1.38b-continuation-amendment-2026-08-20"]["record_status"], "sealed-result")
        self.assertEqual(entries["h1.40-g2-multigeneration-r2"]["record_status"], "sealed-result")
        self.assertIn("WALLTIME_RECOVERY_EXHAUSTED.md", " ".join(entries["h1.40-g2-multigeneration-r2"]["evidence"]))
        self.assertEqual(entries["wb1-trackb-continuation"]["record_status"], "protocol-only")
        self.assertIn("docs/evidence/wb1/results-index.v1.json", entries["wb1-trackb-continuation"]["evidence"])
        for experiment_id in ("h1.34", "h1.35"):
            self.assertIn("inherits H1.30 schema/claim metadata", entries[experiment_id]["claim_boundary"])

    def test_missing_config_variant_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (LEDGER_PATH, CURRENT_CAMPAIGN_PATH, REPORT_PATH):
                source = REPOSITORY_ROOT / relative
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
            ledger = json.loads((root / LEDGER_PATH).read_text(encoding="utf-8"))
            for entry in ledger["experiments"]:
                for key in ("config",):
                    path = entry.get(key)
                    if path:
                        source = REPOSITORY_ROOT / path
                        target = root / path
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(source.read_bytes())
                for path in entry["evidence"]:
                    source = REPOSITORY_ROOT / path
                    target = root / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(source.read_bytes())
            ledger["experiments"] = [
                entry
                for entry in ledger["experiments"]
                if entry.get("config") != "configs/experiment/h1.14-state-critic-recovery-diagnostic.json"
            ]
            (root / LEDGER_PATH).write_text(json.dumps(ledger), encoding="utf-8")

            with self.assertRaisesRegex(AssertionError, "config coverage mismatch"):
                verify(root)


if __name__ == "__main__":
    unittest.main()
