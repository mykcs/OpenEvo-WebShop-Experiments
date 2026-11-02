"""CPU-only tests for the Q17 historical task-identity audit."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("q17_identity", ROOT / "scripts/analysis/q17_frontier_identity_audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

class IdentityAuditTests(unittest.TestCase):
    def clean_row(self, position=0):
        return dict(worker_seed=7, source_session_index=123, instruction_sha256="a" * 64,
                    final_task_score=1.0, won=True, identity_ok=True, trace_complete=True,
                    parser_or_fallback_failure=False, parser_repair_used=False,
                    engineering_invalid=False, error=None, round_index=0,
                    final_panel_access=0, schedule_position=position)

    def test_equal_counts_are_not_equal_identities(self):
        result = audit.compare_identities({"wrong"}, {"correct"}, {"correct"})
        self.assertFalse(result["selected_equals_current_clean_exact"])
        self.assertEqual(result["selected_without_observed_clean_exact"], ["wrong"])

    def test_prior_success_is_not_current_success(self):
        result = audit.compare_identities({"old"}, {"new"}, {"old", "new"})
        self.assertTrue(result["selected_subset_cumulative_clean_exact"])
        self.assertFalse(result["selected_subset_current_clean_exact"])
        self.assertEqual(result["selected_prior_only"], ["old"])

    def test_frozen_success_filters(self):
        self.assertTrue(audit.clean_exact(self.clean_row()))
        for key, value in {"won": False, "identity_ok": False, "trace_complete": False,
                           "parser_or_fallback_failure": True, "parser_repair_used": True,
                           "engineering_invalid": True, "error": "failure",
                           "final_task_score": 0.999}.items():
            row = self.clean_row(); row[key] = value
            self.assertFalse(audit.clean_exact(row), key)
        self.assertEqual(audit.task_identity(self.clean_row()), "7:123:" + "a" * 64)

    def test_out_of_scope_and_duplicate_manifest_fail_closed(self):
        digest = "a" * 64
        for path in ["round-122/attempts.jsonl", "../final_panel.json", "round-001/final.json"]:
            with self.assertRaises(ValueError):
                audit.parse_manifest(digest + "  " + path)
        row = digest + "  round-000/attempts.jsonl\n"
        with self.assertRaises(ValueError):
            audit.parse_manifest(row + row)

    def fixture(self, root):
        rows = [self.clean_row(n) for n in range(128)]
        # A Unicode line separator inside JSON strings must not split a JSONL record.
        rows[0]["note"] = "left\u2028right"
        files = {
            "round-000/ROUND_COMPLETE.json": json.dumps({"status": "PASS", "round_index": 0,
                "rollout_summary": {"attempt_count": 128, "clean_exact_success_count": 128,
                "clean_exact_success_task_count": 1, "engineering_invalid_count": 0}}).encode(),
            "round-000/attempts.jsonl": b"\n".join(json.dumps(r, ensure_ascii=False).encode() for r in rows),
            "round-000/sd-lora/dataset/selection.json": json.dumps({"selected_task_count": 1,
                "bound_task_identities": [audit.task_identity(rows[0])]}).encode(),
        }
        for name, data in files.items():
            target = root / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        return "".join(hashlib.sha256(data).hexdigest() + "  " + name + "\n" for name, data in files.items())

    def test_hash_bound_end_to_end_and_unicode(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(audit, "LAST_ROUND", 0):
            root = Path(tmp); manifest = self.fixture(root)
            result = audit.audit(root, manifest)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["rollout_count"], 128)
            self.assertEqual(result["exact_set_equality_rounds"], 1)
            self.assertEqual(len(result["inputs"]), 3)

    def test_tampered_input_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(audit, "LAST_ROUND", 0):
            root = Path(tmp); manifest = self.fixture(root)
            (root / "round-000/attempts.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                audit.audit(root, manifest)

    def test_frozen_authority_ast_match_and_drift(self):
        implementation = (ROOT / "scripts/analysis/q17_frontier_identity_audit.py").read_text()
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "authority.py"; source.write_text(implementation)
            with patch.object(audit, "AUTHORITY_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()):
                audit.verify_authority(source, implementation)
                changed = implementation.replace("==1.0 and r.get('won')", ">=0.5 and r.get('won')")
                with self.assertRaisesRegex(ValueError, "AST-identical"):
                    audit.verify_authority(source, changed)

if __name__ == "__main__":
    unittest.main()
