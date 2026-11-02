import json
import tempfile
import unittest
from pathlib import Path

from scripts.import_server_evidence_snapshot import build_snapshot, sha256_file
from scripts.server_evidence_inventory import inventory_entry, reclassify_inventory


class ImportServerEvidenceSnapshotTests(unittest.TestCase):
    def test_inventory_labels_raw_episode_the_same_as_importer(self):
        entry = inventory_entry(
            Path("evaluation/base-task1110/episodes/webshop_001110-2026090200.json"),
            size_bytes=17,
            sha256="sha256:" + "a" * 64,
        )

        self.assertEqual(entry["artifact_type"], "raw-trajectory")
        self.assertEqual(entry["scientific_role"], "raw episode or trajectory payload")
        self.assertEqual(entry["selection"], "not-staged")
        self.assertTrue(entry["needs_future_manual_archive"])

    def test_inventory_reclassification_repairs_historical_raw_episode_labels(self):
        document = reclassify_inventory(
            {
                "schema": "openevo.server-evidence-inventory.v1",
                "entries": [
                    {
                        "source_path": "evaluation/base-task1110/episodes/webshop_001110-2026090200.json",
                        "artifact_type": "other",
                        "scientific_role": "unclassified server artifact",
                        "needs_future_manual_archive": False,
                    }
                ],
            }
        )

        entry = document["entries"][0]
        self.assertEqual(entry["artifact_type"], "raw-trajectory")
        self.assertEqual(entry["pre_import_reason"], "raw-trajectory-payload")
        self.assertTrue(entry["needs_future_manual_archive"])

    def test_dry_run_filters_and_deduplicates_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            repo = root / "repo"
            existing = repo / "docs" / "evidence" / "remote-runs"
            output = repo / "docs" / "evidence" / "server-snapshots" / "2026-08-18"
            source.mkdir(parents=True)
            existing.mkdir(parents=True)

            (existing / "known.json").write_text('{"status":"known"}\n', encoding="utf-8")
            (source / "duplicate.json").write_text('{"status":"known"}\n', encoding="utf-8")
            (source / "new.json").write_text('{"status":"new"}\n', encoding="utf-8")
            (source / "bad.json").write_text("not-json\n", encoding="utf-8")
            (source / "raw.jsonl").write_text('{"raw":true}\n', encoding="utf-8")
            episode = source / "evaluation" / "episodes" / "episode.json"
            episode.parent.mkdir(parents=True)
            episode.write_text('{"raw":true}\n', encoding="utf-8")
            (source / "secret.txt").write_text("api_key=super-secret-value\n", encoding="utf-8")
            safe_target = source / "new.json"
            (source / "linked.json").symlink_to(safe_target)
            runtime_state = source / "run" / "ray" / "ray" / "session_1" / "node.json"
            runtime_state.parent.mkdir(parents=True)
            runtime_state.write_text('{"node":"private-runtime-state"}\n', encoding="utf-8")

            document = build_snapshot(
                source=source,
                repository_root=repo,
                existing_evidence_root=existing,
                output_root=output,
                source_label="/server/runs",
                observed_at_utc="2026-08-18T10:04:40Z",
                apply=False,
            )

            self.assertEqual(document["candidate_count"], 8)
            self.assertEqual(document["imported_count"], 1)
            self.assertEqual(document["files"][0]["source_path"], "new.json")
            reasons = {item["source_path"]: item["reason"] for item in document["skipped"]}
            self.assertEqual(reasons["duplicate.json"], "duplicate-existing-content")
            self.assertEqual(reasons["bad.json"], "invalid-json")
            self.assertEqual(reasons["raw.jsonl"], "unsupported-or-raw-suffix")
            self.assertEqual(reasons["secret.txt"], "likely-credential-bearing")
            self.assertEqual(reasons["linked.json"], "symlink-not-followed")
            self.assertEqual(reasons["evaluation/episodes/episode.json"], "raw-trajectory-payload")
            self.assertEqual(
                reasons["run/ray/ray/session_1/node.json"], "runtime-generated-state"
            )
            self.assertFalse(output.exists())

    def test_apply_copies_content_and_writes_hash_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            repo = root / "repo"
            existing = repo / "docs" / "evidence" / "remote-runs"
            output = repo / "docs" / "evidence" / "server-snapshots" / "2026-08-18"
            source.mkdir(parents=True)
            existing.mkdir(parents=True)
            artifact = source / "phase-h1" / "evidence" / "result.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text('{"score":0.5}\n', encoding="utf-8")

            document = build_snapshot(
                source=source,
                repository_root=repo,
                existing_evidence_root=existing,
                output_root=output,
                source_label="/server/runs",
                observed_at_utc="2026-08-18T10:04:40Z",
                apply=True,
            )

            copied = output / "phase-h1" / "evidence" / "result.json"
            index = json.loads((output / "snapshot-index.json").read_text(encoding="utf-8"))
            self.assertEqual(copied.read_bytes(), artifact.read_bytes())
            self.assertEqual(document["files"][0]["sha256"], sha256_file(copied))
            self.assertEqual(index["schema"], "openevo.server-evidence-snapshot.v1")
            self.assertEqual(index["snapshot_import_repo_sha"], index["repository_base_sha"])
            self.assertIsNone(index["scientific_contract_sha"])
            self.assertIsNone(index["execution_orchestration_sha"])

    def test_apply_copies_verified_server_inventory_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            repo = root / "repo"
            existing = repo / "docs" / "evidence" / "remote-runs"
            output = repo / "docs" / "evidence" / "server-snapshots" / "2026-08-18"
            source.mkdir(parents=True)
            existing.mkdir(parents=True)
            artifact = source / "evidence" / "result.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text('{"score":0.5}\n', encoding="utf-8")
            inventory = root / "server-inventory.json"
            inventory.write_text(
                json.dumps(
                    {
                        "schema": "openevo.server-evidence-inventory.v1",
                        "inventory_fingerprint": "sha256:" + "a" * 64,
                        "entries": [
                            {
                                "source_path": "evidence/result.json",
                                "sha256": sha256_file(artifact),
                                "selection": "staged-for-importer",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            document = build_snapshot(
                source=source,
                repository_root=repo,
                existing_evidence_root=existing,
                output_root=output,
                source_label="/server/runs",
                observed_at_utc="2026-08-18T10:04:40Z",
                server_inventory=inventory,
                apply=True,
            )

            copied = json.loads((output / "server-inventory.json").read_text(encoding="utf-8"))
            self.assertEqual(copied["schema"], "openevo.server-evidence-inventory.v1")
            self.assertEqual(document["server_inventory_ledger"], "server-inventory.json")


if __name__ == "__main__":
    unittest.main()
