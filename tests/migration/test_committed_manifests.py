from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MIGRATION_DIR = ROOT / "docs/infrastructure/migration"


class CommittedMigrationManifestTests(unittest.TestCase):
    def test_generated_migration_manifests_are_committed_and_consistent(self) -> None:
        archive = json.loads(
            (MIGRATION_DIR / "ARCHIVE_MIGRATION_MANIFEST.json").read_text(encoding="utf-8")
        )
        source_snapshot = json.loads(
            (MIGRATION_DIR / "SOURCE_SNAPSHOT.json").read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (MIGRATION_DIR / "MIGRATION_MANIFEST.json").read_text(encoding="utf-8")
        )
        private_manifest = json.loads(
            (MIGRATION_DIR / "PRIVATE_OPENEVO_WEBSHOP_MANIFEST.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            source_snapshot["sources"]["openevo_webshop"]["head"],
            archive["sources"]["openevo_webshop"],
        )
        self.assertEqual(
            source_snapshot["sources"]["seed3090"]["head"],
            archive["sources"]["seed3090_main"],
        )
        self.assertEqual(
            source_snapshot["sources"]["private_openevo"]["head"],
            private_manifest["source_head"],
        )
        self.assertNotIn("path", source_snapshot["sources"]["openevo_webshop"])
        self.assertEqual(manifest["record_count"], len(manifest["records"]))
        self.assertGreater(manifest["record_count"], 0)
        self.assertEqual(private_manifest["record_count"], len(private_manifest["records"]))
        self.assertGreater(private_manifest["record_count"], 0)
        for record in manifest["records"] + private_manifest["records"]:
            self.assertRegex(record["sha256"], re.compile(r"^[0-9a-f]{64}$"))
            self.assertFalse(Path(record["source_path"]).is_absolute())
            self.assertFalse(Path(record["destination_path"]).is_absolute())
            self.assertNotIn("..", Path(record["destination_path"]).parts)


if __name__ == "__main__":
    unittest.main()
