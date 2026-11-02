import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.export_server_evidence_bundle import build_export, write_tar


class ExportServerEvidenceBundleTests(unittest.TestCase):
    def test_export_is_explicit_and_filters_unsafe_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            (root / "safe.py").write_text("print('safe')\n", encoding="utf-8")
            (root / "token.txt").write_text("ghp_should_not_export\n", encoding="utf-8")
            (root / "models").mkdir()
            (root / "models" / "config.json").write_text("{}\n", encoding="utf-8")
            manifest, selected = build_export(
                root,
                includes=["safe.py", "token.txt", "models/config.json", "missing.md"],
                observed_at_utc="2026-08-19T00:00:00Z",
            )
            rows = {row["source_path"]: row for row in manifest["entries"]}
            self.assertEqual([path.name for path in selected], ["safe.py"])
            self.assertEqual(rows["token.txt"]["reason"], "likely-credential-bearing")
            self.assertEqual(rows["models/config.json"]["reason"], "blocked-workspace-scope")
            self.assertEqual(rows["missing.md"]["reason"], "missing-at-export")

            archive_path = Path(directory) / "export.tar"
            with archive_path.open("wb") as stream:
                write_tar(manifest, selected, root, stream)
            with tarfile.open(archive_path) as archive:
                self.assertEqual(archive.getnames(), ["server-export-manifest.json", "safe.py"])
                exported_manifest = json.load(archive.extractfile("server-export-manifest.json"))
                self.assertEqual(exported_manifest["selected_count"], 1)


if __name__ == "__main__":
    unittest.main()
