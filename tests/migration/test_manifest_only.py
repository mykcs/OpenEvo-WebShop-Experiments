from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LEGACY_IMPORTER = ROOT / "scripts/migration/import_legacy_experiment_assets.py"
PRIVATE_IMPORTER = ROOT / "scripts/migration/import_private_openevo_webshop.py"


def init_repo(path: Path, files: dict[str, str]) -> None:
    path.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    for relative, content in files.items():
        destination = path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=manifest-test",
            "-c",
            "user.email=manifest-test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )


class ManifestOnlyImporterTests(unittest.TestCase):
    def test_legacy_manifest_only_does_not_copy_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target"
            webshop = root / "openevo-webshop"
            seed = root / "seed3090"
            private = root / "OpenEvo"
            init_repo(target, {"README.md": "target\n"})
            init_repo(webshop, {"docs/source.md": "source\n"})
            init_repo(seed, {"scripts/openevo_alfworld/legacy.py": "VALUE = 1\n"})
            init_repo(private, {"README.md": "core\n"})

            result = subprocess.run(
                [
                    sys.executable,
                    str(LEGACY_IMPORTER),
                    "--target",
                    str(target),
                    "--openevo-webshop",
                    str(webshop),
                    "--seed3090",
                    str(seed),
                    "--private-openevo",
                    str(private),
                    "--manifests-only",
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((target / "docs/infrastructure/migration/SOURCE_SNAPSHOT.json").is_file())
            self.assertTrue((target / "docs/infrastructure/migration/MIGRATION_MANIFEST.json").is_file())
            self.assertFalse((target / "docs/science/webshop/source.md").exists())

    def test_private_manifest_only_does_not_copy_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target"
            private = root / "OpenEvo"
            init_repo(target, {"README.md": "target\n"})
            init_repo(
                private,
                {"benchmarks/webshop/src/openevo_webshop/example.py": "VALUE = 1\n"},
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(PRIVATE_IMPORTER),
                    "--target",
                    str(target),
                    "--private-openevo",
                    str(private),
                    "--manifests-only",
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = target / "docs/infrastructure/migration/PRIVATE_OPENEVO_WEBSHOP_MANIFEST.json"
            self.assertTrue(manifest.is_file())
            self.assertFalse(
                (target / "vendor/openevo-webshop/src/openevo_webshop/example.py").exists()
            )


if __name__ == "__main__":
    unittest.main()
