from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.openevo_webshop import build_q03_step119_source_overlay as overlay


class Step119SourceOverlayTest(unittest.TestCase):
    def test_build_overlay_packages_dataset_adapter_and_manifest_helper(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            seed = root / "seed"
            source_member = "agent_system/multi_turn_rollout/rollout_loop.py"
            source_path = seed / source_member
            source_path.parent.mkdir(parents=True)
            source_path.write_text("pass\n", encoding="utf-8")
            source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()

            def git(_root: Path, *args: str) -> str:
                return "fixture-seed-commit" if args == ("rev-parse", "HEAD") else ""

            with (
                patch.object(overlay, "EXPECTED_SEED_COMMIT", "fixture-seed-commit"),
                patch.object(overlay, "SOURCE_FILES", {source_member: source_sha}),
                patch.object(overlay, "_git", git),
                patch.object(overlay, "patch_rollout_loop", lambda _text: "pass  # patched\n"),
            ):
                result = overlay.build_overlay(seed, root / "published")

            output = Path(result["path"])
            source_dir = Path(overlay.__file__).parent
            members = {
                "verl/utils/dataset/q03_step119_validation_dataset.py": source_dir / "q03_step119_validation_dataset.py",
                "verl/utils/dataset/q03_source_faithful_manifest.py": source_dir / "q03_source_faithful_manifest.py",
            }
            for relative, source in members.items():
                content = source.read_bytes()
                self.assertEqual((output / relative).read_bytes(), content)
                self.assertEqual(
                    result["sources"][relative]["source_sha256"],
                    hashlib.sha256(content).hexdigest(),
                )

    def test_formal_profile_probes_every_dynamic_overlay_module(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        profile_path = repo_root / "configs/experiment/profiles/q03-step119-val128-20261025-profile.json"
        evidence_path = repo_root / "docs/evidence/experiment-passports/20261026/Q03_STEP119_RUNTIME_IMPORT_IDENTITY.json"
        profile_bytes = profile_path.read_bytes()
        profile = json.loads(profile_bytes)
        modules = profile["execution"]["runtime_import_modules"]
        required = {
            "verl.utils.dataset.q03_source_faithful_manifest",
            "verl.utils.dataset.q03_step119_validation_dataset",
        }
        self.assertEqual(len(modules), len(set(modules)))
        self.assertTrue(required.issubset(set(modules)))

        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        runtime = evidence["host_runtime_import_identity"]
        self.assertEqual(runtime["status"], "PASS")
        self.assertTrue(required.issubset(set(runtime["modules"])))
        self.assertEqual(evidence["profile_sha256"], hashlib.sha256(profile_bytes).hexdigest())



if __name__ == "__main__":
    unittest.main()
