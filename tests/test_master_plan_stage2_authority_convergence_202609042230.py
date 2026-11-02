from __future__ import annotations

import ast
import hashlib
import json
from itertools import product
from pathlib import Path
import unittest

from scripts.openevo_webshop.autonomous_campaign import CampaignError, assert_campaign_execution_authority, resolve_manifest

ROOT = Path(__file__).resolve().parents[1]
ROUTER_PATH = ROOT / "configs/experiment/campaigns/current-ceiling1-autonomous.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MasterPlanStage2AuthorityConvergenceTest(unittest.TestCase):
    def setUp(self):
        self.router = json.loads(ROUTER_PATH.read_text())
        manifest_ref = self.router["active_manifest"]
        self.manifest_path = ROOT / manifest_ref["path"]
        self.manifest = json.loads(self.manifest_path.read_text())
        authority_ref = self.router["repository_current_stage2_runtime_authority"]
        self.authority_path = ROOT / authority_ref["path"]
        self.authority = json.loads(self.authority_path.read_text())

    def test_router_hash_binds_one_campaign_and_one_stage2_authority(self):
        self.assertEqual(sha256(self.manifest_path), self.router["active_manifest"]["sha256"])
        authority_ref = self.router["repository_current_stage2_runtime_authority"]
        self.assertEqual(sha256(self.authority_path), authority_ref["sha256"])
        self.assertFalse(self.router["execution_activation_authority"])
        self.assertFalse(authority_ref["execution_activation_authority"])

    def test_real_ceiling_execution_cannot_bypass_stable_router(self):
        activated = dict(self.manifest)
        activated["activation_authority"] = True
        with self.assertRaisesRegex(CampaignError, "stable hash-bound router"):
            assert_campaign_execution_authority(activated, {"kind": "direct"})

    def test_campaign_control_plane_resolves_bound_authority(self):
        manifest, resolution = resolve_manifest(ROOT, ROUTER_PATH)
        self.assertEqual(manifest["campaign_id"], self.manifest["campaign_id"])
        resolved = resolution["repository_current_stage2_runtime_authority"]
        self.assertEqual(resolved["sha256"], self.router["repository_current_stage2_runtime_authority"]["sha256"])
        self.assertEqual(resolved["status"], "READY_FOR_AUDITED_NATURAL_MIGRATION")

    def test_campaign_resolves_the_same_stage2_authority(self):
        bindings = self.manifest["bindings"]
        authority_ref = self.router["repository_current_stage2_runtime_authority"]
        self.assertEqual(bindings["shared_stage2_runtime_authority_path"], authority_ref["path"])
        self.assertEqual(bindings["shared_stage2_runtime_authority_sha256"], authority_ref["sha256"])
        self.assertEqual(
            self.manifest["repository_current_stage2_runtime_authority"]["path"],
            authority_ref["path"],
        )

    def test_authority_runtime_dependency_paths_exist_in_fresh_checkout(self):
        runtime_paths = list(self.authority["resource_backend"].values())
        runtime_paths.remove(self.authority["resource_backend"]["name"])
        runtime_paths.extend(
            value
            for value in self.authority["shared_runtime"].values()
            if isinstance(value, str)
        )
        for rel in runtime_paths:
            with self.subTest(path=rel):
                self.assertTrue((ROOT / rel).is_file(), rel)

        for rel in runtime_paths:
            path = ROOT / rel
            if path.suffix != ".py":
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or not node.module:
                    continue
                prefix = "scripts.openevo_webshop."
                if not node.module.startswith(prefix):
                    continue
                dependency = ROOT / (node.module.replace(".", "/") + ".py")
                with self.subTest(importer=rel, dependency=str(dependency.relative_to(ROOT))):
                    self.assertTrue(dependency.is_file(), dependency)

    def test_every_declared_cross_arm_pair_is_resource_feasible(self):
        cfg_path = ROOT / self.authority["resource_backend"]["config"]
        cfg = json.loads(cfg_path.read_text())
        profiles = list(cfg["profiles"].values())
        three_b = [p for p in profiles if p["arm"] == "3b"]
        qwen3 = [p for p in profiles if p["arm"] == "qwen3-1p7b"]
        self.assertTrue(three_b)
        self.assertTrue(qwen3)
        self.assertGreaterEqual(cfg["process_slots_per_gpu"], 2)
        self.assertEqual(cfg["same_arm_slots_per_gpu"], 1)
        for left, right in product(three_b, qwen3):
            self.assertLessEqual(
                left["vram_request_mib"] + right["vram_request_mib"],
                cfg["gpu_project_budget_mib"],
            )

    def test_legacy_generic_authority_cannot_compete(self):
        legacy = json.loads(
            (ROOT / "configs/experiment/openevo-ceiling1-stage2-runtime-authority-v1.json").read_text()  # legacy-revision-compat
        )
        self.assertEqual(legacy["status"], "SUPERSEDED_HISTORY_ONLY")
        self.assertFalse(legacy["execution_authority"])
        self.assertEqual(legacy["superseded_by"]["path"], str(self.authority_path.relative_to(ROOT)))


if __name__ == "__main__":
    unittest.main()
