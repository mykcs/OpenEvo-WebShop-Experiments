from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts/openevo_webshop/run_mechanism1_m1d_seed_stage1_202609060100.py"
MANIFEST = ROOT / "configs/experiment/manifests/openevo-mechanism1-m1d-seed-stage1-matched-202609060000.json"
AMENDMENT = ROOT / "configs/experiment/amendments/openevo-mechanism1-m1d-matched-world-202609060100.json"
EVIDENCE = ROOT / "docs/evidence/mechanism1/m1d-world-seed-qualification-202609060100.json"
PASSPORT = ROOT / "configs/experiment/designs/openevo-mechanism1-seed-stage1-init-probe-3b-202609052200.json"
CAMPAIGN = ROOT / "configs/experiment/campaigns/202609052200-mechanism1-autonomous.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_driver():
    spec = importlib.util.spec_from_file_location("m1d_driver_test", DRIVER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MechanismM1DReadinessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = load_driver()
        cls.manifest = load(MANIFEST)
        cls.amendment = load(AMENDMENT)
        cls.evidence = load(EVIDENCE)

    def test_manifest_is_exact_and_execution_locked(self):
        doc = self.driver.load_manifest(MANIFEST)
        self.assertEqual(len(doc["tasks"]), 180)
        self.assertEqual(len(self.driver.expanded_identity_rows(doc)), 1440)
        self.assertFalse(doc["activation_authority"])
        self.assertFalse(doc["formal_task_consumption_allowed"])

    def test_world_identity_is_matched_while_sampling_is_method_native(self):
        matched = self.manifest["matched_axes"]
        native = self.manifest["method_native_axes"]
        self.assertEqual(matched["webshop_world_construction_seed"], 2026)
        self.assertEqual(matched["webshop_num_products"], 1000)
        self.assertEqual(matched["lucene_index"], "indexes_1k")
        self.assertEqual(native["existing_openevo_stage1"]["policy_temperature"], 1.0)
        self.assertEqual(native["new_seed_stage1"]["policy_temperature"], 0.4)
        self.assertIsNone(native["new_seed_stage1"]["request_level_sampling_seed"])

    def test_worker_world_seeds_are_constant(self):
        self.assertEqual(self.driver.matched_world_worker_seeds(4), [2026, 2026, 2026, 2026])
        self.assertEqual(self.driver.matched_world_worker_seeds(128), [2026] * 128)
        with self.assertRaises(ValueError):
            self.driver.matched_world_worker_seeds(-1)

    def test_inner_seed_ray_has_zero_gpu_authority(self):
        source = inspect.getsource(self.driver.make_matched_build_manager)
        self.assertIn("num_gpus=0", source)

    def test_small_world_binding_selects_the_small_index_family(self):
        bound = self.driver.bind_world_env_kwargs({"num_products": None, "observation_mode": "text"})
        self.assertEqual(bound["num_products"], 1000)
        self.assertEqual(bound["observation_mode"], "text")

    def test_prospective_amendment_preserves_original_passport(self):
        self.assertEqual(
            hashlib.sha256(PASSPORT.read_bytes()).hexdigest(),
            "0f3528b174a8df206aafa1f45cb55d7fe0f1aa7ccbf647dd708bab61137c6c18",
        )
        self.assertEqual(self.amendment["parent_passport_sha256"], hashlib.sha256(PASSPORT.read_bytes()).hexdigest())
        self.assertFalse(self.amendment["activation_authority"])
        self.assertFalse(self.amendment["formal_task_consumption_allowed"])
        self.assertEqual(self.amendment["decision"]["webshop_world_construction_seed"], 2026)
        self.assertEqual(self.amendment["decision"]["webshop_num_products"], 1000)

    def test_cpu_probe_proves_world_seed_is_task_identity(self):
        obs = self.evidence["observations"]
        self.assertEqual(obs["world_seed_2026"]["instruction_hash_mismatches_vs_frozen_schedule"], 0)
        self.assertEqual(obs["world_seed_2027"]["instruction_hash_mismatches_vs_frozen_schedule"], 180)
        self.assertEqual(self.evidence["formal_task_consumption"], 0)
        self.assertEqual(self.evidence["model_calls"], 0)
        self.assertEqual(self.evidence["trajectory_count"], 0)

    def test_driver_hard_stops_after_upstream_baseline_collection(self):
        source = DRIVER.read_text(encoding="utf-8")
        formal_source = inspect.getsource(self.driver.run_formal)
        self.assertIn("collect_baseline_rollouts", formal_source)
        self.assertNotIn("generate_candidate_skills", source)
        self.assertNotIn("build_sft_exports", source)
        self.assertNotIn("Stage2", formal_source)

    def test_parent_campaign_remains_execution_locked(self):
        campaign = load(CAMPAIGN)
        self.assertFalse(campaign["activation_authority"])
        self.assertFalse(campaign["execution_activation_authority"])
        phase = next(row for row in campaign["phases"] if row["phase"] == "M1-D")
        self.assertFalse(phase["activation_authority"])


if __name__ == "__main__":
    unittest.main()
