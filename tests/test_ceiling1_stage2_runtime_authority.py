from pathlib import Path
import json
import unittest

ROOT = Path(__file__).resolve().parents[1]
CURRENT_AUTH = ROOT / "configs/experiment/openevo-ceiling1-stage2-runtime-authority-202609042025.json"
LEGACY_AUTH = ROOT / "configs/experiment/openevo-ceiling1-stage2-runtime-authority-v1.json"


class Stage2RuntimeAuthorityTest(unittest.TestCase):
    def test_only_timestamped_authority_is_repository_current(self):
        current = json.loads(CURRENT_AUTH.read_text())
        legacy = json.loads(LEGACY_AUTH.read_text())
        self.assertEqual(current["status"], "READY_FOR_AUDITED_NATURAL_MIGRATION")
        self.assertEqual(current["resource_backend"]["name"], "ray-202609042025")
        self.assertEqual(set(current["arms"]), {"3b", "qwen3-1p7b"})
        self.assertEqual(legacy["status"], "SUPERSEDED_HISTORY_ONLY")
        self.assertFalse(legacy["execution_authority"])
        self.assertEqual(legacy["superseded_by"]["path"], str(CURRENT_AUTH.relative_to(ROOT)))

    def test_current_launcher_only_routes_to_timestamped_ray(self):
        launcher = (ROOT / "scripts/openevo_webshop/launch_ceiling1_stage2_current_arm_202609042025.sh").read_text()
        driver = (ROOT / "scripts/openevo_webshop/ceiling1_stage2_current_arm_driver_202609042025.sh").read_text()
        self.assertIn("scripts/oe", launcher)
        self.assertIn(" host ", launcher.replace("\\\n", " "))
        self.assertIn("run_ceiling1_stage2_ray_supervised_202609042025.py", driver)
        self.assertIn("CEILING1_STAGE2_RAY_MIGRATION", driver)

    def test_generic_current_named_entrypoints_fail_closed(self):
        for rel in (
            "scripts/openevo_webshop/launch_ceiling1_stage2_vnext_current_arm.sh",
            "scripts/openevo_webshop/ceiling1_stage2_vnext_current_arm_driver.sh",
            "scripts/openevo_webshop/run_ceiling1_stage2_vnext_ray_v1.py",
        ):
            text = (ROOT / rel).read_text()
            self.assertIn("SUPERSEDED_HISTORY_ONLY", text)
            self.assertIn("NO-GO", text)

    def test_shared_controller_fans_out_before_barrier(self):
        source = (ROOT / "scripts/openevo_webshop/run_ceiling1_stage2_arm_202609042025.py").read_text()
        self.assertIn("ThreadPoolExecutor(max_workers=4", source)
        for token in ("'text_memory'", "'skill_bundle'", "'agent_system'"):
            self.assertIn("pool.submit(self.reflect", source)
            self.assertIn(token, source)
        self.assertIn("pool.submit(self.run_sd", source)
        self.assertLess(source.index("new_mem=f_mem.result()"), source.index("candidate_probe=self.candidate_probe"))


if __name__ == "__main__":
    unittest.main()
