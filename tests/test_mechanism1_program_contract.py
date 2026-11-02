import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / 'configs/experiment/programs/openevo-mechanism-1.0.json'
ROUTER = ROOT / 'configs/experiment/campaigns/current-mechanism1-autonomous.json'
REGISTRY = ROOT / 'configs/experiment/openevo-mechanism1-ray-profile-registry-202609052200.json'
BACKEND = ROOT / 'scripts/openevo_webshop/ceiling1_ray_placement_backend_202609042025.py'

spec = importlib.util.spec_from_file_location('mechanism1_ray_backend', BACKEND)
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


class Mechanism1ProgramContractTest(unittest.TestCase):
    def setUp(self):
        self.program = json.loads(PROGRAM.read_text())
        self.router = json.loads(ROUTER.read_text())
        self.registry = json.loads(REGISTRY.read_text())
        self.passports = {
            Path(path).name: json.loads((ROOT / path).read_text())
            for path in self.program['experiment_passports']
        }

    def test_program_is_new_science_but_execution_locked(self):
        self.assertEqual(self.program['program_id'], 'openevo-mechanism-1.0-202609052200')
        self.assertFalse(self.program['activation_authority'])
        self.assertFalse(self.router['execution_activation_authority'])
        self.assertIn('§2.3', self.program['engineering_platform_authority'])
        self.assertIn('immutable source evidence', self.program['ceiling_relationship'])
    def test_seed_probe_is_bounded_and_has_zero_stage2(self):
        p = self.passports['openevo-mechanism1-seed-stage1-init-probe-3b-202609052200.json']
        self.assertEqual(p['budget']['formal_trajectories'], 1440)
        self.assertEqual(p['budget']['minimax_parse_ok_target'], 1440)
        self.assertEqual(p['budget']['seed_stage2_rollouts'], 0)
        self.assertEqual(p['budget']['final_panel_tasks'], 0)
        self.assertIn('Stage2', p['hard_stop'])

    def test_guided_microtrial_has_fixed_conditional_budget(self):
        p = self.passports['openevo-mechanism1-taskvector-guided-microtrial-202609052200.json']
        self.assertEqual(p['budget']['total_rollouts'], 1536)
        self.assertEqual(p['budget']['rollouts_per_arm'], 512)
        self.assertEqual(len(p['arms']), 3)
        self.assertTrue(any('DO_NOT_EXECUTE' in rule for rule in p['go_gate_priority']))
        self.assertEqual(p['final_panel_access'], 0)

    def test_taskvector_is_counterfactual_not_endpoint_reconstruction(self):
        p = self.passports['openevo-mechanism1-taskvector-causal-transplant-202609052200.json']
        src = p['source_selection']
        self.assertEqual(src['source_vector'], 'theta_100pct - theta_75pct')
        self.assertEqual(src['transplant_target'], 'theta_50pct')
        self.assertIn('theta_50pct + 1.0 * tau_late', p['interventions'])
        self.assertEqual(p['budget']['max_closed_loop_episodes'], 576)

    def test_theme_importance_requires_random_controls(self):
        p = self.passports['openevo-mechanism1-parameter-theme-knockout-202609052200.json']
        self.assertIn('beyond norm-matched random controls', p['causal_importance_definition'])
        self.assertEqual(p['budget']['max_closed_loop_episodes'], 448)
        self.assertEqual(p['final_panel_access'], 0)
    def test_registry_uses_secondary_spare_without_renaming_science_arms(self):
        profiles = self.registry['profiles']
        self.assertTrue(profiles)
        self.assertTrue(all(p['slot_class'] == 'secondary_spare' for p in profiles.values()))
        self.assertEqual(profiles['mechanism1_seed_stage1_3b']['arm'], 'mechanism1-seed-stage1-3b')
        self.assertEqual(
            set(self.registry['compatible_base_config_sha256']),
            {
                'fc680c240c81afcf56497f0db81734172fa38d7ff3c752de64b84048118f0b48',
                '869793ae8e3694f352fca579b2d5d3562c21201a18ec5210678f1101be22157e',
            },
        )

    def test_external_registry_maps_to_q17_resource_lane_only(self):
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            cfg = t / 'base.json'
            cfg.write_text(json.dumps({'profiles': {'legacy3': {'arm': '3b', 'vram_request_mib': 9}}}))
            cfgsha = backend.sha(cfg)
            reg = t / 'registry.json'
            reg.write_text(json.dumps({
                'schema': backend.PROFILE_REGISTRY_SCHEMA,
                'compatible_base_config_sha256': [cfgsha],
                'profiles': {'new': {'arm': 'seed3b', 'slot_class': 'secondary_spare', 'vram_request_mib': 7}},
            }))
            loaded = backend.load_profile_registry(reg, cfgsha)
            external = backend.profile(json.loads(cfg.read_text()), 'new', loaded)
            bundle = backend.bundle_for(external)
            self.assertEqual(external['arm'], 'seed3b')
            self.assertEqual(bundle['arm_q17_slot'], 1.0)
            self.assertNotIn('arm_3b_slot', bundle)

    def test_registry_cannot_shadow_base_or_bind_wrong_config(self):
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            cfg = t / 'base.json'; cfg.write_text(json.dumps({'profiles': {'same': {'arm': '3b', 'vram_request_mib': 9}}}))
            cfgsha = backend.sha(cfg)
            reg = t / 'registry.json'
            reg.write_text(json.dumps({'schema': backend.PROFILE_REGISTRY_SCHEMA, 'compatible_base_config_sha256': [cfgsha], 'profiles': {'same': {'arm': 'other', 'slot_class': 'secondary_spare', 'vram_request_mib': 7}}}))
            loaded = backend.load_profile_registry(reg, cfgsha)
            with self.assertRaisesRegex(RuntimeError, 'may not shadow'):
                backend.profile(json.loads(cfg.read_text()), 'same', loaded)
            with self.assertRaisesRegex(RuntimeError, 'incompatibility'):
                backend.load_profile_registry(reg, '0' * 64)


if __name__ == '__main__':
    unittest.main()
