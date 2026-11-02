from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
class RayCarrierNoopAuthorityStaticTest(unittest.TestCase):
    def test_base_controller_default_denies_science_transition(self):
        s=(ROOT/'scripts/openevo_webshop/ceiling1_stage2_arm_controller.py').read_text()
        self.assertIn('self.carrier_contract_fail_noop_authorized=False',s)
        self.assertIn('carrier contract-fail NOOP requires explicit authorized science amendment',s)
    def test_current_timestamped_ray_controller_binds_authorized_amendment(self):
        s=(ROOT/'scripts/openevo_webshop/run_ceiling1_stage2_ray_202609042025.py').read_text()
        self.assertIn('openevo.ceiling1-stage2-carrier-contract-fail-noop-amendment.v1',s)  # legacy-revision-compat
        self.assertIn('self.carrier_contract_fail_noop_authorized=True',s)
        self.assertIn("'resource_only_successor':science_amendment_sha is None",s)
    def test_current_timestamped_migration_requires_whole_pool_zero_live_boundary(self):
        s=(ROOT/'scripts/openevo_webshop/prepare_ceiling1_ray_continuation_202609042025.py').read_text()
        self.assertIn("'scope':'whole-pool-zero-live-timestamp-successor'",s)
        self.assertIn("'requires_zero_live_both_arms':True",s)
        self.assertIn("'scientific_semantics_changed':False",s)
if __name__=='__main__': unittest.main()
