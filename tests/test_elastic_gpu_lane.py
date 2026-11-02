import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'scripts/openevo_webshop/elastic_gpu_lane.py'
BACKEND = ROOT / 'scripts/openevo_webshop/ceiling1_ray_placement_backend_202609042025.py'

hspec = importlib.util.spec_from_file_location('elastic_gpu_lane', HELPER)
helper = importlib.util.module_from_spec(hspec)
hspec.loader.exec_module(helper)

bspec = importlib.util.spec_from_file_location('elastic_ray_backend', BACKEND)
backend = importlib.util.module_from_spec(bspec)
bspec.loader.exec_module(backend)


def qualified_cfg():
    uuids = {str(i): f'u{i}' for i in [2, 4, 5, 6, 7]}
    return {
        'managed_gpus': [2, 4, 5, 6, 7],
        'managed_gpu_uuids': uuids,
        'global_unallocated_headroom_mib': 4096,
        'gpu_binding_contract': {
            'schema': 'openevo.elastic-gpu-binding.202609181020',
            'mode': helper.QUALIFIED_RESOURCE_CLASS,
            'candidate_gpu_indices': [2, 4, 5, 6, 7],
            'candidate_gpu_uuids': uuids,
            'resource_class': {
                'gpu_name': 'NVIDIA GeForce RTX 5090',
                'min_total_mib': 32000,
            },
            'selection_receipt_role': 'execution_evidence_not_protocol_identity',
            'paper_final_requires_exact_device_set': True,
        },
    }


class ElasticGpuLaneTest(unittest.TestCase):
    def test_legacy_config_remains_exact_device_mode(self):
        cfg = {
            'managed_gpus': [4, 5],
            'managed_gpu_uuids': {'4': 'u4', '5': 'u5'},
        }
        contract = helper.binding_contract(cfg)
        self.assertEqual(contract['mode'], helper.EXACT_DEVICE_SET)
        self.assertTrue(contract['legacy_default'])
        self.assertEqual(
            contract['selection_receipt_role'],
            'scientific_identity_and_execution_evidence',
        )

    def test_explicit_exact_contract_gets_safe_receipt_defaults(self):
        cfg = {
            'managed_gpus': [4, 5],
            'managed_gpu_uuids': {'4': 'u4', '5': 'u5'},
            'gpu_binding_contract': {
                'schema': 'openevo.elastic-gpu-binding.202609181020',
                'mode': helper.EXACT_DEVICE_SET,
                'candidate_gpu_indices': [4, 5],
                'candidate_gpu_uuids': {'4': 'u4', '5': 'u5'},
            },
        }
        contract = helper.binding_contract(cfg)
        self.assertEqual(contract['selection_receipt_role'], 'scientific_identity_and_execution_evidence')
        self.assertTrue(contract['paper_final_requires_exact_device_set'])

    def test_qualified_pool_can_replace_busy_card_without_protocol_rebind(self):
        cfg = qualified_cfg()
        profile = {'vram_request_mib': 7000}
        rows = {
            2: {'uuid': 'u2', 'name': 'NVIDIA GeForce RTX 5090', 'total_mib': 32607, 'free_mib': 32500, 'used_mib': 2, 'util_pct': 0},
            4: {'uuid': 'u4', 'name': 'NVIDIA GeForce RTX 5090', 'total_mib': 32607, 'free_mib': 32400, 'used_mib': 2, 'util_pct': 0},
            5: {'uuid': 'u5', 'name': 'NVIDIA GeForce RTX 5090', 'total_mib': 32607, 'free_mib': 32300, 'used_mib': 2, 'util_pct': 0},
            6: {'uuid': 'u6', 'name': 'NVIDIA GeForce RTX 5090', 'total_mib': 32607, 'free_mib': 6000, 'used_mib': 26364, 'util_pct': 23},
            7: {'uuid': 'u7', 'name': 'NVIDIA GeForce RTX 5090', 'total_mib': 32607, 'free_mib': 32200, 'used_mib': 2, 'util_pct': 0},
        }
        eligible, observed, need = backend.physically_eligible_gpus(cfg, profile, rows)
        self.assertEqual(need, 11096)
        self.assertEqual(eligible, [2, 4, 5, 7])
        self.assertNotIn(6, eligible)

    def test_qualified_pool_rejects_wrong_hardware_class(self):
        cfg = qualified_cfg()
        profile = {'vram_request_mib': 7000}
        rows = {
            2: {'uuid': 'u2', 'name': 'NVIDIA GeForce RTX 4090', 'total_mib': 32607, 'free_mib': 32500, 'used_mib': 2, 'util_pct': 0},
        }
        eligible, _, _ = backend.physically_eligible_gpus(cfg, profile, rows)
        self.assertEqual(eligible, [])

    def test_binding_evidence_keeps_exact_uuid_as_execution_receipt(self):
        cfg = qualified_cfg()
        rows = {
            i: {'uuid': f'u{i}', 'name': 'NVIDIA GeForce RTX 5090', 'total_mib': 32607}
            for i in [2, 4, 5, 7]
        }
        evidence = helper.binding_evidence(cfg, [2, 4, 5, 7], rows, 123.5)
        self.assertEqual(evidence['mode'], helper.QUALIFIED_RESOURCE_CLASS)
        self.assertEqual(
            evidence['selection_receipt_role'],
            'execution_evidence_not_protocol_identity',
        )
        self.assertTrue(evidence['paper_final_requires_exact_device_set'])
        self.assertEqual(
            [d['uuid'] for d in evidence['selected_devices']],
            ['u2', 'u4', 'u5', 'u7'],
        )

    def test_qualified_contract_fails_if_candidate_pool_and_ray_pool_differ(self):
        cfg = qualified_cfg()
        cfg['gpu_binding_contract']['candidate_gpu_indices'] = [4, 5, 6, 7]
        with self.assertRaisesRegex(RuntimeError, 'candidate set must equal Ray managed_gpus'):
            helper.binding_contract(cfg)

    def test_ray_queue_age_does_not_expire_launch_authority(self):
        receipt = helper.queue_gate_freshness(
            helper.RAY_CAPACITY_SHARED_POOL,
            observed_at_epoch=100.0,
            now_epoch=900.0,
        )
        self.assertEqual(receipt['status'], 'DEFER_TO_SELECTION_AND_LAUNCH_LIVE_GATE')
        self.assertFalse(receipt['queue_age_blocks_launch'])
        self.assertEqual(
            receipt['launch_freshness_authority'],
            'RAY_SELECTION_AND_LAUNCH_LIVE_GATE',
        )

    def test_idle_snapshot_queue_gate_still_fails_closed_when_stale(self):
        with self.assertRaisesRegex(RuntimeError, 'idle-snapshot GPU gate is stale'):
            helper.queue_gate_freshness(
                helper.IDLE_GATE_SHARED_POOL,
                observed_at_epoch=100.0,
                now_epoch=161.0,
                max_idle_snapshot_age_seconds=60.0,
            )
        receipt = helper.queue_gate_freshness(
            helper.IDLE_GATE_SHARED_POOL,
            observed_at_epoch=100.0,
            now_epoch=159.0,
            max_idle_snapshot_age_seconds=60.0,
        )
        self.assertTrue(receipt['queue_age_blocks_launch'])
        self.assertEqual(receipt['status'], 'PASS')

    def test_hardware_exposure_audit_is_descriptive_only(self):
        receipts = [
            {'arm': 'off', 'selected_gpu_uuids': ['u2', 'u4', 'u5', 'u7']},
            {'arm': 'on', 'selected_gpu_uuids': ['u2', 'u4', 'u5', 'u7']},
            {'arm': 'off', 'selected_gpu_uuids': ['u2', 'u4', 'u5', 'u7']},
            {'arm': 'on', 'selected_gpu_uuids': ['u4', 'u5', 'u6', 'u7']},
        ]
        audit = helper.hardware_exposure_audit(receipts)
        self.assertEqual(audit['status'], 'DESCRIPTIVE_ONLY')
        self.assertFalse(audit['blocks_progression'])
        self.assertEqual(audit['max_per_uuid_count_delta_across_arms'], 1)

    def test_hardware_exposure_audit_accepts_launch_receipt_wrapper(self):
        receipts = [
            {
                'schema': 'openevo.ceiling1-ray-launch.202609042025',
                'selection': {
                    'arm': 'off',
                    'selected_gpu_uuids': ['u2', 'u4', 'u5', 'u7'],
                },
            },
            {
                'schema': 'openevo.ceiling1-ray-launch.202609042025',
                'selection': {
                    'arm': 'on',
                    'selected_gpu_uuids': ['u2', 'u4', 'u5', 'u7'],
                },
            },
        ]
        audit = helper.hardware_exposure_audit(receipts)
        self.assertEqual(audit['receipt_count'], 2)
        self.assertEqual(set(audit['arms']), {'off', 'on'})
        self.assertEqual(audit['max_per_uuid_count_delta_across_arms'], 0)

    def test_zero_formal_qualification_cache_binds_runtime_and_candidate_pool(self):
        cfg = qualified_cfg()
        rows = {
            i: {'uuid': f'u{i}', 'name': 'NVIDIA GeForce RTX 5090', 'total_mib': 32607}
            for i in [2, 4, 5, 6, 7]
        }
        cache = helper.build_qualification_cache(cfg, rows, 'runtime-A', 10.0)
        self.assertEqual(cache['status'], 'PASS')
        self.assertEqual(cache['formal_task_consumption'], 0)
        self.assertEqual(cache['model_calls'], 0)
        self.assertEqual(cache['teacher_calls'], 0)
        self.assertEqual(cache['training_calls'], 0)
        helper.validate_qualification_cache(cfg, cache, 'runtime-A')
        with self.assertRaisesRegex(RuntimeError, 'runtime drift'):
            helper.validate_qualification_cache(cfg, cache, 'runtime-B')

    def test_launch_time_live_gate_rechecks_resource_class(self):
        cfg = qualified_cfg()
        profile = {'vram_request_mib': 7000}
        old = backend.gpu_rows
        try:
            backend.gpu_rows = lambda: {
                2: {
                    'uuid': 'u2',
                    'name': 'NVIDIA GeForce RTX 5090',
                    'total_mib': 32607,
                    'free_mib': 32000,
                    'used_mib': 607,
                    'util_pct': 0,
                }
            }
            gate = backend.live_gate(cfg, profile, [2])
            self.assertTrue(gate['2']['pass'])
            backend.gpu_rows = lambda: {
                2: {
                    'uuid': 'u2',
                    'name': 'NVIDIA GeForce RTX 4090',
                    'total_mib': 32607,
                    'free_mib': 32000,
                    'used_mib': 607,
                    'util_pct': 0,
                }
            }
            with self.assertRaisesRegex(RuntimeError, 'resource-class qualification drift'):
                backend.live_gate(cfg, profile, [2])
        finally:
            backend.gpu_rows = old


if __name__ == '__main__':
    unittest.main()
