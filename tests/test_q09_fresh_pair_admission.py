"""At-most-once, separate-budget fresh-pair admission; no model or server access."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tests import test_native_eval_admission as fixtures
from scripts.openevo_runtime import native_advisor_task_pool as pool


class FreshPairAdmission(unittest.TestCase):
    write = fixtures.EvalAdmission.write
    ref = fixtures.EvalAdmission.ref

    def setUp(self):
        fixtures.EvalAdmission.setUp(self)
        self.task['native_campaign'] = {'reservation': {'enabled': False, 'expires_at': None}}
        cfg = self.task['native_eval']
        arm = 'r159' if self._testMethodName == 'test_trained_fresh_endpoint_passes_full_chain' else 'r119'
        cfg.pop('zero_consumption_recovery')
        cfg.update(adapter='q09-fixed-checkpoint-fresh64-v1', fresh_endpoint=arm,
                   source_sha=pool.Q09_FRESH_PAIR_SOURCE)
        self.task['id'] = 'q09-fixed-checkpoint-fresh-' + arm + '-gpu4-202610282120'
        ep = self.docs['endpoint']
        identity = pool.Q09_FRESH_PAIR_ENDPOINTS[arm]
        ep.update(endpoint_id=identity[0], evaluation_stage=identity[1], treatment=identity[2],
                  stream_id=identity[3], memory_sha256='9a88356c084783e003d80db90a918be265d1ce481590d769453993d0e21dea6d',
                  skill_sha256='cef31b897dbe4796d61710c13ab11630f64f043bd7df651d1faa12980163b21a',
                  agent_system_sha256='7a87fa050c526907b1999ba4c0ef9c127808e4983a1f36dc646a12c6ab75caf4')
        panel = self.docs['panel']; panel['status'] = 'FROZEN_PREOUTCOME'; panel.pop('panel_sha256')
        panel['panel_sha256'] = hashlib.sha256(json.dumps(panel, ensure_ascii=False,
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.write('panel', panel); ep['panel_sha256'] = panel['panel_sha256']; self.write('endpoint', ep)
        profile = self.docs['profile']; profile['run_id'] = ep['endpoint_id']
        profile['source']['experiment']['revision'] = pool.Q09_FRESH_PAIR_SOURCE
        self.write('profile', profile)
        act = self.docs['activation']; act.update(stage=identity[1], confirmation=True,
            endpoint_id=identity[0], endpoint_sha256=self.ref('endpoint')['sha256'],
            profile_sha256=self.ref('profile')['sha256'])
        act['source']['head'] = pool.Q09_FRESH_PAIR_SOURCE; act['budget']['training_rollouts'] = 0
        act['input_files'][str(self.root/'new-inputs/metadata/endpoint.json')] = self.ref('endpoint')['sha256']
        act['input_files'][str(self.root/'new-inputs/metadata/development-panel.json')] = self.ref('panel')['sha256']
        self.write('activation', act)
        cfg['science_output'] = str(self.root/'new-runs'/identity[0]/'endpoint')
        parent = Path(cfg['science_output']).parent; parent.mkdir(parents=True); parent.chmod(0o750)
        st = parent.stat()
        member = {k:self.ref(k) for k in ('endpoint', 'activation', 'profile')}
        member.update(state_id=ep['state_id'], adapter_sha256=ep['adapter_sha256'],
            endpoint_id=identity[0], dispatch_dir=cfg['dispatch_dir'], science_output=cfg['science_output'],
            prepared_empty_run_root={'path':str(parent), 'device':st.st_dev, 'inode':st.st_ino,
                                     'uid':1001, 'gid':1001, 'mode':0o750})
        analysis = {'comparisons':1, 'bootstrap_resamples':50000, 'random_seed':2026102821}
        budget = {**act['budget'], 'new_evaluation_episodes':128}
        self.write('exclusion_manifest', {'fresh_outcomes_seen':False})
        contract = {'status':'PROSPECTIVE_FROZEN_BEFORE_FRESH_COLLECTION', 'budget':budget,
            'endpoints':{arm:member}, 'confirmation_panel':self.ref('panel'), 'analysis':analysis,
            'exposure':{'checkpoint_pair_selected_after_development':True,
                        'development_result_seen_before_design':True, 'fresh_outcomes_seen':False}}
        self.write('contract', contract)
        proof = {**contract, 'schema':'openevo.q09-fresh-pair-activation.v1',
            'status':'CPU_PREPARED_NOT_ACTIVATED', 'formal_task_consumption_allowed':False,
            'formal_starts_verified_prior':0, 'output_absent_verified':True,
            'gpu_uuids':[pool.GPU_UUIDS[3]], 'contract':self.ref('contract'),
            'exclusion_manifest':self.ref('exclusion_manifest')}
        self.write('fresh_activation_proof', proof)
        release = self.docs['release']
        release['execution_readiness']['profile'] = self.ref('profile')
        release['execution_readiness']['frozen_scientific_artifacts'] = [dict(role=role, **self.ref(name))
            for role,name in [('endpoint','endpoint'),('confirmation_panel','panel'),
                              ('activation','activation'),('fresh_activation','fresh_activation_proof')]]
        self.write('release', release)
        ready = self.docs['readiness_receipt']; ready.update(release_receipt=self.ref('release'), profile=self.ref('profile'))
        self.write('readiness_receipt', ready)
        router = self.docs['current_campaign']; router['release_receipt_sha256'] = self.ref('release')['sha256']
        self.write('current_campaign', router)
        original_stat = Path.stat
        def owned_stat(path, *args, **kwargs):
            value = original_stat(path, *args, **kwargs)
            if path == parent:
                return SimpleNamespace(st_mode=value.st_mode, st_dev=value.st_dev, st_ino=value.st_ino,
                                       st_uid=1001, st_gid=1001)
            return value
        for mocker in (patch.object(pool, 'Q09_FRESH_PAIR_PROOF_SHA256', self.ref('fresh_activation_proof')['sha256']),
                       patch.object(pool, 'Q09_FRESH_PAIR_PANEL', panel['panel_sha256']),
                       patch.object(Path, 'stat', owned_stat), patch.object(pool.time, 'time', return_value=1793220000)):
            mocker.start(); self.addCleanup(mocker.stop)

    def admit(self):
        pool.native_science_admission(self.task)

    def test_exact_separate_fresh_endpoint_passes_full_chain(self):
        self.admit()

    def test_trained_fresh_endpoint_passes_full_chain(self):
        self.admit()

    def test_legacy_adapter_cannot_admit_a_fresh_panel(self):
        self.task['native_eval']['adapter'] = 'q09-stage-a-common-start-dev64-v1'
        with self.assertRaises(ValueError): self.admit()

    def test_unknown_arm_or_extra_claim_is_denied(self):
        self.task['id'] += '-copy'
        with self.assertRaisesRegex(ValueError, 'EXACT_SCOPE'): self.admit()

    def test_proof_repin_does_not_authorize_a_new_plan(self):
        proof = copy.deepcopy(self.docs['fresh_activation_proof']); proof['analysis']['comparisons'] = 2
        self.write('fresh_activation_proof', proof)
        with self.assertRaises(ValueError): self.admit()

    def test_any_current_endpoint_marker_blocks_replay(self):
        out = Path(self.task['native_eval']['science_output']); out.mkdir(); (out/'inflight-000.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'PRIOR_START_OR_OUTPUT'): self.admit()

    def test_genuine_formal_start_blocks_replay(self):
        out = Path(self.task['native_eval']['dispatch_dir']); out.mkdir(); (out/'formal-start-0001.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'PRIOR_START_OR_OUTPUT'): self.admit()

    def test_deadline_and_reservations_are_not_optional(self):
        with patch.object(pool.time, 'time', return_value=1793235600):
            with self.assertRaisesRegex(ValueError, 'ACTIVATION_SCOPE'): self.admit()
        self.task['native_campaign']['reservation']['enabled'] = True
        with self.assertRaisesRegex(ValueError, 'ACTIVATION_SCOPE'): self.admit()
