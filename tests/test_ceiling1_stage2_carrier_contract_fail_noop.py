from __future__ import annotations
import json, tempfile, unittest
from pathlib import Path
from scripts.openevo_webshop.ceiling1_stage2_arm_controller import (
    REFLECTOR_CONTRACT_FAIL_NOOP_REASON, build_reflector_contract_fail_noop,
    reflector_attempt_identity, sha,
)

class ContractFailNoopTest(unittest.TestCase):
    def fixture(self):
        td=tempfile.TemporaryDirectory(); root=Path(td.name); d=root/'text-memory'; d.mkdir()
        evidence=root/'evidence.json'; prior=root/'prior.md'; attempt=d/'model-call-attempt.json'
        evidence.write_text(json.dumps({'x':1})); prior.write_text('# Memory\n## Reusable Knowledge\n- Knowledge: prior.\n')
        state='state-1'; kind='text_memory'; r=5
        ident=reflector_attempt_identity(kind,r,state,evidence,prior,None,None,None)
        attempt.write_text(json.dumps(ident,sort_keys=True,indent=2)+'\n')
        items=[]
        for label,text in [('primary','bad click'),('repair','bad buy')]:
            p=d/f'rejected-{label}.md'; p.write_text(text+'\n')
            items.append({'attempt':label,'output_tokens':3,'saturated':False,'artifact_contract_valid':False,'candidate_sha256':sha(p)})
        rejected={'schema':'openevo.ceiling1-stage2-reflector-rejected-candidates.v1','kind':kind,'round_index':r,'state_id':state,'attempts':items}
        (d/'rejected-candidates.json').write_text(json.dumps(rejected,sort_keys=True,indent=2)+'\n')
        return td,root,d,evidence,prior,attempt,state,kind,r
    def test_two_contract_failures_become_keep_prior_noop(self):
        td,root,d,evidence,prior,attempt,state,kind,r=self.fixture()
        with td:
            out=build_reflector_contract_fail_noop(kind,r,state,evidence,prior,None,None,None,d,attempt)
            self.assertEqual(out['status'],'NOOP'); self.assertEqual(out['reason'],REFLECTOR_CONTRACT_FAIL_NOOP_REASON)
            self.assertEqual(out['output_sha256'],sha(prior)); self.assertEqual(out['model_calls_consumed'],2)
            self.assertEqual(out['primary_calls'],1); self.assertEqual(out['repair_calls'],1)
    def test_single_failure_does_not_convert(self):
        td,root,d,evidence,prior,attempt,state,kind,r=self.fixture()
        with td:
            x=json.loads((d/'rejected-candidates.json').read_text()); x['attempts']=x['attempts'][:1]; (d/'rejected-candidates.json').write_text(json.dumps(x))
            self.assertIsNone(build_reflector_contract_fail_noop(kind,r,state,evidence,prior,None,None,None,d,attempt))
    def test_valid_repair_does_not_convert(self):
        td,root,d,evidence,prior,attempt,state,kind,r=self.fixture()
        with td:
            x=json.loads((d/'rejected-candidates.json').read_text()); x['attempts'][1]['artifact_contract_valid']=True; (d/'rejected-candidates.json').write_text(json.dumps(x))
            self.assertIsNone(build_reflector_contract_fail_noop(kind,r,state,evidence,prior,None,None,None,d,attempt))

    def test_harness201_attempt_contract_is_identity_bound(self):
        td,root,d,evidence,prior,attempt,state,kind,r=self.fixture()
        with td:
            contract='harness201-runtime-memory-v1'
            ident=reflector_attempt_identity(kind,r,state,evidence,prior,None,None,None,contract)
            attempt.write_text(json.dumps(ident,sort_keys=True,indent=2)+'\n')
            out=build_reflector_contract_fail_noop(kind,r,state,evidence,prior,None,None,None,d,attempt,contract)
            self.assertEqual(out['status'],'NOOP')
            with self.assertRaisesRegex(RuntimeError,'attempt identity drift'):
                build_reflector_contract_fail_noop(kind,r,state,evidence,prior,None,None,None,d,attempt,'legacy-authoring-v1')

    def test_rejected_hash_drift_remains_blocked(self):
        td,root,d,evidence,prior,attempt,state,kind,r=self.fixture()
        with td:
            (d/'rejected-repair.md').write_text('mutated\n')
            with self.assertRaisesRegex(RuntimeError,'payload identity mismatch'):
                build_reflector_contract_fail_noop(kind,r,state,evidence,prior,None,None,None,d,attempt)
    def test_attempt_identity_drift_remains_blocked(self):
        td,root,d,evidence,prior,attempt,state,kind,r=self.fixture()
        with td:
            x=json.loads(attempt.read_text()); x['state_id']='wrong'; attempt.write_text(json.dumps(x))
            with self.assertRaisesRegex(RuntimeError,'attempt identity drift'):
                build_reflector_contract_fail_noop(kind,r,state,evidence,prior,None,None,None,d,attempt)
if __name__=='__main__': unittest.main()
