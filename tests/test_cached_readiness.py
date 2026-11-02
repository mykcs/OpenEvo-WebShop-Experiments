"""CPU cache-admission contract tests, exercising training and eval callers."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from tests import test_native_eval_admission as eval_fixture
pool=eval_fixture.pool
class CacheAdmission(unittest.TestCase):
 def setUp(self):
  self.fx=eval_fixture.EvalAdmission();self.fx.setUp();self.addCleanup(self.fx.doCleanups)
  self.task=self.fx.task;self.root=self.fx.root;self.pins=self.fx.pins
  self.request={'runtime_fingerprint_sha256':'sha256:'+'a'*64,'frozen_request':'exact'}
  key=hashlib.sha256(json.dumps(self.request,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  self.cache={'schema':'openevo.runtime-qualification.v1','status':'PASS','failures':[],'cache_key':key,'request':self.request,'formal_task_consumption_allowed':False}
  self.path=self.root/'cache.json';self.write_cache()
  dec={'reason':'exact_request_match','changed_request_fields':[],'environment_mutation_allowed':False}
  self.ready=copy.deepcopy(self.fx.docs['readiness_receipt']);self.ready.update(qualification_cache_key=key,qualification_receipt_path=str(self.path),qualification_receipt_sha256='sha256:'+pool.digest(self.path),runtime_fingerprint_sha256=self.request['runtime_fingerprint_sha256'],runtime_qualification_decision=dec)
  self.ready['stages']=[{'stage':'authority','verdict':'PASS'},{'stage':'runtime-qualification','verdict':'CACHE_HIT','detail':{'cache_key':key[:16],'fingerprint':self.request['runtime_fingerprint_sha256'][:23],'qualification_decision':copy.deepcopy(dec)}}]
 def write_cache(self):
  self.path.write_text(json.dumps(self.cache));self.pins[str(self.path)]=pool.digest(self.path)
 def admit(self,training=False):
  if not training:
   self.fx.write('readiness_receipt',self.ready);return pool.native_science_admission(self.task)
  science=self.root/'science';science.mkdir(exist_ok=True);p=science/'execution-state.json'
  p.write_text(json.dumps({'schema':'openevo.ceiling1-stage2-execution-state.v1','status':'RUNNING','next_round':134,'rounds_complete':list(range(134)),'rollouts_consumed':17152,'final_panel_access_count':0,'external_teacher_calls':0}))
  ready=self.root/'training-ready.json';ready.write_text(json.dumps(self.ready));pins=dict(self.pins);pins[str(p)]=pool.digest(p);pins[str(ready)]=pool.digest(ready)
  return pool.native_science_admission({'id':'q09-cache-training','native_campaign':{'readiness_receipt':str(ready),'science_root':str(science),'next_round':134,'rollouts_consumed':17152},'source_files':pins})
 def reject(self):
  for training in [False,True]:
   with self.subTest(training=training),self.assertRaises(ValueError):self.admit(training)
 def test_exact_cache_passes_both_adapters(self):self.admit();self.admit(True)
 def test_other_stage_cache_rejected(self):self.ready['stages'][1]['stage']='authority';self.reject()
 def test_missing_cache_rejected(self):self.path.unlink();self.reject()
 def test_unbound_cache_rejected(self):self.pins.pop(str(self.path));self.reject()
 def test_drifted_cache_rejected(self):self.path.write_text('{}');self.reject()
 def test_nonpass_cache_rejected_even_repinned(self):
  self.cache['status']='FAIL';self.write_cache();self.ready['qualification_receipt_sha256']='sha256:'+pool.digest(self.path);self.reject()
 def test_failed_checks_rejected_even_repinned(self):
  self.cache['failures']=['failure'];self.write_cache();self.ready['qualification_receipt_sha256']='sha256:'+pool.digest(self.path);self.reject()
 def test_changed_request_rejected(self):self.ready['runtime_qualification_decision']['changed_request_fields']=['image'];self.reject()
 def test_mutation_allowed_rejected(self):self.ready['runtime_qualification_decision']['environment_mutation_allowed']=True;self.reject()
 def test_wrong_decision_rejected(self):self.ready['runtime_qualification_decision']['reason']='request_changed';self.reject()
 def test_stage_decision_drift_rejected(self):self.ready['stages'][1]['detail']['qualification_decision']['reason']='changed';self.reject()
 def test_full_cache_key_drift_rejected(self):self.ready['qualification_cache_key']='b'*64;self.reject()
 def test_short_key_drift_rejected(self):self.ready['stages'][1]['detail']['cache_key']='b'*16;self.reject()
 def test_fingerprint_drift_rejected(self):self.ready['runtime_fingerprint_sha256']='sha256:'+'b'*64;self.reject()
 def test_request_bytes_key_drift_rejected(self):
  self.cache['request']['frozen_request']='changed';self.write_cache();self.ready['qualification_receipt_sha256']='sha256:'+pool.digest(self.path);self.reject()
 def test_cache_not_a_formal_grant(self):
  self.cache['formal_task_consumption_allowed']=True;self.write_cache();self.ready['qualification_receipt_sha256']='sha256:'+pool.digest(self.path);self.reject()
 def test_missing_full_hash_rejected(self):self.ready.pop('qualification_receipt_sha256');self.reject()
 def test_malformed_stages_rejected(self):self.ready['stages']=[None];self.reject()
if __name__=='__main__':unittest.main()
