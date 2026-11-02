"""CPU-only proposed admission tests; no server process, CUDA or model calls."""
import copy,hashlib,importlib.util,json,tempfile,unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
from scripts.openevo_runtime import native_advisor_task_pool as pool
SOURCE_SHA='2f4adc21ed7a3de31b5de390d6487a38107badb3'
ENDPOINT_ID='q09-a-bounded-active-stream1-development-40'
def fixture_documents():
 panel={'status':'FROZEN_EXISTING_DEVELOPMENT_PANEL','tasks':[{'probe_ordinal':i,'task_identity':f'task-{i}','worker_seed':0,'paired_generation_seed':100+i} for i in range(64)]}
 panel['panel_sha256']=hashlib.sha256(json.dumps(panel,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 endpoint={'schema':'openevo.advisor.q09.endpoint-config.v1','endpoint_id':ENDPOINT_ID,'gpu_uuid':pool.GPU_UUIDS[0],'evaluation_stage':'development-40','treatment':'ACTIVE','stream_id':'q09-stream-1','backbone':'SIMPLE_BOUNDED','evaluation_contract':{'episodes':64,'automatic_retries':0},'panel_sha256':panel['panel_sha256'],'state_id':'a'*64,'adapter_sha256':'b'*64,'memory_sha256':'c'*64,'skill_sha256':'d'*64,'agent_system_sha256':'e'*64,'model_revision':'f'*40}
 profile={'schema':'openevo.webshop.run-profile.v1','workspace':'/owned/workspace','run_id':ENDPOINT_ID,'paths':{'models':'old-inputs','runs':'old-runs','cache':'old-cache','data':'frozen-data'},'command':['frozen-producer','--frozen-args'],'source':{'experiment':{'revision':SOURCE_SHA}},'image_id':'sha256:'+'1'*64}
 activation={'status':'CPU_PREPARED_NOT_LAUNCHED','endpoint_id':ENDPOINT_ID,'budget':{'new_evaluation_episodes':64,'teacher_calls':0,'final_panel_access':0,'optimizer_updates':0,'adaptive_replays':0},'source':{'head':SOURCE_SHA},'endpoint_sha256':'2'*64,'profile_sha256':'3'*64,'input_files':{'/owned/old-inputs/metadata/endpoint.json':'2'*64,'/owned/old-inputs/metadata/development-panel.json':'4'*64,'/owned/old-inputs/model/weights':'5'*64},'original_input_files':{'/owned/authority/active-s1-r159.endpoint.json':'2'*64,'/owned/source/q09-a-development-panel-reuse.json':'4'*64,'/owned/original/model/weights':'5'*64}}
 release={'schema':'openevo.experiment-release.v1','status':'launch-authorized','formal_task_consumption_allowed':True,'campaign_id':'original-q09-stage-a','authorized_gpu_uuids':[pool.GPU_UUIDS[0]],'evaluation_episode_limit':64,'rollout_limit':64,'training_rollout_limit':0,'external_teacher_call_limit':0,'final_panel_access_limit':0,'execution_readiness':{}}
 return {'active-s1-r159.endpoint.json':endpoint,'active-s1-r159.activation.json':activation,'active-s1-r159.profile.json':profile,'development-panel.json':panel,'q09-active-s1-dev40-20261026-release.json':release}
get=lambda suffix:copy.deepcopy(fixture_documents()[suffix])

class EvalAdmission(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
  self.pins={};self.docs={}
  self.task={'id':'q09-original-unspent-eval-gpu4','gpu_uuid':pool.GPU_UUIDS[3], 'source_files':self.pins, 'native_eval':{'adapter':'q09-stage-a-dev64-v1','authority':str(self.root),'source_sha':'2f4adc21ed7a3de31b5de390d6487a38107badb3','dispatch_dir':str(self.root/'new-dispatch')}}
  cfg=self.task['native_eval']
  ep=get('active-s1-r159.endpoint.json');act=get('active-s1-r159.activation.json');prof=get('active-s1-r159.profile.json');panel=get('development-panel.json');release=get('q09-active-s1-dev40-20261026-release.json')
  prof['workspace']=str(self.root)
  self.write('original_endpoint',ep);self.write('original_profile',prof);self.write('panel',panel)
  act['endpoint_sha256']=self.ref('original_endpoint')['sha256'];act['profile_sha256']=self.ref('original_profile')['sha256']
  for key in list(act['input_files']):
   if key.endswith('/metadata/endpoint.json'):act['input_files'][key]=act['endpoint_sha256']
   if key.endswith('/metadata/development-panel.json'):act['input_files'][key]=self.ref('panel')['sha256']
  for key in list(act['original_input_files']):
   if key.endswith('/active-s1-r159.endpoint.json'):act['original_input_files'][key]=act['endpoint_sha256']
   if key.endswith('/q09-a-development-panel-reuse.json'):act['original_input_files'][key]=self.ref('panel')['sha256']
  self.write('original_activation',act)
  ep['gpu_uuid']=self.task['gpu_uuid'];self.write('endpoint',ep)
  prof['paths']['runs']='new-runs';prof['paths']['models']='new-inputs';prof['paths']['cache']='new-cache';self.write('profile',prof)
  act['endpoint_sha256']=self.pins[cfg['endpoint']];act['profile_sha256']=self.pins[cfg['profile']]
  oldstage=next(Path(k).parent.parent for k in act['input_files'] if k.endswith('/metadata/endpoint.json'))
  act['input_files']={str(self.root/'new-inputs'/Path(k).relative_to(oldstage)):v for k,v in act['input_files'].items()}
  act['input_files'][str(self.root/'new-inputs/metadata/endpoint.json')]=self.pins[cfg['endpoint']]
  oldkey=next(k for k in act['original_input_files'] if k.endswith('/active-s1-r159.endpoint.json'))
  act['original_input_files'].pop(oldkey);act['original_input_files'][str(self.root/'new-authority/active-s1-r159.endpoint.json')]=self.pins[cfg['endpoint']]
  self.write('activation',act)
  release['authorized_gpu_uuids']=[self.task['gpu_uuid']]
  release['execution_readiness']['profile']=self.ref('profile')
  release['execution_readiness']['frozen_scientific_artifacts']=[{'role':role,**self.ref(name)} for role,name in [('endpoint','endpoint'),('development_panel','panel'),('activation','activation')]]
  self.write('release',release)
  self.write('current_campaign',{'campaign_id':release['campaign_id'],'status':'launch-authorized','formal_task_consumption_allowed':True,'release_receipt':cfg['release'],'release_receipt_sha256':self.pins[cfg['release']]})
  self.write('readiness_receipt',{'schema':'openevo.execution-readiness.v1','verdict':'READY','formal_task_consumption_allowed':True,'authority':{'router_status':'launch-authorized'},'stages':[{'verdict':'PASS'}],'release_receipt':self.ref('release'),'profile':self.ref('profile')})
  failure=self.root/'old-stderr.log';failure.write_text('GPU lock busy before Formal Start\n');self.pins[str(failure)]=pool.digest(failure)
  cfg['science_output']=str(self.root/'new-runs'/ep['endpoint_id']/'endpoint')
  self.write('zero_consumption_recovery',{'schema':'openevo.q09-eval-zero-consumption-recovery.v1','status':'VERIFIED_ZERO_CONSUMPTION_RESOURCE_FAILURE','endpoint_id':ep['endpoint_id'],'new_science_output':cfg['science_output'],'formal_starts':0,'evaluation_episodes':0,'model_calls':0,'original_endpoint':self.ref('original_endpoint'),'original_activation':self.ref('original_activation'),'original_profile':self.ref('original_profile'),'prior_attempts':[{'dispatch_dir':str(self.root/'old-dispatch'),'science_output':str(self.root/'old-runs'/ep['endpoint_id']/'endpoint'),'failure_evidence':{'path':str(failure),'sha256':pool.digest(failure)}}]})
 def write(self,name,value):
  path=self.root/(name+'.json');path.write_text(json.dumps(value,sort_keys=True));self.pins[str(path)]=pool.digest(path);self.docs[name]=copy.deepcopy(value);self.task['native_eval'][name]=str(path);return path
 def ref(self,name):
  path=self.root/(name+'.json');return {'path':str(path),'sha256':pool.digest(path)}
 def admit(self):pool.native_science_admission(self.task)
 def test_exact_original_unspent_eval_passes_without_training_state(self):self.admit()
 def test_wrong_gpu_rejected(self):
  self.task['gpu_uuid']=pool.GPU_UUIDS[0]
  with self.assertRaisesRegex(ValueError,'GPU_SCOPE'):self.admit()
 def test_changed_budget_rejected_even_when_repinned(self):
  r=self.docs['release'];r['evaluation_episode_limit']=65;self.write('release',r)
  ready=self.docs['readiness_receipt'];ready['release_receipt']=self.ref('release');self.write('readiness_receipt',ready)
  router=self.docs['current_campaign'];router['release_receipt_sha256']=self.ref('release')['sha256'];self.write('current_campaign',router)
  with self.assertRaisesRegex(ValueError,'BUDGET_DRIFT'):self.admit()
 def test_missing_old_claim_zero_proof_rejected(self):
  r=self.docs['zero_consumption_recovery'];r['model_calls']=1;self.write('zero_consumption_recovery',r)
  with self.assertRaisesRegex(ValueError,'ZERO_CONSUMPTION'):self.admit()
 def test_any_old_formal_start_refuses_replay(self):
  p=self.root/'old-dispatch/control';p.mkdir(parents=True);(p/'formal-start-attempt-old.json').write_text('{}')
  with self.assertRaisesRegex(ValueError,'FORMAL_START'):self.admit()
 def test_old_paid_marker_or_any_output_refuses_replay(self):
  ep=self.docs['endpoint']['endpoint_id'];p=self.root/'old-runs'/ep/'endpoint';p.mkdir(parents=True);(p/'inflight-000.json').write_text('{}')
  with self.assertRaisesRegex(ValueError,'PAID_OR_UNCLASSIFIED'):self.admit()
 def test_empty_historical_parents_are_preserved(self):
  ep=self.docs['endpoint']['endpoint_id'];(self.root/'old-runs'/ep/'endpoint').mkdir(parents=True);self.admit();self.assertTrue((self.root/'old-runs'/ep/'endpoint').is_dir())
 def test_bytes_drift_rejected(self):
  Path(self.task['native_eval']['endpoint']).write_text('{}')
  with self.assertRaisesRegex(ValueError,'SOURCE_UNBOUND_OR_DRIFTED'):self.admit()
 def test_source_repinned_changed_model_still_rejected(self):
  ep=self.docs['endpoint'];ep['state_id']='b'*64;self.write('endpoint',ep)
  act=self.docs['activation'];act['endpoint_sha256']=self.ref('endpoint')['sha256'];self.write('activation',act)
  release=self.docs['release'];release['execution_readiness']['frozen_scientific_artifacts'][0]=dict(role='endpoint',**self.ref('endpoint'));release['execution_readiness']['frozen_scientific_artifacts'][2]=dict(role='activation',**self.ref('activation'));self.write('release',release)
  ready=self.docs['readiness_receipt'];ready['release_receipt']=self.ref('release');self.write('readiness_receipt',ready)
  router=self.docs['current_campaign'];router['release_receipt_sha256']=self.ref('release')['sha256'];self.write('current_campaign',router)
  with self.assertRaisesRegex(ValueError,'RESOURCE_SUCCESSOR_CHANGED_SCIENCE'):self.admit()
 def test_training_pause_guard_unchanged(self):
  self.task.pop('native_eval');science=self.root/'training';science.mkdir();state=science/'execution-state.json'
  state.write_text(json.dumps({'schema':'openevo.ceiling1-stage2-execution-state.v1','status':'PAUSE_FOR_DIAGNOSTIC','next_round':143,'rounds_complete':list(range(143)),'rollouts_consumed':18304,'final_panel_access_count':0,'external_teacher_calls':0}))
  self.pins[str(state)]=pool.digest(state);ready=self.root/'training-ready.json';ready.write_text(json.dumps({'schema':'openevo.execution-readiness.v1','formal_task_consumption_allowed':True,'authority':{'router_status':'launch-authorized'},'stages':[{'verdict':'PASS'}]}));self.pins[str(ready)]=pool.digest(ready)
  self.task['native_campaign']={'readiness_receipt':str(ready),'science_root':str(science),'next_round':143,'rollouts_consumed':18304}
  with self.assertRaisesRegex(ValueError,'NOT_UNSPENT_READY'):self.admit()

class CommonStartAdmission(unittest.TestCase):
 """Narrow new-budget checks plus the unchanged generic-chain regressions above."""
 def setUp(self):
  EvalAdmission.setUp(self)
  self.task['native_campaign']={'reservation':{'enabled':False,'expires_at':None}}
  cfg=self.task['native_eval'];cfg['adapter']='q09-stage-a-common-start-dev64-v1';cfg.pop('zero_consumption_recovery')
  ep=self.docs['endpoint'];ep.update(endpoint_id='q09-a-bounded-shared-start-development',evaluation_stage='common-start',treatment='SHARED_START',stream_id='shared-common-start',state_id='e5a3480ec014f230de46095b8092813730e4b7e74bc3b5e83499530c33d94770',adapter_sha256='c8ef26c3c96515495703d97148c1f4845c09814b686987f7150022099c67fd11',memory_sha256='9a88356c084783e003d80db90a918be265d1ce481590d769453993d0e21dea6d',skill_sha256='cef31b897dbe4796d61710c13ab11630f64f043bd7df651d1faa12980163b21a',agent_system_sha256='7a87fa050c526907b1999ba4c0ef9c127808e4983a1f36dc646a12c6ab75caf4',panel_sha256='269a07c8f3f4700df8a6d139894c867d4c4ecfdba2e1d957a380d836244448b6');self.write('endpoint',ep)
  prof=self.docs['profile'];prof['run_id']=ep['endpoint_id'];prof['source']['experiment']['path']='control/q09-stage-a-dev-eval-source-20261026';self.write('profile',prof)
  act=self.docs['activation'];act.update(stage='common-start',confirmation=False,endpoint_id=ep['endpoint_id'],endpoint_sha256=self.ref('endpoint')['sha256'],profile_sha256=self.ref('profile')['sha256']);act['input_files'][str(self.root/'new-inputs/metadata/endpoint.json')]=self.ref('endpoint')['sha256'];self.write('activation',act)
  cfg['science_output']=str(self.root/'new-runs'/ep['endpoint_id']/'endpoint')
  self.write('prior_baseline_audit',{'status':'NOT_EXACT_REUSE_DUE_TO_DIFFERENT_CARRIERS'})
  self.write('fresh_activation_proof',dict(schema='openevo.q09-common-start-activation.v1',status='AUTHORIZED_NEW_COMMON_START_DEVELOPMENT_64',endpoint_id=ep['endpoint_id'],budget=act['budget'],prior_baseline_disposition='NOT_EXACT_REUSE_DUE_TO_CARRIERS',science_output=cfg['science_output'],output_absent_verified=True,formal_starts_verified_prior=0,prior_baseline_audit=self.ref('prior_baseline_audit'),user_grant={'quote':'Run a new common-start development baseline on GPU4.','date':'2026-10-28'},source={'path':prof['source']['experiment']['path'],'revision':SOURCE_SHA},**{k:self.ref(k) for k in ('endpoint','activation','profile','panel')}))
 def admit(self):
  cfg=self.task['native_eval']
  def same_ref(ref,path):
   assert ref=={'path':path,'sha256':pool.digest(path)}
  def ref(value):
   path=Path(value['path'])
   if pool.digest(path)!=value['sha256'] or self.pins[str(path)]!=value['sha256']:
    raise ValueError('Q09_EVAL_REFERENCE_DRIFTED')
   return pool.read(path)
  pool._common_start_dev64_admission(self.task,cfg,self.docs['activation'],self.docs['endpoint'],self.docs['panel'],self.docs['profile'],self.docs['fresh_activation_proof'],same_ref,ref)
 def test_exact_new64_passes_without_fake_recovery(self):self.admit()
 def test_bound_prior_baseline_audit_passes(self):self.admit()
 def test_prior_baseline_audit_bytes_drift_rejected(self):
  Path(self.task['native_eval']['prior_baseline_audit']).write_text('{}')
  with self.assertRaisesRegex(ValueError,'REFERENCE_DRIFTED'):self.admit()
 def test_prior_baseline_audit_status_rejected_even_repinned(self):
  self.write('prior_baseline_audit',{'status':'EXACT_REUSABLE'})
  self.docs['fresh_activation_proof']['prior_baseline_audit']=self.ref('prior_baseline_audit')
  with self.assertRaisesRegex(ValueError,'PRIOR_BASELINE_AUDIT'):self.admit()
 def test_wrong_gpu_rejected(self):
  self.task['gpu_uuid']=pool.GPU_UUIDS[0]
  with self.assertRaisesRegex(ValueError,'SCIENTIFIC_IDENTITY'):self.admit()
 def test_each_frozen_input_rejects_repinned_drift(self):
  for key in ('state_id','adapter_sha256','memory_sha256','skill_sha256','agent_system_sha256','panel_sha256'):
   with self.subTest(key=key):
    old=self.docs['endpoint'][key];self.docs['endpoint'][key]='0'*64
    with self.assertRaisesRegex(ValueError,'SCIENTIFIC_IDENTITY'):self.admit()
    self.docs['endpoint'][key]=old
 def test_new_authority_and_exact_budget_required(self):
  for key,val in [('status','VERIFIED_ZERO_CONSUMPTION_RESOURCE_FAILURE'),('formal_starts_verified_prior',1),('output_absent_verified',False),('prior_baseline_disposition','REUSABLE'),('user_grant',{}),('budget',{})]:
   with self.subTest(key=key):
    old=self.docs['fresh_activation_proof'][key];self.docs['fresh_activation_proof'][key]=val
    with self.assertRaisesRegex(ValueError,'NEW_AUTHORIZATION'):self.admit()
    self.docs['fresh_activation_proof'][key]=old
 def test_real_caller_reservation_refused(self):
  self.task['native_campaign']['reservation']['enabled']=True
  with self.assertRaisesRegex(ValueError,'ACTIVATION_SCOPE'):self.admit()
 def test_fake_zero_recovery_refused(self):
  self.task['native_eval']['zero_consumption_recovery']='anything'
  with self.assertRaisesRegex(ValueError,'ACTIVATION_SCOPE'):self.admit()
 def test_confirmation_and_holders_refused(self):
  self.docs['activation']['confirmation']=True
  with self.assertRaisesRegex(ValueError,'ACTIVATION_SCOPE'):self.admit()
  self.docs['activation']['confirmation']=False;self.task['holders']=['holder']
  with self.assertRaisesRegex(ValueError,'ACTIVATION_SCOPE'):self.admit()
 def test_existing_output_and_formal_start_refused(self):
  cfg=self.task['native_eval'];Path(cfg['science_output']).parent.mkdir(parents=True)
  with self.assertRaisesRegex(ValueError,'OUTPUT_ALREADY_PRESENT'):self.admit()
 def test_formal_start_refused(self):
  p=Path(self.task['native_eval']['dispatch_dir']);p.mkdir();(p/'formal-start-attempt-0001.json').write_text('{}')
  with self.assertRaisesRegex(ValueError,'FORMAL_START'):self.admit()
 def prepared_root(self):
  parent=Path(self.task['native_eval']['science_output']).parent;parent.mkdir(parents=True);parent.chmod(0o750)
  st=parent.stat();identity={'path':str(parent),'device':st.st_dev,'inode':st.st_ino,'uid':1001,'gid':1001,'mode':0o750}
  self.docs['fresh_activation_proof']['prepared_empty_run_root']=identity
  original=Path.stat
  def stat(path,*args,**kw):
   value=original(path,*args,**kw)
   if path==parent:return SimpleNamespace(st_mode=value.st_mode,st_dev=value.st_dev,st_ino=value.st_ino,st_uid=1001,st_gid=1001)
   return value
  return parent,stat
 def test_exact_prepared_empty_root_passes(self):
  parent,stat=self.prepared_root()
  with patch.object(Path,'stat',stat):self.admit()
 def test_each_prepared_root_identity_field_rejected(self):
  parent,stat=self.prepared_root();proof=self.docs['fresh_activation_proof']['prepared_empty_run_root']
  for key in ('path','device','inode','uid','gid','mode'):
   old=proof[key];proof[key]=str(parent/'other') if key=='path' else old+1
   with self.subTest(key=key),patch.object(Path,'stat',stat):
    with self.assertRaisesRegex(ValueError,'EMPTY_ROOT_IDENTITY'):self.admit()
   proof[key]=old
 def test_prepared_root_must_be_empty(self):
  parent,stat=self.prepared_root();(parent/'paid-marker').write_text('consumed')
  with patch.object(Path,'stat',stat):
   with self.assertRaisesRegex(ValueError,'OUTPUT_ALREADY_PRESENT'):self.admit()
 def test_actual_ownership_not_only_proof_rejected(self):
  parent,stat=self.prepared_root()
  def wrong(path,*args,**kw):
   value=stat(path,*args,**kw)
   if path==parent:value.st_uid=0
   return value
  with patch.object(Path,'stat',wrong):
   with self.assertRaisesRegex(ValueError,'EMPTY_ROOT_IDENTITY'):self.admit()
 def test_symlink_run_root_rejected(self):
  parent=Path(self.task['native_eval']['science_output']).parent;parent.parent.mkdir(parents=True);target=self.root/'alias-target';target.mkdir();parent.symlink_to(target,target_is_directory=True)
  with self.assertRaisesRegex(ValueError,'OUTPUT_ALREADY_PRESENT'):self.admit()
 write=EvalAdmission.write
 ref=EvalAdmission.ref
if __name__=='__main__':unittest.main()
