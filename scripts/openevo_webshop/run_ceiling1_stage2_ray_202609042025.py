#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path
if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.openevo_webshop.run_ceiling1_stage2_arm_202609042025 import CurrentStage2Controller,H2011_PROMPT_SHA
from scripts.openevo_webshop.ceiling1_stage2_arm_controller import Controller,UUID,atomic_json,cmd,dstatus,load,sha,wait_container,write_once_json
from scripts.openevo_webshop.ceiling1_stage2_state_202609042025 import artifact_digest
from scripts.openevo_webshop.ceiling1_ray_placement_backend_202609042025 import RayAdmissionPaused,acquire_controller_lock
MIGRATION_SCHEMAS={'openevo.ceiling1-stage2-resource-migration.ray-continuation.202609042025','openevo.ceiling1-stage2-resource-migration.ray-v1','openevo.ceiling1-stage2-resource-migration.ray-continuation-v1'}  # legacy-revision-compat
def execution_source_lock_path(root,execution_git_sha,resource_migration_sha):
 return Path(root)/f'{execution_git_sha}--{resource_migration_sha[:16]}.json'
class RayCurrentController(CurrentStage2Controller):
 def __init__(self,a):
  a.harness_profile='201.1';a.step_seed_policy='injective-v1';a.manual_hold_release=None  # legacy-revision-compat
  pre_repo=a.repo.resolve();mig=load(a.recovery_amendment)
  if mig.get('schema') not in MIGRATION_SCHEMAS or mig.get('status') not in {'AUTHORIZED','AUTHORIZED_PENDING_SCIENCE_RECOVERY'}: raise RuntimeError('Ray continuation requires authorized resource migration')
  pre_cfg=pre_repo/'configs/experiment/openevo-ceiling1-ray-placement-backend-202609042025.json';pre_lease=pre_repo/'scripts/openevo_webshop/ceiling1_ray_placement_backend_202609042025.py';pre_ready=Path(mig['ray_pool_ready_path']);pre_root=pre_ready.parent
  if not pre_ready.is_file() or sha(pre_ready)!=mig.get('ray_pool_ready_sha256') or sha(pre_cfg)!=mig.get('ray_backend_config_sha256'): raise RuntimeError('Ray pool/config migration identity drift before controller state load')
  q=cmd(['python3',pre_lease,'--config',pre_cfg,'--state-root',pre_root,'pool-liveness'],check=False,capture=True)
  if q.returncode: raise RuntimeError('Ray pool liveness preflight failed before controller state load: '+(q.stderr.strip() or q.stdout.strip()))
  os.environ['CEILING1_STAGE2_RESOURCE_POLICY']='auto';Controller.__init__(self,a)
  self.recovery=mig
  science_amendment_sha=None
  science_path=mig.get('science_amendment_path')
  if science_path:
   sp=Path(science_path);sx=load(sp)
   if not sp.is_file() or sha(sp)!=mig.get('science_amendment_sha256'): raise RuntimeError('Ray science amendment identity drift')
   if sx.get('schema')!='openevo.ceiling1-stage2-carrier-contract-fail-noop-amendment.v1' or sx.get('status')!='AUTHORIZED': raise RuntimeError('Ray science amendment is not authorized')  # legacy-revision-compat
   if self.arm not in set(sx.get('applies_to_arms') or []) or sx.get('transition',{}).get('reason')!='NOOP_KEEP_PRIOR_TYPED_CONTRACT_REJECTED_AFTER_REPAIR': raise RuntimeError('Ray science amendment scope drift')
   if int(sx.get('unchanged',{}).get('max_primary_calls',-1))!=1 or int(sx.get('unchanged',{}).get('max_repair_calls',-1))!=1 or sx.get('unchanged',{}).get('external_teacher_calls')!=0 or sx.get('unchanged',{}).get('final_panel_access')!=0: raise RuntimeError('Ray science amendment budget drift')
   self.carrier_contract_fail_noop_authorized=True;science_amendment_sha=sha(sp)
  if mig.get('successor_execution_git_sha')!=self.execution_git_sha: raise RuntimeError('Ray successor execution SHA drift')
  if mig.get('schema')=='openevo.ceiling1-stage2-resource-migration.ray-v1':  # legacy-revision-compat
   prior=load(Path(mig['prior_v7_migration_path']))  # legacy-revision-compat
   if prior.get('schema')!='openevo.ceiling1-stage2-resource-migration.v7' or prior.get('status')!='AUTHORIZED': raise RuntimeError('prior v7 migration invalid')  # legacy-revision-compat
   if sha(Path(mig['prior_v7_migration_path']))!=mig.get('prior_v7_migration_sha256') or prior.get('successor_execution_git_sha')!=mig.get('predecessor_execution_git_sha'): raise RuntimeError('v7 -> Ray lineage drift')  # legacy-revision-compat
  else:
   prior_path=Path(mig['prior_ray_migration_path']);prior=load(prior_path)
   if prior.get('schema') not in MIGRATION_SCHEMAS or prior.get('status') not in {'AUTHORIZED','AUTHORIZED_PENDING_SCIENCE_RECOVERY'}: raise RuntimeError('prior Ray migration invalid')
   if sha(prior_path)!=mig.get('prior_ray_migration_sha256') or prior.get('successor_execution_git_sha')!=mig.get('predecessor_execution_git_sha'): raise RuntimeError('Ray -> Ray lineage drift')
  self.design_path=a.design.resolve();self.design=load(self.design_path);self.design_sha=sha(self.design_path)
  activation=load(self.activation);expected='FROZEN_DESIGN_NO_ACTIVATION_AUTHORITY' if self.arm=='3b' else 'FROZEN_PREREG_NEEDS_LAUNCH_READINESS'
  if self.design.get('status')!=expected or self.design.get('activation_authority') is not False or activation.get('vnext_design_sha256')!=self.design_sha: raise RuntimeError('candidate-state design/activation drift')  # legacy-revision-compat
  if sha(self.activation)!=(mig.get('predecessor_activation_sha256') or {}).get(self.arm): raise RuntimeError('scientific activation hash drift')
  self.cfg=self.wt/'configs/experiment/openevo-ceiling1-ray-placement-backend-202609042025.json';self.lease=self.wt/'scripts/openevo_webshop/ceiling1_ray_placement_backend_202609042025.py';self.use_v3_resource=True;self.use_v5_resource=True  # legacy-revision-compat
  if sha(self.cfg)!=mig.get('ray_backend_config_sha256'): raise RuntimeError('Ray backend config drift')
  ready=Path(mig['ray_pool_ready_path']);self.lease_root=ready.parent
  if not ready.is_file() or sha(ready)!=mig.get('ray_pool_ready_sha256'): raise RuntimeError('Ray pool READY identity drift')
  rx=load(ready)
  if rx.get('status')!='READY' or rx.get('config_sha256')!=sha(self.cfg) or set(rx.get('managed_gpus') or [])!={4,5,6,7}: raise RuntimeError('Ray pool not ready for GPU4-7')
  current_digest=artifact_digest(Path(self.state['merged_base_receipt']),Path(self.state['memory']),Path(self.state['skill']),Path(self.state['agent_system']),Path(self.state['current_sd_adapter']) if self.state.get('current_sd_adapter') else None)
  if self.state.get('vnext_design_sha256')!=self.design_sha or self.state.get('vnext_active_artifact_digest')!=current_digest: raise RuntimeError('candidate-state active state drift at Ray migration')  # legacy-revision-compat
  boundary=(mig.get('boundaries') or {}).get(self.arm) or {}
  if int(self.state.get('next_round',-1))!=int(boundary.get('next_round',-2)) or int(self.state.get('rollouts_consumed',-1))!=int(boundary.get('rollouts_consumed',-2)): raise RuntimeError('execution state moved after Ray migration receipt')
  locks=self.out/'execution-source-locks';locks.mkdir(parents=True,exist_ok=True)
  resource_migration_sha=sha(a.recovery_amendment.resolve())
  lock_payload={'schema':'openevo.ceiling1-stage2-execution-source-lock.202609042025','clean_worktree':True,'execution_git_sha':self.execution_git_sha,'vnext_design_sha256':self.design_sha,'resource_migration_sha256':resource_migration_sha,'resource_backend':'ray-202609042025','resource_only_successor':science_amendment_sha is None}  # legacy-revision-compat
  if science_amendment_sha is not None: lock_payload.update({'science_amendment_sha256':science_amendment_sha,'scientific_transition_change':'carrier contract failure after frozen primary+repair becomes NOOP_KEEP_PRIOR'})
  write_once_json(execution_source_lock_path(locks,self.execution_git_sha,resource_migration_sha),lock_payload)
  self.save()
 def allowed_rollout_shas(self,r:int):
  allowed={self.execution_git_sha};b=(self.recovery.get('boundaries') or {}).get(self.arm) or {}
  if int(b.get('partial_round',-1))==r:
   allowed.update(b.get('historical_rollout_execution_shas') or [self.recovery['predecessor_execution_git_sha']])
  return allowed
 def allowed_artifact_execution_shas(self,r:int):
  allowed={self.execution_git_sha};b=(self.recovery.get('boundaries') or {}).get(self.arm) or {}
  if int(b.get('partial_round',-1))==r: allowed.update(b.get('historical_artifact_execution_shas') or [])
  return allowed
 def _select(self,workload,selection,count=None):
  while True:
   args=['python3',str(self.lease),'--config',str(self.cfg),'--state-root',str(self.lease_root),'select','--workload',workload,'--output',str(selection)]
   if count is not None:args+=['--count',str(count)]
   q=cmd(args,check=False,capture=True)
   if q.returncode==0:return list(map(int,load(selection)['selected_gpus'])),selection
   detail=(q.stderr.strip() or q.stdout.strip())
   if 'RAY_ADMISSION_PAUSED' in detail: raise RayAdmissionPaused('Ray admission is paused')
   if 'RAY_POOL_UNAVAILABLE' in detail: raise RuntimeError(detail)
   time.sleep(.5)
 def choose(self,kind,round_idx=None,workload=None):
  if round_idx is None:raise RuntimeError('Ray placement requires round identity')
  workload=workload or f'stage2_rollout_{self.arm}';rr=self.out/f'round-{round_idx:03d}'
  if kind!='rollout':raise RuntimeError('Ray single-GPU workloads use attempt-specific selections')
  selection=rr/'resource-selection.json'
  if selection.is_file():
   x=load(selection)
   if x.get('schema') in {'openevo.ceiling1-ray-selection.202609042025','openevo.ceiling1-ray-selection.v1'}:return list(map(int,x['selected_gpus'])),selection  # legacy-revision-compat
   # Historical partial-round v7 selection is valid only when rollout summary already exists and is reused.  # legacy-revision-compat
   if (rr/'summary.json').is_file():return list(map(int,x['selected_gpus'])),selection
   raise RuntimeError('non-Ray resource selection exists for unconsumed Ray rollout')
  return self._select(workload,selection,None)
 def launch_retry(self,kind,round_idx,workload,launcher,env,vram=None,max_attempts=4):
  if kind=='rollout':return super().launch_retry(kind,round_idx,workload,launcher,env,vram=vram,max_attempts=max_attempts)
  import fcntl
  rr=self.out/f'round-{round_idx:03d}'
  def materialized_success():
   if kind=='training':
    p=env.get('CEILING1_STAGE2_SD_RECEIPT'); return bool(p and Path(p).is_file())
   if kind=='reflector':
    p=env.get('CEILING1_STAGE2_OUTPUT')
    if not p:return False
    q=Path(p);return q.is_file() and q.with_suffix(q.suffix+'.receipt.json').is_file()
   return False
  lock=rr/f'.{workload}.ray-launch.lock';lock.parent.mkdir(parents=True,exist_ok=True)
  with lock.open('a+') as lf:
   fcntl.flock(lf.fileno(),fcntl.LOCK_EX)
   if materialized_success():return 'materialized-existing-success',[]
   last='';attempt=0;launched=0
   while launched<max_attempts:
    stem=f'openevo-ceiling1-stage2-{self.arm}-r{round_idx:03d}-{env.get("CEILING1_STAGE2_KIND",kind)}'
    while True:
     selection=rr/f'{workload}-ray-selection-a{attempt}.json';c=f'{stem}-ray-a{attempt}'
     if not selection.exists() and dstatus(c)=='absent': break
     attempt+=1
    gpus,selection=self._select(workload,selection,1);lid=f'ceiling1-ray-{self.arm}-r{round_idx:03d}-{env.get("CEILING1_STAGE2_KIND",kind)}-a{attempt}'
    launched+=1;attempt+=1;env2={**env,'CEILING1_STAGE2_CONTAINER':c,'CEILING1_STAGE2_GPU_UUID':UUID[gpus[0]]}
    if self.execution_git_sha:env2['CEILING1_STAGE2_EXECUTION_GIT_SHA']=self.execution_git_sha
    ok,last=self.lease_run(lease_id=lid,workload=workload,container=c,launcher=launcher,gpus=gpus,env=env2,vram=vram,selection=selection)
    if not ok:
     if materialized_success():self.lease_reconcile();return c,gpus
     if 'RAY_ADMISSION_PAUSED' in last: raise RayAdmissionPaused('Ray admission paused between selection and launch')
     if 'RAY_POOL_UNAVAILABLE' in last: raise RuntimeError(last)
     time.sleep(.5);continue
    rc=wait_container(c);self.lease_reconcile()
    if rc==0 or materialized_success():return c,gpus
    last=f'container {c} exit={rc}';time.sleep(.5)
  raise RuntimeError(f'{kind} failed after {max_attempts} Ray engineering attempts: {last}')
 def validate_recovery(self):
  cmd(['python3',self.wt/'scripts/openevo_webshop/validate_ceiling1_stage2_resource_migration_ray_202609042025.py','--output',self.out,'--amendment',self.a.recovery_amendment,'--arm',self.arm,'--execution-git-sha',self.execution_git_sha])
 def run(self):
  try:
   if self.a.recovery_amendment and int(self.state['next_round'])>0:self.validate_recovery()
   while int(self.state['next_round'])<160:
    r=int(self.state['next_round']);self.run_round(r)
    if self.a.recovery_amendment and r==0:self.validate_recovery()
   final=self.seal();print(json.dumps(final,sort_keys=True));return 0
  except RayAdmissionPaused as exc:
   atomic_json(self.out/'controller-resource-pause.json',{'schema':'openevo.ceiling1-stage2-resource-pause.202609042025','status':'PAUSED_RESOURCE_ONLY','arm':self.arm,'next_round':int(self.state['next_round']),'rollouts_consumed':int(self.state['rollouts_consumed']),'state_id':self.state['state_id'],'execution_git_sha':self.execution_git_sha,'reason':str(exc),'paused_at_epoch':time.time(),'scientific_state_changed':False,'formal_rollout_replay':0})
   print(json.dumps({'status':'PAUSED_RESOURCE_ONLY','arm':self.arm,'next_round':int(self.state['next_round'])},sort_keys=True));return 0
  except BaseException as exc:
   self.state['status']='BLOCKED';self.state['blocker']=f'{type(exc).__name__}: {exc}';self.state['blocked_at_epoch']=time.time();self.save();raise
def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=('3b','qwen3-1p7b'),required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--analysis-root',type=Path,required=True);p.add_argument('--seed-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stage2-activation',type=Path,required=True);p.add_argument('--recovery-amendment',type=Path,required=True);p.add_argument('--science-recovery-amendment',type=Path);p.add_argument('--design',type=Path,required=True);p.add_argument('--capacity-policy',type=Path,required=True);p.add_argument('--step-seed-amendment',type=Path,required=True);p.add_argument('--harness201-carrier-root',type=Path,required=True);p.add_argument('--execution-git-sha',required=True);a=p.parse_args()
 lock=acquire_controller_lock(a.output.resolve(),a.arm,a.execution_git_sha)
 try:return RayCurrentController(a).run()
 finally:lock.close()
if __name__=='__main__':raise SystemExit(main())
