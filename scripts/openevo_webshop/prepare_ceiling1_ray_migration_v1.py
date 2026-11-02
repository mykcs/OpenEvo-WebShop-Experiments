#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,subprocess,time
from pathlib import Path
SCIENCE_FILES=['configs/experiment/openevo-ceiling-1.0-stage2-final-v1.json','configs/experiment/manifests/h1.44-on-3b-training-schedule-v1.json','scripts/openevo_webshop/ceiling1_harness2011_prompt.py','scripts/openevo_webshop/ceiling1_stage2_format_hard_runtime.py','scripts/openevo_webshop/ceiling1_stage2_rollout_worker.py','scripts/openevo_webshop/ceiling1_stage2_rollout_worker_qwen3.py','scripts/openevo_webshop/ceiling1_stage2_round_driver.py','scripts/openevo_webshop/ceiling1_stage2_reflector.py','scripts/openevo_webshop/ceiling1_stage2_reflector_qwen3.py','scripts/openevo_webshop/ceiling1_stage2_sd_lora.py','scripts/openevo_webshop/ceiling1_stage2_vnext_evidence.py','scripts/openevo_webshop/ceiling1_stage2_vnext_parametric_dataset.py','scripts/openevo_webshop/ceiling1_stage2_vnext_state.py']
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def gbytes(repo,ref,path):return subprocess.check_output(['git','-c',f'safe.directory={repo}','-C',str(repo),'show',f'{ref}:{path}'])
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);text=json.dumps(x,indent=2,sort_keys=True)+'\n'
 if p.exists() and p.read_text()!=text:raise RuntimeError(f'immutable migration conflict {p}')
 if not p.exists():p.write_text(text)
def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--predecessor-sha',required=True);p.add_argument('--successor-sha',required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--v7-state-root',type=Path,required=True);p.add_argument('--ray-state-root',type=Path,required=True);p.add_argument('--ray-config',type=Path,required=True);p.add_argument('--ray-ready',type=Path,required=True);p.add_argument('--qualification',type=Path,required=True);p.add_argument('--prior-v7-migration',type=Path,required=True);p.add_argument('--activation-3b',type=Path,required=True);p.add_argument('--activation-q17',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 repo=a.repo.resolve();root=a.root.resolve()
 if subprocess.check_output(['git','-c',f'safe.directory={repo}','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!=a.successor_sha or subprocess.check_output(['git','-c',f'safe.directory={repo}','-C',str(repo),'status','--porcelain'],text=True).strip():raise RuntimeError('Ray repo not exact clean successor')
 prior=load(a.prior_v7_migration);cfg=load(a.ray_config);ready=load(a.ray_ready);qual=load(a.qualification)
 if prior.get('schema')!='openevo.ceiling1-stage2-resource-migration.v7' or prior.get('status')!='AUTHORIZED' or prior.get('successor_execution_git_sha')!=a.predecessor_sha:raise RuntimeError('prior v7 lineage invalid')
 if cfg.get('schema')!='openevo.ceiling1-ray-placement-backend.v1' or cfg.get('replacement_scope')!='placement-and-reservation-backend-only':raise RuntimeError('Ray config invalid')
 if ready.get('schema')!='openevo.ceiling1-ray-gpu-pool.v1' or ready.get('status')!='READY' or ready.get('config_sha256')!=sha(a.ray_config) or set(ready.get('managed_gpus') or [])!={4,5,6,7}:raise RuntimeError('Ray pool READY invalid')
 if qual.get('schema')!='openevo.ceiling1-ray-backend-qualification.v1' or qual.get('status')!='PASS' or qual.get('formal_task_consumption')!=0 or qual.get('final_panel_access')!=0:raise RuntimeError('Ray backend qualification invalid')
 old=load(a.v7_state_root/'state.json')
 if any(g.get('leases') for g in old.get('gpus',{}).values()) or old.get('reservations'):raise RuntimeError('v7 still has live leases/reservations')
 rs=a.ray_state_root/'state.json'
 if rs.exists():
  rx=load(rs)
  if any(g.get('leases') for g in rx.get('gpus',{}).values()) or rx.get('reservations'):raise RuntimeError('Ray formal state not empty before migration')
 acts={'3b':a.activation_3b.resolve(),'qwen3-1p7b':a.activation_q17.resolve()};boundaries={};actsha={};blocked=False
 for arm in ('3b','qwen3-1p7b'):
  out=root/'stage2'/arm/'formal-vnext-v1';st=load(out/'execution-state.json');nr=int(st['next_round']);cons=int(st['rollouts_consumed']);rr=out/f'round-{nr:03d}';prev=out/f'round-{nr-1:03d}'/'ROUND_COMPLETE.json'
  if st.get('status') not in {'RUNNING','BLOCKED'}:raise RuntimeError(f'{arm} unsupported migration state {st.get("status")}')
  if cons!=nr*128:raise RuntimeError(f'{arm} consumed accounting drift')
  if st.get('status')=='BLOCKED':blocked=True
  if rr.joinpath('ROUND_COMPLETE.json').exists():raise RuntimeError('next_round points to already-complete round')
  if rr.joinpath('summary.json').is_file():
   sm=load(rr/'summary.json')
   if sm.get('status')!='PASS' or int(sm.get('attempt_count',-1))!=128 or int(sm.get('engineering_invalid_count',-1))!=0:raise RuntimeError(f'{arm} partial rollout is not reusable PASS')
   kind='PARTIAL_ROUND_AFTER_ROLLOUT';partial=nr;attempt_sha=sha(rr/'attempts.jsonl');summary_sha=sha(rr/'summary.json')
  else:
   if not prev.is_file():raise RuntimeError(f'{arm} neither exact boundary nor reusable partial rollout')
   kind='EXACT_ROUND_BOUNDARY';partial=None;attempt_sha=None;summary_sha=None
  actsha[arm]=sha(acts[arm]);boundaries[arm]={'kind':kind,'next_round':nr,'rollouts_consumed':cons,'status':st.get('status'),'blocker':st.get('blocker'),'state_id':st['state_id'],'execution_state_sha256':sha(out/'execution-state.json'),'partial_round':partial,'partial_attempts_sha256':attempt_sha,'partial_summary_sha256':summary_sha,'formal_rollout_replay':0}
 parity={}
 for f in SCIENCE_FILES:
  x=gbytes(repo,a.predecessor_sha,f);y=gbytes(repo,a.successor_sha,f);ok=x==y;parity[f]={'identical':ok,'predecessor_sha256':hashlib.sha256(x).hexdigest(),'successor_sha256':hashlib.sha256(y).hexdigest()}
  if not ok:raise RuntimeError(f'science-critical file changed: {f}')
 status='AUTHORIZED_PENDING_SCIENCE_RECOVERY' if blocked else 'AUTHORIZED'
 activation={'schema':'openevo.ceiling1-ray-placement-activation.v1','status':'PASS','mode':'RAY_PLACEMENT_BACKEND_V1','config_sha256':sha(a.ray_config),'managed_gpus':[4,5,6,7],'unmanaged_gpus':[0,1,2,3],'pool_ready_path':str(a.ray_ready.resolve()),'pool_ready_sha256':sha(a.ray_ready),'qualification_path':str(a.qualification.resolve()),'qualification_sha256':sha(a.qualification),'resource_only':True,'activated_at_epoch':time.time()}
 actpath=a.ray_state_root/'activations/FORMAL_RAY_PLACEMENT_V1.json';atomic(actpath,activation)
 mig={'schema':'openevo.ceiling1-stage2-resource-migration.ray-v1','status':status,'scope':'placement-and-reservation-backend-only-no-scientific-semantic-change','predecessor_execution_git_sha':a.predecessor_sha,'successor_execution_git_sha':a.successor_sha,'prior_v7_migration_path':str(a.prior_v7_migration.resolve()),'prior_v7_migration_sha256':sha(a.prior_v7_migration),'predecessor_activation_sha256':actsha,'ray_backend_config_path':str(a.ray_config.resolve()),'ray_backend_config_sha256':sha(a.ray_config),'ray_pool_ready_path':str(a.ray_ready.resolve()),'ray_pool_ready_sha256':sha(a.ray_ready),'ray_backend_qualification_path':str(a.qualification.resolve()),'ray_backend_qualification_sha256':sha(a.qualification),'ray_activation_path':str(actpath),'ray_activation_sha256':sha(actpath),'boundaries':boundaries,'science_runtime_parity':parity,'accounting':{'formal_rollout_replay':0,'completed_round_rewrite':False,'external_teacher_calls_delta':0,'final_panel_access_delta':0,'scientific_semantics_changed':False},'placement_philosophy':'resource envelopes and live capacity determine co-residency; phase names do not','authorized_at_epoch':time.time()};atomic(a.output,mig);print(json.dumps({'status':status,'migration_sha256':sha(a.output),'activation_sha256':sha(actpath),'boundaries':boundaries},indent=2,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
