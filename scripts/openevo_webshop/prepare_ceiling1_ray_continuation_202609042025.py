#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,subprocess,time
from pathlib import Path
SCIENCE_FILES=['configs/experiment/openevo-ceiling-1.0-stage2-final-v1.json','configs/experiment/manifests/h1.44-on-3b-training-schedule-v1.json','scripts/openevo_webshop/ceiling1_harness2011_prompt.py','scripts/openevo_webshop/ceiling1_stage2_format_hard_runtime.py','scripts/openevo_webshop/ceiling1_stage2_rollout_worker.py','scripts/openevo_webshop/ceiling1_stage2_rollout_worker_qwen3.py','scripts/openevo_webshop/ceiling1_stage2_round_driver.py','scripts/openevo_webshop/ceiling1_stage2_reflector.py','scripts/openevo_webshop/ceiling1_stage2_reflector_qwen3.py','scripts/openevo_webshop/ceiling1_stage2_sd_lora.py','scripts/openevo_webshop/ceiling1_stage2_vnext_evidence.py','scripts/openevo_webshop/ceiling1_stage2_vnext_parametric_dataset.py',]  # legacy-revision-compat
def load(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def gbytes(repo,ref,path): return subprocess.check_output(['git','-c',f'safe.directory={repo}','-C',str(repo),'show',f'{ref}:{path}'])
def rows_shas(path):
 vals=set()
 for line in Path(path).read_text().split('\n'):
  if line.strip():
   x=json.loads(line);v=x.get('experiment_git_sha')
   if v: vals.add(v)
 return sorted(vals)
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);text=json.dumps(x,indent=2,sort_keys=True)+'\n'
 if p.exists() and p.read_text()!=text: raise RuntimeError(f'immutable continuation conflict {p}')
 if not p.exists(): p.write_text(text)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--predecessor-sha',required=True);ap.add_argument('--successor-sha',required=True);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--ray-state-root',type=Path,required=True);ap.add_argument('--ray-config',type=Path,required=True);ap.add_argument('--ray-ready',type=Path,required=True);ap.add_argument('--prior-ray-migration',type=Path,required=True);ap.add_argument('--activation-3b',type=Path,required=True);ap.add_argument('--activation-q17',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 repo=a.repo.resolve();root=a.root.resolve();prior=load(a.prior_ray_migration);cfg=load(a.ray_config);ready=load(a.ray_ready)
 if subprocess.check_output(['git','-c',f'safe.directory={repo}','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!=a.successor_sha or subprocess.check_output(['git','-c',f'safe.directory={repo}','-C',str(repo),'status','--porcelain'],text=True).strip(): raise RuntimeError('repo not exact clean successor')
 if prior.get('schema') not in {'openevo.ceiling1-stage2-resource-migration.ray-continuation.202609042025','openevo.ceiling1-stage2-resource-migration.ray-v1','openevo.ceiling1-stage2-resource-migration.ray-continuation-v1'} or prior.get('successor_execution_git_sha')!=a.predecessor_sha: raise RuntimeError('prior Ray lineage invalid')  # legacy-revision-compat
 if cfg.get('schema')!='openevo.ceiling1-ray-placement-backend.202609042025' or ready.get('status')!='READY' or ready.get('config_sha256')!=sha(a.ray_config): raise RuntimeError('Ray pool/config invalid')
 rs=a.ray_state_root/'state.json'
 if rs.exists():
  x=load(rs)
  if any(g.get('leases') for g in x.get('gpus',{}).values()) or x.get('reservations'): raise RuntimeError('Ray pool has live work; continue only at zero-live-workload boundary')
 acts={'3b':a.activation_3b.resolve(),'qwen3-1p7b':a.activation_q17.resolve()};boundaries={};blocked=False
 for arm in ('3b','qwen3-1p7b'):
  out=root/'stage2'/arm/'formal-vnext-v1';st=load(out/'execution-state.json');nr=int(st['next_round']);rr=out/f'round-{nr:03d}';cons=int(st['rollouts_consumed'])  # legacy-revision-compat
  if st.get('status') not in {'RUNNING','BLOCKED'} or cons!=nr*128: raise RuntimeError(f'{arm} state/accounting invalid')
  blocked |= st.get('status')=='BLOCKED';partial=None;attempt_sha=None;summary_sha=None;rollout_shas=[];artifact_shas=[];kind='EXACT_ROUND_BOUNDARY'
  if (rr/'summary.json').is_file():
   sm=load(rr/'summary.json')
   if sm.get('status')!='PASS' or int(sm.get('attempt_count',-1))!=128 or int(sm.get('engineering_invalid_count',-1))!=0: raise RuntimeError(f'{arm} partial rollout invalid')
   kind='PARTIAL_ROUND_AFTER_ROLLOUT';partial=nr;attempt_sha=sha(rr/'attempts.jsonl');summary_sha=sha(rr/'summary.json');rollout_shas=rows_shas(rr/'attempts.jsonl')
   u=rr/'sd-lora/update.json'
   if u.is_file():
    ux=load(u);v=ux.get('experiment_git_sha')
    if v: artifact_shas=[v]
  boundaries[arm]={'kind':kind,'next_round':nr,'rollouts_consumed':cons,'status':st.get('status'),'blocker':st.get('blocker'),'state_id':st['state_id'],'execution_state_sha256':sha(out/'execution-state.json'),'partial_round':partial,'partial_attempts_sha256':attempt_sha,'partial_summary_sha256':summary_sha,'historical_rollout_execution_shas':rollout_shas,'historical_artifact_execution_shas':artifact_shas,'formal_rollout_replay':0}
 parity={}
 for f in SCIENCE_FILES:
  x=gbytes(repo,a.predecessor_sha,f);y=gbytes(repo,a.successor_sha,f);ok=x==y;parity[f]={'identical':ok,'predecessor_sha256':hashlib.sha256(x).hexdigest(),'successor_sha256':hashlib.sha256(y).hexdigest()}
  if not ok: raise RuntimeError(f'science-critical file changed: {f}')
 old_state='scripts/openevo_webshop/ceiling1_stage2_vnext_state.py'  # legacy-revision-compat
 new_state='scripts/openevo_webshop/ceiling1_stage2_state_202609042025.py'
 x=gbytes(repo,a.predecessor_sha,old_state);y=gbytes(repo,a.successor_sha,new_state);ok=x==y
 parity['candidate_state_logic_renamed']={'identical':ok,'predecessor_path':old_state,'successor_path':new_state,'predecessor_sha256':hashlib.sha256(x).hexdigest(),'successor_sha256':hashlib.sha256(y).hexdigest()}
 if not ok: raise RuntimeError('candidate-state acceptance logic changed during timestamp successor')
 mig={'schema':'openevo.ceiling1-stage2-resource-migration.ray-continuation.202609042025','status':'AUTHORIZED_PENDING_SCIENCE_RECOVERY' if blocked else 'AUTHORIZED','scope':'whole-pool-zero-live-timestamp-successor','predecessor_execution_git_sha':a.predecessor_sha,'successor_execution_git_sha':a.successor_sha,'prior_ray_migration_path':str(a.prior_ray_migration.resolve()),'prior_ray_migration_sha256':sha(a.prior_ray_migration),'predecessor_activation_sha256':{k:sha(v) for k,v in acts.items()},'ray_backend_config_path':str(a.ray_config.resolve()),'ray_backend_config_sha256':sha(a.ray_config),'ray_pool_ready_path':str(a.ray_ready.resolve()),'ray_pool_ready_sha256':sha(a.ray_ready),'boundaries':boundaries,'science_runtime_parity':parity,'accounting':{'formal_rollout_replay':0,'completed_round_rewrite':False,'external_teacher_calls_delta':0,'final_panel_access_delta':0,'scientific_semantics_changed':False},'identity_timestamp':'202609042025','requires_zero_live_both_arms':True,'authorized_at_epoch':time.time()}
 atomic(a.output,mig);print(json.dumps({'status':mig['status'],'migration_sha256':sha(a.output),'boundaries':boundaries},indent=2,sort_keys=True));return 0
if __name__=='__main__': raise SystemExit(main())
