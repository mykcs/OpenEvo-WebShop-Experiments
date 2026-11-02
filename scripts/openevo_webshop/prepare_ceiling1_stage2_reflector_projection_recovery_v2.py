#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
from scripts.openevo_webshop.ceiling1_stage2_vnext_evidence import current_round_summaries
PRE='76615acc680dd7b17d65ae7290df9ab5f98cda22'
OLD3='e73b1022b76b23875a40bffc9b7975feda57816e'; OLDQ='e439cbba64d2071ad7c1a239b84e893fd135bc78'
SAFE_UNCHANGED=['configs/experiment/openevo-ceiling-1.0-stage2-final-v1.json','configs/experiment/manifests/h1.44-on-3b-training-schedule-v1.json','scripts/openevo_webshop/ceiling1_stage2_format_hard_runtime.py','scripts/openevo_webshop/ceiling1_stage2_rollout_worker.py','scripts/openevo_webshop/ceiling1_stage2_rollout_worker_qwen3.py','scripts/openevo_webshop/ceiling1_stage2_round_driver.py','scripts/openevo_webshop/ceiling1_stage2_sd_lora.py','scripts/openevo_webshop/ceiling1_stage2_vnext_state.py']
ALLOWED_PREFIX='scripts/openevo_webshop/'
ALLOWED_NAMES={'ceiling1_stage2_vnext_evidence.py','ceiling1_stage2_reflector.py','run_ceiling1_stage2_vnext_arm.py','run_ceiling1_stage2_vnext_v6.py','recover_ceiling1_stage2_reflector_projection_v2.py','validate_ceiling1_stage2_reflector_projection_recovery_v2.py','prepare_ceiling1_stage2_reflector_projection_recovery_v2.py'}
def sha(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p:Path): return json.loads(p.read_text())
def gitbytes(repo:Path,ref:str,path:str): return subprocess.check_output(['git','-C',str(repo),'show',f'{ref}:{path}'])
def jbytes(x): return (json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()
def rows(p): return [json.loads(x) for x in p.open() if x.strip()]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--repo',type=Path,required=True); p.add_argument('--successor-sha',required=True); p.add_argument('--root',type=Path,required=True); p.add_argument('--resource-migration-amendment',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
 repo=a.repo.resolve(); root=a.root.resolve(); head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
 if head!=a.successor_sha or subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip(): raise RuntimeError('successor repo not exact clean')
 if load(a.resource_migration_amendment).get('successor_execution_git_sha')!=PRE: raise RuntimeError('resource migration predecessor drift')
 changed=subprocess.check_output(['git','-C',str(repo),'diff','--name-only',f'{PRE}..{head}'],text=True).splitlines()
 for f in changed:
  if not (f.startswith(ALLOWED_PREFIX) and Path(f).name in ALLOWED_NAMES) and not f.startswith('tests/'): raise RuntimeError(f'unauthorized recovery code change: {f}')
 for f in SAFE_UNCHANGED:
  if gitbytes(repo,PRE,f)!=gitbytes(repo,head,f): raise RuntimeError(f'science-critical runtime changed: {f}')
 arm_specs={'3b':1,'qwen3-1p7b':8}; arms={}
 for arm,r in arm_specs.items():
  out=root/'stage2'/arm/'formal-vnext-v1'; rr=out/f'round-{r:03d}'; statep=out/'execution-state.json'; st=load(statep)
  if st.get('status')!='BLOCKED' or int(st.get('next_round',-1))!=r or (rr/'ROUND_COMPLETE.json').exists(): raise RuntimeError(f'{arm} not at expected blocked unsealed round')
  evp=rr/'evidence/evidence.json'; old=load(evp); un=rows(rr/'evidence/candidate-evidence-unconsumed.jsonl'); partial=[x for x in un if x['evidence_type']=='step']; exact=[x for x in un if x['evidence_type']=='clean_exact_trajectory']
  corrected=dict(old); corrected['summaries']=current_round_summaries(partial,exact,r); corrected['summary_scope']='current-round-unconsumed-step-evidence-v2'; corrected['summary_count']=len(corrected['summaries'])
  if corrected['summary_count']>int(corrected['task_identity_count']) or len(old.get('summaries') or [])<=int(old['task_identity_count']): raise RuntimeError(f'{arm} summary recovery shape invalid')
  corrected_sha=hashlib.sha256(jbytes(corrected)).hexdigest(); receipt_path=rr/'REFLECTOR_PROJECTION_RECOVERY_V2.json'
  rec={'schema':'openevo.ceiling1-stage2-reflector-projection-recovery-apply.v2','status':'PASS','arm':arm,'round_index':r,'source_evidence_sha256':sha(evp),'corrected_evidence_sha256':corrected_sha,'source_summary_count':len(old.get('summaries') or []),'corrected_summary_count':len(corrected['summaries']),'summary_scope':'current-round-unconsumed-step-evidence-v2','candidate_evidence_digest':old.get('candidate_evidence_digest'),'gates_unchanged':True,'candidate_payload_files_unchanged':True,'formal_rollout_replay':0,'external_teacher_calls':0,'source_evidence_dir':str(rr/'evidence-superseded-unbounded-v1'),'corrected_evidence_dir':str(rr/'evidence'),'superseded_reflector_dir':str(rr/'text-memory-superseded-unbounded-v1')}
  arms[arm]={'recovery_round':r,'blocked_execution_state_sha256':sha(statep),'source_evidence_sha256':sha(evp),'corrected_evidence_sha256':corrected_sha,'source_summary_count':len(old.get('summaries') or []),'corrected_summary_count':len(corrected['summaries']),'candidate_evidence_digest':old.get('candidate_evidence_digest'),'apply_receipt_path':str(receipt_path),'apply_receipt_sha256':hashlib.sha256(jbytes(rec)).hexdigest(),'blocked_state_id':st.get('state_id')}
 hist={'3b':{'0':[OLD3],'1':[PRE]},'qwen3-1p7b':{'0':[OLDQ],'1':[OLDQ],**{str(i):[PRE] for i in range(2,9)}}}
 doc={'schema':'openevo.ceiling1-stage2-reflector-bounded-projection-recovery.v2','status':'AUTHORIZED','scope':'restore frozen bounded reflector evidence projection; no rollout/harness/sampling/reward/update-rule change','predecessor_execution_git_sha':PRE,'successor_execution_git_sha':head,'resource_migration_amendment_path':str(a.resource_migration_amendment.resolve()),'resource_migration_amendment_sha256':sha(a.resource_migration_amendment.resolve()),'arms':arms,'historical_rollout_reuse':hist,'changed_files':changed,'unchanged_science_runtime_files':SAFE_UNCHANGED,'formal_rollout_replay':0,'final_panel_access_delta':0,'external_teacher_calls_delta':0,'reflector_call_budget':'one primary plus at most one fixed repair under corrected bounded projection','authority_basis':'OPEN_EVO_CEILING_1.0_STAGE2_REFLECTOR_BOUNDED_RECOVERY_2026-08-31 retains current-round semantic summaries and omits accumulated historical support IDs'}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_bytes(jbytes(doc)); print(json.dumps(doc,indent=2,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
