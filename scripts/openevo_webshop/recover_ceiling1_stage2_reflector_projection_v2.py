#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path
from scripts.openevo_webshop.ceiling1_stage2_vnext_evidence import current_round_summaries

def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p:Path): return json.loads(p.read_text())
def atomic(p:Path,x:dict):
 text=json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n'; t=p.with_name('.'+p.name+'.tmp'); t.write_text(text); t.replace(p)
def rows(p:Path): return [json.loads(line) for line in p.open() if line.strip()]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--arm',choices=('3b','qwen3-1p7b'),required=True); ap.add_argument('--round-index',type=int,required=True); ap.add_argument('--amendment',type=Path,required=True); a=ap.parse_args()
 amend=load(a.amendment); ar=(amend.get('arms') or {}).get(a.arm) or {}; r=a.round_index
 if amend.get('schema')!='openevo.ceiling1-stage2-reflector-bounded-projection-recovery.v2' or amend.get('status')!='AUTHORIZED' or int(ar.get('recovery_round',-1))!=r: raise RuntimeError('recovery amendment scope mismatch')
 out=a.root/'stage2'/a.arm/'formal-vnext-v1'; rr=out/f'round-{r:03d}'; state=load(out/'execution-state.json')
 if state.get('status')!='BLOCKED' or int(state.get('next_round',-1))!=r or sha(out/'execution-state.json')!=ar.get('blocked_execution_state_sha256'): raise RuntimeError('blocked execution state drift')
 if (rr/'ROUND_COMPLETE.json').exists(): raise RuntimeError('round already sealed')
 if int(load(rr/'summary.json').get('attempt_count',-1))!=128: raise RuntimeError('cached rollout denominator drift')
 old=rr/'evidence'; sup=rr/'evidence-superseded-unbounded-v1'; failed=rr/'text-memory'; failed_sup=rr/'text-memory-superseded-unbounded-v1'
 if sup.exists() or failed_sup.exists(): raise RuntimeError('recovery already applied')
 old_ev=load(old/'evidence.json'); old_sha=sha(old/'evidence.json')
 if old_sha!=ar.get('source_evidence_sha256') or len(old_ev.get('summaries') or [])<=int(old_ev.get('task_identity_count',-1)): raise RuntimeError('source evidence is not the expected unbounded failure')
 un=rows(old/'candidate-evidence-unconsumed.jsonl'); partial=[x for x in un if x['evidence_type']=='step']; exact=[x for x in un if x['evidence_type']=='clean_exact_trajectory']
 new_ev=dict(old_ev); new_ev['summaries']=current_round_summaries(partial,exact,r); new_ev['summary_scope']='current-round-unconsumed-step-evidence-v2'; new_ev['summary_count']=len(new_ev['summaries'])
 if new_ev['summary_count']>int(new_ev['task_identity_count']): raise RuntimeError('corrected summary count exceeds task denominator')
 expected=json.dumps(new_ev,ensure_ascii=False,sort_keys=True,indent=2)+'\n'
 if hashlib.sha256(expected.encode()).hexdigest()!=ar.get('corrected_evidence_sha256'): raise RuntimeError('corrected evidence does not match amendment')
 old.rename(sup); shutil.copytree(sup,old,copy_function=shutil.copy2)
 atomic(old/'evidence.json',new_ev)
 for name in ['candidate-evidence-current.jsonl','candidate-evidence-unconsumed.jsonl','step-evidence.jsonl','history-after.json']:
  if sha(old/name)!=sha(sup/name): raise RuntimeError(f'candidate evidence payload changed: {name}')
 check=load(old/'evidence.json')
 for key in ['gates','candidate_evidence_digest','candidate_evidence_count','partial_candidate_evidence_ids','clean_exact_candidate_evidence_ids','consumed_evidence_count']:
  if check.get(key)!=old_ev.get(key): raise RuntimeError(f'evidence selection/gate changed: {key}')
 if not failed.is_dir() or (failed/'terminal-receipt.json').exists(): raise RuntimeError('expected failed unsealed Text Memory directory missing/drifted')
 failed.rename(failed_sup)
 rec={'schema':'openevo.ceiling1-stage2-reflector-projection-recovery-apply.v2','status':'PASS','arm':a.arm,'round_index':r,'source_evidence_sha256':old_sha,'corrected_evidence_sha256':sha(old/'evidence.json'),'source_summary_count':len(old_ev.get('summaries') or []),'corrected_summary_count':len(check.get('summaries') or []),'summary_scope':check.get('summary_scope'),'candidate_evidence_digest':check.get('candidate_evidence_digest'),'gates_unchanged':True,'candidate_payload_files_unchanged':True,'formal_rollout_replay':0,'external_teacher_calls':0,'source_evidence_dir':str(sup),'corrected_evidence_dir':str(old),'superseded_reflector_dir':str(failed_sup)}
 receipt=rr/'REFLECTOR_PROJECTION_RECOVERY_V2.json'; atomic(receipt,rec)
 if sha(receipt)!=ar.get('apply_receipt_sha256'): raise RuntimeError('apply receipt digest differs from amendment')
 print(json.dumps(rec,indent=2,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
