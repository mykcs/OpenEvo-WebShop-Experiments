#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p:Path): return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--amendment',type=Path,required=True); p.add_argument('--arm',choices=('3b','qwen3-1p7b'),required=True); p.add_argument('--execution-git-sha',required=True); a=p.parse_args()
 m=load(a.amendment); ar=(m.get('arms') or {}).get(a.arm) or {}
 if m.get('schema')!='openevo.ceiling1-stage2-reflector-bounded-projection-recovery.v2' or m.get('status')!='AUTHORIZED': raise RuntimeError('invalid science recovery amendment')
 if m.get('successor_execution_git_sha')!=a.execution_git_sha or ar.get('recovery_round') is None: raise RuntimeError('science recovery execution/arm scope drift')
 rr=a.output/f"round-{int(ar['recovery_round']):03d}"; receipt=Path(ar['apply_receipt_path'])
 if not receipt.is_file() or sha(receipt)!=ar.get('apply_receipt_sha256'): raise RuntimeError('science recovery apply receipt drift')
 x=load(receipt); ev=rr/'evidence/evidence.json'; old=rr/'evidence-superseded-unbounded-v1/evidence.json'
 if x.get('status')!='PASS' or x.get('formal_rollout_replay')!=0 or x.get('external_teacher_calls')!=0: raise RuntimeError('science recovery accounting drift')
 if not ev.is_file() or sha(ev)!=ar.get('corrected_evidence_sha256') or not old.is_file() or sha(old)!=ar.get('source_evidence_sha256'): raise RuntimeError('science recovery evidence identity drift')
 e=load(ev)
 if e.get('summary_scope')!='current-round-unconsumed-step-evidence-v2' or int(e.get('summary_count',-1))!=len(e.get('summaries') or []) or int(e['summary_count'])>int(e['task_identity_count']): raise RuntimeError('bounded summary invariant drift')
 print(json.dumps({'status':'PASS','arm':a.arm,'round_index':ar['recovery_round'],'summary_count':e['summary_count'],'candidate_evidence_digest':e.get('candidate_evidence_digest')},sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
