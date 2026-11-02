#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def loadj(p:Path):return [json.loads(x) for x in p.read_text().split('\n') if x.strip()]
def atomic(p:Path,x:Any):p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name('.'+p.name+'.tmp');q.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');q.replace(p)
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--candidate-evidence',type=Path,required=True);ap.add_argument('--state-before',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--round-index',type=int,required=True);ap.add_argument('--max-training-examples',type=int,default=256);a=ap.parse_args()
 if a.output.exists(): raise RuntimeError('vNext parametric dataset output is write-once')
 state=json.loads(a.state_before.read_text()); excluded_tasks=set(state.get('vnext_sd_consumed_task_identities') or []);ev=[x for x in loadj(a.candidate_evidence) if x.get('evidence_type')=='clean_exact_trajectory' and x.get('task_identity') not in excluded_tasks];ev.sort(key=lambda x:(int(x['source_round']),int(x['schedule_position'])))
 seen=set();chosen=[];rows=[];traces=0
 cache={}
 for e in ev:
  tid=e['task_identity']
  if tid in seen: continue
  p=Path(e['source_attempts_path']); rr=cache.get(str(p))
  if rr is None: rr={int(x['schedule_position']):x for x in loadj(p)};cache[str(p)]=rr
  r=rr[int(e['schedule_position'])]; tr=(r.get('episode') or {}).get('trace',[]); n=len(tr)
  clean=bool(float(r.get('final_task_score',-1))==1.0 and r.get('won') and r.get('identity_ok') and r.get('trace_complete') and not r.get('parser_or_fallback_failure') and not r.get('parser_repair_used') and not r.get('engineering_invalid') and not r.get('error'))
  if not clean or not 2<=n<=15: raise RuntimeError('vNext exact evidence source trajectory drift')
  if traces+n>a.max_training_examples: continue
  seen.add(tid);chosen.append((e,r));traces+=n
 a.output.mkdir(parents=True)
 if not chosen:
  atomic(a.output/'selection.json',{'schema':'openevo.ceiling1-stage2-vnext-parametric-selection.v1','status':'NOOP_NO_UNCONSUMED_CLEAN_EXACT','round_index':a.round_index,'selected_task_count':0,'trace_count':0,'max_training_examples':a.max_training_examples,'candidate_evidence_sha256':sha(a.candidate_evidence),'bound_evidence_ids':[],'bound_task_identities':[],'excluded_consumed_task_identity_count':len(excluded_tasks)});return 0
 records=[]
 for e,r in chosen:
  trs=[]
  for i,s in enumerate((r.get('episode') or {}).get('trace',[])):
   trs.append({'prompt_messages':s.get('runtime_messages') or [{'role':'user','content':s['prompt']}],'response_messages':[{'role':'assistant','content':s['raw_completion']}],'metadata':{'source_step':int(s.get('step',i)),'source_schedule_position':int(r['schedule_position']),'source_task_identity':e['task_identity'],'source_evidence_id':e['evidence_id']}})
  records.append({'event_id':f"ceiling1-vnext:{e['source_round']}:{int(r['schedule_position'])}",'task_id':e['task_identity'],'session_id':f"source-round-{e['source_round']:03d}:{int(r['schedule_position'])}",'status':'completed','reward':1.0,'traces':trs})
 payload=b''.join(json.dumps(x,sort_keys=True,allow_nan=False).encode()+b'\n' for x in records);rp=a.output/'records.jsonl';rp.write_bytes(payload)
 man={'schema':'openevo.dataset.v1','name':f'openevo-ceiling1-stage2-vnext-round-{a.round_index:03d}-parametric','source':'persistent unconsumed clean exact trajectory evidence; earliest per distinct task identity; chronological capacity packing','minimum_reward':1.0,'record_count':len(records),'trace_count':traces,'max_records':256,'max_training_examples':a.max_training_examples,'records_path':'records.jsonl','records_uri':rp.resolve().as_uri(),'records_byte_size':len(payload),'records_sha256':hashlib.sha256(payload).hexdigest(),'qualification_only':False,'formal_task_consumption_delta':0,'bound_evidence_ids':[e['evidence_id'] for e,_ in chosen],'bound_task_identities':[e['task_identity'] for e,_ in chosen]};mp=a.output/'manifest.json';mp.write_bytes(json.dumps(man,indent=2,sort_keys=True,allow_nan=False).encode())
 atomic(a.output/'selection.json',{'schema':'openevo.ceiling1-stage2-vnext-parametric-selection.v1','status':'DATASET_READY','round_index':a.round_index,'selected_task_count':len(records),'trace_count':traces,'max_training_examples':a.max_training_examples,'candidate_evidence_sha256':sha(a.candidate_evidence),'bound_evidence_ids':man['bound_evidence_ids'],'bound_task_identities':man['bound_task_identities'],'excluded_consumed_task_identity_count':len(excluded_tasks),'manifest':str(mp),'manifest_sha256':sha(mp),'records':str(rp),'records_sha256':sha(rp)});print(json.dumps({'status':'DATASET_READY','records':len(records),'traces':traces},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
