#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from collections import defaultdict
from typing import Any

def atomic_json(p:Path,x:Any): p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name('.'+p.name+'.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); t.replace(p)
def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(r): return bool(float(r.get('final_task_score',-1))==1.0 and r.get('won') and r.get('identity_ok') and r.get('trace_complete') and not r.get('parser_or_fallback_failure') and not r.get('parser_repair_used') and not r.get('engineering_invalid') and not r.get('error'))
def ident(r): return f"{int(r['worker_seed'])}:{int(r['source_session_index'])}:{r['instruction_sha256']}"
def main()->int:
    p=argparse.ArgumentParser(); p.add_argument('--attempts',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--round-index',type=int,required=True); p.add_argument('--max-records',type=int,default=256); a=p.parse_args()
    rows=[json.loads(x) for x in a.attempts.read_text().splitlines() if x.strip()]; rows.sort(key=lambda r:int(r['schedule_position']))
    if len(rows)!=128 or any(r.get('engineering_invalid') for r in rows): raise ValueError('parametric dataset requires one sealed 128-rollout round')
    groups=defaultdict(list)
    for r in rows: groups[ident(r)].append(r)
    if len(groups)!=16 or any(len(v)!=8 for v in groups.values()): raise ValueError('parametric dataset requires 16 task identities x 8 rollouts')
    if a.max_records!=256: raise ValueError('Ceiling Stage2 native SD-LoRA max_records must remain 256')
    attempts_sha256=sha(a.attempts)
    seen=set(); candidates=[]
    for r in rows:
        if clean(r) and ident(r) not in seen: seen.add(ident(r)); candidates.append(r)
    if not candidates:
        atomic_json(a.output/'selection.json',{'schema':'openevo.ceiling1-stage2-parametric-selection.v2','status':'NOOP_NO_SUCCESS','round_index':a.round_index,'selected_task_count':0,'trace_count':0,'max_records':a.max_records,'attempts_sha256':attempts_sha256}); return 0
    selected=[]; traces=0
    if len(candidates)>16: raise RuntimeError('more than 16 selected task identities in one 128-rollout round')
    for r in candidates:
        n=len((r.get('episode') or {}).get('trace',[]))
        if not 2<=n<=15: raise RuntimeError(f'clean exact-success trajectory has invalid WebShop step count {n}')
        selected.append(r); traces+=n
    if traces>240 or traces>a.max_records: raise RuntimeError(f'128-round parametric dataset exceeded frozen 16x15 bound: {traces}')
    excluded=[]
    status='DATASET_READY' if traces>=2 else 'NOOP_NATIVE_MIN_OPT_STEPS'
    result={'schema':'openevo.ceiling1-stage2-parametric-selection.v2','status':status,'round_index':a.round_index,'candidate_task_count':len(candidates),'selected_task_count':len(selected),'trace_count':traces,'max_records':a.max_records,'attempts_sha256':attempts_sha256,'selected_schedule_positions':[int(r['schedule_position']) for r in selected],'capacity_exclusions':excluded}
    a.output.mkdir(parents=True,exist_ok=True)
    if status=='DATASET_READY':
        records=[]
        for r in selected:
            tr=[]
            for i,s in enumerate((r.get('episode') or {}).get('trace',[])):
                tr.append({'prompt_messages':s.get('runtime_messages') or [{'role':'user','content':s['prompt']}],'response_messages':[{'role':'assistant','content':s['raw_completion']}],'metadata':{'source_step':int(s.get('step',i)),'source_schedule_position':int(r['schedule_position']),'source_task_identity':ident(r)}})
            records.append({'event_id':f"ceiling1-stage2:{int(r['schedule_position'])}",'task_id':ident(r),'session_id':f"round-{a.round_index:03d}:{int(r['schedule_position'])}",'status':'completed','reward':1.0,'traces':tr})
        payload=b''.join(json.dumps(x,sort_keys=True,allow_nan=False).encode()+b'\n' for x in records); rp=a.output/'records.jsonl'; rp.write_bytes(payload)
        manifest={'schema':'openevo.dataset.v1','name':f'openevo-ceiling1-stage2-round-{a.round_index:03d}-parametric','source':'current-round earliest clean exact success per distinct task identity','minimum_reward':1.0,'record_count':len(records),'trace_count':traces,'max_records':a.max_records,'records_path':'records.jsonl','records_uri':rp.resolve().as_uri(),'records_byte_size':len(payload),'records_sha256':hashlib.sha256(payload).hexdigest(),'selection':'earliest clean exact success per distinct task; chronological whole-trajectory packing; no cross-round raw buffer','capacity_exclusions':excluded}; mp=a.output/'manifest.json'; mp.write_text(json.dumps(manifest,indent=2,sort_keys=True))
        result.update({'manifest':str(mp),'manifest_sha256':sha(mp),'records':str(rp),'records_sha256':manifest['records_sha256']})
    atomic_json(a.output/'selection.json',result); print(json.dumps(result,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
