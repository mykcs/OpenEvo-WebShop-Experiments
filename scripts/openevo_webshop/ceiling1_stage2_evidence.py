#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, re
from collections import defaultdict
from pathlib import Path
from typing import Any

def read_jsonl(p:Path): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def atomic_json(p:Path,x:Any): p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name('.'+p.name+'.tmp'); t.write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n'); t.replace(p)
def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(r): return bool(float(r.get('final_task_score',-1))==1.0 and r.get('won') and r.get('identity_ok') and r.get('trace_complete') and not r.get('parser_or_fallback_failure') and not r.get('parser_repair_used') and not r.get('engineering_invalid') and not r.get('error'))
def ident(r): return f"{int(r['worker_seed'])}:{int(r['source_session_index'])}:{r['instruction_sha256']}"
def action_kind(a:str|None)->str:
    if not a: return 'invalid'
    if a.startswith('search['): return 'search'
    low=a.lower()
    if 'buy now' in low: return 'buy'
    if 'next >' in low or 'next]' in low: return 'paginate'
    if 'back to search' in low: return 'back_search'
    if a.startswith('click['): return 'click'
    return a.split('[',1)[0][:32]
def sequence(r):
    ks=[action_kind(x.get('action')) for x in (r.get('episode') or {}).get('trace',[])]; out=[]
    for k in ks:
        if not out or out[-1]!=k: out.append(k)
    return '>'.join(out[:12]) or 'empty'
def failure_sig(r):
    if r.get('terminated_on_invalid_action'): return 'invalid_action'
    if r.get('parser_repair_used'): return 'parser_repair'
    tr=(r.get('episode') or {}).get('trace',[]); acts=[x.get('action') for x in tr if x.get('action')]
    if len(acts)>=3 and len(set(acts[-3:]))==1: return 'repeated_same_action:'+action_kind(acts[-1])
    return 'unsuccessful_sequence:'+sequence(r)
def compact(r):
    trace=(r.get('episode') or {}).get('trace',[]); actions=[str(x.get('action') or '') for x in trace if x.get('action')]
    first=str(trace[0].get('base_prompt') or trace[0].get('prompt') or '')[:800] if trace else ''
    last=str(trace[-1].get('base_prompt') or trace[-1].get('prompt') or '')[:800] if trace else ''
    return {'schedule_position':int(r['schedule_position']),'task_identity':ident(r),'score':float(r['final_task_score']),'clean_exact_success':clean(r),'sequence':sequence(r),'actions':actions[:15],'first_prompt_excerpt':first,'last_prompt_excerpt':last,'failure_signature':None if clean(r) else failure_sig(r)}
def main()->int:
    p=argparse.ArgumentParser(); p.add_argument('--attempts',type=Path,required=True); p.add_argument('--prior-history',type=Path); p.add_argument('--output',type=Path,required=True); p.add_argument('--round-index',type=int,required=True); a=p.parse_args()
    rows=read_jsonl(a.attempts); rows.sort(key=lambda r:int(r['schedule_position']))
    if len(rows)!=128 or any(r.get('engineering_invalid') for r in rows): raise ValueError('evidence requires sealed 128 valid rollouts')
    groups=defaultdict(list)
    for r in rows: groups[ident(r)].append(r)
    if len(groups)!=16 or any(len(v)!=8 for v in groups.values()): raise ValueError('round must be 16 task identities x 8 rollouts')
    history={'schema':'openevo.ceiling1-stage2-evidence-history.v1','through_round':-1,'patterns':{}}
    if a.prior_history and a.prior_history.is_file(): history=json.loads(a.prior_history.read_text())
    if int(history.get('through_round',-1))!=a.round_index-1: raise ValueError('evidence history round mismatch')
    patterns={k:{'kind':v['kind'],'task_ids':list(v.get('task_ids',[])),'first_round':int(v['first_round']),'last_round':int(v['last_round'])} for k,v in history.get('patterns',{}).items()}
    current_support=defaultdict(set); summaries=[]; contrast=[]; clean_rows=[]; failure_support=defaultdict(set)
    for tid,rs in sorted(groups.items()):
        scores=[float(r['final_task_score']) for r in rs]; cr=[r for r in rs if clean(r)]; fr=[r for r in rs if not clean(r)]
        if max(scores)-min(scores)>1e-12: contrast.append(tid)
        for r in cr:
            key='success_sequence:'+sequence(r); current_support[key].add(tid); clean_rows.append(r)
        for r in fr:
            key='failure:'+failure_sig(r); current_support[key].add(tid); failure_support[failure_sig(r)].add(tid)
        summaries.append({'task_identity':tid,'best_score':max(scores),'worst_score':min(scores),'score_values':sorted(set(scores)),'clean_success_count':len(cr),'best_examples':[compact(r) for r in sorted(rs,key=lambda x:(-float(x['final_task_score']),int(x['schedule_position'])))[:2]],'failure_signatures':sorted({failure_sig(r) for r in fr})})
    for key,ids in current_support.items():
        kind='success_sequence' if key.startswith('success_sequence:') else 'failure_pattern'; item=patterns.get(key,{'kind':kind,'task_ids':[],'first_round':a.round_index,'last_round':a.round_index}); item['task_ids']=sorted(set(item['task_ids'])|set(ids)); item['last_round']=a.round_index; patterns[key]=item
    repeated_failure=sorted(k for k,ids in failure_support.items() if len(ids)>=2)
    text_eligible=bool(clean_rows or contrast or repeated_failure)
    skill_patterns=[]; agent_patterns=[]
    for key,item in sorted(patterns.items()):
        cur=len(current_support.get(key,set())); total=len(set(item['task_ids']))
        if cur>=1 and total>=2: skill_patterns.append({'pattern':key,'kind':item['kind'],'total_task_support':total,'current_task_support':cur,'task_ids':item['task_ids']})
        if cur>=1 and total>=3: agent_patterns.append({'pattern':key,'kind':item['kind'],'total_task_support':total,'current_task_support':cur,'task_ids':item['task_ids']})
    selected=[]; seen=set()
    for r in rows:
        if not clean(r): continue
        tid=ident(r)
        if tid in seen: continue
        seen.add(tid); selected.append({'task_identity':tid,'schedule_position':int(r['schedule_position']),'steps':len((r.get('episode') or {}).get('trace',[])),'trajectory':compact(r)})
    evidence={'schema':'openevo.ceiling1-stage2-round-evidence.v1','status':'PASS','round_index':a.round_index,'attempt_count':128,'task_identity_count':16,'attempts_sha256':sha(a.attempts),'prior_history_sha256':sha(a.prior_history) if a.prior_history and a.prior_history.is_file() else None,'summaries':summaries,'gates':{
      'text_memory':{'eligible':text_eligible,'clean_success_count':len(clean_rows),'contrast_task_count':len(contrast),'repeated_failure_patterns':repeated_failure,'noop_reason':None if text_eligible else 'TEXT_MEMORY_NOOP_NO_NEW_GROUNDED_EVIDENCE'},
      'parametric_sd_lora':{'eligible':bool(selected),'selected_task_count':len(selected),'selected':selected,'noop_reason':None if selected else 'PARAMETRIC_NOOP_NO_CLEAN_EXACT_SUCCESS'},
      'skill_bundle':{'eligible':bool(skill_patterns),'supporting_patterns':skill_patterns,'noop_reason':None if skill_patterns else 'SKILL_BUNDLE_NOOP_INSUFFICIENT_CROSS_TASK_SUPPORT'},
      'agent_system':{'eligible':bool(agent_patterns),'supporting_patterns':agent_patterns,'noop_reason':None if agent_patterns else 'AGENT_SYSTEM_NOOP_INSUFFICIENT_CROSS_TASK_SUPPORT'}},'history_before_round':a.round_index-1,'history_after_round':a.round_index}
    a.output.mkdir(parents=True,exist_ok=True); history_after=a.output/'history-after.json'; atomic_json(history_after,{'schema':'openevo.ceiling1-stage2-evidence-history.v1','through_round':a.round_index,'patterns':patterns}); evidence['history_after_sha256']=sha(history_after); atomic_json(a.output/'evidence.json',evidence); print(json.dumps({k:v['eligible'] for k,v in evidence['gates'].items()},sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
