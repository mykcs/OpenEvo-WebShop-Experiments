#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, re, time
from pathlib import Path

FREEZE_ID='202609030400'
EXEC_SHA='362271d3cdc411581dd7f09d98d5ce69cfb1cb57'
ARMS=('qwen25-3b','qwen3-1p7b')

def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic_json(p:Path,v)->None:
    p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name('.'+p.name+'.tmp')
    t.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n',encoding='utf-8'); t.replace(p)
def task_from_prompt(prompt:str)->str:
    m=re.search(r'(?:^|\n)Task:\s*(.+?)\nPrior to this step',prompt,re.S)
    if not m: m=re.search(r'(?:^|\n)Task:\s*(.+?)\nRecent history',prompt,re.S)
    return ' '.join((m.group(1) if m else '').split())
def read_complete_rows(root:Path,arm:str):
    rows=[]
    for p in sorted(root.glob('shard-*/episodes/*.json')):
        r=json.loads(p.read_text(encoding='utf-8')); pos=int(r['schedule_position'])
        if r.get('arm')!=arm or r.get('freeze_id')!=FREEZE_ID or r.get('experiment_git_sha')!=EXEC_SHA: raise ValueError(f'identity drift {p}')
        if r.get('engineering_invalid') or r.get('terminated_on_invalid_action') or r.get('parser_repair_used'): raise ValueError(f'mechanical failure pos={pos}')
        ep=r.get('episode') or {}
        if ep.get('trace_complete') is not True or not (ep.get('trace') or []): raise ValueError(f'incomplete pos={pos}')
        rows.append(r)
    rows.sort(key=lambda r:int(r['schedule_position']))
    if len({int(r['schedule_position']) for r in rows})!=len(rows): raise ValueError('duplicate schedule positions')
    return rows
def validate_step(st:dict,pos:int)->None:
    rec=st.get('runtime_context_receipt') or {}; samp=rec.get('sampling') or {}
    ch=st.get('action_channel') or rec.get('action_channel') or {}; contract=rec.get('successor_contract') or {}
    if not (samp.get('temperature')==1.0 and samp.get('top_p')==1.0 and samp.get('top_k')==0): raise ValueError(f'sampling drift pos={pos}')
    if ch.get('admissible') is not True or not ch: raise ValueError(f'action channel drift pos={pos}')
    if not (contract.get('minimum_pre_action_model_tokens')==4 and contract.get('minimum_deliberation_enforced_by_interface') is True): raise ValueError(f'deliberation contract drift pos={pos}')
    if contract.get('observed_pre_action_nonempty') is not True: raise ValueError(f'empty deliberation pos={pos}')

def convert(r:dict,arm:str)->dict:
    pos=int(r['schedule_position']); trace=(r.get('episode') or {}).get('trace') or []
    for st in trace: validate_step(st,pos)
    rollout=(pos-1)%8; task_num=(pos-1)//8; task_id=f'shared030400-{arm}-task{task_num:03d}'
    desc=task_from_prompt(str(trace[0].get('prompt') or ''))
    if not desc: raise ValueError(f'missing task description pos={pos}')
    steps=[]
    for i,st in enumerate(trace):
        rec=st.get('runtime_context_receipt') or {}; ch=st.get('action_channel') or rec.get('action_channel') or {}
        valid=bool(st.get('action_valid',ch.get('admissible',False))); obs=str(st.get('formatted_observation') or '')
        steps.append({'step_idx':int(st.get('step',i)),'observation':obs,'observation_prompt':str(st.get('prompt') or ''),
                      'model_response':str(st.get('raw_completion') or ''),'action_valid':valid,
                      'info':{'is_action_valid':valid,'task_score':float(st.get('task_score',0.0) or 0)}})
    return {'task_id':task_id,'task_type':'webshop','goal_idx':int(r['source_session_index']),'rollout_id':rollout,
            'baseline_key':f'{task_id}:{rollout}','task_description':desc,'steps':steps,'success':bool(r.get('won',False)),
            'completed':True,'num_steps':len(steps),'final_reward':float(r.get('final_task_score',0) or 0),
            'final_task_score':float(r.get('final_task_score',0) or 0),'source_schedule_position':pos,
            'source_instruction_sha256':r.get('instruction_sha256'),'source_rollout_seed':int(r.get('rollout_seed',0)),
            'source_execution_git_sha':EXEC_SHA,'source_freeze_id':FREEZE_ID}
def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--arm',choices=ARMS,required=True)
    ap.add_argument('--stage1-root',type=Path,required=True); ap.add_argument('--output-root',type=Path,required=True)
    a=ap.parse_args(); rows=read_complete_rows(a.stage1_root.resolve(),a.arm); converted=[convert(r,a.arm) for r in rows]
    a.output_root.mkdir(parents=True,exist_ok=True); out=a.output_root/f'{a.arm}-student-trajectories.jsonl'
    out.write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in converted),encoding='utf-8')
    receipt={'schema':'openevo.shared-stage1.provisional-minimax-snapshot.v1','status':'PASS','arm':a.arm,
             'freeze_id':FREEZE_ID,'formal_generation_code_sha':EXEC_SHA,'trajectory_count':len(converted),
             'positions':[int(r['schedule_position']) for r in rows],'corpus_sha256':sha(out),
             'source_episode_sha256':[{'path':str(p),'sha256':sha(p)} for p in sorted(a.stage1_root.glob('shard-*/episodes/*.json'))],
             'sampling':{'temperature':1.0,'top_p':1.0,'top_k':0},'minimum_deliberation_required':True,
             'provisional_only':True,'no_feedback_to_stage1':True,'new_webshop_calls':0,'created_at_epoch':time.time()}
    atomic_json(a.output_root/f'PROVISIONAL_SNAPSHOT_{a.arm}.json',receipt)
    print(json.dumps({'status':'PASS','arm':a.arm,'trajectory_count':len(converted),'corpus_sha256':receipt['corpus_sha256']},sort_keys=True)); return 0

if __name__=='__main__': raise SystemExit(main())
