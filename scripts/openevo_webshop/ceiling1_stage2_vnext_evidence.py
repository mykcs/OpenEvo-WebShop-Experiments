#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openevo_webshop.comparison_contract import extract_goal
from openevo_webshop.env_compat import WebshopEnv, WebshopEnvConfig

def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def loadj(p:Path): return [json.loads(x) for x in p.read_text().split('\n') if x.strip()]
def canon(x:Any)->bytes: return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def evid(x:dict)->str: return hashlib.sha256(canon(x)).hexdigest()
def task_identity(r:dict)->str: return f"{int(r['worker_seed'])}:{int(r['source_session_index'])}:{r['instruction_sha256']}"
def is_buy(a:str|None)->bool: return str(a or '').strip().lower()=='click[buy now]'
def clean_exact(r:dict)->bool:
 return bool(float(r.get('final_task_score',-1))==1.0 and r.get('won') and r.get('identity_ok') and r.get('trace_complete') and not r.get('parser_or_fallback_failure') and not r.get('parser_repair_used') and not r.get('engineering_invalid') and not r.get('error'))
def atomic_json(p:Path,x:Any): p.parent.mkdir(parents=True,exist_ok=True); q=p.with_name('.'+p.name+'.tmp'); q.write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n'); q.replace(p)

def current_round_summaries(partial:list[dict],exact_u:list[dict],round_index:int)->list[dict]:
 current_partial=[x for x in partial if int(x['source_round'])==round_index]
 current_exact=[x for x in exact_u if int(x['source_round'])==round_index]
 by_task=defaultdict(list)
 for x in current_partial: by_task[x['task_identity']].append(x)
 summaries=[]
 for tid,es in sorted(by_task.items()):
  scores=[float(x['source_final_score']) for x in es]; mistakes=sorted({x['kind'] for x in es if x['label']=='verified_mistake'}); best=sorted(es,key=lambda x:(x['label']!='verified_positive',-float(x['source_final_score']),int(x['schedule_position']),int(x.get('step',-1))))[0]
  summaries.append({'task_identity':tid,'best_score':max(scores),'worst_score':min(scores),'score_values':sorted(set(scores)),'clean_success_count':sum(1 for x in current_exact if x['task_identity']==tid),'failure_signatures':['verified_mistake:'+x for x in mistakes],'best_examples':[{'schedule_position':best['schedule_position'],'score':best['source_final_score'],'clean_exact_success':False,'sequence':best['kind'],'actions':[best['action']],'first_prompt_excerpt':best['prompt_excerpt'],'failure_signature':('verified_mistake:'+best['kind']) if best['label']=='verified_mistake' else None}]})
 return summaries

def classify_recorded(rows:list[dict],seed_root:Path,round_index:int,attempts:Path)->tuple[list[dict],dict]:
 webshop=seed_root/'agent_system/environments/env_package/webshop/webshop'; sys.path.insert(0,str(webshop)); from web_agent_site.engine.normalize import normalize_color
 steps=[]; missing=0
 for row in rows:
  tid=task_identity(row)
  for st in (row.get('episode') or {}).get('trace',[]):
   action=st.get('action'); vs=st.get('verifier_state')
   if not action: continue
   if not isinstance(vs,dict) or not isinstance(vs.get('before'),dict) or (vs.get('environment_step_executed') and not isinstance(vs.get('after'),dict)):
    missing+=1; continue
   before=vs['before']; after=vs.get('after') or before; goal_asin=before.get('goal_asin'); goal_options=before.get('goal_options'); before_asin=before.get('asin'); after_asin=after.get('asin'); before_opts=dict(before.get('options') or {}); after_opts=dict(after.get('options') or {})
   label='unknown';kind='unscored'
   if is_buy(action) and vs.get('environment_step_executed'):
    score=float(vs.get('terminal_task_score',0) or 0);label='verified_positive' if math.isclose(score,1.0,abs_tol=1e-12) else 'verified_mistake';kind='exact_success_buy' if label=='verified_positive' else 'premature_or_inexact_buy'
   elif vs.get('environment_step_executed'):
    changed=[k for k in set(before_opts)|set(after_opts) if before_opts.get(k)!=after_opts.get(k)]
    if changed:
     k=changed[0];target=(goal_options or {}).get(k) if isinstance(goal_options,dict) else None;val=after_opts.get(k)
     if target is not None:
      matched=normalize_color(str(val))==normalize_color(str(target));on_target=str(after_asin or '').upper()==str(goal_asin or '').upper()
      if matched and on_target:label='verified_positive';kind='goal_option_match'
      elif matched:label='unknown';kind='goal_option_match_on_non_target_product_rejected_v2'
      else:label='verified_mistake';kind='goal_option_mismatch'
    elif after_asin!=before_asin and after_asin is not None and str(after_asin).upper()==str(goal_asin).upper():label='verified_positive';kind='exact_target_product'
   base={'schema':'openevo.ceiling1-stage2-vnext-step-evidence.v1','evidence_type':'step','source_round':round_index,'schedule_position':int(row['schedule_position']),'task_id':row['task_id'],'task_identity':tid,'step':int(st['step']),'action':action,'label':label,'kind':kind,'goal_asin':goal_asin,'goal_options':goal_options,'before_asin':before_asin,'after_asin':after_asin,'source_final_score':float(row.get('final_task_score',0) or 0),'instruction_sha256':row['instruction_sha256'],'source_attempts_path':str(attempts.resolve()),'prompt_excerpt':str(st.get('base_prompt') or st.get('prompt') or '')[:800]}
   base['evidence_id']=evid({k:v for k,v in base.items() if k!='evidence_id'});steps.append(base)
 return steps,{'replayed_rows':0,'identity_failures':0,'terminal_reward_parity_failures':0,'recorded_state_steps':len(steps),'missing_recorded_state_steps':missing,'source':'recorded-rollout-state'}

def replay_current(rows:list[dict],seed_root:Path,round_index:int,attempts:Path)->tuple[list[dict],dict]:
 webshop=seed_root/'agent_system/environments/env_package/webshop/webshop'; sys.path.insert(0,str(webshop)); from web_agent_site.engine.normalize import normalize_color
 envs={}; steps=[]; identity_fail=0; parity_fail=0; replayed=0
 try:
  for row in rows:
   seed=int(row['worker_seed']); env=envs.get(seed)
   if env is None: env=WebshopEnv(WebShopConfig(seed_root,seed)); envs[seed]=env
   sid=str(int(row['source_session_index'])); obs,_=env.reset(int(row['source_session_index']),seed=seed); goal_text=extract_goal(obs)
   if hashlib.sha256(goal_text.encode()).hexdigest()!=row['instruction_sha256']: identity_fail+=1; continue
   rawenv=env._build(); server=rawenv.server; goal=server.user_sessions[sid]['goal']; tid=task_identity(row)
   for st in (row.get('episode') or {}).get('trace',[]):
    action=st.get('action')
    if not action: break
    sess=server.user_sessions[sid]; before_asin=sess.get('asin'); before_opts=dict(sess.get('options') or {})
    _,_,done,info=env.step(action); info=info or {}; sess2=server.user_sessions[sid]; after_asin=sess2.get('asin'); after_opts=dict(sess2.get('options') or {})
    label='unknown'; kind='unscored'
    if is_buy(action):
     score=float(info.get('task_score',0) or 0); label='verified_positive' if math.isclose(score,1.0,abs_tol=1e-12) else 'verified_mistake'; kind='exact_success_buy' if label=='verified_positive' else 'premature_or_inexact_buy'
    else:
     changed=[k for k in set(before_opts)|set(after_opts) if before_opts.get(k)!=after_opts.get(k)]
     if changed:
      k=changed[0]; target=(goal.get('goal_options') or {}).get(k) if isinstance(goal.get('goal_options'),dict) else None; val=after_opts.get(k)
      if target is not None:
       matched=normalize_color(str(val))==normalize_color(str(target)); on_target=str(after_asin or '').upper()==str(goal.get('asin') or '').upper()
       if matched and on_target: label='verified_positive'; kind='goal_option_match'
       elif matched and not on_target: label='unknown'; kind='goal_option_match_on_non_target_product_rejected_v2'
       else: label='verified_mistake'; kind='goal_option_mismatch'
     elif after_asin!=before_asin and after_asin is not None and str(after_asin).upper()==str(goal.get('asin')).upper(): label='verified_positive'; kind='exact_target_product'
    base={'schema':'openevo.ceiling1-stage2-vnext-step-evidence.v1','evidence_type':'step','source_round':round_index,'schedule_position':int(row['schedule_position']),'task_id':row['task_id'],'task_identity':tid,'step':int(st['step']),'action':action,'label':label,'kind':kind,'goal_asin':goal.get('asin'),'goal_options':goal.get('goal_options'),'before_asin':before_asin,'after_asin':after_asin,'source_final_score':float(row.get('final_task_score',0) or 0),'instruction_sha256':row['instruction_sha256'],'source_attempts_path':str(attempts.resolve()),'prompt_excerpt':str(st.get('base_prompt') or st.get('prompt') or '')[:800]}
    base['evidence_id']=evid({k:v for k,v in base.items() if k!='evidence_id'}); steps.append(base)
    if done:
     if not math.isclose(float(row.get('final_task_score',0) or 0),float(info.get('task_score',0) or 0),rel_tol=0,abs_tol=1e-12): parity_fail+=1
     break
   replayed+=1
 finally:
  for env in envs.values(): env.close()
 return steps,{'replayed_rows':replayed,'identity_failures':identity_fail,'terminal_reward_parity_failures':parity_fail}

class WebShopConfig(WebshopEnvConfig):
 def __init__(self,seed_root:Path,seed:int): super().__init__(seed_root=str(seed_root),num_products=1000,seed=seed)

def main()->int:
 ap=argparse.ArgumentParser(); ap.add_argument('--require-recorded-state',action='store_true'); ap.add_argument('--attempts',type=Path,required=True); ap.add_argument('--seed-root',type=Path,required=True); ap.add_argument('--stage2-root',type=Path,required=True); ap.add_argument('--state-before',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--round-index',type=int,required=True); a=ap.parse_args()
 rows=loadj(a.attempts); rows.sort(key=lambda x:int(x['schedule_position']))
 if len(rows)!=128 or any(x.get('engineering_invalid') for x in rows): raise RuntimeError('vNext evidence requires exact 128 engineering-valid rows')
 if a.output.exists(): raise RuntimeError('vNext evidence output is write-once')
 state=json.loads(a.state_before.read_text()); consumed=set(state.get('vnext_consumed_evidence_ids') or [])
 has_recorded=all((not st.get('action')) or isinstance(st.get('verifier_state'),dict) for row in rows for st in (row.get('episode') or {}).get('trace',[]))
 if a.require_recorded_state and not has_recorded: raise RuntimeError('vNext formal evidence requires recorded verifier_state on every actionable step')
 steps,tele=classify_recorded(rows,a.seed_root.resolve(),a.round_index,a.attempts) if has_recorded else replay_current(rows,a.seed_root.resolve(),a.round_index,a.attempts)
 if tele['identity_failures'] or tele['terminal_reward_parity_failures'] or int(tele.get('missing_recorded_state_steps',0)): raise RuntimeError(f"deterministic verifier parity failure {tele}")
 current_candidates=[x for x in steps if x['label'] in {'verified_positive','verified_mistake'}]
 exact=[]
 for row in rows:
  if not clean_exact(row): continue
  x={'schema':'openevo.ceiling1-stage2-vnext-trajectory-evidence.v1','evidence_type':'clean_exact_trajectory','source_round':a.round_index,'schedule_position':int(row['schedule_position']),'task_id':row['task_id'],'task_identity':task_identity(row),'instruction_sha256':row['instruction_sha256'],'source_attempts_path':str(a.attempts.resolve()),'trace_count':len((row.get('episode') or {}).get('trace',[])),'source_final_score':1.0}
  x['evidence_id']=evid({k:v for k,v in x.items() if k!='evidence_id'}); exact.append(x)
 current_candidates += exact
 prior=[]
 for r in range(a.round_index):
  rr=a.stage2_root/f'round-{r:03d}'
  if not (rr/'ROUND_COMPLETE.json').is_file(): raise RuntimeError(f'prior round {r} not sealed before vNext evidence build')
  p=rr/'evidence/candidate-evidence-current.jsonl'
  if p.is_file(): prior.extend(loadj(p))
 all_by_id={x['evidence_id']:x for x in [*prior,*current_candidates]}; unconsumed=[x for x in all_by_id.values() if x['evidence_id'] not in consumed]
 unconsumed.sort(key=lambda x:(int(x['source_round']),int(x['schedule_position']),0 if x['evidence_type']=='step' else 1,int(x.get('step',-1)),x['evidence_id']))
 cand_ids=[x['evidence_id'] for x in unconsumed]; digest=hashlib.sha256(''.join(cand_ids).encode()).hexdigest() if cand_ids else None
 last=state.get('vnext_last_candidate_attempt') or {}; same=bool(digest and last.get('active_artifact_digest')==state.get('vnext_active_artifact_digest') and last.get('evidence_digest')==digest)
 should_attempt=bool(unconsumed) and not same
 partial=[x for x in unconsumed if x['evidence_type']=='step']; exact_u=[x for x in unconsumed if x['evidence_type']=='clean_exact_trajectory']
 summaries=current_round_summaries(partial,exact_u,a.round_index)
 support=defaultdict(set)
 for x in partial: support[f"{x['label']}:{x['kind']}"] .add(x['task_identity'])
 skill=[{'pattern':k,'kind':'typed_transition','total_task_support':len(v),'current_task_support':len(v),'task_ids':sorted(v)} for k,v in sorted(support.items()) if len(v)>=2]
 agent=[x for x in skill if int(x['total_task_support'])>=3]
 repeated=[k for k,v in support.items() if k.startswith('verified_mistake:') and len(v)>=2]
 gates={'text_memory':{'eligible':bool(should_attempt and partial),'clean_success_count':len(exact_u),'contrast_task_count':0,'repeated_failure_patterns':sorted(repeated),'noop_reason':None if should_attempt and partial else 'VNEXT_MEMORY_NO_NEW_UNATTEMPTED_TYPED_EVIDENCE'},'skill_bundle':{'eligible':bool(should_attempt and skill),'supporting_patterns':skill,'noop_reason':None if should_attempt and skill else 'VNEXT_SKILL_NO_CROSS_TASK_TYPED_SUPPORT'},'agent_system':{'eligible':bool(should_attempt and agent),'supporting_patterns':agent,'noop_reason':None if should_attempt and agent else 'VNEXT_AGENT_NO_CROSS_TASK_TYPED_SUPPORT'},'parametric_sd_lora':{'eligible':bool(should_attempt and exact_u),'selected_task_count':len({x['task_identity'] for x in exact_u}),'noop_reason':None if should_attempt and exact_u else 'VNEXT_PARAMETRIC_NO_UNATTEMPTED_CLEAN_EXACT'}}
 a.output.mkdir(parents=True)
 (a.output/'step-evidence.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in steps))
 (a.output/'candidate-evidence-current.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in current_candidates))
 (a.output/'candidate-evidence-unconsumed.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in unconsumed))
 ev={'schema':'openevo.ceiling1-stage2-vnext-round-evidence.v1','status':'PASS','round_index':a.round_index,'attempt_count':128,'task_identity_count':16,'attempts_sha256':sha(a.attempts),'summaries':summaries,'summary_scope':'current-round-unconsumed-step-evidence-v2','summary_count':len(summaries),'gates':gates,'candidate_evidence_digest':digest,'candidate_evidence_count':len(unconsumed),'new_candidate_evidence_count':sum(x['source_round']==a.round_index for x in unconsumed),'should_attempt_candidate':should_attempt,'partial_candidate_evidence_ids':[x['evidence_id'] for x in partial],'clean_exact_candidate_evidence_ids':[x['evidence_id'] for x in exact_u],'consumed_evidence_count':len(consumed),'verifier':tele,'label_counts':dict(Counter(x['label'] for x in steps))}
 atomic_json(a.output/'evidence.json',ev); atomic_json(a.output/'history-after.json',{'schema':'openevo.ceiling1-stage2-vnext-evidence-history.v1','through_round':a.round_index,'candidate_evidence_digest':digest,'candidate_evidence_count':len(unconsumed),'consumed_evidence_count':len(consumed)})
 print(json.dumps({'status':'PASS','round':a.round_index,'labels':ev['label_counts'],'candidate_evidence_count':len(unconsumed),'should_attempt':should_attempt,'gates':{k:v['eligible'] for k,v in gates.items()}},sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
