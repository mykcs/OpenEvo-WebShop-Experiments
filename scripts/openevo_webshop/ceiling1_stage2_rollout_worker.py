#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
from typing import Any
from openevo_webshop.comparison_contract import extract_goal, format_webshop_actions, render_compact_webshop_prompt
from openevo_webshop.env_compat import WebshopEnv, WebshopEnvConfig
from scripts.openevo_webshop.webshop_comparison_contract import parse_action_command_contract, repair_action_command
from scripts.openevo_webshop.h145_rollouts import load_rows
from scripts.openevo_webshop.parametric_rnd import FixedEnvironmentSeed, gpu_snapshot
from scripts.openevo_webshop.ceiling1_runtime import CeilingRuntime
from scripts.openevo_webshop.ceiling1_harness2_runtime import Harness201Runtime
from scripts.openevo_webshop.ceiling1_harness201_prompt import PROMPT_CONTRACT_SHA256, PROMPT_PROFILE, render_harness201_compact_webshop_prompt
from scripts.openevo_webshop.ceiling1_harness2011_prompt import PROMPT_CONTRACT_SHA256 as H2011_PROMPT_SHA256, PROMPT_PROFILE as H2011_PROMPT_PROFILE, render_harness2011_compact_webshop_prompt
from scripts.openevo_webshop.ceiling1_harness2011_runtime import Harness2011Runtime
MAX_STEPS=15; MAX_NEW=512; SEED_BASE=148_000_000; STEP_SEED_STRIDE=100_000

def step_generation_seed(rollout_seed:int, step_idx:int, policy:str)->int:
    if policy=="legacy": return rollout_seed+step_idx
    if policy=="injective-v1": return rollout_seed+step_idx*STEP_SEED_STRIDE
    raise ValueError(f"unknown step seed policy {policy}")

def atomic_json(p:Path,x:Any)->None:
    p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name('.'+p.name+'.tmp'); t.write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n'); t.replace(p)
def sha256(p:Path|None)->str|None:
    if p is None: return None
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
def admissible(action:str, available:dict[str,Any])->bool:
    if action.startswith('search['): return bool(available.get('has_search_bar'))
    return action in format_webshop_actions(available)

def verifier_snapshot(env:Any, session_index:int)->dict[str,Any]:
    raw=env._build(); server=raw.server; sess=server.user_sessions[str(int(session_index))]; goal=sess['goal']
    return {'goal_asin':goal.get('asin'),'goal_options':goal.get('goal_options'),'asin':sess.get('asin'),'options':dict(sess.get('options') or {})}

def run_episode(env:Any, rt:Any, task:dict[str,Any], seed:int, *, harness_profile:str, step_seed_policy:str)->dict[str,Any]:
    obs,info=env.reset(int(task['source_session_index']),seed=int(task['worker_seed'])); goal=extract_goal(obs); hist=[]; trace=[]; repaired_any=False; env_steps=0
    for step_idx in range(MAX_STEPS):
        available=env.get_available_actions(); verifier_before=verifier_snapshot(env,int(task['source_session_index'])) if harness_profile=='201.1' else None
        if harness_profile=='201.1': renderer=render_harness2011_compact_webshop_prompt
        elif harness_profile=='2.0.1': renderer=render_harness201_compact_webshop_prompt
        else: renderer=render_compact_webshop_prompt
        prompt=renderer(task_description=goal,current_observation=obs,available_actions=available,step_count=step_idx,history_length=2,action_history=' | '.join(hist[-2:]) or '(none)',max_prompt_chars=20_000)
        generation_seed=step_generation_seed(seed,step_idx,step_seed_policy)
        kwargs=dict(generation_seed=generation_seed,max_new_tokens=MAX_NEW,do_sample=True,temperature=1.0,top_p=1.0,top_k=0,repetition_penalty=1.0)
        if harness_profile=='201.1':
            try: raw,receipt=rt.generate_h2(prompt,available_actions=available,**kwargs)
            except RuntimeError as exc:
                if 'action prefix escaped' in str(exc) or 'no compilable action schema' in str(exc):
                    trace.append({'step':step_idx,'base_prompt':prompt,'generation_seed':generation_seed,'step_seed_policy':step_seed_policy,'constraint_escape':True,'parser_repair_used':False,'error':f'{type(exc).__name__}: {exc}'})
                    return {'steps':step_idx+1,'task_score':0.0,'won':False,'trace':trace,'trace_complete':False,'terminated_on_invalid_action':True,'parser_repair_used':False,'constraint_escape':True,'environment_action_schema_violation':False,'webshop_env_step_calls':env_steps,'interface_failure_kind':'constraint_escape'}
                raise
            ch=receipt.get('action_channel') or {}; action=ch.get('action'); valid=bool(ch.get('admissible')); repaired=False; prepared=raw
        else:
            raw,receipt=(rt.generate_h2(prompt,available_actions=available,**kwargs) if harness_profile=='2.0.1' else rt.generate(prompt,**kwargs))
            prepared,action,repaired=repair_action_command(raw); parsed,valid0=parse_action_command_contract(prepared,require_think=True); valid=bool(valid0 and action==parsed and admissible(action,available)); ch=None
        step={'step':step_idx,'base_prompt':prompt,'prompt':receipt.get('decoded_input') or receipt.get('effective_user_prompt') or prompt,'runtime_messages':receipt.get('messages'),'runtime_sampling':receipt.get('sampling'),'runtime_active_adapters':receipt.get('active_adapters'),'runtime_adapters':receipt.get('adapters'),'runtime_carrier_readback':receipt.get('carrier_readback'),'runtime_harness':receipt.get('harness') or 'legacy-ceiling-runtime','runtime_context_receipt':receipt if harness_profile in {'2.0.1','201.1'} else None,'raw_completion':raw,'repaired_completion':prepared if repaired else None,'action':action if valid else None,'action_valid':valid,'parser_failure':not valid,'parser_repair_used':bool(repaired),'generation_seed':generation_seed,'step_seed_policy':step_seed_policy,'runtime_input_tokens':receipt.get('input_tokens'),'constraint_escape':False}
        if ch is not None: step['action_channel']=ch
        if verifier_before is not None: step['verifier_state']={'before':verifier_before,'after':None,'environment_step_executed':False}
        trace.append(step)
        if not valid:
            kind='missing_action_channel' if harness_profile=='201.1' and not ch.get('action_channel_entered') else 'inadmissible_action_channel_output' if harness_profile=='201.1' else 'parser_or_state_invalid'
            return {'steps':step_idx+1,'task_score':0.0,'won':False,'trace':trace,'trace_complete':False,'terminated_on_invalid_action':True,'parser_repair_used':repaired_any or bool(repaired),'constraint_escape':False,'environment_action_schema_violation':False,'webshop_env_step_calls':env_steps,'interface_failure_kind':kind}
        repaired_any=repaired_any or bool(repaired)
        try: obs,reward,done,info=env.step(action); env_steps+=1
        except BaseException as exc:
            if harness_profile!='201.1': raise
            step['environment_error']=f'{type(exc).__name__}: {exc}'; step['environment_action_schema_violation']=True
            return {'steps':step_idx+1,'task_score':0.0,'won':False,'trace':trace,'trace_complete':False,'terminated_on_invalid_action':True,'parser_repair_used':False,'constraint_escape':False,'environment_action_schema_violation':True,'webshop_env_step_calls':env_steps,'interface_failure_kind':'environment_rejected_compiled_action'}
        info=info or {}; hist.append(f"obs:{str(obs)[:60]} -> act:{action}"); step['task_score']=float(info.get('task_score',reward) or 0.0); step['environment_accepted']=True
        if verifier_before is not None: step['verifier_state']={'before':verifier_before,'after':verifier_snapshot(env,int(task['source_session_index'])),'environment_step_executed':True,'terminal_task_score':float(info.get('task_score',0) or 0) if done else None}
        if done:
            won=bool(info.get('won',reward==1.0)); return {'steps':step_idx+1,'task_score':max(float(step['task_score']),0.0),'won':won,'trace':trace,'trace_complete':True,'terminated_on_invalid_action':False,'parser_repair_used':repaired_any,'constraint_escape':False,'environment_action_schema_violation':False,'webshop_env_step_calls':env_steps,'interface_failure_kind':None}
    return {'steps':MAX_STEPS,'task_score':max((float(x.get('task_score',0.0)) for x in trace),default=0.0),'won':False,'trace':trace,'trace_complete':True,'terminated_on_invalid_action':False,'parser_repair_used':repaired_any,'constraint_escape':False,'environment_action_schema_violation':False,'webshop_env_step_calls':env_steps,'interface_failure_kind':None}

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument('--schedule',type=Path,required=True); p.add_argument('--start-position',type=int,required=True); p.add_argument('--end-position',type=int,required=True); p.add_argument('--seed-root',type=Path,required=True)
    p.add_argument('--model',required=True); p.add_argument('--model-revision',required=True); p.add_argument('--sd-adapter',type=Path); p.add_argument('--memory',type=Path,required=True); p.add_argument('--skill',type=Path,required=True); p.add_argument('--agent-system',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--gpu-uuid',required=True); p.add_argument('--round-index',type=int,required=True); p.add_argument('--resume',action='store_true'); p.add_argument('--state-id',required=True); p.add_argument('--experiment-git-sha',required=True); p.add_argument('--config-sha256',required=True); p.add_argument('--harness-profile',choices=('legacy','2.0.1','201.1'),default='legacy'); p.add_argument('--step-seed-policy',choices=('legacy','injective-v1'),default='legacy'); p.add_argument('--qualification-only',action='store_true'); a=p.parse_args()
    if (a.harness_profile in {'2.0.1','201.1'}) != (a.step_seed_policy=='injective-v1'): raise SystemExit('Harness successor profiles require injective-v1 as one bundle')
    if os.environ.get('CUDA_VISIBLE_DEVICES','').strip()!=a.gpu_uuid: raise SystemExit('CUDA_VISIBLE_DEVICES mismatch')
    tasks=load_rows(a.schedule.resolve(),a.start_position,a.end_position)
    if any(x.get('stage')!='stage2' for x in tasks): raise SystemExit('non-stage2 schedule row')
    mem=a.memory.read_text(); skill=a.skill.read_text(); agent=a.agent_system.read_text(); a.output.mkdir(parents=True,exist_ok=True); (a.output/'episodes').mkdir(exist_ok=True)
    resume_identity={'state_id':a.state_id,'round_index':a.round_index,'memory_sha256':sha256(a.memory),'skill_sha256':sha256(a.skill),'agent_system_sha256':sha256(a.agent_system),'sd_adapter_sha256':sha256(a.sd_adapter/'adapter_model.safetensors') if a.sd_adapter else None,'experiment_git_sha':a.experiment_git_sha,'config_sha256':a.config_sha256,'harness_profile':a.harness_profile,'step_seed_policy':a.step_seed_policy,'prompt_contract_sha256':H2011_PROMPT_SHA256 if a.harness_profile=='201.1' else PROMPT_CONTRACT_SHA256 if a.harness_profile=='2.0.1' else None}
    if a.resume:
      for task in tasks:
        pos=int(task['position']); ep_path=a.output/'episodes'/f'{pos:06d}-{task["task_id"]}.json'
        if not ep_path.is_file(): continue
        prior=json.loads(ep_path.read_text())
        if int(prior.get('schedule_position',-1))!=pos or any(prior.get(key)!=value for key,value in resume_identity.items()): raise RuntimeError('resume episode identity mismatch before model execution')
    atomic_json(a.output/'gpu-before.json',gpu_snapshot(a.gpu_uuid)); runtime_cls=Harness2011Runtime if a.harness_profile=='201.1' else Harness201Runtime if a.harness_profile=='2.0.1' else CeilingRuntime; rt=runtime_cls(a.model,a.model_revision,sd_adapter=a.sd_adapter,memory=mem,skill_bundle=skill,agent_system=agent)
    rows=[]; started=time.time()
    try:
      for task in tasks:
        pos=int(task['position']); ep_path=a.output/'episodes'/f'{pos:06d}-{task["task_id"]}.json'
        if a.resume and ep_path.is_file():
            prior=json.loads(ep_path.read_text())
            if not prior.get('engineering_invalid'):
                rows.append(prior); continue
        raw_env=WebshopEnv(WebshopEnvConfig(seed_root=str(a.seed_root),num_products=1000,seed=int(task['worker_seed']))); error=None
        try: ep=run_episode(FixedEnvironmentSeed(raw_env,int(task['worker_seed'])),rt,task,SEED_BASE+pos,harness_profile=a.harness_profile,step_seed_policy=a.step_seed_policy)
        except BaseException as exc: ep={'steps':0,'task_score':0.0,'won':False,'trace':[],'trace_complete':False,'terminated_on_invalid_action':False,'parser_repair_used':False}; error=f'{type(exc).__name__}: {exc}'
        finally: raw_env.close()
        tr=ep['trace']; identity_ok=bool(tr and task['instruction'] in tr[0].get('prompt','')); parser_failure=bool(error) or any(x.get('parser_failure') for x in tr)
        row={'schema':'openevo.ceiling1-stage2-rollout.v1','schedule_position':pos,'round_index':a.round_index,'task_id':task['task_id'],'worker_seed':int(task['worker_seed']),'source_session_index':int(task['source_session_index']),'instruction_sha256':task['instruction_sha256'],'rollout_seed':SEED_BASE+pos,'final_task_score':float(ep.get('task_score',0.0)),'won':bool(ep.get('won')),'identity_ok':identity_ok,'trace_complete':bool(ep.get('trace_complete')),'parser_or_fallback_failure':parser_failure,'parser_repair_used':bool(ep.get('parser_repair_used')),'terminated_on_invalid_action':bool(ep.get('terminated_on_invalid_action')),'task_bearing_budget_consumed':not bool(error),'qualification_only':bool(a.qualification_only),'formal_task_consumption':0 if a.qualification_only else None,'final_panel_access':0,'external_teacher_calls':0,'engineering_invalid':bool(error),'error':error,'episode':ep,'state_id':a.state_id,'memory_sha256':sha256(a.memory),'skill_sha256':sha256(a.skill),'agent_system_sha256':sha256(a.agent_system),'sd_adapter_sha256':sha256(a.sd_adapter/'adapter_model.safetensors') if a.sd_adapter else None,'experiment_git_sha':a.experiment_git_sha,'config_sha256':a.config_sha256,'harness_profile':a.harness_profile,'step_seed_policy':a.step_seed_policy,'prompt_profile':H2011_PROMPT_PROFILE if a.harness_profile=='201.1' else PROMPT_PROFILE if a.harness_profile=='2.0.1' else None,'prompt_contract_sha256':H2011_PROMPT_SHA256 if a.harness_profile=='201.1' else PROMPT_CONTRACT_SHA256 if a.harness_profile=='2.0.1' else None}
        atomic_json(ep_path,row); rows.append(row)
      rows.sort(key=lambda r:int(r['schedule_position'])); expected=list(range(a.start_position,a.end_position+1))
      if [int(r['schedule_position']) for r in rows]!=expected: raise RuntimeError('shard schedule coverage mismatch')
      (a.output/'attempts.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows))
      clean=lambda r: r['final_task_score']==1.0 and r['won'] and r['identity_ok'] and r['trace_complete'] and not r['parser_or_fallback_failure'] and not r['parser_repair_used'] and not r['engineering_invalid'] and not r['error']
      atomic_json(a.output/'summary.json',{'schema':'openevo.ceiling1-stage2-rollout-shard-summary.v1','status':'pass','round_index':a.round_index,'start_position':a.start_position,'end_position':a.end_position,'attempt_count':len(rows),'engineering_invalid_count':sum(int(r['engineering_invalid']) for r in rows),'clean_exact_success_count':sum(int(clean(r)) for r in rows),'mean_score':sum(r['final_task_score'] for r in rows)/len(rows),'gpu_uuid':a.gpu_uuid,'state_id':a.state_id,'harness_profile':a.harness_profile,'step_seed_policy':a.step_seed_policy,'qualification_only':bool(a.qualification_only),'prompt_profile':H2011_PROMPT_PROFILE if a.harness_profile=='201.1' else PROMPT_PROFILE if a.harness_profile=='2.0.1' else None,'prompt_contract_sha256':H2011_PROMPT_SHA256 if a.harness_profile=='201.1' else PROMPT_CONTRACT_SHA256 if a.harness_profile=='2.0.1' else None,'sampling':{'temperature':1.0,'do_sample':True,'top_p':1.0,'top_k':0,'repetition_penalty':1.0,'max_steps':15,'max_new_tokens':512,'generation_seed_base':SEED_BASE},'wall_time_seconds':time.time()-started})
      atomic_json(a.output/'gpu-after.json',gpu_snapshot(a.gpu_uuid))
    finally: rt.close()
    return 0
if __name__=='__main__': raise SystemExit(main())
