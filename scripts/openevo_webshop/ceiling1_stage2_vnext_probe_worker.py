#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,os,sys,time
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from openevo_webshop.comparison_contract import extract_goal
from openevo_webshop.env_compat import WebshopEnv,WebshopEnvConfig
from scripts.openevo_webshop.ceiling1_harness2011_prompt import render_harness2011_compact_webshop_prompt,PROMPT_CONTRACT_SHA256
from scripts.openevo_webshop.ceiling1_stage2_format_hard_runtime import Stage2FormatHardRuntime,Qwen3Stage2FormatHardRuntime
from scripts.openevo_webshop.parametric_rnd import FixedEnvironmentSeed
MAX_STEPS=15;BASE=148000000;STRIDE=100000
def sha(p:Path|None)->str|None:return hashlib.sha256(p.read_bytes()).hexdigest() if p else None
def atomic(p:Path,x):p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name('.'+p.name+'.tmp');q.write_text(json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False)+'\n');q.replace(p)
def run_episode(env,rt,task):
    obs,_=env.reset(int(task['source_session_index']),seed=int(task['worker_seed']));goal=extract_goal(obs);hist=[];trace=[];env_steps=0;pos=int(task['position'])
    for step in range(MAX_STEPS):
        avail=env.get_available_actions();prompt=render_harness2011_compact_webshop_prompt(task_description=goal,current_observation=obs,available_actions=avail,step_count=step,history_length=2,action_history=' | '.join(hist[-2:]) or '(none)',max_prompt_chars=20000);seed=BASE+pos+step*STRIDE
        try:raw,rec=rt.generate_h2(prompt,available_actions=avail,generation_seed=seed,max_new_tokens=512,do_sample=True,temperature=1.0,top_p=1.0,top_k=0,repetition_penalty=1.0)
        except RuntimeError as exc:
            if 'action prefix escaped' in str(exc) or 'no compilable action schema' in str(exc):return {'trace':trace,'steps':step+1,'score':0.0,'won':False,'invalid_terminated':True,'constraint_escape':True,'environment_action_schema_violation':False,'webshop_env_step_calls':env_steps}
            raise
        ch=rec.get('action_channel') or {};action=ch.get('action');valid=bool(ch.get('admissible'));trace.append({'step':step,'generation_seed':seed,'raw_completion':raw,'raw_completion_sha256':hashlib.sha256(raw.encode()).hexdigest(),'action':action if valid else None,'action_valid':valid,'action_channel':ch,'runtime_input_tokens':rec.get('input_tokens'),'harness_readback':rec.get('harness')})
        if not valid:return {'trace':trace,'steps':step+1,'score':0.0,'won':False,'invalid_terminated':True,'constraint_escape':False,'environment_action_schema_violation':False,'webshop_env_step_calls':env_steps}
        try:obs,reward,done,info=env.step(action);env_steps+=1
        except BaseException as exc:trace[-1]['environment_error']=f'{type(exc).__name__}: {exc}';return {'trace':trace,'steps':step+1,'score':0.0,'won':False,'invalid_terminated':True,'constraint_escape':False,'environment_action_schema_violation':True,'webshop_env_step_calls':env_steps}
        info=info or {};hist.append(f'act:{action} -> obs:{str(obs)[:60]}');trace[-1]['score']=float(info.get('task_score',reward) or 0.0)
        if done:return {'trace':trace,'steps':step+1,'score':float(info.get('task_score',reward) or 0.0),'won':bool(info.get('won',reward==1.0)),'invalid_terminated':False,'constraint_escape':False,'environment_action_schema_violation':False,'webshop_env_step_calls':env_steps}
    return {'trace':trace,'steps':MAX_STEPS,'score':max([float(s.get('score',0)) for s in trace] or [0]),'won':False,'invalid_terminated':False,'constraint_escape':False,'environment_action_schema_violation':False,'webshop_env_step_calls':env_steps}
def main()->int:
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('3b','qwen3-1p7b'),default='3b');p.add_argument('--mode',choices=('pre','post'),required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--seed-root',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--revision',required=True);p.add_argument('--sd-adapter',type=Path);p.add_argument('--memory',type=Path,required=True);p.add_argument('--skill',type=Path,required=True);p.add_argument('--agent',type=Path,required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES','').strip()!=a.gpu_uuid:raise SystemExit('CUDA_VISIBLE_DEVICES mismatch')
    if a.mode=='pre' and a.sd_adapter is not None:raise RuntimeError('pre probe must have no SD adapter')
    if a.mode=='post' and a.sd_adapter is None:raise RuntimeError('post probe requires SD adapter')
    rows=json.load(open(a.manifest))['rows'];runtime_cls=Qwen3Stage2FormatHardRuntime if a.arm=='qwen3-1p7b' else Stage2FormatHardRuntime;rt=runtime_cls(a.model,a.revision,sd_adapter=a.sd_adapter,memory=a.memory.read_text(),skill_bundle=a.skill.read_text(),agent_system=a.agent.read_text());out=[];started=time.time()
    try:
        for task in rows:
            env0=WebshopEnv(WebshopEnvConfig(seed_root=str(a.seed_root),num_products=1000,seed=int(task['worker_seed'])));err=None
            try:ep=run_episode(FixedEnvironmentSeed(env0,int(task['worker_seed'])),rt,task)
            except BaseException as exc:ep={'trace':[],'steps':0,'score':0.0,'won':False,'invalid_terminated':False,'constraint_escape':False,'environment_action_schema_violation':False,'webshop_env_step_calls':0};err=f'{type(exc).__name__}: {exc}'
            finally:env0.close()
            out.append({'position':int(task['position']),'task_id':task['task_id'],'instruction_sha256':task['instruction_sha256'],'worker_seed':int(task['worker_seed']),'source_session_index':int(task['source_session_index']),'mode':a.mode,'adapter_sha256':sha(a.sd_adapter/'adapter_model.safetensors') if a.sd_adapter else None,'engineering_invalid':bool(err),'error':err,'episode':ep})
    finally:rt.close()
    summary={'schema':'openevo.ceiling1-harness2011-g3-shadow-probe.v1','status':'PASS' if len(out)==16 and not any(r['engineering_invalid'] for r in out) and not any(r['episode']['invalid_terminated'] or r['episode']['constraint_escape'] or r['episode']['environment_action_schema_violation'] for r in out) and all(all(bool((st.get('action_channel') or {}).get('interface_owned_action_opener')) and bool((st.get('action_channel') or {}).get('admissible')) for st in r['episode']['trace']) for r in out) else 'FAIL','arm':a.arm,'mode':a.mode,'qualification_only':True,'formal_task_consumption':0,'final_panel_access':0,'external_teacher_calls':0,'attempt_count':len(out),'exact_success':sum(r['episode']['won'] for r in out),'positive_score':sum(float(r['episode']['score'])>0 for r in out),'mean_score':sum(float(r['episode']['score']) for r in out)/max(1,len(out)),'invalid_terminated':sum(r['episode']['invalid_terminated'] for r in out),'constraint_escape_count':sum(r['episode']['constraint_escape'] for r in out),'environment_action_schema_violation_count':sum(r['episode']['environment_action_schema_violation'] for r in out),'rows':out,'wall_time_seconds':time.time()-started,'prompt_contract_sha256':PROMPT_CONTRACT_SHA256,'seed_policy':'injective-step-seed-v1'};atomic(a.output,summary);print(json.dumps({k:summary[k] for k in ['status','mode','attempt_count','exact_success','positive_score','mean_score','invalid_terminated','constraint_escape_count','environment_action_schema_violation_count']},sort_keys=True));return 0 if summary['status']=='PASS' else 2
if __name__=='__main__':raise SystemExit(main())
