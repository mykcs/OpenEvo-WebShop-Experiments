#!/usr/bin/env python3
from __future__ import annotations
import argparse, collections, hashlib, json, time
from pathlib import Path

FREEZE_ID='202609030400'
PARENT_FREEZE_ID='202609021800'
SCHEDULE_SHA='1b3ac4252cda4d25d9542713e68e2fa4aacd90ceb377e5c03c8c1ad7c6878428'
PROFILE='harness201.1-stage1-shared-seed-native-min-deliberation-v1'
PROMPT_PROFILE='harness201.1-stage1-shared-seed-native-soft-deliberation-v1'
PROMPT_CONTRACT='76b780231fe0298b44ddb25c397537ab81cfdba645258f5643a49514a50b7e55'
ARM_SPEC={
 'qwen25-3b':{'revision':'aa8e72537993ba99e69dfaafa59ed015b17504d1','dtype':'bfloat16','thinking':None},
 'qwen3-1p7b':{'revision':'70d244cc86ccca08cf5af4e1e306ecf908b1ad5e','dtype':'float16','thinking':False},
}

def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def read_rows(paths):
    rows=[]
    for p in paths:
        with p.open(encoding='utf-8') as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return sorted(rows,key=lambda r:int(r['schedule_position']))
def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--arm',choices=tuple(ARM_SPEC),required=True)
    ap.add_argument('--attempts',type=Path,action='append',required=True)
    ap.add_argument('--process-exit',type=int,action='append',required=True)
    ap.add_argument('--execution-sha',required=True)
    ap.add_argument('--design',type=Path,required=True)
    ap.add_argument('--freeze-manifest',type=Path,required=True)
    ap.add_argument('--output-root',type=Path,required=True)
    a=ap.parse_args()
    rows=read_rows(a.attempts); spec=ARM_SPEC[a.arm]
    design=json.loads(a.design.read_text(encoding='utf-8'))
    manifest=json.loads(a.freeze_manifest.read_text(encoding='utf-8'))
    design_sha=sha(a.design); freeze_sha=sha(a.freeze_manifest)
    mechanics=collections.Counter()
    task_counts=collections.Counter()
    for r in rows:
        task_counts[(r.get('worker_seed'),r.get('source_session_index'),r.get('instruction_sha256'))]+=1
        ep=r.get('episode') or {}
        mechanics['engineering']+=int(bool(r.get('engineering_invalid')))
        mechanics['invalid']+=int(bool(r.get('terminated_on_invalid_action')))
        mechanics['repair']+=int(bool(r.get('parser_repair_used')))
        mechanics['incomplete']+=int(ep.get('trace_complete') is not True)
        mechanics['constraint']+=int(bool(ep.get('constraint_escape')))
        mechanics['env_schema']+=int(bool(ep.get('environment_action_schema_violation')))
        for st in ep.get('trace') or []:
            rec=st.get('runtime_context_receipt') or {}; ch=st.get('action_channel') or rec.get('action_channel') or {}; samp=rec.get('sampling') or {}; contract=rec.get('successor_contract') or {}
            mechanics['missing_channel']+=int(not ch)
            mechanics['inadmissible']+=int(bool(ch) and ch.get('admissible') is not True)
            mechanics['sampling_drift']+=int(not (samp.get('temperature')==1.0 and samp.get('top_p')==1.0 and samp.get('top_k')==0))
            mechanics['min_contract_drift']+=int(not (contract.get('minimum_pre_action_model_tokens')==4 and contract.get('minimum_deliberation_enforced_by_interface') is True and contract.get('deliberation_content_guidance') is False))
            mechanics['empty_deliberation']+=int(contract.get('observed_pre_action_nonempty') is not True)
    checks={
      'design_frozen':design.get('freeze_id')==FREEZE_ID and design.get('status')=='FROZEN',
      'manifest_frozen':manifest.get('freeze_id')==FREEZE_ID and manifest.get('status')=='FROZEN',
      'attempt_count_1440':len(rows)==1440,
      'positions_exact':[int(r['schedule_position']) for r in rows]==list(range(1,1441)),
      'one_eighty_tasks_x_eight':len(task_counts)==180 and set(task_counts.values())=={8},
      'freeze_id_exact':all(r.get('freeze_id')==FREEZE_ID and r.get('shared_harness_freeze_id')==FREEZE_ID for r in rows),
      'upstream_freeze_exact':all(r.get('upstream_source_freeze_id')==PARENT_FREEZE_ID for r in rows),
      'shared_design_exact':all(r.get('shared_harness_design_sha256')==design_sha for r in rows),
      'shared_freeze_manifest_exact':all(r.get('shared_harness_freeze_manifest_sha256')==freeze_sha for r in rows),
      'arm_exact':all(r.get('arm')==a.arm for r in rows),
      'execution_sha_exact':all(r.get('experiment_git_sha')==a.execution_sha for r in rows),
      'model_revision_exact':all(r.get('model_revision')==spec['revision'] for r in rows),
      'dtype_exact':all(r.get('model_dtype')==spec['dtype'] for r in rows),
      'thinking_exact':all(r.get('chat_template_enable_thinking') is spec['thinking'] for r in rows),
      'harness_profile_exact':all(r.get('harness_profile')==PROFILE for r in rows),
      'prompt_profile_exact':all(r.get('prompt_profile')==PROMPT_PROFILE for r in rows),
      'prompt_contract_exact':all(r.get('prompt_contract_sha256')==PROMPT_CONTRACT for r in rows),
      'schedule_sha_exact':all(r.get('source_schedule_sha256')==SCHEDULE_SHA for r in rows),
      'strategy_neutral':all(r.get('prompt_strategy_neutral') is True for r in rows),
      'qualification_only_false':all(r.get('qualification_only') is False for r in rows),
      'zero_teacher_calls':all(int(r.get('external_teacher_calls',0))==0 for r in rows),
      'no_carriers':all(r.get('carrier_mode')=='none' and all(r.get(k) is None for k in ('memory_sha256','skill_sha256','agent_system_sha256','sd_adapter_sha256')) for r in rows),
      'all_process_exits_zero':bool(a.process_exit) and all(int(x)==0 for x in a.process_exit),
      'zero_engineering_invalid':mechanics['engineering']==0,'zero_invalid_action_termination':mechanics['invalid']==0,'zero_parser_repair':mechanics['repair']==0,'trace_complete':mechanics['incomplete']==0,
      'zero_constraint_escape':mechanics['constraint']==0,'zero_environment_schema_violation':mechanics['env_schema']==0,
      'action_channel_complete':mechanics['missing_channel']==0,'all_actions_admissible':mechanics['inadmissible']==0,'sampling_exact_t1':mechanics['sampling_drift']==0,'minimum_deliberation_contract_exact':mechanics['min_contract_drift']==0,'pre_action_nonempty_100pct':mechanics['empty_deliberation']==0,
    }
    a.output_root.mkdir(parents=True,exist_ok=True)
    corpus=a.output_root/f'{a.arm}-stage1-202609030400.jsonl'
    corpus.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows),encoding='utf-8')
    scores=[float(r.get('final_task_score',0) or 0) for r in rows]
    steps=[int((r.get('episode') or {}).get('steps',0) or 0) for r in rows]
    result={
      'schema':'openevo.shared-stage1.formal-closeout.v1','freeze_id':FREEZE_ID,
      'arm':a.arm,'status':'PASS' if all(checks.values()) else 'FAIL',
      'execution_git_sha':a.execution_sha,'checks':checks,'mechanics':dict(mechanics),
      'behavioral_metrics_not_pass_gates':{
        'positive_score_count':sum(x>0 for x in scores),
        'exact_success_count':sum(bool(r.get('won')) and float(r.get('final_task_score',0) or 0)==1 for r in rows),
        'mean_raw_score':sum(scores)/max(1,len(scores)),
        'max_step_count':sum(x>=15 for x in steps),
        'mean_steps':sum(steps)/max(1,len(steps)),
      },
      'corpus_path':str(corpus),'corpus_sha256':sha(corpus),
      'design_sha256':design_sha,'freeze_manifest_sha256':freeze_sha,
      'source_attempts':[{'path':str(p),'sha256':sha(p)} for p in a.attempts],
      'process_exit_codes':list(map(int,a.process_exit)),
      'reward_or_success_used_as_pass_gate':False,
      'downstream_not_started':True,'sealed_at_epoch':time.time(),
    }
    receipt=a.output_root/f'STAGE1_COMPLETE_{FREEZE_ID}.json'
    receipt.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2))
    return 0 if result['status']=='PASS' else 1

if __name__=='__main__': raise SystemExit(main())
