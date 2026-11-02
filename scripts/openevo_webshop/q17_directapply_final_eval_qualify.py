#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path

from openevo_webshop.env_compat import WebshopEnv, WebshopEnvConfig
from scripts.openevo_webshop.ceiling1_stage2_format_hard_runtime import Qwen3Stage2FormatHardRuntime
from scripts.openevo_webshop.parametric_rnd import FixedEnvironmentSeed
from scripts.openevo_webshop.q17_directapply_final_eval_worker import episode, atomic_json

GOAL_IDS = [870, 4996, 6325, 5676]
SEEDS = [202609120001000 + i for i in range(4)]

def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument('--seed-root',type=Path,required=True)
    p.add_argument('--model',required=True); p.add_argument('--model-revision',required=True)
    p.add_argument('--sd-adapter',type=Path,required=True); p.add_argument('--memory',type=Path,required=True)
    p.add_argument('--skill',type=Path,required=True); p.add_argument('--agent-system',type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES','').strip()!=a.gpu_uuid: raise SystemExit('CUDA_VISIBLE_DEVICES mismatch')
    runtime=Qwen3Stage2FormatHardRuntime(a.model,a.model_revision,sd_adapter=a.sd_adapter,
        memory=a.memory.read_text(),skill_bundle=a.skill.read_text(),agent_system=a.agent_system.read_text())
    rows=[]; started=time.time()
    try:
        for i,(goal_idx,seed) in enumerate(zip(GOAL_IDS,SEEDS,strict=True)):
            env=WebshopEnv(WebshopEnvConfig(seed_root=str(a.seed_root),num_products=1000,seed=0))
            try: result=episode(FixedEnvironmentSeed(env,0),runtime,{'goal_idx':goal_idx,'generation_seed':seed})
            finally: env.close()
            rows.append({'index':i,'goal_idx':goal_idx,'generation_seed':seed,'score':float(result['task_score']),
                'won':bool(result['won']),'trace_complete':bool(result['trace_complete']),
                'invalid_terminated':bool(result.get('terminated_on_invalid_action')),
                'parser_repair_used':bool(result.get('parser_repair_used')),
                'interface_failure_kind':result.get('interface_failure_kind'),'environment_steps':int(result.get('webshop_env_step_calls',0))})
    finally: runtime.close()
    ok=len(rows)==4 and all(x['trace_complete'] and not x['invalid_terminated'] and not x['parser_repair_used'] and not x['interface_failure_kind'] for x in rows)
    out={'schema':'openevo.q17-directapply-final-interface-qualification.202609120001','status':'PASS' if ok else 'FAIL',
        'qualification_only':True,'source':'already-consumed Stage2 task identities outside held-out 0..499 pool',
        'final_panel_access':0,'training_mutation':False,'external_teacher_calls':0,'new_training_rollouts':0,
        'chat_template_enable_thinking':False,'do_sample':False,'rows':rows,'wall_time_seconds':time.time()-started}
    atomic_json(a.output,out); print(json.dumps({k:out[k] for k in ['status','final_panel_access','training_mutation']},sort_keys=True))
    return 0 if ok else 2

if __name__=='__main__': raise SystemExit(main())
