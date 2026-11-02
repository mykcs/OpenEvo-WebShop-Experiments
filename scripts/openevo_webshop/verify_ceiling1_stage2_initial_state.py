#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
from scripts.openevo_webshop.ceiling1_harness201_carrier_contract import require_evolution_compatible_bundle
from scripts.openevo_webshop.ceiling1_runtime import CeilingRuntime
from scripts.openevo_webshop.ceiling1_harness2_runtime import Harness201Runtime
from scripts.openevo_webshop.ceiling1_harness201_prompt import PROMPT_CONTRACT_SHA256, PROMPT_PROFILE
from scripts.openevo_webshop.ceiling1_harness2011_prompt import PROMPT_CONTRACT_SHA256 as H2011_PROMPT_CONTRACT_SHA256, PROMPT_PROFILE as H2011_PROMPT_PROFILE
from scripts.openevo_webshop.ceiling1_harness2011_runtime import Harness2011Runtime
from scripts.openevo_webshop.ceiling1_harness2011_runtime_qwen3 import Qwen3Harness2011Runtime

def sha(p:Path): h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
def main()->int:
 p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--revision',required=True); p.add_argument('--memory',type=Path,required=True); p.add_argument('--skill',type=Path,required=True); p.add_argument('--agent-system',type=Path,required=True); p.add_argument('--gpu-uuid',required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--stage1-state-id',required=True); p.add_argument('--stage2-config-sha256',required=True); p.add_argument('--harness-profile',choices=('legacy','2.0.1','201.1'),default='legacy'); p.add_argument('--carrier-seed-receipt',type=Path); p.add_argument('--chat-template-enable-thinking',choices=('auto','false'),default='auto'); a=p.parse_args()
 if os.environ.get('CUDA_VISIBLE_DEVICES','').strip()!=a.gpu_uuid: raise SystemExit('CUDA_VISIBLE_DEVICES mismatch')
 mem=a.memory.read_text(); skill=a.skill.read_text(); agent=a.agent_system.read_text()
 if a.harness_profile=='legacy':
  rt=CeilingRuntime(a.model,a.revision,memory=mem,skill_bundle=skill,agent_system=agent)
  try: _,r=rt.prepare('TRAIN-ONLY PRECHECK: locate a product while respecting all explicit constraints.')
  finally: rt.close()
  if r['active_adapters']: raise RuntimeError('post-OPSD merged base must not require a bootstrap adapter at Stage2 runtime')
  for k in ('text_memory','skill_bundle','agent_system'):
   if not r['carrier_readback'][k]['requested'] or not r['carrier_readback'][k]['present_in_decoded_input']: raise RuntimeError(f'{k} absent from legacy Stage2 runtime')
  readback={'legacy':r}; carrier_seed_receipt_sha256=None
 else:
  if not a.carrier_seed_receipt or not a.carrier_seed_receipt.is_file(): raise RuntimeError('Harness201 successor preflight requires carrier seed receipt')
  seed=json.loads(a.carrier_seed_receipt.read_text())
  if seed.get('status')!='PASS' or seed.get('schema')!='openevo.ceiling1-harness2.0.1-carrier-finalize-receipt.v2': raise RuntimeError('Harness201 carrier seed receipt invalid')
  require_evolution_compatible_bundle(mem,skill,agent,context='new Stage2 runtime preflight')
  runtime_cls=(Qwen3Harness2011Runtime if a.chat_template_enable_thinking=='false' else Harness2011Runtime) if a.harness_profile=='201.1' else Harness201Runtime
  rt=runtime_cls(a.model,a.revision,sd_adapter=None,memory=mem,skill_bundle=skill,agent_system=agent)
  expected_harness='201.1' if a.harness_profile=='201.1' else '2.0.1-thin'
  try:
   search_enc,search=rt.prepare_h2('TRAIN-ONLY PRECHECK SEARCH',available_actions={'has_search_bar':True,'clickables':['search']})
   results_enc,results=rt.prepare_h2('TRAIN-ONLY PRECHECK RESULTS',available_actions={'has_search_bar':False,'clickables':['example-product']})
  finally: rt.close()
  for rec in (search,results):
   if rec.get('harness')!=expected_harness or (rec.get('encoded_budget') or {}).get('status')!='PASS': raise RuntimeError('Harness201 runtime/budget readback failed')
   if a.harness_profile=='201.1' and a.chat_template_enable_thinking=='false' and rec.get('chat_template_enable_thinking') is not False: raise RuntimeError('Qwen3 201.1 thinking readback failed')
   if sum(int((rec.get('carrier_readback') or {}).get(k,{}).get('selected')) for k in ('text_memory','skill_bundle'))>1: raise RuntimeError('Harness201 selected more than one non-global carrier')
   if not (rec.get('carrier_readback') or {}).get('agent_system',{}).get('selected'): raise RuntimeError('Harness201 Agent global authority missing')
  if not search['carrier_readback']['skill_bundle']['selected'] or search['carrier_readback']['text_memory']['selected']: raise RuntimeError('Harness201 Search route drift')
  if not results['carrier_readback']['text_memory']['selected'] or results['carrier_readback']['skill_bundle']['selected']: raise RuntimeError('Harness201 Results route drift')
  readback={'search':search,'results':results}; carrier_seed_receipt_sha256=sha(a.carrier_seed_receipt)
 prompt_profile=(H2011_PROMPT_PROFILE if a.harness_profile=='201.1' else (PROMPT_PROFILE if a.harness_profile=='2.0.1' else None))
 prompt_sha=(H2011_PROMPT_CONTRACT_SHA256 if a.harness_profile=='201.1' else (PROMPT_CONTRACT_SHA256 if a.harness_profile=='2.0.1' else None))
 model_context=int(json.loads((Path(a.model)/'config.json').read_text()).get('max_position_embeddings',0)) if Path(a.model).is_dir() else 0
 if a.harness_profile=='201.1' and (model_context<=0 or any(int(rec.get('input_tokens',0))>model_context for rec in (search,results))): raise RuntimeError('Harness201.1 model context readback failed')
 x={'schema':'openevo.ceiling1-stage2-initial-runtime-preflight.v2','status':'PASS','stage1_state_id':a.stage1_state_id,'stage2_config_sha256':a.stage2_config_sha256,'model':a.model,'harness_profile':a.harness_profile,'prompt_profile':prompt_profile,'prompt_contract_sha256':prompt_sha,'chat_template_enable_thinking':False if a.chat_template_enable_thinking=='false' else None,'context_window':model_context if a.harness_profile=='201.1' else None,'memory_sha256':sha(a.memory),'skill_sha256':sha(a.skill),'agent_system_sha256':sha(a.agent_system),'carrier_seed_receipt_sha256':carrier_seed_receipt_sha256,'runtime_readback':readback,'gpu_uuid':a.gpu_uuid,'formal_webshop_calls':0,'final_panel_consumption':0}
 a.output.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); print(json.dumps(x,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
