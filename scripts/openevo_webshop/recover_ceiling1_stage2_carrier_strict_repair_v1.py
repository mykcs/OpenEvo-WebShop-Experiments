#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
from scripts.openevo_webshop.ceiling1_stage2_reflector import render, system_prompt
from scripts.openevo_webshop.ceiling1_harness201_carrier_contract import validate_carrier

def sha(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p:Path,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name('.'+p.name+'.tmp');q.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');q.replace(p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=('3b','qwen3-1p7b'),required=True);ap.add_argument('--kind',choices=('skill_bundle','agent_system'),required=True);ap.add_argument('--evidence',type=Path,required=True);ap.add_argument('--prior',type=Path,required=True);ap.add_argument('--memory',type=Path);ap.add_argument('--skill',type=Path);ap.add_argument('--model',required=True);ap.add_argument('--sd-adapter',type=Path);ap.add_argument('--output-root',type=Path,required=True);ap.add_argument('--gpu-uuid',required=True);ap.add_argument('--state-id',required=True);ap.add_argument('--round-index',type=int,required=True);ap.add_argument('--existing-primary',type=Path,required=True);ap.add_argument('--existing-receipt',type=Path,required=True);ap.add_argument('--artifact-contract',required=True);a=ap.parse_args()
 if os.environ.get('CUDA_VISIBLE_DEVICES','').strip()!=a.gpu_uuid: raise SystemExit('CUDA_VISIBLE_DEVICES mismatch')
 rec=json.loads(a.existing_receipt.read_text());expected={'kind':a.kind,'round_index':a.round_index,'state_id':a.state_id,'evidence_sha256':sha(a.evidence),'prior_sha256':sha(a.prior),'artifact_contract':a.artifact_contract,'primary_calls':1,'repair_calls':0}
 if any(rec.get(k)!=v for k,v in expected.items()): raise RuntimeError('existing primary receipt identity drift')
 if rec.get('output_sha256')!=sha(a.existing_primary): raise RuntimeError('existing primary payload drift')
 expected_sd=sha(a.sd_adapter/'adapter_model.safetensors') if a.sd_adapter else None
 if rec.get('sd_adapter_sha256')!=expected_sd: raise RuntimeError('existing primary SD adapter identity drift')
 primary_check=validate_carrier(a.kind,a.existing_primary.read_text())
 if primary_check['status']!='FAIL': raise RuntimeError('existing primary is already strict-valid; repair not authorized')
 evidence=json.loads(a.evidence.read_text());prior=a.prior.read_text();memory=a.memory.read_text() if a.memory else '';skill=a.skill.read_text() if a.skill else ''
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from peft import PeftModel
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);base=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda').eval();model=base
 if a.sd_adapter: model=PeftModel.from_pretrained(base,str(a.sd_adapter),is_trainable=False,local_files_only=True,autocast_adapter_dtype=False).eval()
 user,projection=render(a.kind,evidence,prior,memory,skill,repair=True,artifact_contract=a.artifact_contract);messages=[{'role':'system','content':system_prompt(a.kind,a.artifact_contract)},{'role':'user','content':user}]
 kwargs={'add_generation_prompt':True,'return_tensors':'pt','return_dict':True};thinking=False if a.arm=='qwen3-1p7b' else None
 if thinking is False: kwargs['enable_thinking']=False
 enc=tok.apply_chat_template(messages,**kwargs);n=int(enc['input_ids'].shape[1]);context=int(getattr(model.config,'max_position_embeddings',0) or 0);expected_context=40960 if a.arm=='qwen3-1p7b' else 32768
 if context!=expected_context or n+1024>context: raise RuntimeError('strict repair context contract drift/overflow')
 enc={k:v.to(model.device) for k,v in enc.items()};started=time.time()
 with torch.no_grad(): out=model.generate(**enc,max_new_tokens=1024,do_sample=False,pad_token_id=tok.eos_token_id)
 ids=out[0][n:];n_out=int(ids.shape[0]);candidate=tok.decode(ids,skip_special_tokens=True).strip();check=validate_carrier(a.kind,candidate);a.output_root.mkdir(parents=True,exist_ok=True);path=a.output_root/'repair.md';path.write_text(candidate.rstrip()+'\n')
 pbytes=json.dumps(projection,ensure_ascii=False,sort_keys=True).encode();receipt={'schema':'openevo.ceiling1-stage2-carrier-strict-repair.v1','status':'PASS' if check['status']=='PASS' and n_out<1024 else 'REJECTED','arm':a.arm,'kind':a.kind,'round_index':a.round_index,'state_id':a.state_id,'artifact_contract':a.artifact_contract,'evidence_sha256':sha(a.evidence),'prior_sha256':sha(a.prior),'memory_sha256':sha(a.memory) if a.memory else None,'skill_sha256':sha(a.skill) if a.skill else None,'sd_adapter_sha256':expected_sd,'existing_primary_path':str(a.existing_primary.resolve()),'existing_primary_sha256':sha(a.existing_primary),'existing_primary_receipt_sha256':sha(a.existing_receipt),'existing_primary_strict_validation':primary_check,'repair_path':str(path),'repair_sha256':sha(path),'repair_input_tokens':n,'repair_output_tokens':n_out,'repair_strict_validation':check,'projection_schema':projection['schema'],'projection_sha256':hashlib.sha256(pbytes).hexdigest(),'projection_bytes':len(pbytes),'primary_calls_total':1,'repair_calls_total':1,'new_model_calls':1,'external_teacher_calls':0,'formal_rollout_replay':0,'do_sample':False,'chat_template_enable_thinking':thinking,'context_window':context,'max_new_tokens':1024,'wall_time_seconds':time.time()-started}
 atomic(a.output_root/'repair-receipt.json',receipt);print(json.dumps(receipt,sort_keys=True));return 0 if receipt['status']=='PASS' else 3
if __name__=='__main__': raise SystemExit(main())
