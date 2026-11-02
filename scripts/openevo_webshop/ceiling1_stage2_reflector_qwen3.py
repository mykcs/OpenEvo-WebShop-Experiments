#!/usr/bin/env python3
"""Qwen3 binding for the shared bounded Stage2 reflector."""
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
from scripts.openevo_webshop.ceiling1_stage2_reflector import (
    DEFAULT_ARTIFACT_CONTRACT, H201_RUNTIME_AGENT_CONTRACT, H201_RUNTIME_MEMORY_CONTRACT,
    H201_RUNTIME_SKILL_CONTRACT, MAX_NEW_TOKENS, accepted_attempt_health, atomic_json,
    candidate_validation, load_scope_manifest, mechanically_valid, record_rejected_candidate,
    render, sha, system_prompt,
)

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument('--kind',choices=('text_memory','skill_bundle','agent_system'),required=True); p.add_argument('--evidence',type=Path,required=True); p.add_argument('--prior',type=Path,required=True); p.add_argument('--memory',type=Path); p.add_argument('--skill',type=Path)
    p.add_argument('--model',required=True); p.add_argument('--sd-adapter',type=Path); p.add_argument('--output',type=Path,required=True); p.add_argument('--gpu-uuid',required=True); p.add_argument('--state-id',required=True); p.add_argument('--round-index',type=int,required=True); p.add_argument('--artifact-contract',choices=(DEFAULT_ARTIFACT_CONTRACT,H201_RUNTIME_MEMORY_CONTRACT,H201_RUNTIME_SKILL_CONTRACT,H201_RUNTIME_AGENT_CONTRACT),default=DEFAULT_ARTIFACT_CONTRACT); p.add_argument('--scope-manifest',type=Path); a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES','').strip()!=a.gpu_uuid: raise SystemExit('CUDA_VISIBLE_DEVICES mismatch')
    evidence=json.loads(a.evidence.read_text()); gate=evidence['gates'][a.kind]
    if not gate['eligible']: raise ValueError('reflector invoked for NOOP gate')
    scope_manifest=load_scope_manifest(a.scope_manifest,expected_round_index=a.round_index) if a.scope_manifest else None
    scope_manifest_sha256=sha(a.scope_manifest) if a.scope_manifest else None
    prior=a.prior.read_text(); memory=a.memory.read_text() if a.memory else ''; skill=a.skill.read_text() if a.skill else ''
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda').eval(); model=base
    if a.sd_adapter:
        model=PeftModel.from_pretrained(base,str(a.sd_adapter),is_trainable=False,local_files_only=True,autocast_adapter_dtype=False).eval()
    context_window=int(getattr(model.config,'max_position_embeddings',0) or 0)
    if context_window!=40960: raise RuntimeError(f'unexpected Qwen3 reflector context window {context_window}')
    primary=0; repairs=0; total_in=0; total_out=0; started=time.time(); candidate=''; attempt_tokens=[]; attempt_input_tokens=[]; saturation_events=0; projection=None; rejected=[]
    for attempt in range(2):
        repair=attempt==1
        if repair: repairs=1
        else: primary=1
        user,projection=render(a.kind,evidence,prior,memory,skill,repair=repair,artifact_contract=a.artifact_contract)
        enc=tok.apply_chat_template(
            [{'role':'system','content':system_prompt(a.kind,a.artifact_contract)},{'role':'user','content':user}],
            add_generation_prompt=True,return_tensors='pt',return_dict=True,enable_thinking=False,
        )
        n=int(enc['input_ids'].shape[1])
        if n+MAX_NEW_TOKENS>context_window: raise RuntimeError(f'REFLECTOR_INPUT_CONTEXT_OVERFLOW kind={a.kind} input={n} max_new={MAX_NEW_TOKENS} context={context_window}')
        enc={k:v.to(model.device) for k,v in enc.items()}
        with torch.no_grad(): out=model.generate(**enc,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,pad_token_id=tok.eos_token_id)
        ids=out[0][n:]; n_out=int(ids.shape[0]); candidate=tok.decode(ids,skip_special_tokens=True).strip()
        total_in+=n; total_out+=n_out; attempt_input_tokens.append(n); attempt_tokens.append(n_out); saturation_events+=int(n_out>=MAX_NEW_TOKENS)
        if mechanically_valid(a.kind,candidate,n_out,artifact_contract=a.artifact_contract,scope_manifest=scope_manifest): break
        label='repair' if repair else 'primary'
        record_rejected_candidate(
            output=a.output, kind=a.kind, round_index=a.round_index, state_id=a.state_id,
            label=label, candidate=candidate, output_tokens=n_out,
            artifact_contract=a.artifact_contract, rejected=rejected, scope_manifest=scope_manifest,
        )
    accepted_tokens=attempt_tokens[-1] if attempt_tokens else 0
    if not mechanically_valid(a.kind,candidate,accepted_tokens,artifact_contract=a.artifact_contract,scope_manifest=scope_manifest): raise RuntimeError('reflector output failed after one allowed repair')
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(candidate.rstrip()+'\n')
    accepted_validation=candidate_validation(a.kind,candidate,a.artifact_contract,scope_manifest=scope_manifest)
    attempt_health=accepted_attempt_health(a.output,rejected,repairs,accepted_validation)
    projection_bytes=json.dumps(projection,ensure_ascii=False,sort_keys=True).encode()
    receipt={'schema':'openevo.ceiling1-stage2-reflector.v2','status':'UPDATE','kind':a.kind,'artifact_contract':a.artifact_contract,'carrier_contract_health_version':'v2','generation_attempted':True,'scope_manifest_sha256':scope_manifest_sha256,**attempt_health,'round_index':a.round_index,'state_id':a.state_id,'gpu_uuid':a.gpu_uuid,'evidence_sha256':sha(a.evidence),'projection_schema':projection['schema'],'projection_sha256':hashlib.sha256(projection_bytes).hexdigest(),'projection_bytes':len(projection_bytes),'prior_sha256':sha(a.prior),'memory_sha256':sha(a.memory) if a.memory else None,'skill_sha256':sha(a.skill) if a.skill else None,'output_sha256':sha(a.output),'sd_adapter_sha256':sha(a.sd_adapter/'adapter_model.safetensors') if a.sd_adapter else None,'primary_calls':primary,'repair_calls':repairs,'input_tokens':total_in,'attempt_input_tokens':attempt_input_tokens,'output_tokens':total_out,'accepted_output_tokens':accepted_tokens,'attempt_output_tokens':attempt_tokens,'saturation_events':saturation_events,'primary_saturated':bool(attempt_tokens and attempt_tokens[0]>=MAX_NEW_TOKENS),'repair_saturated':bool(len(attempt_tokens)>1 and attempt_tokens[1]>=MAX_NEW_TOKENS),'external_teacher_calls':0,'max_primary_calls':1,'max_repair_calls':1,'max_new_tokens_per_call':MAX_NEW_TOKENS,'context_window':context_window,'hard_context_preflight':True,'silent_truncation_allowed':False,'do_sample':False,'chat_template_enable_thinking':False,'wall_time_seconds':time.time()-started}
    atomic_json(a.output.with_suffix(a.output.suffix+'.receipt.json'),receipt); print(json.dumps(receipt,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
