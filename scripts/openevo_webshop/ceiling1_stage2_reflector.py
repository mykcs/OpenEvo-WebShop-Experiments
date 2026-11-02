#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
from typing import Any, Iterable
from scripts.openevo_webshop.ceiling1_harness201_carrier_contract import (
    load_scope_manifest, task_local_scope_violations,
    task_local_scope_violations_from_manifest, validate_carrier,
)

def sha(p:Path|None):
    if p is None: return None
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
def atomic_json(p:Path,x:Any): p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name('.'+p.name+'.tmp'); t.write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n'); t.replace(p)
SYSTEMS={
 'text_memory':'You maintain concise reusable long-term WebShop memory. Use only grounded evidence supplied below. Preserve still-valid prior rules, add or revise only when evidence supports it, and avoid task-specific product answers. Return only Markdown.',
 'skill_bundle':'You maintain a reusable WebShop procedural SKILL.md. Use only grounded cross-task evidence. Every new or revised skill must state Trigger, Procedure, Validation, and Evidence task identities. Do not copy task-specific product answers. Return only SKILL.md Markdown.',
 'agent_system':'You maintain global WebShop agent operating rules. Use only cross-task grounded evidence. Return Markdown whose first line is exactly # Agent System and whose required sections are ## Trigger, ## Action, and ## Validation in that order. The Agent System is a distinct carrier: never return SKILL.md, never copy the Skill Bundle, and never use a # SKILL.md heading. Do not copy task-specific product answers.'}
LIMIT={'text_memory':65536,'skill_bundle':65536,'agent_system':12000}
MAX_NEW_TOKENS=1024
DEFAULT_ARTIFACT_CONTRACT='legacy-authoring-v1'
H201_RUNTIME_MEMORY_CONTRACT='harness201-runtime-memory-v1'
H201_RUNTIME_SKILL_CONTRACT='harness201-runtime-skill-v1'
H201_RUNTIME_AGENT_CONTRACT='harness201-runtime-agent-v1'
H201_RUNTIME_MEMORY_SYSTEM='You maintain the thin Harness WebShop runtime Memory carrier from grounded cross-task evidence. Return only Markdown beginning with # Memory and exactly one level-2 section ## Reusable Knowledge containing 1 to 4 concise reusable-knowledge bullets. Preserve still-valid prior knowledge unless evidence supports a change. Do not emit UI/action-shaped vocabulary or task-specific product answers.'
H201_RUNTIME_SKILL_SYSTEM='You maintain the thin Harness 2.0.1 WebShop runtime Skill carrier from grounded cross-task evidence. Return only Markdown beginning with # Skill Bundle and exactly these level-2 sections in order: ## Search State, ## Results State, ## Product State, ## Any State. Keep it concise and runtime-safe. Preserve prior guidance unless evidence supports a change. Do not emit Trigger/Procedure/Validation/Evidence authoring sections or task-specific product answers.'
H201_RUNTIME_AGENT_SYSTEM='You maintain the thin Harness 2.0.1 WebShop runtime Agent System from grounded cross-task evidence. Return only Markdown beginning with # Agent System and exactly these level-2 sections in order: ## Authority and ## Coordination. Keep it minimal, global, and subordinate to the current environment. Preserve prior guidance unless evidence supports a change. Do not emit Trigger/Action/Validation authoring sections, SKILL.md headings, or task-specific product answers.'

def system_prompt(kind:str,artifact_contract:str)->str:
    if kind=='text_memory' and artifact_contract==H201_RUNTIME_MEMORY_CONTRACT:
        return H201_RUNTIME_MEMORY_SYSTEM
    if kind=='skill_bundle' and artifact_contract==H201_RUNTIME_SKILL_CONTRACT:
        return H201_RUNTIME_SKILL_SYSTEM
    if kind=='agent_system' and artifact_contract==H201_RUNTIME_AGENT_CONTRACT:
        return H201_RUNTIME_AGENT_SYSTEM
    return SYSTEMS[kind]

def _project_summary(s:dict)->dict:
    ex=(s.get('best_examples') or [])[:1]
    examples=[]
    for x in ex:
        examples.append({'schedule_position':x.get('schedule_position'),'score':x.get('score'),'clean_exact_success':x.get('clean_exact_success'),'sequence':x.get('sequence'),'actions':(x.get('actions') or [])[:15],'failure_signature':x.get('failure_signature'),'task_prompt_excerpt':str(x.get('first_prompt_excerpt') or '')[:400]})
    return {'task_identity':s.get('task_identity'),'best_score':s.get('best_score'),'worst_score':s.get('worst_score'),'score_values':s.get('score_values'),'clean_success_count':s.get('clean_success_count'),'failure_signatures':s.get('failure_signatures'),'examples':examples}

def _project_gate(kind:str,g:dict)->dict:
    if kind=='text_memory':
        return {'eligible':g.get('eligible'),'clean_success_count':g.get('clean_success_count'),'contrast_task_count':g.get('contrast_task_count'),'repeated_failure_patterns':g.get('repeated_failure_patterns') or []}
    return {'eligible':g.get('eligible'),'supporting_patterns':[{k:x.get(k) for k in ('pattern','kind','current_task_support','total_task_support')} for x in (g.get('supporting_patterns') or [])]}

def bounded_projection(kind:str,evidence:dict)->dict:
    rules={'all_semantic_task_summaries_retained':True,'historical_support_task_id_lists_omitted':True,'best_examples_per_task':1,'task_prompt_excerpt_chars':400,'outcome_adaptive_projection':False}
    if evidence.get('summary_scope') is not None: rules['summary_scope']=evidence['summary_scope']
    return {'schema':'openevo.ceiling1-stage2-reflector-evidence-projection.v1','round_index':evidence['round_index'],'source_attempt_count':evidence.get('attempt_count'),'source_task_identity_count':evidence.get('task_identity_count'),'gate':_project_gate(kind,evidence['gates'][kind]),'task_summaries':[_project_summary(x) for x in evidence.get('summaries') or []],'projection_rules':rules}

def render(kind:str,evidence:dict,prior:str,memory:str,skill:str,repair:bool=False,artifact_contract:str=DEFAULT_ARTIFACT_CONTRACT)->tuple[str,dict]:
    proj=bounded_projection(kind,evidence)
    parts=[f'# Stage2 {kind} reflection',f'Round: {evidence["round_index"]}','The gate has already been deterministically satisfied. Revise only from the bounded grounded projection below.','', '## Prior artifact','',prior.strip() or '(none)','', '## Current round bounded grounded evidence','',json.dumps(proj,ensure_ascii=False,sort_keys=True)]
    if kind=='agent_system':
        parts += ['','## Prior State_k reference carriers (context only; never copy as output)','', '### Text Memory',memory.strip(),'','### Skill Bundle',skill.strip()]
        if artifact_contract==H201_RUNTIME_AGENT_CONTRACT:
            parts += ['','## Harness 2.0.1 runtime Agent output boundary','Begin exactly with `# Agent System`. Then emit exactly these two level-2 sections in order: `## Authority` and `## Coordination`. Keep only minimal global guidance subordinate to the current environment. Preserve prior guidance unless bounded evidence supports a change. Do not emit Trigger/Action/Validation authoring sections, tables, preambles, SKILL.md headings, or a copy/restatement of the Skill Bundle.']
        else:
            parts += ['','## Agent System output boundary','Begin exactly with `# Agent System`. Then emit exactly these three top-level sections in order: `## Trigger`, `## Action`, and `## Validation`. Put all grounded rules inside those sections. Do not emit tables, preambles, SKILL.md headings, or a copy/restatement of the Skill Bundle.']
        parts += ['','## Prior Agent System (authoritative output base, reiterated after reference carriers)',prior.strip() or '(none)']
    if kind=='text_memory' and artifact_contract==H201_RUNTIME_MEMORY_CONTRACT: parts += ['','## Harness runtime Memory output boundary','Begin exactly with `# Memory`, then `## Reusable Knowledge`. Under that section emit only 1 to 4 concise `- ` bullets. Preserve grounded reusable content; do not emit any other headings, prose, UI/action-shaped vocabulary, or task-specific answers.']
    if kind=='skill_bundle' and artifact_contract==H201_RUNTIME_SKILL_CONTRACT: parts += ['','## Harness 2.0.1 runtime Skill output boundary','This is the thin runtime carrier, not the authoring SKILL.md schema. Begin exactly with `# Skill Bundle`. Then emit exactly these four level-2 sections in order: `## Search State`, `## Results State`, `## Product State`, and `## Any State`. Keep each section concise; `- No additional guidance.` is valid. Do not emit Trigger/Procedure/Validation/Evidence headings. Preserve prior guidance unless the bounded evidence supports a change.']
    if repair: parts += ['','## Fixed one-time repair','The primary candidate hit the output hard cap or failed the mechanical artifact contract. Using the exact same prior artifact and exact same bounded evidence projection, rewrite once more compactly. Merge duplicate rules, omit commentary, satisfy the required structure, and stop well before the same 1024-token hard cap.']
    else: parts += ['','Keep the result concise and finish well before the 1024-token hard cap. Do not invent unsupported facts.']
    return '\n'.join(parts),proj

def _typed_valid(kind:str,text:str,artifact_contract:str)->bool:
    if not text.strip() or len(text.encode())>LIMIT[kind]: return False
    low=text.lower()
    if kind=='text_memory' and artifact_contract==H201_RUNTIME_MEMORY_CONTRACT:
        return validate_carrier('text_memory',text)['status']=='PASS'
    if kind=='skill_bundle' and artifact_contract==H201_RUNTIME_SKILL_CONTRACT:
        return validate_carrier('skill_bundle',text)['status']=='PASS'
    if kind=='agent_system' and artifact_contract==H201_RUNTIME_AGENT_CONTRACT:
        return validate_carrier('agent_system',text)['status']=='PASS'
    if kind=='skill_bundle':
        return all(k in low for k in ('trigger','procedure','validation','evidence'))
    if kind=='agent_system':
        lines=[line.strip().lower() for line in text.splitlines() if line.strip()]
        if not lines or lines[0]!='# agent system' or any(line=='# skill.md' for line in lines): return False
        required=['## trigger','## action','## validation']; positions=[]
        for heading in required:
            if heading not in lines: return False
            positions.append(lines.index(heading))
        return positions==sorted(positions) and len(set(positions))==len(required)
    return True


def candidate_validation(
    kind:str, text:str, artifact_contract:str=DEFAULT_ARTIFACT_CONTRACT,
    forbidden_literals:Iterable[str]|None=None, scope_manifest:dict|None=None,
)->dict:
    if forbidden_literals is not None and scope_manifest is not None:
        raise ValueError("supply forbidden_literals or scope_manifest, not both")
    typed_ok=_typed_valid(kind,text,artifact_contract)
    if scope_manifest is not None:
        scope=task_local_scope_violations_from_manifest(text,scope_manifest)
        scope_status='PASS' if not scope else 'FAIL'
    elif forbidden_literals is not None:
        scope=task_local_scope_violations(text,forbidden_literals)
        scope_status='PASS' if not scope else 'FAIL'
    else:
        scope=[]; scope_status='NOT_EVALUATED'
    return {'artifact_contract_valid':bool(typed_ok and not scope),'typed_contract_valid':typed_ok,'scope_status':scope_status,'scope_violations':scope}


def valid(kind:str,text:str,artifact_contract:str=DEFAULT_ARTIFACT_CONTRACT,forbidden_literals:Iterable[str]|None=None,scope_manifest:dict|None=None)->bool:
    return bool(candidate_validation(kind,text,artifact_contract,forbidden_literals,scope_manifest)['artifact_contract_valid'])


def mechanically_valid(kind:str,text:str,output_tokens:int,cap:int=MAX_NEW_TOKENS,artifact_contract:str=DEFAULT_ARTIFACT_CONTRACT,forbidden_literals:Iterable[str]|None=None,scope_manifest:dict|None=None)->bool:
    return output_tokens < cap and valid(kind,text,artifact_contract,forbidden_literals,scope_manifest)


def record_rejected_candidate(
    *, output: Path, kind: str, round_index: int, state_id: str, label: str,
    candidate: str, output_tokens: int, artifact_contract: str, rejected: list[dict],
    scope_manifest: dict | None = None,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    rejected_path = output.parent / f"rejected-{label}.md"
    rejected_path.write_text(candidate.rstrip() + "\n")
    validation=candidate_validation(kind,candidate,artifact_contract,scope_manifest=scope_manifest)
    item = {
        "attempt": label, "output_tokens": int(output_tokens),
        "saturated": int(output_tokens) >= MAX_NEW_TOKENS,
        "artifact_contract_valid": validation["artifact_contract_valid"],
        "typed_contract_valid": validation["typed_contract_valid"],
        "scope_status": validation["scope_status"],
        "scope_violations": validation["scope_violations"],
        "candidate_sha256": sha(rejected_path),
    }
    if label == "repair":
        if len(rejected) != 1 or rejected[0].get("attempt") != "primary":
            raise RuntimeError("repair evidence missing unique primary candidate")
        item["repair_changed_bytes"] = item["candidate_sha256"] != rejected[0]["candidate_sha256"]
    rejected.append(item)
    primary = rejected[0] if rejected else None; repair = rejected[1] if len(rejected) > 1 else None
    atomic_json(output.parent / "rejected-candidates.json", {
        "schema": "openevo.ceiling1-stage2-reflector-rejected-candidates.v1",
        "carrier_contract_health_version": "v2", "kind": kind,
        "round_index": round_index, "state_id": state_id,
        "primary_payload_sha256": primary.get("candidate_sha256") if primary else None,
        "primary_contract_valid": primary.get("artifact_contract_valid") if primary else None,
        "primary_scope_status": primary.get("scope_status") if primary else None,
        "repair_requested": repair is not None,
        "repair_payload_sha256": repair.get("candidate_sha256") if repair else None,
        "repair_changed_bytes": repair.get("repair_changed_bytes") if repair else None,
        "repair_contract_valid": repair.get("artifact_contract_valid") if repair else None,
        "repair_scope_status": repair.get("scope_status") if repair else None,
        "attempts": rejected,
    })
    return item


def accepted_attempt_health(output: Path, rejected: list[dict], repair_calls: int, accepted_validation:dict) -> dict:
    accepted_sha = sha(output)
    if int(repair_calls):
        if len(rejected) != 1 or rejected[0].get("attempt") != "primary":
            raise RuntimeError("accepted repair evidence missing unique rejected primary")
        primary = rejected[0]
        return {
            "primary_payload_sha256": primary["candidate_sha256"],
            "primary_contract_valid": bool(primary.get("artifact_contract_valid")),
            "primary_scope_status": primary.get("scope_status"),
            "repair_requested": True, "repair_payload_sha256": accepted_sha,
            "repair_changed_bytes": accepted_sha != primary["candidate_sha256"],
            "repair_contract_valid": bool(accepted_validation["artifact_contract_valid"]),
            "repair_scope_status": accepted_validation["scope_status"],
            "accepted_payload_sha256": accepted_sha,
        }
    return {
        "primary_payload_sha256": accepted_sha,
        "primary_contract_valid": bool(accepted_validation["artifact_contract_valid"]),
        "primary_scope_status": accepted_validation["scope_status"],
        "repair_requested": False, "repair_payload_sha256": None,
        "repair_changed_bytes": None, "repair_contract_valid": None,
        "repair_scope_status": None, "accepted_payload_sha256": accepted_sha,
    }


def main()->int:
    p=argparse.ArgumentParser(); p.add_argument('--kind',choices=tuple(SYSTEMS),required=True); p.add_argument('--evidence',type=Path,required=True); p.add_argument('--prior',type=Path,required=True); p.add_argument('--memory',type=Path); p.add_argument('--skill',type=Path)
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
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True); base=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda').eval(); model=base
    if a.sd_adapter:
        model=PeftModel.from_pretrained(base,str(a.sd_adapter),is_trainable=False,local_files_only=True,autocast_adapter_dtype=False).eval()
    primary=0; repairs=0; total_in=0; total_out=0; started=time.time(); candidate=''
    attempt_tokens=[]; attempt_input_tokens=[]; saturation_events=0; projection=None; rejected=[]
    context_window=int(getattr(model.config,'max_position_embeddings',0) or 0)
    if context_window!=32768: raise RuntimeError(f'unexpected reflector context window {context_window}')
    for attempt in range(2):
        repair=attempt==1
        if repair: repairs=1
        else: primary=1
        user,projection=render(a.kind,evidence,prior,memory,skill,repair=repair,artifact_contract=a.artifact_contract)
        enc=tok.apply_chat_template([{'role':'system','content':system_prompt(a.kind,a.artifact_contract)},{'role':'user','content':user}],add_generation_prompt=True,return_tensors='pt',return_dict=True)
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
    if not mechanically_valid(a.kind,candidate,accepted_tokens,artifact_contract=a.artifact_contract,scope_manifest=scope_manifest):
        raise RuntimeError('reflector output failed after one allowed repair')
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(candidate.rstrip()+'\n')
    accepted_validation=candidate_validation(a.kind,candidate,a.artifact_contract,scope_manifest=scope_manifest)
    attempt_health=accepted_attempt_health(a.output,rejected,repairs,accepted_validation)
    projection_bytes=json.dumps(projection,ensure_ascii=False,sort_keys=True).encode()
    receipt={'schema':'openevo.ceiling1-stage2-reflector.v2','status':'UPDATE','kind':a.kind,'artifact_contract':a.artifact_contract,'carrier_contract_health_version':'v2','generation_attempted':True,'scope_manifest_sha256':scope_manifest_sha256,**attempt_health,'round_index':a.round_index,'state_id':a.state_id,'gpu_uuid':a.gpu_uuid,'evidence_sha256':sha(a.evidence),'projection_schema':projection['schema'],'projection_sha256':hashlib.sha256(projection_bytes).hexdigest(),'projection_bytes':len(projection_bytes),'prior_sha256':sha(a.prior),'memory_sha256':sha(a.memory) if a.memory else None,'skill_sha256':sha(a.skill) if a.skill else None,'output_sha256':sha(a.output),'sd_adapter_sha256':sha(a.sd_adapter/'adapter_model.safetensors') if a.sd_adapter else None,'primary_calls':primary,'repair_calls':repairs,'input_tokens':total_in,'attempt_input_tokens':attempt_input_tokens,'output_tokens':total_out,'accepted_output_tokens':accepted_tokens,'attempt_output_tokens':attempt_tokens,'saturation_events':saturation_events,'primary_saturated':bool(attempt_tokens and attempt_tokens[0]>=MAX_NEW_TOKENS),'repair_saturated':bool(len(attempt_tokens)>1 and attempt_tokens[1]>=MAX_NEW_TOKENS),'external_teacher_calls':0,'max_primary_calls':1,'max_repair_calls':1,'max_new_tokens_per_call':MAX_NEW_TOKENS,'context_window':context_window,'hard_context_preflight':True,'silent_truncation_allowed':False,'do_sample':False,'wall_time_seconds':time.time()-started}
    atomic_json(a.output.with_suffix(a.output.suffix+'.receipt.json'),receipt); print(json.dumps(receipt,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
