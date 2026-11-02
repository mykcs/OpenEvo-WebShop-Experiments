#!/usr/bin/env python3
"""Deterministically compile sealed Stage1 carriers into Harness-2.0 roles.

Qualification-only tool. It never touches formal Stage2 state or the final panel.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.openevo_webshop.ceiling1_harness2_compile_carriers import CAPS, abstract_ui_literals, build_prompt, validate


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name('.'+path.name+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2)+'\n')
    tmp.replace(path)


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--memory', type=Path, required=True)
    p.add_argument('--skill', type=Path, required=True)
    p.add_argument('--agent', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    p.add_argument('--gpu-uuid')
    p.add_argument('--source-activation', type=Path, required=True)
    p.add_argument('--experiment-git-sha', required=True)
    p.add_argument('--chat-template-enable-thinking', choices=('auto','false'), default='auto')
    a=p.parse_args()
    thinking=False if a.chat_template_enable_thinking=='false' else None
    if a.device=='cuda':
        if not a.gpu_uuid: raise SystemExit('--gpu-uuid required for cuda compiler')
        if os.environ.get('CUDA_VISIBLE_DEVICES','').strip()!=a.gpu_uuid: raise SystemExit('CUDA_VISIBLE_DEVICES mismatch')
    for q in (a.model,a.memory,a.skill,a.agent,a.source_activation):
        if not q.exists(): raise SystemExit(f'missing input: {q}')
    out=a.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    tok=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True,trust_remote_code=False)
    if a.device=='cuda':
        model=AutoModelForCausalLM.from_pretrained(str(a.model),torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},local_files_only=True,trust_remote_code=False)
    else:
        model=AutoModelForCausalLM.from_pretrained(str(a.model),torch_dtype=torch.bfloat16,local_files_only=True,trust_remote_code=False).to('cpu')
    model.eval()
    sources={'memory':a.memory,'skill':a.skill,'agent':a.agent}
    compiled={}
    calls=[]
    started=time.time()
    try:
        for kind in ('memory','skill','agent'):
            prompt=build_prompt(kind,sources[kind].read_text())
            messages=[{'role':'system','content':'You are a deterministic safety compiler for OpenEVO WebShop carriers. Follow the requested output schema exactly.'},{'role':'user','content':prompt}]
            template_kwargs={'add_generation_prompt':True,'return_tensors':'pt','return_dict':True}
            if thinking is not None: template_kwargs['enable_thinking']=thinking
            enc=tok.apply_chat_template(messages,**template_kwargs)
            enc={k:v.to(model.device) for k,v in enc.items()}
            n_in=int(enc['input_ids'].shape[1])
            with torch.no_grad():
                ids=model.generate(**enc,max_new_tokens=int(CAPS[kind]),do_sample=False,pad_token_id=tok.eos_token_id)
            raw_text=tok.decode(ids[0][n_in:],skip_special_tokens=True).strip()+'\n'
            raw_path=out/f'{kind}.raw.md'; raw_path.write_text(raw_text)
            text,ui_replacements=abstract_ui_literals(raw_text); text=text.strip()+'\n'
            path=out/f'{kind}.md'; path.write_text(text)
            compiled[kind]=text
            calls.append({'kind':kind,'raw_output_sha256':sha256(raw_path),'ui_literal_abstractions':ui_replacements,'source_path':str(sources[kind].resolve()),'source_sha256':sha256(sources[kind]),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'input_tokens':n_in,'max_output_tokens':int(CAPS[kind]),'output_sha256':sha256(path),'do_sample':False,'chat_template_enable_thinking':thinking})
    finally:
        del model
        if a.device=='cuda' and torch.cuda.is_available(): torch.cuda.empty_cache()
    result=validate(compiled['memory'],compiled['skill'],compiled['agent'])
    receipt={'schema':'openevo.ceiling1-harness2-carrier-compile-receipt.v1','status':result['status'],'qualification_only':True,'formal_task_consumption':0,'final_panel_access':0,'external_teacher_calls':0,'source_activation':str(a.source_activation.resolve()),'source_activation_sha256':sha256(a.source_activation),'model_path':str(a.model.resolve()),'model_config_sha256':sha256(a.model/'config.json'),'execution_device':a.device,'gpu_uuid':a.gpu_uuid if a.device=='cuda' else None,'experiment_git_sha':a.experiment_git_sha,'chat_template_enable_thinking':thinking,'calls':calls,'validation':result,'wall_time_seconds':time.time()-started}
    atomic_json(out/'compile-receipt.json',receipt)
    print(json.dumps(receipt,sort_keys=True))
    return 0 if result['status']=='PASS' else 2

if __name__=='__main__': raise SystemExit(main())
