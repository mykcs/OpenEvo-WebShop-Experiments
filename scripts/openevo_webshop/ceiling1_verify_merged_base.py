#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, os, time
from pathlib import Path
from typing import Any

def sha(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def atomic_json(p:Path,x:Any): p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name('.'+p.name+'.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); t.replace(p)
def main()->int:
    p=argparse.ArgumentParser(); p.add_argument('--base',type=Path,required=True); p.add_argument('--adapter',type=Path,required=True); p.add_argument('--merged',type=Path,required=True); p.add_argument('--revision',required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--arm',choices=('3b','7b','qwen3-1p7b'),required=True); p.add_argument('--gpu-uuid',required=True); p.add_argument('--experiment-git-sha',required=True); p.add_argument('--stage2-config-sha256',required=True); a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES','').strip()!=a.gpu_uuid: raise SystemExit('CUDA_VISIBLE_DEVICES mismatch')
    import torch, gc
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    text='Ceiling Stage2 merge-equivalence sentinel: search and verify every explicit constraint before buying.'; started=time.time(); tok=AutoTokenizer.from_pretrained(str(a.base),local_files_only=True); enc=tok(text,return_tensors='pt').to('cuda')
    base=AutoModelForCausalLM.from_pretrained(str(a.base),torch_dtype=torch.bfloat16,local_files_only=True).to('cuda').eval(); peft=PeftModel.from_pretrained(base,str(a.adapter),is_trainable=False,local_files_only=True,autocast_adapter_dtype=False).eval()
    with torch.no_grad(): before=peft(**enc).logits[:,-1,:].float().cpu()
    del peft,base,enc; gc.collect(); torch.cuda.empty_cache()
    merged=AutoModelForCausalLM.from_pretrained(str(a.merged),torch_dtype=torch.bfloat16,local_files_only=True).to('cuda').eval(); enc2=tok(text,return_tensors='pt').to('cuda')
    with torch.no_grad(): after=merged(**enc2).logits[:,-1,:].float().cpu()
    diff=(before-after).abs(); max_abs=float(diff.max()); mean_abs=float(diff.mean()); pb=torch.softmax(before,dim=-1); pa=torch.softmax(after,dim=-1); kl=float((pb*(pb.clamp_min(1e-12).log()-pa.clamp_min(1e-12).log())).sum()); top1b=int(before.argmax()); top1a=int(after.argmax())
    if not(top1b==top1a and kl<=1e-3 and mean_abs<=0.1): raise RuntimeError(f'merged-base equivalence failed top1={top1b}/{top1a} kl={kl} mean={mean_abs}')
    files=[]
    for fp in sorted(a.merged.iterdir()):
        if fp.is_file() and fp.name not in ('merge-receipt.json','verification-receipt.json'): files.append({'name':fp.name,'bytes':fp.stat().st_size,'sha256':sha(fp)})
    original=a.merged/'merge-receipt.json'; out={'schema':'openevo.ceiling1-stage2-merged-base-verification.v1','status':'PASS','arm':a.arm,'base_revision':a.revision,'base_path':str(a.base.resolve()),'opsd_adapter_path':str(a.adapter.resolve()),'opsd_adapter_sha256':sha(a.adapter/'adapter_model.safetensors'),'merged_path':str(a.merged.resolve()),'merged_files':files,'original_merge_receipt_sha256':sha(original) if original.is_file() else None,'equivalence':{'top1_before':top1b,'top1_after':top1a,'softmax_kl':kl,'mean_abs_last_token_logits':mean_abs,'max_abs_last_token_logits':max_abs,'threshold_kl':1e-3,'threshold_mean_abs':0.1},'gpu_uuid':a.gpu_uuid,'experiment_git_sha':a.experiment_git_sha,'stage2_config_sha256':a.stage2_config_sha256,'formal_task_consumption':0,'final_panel_consumption':0,'wall_time_seconds':time.time()-started}
    atomic_json(a.output,out); print(json.dumps(out,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
