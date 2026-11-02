#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path

def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def atomic_json(p: Path, x):
    p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name('.'+p.name+'.tmp')
    t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); t.replace(p)

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,required=True); ap.add_argument('--adapter',type=Path,required=True)
    ap.add_argument('--revision',required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--arm',choices=('3b','7b','qwen3-1p7b'),required=True)
    ap.add_argument('--gpu-uuid',required=True); ap.add_argument('--source-sha',required=True); a=ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES','').strip()!=a.gpu_uuid: raise SystemExit('CUDA_VISIBLE_DEVICES mismatch')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    started=time.time(); out=a.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(str(a.base),local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(str(a.base),torch_dtype=torch.bfloat16,local_files_only=True).to('cuda').eval()
    peft=PeftModel.from_pretrained(base,str(a.adapter),is_trainable=False,local_files_only=True,autocast_adapter_dtype=False).eval()
    text='Ceiling Stage2 merge-equivalence sentinel: search and verify every explicit constraint before buying.'
    enc=tok(text,return_tensors='pt').to('cuda')
    with torch.no_grad(): before=peft(**enc).logits[:,-1,:].float().cpu()
    merged=peft.merge_and_unload().eval()
    with torch.no_grad(): after=merged(**enc).logits[:,-1,:].float().cpu()
    diff=(before-after).abs(); max_abs=float(diff.max()); mean_abs=float(diff.mean())
    p_before=torch.softmax(before,dim=-1); p_after=torch.softmax(after,dim=-1)
    kl=float((p_before*(p_before.clamp_min(1e-12).log()-p_after.clamp_min(1e-12).log())).sum())
    top1_before=int(before.argmax(dim=-1).item()); top1_after=int(after.argmax(dim=-1).item())
    if not (top1_before==top1_after and kl <= 1e-3 and mean_abs <= 0.1): raise RuntimeError(f'merge equivalence failed top1={top1_before}/{top1_after} kl={kl} max={max_abs} mean={mean_abs}')
    merged.save_pretrained(str(out),safe_serialization=True,max_shard_size='5GB'); tok.save_pretrained(str(out))
    files=[]
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name!='merge-receipt.json': files.append({'name':p.name,'bytes':p.stat().st_size,'sha256':sha256(p)})
    receipt={'schema':'openevo.ceiling1-stage2-merged-base.v1','status':'PASS','arm':a.arm,'base_path':str(a.base.resolve()),
      'base_revision':a.revision,'opsd_adapter_path':str(a.adapter.resolve()),'opsd_adapter_sha256':sha256(a.adapter/'adapter_model.safetensors'),
      'gpu_uuid':a.gpu_uuid,'equivalence':{'max_abs_last_token_logits':max_abs,'mean_abs_last_token_logits':mean_abs,'softmax_kl':kl,'top1_before':top1_before,'top1_after':top1_after,'threshold_kl':1e-3,'threshold_mean_abs':0.1},
      'files':files,'source_git_sha':a.source_sha,'formal_task_consumption':0,'final_panel_consumption':0,'wall_time_seconds':time.time()-started}
    atomic_json(out/'merge-receipt.json',receipt); print(json.dumps(receipt,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
