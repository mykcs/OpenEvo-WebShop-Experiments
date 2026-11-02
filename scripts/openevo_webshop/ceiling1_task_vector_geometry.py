#!/usr/bin/env python3
"""Exact low-rank Task Vector geometry for cumulative native SD-LoRA states."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
from typing import Any

def atomic_json(p:Path,x:Any): p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name('.'+p.name+'.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); t.replace(p)
def sha(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def coeffs(man): return [float(x['coefficient']) for x in man['components']]
def qform(x,G,y): return sum(x[i]*G[i][j]*y[j] for i in range(len(x)) for j in range(len(y)))
def safe_cos(dot,a2,b2): return None if a2<=0 or b2<=0 else max(-1.0,min(1.0,dot/math.sqrt(a2*b2)))
def component_keys(tensors,idx):
    prefix=f'components.{idx}.'; names=sorted({k[len(prefix):-2] for k in tensors if k.startswith(prefix) and k.endswith('.A')}); return names
def comp_digest(tensors,idx,names):
    h=hashlib.sha256()
    for n in names:
        for m in ('A','B'):
            k=f'components.{idx}.{n}.{m}'; v=tensors[k].contiguous().cpu(); h.update(k.encode()); h.update(str(tuple(v.shape)).encode()); h.update(v.numpy().tobytes())
    return h.hexdigest()
def inner(tensors,i,j,names):
    import torch
    total=torch.zeros((),dtype=torch.float64)
    for n in names:
        Ai=tensors[f'components.{i}.{n}.A'].double(); Bi=tensors[f'components.{i}.{n}.B'].double(); Aj=tensors[f'components.{j}.{n}.A'].double(); Bj=tensors[f'components.{j}.{n}.B'].double()
        total += torch.sum((Bi.T @ Bj) * (Aj @ Ai.T).T)
    return float(total.item())
def main()->int:
    p=argparse.ArgumentParser(); p.add_argument('--current-adapter',type=Path,required=True); p.add_argument('--prior-adapter',type=Path); p.add_argument('--prior-geometry',type=Path); p.add_argument('--output',type=Path,required=True); p.add_argument('--round-index',type=int,required=True); p.add_argument('--generation-index',type=int,required=True); p.add_argument('--arm',choices=('3b','7b','qwen3-1p7b'),required=True); a=p.parse_args()
    from safetensors.torch import load_file
    cur=a.current_adapter.resolve(); man=json.loads((cur/'openevo_sd_lora_state.json').read_text()); N=int(man['component_count']); c=coeffs(man)
    if N!=len(c) or N<1: raise ValueError('current SD-LoRA component accounting invalid')
    tensors=load_file(str(cur/'openevo_sd_lora_state.safetensors'),device='cpu'); names=component_keys(tensors,0)
    if not names or any(component_keys(tensors,i)!=names for i in range(N)): raise ValueError('component module inventory drift')
    digests=[comp_digest(tensors,i,names) for i in range(N)]
    if a.prior_adapter is None:
        if a.prior_geometry is not None or N!=1: raise ValueError('first geometry must be component_count=1 with no prior')
        G=[[inner(tensors,0,0,names)]]; pcoeff=[0.0]; prevdelta=None; prior_path=0.0; prior_sha=None
    else:
        if a.prior_geometry is None or not a.prior_geometry.is_file(): raise ValueError('prior adapter requires prior geometry')
        prior=json.loads((a.prior_adapter/'openevo_sd_lora_state.json').read_text()); pc=coeffs(prior); pg=json.loads(a.prior_geometry.read_text())
        if N!=len(pc)+1 or pg['component_count']!=len(pc): raise ValueError('native SD-LoRA must append exactly one component')
        if pg['component_factor_sha256']!=digests[:-1]: raise ValueError('prior component direction identity drift')
        oldG=pg['gram_matrix'];
        if len(oldG)!=N-1 or any(len(row)!=N-1 for row in oldG): raise ValueError('prior Gram shape mismatch')
        newrow=[inner(tensors,i,N-1,names) for i in range(N)]; G=[list(row)+[newrow[i]] for i,row in enumerate(oldG)]; G.append(newrow)
        pcoeff=pc+[0.0]; prevdelta=list(pg['delta_coefficients'])+[0.0]; prior_path=float(pg['path_length']); prior_sha=sha(a.prior_geometry)
    diag=[G[i][i] for i in range(N)]
    if any(not math.isfinite(x) or abs(x-1.0)>5e-3 for x in diag): raise RuntimeError(f'normalized component Gram diagonal drift: {diag[-3:]}')
    d=[c[i]-pcoeff[i] for i in range(N)]; d2=max(0.0,qform(d,G,d)); cn2=max(0.0,qform(c,G,c)); delta_norm=math.sqrt(d2); cumulative_norm=math.sqrt(cn2)
    adj=None
    if prevdelta is not None:
        pd2=max(0.0,qform(prevdelta,G,prevdelta)); adj=safe_cos(qform(d,G,prevdelta),d2,pd2)
    alignment=safe_cos(qform(d,G,c),d2,cn2); path=prior_path+delta_norm; effrank=int(man['effective_rank']); max_ceiling_rank=64*8
    c2=sum(x*x for x in c); c4=sum((x*x)*(x*x) for x in c); eff_components=(c2*c2/c4) if c4>0 else 0.0
    out={'schema':'openevo.ceiling1-task-vector-geometry.v1','status':'PASS','arm':a.arm,'round_index':a.round_index,'generation_index':a.generation_index,'definition':'tau_r = cumulative native SD-LoRA Delta-W after current update minus cumulative Delta-W before current update; exact Frobenius geometry computed from normalized low-rank component factors and learned coefficients','component_count':N,'component_rank':8,'component_factor_sha256':digests,'gram_matrix':G,'current_coefficients':c,'prior_coefficients_padded':pcoeff,'delta_coefficients':d,'delta_norm':delta_norm,'path_length':path,'adjacent_update_cosine':adj,'cumulative_alignment':alignment,'cumulative_delta_norm':cumulative_norm,'effective_rank':effrank,'effective_rank_participation':effrank/max_ceiling_rank,'coefficient_effective_component_count':eff_components,'current_adapter_state_sha256':sha(cur/'openevo_sd_lora_state.safetensors'),'current_adapter_manifest_sha256':sha(cur/'openevo_sd_lora_state.json'),'prior_geometry_sha256':prior_sha,'online_use':'diagnostic-only','modify_training':False,'modify_next_round':False}
    for k in ('delta_norm','path_length','cumulative_delta_norm','effective_rank_participation','coefficient_effective_component_count'):
        if not math.isfinite(float(out[k])): raise RuntimeError(f'non-finite geometry {k}')
    atomic_json(a.output,out); print(json.dumps({k:out[k] for k in ['component_count','delta_norm','path_length','adjacent_update_cosine','cumulative_alignment','effective_rank_participation']},sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
