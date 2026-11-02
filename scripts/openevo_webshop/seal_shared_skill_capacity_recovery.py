#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path

def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p:Path): return json.loads(p.read_text(encoding='utf-8'))
def atomic_json(p:Path,x):
    p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name('.'+p.name+'.tmp')
    t.write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8'); t.replace(p)
def main()->int:
    p=argparse.ArgumentParser(); p.add_argument('--repo',type=Path,required=True); p.add_argument('--root',type=Path,required=True); p.add_argument('--arm',choices=('qwen25-3b','qwen3-1p7b'),required=True); p.add_argument('--skill-root',type=Path,required=True); a=p.parse_args()
    repo=a.repo.resolve(); root=a.root.resolve(); d=a.skill_root.resolve()
    authority=repo/'configs/experiment/amendments/openevo-ceiling-1.0-stage1-skill-capacity-recovery-20260903-v1.json'
    r=load(d/'coverage-receipt.json'); mm=load(root/f'MINIMAX_COMPLETE_{a.arm}.json')
    if r.get('schema')!='openevo.ceiling1-skill-bundle-stage1.capacity-v1' or r.get('status')!='pass' or r.get('scientific_result') is not True or r.get('qualification') is not False: raise RuntimeError('Skill capacity result not scientific PASS')
    if r.get('method')!='local_qwen_hierarchical_skill_bundle_bounded_capacity_v1': raise RuntimeError('Skill capacity method mismatch')
    if r.get('split_policy')!='max 20 inputs; capacity-only deterministic frozen-order midpoint bisection; max depth 5; no reward/outcome selection': raise RuntimeError('Skill capacity split policy mismatch')
    if int(r.get('max_split_depth',-1))!=5 or r.get('outcome_based_split') is not False or r.get('adaptive_tree') is not False: raise RuntimeError('Skill capacity bounds mismatch')
    if r.get('authority_sha256')!=sha(authority): raise RuntimeError('Skill capacity authority SHA mismatch')
    if r.get('analysis_sha256')!=mm.get('analysis_corpus_sha256'): raise RuntimeError('Skill analysis lineage mismatch')
    if int(r.get('consumed_rows_total',-1))!=1440 or int(r.get('unique_source_ids',-1))!=1440: raise RuntimeError('Skill coverage mismatch')
    for k in ('omitted_rows','duplicates','unexpected_rows'):
        if r.get(k): raise RuntimeError(f'Skill contains {k}')
    skill=d/'SKILL.md'
    if not skill.is_file() or sha(skill)!=r.get('skill_sha256'): raise RuntimeError('Skill payload hash mismatch')
    out={**r,'arm':a.arm,'host_output_root':str(d),'host_skill_path':str(skill),'capacity_authority_path':str(authority),'capacity_authority_sha256':sha(authority),'sealed_at_epoch':time.time()}
    marker=root/f'SKILL_BUNDLE_COMPLETE_{a.arm}.json'; atomic_json(marker,out)
    print(json.dumps({'status':'PASS','marker':str(marker),'sha256':sha(marker),'split_events':r.get('split_events')},sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
