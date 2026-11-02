#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any

def canonical_sha(v:Any)->str:
    return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read_jsonl(p:Path):
    if not p.exists(): return []
    with p.open(encoding='utf-8') as f: return [json.loads(x) for x in f if x.strip()]
def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--arm',required=True); ap.add_argument('--canonical-source',type=Path,required=True)
    ap.add_argument('--provisional-analysis',type=Path,action='append',default=[]); ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); src=read_jsonl(a.canonical_source); source={str(r['baseline_key']):r for r in src}
    if len(source)!=1440: raise SystemExit('canonical source must contain 1440 unique trajectories')
    promoted={}; rejected=[]
    for p in a.provisional_analysis:
        for r in read_jsonl(p):
            tid=str(r.get('trajectory_id',''))
            if r.get('parse_ok') is not True: rejected.append((tid,'parse_not_ok')); continue
            s=source.get(tid)
            if s is None: rejected.append((tid,'not_in_canonical')); continue
            if r.get('trajectory_sha256')!=canonical_sha(s): rejected.append((tid,'sha_mismatch')); continue
            promoted[tid]=r
    ordered=[promoted[str(s['baseline_key'])] for s in src if str(s['baseline_key']) in promoted]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in ordered),encoding='utf-8')
    receipt={'schema':'openevo.shared-stage1.provisional-minimax-promotion.v1','status':'PASS','arm':a.arm,
             'canonical_count':1440,'promoted_parse_ok':len(ordered),'rejected_count':len(rejected),
             'rejected_sample':rejected[:20],'output':str(a.output)}
    rp=a.output.with_suffix('.promotion.json'); rp.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps(receipt,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
