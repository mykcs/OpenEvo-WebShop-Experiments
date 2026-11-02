#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, json, tempfile
from pathlib import Path

REPO=Path('/workspace/repo')
S=REPO/'scripts/openevo_webshop'

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def main():
    ana=load_module('shared_ana',S/'h146_analyze_trajectories.py')
    mem=load_module('shared_mem',S/'ceiling1_memory_contracts.py')
    build=load_module('shared_build',S/'build_ceiling1_opsd_dataset.py')
    train=load_module('shared_train',S/'train_ceiling1_opsd.py')
    row={'x':'before\u2028after','y':'before\u2029after'}
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'u.jsonl'; p.write_text(json.dumps(row,ensure_ascii=False)+'\n',encoding='utf-8')
        for label,reader in [('analyzer',ana.read_jsonl),('memory',mem.read_jsonl),('opsd-build',build.read_jsonl),('opsd-train',train.read_jsonl)]:
            got=reader(p); assert got==[row],(label,got)
    ids=['ceiling1-3b-task000','ceiling1-qwen3-1p7b-task179','shared030400-qwen25-3b-task042','shared030400-qwen3-1p7b-task178']
    assert [mem.baseline_ordinal(x) for x in ids]==[0,179,42,178]
    print('SYNTHETIC_JSONL_AND_ARM_IDS_PASS')

if __name__=='__main__': main()
