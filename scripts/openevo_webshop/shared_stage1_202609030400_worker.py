#!/usr/bin/env python3
"""Formal shared Stage1 worker for freeze 202609030400.

Generation semantics are the already-qualified shared min-deliberation runtime.
This wrapper only updates formal provenance from the parent 202609021800 source contract.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from scripts.openevo_webshop import q25_stage1_seed_native_worker as seed_worker
from scripts.openevo_webshop.stage1_202609021800 import worker as w
from scripts.openevo_webshop.shared_stage1_min_deliberation_runtime import (
    PROFILE, SharedMinimumDeliberationRuntime,
)
from scripts.openevo_webshop.shared_stage1_seed_native_soft_deliberation_prompt import (
    PROMPT_CONTRACT_SHA256, PROMPT_PROFILE, render_shared_seed_native_soft_prompt,
)

FREEZE_ID='202609030400'
PARENT_FREEZE_ID='202609021800'
ROOT=Path(os.environ.get('SHARED_STAGE1_REPO_ROOT', str(Path.cwd())))
DESIGN_PATH=Path(os.environ.get('SHARED_STAGE1_DESIGN_PATH', str(ROOT/'configs/experiment/stage1/202609030400/design.json')))
FREEZE_PATH=Path(os.environ.get('SHARED_STAGE1_FREEZE_PATH', str(ROOT/'configs/experiment/stage1/202609030400/freeze.json')))

def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def postprocess(output:Path)->None:
    design_sha=sha(DESIGN_PATH)
    freeze_sha=sha(FREEZE_PATH)
    episode_files=sorted((output/'episodes').glob('*.json'))
    rows=[]
    for path in episode_files:
        row=json.loads(path.read_text(encoding='utf-8'))
        row['schema']='openevo.stage1.rollout.202609030400'
        row['freeze_id']=FREEZE_ID
        row['shared_harness_freeze_id']=FREEZE_ID
        row['shared_harness_design_sha256']=design_sha
        row['shared_harness_freeze_manifest_sha256']=freeze_sha
        row['upstream_source_freeze_id']=PARENT_FREEZE_ID
        row['upstream_source_design_sha256']=row.get('source_design_sha256')
        path.write_text(json.dumps(row,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
        rows.append(row)
    rows.sort(key=lambda r:int(r['schedule_position']))
    (output/'attempts.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows),encoding='utf-8')
    summary_path=output/'summary.json'
    summary=json.loads(summary_path.read_text(encoding='utf-8'))
    summary['schema']='openevo.stage1.shard-summary.202609030400'
    summary['freeze_id']=FREEZE_ID
    summary['shared_harness_freeze_id']=FREEZE_ID
    summary['shared_harness_design_sha256']=design_sha
    summary['shared_harness_freeze_manifest_sha256']=freeze_sha
    summary['upstream_source_freeze_id']=PARENT_FREEZE_ID
    summary['upstream_source_design_sha256']=summary.get('source_design_sha256')
    summary_path.write_text(json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
def configure_shared_runtime()->None:
    design=json.loads(DESIGN_PATH.read_text(encoding='utf-8'))
    manifest=json.loads(FREEZE_PATH.read_text(encoding='utf-8'))
    if design.get('freeze_id')!=FREEZE_ID or design.get('status')!='FROZEN':
        raise RuntimeError('shared Stage1 design identity mismatch')
    if manifest.get('freeze_id')!=FREEZE_ID or manifest.get('status')!='FROZEN':
        raise RuntimeError('shared Stage1 freeze manifest identity mismatch')
    seed_worker.render_seed_native_prompt=render_shared_seed_native_soft_prompt
    seed_worker.PROMPT_PROFILE=PROMPT_PROFILE
    seed_worker.PROMPT_CONTRACT_SHA256=PROMPT_CONTRACT_SHA256
    w.FREEZE_ID=FREEZE_ID
    w.run_episode=seed_worker.corrected_run_episode
    w.TEMPERATURE=1.0
    w.TOP_P=1.0
    w.TOP_K=0
    w.REPETITION_PENALTY=1.0
    w.PROFILE=PROFILE
    w.PROMPT_PROFILE=PROMPT_PROFILE
    w.PROMPT_CONTRACT_SHA256=PROMPT_CONTRACT_SHA256
    w.Harness2011Stage1InterfaceRuntime=SharedMinimumDeliberationRuntime

if __name__=='__main__':
    configure_shared_runtime()
    original=list(sys.argv)
    rc=w.main()
    out=Path(original[original.index('--output')+1])
    postprocess(out)
    raise SystemExit(rc)
