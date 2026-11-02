#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, importlib, json, subprocess, sys, time
from pathlib import Path
REPO_ROOT=Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from scripts.openevo_webshop import autonomous_campaign as ac

def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def git(repo:Path,*a:str)->str: return subprocess.check_output(['git','-C',str(repo),*a],text=True).strip()

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,default=Path('configs/experiment/campaigns/current-ceiling1-autonomous.json')); p.add_argument('--output',type=Path,required=True); p.add_argument('--allow-dirty',action='store_true'); a=p.parse_args()
    repo=REPO_ROOT; manifest_path=a.manifest if a.manifest.is_absolute() else repo/a.manifest
    head=git(repo,'rev-parse','HEAD'); dirty=git(repo,'status','--porcelain')
    if dirty and not a.allow_dirty: raise SystemExit('qualification refuses dirty control-plane source')
    tests=importlib.import_module('tests.test_autonomous_campaign'); executed=[]
    for name in sorted(dir(tests)):
        if name.startswith('test_'):
            getattr(tests,name)(); executed.append(name)
    manifest,resolution=ac.resolve_manifest(repo,a.manifest); ac.validate_manifest(repo,manifest); live=ac.evaluate(repo,manifest); live['manifest_resolution']=resolution
    violations=[]
    for arm,rows in live['arms'].items():
        seen_incomplete=False
        for row in rows:
            if seen_incomplete and row['state'] not in {'WAITING_DEPENDENCY','BLOCKED_SCIENCE','BLOCKED_AUTHORITY'}:
                violations.append(f"{arm}:{row['phase']}: downstream escaped dependency gate as {row['state']}")
            if row['state']!='COMPLETE': seen_incomplete=True
            if row['phase']=='final_evaluation' and row['state'] in {'READY','RUNNING','RESUME_REQUIRED','ENGINEERING_RETRYABLE'}:
                violations.append(f"{arm}: final evaluation became actionable before frozen final gate")
    status='PASS' if not violations else 'FAIL'
    receipt={'schema':'openevo.autonomous-campaign-qualification.v1','status':status,'campaign_id':manifest['campaign_id'],'control_plane_git_sha':head,'control_plane_clean':not bool(dirty),'entry_manifest_path':str(manifest_path),'entry_manifest_sha256':sha(manifest_path),'manifest_resolution':resolution,'fault_test_count':len(executed),'fault_tests':executed,'live_black_box':live,'violations':violations,'mutation':False,'budget_delta':{'new_webshop_calls':0,'external_teacher_calls':0,'gpu_training_launches':0,'final_panel_access':0},'qualified_at_epoch':time.time()}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n"); print(json.dumps({'status':status,'fault_test_count':len(executed),'violations':violations},sort_keys=True)); return 0 if status=='PASS' else 2
if __name__=='__main__': raise SystemExit(main())
