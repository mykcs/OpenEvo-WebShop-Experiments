#!/usr/bin/env python3
"""Q06-specific zero-task GPU smoke using the canonical prelaunch owner."""
from __future__ import annotations
import argparse,json,os,pathlib,subprocess,uuid
from scripts import oe_impl as oe
from scripts.openevo_webshop import prelaunch_zero_formal as pre
from scripts.openevo_webshop.q06_context_probe_inputs import validate_inputs,SOURCE_SHA
from scripts.openevo_webshop.q06_context_host_orchestrator import MANIFEST_SHA,sha,require
SCHEMA='openevo.q06.zero-formal-qualification.202610011303'
POOL={'GPU-00000000-0000-0000-0000-000000000000','GPU-00000000-0000-0000-0000-000000000000','GPU-00000000-0000-0000-0000-000000000000','GPU-00000000-0000-0000-0000-000000000000','GPU-00000000-0000-0000-0000-000000000000','GPU-00000000-0000-0000-0000-000000000000','GPU-00000000-0000-0000-0000-000000000000','GPU-00000000-0000-0000-0000-000000000000'}

def main():
 p=argparse.ArgumentParser(__doc__)
 p.add_argument('--authority-repo',type=pathlib.Path,required=True)
 p.add_argument('--execution-source',type=pathlib.Path,required=True)
 p.add_argument('--profile',type=pathlib.Path,required=True)
 p.add_argument('--namespace-receipt',type=pathlib.Path,required=True)
 p.add_argument('--namespace-sha256',required=True)
 p.add_argument('--gpu-uuid',choices=sorted(POOL),required=True)
 p.add_argument('--control-root',type=pathlib.Path,required=True)
 a=p.parse_args();repo=a.authority_repo.resolve();source=a.execution_source.resolve();gpu=a.gpu_uuid
 require(os.geteuid()==0,'Q06 GPU qualification requires root control plane')
 require(os.getenv('OPENEVO_AUTHORIZED_GPU_UUIDS')==gpu,'Q06 exact authorized UUID env required')
 require(not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip(),'Q06 authority must be clean')
 validate_inputs(repo,source,MANIFEST_SHA)
 require(sha(a.namespace_receipt)==a.namespace_sha256,'Q06 namespace receipt drift')
 ns=json.loads(a.namespace_receipt.read_text())
 require(ns.get('status')=='PASS' and ns.get('gpu_used') is False and ns.get('model_calls')==ns.get('environment_steps')==0 and ns.get('manifest_sha256')==MANIFEST_SHA,'Q06 CPU namespace not PASS')
 profile=json.loads(a.profile.read_text());profile_sha=sha(a.profile)
 require(profile['source']['experiment']['revision']==SOURCE_SHA and profile['identity']=={'uid':1001,'gid':1001},'Q06 profile source/UID drift')
 root=a.control_root.absolute();require(not root.exists() and root.parent.is_dir(),'Q06 control root must be fresh')
 root.mkdir(mode=0o750);os.chown(root,1001,1001)
 receipt={'schema':SCHEMA,'status':'FAIL','campaign_id':profile['campaign_id'],'execution_sha':SOURCE_SHA,'authority_sha':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),'selected_gpu_uuids':[gpu],'candidate_hashes':{'runtime_profile':profile_sha},'manifest_sha256':MANIFEST_SHA,'namespace_receipt':{'path':str(a.namespace_receipt),'sha256':a.namespace_sha256},'formal_task_consumed':False,'scientific_state_mutated':False,'formal_task_budget':0,'task_bearing_calls':0,'model_calls':0,'training_calls':0,'final_panel_access':0,'scientific_authority_granted_by_receipt':False}
 try:
  with oe.gpu_locks([gpu],oe.workspace_path(profile,profile['lock_root'])):
   oe.wait_for_idle({**profile,'wait':False},[gpu],root)
   smoke_root=oe.workspace_path(profile,profile['paths']['runs'])/('.q06-zero-formal-'+uuid.uuid4().hex)
   smoke_root.mkdir(mode=0o700);os.chown(smoke_root,1001,1001)
   smoke={**profile,'run_id':smoke_root.name}
   result=oe.run_selected_gpu_preflight(smoke,[gpu],smoke_root,profile['image_id'],root)
   receipt.update(status='ZERO_FORMAL_QUALIFIED',prelaunch_receipt={'path':result['path'],'sha256':result['sha256']})
 except Exception as exc:
  receipt['failure_reason']=type(exc).__name__+': '+str(exc)
 finally:
  receipt['receipt_sha256']=pre.canonical_sha256(receipt)
  with (root/'qualification.json').open('x') as f:
   json.dump(receipt,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':receipt['status'],'receipt':str(root/'qualification.json'),'new_task_calls':0}))
 return 0 if receipt['status']=='ZERO_FORMAL_QUALIFIED' else 1
if __name__=='__main__':raise SystemExit(main())
