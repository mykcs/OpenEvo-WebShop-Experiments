#!/usr/bin/env python3
"""Ray virtual-GPU pool for OpenEVO Stage2.

Each physical GPU4-7 is represented as one Ray node with numerical resource
budgets. Ray schedules generic workload intents; formal CUDA remains inside the
existing Docker launchers, so scientific runtime semantics are unchanged.
"""
from __future__ import annotations
import argparse, hashlib, json, os, signal, sys, time
from pathlib import Path

def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p:Path,x:dict):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name('.'+p.name+'.tmp')
 t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); t.replace(p)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,required=True); ap.add_argument('--state-root',type=Path,required=True); a=ap.parse_args()
 cfg=json.loads(a.config.read_text()); root=a.state_root.resolve(); root.mkdir(parents=True,exist_ok=True)
 if cfg.get('schema')!='openevo.ceiling1-ray-placement-backend.v1': raise RuntimeError('Ray placement config schema drift')
 runtime=Path(cfg['ray_runtime_path']); sys.path.insert(0,str(runtime)); os.environ['PYTHONPATH']=str(runtime)+((':'+os.environ['PYTHONPATH']) if os.environ.get('PYTHONPATH') else ''); os.environ['RAY_TMPDIR']=cfg['ray_tmp_dir']; os.environ['CUDA_VISIBLE_DEVICES']=''; os.environ.setdefault('RAY_raylet_start_wait_time_s','60')
 import ray
 from ray.cluster_utils import Cluster
 if ray.__version__!=cfg['ray_version']: raise RuntimeError(f'Ray version drift {ray.__version__}')
 cluster=Cluster(shutdown_at_exit=True)
 budget=float(cfg['gpu_project_budget_mib']); slots=float(cfg['process_slots_per_gpu']); same=float(cfg['same_arm_slots_per_gpu'])
 for pos,idx in enumerate(cfg['managed_gpus']):
  kwargs={'num_cpus':1,'num_gpus':0,'include_dashboard':False,'object_store_memory':80*1024*1024,'resources':{'vram_mib':budget,'process_slot':slots,'arm_3b_slot':same,'arm_q17_slot':same,f'gpu_index_{idx}':1}}
  cluster.add_node(**kwargs)
  # Local multi-node Ray needs the head/GCS to observe each raylet before the next one joins.
  time.sleep(.75)
 ray.init(address=cluster.address,namespace=cfg['ray_namespace'],include_dashboard=False,log_to_driver=False)
 node_map={}
 for n in ray.nodes():
  if not n['Alive']: continue
  for k,v in n['Resources'].items():
   if k.startswith('gpu_index_') and v>=1: node_map[n['NodeID']]=int(k.rsplit('_',1)[1])
 if set(node_map.values())!=set(cfg['managed_gpus']): raise RuntimeError('Ray virtual GPU node mapping incomplete')
 payload={'schema':'openevo.ceiling1-ray-gpu-pool.v1','status':'READY','ray_version':ray.__version__,'address':cluster.address,'pid':os.getpid(),'config_sha256':sha(a.config),'managed_gpus':cfg['managed_gpus'],'node_to_gpu':node_map,'started_at_epoch':time.time(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES','')}
 atomic(root/'cluster.json',payload); atomic(root/'READY.json',payload)
 stop=False
 def handler(*_):
  nonlocal stop; stop=True
 signal.signal(signal.SIGTERM,handler); signal.signal(signal.SIGINT,handler)
 while not stop:
  live={n['NodeID'] for n in ray.nodes() if n['Alive']}
  if not set(node_map)<=live: raise RuntimeError('Ray virtual GPU node died')
  time.sleep(1)
 atomic(root/'STOPPING.json',{'schema':'openevo.ceiling1-ray-gpu-pool-stop.v1','stopped_at_epoch':time.time()})
 ray.shutdown(); cluster.shutdown(); return 0
if __name__=='__main__': raise SystemExit(main())
