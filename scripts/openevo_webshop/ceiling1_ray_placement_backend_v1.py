#!/usr/bin/env python3
"""Drop-in Ray placement/reservation backend for Stage2 formal Docker workloads.

No phase-name co-residency blacklist exists here. A workload declares an arm,
GPU count, and conservative VRAM request. Ray's virtual GPU nodes enforce:
  * aggregate VRAM budget,
  * at most two project processes per physical GPU,
  * at most one live workload from the same arm per physical GPU.
A separate live nvidia-smi gate protects against drift/external occupancy.
"""
from __future__ import annotations
import argparse, fcntl, hashlib, json, os, subprocess, sys, time
from pathlib import Path

def load(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name('.'+p.name+'.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');t.replace(p)
def write_once(p,x):
 p=Path(p); text=json.dumps(x,indent=2,sort_keys=True)+'\n';p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  if p.read_text()!=text: raise RuntimeError(f'write-once conflict: {p}')
  return
 p.write_text(text)
def dstatus(name):
 q=subprocess.run(['docker','inspect','-f','{{.State.Status}}',name],text=True,capture_output=True);return q.stdout.strip() if q.returncode==0 else 'absent'
def running(name): return dstatus(name)=='running'
def gpu_rows():
 q=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True);out={}
 for line in q.splitlines():
  a=[x.strip() for x in line.split(',')];out[int(a[0])]={'uuid':a[1],'used_mib':int(a[2]),'free_mib':int(a[3]),'util_pct':int(a[4])}
 return out
class LockedState:
 def __init__(self,root,cfgsha,managed): self.root=Path(root);self.cfgsha=cfgsha;self.managed=managed
 def __enter__(self):
  self.root.mkdir(parents=True,exist_ok=True);self.f=(self.root/'state.lock').open('a+');fcntl.flock(self.f.fileno(),fcntl.LOCK_EX);p=self.root/'state.json'
  if not p.exists(): dump(p,{'schema':'openevo.ceiling1-ray-placement-state.v1','config_sha256':self.cfgsha,'gpus':{str(i):{'leases':[]} for i in self.managed},'reservations':{},'updated_at_epoch':time.time()})
  self.p=p;self.s=load(p)
  if self.s.get('config_sha256')!=self.cfgsha: raise RuntimeError('Ray placement state config drift')
  return self.s
 def __exit__(self,*_): self.s['updated_at_epoch']=time.time();dump(self.p,self.s);fcntl.flock(self.f.fileno(),fcntl.LOCK_UN);self.f.close()
def ray_api(cfg,root):
 runtime=Path(cfg['ray_runtime_path']);sys.path.insert(0,str(runtime));os.environ['RAY_TMPDIR']=cfg['ray_tmp_dir'];os.environ['CUDA_VISIBLE_DEVICES']=''
 import ray
 if ray.__version__!=cfg['ray_version']: raise RuntimeError('Ray version drift')
 cluster=load(Path(root)/'cluster.json')
 ray.init(address=cluster['address'],namespace=cfg['ray_namespace'],include_dashboard=False,log_to_driver=False,ignore_reinit_error=True)
 from ray.util.placement_group import placement_group,get_placement_group,remove_placement_group
 from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
 return ray,cluster,placement_group,get_placement_group,remove_placement_group,PlacementGroupSchedulingStrategy
def profile(cfg,workload):
 p=(cfg.get('profiles') or {}).get(workload)
 if not p: raise RuntimeError(f'unknown Ray workload intent: {workload}')
 return dict(p)
def bundle_for(p):
 armkey='arm_3b_slot' if p['arm']=='3b' else 'arm_q17_slot'
 return {'vram_mib':float(p['vram_request_mib']),'process_slot':1.0,armkey:1.0}
def pg_name(output,workload,count): return 'ce1-'+hashlib.sha256(f'{Path(output).resolve()}|{workload}|{count}'.encode()).hexdigest()[:24]
def live_gate(cfg,p,gpus):
 rows=gpu_rows(); need=int(p['vram_request_mib'])+int(cfg['global_unallocated_headroom_mib']);out={}
 for i in gpus:
  if i not in cfg['managed_gpus']: raise RuntimeError('Ray selected unmanaged GPU')
  r=rows[i];ok=r['free_mib']>=need;out[str(i)]={**r,'required_free_mib':need,'pass':ok}
  if not ok: raise RuntimeError(f'GPU{i} live free {r["free_mib"]} < required {need}')
 return out
def remove_pg_quiet(get_pg,remove_pg,name):
 try: remove_pg(get_pg(name))
 except Exception: pass
def reconcile(cfg,root):
 ray,cluster,pg,get_pg,remove_pg,strategy=ray_api(cfg,root);now=time.time();cfgsha=sha(Path(args_global.config));managed=cfg['managed_gpus']
 with LockedState(root,cfgsha,managed) as s:
  for rid,r in list((s.get('reservations') or {}).items()):
   if float(r.get('expires_at_epoch',0))<=now:
    remove_pg_quiet(get_pg,remove_pg,r['placement_group_name']);s['reservations'].pop(rid,None)
  for i in managed:
   keep=[]
   for lease in s['gpus'][str(i)].get('leases',[]):
    if running(lease['container']): keep.append(lease)
    else: remove_pg_quiet(get_pg,remove_pg,lease['placement_group_name'])
   s['gpus'][str(i)]['leases']=keep
 ray.shutdown()
def cmd_select(a,cfg):
 ray,cluster,placement_group,get_pg,remove_pg,PGS=ray_api(cfg,a.state_root);p=profile(cfg,a.workload);bundle=bundle_for(p);desired=int(a.count or p['target_gpus']);minimum=int(p['min_gpus']);maximum=int(p['max_gpus']);desired=min(desired,maximum)
 if desired<minimum: raise RuntimeError('requested GPU count below minimum')
 selected=None; chosen_pg=None; chosen_name=None; ready_ms=None
 @ray.remote(num_cpus=0)
 def where(): return ray.get_runtime_context().get_node_id()
 try:
  for count in range(desired,minimum-1,-1):
   name=pg_name(a.output,a.workload,count)
   try: old=get_pg(name); remove_pg(old); time.sleep(.05)
   except Exception: pass
   pg=placement_group([bundle for _ in range(count)],strategy=(cfg['rollout_strategy'] if count>1 else cfg['single_gpu_strategy']),name=name,lifetime='detached')
   t0=time.perf_counter()
   ready=ray.wait([pg.ready()],timeout=.35)[0]
   if not ready: remove_pg(pg);continue
   ids=[]
   for bi in range(count):
    r=where.options(resources=bundle,scheduling_strategy=PGS(placement_group=pg,placement_group_bundle_index=bi)).remote();ids.append(r)
   node_ids=ray.get(ids);node_map={str(k):int(v) for k,v in cluster['node_to_gpu'].items()};gpus=sorted(node_map[x] for x in node_ids)
   live=live_gate(cfg,p,gpus);selected=gpus;chosen_pg=pg;chosen_name=name;ready_ms=(time.perf_counter()-t0)*1000;break
  if selected is None: raise RuntimeError('Ray has no immediately feasible placement')
  rid=hashlib.sha256(f'{Path(a.output).resolve()}|{a.workload}'.encode()).hexdigest()[:24];now=time.time()
  rec={'schema':'openevo.ceiling1-ray-selection.v1','workload':a.workload,'arm':p['arm'],'selected_gpus':selected,'selected_gpu_uuids':[gpu_rows()[i]['uuid'] for i in selected],'placement_group_name':chosen_name,'reservation_id':rid,'config_sha256':sha(a.config),'vram_request_mib_per_gpu':p['vram_request_mib'],'live_gate':live,'ray_ready_ms':ready_ms,'selected_at_epoch':now,'no_mid_workload_reshard':True,'scheduler':'ray'}
  with LockedState(a.state_root,sha(a.config),cfg['managed_gpus']) as s: s.setdefault('reservations',{})[rid]={'placement_group_name':chosen_name,'workload':a.workload,'gpus':selected,'created_at_epoch':now,'expires_at_epoch':now+int(cfg['reservation_ttl_seconds']),'selection_path':str(a.output.resolve())}
  write_once(a.output,rec);print(json.dumps(rec,indent=2,sort_keys=True));return 0
 finally: ray.shutdown()
def cmd_run(a,cfg):
 ray,cluster,pg,get_pg,remove_pg,PGS=ray_api(cfg,a.state_root);p=profile(cfg,a.workload);sel=load(a.selection_receipt) if a.selection_receipt else None
 if not sel or sel.get('schema')!='openevo.ceiling1-ray-selection.v1' or sel.get('workload')!=a.workload: raise RuntimeError('Ray run requires matching selection receipt')
 if sel.get('config_sha256')!=sha(a.config) or list(map(int,sel['selected_gpus']))!=list(map(int,a.gpus)): raise RuntimeError('Ray selection/run identity drift')
 get_pg(sel['placement_group_name']);live=live_gate(cfg,p,a.gpus)
 env={**os.environ,'CEILING1_LEASE_ID':a.lease_id,'CEILING1_LEASE_WORKLOAD':a.workload,'CEILING1_RAY_RESERVATION_ID':sel['reservation_id'],'CEILING1_RAY_PLACEMENT_GROUP':sel['placement_group_name']}
 q=subprocess.run([str(a.launch_script)],env=env)
 if q.returncode: raise RuntimeError('science launch script failed')
 deadline=time.monotonic()+4
 while time.monotonic()<deadline and not running(a.container): time.sleep(.05)
 if not running(a.container): raise RuntimeError('science container did not become running')
 lease={'lease_id':a.lease_id,'workload':a.workload,'container':a.container,'gpus':list(map(int,a.gpus)),'placement_group_name':sel['placement_group_name'],'reservation_id':sel['reservation_id'],'acquired':time.time(),'vram_request_mib_per_gpu':p['vram_request_mib']}
 with LockedState(a.state_root,sha(a.config),cfg['managed_gpus']) as s:
  s.get('reservations',{}).pop(sel['reservation_id'],None)
  for i in a.gpus:s['gpus'][str(i)].setdefault('leases',[]).append(lease)
 dump(Path(a.state_root)/'receipts'/f'{a.lease_id}.launch.json',{'schema':'openevo.ceiling1-ray-launch.v1','status':'RUNNING','selection':sel,'live_gate':live,'lease':lease})
 watcher=[sys.executable,str(Path(__file__).resolve()),'--config',str(a.config),'--state-root',str(a.state_root),'watch-release','--lease-id',a.lease_id,'--container',a.container,'--placement-group-name',sel['placement_group_name']]
 subprocess.Popen(watcher,stdout=(Path(a.state_root)/'watchers.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
 print(json.dumps(lease,indent=2,sort_keys=True));ray.shutdown();return 0
def cmd_watch(a,cfg):
 while running(a.container):time.sleep(1)
 ray,cluster,pg,get_pg,remove_pg,PGS=ray_api(cfg,a.state_root);remove_pg_quiet(get_pg,remove_pg,a.placement_group_name)
 with LockedState(a.state_root,sha(a.config),cfg['managed_gpus']) as s:
  for i in cfg['managed_gpus']:s['gpus'][str(i)]['leases']=[x for x in s['gpus'][str(i)].get('leases',[]) if x.get('lease_id')!=a.lease_id]
 ray.shutdown();return 0
def main():
 global args_global
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--state-root',type=Path,required=True);sp=ap.add_subparsers(dest='cmd',required=True)
 sp.add_parser('validate');sp.add_parser('status');sp.add_parser('reconcile')
 q=sp.add_parser('select');q.add_argument('--workload',required=True);q.add_argument('--count',type=int);q.add_argument('--output',type=Path,required=True)
 q=sp.add_parser('run');q.add_argument('--lease-id',required=True);q.add_argument('--workload',required=True);q.add_argument('--container',required=True);q.add_argument('--launch-script',type=Path,required=True);q.add_argument('--gpus',nargs='+',type=int,required=True);q.add_argument('--vram-qualification',type=Path);q.add_argument('--selection-receipt',type=Path,required=True)
 q=sp.add_parser('watch-release');q.add_argument('--lease-id',required=True);q.add_argument('--container',required=True);q.add_argument('--placement-group-name',required=True)
 args_global=ap.parse_args();cfg=load(args_global.config)
 if cfg.get('schema')!='openevo.ceiling1-ray-placement-backend.v1': raise RuntimeError('config schema drift')
 if args_global.cmd=='validate': print('CEILING1_RAY_PLACEMENT_CONFIG=PASS');return 0
 if args_global.cmd=='status': reconcile(cfg,args_global.state_root);print(json.dumps(load(Path(args_global.state_root)/'state.json'),indent=2,sort_keys=True));return 0
 if args_global.cmd=='reconcile': reconcile(cfg,args_global.state_root);return 0
 if args_global.cmd=='select': return cmd_select(args_global,cfg)
 if args_global.cmd=='run': return cmd_run(args_global,cfg)
 if args_global.cmd=='watch-release': return cmd_watch(args_global,cfg)
if __name__=='__main__': raise SystemExit(main())
