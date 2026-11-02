#!/usr/bin/env python3
"""Ray virtual-GPU pool for OpenEVO Stage2.

Each physical GPU4-7 is represented as one logical Ray node. The pool never
sees CUDA; formal CUDA remains inside Docker scientific launchers. A separate
heartbeat prevents stale READY files from masquerading as a live scheduler.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p: Path, x: dict):
    p.parent.mkdir(parents=True, exist_ok=True); t = p.with_name('.' + p.name + '.tmp')
    t.write_text(json.dumps(x, indent=2, sort_keys=True) + '\n'); t.replace(p)
def pid_alive(pid: int) -> bool:
    try: os.kill(pid, 0); return True
    except (OSError, ValueError): return False


def physical_gpu_uuids() -> dict[int,str]:
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
    out={}
    for line in text.splitlines():
        idx,uuid=[x.strip() for x in line.split(',',1)];out[int(idx)]=uuid
    return out


def container_running(name: str) -> bool:
    q=subprocess.run(['docker','inspect','-f','{{.State.Running}}',name],text=True,capture_output=True)
    return q.returncode==0 and q.stdout.strip()=='true'

def recorded_live_containers(state: dict) -> list[str]:
    names=[]
    for gpu in (state.get('gpus') or {}).values():
        for lease in gpu.get('leases',[]):
            name=str(lease.get('container') or '')
            if name and container_running(name): names.append(name)
    for r in (state.get('reservations') or {}).values():
        if not r.get('launching'): continue
        name=str(r.get('container') or '')
        if name and container_running(name): names.append(name)
    return sorted(set(names))

def overlapping_pool_processes(managed: list[int], proc_root: Path=Path('/proc'), self_pid: int|None=None) -> list[dict]:
    overlap=[]; mine=set(map(int,managed)); self_pid=os.getpid() if self_pid is None else int(self_pid)
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name)==self_pid: continue
        try:
            raw=(entry/'cmdline').read_bytes(); argv=[x.decode(errors='replace') for x in raw.split(b'\0') if x]
            if not any('ceiling1_ray_gpu_pool_' in x for x in argv): continue
            if '--config' not in argv: continue
            cp=Path(argv[argv.index('--config')+1]); other=json.loads(cp.read_text()); theirs=set(map(int,other.get('managed_gpus') or []))
            if mine & theirs: overlap.append({'pid':int(entry.name),'config':str(cp),'managed_gpus':sorted(theirs),'overlap':sorted(mine & theirs)})
        except Exception:
            continue
    return overlap

def acquire_host_pool_lock(path: Path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);f=path.open('a+')
    try: fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        f.close();raise RuntimeError(f'Ray GPU pool host lock already held: {path}')
    return f


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', type=Path, required=True); ap.add_argument('--state-root', type=Path, required=True); a = ap.parse_args()
    cfg = json.loads(a.config.read_text()); root = a.state_root.resolve(); root.mkdir(parents=True, exist_ok=True); cfgsha = sha(a.config)
    if cfg.get('schema') != 'openevo.ceiling1-ray-placement-backend.202609042025': raise RuntimeError('Ray placement config schema drift')
    expected={int(k):v for k,v in (cfg.get('managed_gpu_uuids') or {}).items()}; observed=physical_gpu_uuids()
    if set(expected)!=set(cfg['managed_gpus']) or any(observed.get(i)!=expected[i] for i in cfg['managed_gpus']): raise RuntimeError(f'Ray managed GPU index/UUID identity drift: expected={expected} observed={observed}')
    host_lock=acquire_host_pool_lock(Path(cfg['host_pool_lock_path']))
    overlaps=overlapping_pool_processes(cfg['managed_gpus'])
    if overlaps:
        host_lock.close();raise RuntimeError(f'Ray overlapping GPU pool already running: {overlaps}')
    state_path=root/'state.json'
    if state_path.is_file():
        prior=json.loads(state_path.read_text())
        if prior.get('config_sha256')!=cfgsha: host_lock.close();raise RuntimeError('Ray state root belongs to a different config; archive it or use a new state root')
        live=recorded_live_containers(prior)
        if live: host_lock.close();raise RuntimeError(f'Ray pool restart forbidden while recorded live containers exist: {live}')
    ready_path = root / 'READY.json'; heartbeat_path = root / 'HEARTBEAT.json'
    if ready_path.is_file():
        try:
            old = json.loads(ready_path.read_text()); old_pid = int(old.get('pid', -1)); hb = json.loads(heartbeat_path.read_text()) if heartbeat_path.is_file() else {}
            fresh = hb.get('status')=='READY' and hb.get('pid')==old_pid and hb.get('config_sha256')==cfgsha and time.time()-float(hb.get('updated_at_epoch',0)) <= float(cfg.get('pool_heartbeat_max_age_seconds',5))
        except Exception:
            old_pid = -1; fresh = False
        if old_pid > 0 and fresh and pid_alive(old_pid): raise RuntimeError(f'Ray pool already running pid={old_pid}')
    for stale in ('STOPPING.json', 'STOPPED.json'):
        (root / stale).unlink(missing_ok=True)
    runtime = Path(cfg['ray_runtime_path']); sys.path.insert(0, str(runtime))
    os.environ['PYTHONPATH'] = str(runtime) + ((':' + os.environ['PYTHONPATH']) if os.environ.get('PYTHONPATH') else '')
    os.environ['RAY_TMPDIR'] = cfg['ray_tmp_dir']; os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ.setdefault('RAY_raylet_start_wait_time_s', '60')
    import ray
    from ray.cluster_utils import Cluster
    if ray.__version__ != cfg['ray_version']: raise RuntimeError(f'Ray version drift {ray.__version__}')
    cluster = Cluster(shutdown_at_exit=True); node_map = {}; started = time.time(); stop = False
    def handler(*_):
        nonlocal stop; stop = True
    signal.signal(signal.SIGTERM, handler); signal.signal(signal.SIGINT, handler)
    try:
        budget = float(cfg['gpu_project_budget_mib']); slots = float(cfg['process_slots_per_gpu']); same = float(cfg['same_arm_slots_per_gpu'])
        for idx in cfg['managed_gpus']:
            cluster.add_node(num_cpus=1, num_gpus=0, include_dashboard=False, object_store_memory=80 * 1024 * 1024, resources={'vram_mib': budget, 'process_slot': slots, 'arm_3b_slot': same, 'arm_q17_slot': same, f'gpu_index_{idx}': slots})
            time.sleep(.75)
        ray.init(address=cluster.address, namespace=cfg['ray_namespace'], include_dashboard=False, log_to_driver=False)
        for n in ray.nodes():
            if not n['Alive']: continue
            for k, v in n['Resources'].items():
                if k.startswith('gpu_index_') and v >= 1: node_map[n['NodeID']] = int(k.rsplit('_', 1)[1])
        if set(node_map.values()) != set(cfg['managed_gpus']): raise RuntimeError('Ray virtual GPU node mapping incomplete')
        payload = {'schema': 'openevo.ceiling1-ray-gpu-pool.202609042025', 'status': 'READY', 'ray_version': ray.__version__, 'address': cluster.address, 'pid': os.getpid(), 'config_sha256': cfgsha, 'managed_gpus': cfg['managed_gpus'], 'node_to_gpu': node_map, 'started_at_epoch': started, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', '')}
        atomic(root / 'cluster.json', payload); atomic(ready_path, payload)
        while not stop:
            live = {n['NodeID'] for n in ray.nodes() if n['Alive']}
            if not set(node_map) <= live: raise RuntimeError('Ray virtual GPU node died')
            atomic(heartbeat_path, {'schema': 'openevo.ceiling1-ray-gpu-pool-heartbeat.202609042025', 'status': 'READY', 'pid': os.getpid(), 'config_sha256': cfgsha, 'address': cluster.address, 'updated_at_epoch': time.time()})
            time.sleep(1)
        return 0
    finally:
        atomic(root / 'STOPPING.json', {'schema': 'openevo.ceiling1-ray-gpu-pool-stop.202609042025', 'status': 'STOPPING', 'pid': os.getpid(), 'started_at_epoch': started, 'stopping_at_epoch': time.time(), 'config_sha256': cfgsha})
        try: ray.shutdown()
        except Exception: pass
        try: cluster.shutdown()
        except Exception: pass
        # Only retire this process's READY/heartbeat. A newer pool must never be erased by an older process exiting late.
        try:
            current = json.loads(ready_path.read_text()) if ready_path.is_file() else {}
            if int(current.get('pid', -1)) == os.getpid(): ready_path.unlink(missing_ok=True)
        except Exception: pass
        try:
            current = json.loads(heartbeat_path.read_text()) if heartbeat_path.is_file() else {}
            if int(current.get('pid', -1)) == os.getpid(): heartbeat_path.unlink(missing_ok=True)
        except Exception: pass
        atomic(root / 'STOPPED.json', {'schema': 'openevo.ceiling1-ray-gpu-pool-stop.202609042025', 'status': 'STOPPED', 'pid': os.getpid(), 'started_at_epoch': started, 'stopped_at_epoch': time.time(), 'config_sha256': cfgsha})
        try: host_lock.close()
        except Exception: pass


if __name__ == '__main__': raise SystemExit(main())
