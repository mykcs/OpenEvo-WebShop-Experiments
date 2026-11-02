#!/usr/bin/env python3
"""Thin liveness supervisor for one Ray Stage2 arm.

It never retries BLOCKED science state. If the controller process disappears
while a Docker workload is still alive, it waits for that arm's Ray leases to
drain and only then restarts the exact controller argv.
"""
from __future__ import annotations
import json, signal, subprocess, sys, time
from pathlib import Path

def load(p): return json.loads(Path(p).read_text())
def arg(flag):
    try: return sys.argv[sys.argv.index(flag)+1]
    except (ValueError,IndexError): raise SystemExit(f'missing {flag}')
def main():
    arm=arg('--arm'); repo=Path(arg('--repo')).resolve(); out=Path(arg('--output')).resolve(); mig=load(Path(arg('--recovery-amendment')))
    rayroot=Path(mig['ray_pool_ready_path']).resolve().parent; cfg=load(repo/'configs/experiment/openevo-ceiling1-ray-placement-backend-202609042025.json')
    log=out/'controller-supervisor.jsonl'; stopping=False; child=None
    def live_arm_work():
        p=rayroot/'state.json'
        if not p.is_file(): return False
        s=load(p); profiles=cfg.get('profiles') or {}
        for g in (s.get('gpus') or {}).values():
            for lease in g.get('leases',[]):
                if (profiles.get(lease.get('workload')) or {}).get('arm')==arm: return True
        return False
    def admission_paused():
        p=rayroot/'state.json'
        return p.is_file() and bool((load(p).get('admission') or {}).get('paused'))
    def record(event,**extra):
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as f:f.write(json.dumps({'event':event,'arm':arm,'at_epoch':time.time(),**extra},sort_keys=True)+'\n')
    def stop(*_):
        nonlocal stopping;stopping=True
        if child is not None and child.poll() is None: child.terminate()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    runner=repo/'scripts/openevo_webshop/run_ceiling1_stage2_ray_202609042025.py'; raw=sys.argv[1:]
    while not stopping:
        child=subprocess.Popen([sys.executable,str(runner),*raw]);record('controller_started',pid=child.pid)
        rc=child.wait();record('controller_exited',pid=child.pid,returncode=rc)
        if stopping:return 0
        state=load(out/'execution-state.json')
        if state.get('status')=='STAGE2_SEALED':return 0
        if state.get('status')=='BLOCKED':return rc or 1
        if admission_paused():return 0
        if state.get('status')!='RUNNING':return rc or 1
        while live_arm_work() and not stopping:time.sleep(1)
        if stopping:return 0
        state=load(out/'execution-state.json')
        if state.get('status')!='RUNNING' or admission_paused():return 0 if admission_paused() else (rc or 1)
        record('controller_restart_after_live_work_drained');time.sleep(1)
    return 0
if __name__=='__main__':raise SystemExit(main())
