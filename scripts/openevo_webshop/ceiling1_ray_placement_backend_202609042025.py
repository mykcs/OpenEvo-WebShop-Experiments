#!/usr/bin/env python3
"""Ray placement/reservation backend for shared small-arm Stage2.

Science declares workload intents. This backend only controls placement,
co-residency, admission, and lifecycle accounting. Phase names never imply
exclusivity. Existing leases are never preempted by an operator pause.
"""
from __future__ import annotations

import argparse
import fcntl
import itertools
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.openevo_webshop.elastic_gpu_lane import binding_contract, binding_evidence, row_qualifies


def load(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name('.' + p.name + '.tmp')
    t.write_text(json.dumps(x, indent=2, sort_keys=True) + '\n')
    t.replace(p)
def write_once(p, x):
    p = Path(p); text = json.dumps(x, indent=2, sort_keys=True) + '\n'
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        if p.read_text() != text: raise RuntimeError(f'write-once conflict: {p}')
        return
    p.write_text(text)
def dstatus(name):
    q = subprocess.run(['docker', 'inspect', '-f', '{{.State.Status}}', name], text=True, capture_output=True)
    return q.stdout.strip() if q.returncode == 0 else 'absent'
def running(name): return dstatus(name) == 'running'
def gpu_rows():
    q = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    out = {}
    for line in q.splitlines():
        a = [x.strip() for x in line.split(',')]
        out[int(a[0])] = {'uuid': a[1], 'name': a[2], 'total_mib': int(a[3]), 'used_mib': int(a[4]), 'free_mib': int(a[5]), 'util_pct': int(a[6])}
    return out


class RayAdmissionPaused(RuntimeError):
    pass



def acquire_controller_lock(output, arm, execution_sha):
    root=Path(output);root.mkdir(parents=True,exist_ok=True);path=root/'controller.lock';f=path.open('a+')
    try: fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        f.seek(0);holder=f.read().strip();f.close();raise RuntimeError(f'CONTROLLER_SINGLE_WRITER_HELD arm={arm} output={root} holder={holder}')
    f.seek(0);f.truncate();f.write(json.dumps({'schema':'openevo.ceiling1-stage2-controller-lock.202609042025','arm':arm,'execution_git_sha':execution_sha,'pid':os.getpid(),'acquired_at_epoch':time.time(),'lock_authority':'os-flock; file content is diagnostic only'},sort_keys=True)+'\n');f.flush();os.fsync(f.fileno());return f

class LockedState:
    def __init__(self, root, cfgsha, managed): self.root = Path(root); self.cfgsha = cfgsha; self.managed = managed
    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.f = (self.root / 'state.lock').open('a+')
        fcntl.flock(self.f.fileno(), fcntl.LOCK_EX)
        p = self.root / 'state.json'
        if not p.exists():
            dump(p, {'schema': 'openevo.ceiling1-ray-placement-state.202609042025', 'config_sha256': self.cfgsha, 'gpus': {str(i): {'leases': []} for i in self.managed}, 'reservations': {}, 'admission': {'paused': False, 'reason': None, 'updated_at_epoch': time.time()}, 'updated_at_epoch': time.time()})
        self.p = p; self.s = load(p)
        if self.s.get('config_sha256') != self.cfgsha: raise RuntimeError('Ray placement state config drift')
        self.s.setdefault('reservations', {})
        self.s.setdefault('admission', {'paused': False, 'reason': None, 'updated_at_epoch': time.time()})
        for i in self.managed: self.s.setdefault('gpus', {}).setdefault(str(i), {'leases': []})
        return self.s
    def __exit__(self, *_):
        self.s['updated_at_epoch'] = time.time(); dump(self.p, self.s)
        fcntl.flock(self.f.fileno(), fcntl.LOCK_UN); self.f.close()


def assert_admission_open(state):
    admission = state.get('admission') or {}
    if admission.get('paused') is True:
        raise RayAdmissionPaused(f"RAY_ADMISSION_PAUSED: {admission.get('reason') or 'operator resource pause'}")


def pool_liveness(cfg, root, config_path):
    root = Path(root); config_path = Path(config_path)
    ready_path = root / 'READY.json'; heartbeat_path = root / 'HEARTBEAT.json'
    if not ready_path.is_file(): raise RuntimeError('RAY_POOL_UNAVAILABLE: READY.json missing')
    ready = load(ready_path); expected_cfg = sha(config_path)
    if ready.get('status') != 'READY' or ready.get('config_sha256') != expected_cfg:
        raise RuntimeError('RAY_POOL_UNAVAILABLE: READY identity drift')
    if not heartbeat_path.is_file(): raise RuntimeError('RAY_POOL_UNAVAILABLE: HEARTBEAT.json missing')
    heartbeat = load(heartbeat_path)
    if heartbeat.get('status') != 'READY' or heartbeat.get('config_sha256') != expected_cfg or heartbeat.get('pid') != ready.get('pid'):
        raise RuntimeError('RAY_POOL_UNAVAILABLE: heartbeat identity drift')
    age = time.time() - float(heartbeat.get('updated_at_epoch', 0))
    max_age = float(cfg.get('pool_heartbeat_max_age_seconds', 5))
    if age < 0 or age > max_age: raise RuntimeError(f'RAY_POOL_UNAVAILABLE: stale heartbeat age={age:.3f}s max={max_age:.3f}s')
    pid = int(ready.get('pid', -1))
    try: os.kill(pid, 0)
    except (OSError, ValueError): raise RuntimeError(f'RAY_POOL_UNAVAILABLE: pool pid {pid} not alive')
    return ready, heartbeat


def ray_api(cfg, root, config_path):
    pool_liveness(cfg, root, config_path)
    runtime = Path(cfg['ray_runtime_path']); sys.path.insert(0, str(runtime))
    os.environ['RAY_TMPDIR'] = cfg['ray_tmp_dir']; os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import ray
    if ray.__version__ != cfg['ray_version']: raise RuntimeError('Ray version drift')
    cluster = load(Path(root) / 'cluster.json')
    ray.init(address=cluster['address'], namespace=cfg['ray_namespace'], include_dashboard=False, log_to_driver=False, ignore_reinit_error=True)
    from ray.util.placement_group import placement_group, get_placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    return ray, cluster, placement_group, get_placement_group, remove_placement_group, PlacementGroupSchedulingStrategy


PROFILE_REGISTRY_SCHEMA = 'openevo.ray-workload-profile-registry.202609052200'
SLOT_CLASS_RESOURCE = {
    '3b_primary': 'arm_3b_slot',
    'q17_primary': 'arm_q17_slot',
    'secondary_spare': 'arm_q17_slot',
}

def load_profile_registry(path, base_config_sha256):
    if not path:
        return None
    path = Path(path)
    reg = load(path)
    if reg.get('schema') != PROFILE_REGISTRY_SCHEMA:
        raise RuntimeError('Ray profile registry schema drift')
    compatible = set(reg.get('compatible_base_config_sha256') or [])
    if base_config_sha256 not in compatible:
        raise RuntimeError('Ray profile registry/base-config incompatibility')
    if not isinstance(reg.get('profiles'), dict) or not reg['profiles']:
        raise RuntimeError('Ray profile registry has no profiles')
    reg = dict(reg)
    reg['_sha256'] = sha(path)
    reg['_path'] = str(path.resolve())
    return reg

def profile(cfg, workload, registry=None):
    base = (cfg.get('profiles') or {}).get(workload)
    extra = (registry or {}).get('profiles', {}).get(workload) if registry else None
    if base is not None and extra is not None:
        raise RuntimeError(f'Ray profile registry may not shadow base workload: {workload}')
    p = base if base is not None else extra
    if not p:
        raise RuntimeError(f'unknown Ray workload intent: {workload}')
    p = dict(p)
    p['_profile_source'] = 'base' if base is not None else 'registry'
    if p['_profile_source'] == 'registry' and not p.get('slot_class'):
        raise RuntimeError(f'external Ray workload missing slot_class: {workload}')
    return p

def bundle_for(p):
    slot_class = p.get('slot_class')
    if slot_class:
        armkey = SLOT_CLASS_RESOURCE.get(slot_class)
        if not armkey:
            raise RuntimeError(f'unknown Ray slot_class: {slot_class}')
    else:
        armkey = 'arm_3b_slot' if p['arm'] == '3b' else 'arm_q17_slot'
    return {'vram_mib': float(p['vram_request_mib']), 'process_slot': 1.0, armkey: 1.0}
def pg_name(output, workload, count, combo):
    physical = ','.join(map(str, combo))
    return 'ce1-' + hashlib.sha256(f'{Path(output).resolve()}|{workload}|{count}|{physical}'.encode()).hexdigest()[:24]
def physically_eligible_gpus(cfg,p,rows=None):
    rows=rows or gpu_rows();need=int(p['vram_request_mib'])+int(cfg['global_unallocated_headroom_mib']);eligible=[]
    contract=binding_contract(cfg)
    for i in cfg['managed_gpus']:
        r=rows.get(i)
        if not r:continue
        qualified,_=row_qualifies(contract,i,r)
        if not qualified:continue
        if int(r.get('free_mib',-1))>=need:eligible.append(i)
    eligible.sort(key=lambda i:(-int(rows[i]['free_mib']),i))
    return eligible,rows,need

def selection_count_bounds(p, requested_count=None, exact_count=False):
    if exact_count and requested_count is None: raise RuntimeError('--exact-count requires --count')
    desired=min(int(requested_count or p['target_gpus']),int(p['max_gpus']))
    minimum=desired if exact_count else int(p['min_gpus'])
    if desired<minimum: raise RuntimeError('requested GPU count below minimum')
    return desired,minimum

def live_gate(cfg, p, gpus):
    rows = gpu_rows(); need = int(p['vram_request_mib']) + int(cfg['global_unallocated_headroom_mib']); out = {}
    contract=binding_contract(cfg)
    for i in gpus:
        if i not in cfg['managed_gpus']: raise RuntimeError('Ray selected unmanaged GPU')
        r = rows[i]; expected_uuid=(cfg.get('managed_gpu_uuids') or {}).get(str(i))
        qualified,reason=row_qualifies(contract,i,r)
        if not qualified:
            if reason == 'uuid_mismatch': raise RuntimeError(f'GPU{i} UUID identity drift: observed {r["uuid"]} expected {expected_uuid}')
            raise RuntimeError(f'GPU{i} resource-class qualification drift: {reason}')
        ok = r['free_mib'] >= need; out[str(i)] = {**r, 'expected_uuid': expected_uuid, 'required_free_mib': need, 'pass': ok}
        if not ok: raise RuntimeError(f'GPU{i} live free {r["free_mib"]} < required {need}')
    return out
def remove_pg_quiet(get_pg, remove_pg, name):
    """Request removal only for the named Ray PG and preserve uncertain cleanup.

    Ray 2.57.0 raises ValueError when a named group is *absent*. A disconnected
    GCS/provider can instead fail with other exceptions; those do not prove
    resources were released and must retain the cleanup_pending reservation.
    True on remove_pg means removal was requested, not a host GPU-release proof.
    """
    try:
        group = get_pg(name)
    except ValueError:
        return True  # Ray 2.57.0: named PG is already absent
    except Exception:
        return False  # provider/connection unknown: retain reservation
    try:
        remove_pg(group)
        return True
    except Exception:
        return False

def lease_from_launching_reservation(r: dict, rid: str, now: float, recovered: bool=False) -> dict:
    required=('lease_id','workload','container','gpus','placement_group_name','vram_request_mib_per_gpu')
    missing=[k for k in required if not r.get(k) and r.get(k)!=0]
    if missing: raise RuntimeError(f'launching reservation missing lease identity: {missing}')
    return {'lease_id':r['lease_id'],'workload':r['workload'],'container':r['container'],'gpus':list(map(int,r['gpus'])),'placement_group_name':r['placement_group_name'],'reservation_id':rid,'acquired':float(r.get('launching_at_epoch') or now),'vram_request_mib_per_gpu':int(r['vram_request_mib_per_gpu']),'recovered_from_launching_reservation':bool(recovered)}

def install_lease_once(state: dict, lease: dict):
    lid=lease['lease_id']
    for i in lease['gpus']:
        bucket=state['gpus'][str(i)].setdefault('leases',[])
        if not any(x.get('lease_id')==lid for x in bucket): bucket.append(dict(lease))

def spawn_recovered_watcher(config_path: Path, root: Path, lease: dict):
    watcher=[sys.executable,str(Path(__file__).resolve()),'--config',str(config_path),'--state-root',str(root),'watch-release','--lease-id',lease['lease_id'],'--container',lease['container'],'--placement-group-name',lease['placement_group_name']]
    subprocess.Popen(watcher,stdout=(Path(root)/'watchers.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)

def reconcile(cfg, root, config_path):
    ray, cluster, pg, get_pg, remove_pg, strategy = ray_api(cfg, root, config_path)
    now = time.time(); cfgsha = sha(Path(config_path)); managed = cfg['managed_gpus']
    promoted=[]
    try:
        with LockedState(root, cfgsha, managed) as s:
            grace=float(cfg.get('launch_transaction_grace_seconds',10))
            for rid, r in list((s.get('reservations') or {}).items()):
                if r.get('launching'):
                    name=str(r.get('container') or '')
                    if name and running(name):
                        lease=lease_from_launching_reservation(r,rid,now,recovered=True);install_lease_once(s,lease);s['reservations'].pop(rid,None);promoted.append(lease);continue
                    age=now-float(r.get('launching_at_epoch') or now)
                    if age <= grace: continue
                    if remove_pg_quiet(get_pg,remove_pg,r['placement_group_name']): s['reservations'].pop(rid,None)
                    else: r['cleanup_pending']=True
                    continue
                if r.get('cleanup_pending') or float(r.get('expires_at_epoch', 0)) <= now:
                    if remove_pg_quiet(get_pg, remove_pg, r['placement_group_name']): s['reservations'].pop(rid, None)
                    else: r['cleanup_pending'] = True
            for i in managed:
                keep = []
                for lease in s['gpus'][str(i)].get('leases', []):
                    if running(lease['container']): keep.append(lease)
                    elif remove_pg_quiet(get_pg, remove_pg, lease['placement_group_name']): pass
                    else:
                        lease['release_pending'] = True; keep.append(lease)
                s['gpus'][str(i)]['leases'] = keep
        for lease in promoted:
            dump(Path(root)/'receipts'/f"{lease['lease_id']}.recovered-launch.json",{'schema':'openevo.ceiling1-ray-launch-recovery.202609042102','status':'RECOVERED_RUNNING_LEASE','lease':lease,'recovered_at_epoch':time.time()})
            spawn_recovered_watcher(Path(config_path),Path(root),lease)
    finally:
        ray.shutdown()


def cmd_select(a, cfg):
    cfgsha=sha(a.config)
    registry=load_profile_registry(getattr(a,'profile_registry',None),cfgsha)
    with LockedState(a.state_root,cfgsha,cfg['managed_gpus']) as state:assert_admission_open(state)
    ray,cluster,placement_group,get_pg,remove_pg,PGS=ray_api(cfg,a.state_root,a.config)
    p=profile(cfg,a.workload,registry);base_bundle=bundle_for(p);desired,minimum=selection_count_bounds(p,a.count,getattr(a,'exact_count',False))
    eligible,physical_rows,need=physically_eligible_gpus(cfg,p)
    selected=None;chosen_pg=None;chosen_name=None;ready_ms=None;live=None;committed=False
    try:
        for count in range(min(desired,len(eligible)),minimum-1,-1):
            for combo in itertools.combinations(eligible,count):
                name=pg_name(a.output,a.workload,count,combo)
                bundles=[]
                for idx in combo:
                    b=dict(base_bundle);b[f'gpu_index_{idx}']=1.0;bundles.append(b)
                try:old=get_pg(name);remove_pg(old);time.sleep(.05)
                except Exception:pass
                pg=placement_group(bundles,strategy=(cfg['rollout_strategy'] if count>1 else cfg['single_gpu_strategy']),name=name,lifetime='detached')
                t0=time.perf_counter();ready=ray.wait([pg.ready()],timeout=float(cfg.get('placement_ready_timeout_seconds',2.0)))[0]
                if not ready:remove_pg(pg);continue
                try:live=live_gate(cfg,p,list(combo))
                except RuntimeError:remove_pg(pg);continue
                selected=list(combo);chosen_pg=pg;chosen_name=name;ready_ms=(time.perf_counter()-t0)*1000;break
            if selected is not None:break
        if selected is None:raise RuntimeError(f'RAY_NO_FEASIBLE_PLACEMENT: physically_eligible={eligible} required_free_mib={need}')
        rid=hashlib.sha256(f'{Path(a.output).resolve()}|{a.workload}'.encode()).hexdigest()[:24];now=time.time()
        rec={'schema':'openevo.ceiling1-ray-selection.202609042025','workload':a.workload,'arm':p['arm'],'selected_gpus':selected,'selected_gpu_uuids':[physical_rows[i]['uuid'] for i in selected],'gpu_binding':binding_evidence(cfg,selected,physical_rows,now),'placement_group_name':chosen_name,'reservation_id':rid,'config_sha256':cfgsha,'vram_request_mib_per_gpu':p['vram_request_mib'],'live_gate':live,'physical_prefilter':{'eligible_gpus':eligible,'required_free_mib':need},'ray_ready_ms':ready_ms,'selected_at_epoch':now,'no_mid_workload_reshard':True,'scheduler':'ray','profile_source':p.get('_profile_source','base')}
        if registry is not None:
            rec['profile_registry_sha256']=registry['_sha256'];rec['profile_registry_path']=registry['_path'];rec['slot_class']=p['slot_class']
        reservation={'placement_group_name':chosen_name,'workload':a.workload,'gpus':selected,'created_at_epoch':now,'expires_at_epoch':now+int(cfg['reservation_ttl_seconds']),'selection_path':str(a.output.resolve()),'launching':False}
        with LockedState(a.state_root,cfgsha,cfg['managed_gpus']) as state:
            assert_admission_open(state);state['reservations'][rid]=reservation
            try:write_once(a.output,rec)
            except Exception:state['reservations'].pop(rid,None);raise
        committed=True;print(json.dumps(rec,indent=2,sort_keys=True));return 0
    finally:
        if chosen_pg is not None and not committed:remove_pg_quiet(get_pg,remove_pg,chosen_name)
        ray.shutdown()


def ensure_selection_reservation(a,cfg,p,sel,ray,placement_group,get_pg,remove_pg):
    cfgsha=sha(a.config);rid=sel['reservation_id'];name=sel['placement_group_name'];selected=list(map(int,sel['selected_gpus']))
    with LockedState(a.state_root,cfgsha,cfg['managed_gpus']) as state:
        assert_admission_open(state)
        existing=state.get('reservations',{}).get(rid)
        if existing:return existing
    bundles=[]
    for idx in selected:
        b=bundle_for(p);b[f'gpu_index_{idx}']=1.0;bundles.append(b)
    group=None;committed=False
    try:
        remove_pg_quiet(get_pg,remove_pg,name)
        group=placement_group(bundles,strategy=(cfg['rollout_strategy'] if len(selected)>1 else cfg['single_gpu_strategy']),name=name,lifetime='detached')
        ready=ray.wait([group.ready()],timeout=float(cfg.get('placement_ready_timeout_seconds',2.0)))[0]
        if not ready:raise RuntimeError('Ray could not reconstruct immutable selection reservation')
        live_gate(cfg,p,selected);now=time.time();rec={'placement_group_name':name,'workload':a.workload,'gpus':selected,'created_at_epoch':now,'expires_at_epoch':now+int(cfg['reservation_ttl_seconds']),'selection_path':str(Path(a.selection_receipt).resolve()),'launching':False,'reconstructed_from_immutable_selection':True}
        with LockedState(a.state_root,cfgsha,cfg['managed_gpus']) as state:
            assert_admission_open(state);state['reservations'][rid]=rec
        committed=True;return rec
    finally:
        if group is not None and not committed:
            cleaned=remove_pg_quiet(get_pg,remove_pg,name)
            if not cleaned:
                now=time.time();pending={'placement_group_name':name,'workload':a.workload,'gpus':selected,'created_at_epoch':now,'expires_at_epoch':now,'selection_path':str(Path(a.selection_receipt).resolve()),'launching':False,'reconstructed_from_immutable_selection':True,'cleanup_pending':True}
                with LockedState(a.state_root,cfgsha,cfg['managed_gpus']) as state:state.setdefault('reservations',{})[rid]=pending

def cmd_run(a, cfg):
    cfgsha = sha(a.config); registry=load_profile_registry(getattr(a,'profile_registry',None),cfgsha); p = profile(cfg, a.workload, registry); sel = load(a.selection_receipt) if a.selection_receipt else None
    if not sel or sel.get('schema') != 'openevo.ceiling1-ray-selection.202609042025' or sel.get('workload') != a.workload: raise RuntimeError('Ray run requires matching selection receipt')
    if sel.get('config_sha256') != cfgsha or list(map(int, sel['selected_gpus'])) != list(map(int, a.gpus)): raise RuntimeError('Ray selection/run identity drift')
    expected_registry_sha = registry['_sha256'] if registry is not None else None
    if sel.get('profile_registry_sha256') != expected_registry_sha: raise RuntimeError('Ray selection/profile-registry identity drift')
    ray = None; get_pg = remove_pg = None; lease_acquired = False
    try:
        ray, cluster, placement_group, get_pg, remove_pg, PGS = ray_api(cfg, a.state_root, a.config)
        reservation=ensure_selection_reservation(a,cfg,p,sel,ray,placement_group,get_pg,remove_pg)
        if reservation.get('placement_group_name') != sel['placement_group_name']: raise RuntimeError('Ray selection reservation drift before launch')
        with LockedState(a.state_root,cfgsha,cfg['managed_gpus']) as state:
            assert_admission_open(state);r=state['reservations'][sel['reservation_id']];r.update({'launching':True,'launching_at_epoch':time.time(),'lease_id':a.lease_id,'container':a.container,'gpus':list(map(int,a.gpus)),'vram_request_mib_per_gpu':int(p['vram_request_mib'])})
        get_pg(sel['placement_group_name']); live = live_gate(cfg, p, a.gpus)
        with LockedState(a.state_root, cfgsha, cfg['managed_gpus']) as s:
            assert_admission_open(s)
            if sel['reservation_id'] not in s.get('reservations', {}): raise RuntimeError('Ray reservation disappeared before launch')
        env = {**os.environ, 'CEILING1_LEASE_ID': a.lease_id, 'CEILING1_LEASE_WORKLOAD': a.workload, 'CEILING1_RAY_RESERVATION_ID': sel['reservation_id'], 'CEILING1_RAY_PLACEMENT_GROUP': sel['placement_group_name']}
        q = subprocess.run([str(a.launch_script)], env=env)
        if q.returncode: raise RuntimeError(f'science launch script failed rc={q.returncode}')
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline and not running(a.container): time.sleep(.05)
        if not running(a.container): raise RuntimeError('science container did not become running')
        with LockedState(a.state_root,cfgsha,cfg['managed_gpus']) as s:
            r=dict(s.get('reservations',{}).get(sel['reservation_id']) or {});lease=lease_from_launching_reservation(r,sel['reservation_id'],time.time(),recovered=False);s.get('reservations',{}).pop(sel['reservation_id'],None);install_lease_once(s,lease)
        lease_acquired = True
        launch_rows={int(k):v for k,v in live.items()}; launch_binding=binding_evidence(cfg,list(map(int,a.gpus)),launch_rows,time.time()); dump(Path(a.state_root) / 'receipts' / f'{a.lease_id}.launch.json', {'schema': 'openevo.ceiling1-ray-launch.202609042025', 'status': 'RUNNING', 'selection': sel, 'live_gate': live, 'launch_gpu_binding': launch_binding, 'lease': lease})
        watcher = [sys.executable, str(Path(__file__).resolve()), '--config', str(a.config), '--state-root', str(a.state_root), 'watch-release', '--lease-id', a.lease_id, '--container', a.container, '--placement-group-name', sel['placement_group_name']]
        subprocess.Popen(watcher, stdout=(Path(a.state_root) / 'watchers.log').open('a'), stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(lease, indent=2, sort_keys=True)); return 0
    except Exception as exc:
        # If a container is unexpectedly running, preserve its reservation/PG rather than lying about free capacity.
        container_running = running(a.container)
        if not lease_acquired and not container_running:
            cleaned = bool(get_pg is not None and remove_pg is not None and remove_pg_quiet(get_pg, remove_pg, sel['placement_group_name']))
            with LockedState(a.state_root, cfgsha, cfg['managed_gpus']) as s:
                if cleaned: s.get('reservations', {}).pop(sel['reservation_id'], None)
                elif sel['reservation_id'] in s.get('reservations', {}): s['reservations'][sel['reservation_id']]['cleanup_pending'] = True
        dump(Path(a.state_root) / 'receipts' / f'{a.lease_id}.failed.json', {'schema': 'openevo.ceiling1-ray-launch-failure.202609042025', 'status': 'FAILED_BEFORE_LEASE' if not container_running else 'CONTAINER_RUNNING_UNCONFIRMED', 'lease_id': a.lease_id, 'workload': a.workload, 'reservation_id': sel['reservation_id'], 'placement_group_name': sel['placement_group_name'], 'container': a.container, 'container_running': container_running, 'error': f'{type(exc).__name__}: {exc}', 'failed_at_epoch': time.time()})
        raise
    finally:
        if ray is not None: ray.shutdown()


def cmd_watch(a, cfg):
    while running(a.container): time.sleep(1)
    ray, cluster, pg, get_pg, remove_pg, PGS = ray_api(cfg, a.state_root, a.config)
    try:
        cleaned = remove_pg_quiet(get_pg, remove_pg, a.placement_group_name)
        with LockedState(a.state_root, sha(a.config), cfg['managed_gpus']) as s:
            for i in cfg['managed_gpus']:
                keep=[]
                for x in s['gpus'][str(i)].get('leases', []):
                    if x.get('lease_id') != a.lease_id: keep.append(x)
                    elif not cleaned: x['release_pending']=True; keep.append(x)
                s['gpus'][str(i)]['leases']=keep
        if not cleaned: raise RuntimeError('Ray placement group release pending after container exit')
    finally:
        ray.shutdown()
    return 0


def cmd_pause(a, cfg):
    cfgsha = sha(a.config); pending = []
    with LockedState(a.state_root, cfgsha, cfg['managed_gpus']) as s:
        s['admission'] = {'paused': True, 'reason': a.reason, 'updated_at_epoch': time.time()}
        for rid, r in list(s.get('reservations', {}).items()):
            if not r.get('launching'): pending.append((rid, dict(r)))
    removed = 0; warning = None
    try:
        ray, cluster, pg, get_pg, remove_pg, PGS = ray_api(cfg, a.state_root, a.config)
        try:
            cleaned=[]
            for rid, r in pending:
                if remove_pg_quiet(get_pg, remove_pg, r['placement_group_name']): cleaned.append(rid); removed += 1
            with LockedState(a.state_root, cfgsha, cfg['managed_gpus']) as s:
                for rid in cleaned: s.get('reservations', {}).pop(rid, None)
                for rid, _ in pending:
                    if rid in s.get('reservations', {}): s['reservations'][rid]['cleanup_pending'] = True
        finally: ray.shutdown()
    except Exception as exc:
        warning = f'{type(exc).__name__}: {exc}'
        with LockedState(a.state_root, cfgsha, cfg['managed_gpus']) as s:
            for rid, _ in pending:
                if rid in s.get('reservations', {}): s['reservations'][rid]['cleanup_pending'] = True
    print(json.dumps({'status': 'RAY_ADMISSION_PAUSED', 'reason': a.reason, 'pending_reservations_seen': len(pending), 'placement_groups_removed': removed, 'cleanup_pending': len(pending)-removed, 'pool_warning': warning}, sort_keys=True)); return 0

def cmd_resume(a, cfg):
    pool_liveness(cfg, a.state_root, a.config)
    with LockedState(a.state_root, sha(a.config), cfg['managed_gpus']) as s:
        s['admission'] = {'paused': False, 'reason': None, 'updated_at_epoch': time.time()}
    print(json.dumps({'status': 'RAY_ADMISSION_OPEN'}, sort_keys=True)); return 0


def cmd_drain_status(a, cfg):
    p = Path(a.state_root) / 'state.json'
    state = load(p) if p.is_file() else {'gpus': {}, 'reservations': {}, 'admission': {'paused': False}}
    leases = []
    for i, g in (state.get('gpus') or {}).items():
        for lease in g.get('leases', []):
            if lease.get('lease_id') not in {x.get('lease_id') for x in leases}: leases.append(lease)
    out = {'status': 'DRAINED' if not leases and not state.get('reservations') else 'DRAINING', 'admission': state.get('admission'), 'live_lease_count': sum(1 for x in leases if running(x.get('container', ''))), 'recorded_lease_count': len(leases), 'reservation_count': len(state.get('reservations') or {}), 'leases': leases}
    print(json.dumps(out, indent=2, sort_keys=True)); return 0


def main():
    global args_global
    ap = argparse.ArgumentParser(); ap.add_argument('--config', type=Path, required=True); ap.add_argument('--state-root', type=Path, required=True); ap.add_argument('--profile-registry', type=Path); sp = ap.add_subparsers(dest='cmd', required=True)
    sp.add_parser('validate'); sp.add_parser('pool-liveness'); sp.add_parser('status'); sp.add_parser('reconcile'); sp.add_parser('drain-status')
    q = sp.add_parser('pause-admission'); q.add_argument('--reason', required=True)
    sp.add_parser('resume-admission')
    q = sp.add_parser('select'); q.add_argument('--workload', required=True); q.add_argument('--count', type=int); q.add_argument('--exact-count', action='store_true'); q.add_argument('--output', type=Path, required=True)
    q = sp.add_parser('run'); q.add_argument('--lease-id', required=True); q.add_argument('--workload', required=True); q.add_argument('--container', required=True); q.add_argument('--launch-script', type=Path, required=True); q.add_argument('--gpus', nargs='+', type=int, required=True); q.add_argument('--vram-qualification', type=Path); q.add_argument('--selection-receipt', type=Path, required=True)
    q = sp.add_parser('watch-release'); q.add_argument('--lease-id', required=True); q.add_argument('--container', required=True); q.add_argument('--placement-group-name', required=True)
    args_global = ap.parse_args(); cfg = load(args_global.config)
    if cfg.get('schema') != 'openevo.ceiling1-ray-placement-backend.202609042025': raise RuntimeError('config schema drift')
    binding_contract(cfg)
    registry = load_profile_registry(args_global.profile_registry, sha(args_global.config))
    if args_global.cmd == 'validate':
        print('CEILING1_RAY_PLACEMENT_CONFIG=PASS')
        if registry is not None: print('RAY_PROFILE_REGISTRY=PASS')
        return 0
    if args_global.cmd == 'pool-liveness': pool_liveness(cfg,args_global.state_root,args_global.config); print('CEILING1_RAY_POOL=LIVE'); return 0
    if args_global.cmd == 'status': reconcile(cfg, args_global.state_root, args_global.config); print(json.dumps(load(Path(args_global.state_root) / 'state.json'), indent=2, sort_keys=True)); return 0
    if args_global.cmd == 'reconcile': reconcile(cfg, args_global.state_root, args_global.config); return 0
    if args_global.cmd == 'drain-status': return cmd_drain_status(args_global, cfg)
    if args_global.cmd == 'pause-admission': return cmd_pause(args_global, cfg)
    if args_global.cmd == 'resume-admission': return cmd_resume(args_global, cfg)
    if args_global.cmd == 'select': return cmd_select(args_global, cfg)
    if args_global.cmd == 'run': return cmd_run(args_global, cfg)
    if args_global.cmd == 'watch-release': return cmd_watch(args_global, cfg)
if __name__ == '__main__': raise SystemExit(main())
