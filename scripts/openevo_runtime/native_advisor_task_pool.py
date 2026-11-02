"""Durable, at-most-once dispatcher for already authorized exact launchers.

This does not create experiments or relax their admission gates. A launcher still
owns Formal Start, frozen science, storage, live GPU ownership and phase locks.

This is a reviewed *replacement candidate* for the existing Q04 writer, never a
second service: only deploy by re-binding the frozen source and stopping the old
writer under its existing lock with an authorized continuity handoff.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

GPU_UUIDS = (
    'GPU-00000000-0000-0000-0000-000000000000',
    'GPU-00000000-0000-0000-0000-000000000000',
    'GPU-00000000-0000-0000-0000-000000000000',
    'GPU-00000000-0000-0000-0000-000000000000',
)
RETIRED_GPU_UUIDS = ('GPU-00000000-0000-0000-0000-000000000000',)
# 2026-10-28 explicit owner withdrawal. This bound is independent of caller
# flags: an accidental restart without --exclude must never re-admit GPU1.
# Changing it requires new explicit owner authorization and reviewed cutover.
CURRENT_OWNER_WITHDRAWN_UUIDS = frozenset({'GPU-00000000-0000-0000-0000-000000000000'})
# Exact reviewed #746 source at 5a174ac50; repin only with reviewed source migration.
PROJECTION_SHA256 = '394e86431ad3bd2b0dcf47d5946b8d569cce0a89cd2cbcee18270738efbd0002'


def read(p):
    return json.loads(Path(p).read_text())


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def atomic(p, value):
    temporary = p.with_suffix('.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(p)


def processes():
    # Snapshot numeric PIDs without probing each disappearing /proc/PID path
    # during glob enumeration. A failed top-level snapshot is NOT an idle GPU.
    result = []
    try:
        with os.scandir('/proc') as entries:
            pids = [entry.name for entry in entries if entry.name.isdecimal()]
    except OSError as exc:
        raise RuntimeError('PROCESS_INVENTORY_UNAVAILABLE') from exc
    for pid in pids:
        try:
            argv = (Path('/proc') / pid / 'cmdline').read_bytes().decode().strip('\0').split('\0')
            result.append((int(pid), argv))
        except (OSError, UnicodeError):
            # A PID can vanish after readdir; its cmdline was never observed.
            pass
    return result


def running(task, live):
    # Exact argv tokens, not substring/pgrep matches or stale PID reuse.
    match = task.get('worker_tokens', [])
    return [pid for pid, argv in live if match and all(t in argv for t in match)]


def validate_manifest(m):
    assert m['schema'] == 'openevo.advisor-task-pool.v1'
    assert m['authorized_gpu_uuids'] == list(GPU_UUIDS)
    ids = [t['id'] for t in m['tasks']]
    assert len(ids) == len(set(ids))
    for t in m['tasks']:
        assert t['gpu_uuid'] in (GPU_UUIDS if t['mode'] == 'queued' else GPU_UUIDS + RETIRED_GPU_UUIDS)
        assert t['mode'] in ('adopt', 'queued', 'hold')
        assert set(t.get('depends_on', [])) <= set(ids)
        assert t['id'] not in t.get('depends_on', [])
        if t['mode'] == 'queued':
            assert t['argv'] and t['source_files'] and t['worker_tokens']
            assert t['peak_mib'] > 0 and t['headroom_mib'] >= 4096
            assert t['output_root'] and t['completion']
    return m


def snapshot_gpu():
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=uuid,memory.total,memory.free',
        '--format=csv,noheader,nounits'], text=True)
    rows = {}
    for line in output.splitlines():
        uuid, total, free = [x.strip() for x in line.split(',')]
        if uuid in GPU_UUIDS:
            rows[uuid] = {'total': int(total), 'free': int(free)}
    assert set(rows) == set(GPU_UUIDS)
    return rows


def verified_owner_admission(snapshot, gpu_uuid, *, now=None):
    """Consume ZJU #216 observation at dispatch; it never grants science permission.

    Only a verified, physically idle and ownerless card is eligible here.
    Same-owner co-residency and Holder hand-back require a separately qualified
    authority and are intentionally not inferred from free VRAM.
    """
    if (not isinstance(snapshot, dict)
            or snapshot.get('schema') != 'zju.gpu-ownership-snapshot.v1'
            or snapshot.get('mutation') is not False
            or snapshot.get('execution_authority') is not False
            or snapshot.get('gpu_attribution_source') != 'verified-observer'
            or snapshot.get('gpu_attribution_error') is not None):
        raise ValueError('GPU_OWNER_OBSERVER_NOT_VERIFIED')
    unresolved = snapshot.get('unresolved_global_gpu_uuids')
    if not isinstance(unresolved, list) or any(not isinstance(v, str) for v in unresolved):
        raise ValueError('GPU_OWNER_UNRESOLVED_SET_INVALID')
    observed_at = snapshot.get('observed_at_utc')
    if not isinstance(observed_at, str):
        raise ValueError('GPU_OWNER_TIMESTAMP_MISSING')
    try:
        observed = datetime.fromisoformat(observed_at.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError('GPU_OWNER_TIMESTAMP_INVALID') from exc
    if observed.tzinfo is None:
        raise ValueError('GPU_OWNER_TIMESTAMP_NAIVE')
    now = now or datetime.now(timezone.utc)
    age_seconds = (now - observed).total_seconds()
    if not -2 <= age_seconds <= 10:
        raise ValueError('GPU_OWNER_OBSERVATION_STALE')
    rows = snapshot.get('gpus')
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError('GPU_OWNER_INVENTORY_INCOMPLETE')
    ids = [r.get('uuid') for r in rows if isinstance(r, dict)]
    indices = [r.get('index') for r in rows if isinstance(r, dict)]
    if (len(ids) != 8 or any(not isinstance(v, str) or not v.startswith('GPU-') for v in ids)
            or len(set(ids)) != 8 or len(set(indices)) != 8
            or any(type(i) is not int or not 0 <= i < 8 for i in indices)):
        raise ValueError('GPU_OWNER_UUID_INDEX_COLLISION')
    matching = [r for r in rows if r['uuid'] == gpu_uuid]
    if len(matching) != 1:
        raise ValueError('GPU_OWNER_REQUESTED_UUID_NOT_FOUND')
    row = matching[0]
    if gpu_uuid in unresolved or (row.get('attribution') != 'IDLE'
            or row.get('owner') != 'idle' or row.get('owners') != []):
        raise ValueError('GPU_OWNER_NOT_EXCLUSIVELY_IDLE')
    used, total, util = (row.get('memory_used_mib'), row.get('memory_total_mib'),
                         row.get('utilization_gpu_percent'))
    if (any(type(v) is not int for v in (used, total, util))
            or not 0 <= used <= 16 or total <= used or util != 0):
        raise ValueError('GPU_OWNER_IDLE_TELEMETRY_MISMATCH')
    return {'gpu_uuid': gpu_uuid, 'gpu_index': row['index'], 'observed_at_utc': observed_at,
            'free_mib': total - used,
            'snapshot_sha256': hashlib.sha256(json.dumps(
                snapshot, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}


def snapshot_verified_owner(gpu_uuid):
    """One bounded read through the existing host observer, not a new daemon."""
    proc = subprocess.run(['gpu-who', '--json'], capture_output=True, text=True,
                          timeout=20, check=False,
                          env={**os.environ, 'GPU_WHO_REQUIRE_OBSERVER': '1'})
    if proc.returncode != 0 or len(proc.stdout) > 1024 * 1024:
        raise ValueError('GPU_OWNER_OBSERVER_FAILED')
    try:
        return verified_owner_admission(json.loads(proc.stdout), gpu_uuid)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError('GPU_OWNER_JSON_INVALID') from exc


def readiness_stages_pass(readiness, source_files):
    """Accept native exact-cache reuse only with its pinned PASS receipt."""
    stages = readiness.get('stages')
    if not isinstance(stages, list) or not stages:
        return False
    for stage in stages:
        if not isinstance(stage, dict):
            return False
        if stage.get('verdict') == 'PASS':
            continue
        if (stage.get('stage') != 'runtime-qualification'
                or stage.get('verdict') != 'CACHE_HIT'):
            return False
        try:
            path = Path(readiness['qualification_receipt_path'])
            expected = readiness['qualification_receipt_sha256']
            key = readiness['qualification_cache_key']
            fingerprint = readiness['runtime_fingerprint_sha256']
            detail = stage['detail']
            decision = readiness['runtime_qualification_decision']
            if (not path.is_absolute() or path.is_symlink() or not path.is_file()
                    or not isinstance(expected, str)
                    or not expected.startswith('sha256:')
                    or len(expected) != 71
                    or any(c not in '0123456789abcdef' for c in expected[7:])
                    or source_files.get(str(path)) != expected[7:]
                    or not isinstance(key, str) or len(key) != 64
                    or any(c not in '0123456789abcdef' for c in key)
                    or not isinstance(fingerprint, str)
                    or not fingerprint.startswith('sha256:') or len(fingerprint) != 71
                    or any(c not in '0123456789abcdef' for c in fingerprint[7:])
                    or not isinstance(decision, dict)
                    or decision.get('reason') != 'exact_request_match'
                    or decision.get('changed_request_fields') != []
                    or decision.get('environment_mutation_allowed') is not False
                    or detail.get('qualification_decision') != decision
                    or detail.get('cache_key') != key[:16]
                    or detail.get('fingerprint') != fingerprint[:23]):
                return False
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected[7:]:
                return False
            cached = json.loads(payload)
            request = cached['request']
            request_key = hashlib.sha256(json.dumps(
                request, ensure_ascii=False, sort_keys=True,
                separators=(',', ':')).encode('utf-8')).hexdigest()
            if (cached.get('schema') != 'openevo.runtime-qualification.v1'
                    or cached.get('status') != 'PASS' or cached.get('failures') != []
                    or cached.get('formal_task_consumption_allowed') is not False
                    or cached.get('cache_key') != key or request_key != key
                    or request.get('runtime_fingerprint_sha256') != fingerprint):
                return False
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return False
    return True


def _common_start_dev64_admission(task, cfg, activation, endpoint, panel, profile, proof, same_ref, ref):
    """New owner-authorized64; never reinterpret historical failure as zero recovery."""
    pins = {
        'state_id': 'e5a3480ec014f230de46095b8092813730e4b7e74bc3b5e83499530c33d94770',
        'adapter_sha256': 'c8ef26c3c96515495703d97148c1f4845c09814b686987f7150022099c67fd11',
        'memory_sha256': '9a88356c084783e003d80db90a918be265d1ce481590d769453993d0e21dea6d',
        'skill_sha256': 'cef31b897dbe4796d61710c13ab11630f64f043bd7df651d1faa12980163b21a',
        'agent_system_sha256': '7a87fa050c526907b1999ba4c0ef9c127808e4983a1f36dc646a12c6ab75caf4',
        'panel_sha256': '269a07c8f3f4700df8a6d139894c867d4c4ecfdba2e1d957a380d836244448b6',
    }
    source = '2f4adc21ed7a3de31b5de390d6487a38107badb3'
    if (task['gpu_uuid'] != GPU_UUIDS[3] or any(endpoint.get(k) != v for k, v in pins.items())
            or cfg.get('source_sha') != source
            or activation.get('source', {}).get('head') != source
            or profile.get('source', {}).get('experiment', {}).get('revision') != source):
        raise ValueError('Q09_COMMON_START_SCIENTIFIC_IDENTITY_DRIFT')
    if (activation.get('status') != 'CPU_PREPARED_NOT_LAUNCHED'
            or activation.get('stage') != 'common-start'
            or activation.get('confirmation') is not False
            or task.get('holders') not in (None, [])
            or task.get('native_campaign', {}).get('reservation') != {'enabled': False, 'expires_at': None}
            or 'zero_consumption_recovery' in cfg):
        raise ValueError('Q09_COMMON_START_ACTIVATION_SCOPE_DRIFT')
    if (proof.get('schema') != 'openevo.q09-common-start-activation.v1'
            or proof.get('status') != 'AUTHORIZED_NEW_COMMON_START_DEVELOPMENT_64'
            or proof.get('endpoint_id') != endpoint['endpoint_id']
            or proof.get('budget') != activation['budget']
            or proof.get('prior_baseline_disposition') != 'NOT_EXACT_REUSE_DUE_TO_CARRIERS'
            or proof.get('science_output') != cfg['science_output']
            or proof.get('output_absent_verified') is not True
            or type(proof.get('formal_starts_verified_prior')) is not int
            or proof['formal_starts_verified_prior'] != 0
            or not isinstance(proof.get('user_grant'), dict)
            or not isinstance(proof['user_grant'].get('quote'), str)
            or not proof['user_grant']['quote'].strip()
            or proof['user_grant'].get('date') != '2026-10-28'):
        raise ValueError('Q09_COMMON_START_NEW_AUTHORIZATION_MISSING')
    baseline = ref(proof.get('prior_baseline_audit'))
    if baseline.get('status') != 'NOT_EXACT_REUSE_DUE_TO_DIFFERENT_CARRIERS':
        raise ValueError('Q09_COMMON_START_PRIOR_BASELINE_AUDIT_DRIFT')
    for name in ('endpoint', 'activation', 'profile', 'panel'):
        same_ref(proof.get(name), cfg[name])
    source_ref = proof.get('source') or {}
    if (source_ref.get('revision') != source
            or source_ref.get('path') != profile.get('source', {}).get('experiment', {}).get('path')):
        raise ValueError('Q09_COMMON_START_SOURCE_REFERENCE_DRIFT')
    inputs = activation.get('input_files') or {}
    anchors = [Path(k) for k in inputs if k.endswith('/metadata/endpoint.json')]
    if len(anchors) != 1:
        raise ValueError('Q09_COMMON_START_STAGING_ROOT_AMBIGUOUS')
    stage = anchors[0].parent.parent
    if (inputs[str(anchors[0])] != digest(cfg['endpoint'])
            or inputs.get(str(stage/'metadata/development-panel.json')) != digest(cfg['panel'])):
        raise ValueError('Q09_COMMON_START_STAGED_METADATA_DRIFT')
    expected = Path(profile['workspace']) / profile['paths']['runs'] / profile['run_id'] / 'endpoint'
    output = Path(cfg['science_output'])
    control = Path(cfg['dispatch_dir'])
    if output != expected:
        raise ValueError('Q09_COMMON_START_OUTPUT_PROFILE_MISMATCH')
    if (not control.is_absolute() or any(p.is_symlink() for p in (control, *control.parents))
            or any(control.rglob('formal-start*.json'))):
        raise ValueError('Q09_COMMON_START_FORMAL_START_OR_CONTROL_DRIFT')
    if (not output.is_absolute() or any(p.is_symlink() for p in (output, *output.parents))
            or output.exists()):
        raise ValueError('Q09_COMMON_START_OUTPUT_ALREADY_PRESENT')
    prepared = proof.get('prepared_empty_run_root')
    if output.parent.exists():
        parent = output.parent
        if not isinstance(prepared, dict) or not parent.is_dir() or any(parent.iterdir()):
            raise ValueError('Q09_COMMON_START_OUTPUT_ALREADY_PRESENT')
        observed = parent.stat()
        identity = {'path': str(parent), 'device': observed.st_dev, 'inode': observed.st_ino,
                    'uid': observed.st_uid, 'gid': observed.st_gid, 'mode': observed.st_mode & 0o7777}
        if (prepared != identity or identity['uid'] != 1001 or identity['gid'] != 1001
                or identity['mode'] != 0o750):
            raise ValueError('Q09_COMMON_START_EMPTY_ROOT_IDENTITY_DRIFT')
    elif prepared is not None:
        raise ValueError('Q09_COMMON_START_EMPTY_ROOT_IDENTITY_DRIFT')


Q09_FRESH_PAIR_PROOF_SHA256 = '67f031c0c4674df63f51e74b406aeca2fa679d562885cbe4f8378e6a9e3ee850'
Q09_FRESH_PAIR_SOURCE = '160db1c85c6ccf247ceb69239d4db1eb61e95c22'
Q09_FRESH_PAIR_PANEL = 'bcb91ad087ba166b04b75baab1c99f3bf44fb45565aea8ad1b97a33c945ed109'
Q09_FRESH_PAIR_ENDPOINTS = {
    'r119': ('q09-fixed-common-start-fresh64-202610282120', 'common-start-confirmation',
             'SHARED_START', 'shared-common-start'),
    'r159': ('q09-fixed-s1-r159-fresh64-202610282120', 'confirmation-40',
             'ACTIVE', 'q09-stream-1'),
}


def _fresh_pair64_admission(task, cfg, activation, endpoint, panel, profile, proof, same_ref, ref):
    """Admit only the two reviewed members of one prospective 128-evaluation study."""
    arm = cfg.get('fresh_endpoint')
    if (arm not in Q09_FRESH_PAIR_ENDPOINTS
            or digest(cfg['fresh_activation_proof']) != Q09_FRESH_PAIR_PROOF_SHA256
            or task.get('id') != 'q09-fixed-checkpoint-fresh-' + arm + '-gpu4-202610282120'
            or task.get('gpu_uuid') != GPU_UUIDS[3]
            or cfg.get('source_sha') != Q09_FRESH_PAIR_SOURCE
            or activation.get('source', {}).get('head') != Q09_FRESH_PAIR_SOURCE
            or profile.get('source', {}).get('experiment', {}).get('revision') != Q09_FRESH_PAIR_SOURCE):
        raise ValueError('Q09_FRESH_PAIR_EXACT_SCOPE_DRIFT')
    if (time.time() >= 1793235600
            or task.get('holders') not in (None, [])
            or task.get('native_campaign', {}).get('reservation') != {'enabled': False, 'expires_at': None}
            or 'zero_consumption_recovery' in cfg
            or activation.get('confirmation') is not True
            or activation.get('status') != 'CPU_PREPARED_NOT_LAUNCHED'):
        raise ValueError('Q09_FRESH_PAIR_ACTIVATION_SCOPE_DRIFT')
    if (proof.get('schema') != 'openevo.q09-fresh-pair-activation.v1'
            or proof.get('status') != 'CPU_PREPARED_NOT_ACTIVATED'
            or proof.get('formal_task_consumption_allowed') is not False
            or proof.get('formal_starts_verified_prior') != 0
            or proof.get('output_absent_verified') is not True
            or proof.get('gpu_uuids') != [GPU_UUIDS[3]]
            or proof.get('budget') != {'new_evaluation_episodes': 128, 'teacher_calls': 0,
                'final_panel_access': 0, 'optimizer_updates': 0, 'adaptive_replays': 0, 'training_rollouts': 0}):
        raise ValueError('Q09_FRESH_PAIR_CONTRACT_DRIFT')
    contract = ref(proof['contract'])
    ref(proof['exclusion_manifest'])
    same_ref(proof['confirmation_panel'], cfg['panel'])
    if (contract.get('status') != 'PROSPECTIVE_FROZEN_BEFORE_FRESH_COLLECTION'
            or contract.get('budget') != proof['budget']
            or contract.get('endpoints') != proof['endpoints']
            or contract.get('confirmation_panel') != proof['confirmation_panel']
            or contract.get('exposure') != {'checkpoint_pair_selected_after_development': True,
                'development_result_seen_before_design': True, 'fresh_outcomes_seen': False}
            or contract.get('analysis') != proof['analysis']
            or proof['analysis'].get('comparisons') != 1
            or proof['analysis'].get('bootstrap_resamples') != 50000
            or proof['analysis'].get('random_seed') != 2026102821):
        raise ValueError('Q09_FRESH_PAIR_PROSPECTIVE_PLAN_DRIFT')
    member = proof['endpoints'][arm]
    for name in ('endpoint', 'activation', 'profile'):
        same_ref(member[name], cfg[name])
    if (endpoint.get('state_id') != member['state_id']
            or endpoint.get('adapter_sha256') != member['adapter_sha256']
            or endpoint.get('panel_sha256') != Q09_FRESH_PAIR_PANEL
            or any(endpoint.get(k) != v for k, v in {
                'memory_sha256': '9a88356c084783e003d80db90a918be265d1ce481590d769453993d0e21dea6d',
                'skill_sha256': 'cef31b897dbe4796d61710c13ab11630f64f043bd7df651d1faa12980163b21a',
                'agent_system_sha256': '7a87fa050c526907b1999ba4c0ef9c127808e4983a1f36dc646a12c6ab75caf4',
            }.items())
            or cfg['dispatch_dir'] != member['dispatch_dir']
            or cfg['science_output'] != member['science_output']):
        raise ValueError('Q09_FRESH_PAIR_INPUT_IDENTITY_DRIFT')
    inputs = activation.get('input_files') or {}
    anchors = [Path(k) for k in inputs if k.endswith('/metadata/endpoint.json')]
    if (len(anchors) != 1 or inputs[str(anchors[0])] != digest(cfg['endpoint'])
            or inputs.get(str(anchors[0].parent/'development-panel.json')) != digest(cfg['panel'])):
        raise ValueError('Q09_FRESH_PAIR_STAGED_METADATA_DRIFT')
    output, control = Path(cfg['science_output']), Path(cfg['dispatch_dir'])
    expected = Path(profile['workspace']) / profile['paths']['runs'] / profile['run_id'] / 'endpoint'
    if (output != expected or not output.is_absolute() or not control.is_absolute()
            or any(p.is_symlink() for p in (output, *output.parents, control, *control.parents))
            or output.exists() or any(control.rglob('formal-start*.json'))):
        raise ValueError('Q09_FRESH_PAIR_PRIOR_START_OR_OUTPUT')
    parent = output.parent
    prepared = member['prepared_empty_run_root']
    if not parent.is_dir() or any(parent.iterdir()):
        raise ValueError('Q09_FRESH_PAIR_EMPTY_ROOT_REQUIRED')
    st = parent.stat()
    actual = {'path': str(parent), 'device': st.st_dev, 'inode': st.st_ino,
              'uid': st.st_uid, 'gid': st.st_gid, 'mode': st.st_mode & 0o7777}
    if prepared != actual or actual['uid'] != 1001 or actual['gid'] != 1001 or actual['mode'] != 0o750:
        raise ValueError('Q09_FRESH_PAIR_EMPTY_ROOT_IDENTITY_DRIFT')


def native_eval_science_admission(task):
    """Admit one reviewed, still-unspent Q09 Stage-A endpoint, never training.

    This only checks CPU evidence. The existing runner remains responsible for
    model bytes, native Formal Start, live ownership, locks and completion.
    """
    cfg = task.get('native_eval')
    if (not isinstance(cfg, dict) or cfg.get('adapter') not in {'q09-stage-a-dev64-v1', 'q09-stage-a-common-start-dev64-v1', 'q09-fixed-checkpoint-fresh64-v1'}
            or not task.get('id', '').startswith('q09-')):
        raise ValueError('Q09_EVAL_ADAPTER_UNRECOGNIZED')
    pins = task.get('source_files') or {}
    authority = Path(cfg['authority'])

    def bound(path, *, json_value=True):
        path = Path(path)
        if (not path.is_absolute() or path.is_symlink() or not path.is_file()
                or str(path) not in pins or digest(path) != pins[str(path)]):
            raise ValueError('Q09_EVAL_SOURCE_UNBOUND_OR_DRIFTED')
        return read(path) if json_value else path.read_bytes()

    def ref(value, *, json_value=True):
        if not isinstance(value, dict):
            raise ValueError('Q09_EVAL_REFERENCE_MISSING')
        path = Path(value['path'])
        if not path.is_absolute():
            path = authority / path
        result = bound(path, json_value=json_value)
        if digest(path) != value['sha256'].removeprefix('sha256:'):
            raise ValueError('Q09_EVAL_REFERENCE_DRIFTED')
        return result

    def same_ref(value, path):
        target = Path(value['path'])
        if not target.is_absolute():
            target = authority / target
        if target != Path(path):
            raise ValueError('Q09_EVAL_REFERENCE_TARGET_MISMATCH')
        return ref(value)

    ready = bound(cfg['readiness_receipt'])
    release = bound(cfg['release'])
    router = bound(cfg['current_campaign'])
    profile = bound(cfg['profile'])
    activation = bound(cfg['activation'])
    endpoint = bound(cfg['endpoint'])
    panel = bound(cfg['panel'])
    common_start = cfg['adapter'] == 'q09-stage-a-common-start-dev64-v1'
    fresh_pair = cfg['adapter'] == 'q09-fixed-checkpoint-fresh64-v1'
    recovery = bound(cfg['fresh_activation_proof'] if common_start or fresh_pair else cfg['zero_consumption_recovery'])
    if (ready.get('schema') != 'openevo.execution-readiness.v1'
            or ready.get('verdict') != 'READY'
            or ready.get('formal_task_consumption_allowed') is not True
            or ready.get('authority', {}).get('router_status') != 'launch-authorized'
            or not readiness_stages_pass(ready, pins)):
        raise ValueError('Q09_EVAL_READINESS_NOT_PASS')
    same_ref(ready['release_receipt'], cfg['release'])
    same_ref(ready['profile'], cfg['profile'])
    if (router.get('status') != 'launch-authorized'
            or router.get('formal_task_consumption_allowed') is not True
            or release.get('schema') != 'openevo.experiment-release.v1'
            or release.get('status') != 'launch-authorized'
            or release.get('formal_task_consumption_allowed') is not True
            or router.get('campaign_id') != release.get('campaign_id')):
        raise ValueError('Q09_EVAL_RELEASE_NOT_AUTHORIZED')
    same_ref({'path': router['release_receipt'],
              'sha256': router['release_receipt_sha256']}, cfg['release'])
    same_ref(release['execution_readiness']['profile'], cfg['profile'])
    artifacts = {x['role']: x for x in release['execution_readiness']['frozen_scientific_artifacts']}
    required_roles = {'endpoint', 'confirmation_panel' if fresh_pair else 'development_panel', 'activation'}
    if fresh_pair:
        required_roles.add('fresh_activation')
    if not required_roles.issubset(artifacts):
        raise ValueError('Q09_EVAL_FROZEN_ARTIFACT_MISSING')
    for role, name in [('endpoint', 'endpoint'), ('confirmation_panel' if fresh_pair else 'development_panel', 'panel'), ('activation', 'activation')]:
        same_ref(artifacts[role], cfg[name])
    if (release.get('authorized_gpu_uuids') != [task['gpu_uuid']]
            or endpoint.get('gpu_uuid') != task['gpu_uuid']):
        raise ValueError('Q09_EVAL_GPU_SCOPE_MISMATCH')
    if (any(type(release.get(key)) is not int or release[key] != limit for key, limit in
            [('evaluation_episode_limit', 64), ('rollout_limit', 64), ('training_rollout_limit', 0),
             ('external_teacher_call_limit', 0), ('final_panel_access_limit', 0)])
            or activation.get('budget') != {'new_evaluation_episodes': 64, 'teacher_calls': 0,
               'final_panel_access': 0, 'optimizer_updates': 0, 'adaptive_replays': 0,
               **({'training_rollouts': 0} if fresh_pair else {})}
            or endpoint.get('evaluation_contract', {}).get('episodes') != 64
            or endpoint['evaluation_contract'].get('automatic_retries') != 0):
        raise ValueError('Q09_EVAL_BUDGET_DRIFT')
    fresh_identity = Q09_FRESH_PAIR_ENDPOINTS.get(cfg.get('fresh_endpoint'), ('', '', '', ''))
    endpoint_id = (fresh_identity[0] if fresh_pair else
                   'q09-a-bounded-shared-start-development' if common_start
                   else 'q09-a-bounded-active-stream1-development-40')
    if (endpoint.get('schema') != 'openevo.advisor.q09.endpoint-config.v1'
            or endpoint.get('endpoint_id') != endpoint_id
            or activation.get('endpoint_id') != endpoint_id
            or profile.get('run_id') != endpoint_id
            or (endpoint.get('evaluation_stage'), endpoint.get('treatment'), endpoint.get('stream_id'),
                endpoint.get('backbone')) != ((*fresh_identity[1:], 'SIMPLE_BOUNDED') if fresh_pair else ('common-start', 'SHARED_START', 'shared-common-start', 'SIMPLE_BOUNDED') if common_start
                    else ('development-40', 'ACTIVE', 'q09-stream-1', 'SIMPLE_BOUNDED'))
            or activation.get('endpoint_sha256') != digest(cfg['endpoint'])
            or activation.get('profile_sha256') != digest(cfg['profile'])):
        raise ValueError('Q09_EVAL_ENDPOINT_SCOPE_DRIFT')
    canonical = dict(panel)
    panel_sha = canonical.pop('panel_sha256', None)
    if (panel.get('status') != ('FROZEN_PREOUTCOME' if fresh_pair else 'FROZEN_EXISTING_DEVELOPMENT_PANEL')
            or hashlib.sha256(json.dumps(canonical, ensure_ascii=False, sort_keys=True,
                 separators=(',', ':')).encode()).hexdigest() != panel_sha
            or endpoint.get('panel_sha256') != panel_sha
            or [x.get('probe_ordinal') for x in panel.get('tasks', [])] != list(range(64))
            or len({x.get('task_identity') for x in panel['tasks']}) != 64):
        raise ValueError('Q09_EVAL_PANEL_OR_SEEDS_DRIFT')
    if fresh_pair:
        same_ref(artifacts['fresh_activation'], cfg['fresh_activation_proof'])
        return _fresh_pair64_admission(task, cfg, activation, endpoint, panel, profile, recovery, same_ref, ref)
    if common_start:
        return _common_start_dev64_admission(task, cfg, activation, endpoint, panel, profile, recovery, same_ref, ref)
    if (recovery.get('schema') != 'openevo.q09-eval-zero-consumption-recovery.v1'
            or recovery.get('status') != 'VERIFIED_ZERO_CONSUMPTION_RESOURCE_FAILURE'
            or recovery.get('endpoint_id') != endpoint_id
            or recovery.get('new_science_output') != cfg['science_output']
            or any(type(recovery.get(k)) is not int or recovery[k] != 0
                   for k in ('formal_starts', 'evaluation_episodes', 'model_calls'))):
        raise ValueError('Q09_EVAL_ZERO_CONSUMPTION_PROOF_MISSING')
    original_endpoint = ref(recovery['original_endpoint'])
    original_activation = ref(recovery['original_activation'])
    original_profile = ref(recovery['original_profile'])
    if (original_activation.get('endpoint_sha256') != recovery['original_endpoint']['sha256'].removeprefix('sha256:')
            or original_activation.get('profile_sha256') != recovery['original_profile']['sha256'].removeprefix('sha256:')):
        raise ValueError('Q09_EVAL_ORIGINAL_AUTHORITY_CHAIN_MISMATCH')
    if ({k: v for k, v in endpoint.items() if k != 'gpu_uuid'} !=
            {k: v for k, v in original_endpoint.items() if k != 'gpu_uuid'}):
        raise ValueError('Q09_EVAL_RESOURCE_SUCCESSOR_CHANGED_SCIENCE')
    for item in (profile, original_profile):
        if not isinstance(item.get('paths'), dict):
            raise ValueError('Q09_EVAL_PROFILE_PATHS_MISSING')
    without_resource = lambda x: {**x, 'paths': {k: v for k, v in x['paths'].items()
                                               if k not in {'models', 'runs', 'cache'}}}
    if without_resource(profile) != without_resource(original_profile):
        raise ValueError('Q09_EVAL_PROFILE_CHANGED_PRODUCER')
    allowed = {'endpoint_sha256', 'profile_sha256', 'input_files', 'original_input_files', 'resource_amendment'}
    if ({k: v for k, v in activation.items() if k not in allowed} !=
            {k: v for k, v in original_activation.items() if k not in allowed}):
        raise ValueError('Q09_EVAL_ACTIVATION_CHANGED_SCIENCE')
    staged = []
    for value in (original_activation, activation):
        inputs = value['input_files']
        anchors = [Path(k) for k in inputs if k.endswith('/metadata/endpoint.json')]
        if len(anchors) != 1:
            raise ValueError('Q09_EVAL_STAGING_ROOT_AMBIGUOUS')
        stage = anchors[0].parent.parent
        mapped = {str(Path(k).relative_to(stage)): v for k, v in inputs.items()}
        if mapped.pop('metadata/endpoint.json') != value['endpoint_sha256']:
            raise ValueError('Q09_EVAL_STAGED_ENDPOINT_DRIFT')
        staged.append(mapped)
    if staged[1].get('metadata/development-panel.json') != digest(cfg['panel']):
        raise ValueError('Q09_EVAL_STAGED_PANEL_DRIFT')
    if staged[0] != staged[1]:
        raise ValueError('Q09_EVAL_STAGED_INPUTS_CHANGED')
    originals = []
    for value in (original_activation, activation):
        inputs = dict(value['original_input_files'])
        endpoint_keys = [k for k in inputs if k.endswith('/active-s1-r159.endpoint.json')]
        if len(endpoint_keys) != 1 or inputs.pop(endpoint_keys[0]) != value['endpoint_sha256']:
            raise ValueError('Q09_EVAL_ORIGINAL_ENDPOINT_REFERENCE_DRIFT')
        originals.append(inputs)
    if originals[0] != originals[1]:
        raise ValueError('Q09_EVAL_ORIGINAL_INPUTS_CHANGED')
    if (activation.get('source', {}).get('head') != cfg.get('source_sha')
            or profile.get('source', {}).get('experiment', {}).get('revision') != cfg.get('source_sha')):
        raise ValueError('Q09_EVAL_SOURCE_COMMIT_CHANGED')
    expected_output = Path(profile['workspace']) / profile['paths']['runs'] / profile['run_id'] / 'endpoint'
    if Path(cfg['science_output']) != expected_output:
        raise ValueError('Q09_EVAL_OUTPUT_PROFILE_MISMATCH')
    attempts = recovery.get('prior_attempts')
    if not isinstance(attempts, list) or not attempts:
        raise ValueError('Q09_EVAL_PRIOR_ATTEMPT_UNBOUND')
    controls = [Path(cfg['dispatch_dir'])]
    science = [Path(cfg['science_output'])]
    for attempt in attempts:
        if not ref(attempt['failure_evidence'], json_value=False):
            raise ValueError('Q09_EVAL_FAILURE_EVIDENCE_EMPTY')
        controls.append(Path(attempt['dispatch_dir']))
        science.append(Path(attempt['science_output']))
    for control in controls:
        if not control.is_absolute() or control.is_symlink():
            raise ValueError('Q09_EVAL_CONTROL_PATH_INVALID')
        if any(control.rglob('formal-start*.json')):
            raise ValueError('Q09_EVAL_FORMAL_START_ALREADY_PRESENT')
    for output in science:
        if not output.is_absolute() or output.is_symlink() or output.parent.is_symlink():
            raise ValueError('Q09_EVAL_SCIENCE_PATH_INVALID')
        # Empty historical parents remain untouched; any file is evidence to audit.
        if output.parent.exists() and any(p.is_file() or p.is_symlink() for p in output.parent.rglob('*')):
            raise ValueError('Q09_EVAL_PAID_OR_UNCLASSIFIED_OUTPUT_ALREADY_PRESENT')


def native_science_admission(task):
    """Check actual native release inputs; never mint a successor from status."""
    if 'native_eval' in task:
        return native_eval_science_admission(task)
    native = task.get('native_campaign')
    if native is None:
        if task['id'].startswith('q04-'):
            return  # Original Q04 launcher still owns its frozen science.
        raise ValueError('NATIVE_RELEASE_ADAPTER_MISSING')
    if not isinstance(native, dict) or not task['id'].startswith('q09-'):
        raise ValueError('NATIVE_RELEASE_ADAPTER_MISSING')
    source_files = task.get('source_files', {})
    readiness_path = native.get('readiness_receipt')
    state_path = str(Path(native.get('science_root', '')) / 'execution-state.json')
    if (not isinstance(readiness_path, str)
            or readiness_path not in source_files or state_path not in source_files
            or not isinstance(native.get('next_round'), int)
            or not isinstance(native.get('rollouts_consumed'), int)):
        raise ValueError('Q09_NATIVE_RELEASE_IDENTITY_UNBOUND')
    if any(digest(p) != source_files[p] for p in (readiness_path, state_path)):
        raise ValueError('Q09_NATIVE_RELEASE_BYTES_DRIFTED')
    readiness = read(readiness_path)
    state = read(state_path)
    authority = readiness.get('authority')
    if (readiness.get('schema') != 'openevo.execution-readiness.v1'
            or readiness.get('formal_task_consumption_allowed') is not True
            or not isinstance(authority, dict)
            or authority.get('router_status') != 'launch-authorized'
            or not readiness_stages_pass(readiness, source_files)):
        raise ValueError('Q09_NATIVE_EXECUTION_READINESS_NOT_PASS')
    if (state.get('schema') != 'openevo.ceiling1-stage2-execution-state.v1'
            or state.get('status') != 'RUNNING'
            or type(state.get('next_round')) is not int
            or state.get('rounds_complete') != list(range(state['next_round']))
            or state.get('next_round') != native['next_round']
            or state.get('rollouts_consumed') != native['rollouts_consumed']
            or state.get('rollouts_consumed') != 128 * state.get('next_round', -1)
            or not 120 <= state['next_round'] < 160
            or state.get('final_panel_access_count') != 0
            or state.get('external_teacher_calls') != 0):
        raise ValueError('Q09_NATIVE_SCIENCE_STATE_NOT_UNSPENT_READY')
    # The native runner is the final authority for live PID, health resolutions,
    # exact saved phase, step budget and any consumer claim not represented here.


def source_bound_candidates(root):
    """Query the existing #746 observer from this writer's exclusive lock.

    A CANDIDATE is only a necessary gate. Native science/owner launchers still
    decide whether a real paid operation may begin.
    """
    projection_path = Path(__file__).with_name('science_ready_projection.py')
    if projection_path.is_symlink() or not projection_path.is_file():
        raise ValueError('SCIENCE_PROJECTION_SOURCE_MISSING')
    if digest(projection_path) != PROJECTION_SHA256:
        raise ValueError('SCIENCE_PROJECTION_SOURCE_DRIFTED')
    spec = importlib.util.spec_from_file_location('science_ready_projection', projection_path)
    if spec is None or spec.loader is None:
        raise ValueError('SCIENCE_PROJECTION_IMPORT_UNAVAILABLE')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    projected = module.project_pool(root)
    if projected.get('execution_authority') is not False or projected.get('mutation') is not False:
        raise ValueError('SCIENCE_PROJECTION_SCOPE_CHANGED')
    return {row['task_id'] for row in projected['tasks']
            if row['verdict'] == 'CANDIDATE_FOR_EXISTING_LAUNCHER'}


def observe(t, live, prior):
    pids = running(t, live)
    if pids:
        return {'status': 'RUNNING', 'pids': pids}
    marker = t.get('completion')
    if marker and Path(marker['path']).exists():
        value = read(marker['path'])
        if all(value.get(k) == v for k, v in marker['equals'].items()):
            return {'status': 'COMPLETE', 'completion_sha256': digest(marker['path'])}
        return {'status': 'DIAGNOSTIC_REQUIRED', 'reason': 'completion marker differs'}
    state_path = t.get('state_path')
    if state_path and Path(state_path).exists():
        state = read(state_path)
        info = {k: state.get(k) for k in ('state_id', 'next_round')}
        if state.get('status') == 'PAUSE_FOR_DIAGNOSTIC':
            return {'status': 'DIAGNOSTIC_REQUIRED', **info}
        if t['mode'] == 'adopt' or prior.get('attempted'):
            return {'status': 'CONTINUATION_REQUIRED', **info}
    if t['mode'] == 'hold':
        return {'status': 'CONTINUATION_REQUIRED', 'reason': t['reason']}
    if prior.get('attempted'):
        return {'status': 'DIAGNOSTIC_REQUIRED', 'reason': 'launcher exited without verified completion; no automatic retry'}
    if t['mode'] == 'adopt':
        return {'status': 'DIAGNOSTIC_REQUIRED', 'reason': 'adopted worker absent'}
    return {'status': 'QUEUED'}


# The current human turn explicitly permits GPU1/4 in this report task
# ("你可以积极利用 GPU1 和 4"), after the older resource withdrawal. This reviewed
# exception permits only the named same-source claim before 2026-10-29 01:00
# UTC. It grants no holder and does not authorize other GPU1 work or replay.
GPU1_REAUTHORIZED_CLAIM = 'q09-s2-r143-same-source-resume-gpu1-20261028'
GPU1_REAUTHORIZATION_START = datetime(2026, 10, 28, 16, 40, tzinfo=timezone.utc).timestamp()
GPU1_REAUTHORIZATION_DEADLINE = datetime(2026, 10, 29, 1, tzinfo=timezone.utc).timestamp()
GPU1_REAUTHORIZED_RUNNER_SHA256 = '818f7d2147d3ae25b5ccced16497c55a86e8380c5ae77b0afebef360dc123205'
GPU1_PRESERVED101_CLAIM = 'q09-s2-r143-preserved101-gpu1-202610281958'
# Exact separately reviewed runner; private grant evidence remains in its bound proof.
GPU1_PRESERVED101_RUNNER_SHA256 = '219fd3b6f5c1030930e7397f2ffaa8744abde01d15d914f9e2cadf832e8c25c5'
GPU1_FIRSTPARTY_CLAIM = 'q09-s2-r143-firstparty-grant-gpu1-202610282118'
GPU1_FIRSTPARTY_RUNNER_SHA256 = '964ac7f0546be57bbbcd99df850f539834d6585e169ca4450063eff9f1e4f696'
GPU1_FIRSTPARTY_PROOF_SHA256 = '9b3b2455a7a336a418f9dc7d7eae3dc40208d490e43101aa647988585c871e11'
GPU1_FIRSTPARTY_EVENT_SHA256 = 'c170722143d08840aca631d800089686235f9415966bba8ad8f9123cc24e32cb'


def scoped_gpu1_reauthorization(task):
    """Resource exception only; native science and durable-claim gates still run."""
    if not GPU1_REAUTHORIZATION_START <= time.time() < GPU1_REAUTHORIZATION_DEADLINE:
        return False
    try:
        base = '/workspace'
        runner = base + '/maintenance/prepared_q09_runner_202610281815.py'
        proof = base + '/maintenance/Q09-S2-R143-SAME-SOURCE-RESUME-20261028/reviewed-partial-resume.json'
        claim = task['id']
        runner_sha = GPU1_REAUTHORIZED_RUNNER_SHA256
        output = base + '/maintenance/Q09-S2-R143-SAME-SOURCE-POOL-RUN-20261028'
        if claim == GPU1_PRESERVED101_CLAIM:
            runner = base + '/maintenance/prepared_q09_runner_202610281958.py'
            proof = base + '/maintenance/Q09-S2-R143-PRESERVED101-202610281958/reviewed-partial-resume.json'
            output = base + '/maintenance/Q09-S2-R143-PRESERVED101-POOL-RUN-202610281958'
            runner_sha = GPU1_PRESERVED101_RUNNER_SHA256
        elif claim == GPU1_FIRSTPARTY_CLAIM:
            runner = base + '/maintenance/prepared_q09_runner_202610282132.py'
            proof = base + '/maintenance/Q09-S2-R143-FIRSTPARTY-GRANT-202610282118/reviewed-partial-resume.json'
            output = base + '/maintenance/Q09-S2-R143-FIRSTPARTY-GRANT-POOL-RUN-202610282118'
            runner_sha = GPU1_FIRSTPARTY_RUNNER_SHA256
            event_path = base + '/maintenance/q09-firstparty-event-verification-202610282118.json'
            pins = task['source_files']
            if (pins.get(proof) != GPU1_FIRSTPARTY_PROOF_SHA256
                    or digest(proof) != GPU1_FIRSTPARTY_PROOF_SHA256
                    or pins.get(event_path) != GPU1_FIRSTPARTY_EVENT_SHA256
                    or Path(event_path).is_symlink()
                    or digest(event_path) != GPU1_FIRSTPARTY_EVENT_SHA256):
                return False
            event = read(event_path)
            if (event.get('source') != 'original_native_codex_session_event_direct_mac_read'
                    or event.get('event_type') != 'response_item'
                    or event.get('payload_type') != 'message' or event.get('role') != 'user'
                    or event.get('timestamp') != '2026-10-28T16:39:58.750Z'
                    or event.get('line') != 16
                    or event.get('raw_line_sha256') != '8e4604051b8829e2fae4fb1950a38c6275fcc5ace2fa108a579066d1a9701a7a'
                    or event.get('verbatim_grant') != '你可以积极利用 GPU1 和 4'
                    or event.get('exact_grant_present') is not True
                    or event.get('raw_user_text_exported') is not False):
                return False
        elif claim != GPU1_REAUTHORIZED_CLAIM:
            return False
        if not isinstance(runner_sha, str) or len(runner_sha) != 64:
            return False
        cfg = task['native_campaign']
        pins = task['source_files']
        return (task['gpu_uuid'] == GPU_UUIDS[0]
                and task['argv'] == ['python3', runner, '--root',
                                     base + '/maintenance/Q04-POOL-20261022',
                                     '--task-id', claim]
                and task['output_root'] == output
                and cfg['science_root'] == base + '/runs/q09-active-s2-r120-gpu2-formal-20261020'
                and cfg['dispatch_dir'] == base + '/control/q09-s2-r143-reviewed-dispatch-20261028'
                and cfg['source_sha'] == 'f952c09bbdefc0aa6cef570003702bd09bec87cc'
                and cfg['authority_sha'] == '33cbc3ac082c0764094c1ca68012e87c6bb29161'
                and cfg['next_round'] == 143 and cfg['rollouts_consumed'] == 18304
                and cfg['reservation'] == {'enabled': False, 'expires_at': None}
                and cfg['reviewed_partial_resume'] == proof
                and pins.get(runner) == runner_sha
                and not Path(runner).is_symlink()
                and digest(runner) == runner_sha
                and not Path(proof).is_symlink() and pins.get(proof) == digest(proof))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def tick(manifest, root, *, gpu_query=snapshot_gpu, live_query=processes,
         owner_query=snapshot_verified_owner, candidate_query=source_bound_candidates,
         excluded_gpu_uuids=()):
    root = Path(root)
    manifest = Path(manifest)
    if (manifest.is_symlink() or (root / 'manifest.json').is_symlink()
            or manifest.resolve() != (root / 'manifest.json').resolve()):
        raise ValueError('SINGLE_WRITER_MANIFEST_PATH_MISMATCH')
    manifest_sha = digest(manifest)
    m = validate_manifest(read(manifest))
    excluded = tuple(excluded_gpu_uuids)
    if len(set(excluded)) != len(excluded) or not set(excluded).issubset(set(GPU_UUIDS)):
        raise ValueError('GPU_EXCLUSION_INVALID')
    effective_excluded = set(excluded) | CURRENT_OWNER_WITHDRAWN_UUIDS
    prior = read(root / 'state.json') if (root / 'state.json').exists() else {'tasks': {}}
    live = live_query()
    state = {'time': time.time(), 'manifest_sha256': manifest_sha, 'tasks': {}}
    for t in m['tasks']:
        old = prior['tasks'].get(t['id'], {})
        observation = observe(t, live, old)
        current = {**old, **observation}
        if observation['status'] in ('COMPLETE', 'RUNNING'):
            # Queue admission and a previous failure are observations, not
            # durable claims. Do not carry their reason into a fresh sealed
            # or live state. Preserve attempted and completion receipts.
            current.pop('wait_reason', None)
            if 'reason' not in observation:
                current.pop('reason', None)
            if observation['status'] == 'COMPLETE':
                current.pop('pids', None)  # stale launch PIDs are not live proof
        state['tasks'][t['id']] = current
    gpu = gpu_query()
    state['gpu'] = gpu
    candidates = set()
    projection_problem = None
    if any(s['status'] == 'QUEUED' for s in state['tasks'].values()):
        try:
            candidates = candidate_query(root)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            projection_problem = type(exc).__name__
    launched_ids = []
    for t in sorted(m['tasks'], key=lambda x: (x.get('priority', 100), x['id'])):
        s = state['tasks'][t['id']]
        if s['status'] != 'QUEUED':
            continue
        s.pop('wait_reason', None)
        scoped_gpu1 = scoped_gpu1_reauthorization(t) if t['gpu_uuid'] == GPU_UUIDS[0] else False
        if t['gpu_uuid'] in effective_excluded and not scoped_gpu1:
            s['wait_reason'] = 'GPU_EXCLUDED_BY_CURRENT_OWNER_INSTRUCTION'
            continue
        if t['id'] not in candidates:
            s['wait_reason'] = ('SCIENCE_PROJECTION_NOT_CANDIDATE' if projection_problem is None
                                else 'SCIENCE_PROJECTION_UNTRUSTED: ' + projection_problem)
            continue
        if any(state['tasks'][d]['status'] != 'COMPLETE' for d in t.get('depends_on', [])):
            s['wait_reason'] = 'dependency not complete'
            continue
        active = [x for x in m['tasks'] if x['gpu_uuid'] == t['gpu_uuid']
                  and state['tasks'][x['id']]['status'] == 'RUNNING']
        # Co-location requires the same verified phase coordinator. Independent
        # whole-job reservations never rely solely on instantaneous free memory.
        cohort = t.get('phase_group')
        if active and (not cohort or any(x.get('phase_group') != cohort for x in active)):
            s['wait_reason'] = 'active worker has a different capacity coordinator'
            continue
        if len(active) >= 2:
            s['wait_reason'] = 'two experiments already on card'
            continue
        if gpu[t['gpu_uuid']]['free'] < t['peak_mib'] + t['headroom_mib']:
            s['wait_reason'] = 'GPU memory admission'
            continue
        if os.statvfs(root).f_bavail * os.statvfs(root).f_frsize < m['min_free_bytes']:
            s['wait_reason'] = 'shared disk admission'
            continue
        if any(digest(p) != sha for p, sha in t['source_files'].items()):
            s.update(status='DIAGNOSTIC_REQUIRED', reason='exact launcher/source identity changed')
            continue
        try:
            native_science_admission(t)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            s['wait_reason'] = 'SCIENCE_NOT_READY: ' + str(exc)
            continue
        output = Path(t['output_root'])
        if output.exists() and any(output.iterdir()):
            s.update(status='DIAGNOSTIC_REQUIRED', reason='nonempty scientific output refuses replay')
            continue
        attempt = root / (t['id'] + '-dispatch.json')
        if attempt.exists():
            s.update(status='DIAGNOSTIC_REQUIRED', attempted=True, reason='existing launch claim; refuses replay')
            continue
        if digest(manifest) != manifest_sha:
            s['wait_reason'] = 'SCIENCE_MANIFEST_CHANGED_DURING_ADMISSION'
            continue
        try:
            owner_pre = owner_query(t['gpu_uuid'])
            if owner_pre['gpu_uuid'] != t['gpu_uuid']:
                raise ValueError('GPU_OWNER_UUID_DRIFT')
            if owner_pre['free_mib'] < t['peak_mib'] + t['headroom_mib']:
                raise ValueError('GPU_OWNER_FREE_MEMORY_INSUFFICIENT')
        except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
            s['wait_reason'] = 'RESOURCE_NOT_ADMITTED: ' + str(exc)
            continue
        # Durable claim precedes launch. A crash here sacrifices availability,
        # never permits duplicate paid work. Restart requires diagnosis.
        with attempt.open('x') as f:
            json.dump({'task': t, 'time': time.time(), 'owner_pre': owner_pre}, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        try:
            owner_post = owner_query(t['gpu_uuid'])
            if owner_post['gpu_uuid'] != t['gpu_uuid']:
                raise ValueError('GPU_OWNER_UUID_CHANGED_AFTER_CLAIM')
            if owner_post['free_mib'] < t['peak_mib'] + t['headroom_mib']:
                raise ValueError('GPU_OWNER_CAPACITY_CHANGED_AFTER_CLAIM')
        except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
            s.update(status='DIAGNOSTIC_REQUIRED', attempted=True,
                     reason='RESOURCE_CHANGED_AFTER_DURABLE_CLAIM: ' + str(exc))
            continue  # Never launch or silently retry this exact claimed cell.
        try:
            source_changed = (digest(manifest) != manifest_sha or any(
                digest(p) != sha for p, sha in t['source_files'].items()))
        except OSError:
            source_changed = True
        if source_changed:
            s.update(status='DIAGNOSTIC_REQUIRED', attempted=True,
                     reason='SCIENCE_SOURCE_CHANGED_AFTER_DURABLE_CLAIM')
            continue
        if scoped_gpu1 and not scoped_gpu1_reauthorization(t):
            s.update(status='DIAGNOSTIC_REQUIRED', attempted=True,
                     reason='SCOPED_GPU1_AUTHORIZATION_EXPIRED_OR_CHANGED_AFTER_CLAIM')
            continue
        with (root / (t['id'] + '-owner-admission.json')).open('x') as f:
            json.dump({'task_id': t['id'], 'pre': owner_pre, 'post': owner_post}, f)
            f.flush()
            os.fsync(f.fileno())
        env = {**os.environ, **t.get('environment', {}),
               'OPENEVO_AUTHORIZED_GPU_UUIDS': t['gpu_uuid']}
        with (root / (t['id'] + '.log')).open('x') as log:
            p = subprocess.Popen(t['argv'], stdout=log, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
        s.update(status='RUNNING', attempted=True, launcher_pid=p.pid, pids=[p.pid])
        launched_ids.append(t['id'])
        # Conservative within-tick reservation; fresh NVML is read next tick.
        gpu[t['gpu_uuid']]['free'] -= t['peak_mib'] + t['headroom_mib']
    # Report the *known pool* disposition, never claim global science READY
    # coverage from a partial cross-campaign projection.
    known_statuses = ('QUEUED', 'RUNNING', 'CONTINUATION_REQUIRED',
                      'DIAGNOSTIC_REQUIRED', 'COMPLETE')
    counts = {status: sum(s['status'] == status for s in state['tasks'].values())
              for status in known_statuses}
    if launched_ids:
        decision = 'DISPATCH_STARTED'
    elif projection_problem is not None:
        decision = 'SCIENCE_PROJECTION_UNTRUSTED'
    elif counts['QUEUED'] == 0:
        decision = 'NO_REGISTERED_QUEUED_TASKS'
    elif any(task_id in candidates for task_id, s in state['tasks'].items()
             if s['status'] == 'QUEUED'):
        decision = 'CANDIDATE_ADMISSION_BLOCKED'
    else:
        decision = 'NO_PROJECTED_LAUNCH_CANDIDATES'
    state['tick_decision'] = {
        'schema': 'openevo.advisor-pool-decision.202610261920',
        'code': decision, 'task_counts': counts,
        'projection_candidate_count': len(candidates),
        'spawned_task_ids': launched_ids,
        'global_ready_supply': 'UNVERIFIED_CROSS_CAMPAIGN',
    }
    atomic(root / 'state.json', state)
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--exclude-gpu-uuid', action='append', default=[],
                    help='current owner withdrawal: never admit work on this exact GPU UUID')
    ap.add_argument('--prepare-module', type=Path)
    ap.add_argument('--prepare-module-sha256')
    ap.add_argument('--prepare-plan', type=Path)
    ap.add_argument('--prepare-plan-sha256')
    a = ap.parse_args()
    # Fail before creating a pool root/lock if the launch command dropped the
    # mandatory owner-withdrawal flag. The tick gate above protects direct callers.
    if not CURRENT_OWNER_WITHDRAWN_UUIDS.issubset(set(a.exclude_gpu_uuid)):
        ap.error('GPU_OWNER_WITHDRAWAL_FLAG_REQUIRED')
    preparation = None
    if any((a.prepare_module,a.prepare_module_sha256,a.prepare_plan,a.prepare_plan_sha256)):
        assert all((a.prepare_module,a.prepare_module_sha256,a.prepare_plan,a.prepare_plan_sha256))
        assert digest(a.prepare_module)==a.prepare_module_sha256
        assert digest(a.prepare_plan)==a.prepare_plan_sha256
        spec=importlib.util.spec_from_file_location('prepared_endpoints',a.prepare_module)
        preparation=importlib.util.module_from_spec(spec);spec.loader.exec_module(preparation)

    a.root.mkdir(parents=True, exist_ok=True)
    last_preparation = 0.0
    with (a.root / 'dispatcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                tick(a.manifest, a.root, excluded_gpu_uuids=a.exclude_gpu_uuid)
                now = time.monotonic()
                if preparation is not None and now - last_preparation >= 3:
                    error_path = a.prepare_plan.parent/'preparation-error.json'
                    if not error_path.exists():
                        try:
                            assert digest(a.prepare_module)==a.prepare_module_sha256
                            events=preparation.prepare_once(a.prepare_plan,a.prepare_plan_sha256)
                            atomic(a.prepare_plan.parent/'preparation-status.json',{'time':time.time(),'events':events})
                        except Exception as e:
                            # A preparation fault never blocks already-ready native jobs.
                            # Latch it for engineering, without scientific replay or costly repeated validation.
                            atomic(error_path,{'time':time.time(),'error':repr(e),'plan_sha256':a.prepare_plan_sha256})
                    last_preparation = time.monotonic()
                    tick(a.manifest, a.root, excluded_gpu_uuids=a.exclude_gpu_uuid)
            except Exception as e:
                atomic(a.root / 'dispatcher-error.json', {'time': time.time(), 'error': repr(e)})
                raise
            if a.once:
                break
            time.sleep(1)


if __name__ == '__main__':
    main()
