"""Behavioral tests: a dispatch error must not turn into an automatic retry loop."""
from pathlib import Path
import json
import pytest
from scripts.openevo_runtime import unattended_supervisor as us
from tests.test_unattended_supervisor import _base_config


def prepare(tmp_path, monkeypatch, arm_count=1):
    p, _ = _base_config(tmp_path, arm_count=arm_count)
    cfg = us.config(p)
    monkeypatch.setattr(us, 'git_clean', lambda *args: (True, 'clean'))
    monkeypatch.setattr(us, 'git_head', lambda *args: 'a' * 40)
    monkeypatch.setattr(us, 'lock_busy', lambda *args: False)
    return cfg


def test_unchanged_dispatch_exception_attempted_once_across_polls_and_restart(tmp_path, monkeypatch):
    cfg = prepare(tmp_path, monkeypatch)
    attempted = []
    def fail(d, arm, lane):
        attempted.append(arm['id'])
        raise PermissionError('fixture: executable denied before launch')
    monkeypatch.setattr(us, 'start', fail)
    result = us.run(cfg, execute=True, max_cycles=3, sleep_fn=lambda _: None)
    assert attempted == ['rank-0']
    assert result['arms']['rank-0']['action'] == 'BLOCKED_NEEDS_HUMAN'
    us.run(cfg, execute=True, max_cycles=2, sleep_fn=lambda _: None)
    assert attempted == ['rank-0']
    marker = json.loads(us.terminal_marker_path(cfg['arms'][0]).read_text())
    assert marker['kind'] == 'dispatch_exception'
    assert marker['side_effect_status'] == 'UNKNOWN_REQUIRES_READBACK'
    assert marker['scientific_authority_granted_by_supervisor'] is False


def test_blocked_arm_does_not_prevent_independent_ready_arm_dispatch(tmp_path, monkeypatch):
    cfg = prepare(tmp_path, monkeypatch, arm_count=2)
    attempted = []
    class Proc:
        pid = 999999
        def poll(self): return None
        def terminate(self): pytest.fail('must not terminate independent arm')
    def launch(d, arm, lane):
        attempted.append(arm['id'])
        if arm['id'] == 'rank-0': raise PermissionError('fixture denied')
        return {'arm_id': arm['id'], 'lane_id': lane['id'], 'process': Proc(),
                'stream': (tmp_path/'fake-child.log').open('ab'), 'dispatch_id':'fixture'}
    monkeypatch.setattr(us, 'start', launch)
    us.run(cfg, execute=True, max_cycles=3, sleep_fn=lambda _: None)
    assert attempted == ['rank-0', 'rank-1']


def test_fresh_arm_lock_is_respected_before_execution_state_exists(tmp_path, monkeypatch):
    cfg = prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(us, 'lock_busy', lambda *args: True)
    assert us.discover(cfg['arms'][0])['action'] == 'RUNNING_EXTERNAL'
    assert us.plan(cfg)['assignments'] == []


def test_unreaped_dispatch_intent_blocks_restart_without_progress(tmp_path, monkeypatch):
    """A controller crash may leave only the durable pre-spawn intent."""
    cfg = prepare(tmp_path, monkeypatch)
    arm = cfg['arms'][0]
    recovery = us.recovery_identity(arm)
    intent = Path(cfg['state_root']) / 'dispatches' / arm['id'] / 'orphan' / 'intent.json'
    us.atomic(intent, {'dispatch_id': 'orphan', 'arm_id': arm['id'], 'kind': 'launch',
                      'progress_cwd': arm['launch']['cwd'], 'pre_git_head': 'a' * 40,
                      'pre_recovery_sha256': recovery['sha256'],
                      'pre_recovery_projection': recovery['projection']})
    attempted = []
    def forbidden(*args):
        attempted.append(1)
        raise AssertionError('unacknowledged dispatch must not be replayed')
    monkeypatch.setattr(us, 'start', forbidden)
    for _ in range(2):
        result = us.run(cfg, execute=True, max_cycles=2, sleep_fn=lambda _: None)
        assert result['arms'][arm['id']]['action'] == 'BLOCKED_NEEDS_HUMAN'
    assert attempted == []


def test_corrupt_arm_state_does_not_hide_independent_ready_arm(tmp_path, monkeypatch):
    cfg = prepare(tmp_path, monkeypatch, arm_count=2)
    arm = cfg['arms'][0]
    (Path(arm['run_output_root']) / 'execution-state.json').write_text('{')
    result = us.run(cfg, execute=False)
    assert result['arms'][arm['id']]['action'] == 'BLOCKED_NEEDS_HUMAN'
    assert result['assignments'][0]['arm_id'] == 'rank-1'
    assert 'unreadable JSON' in result['arms'][arm['id']]['reason']


def test_dispatch_readback_race_is_persistent_and_arm_scoped(tmp_path, monkeypatch):
    cfg = prepare(tmp_path, monkeypatch, arm_count=2)
    original = us.recovery_identity
    def unavailable(arm):
        if arm['id'] == 'rank-0':
            raise PermissionError('fixture state changed after planning')
        return original(arm)
    monkeypatch.setattr(us, 'recovery_identity', unavailable)
    attempted = []
    def fail(d, arm, lane):
        attempted.append(arm['id'])
        raise PermissionError('fixture sibling launch')
    monkeypatch.setattr(us, 'start', fail)
    us.run(cfg, execute=True, max_cycles=2, sleep_fn=lambda _: None)
    assert attempted == ['rank-1']
    marker = us.load(us.terminal_marker_path(cfg['arms'][0]))
    assert 'state changed after planning' in marker['error']


@pytest.mark.parametrize('reap_interrupted', [False, True])
def test_real_child_exit_without_arm_marker_cannot_replay(tmp_path, reap_interrupted):
    import subprocess
    import sys
    p, doc = _base_config(tmp_path, arm_count=1)
    arm = doc['arms'][0]
    cwd = Path(arm['launch']['cwd'])
    executable = cwd / 'scripts' / 'oe'
    executable.write_text(f'#!{sys.executable}\n' + '''import sys
from pathlib import Path
root = Path(sys.argv[sys.argv.index('--run-output-root') + 1])
with (root / 'calls').open('a') as stream:
    stream.write('called\\n')
raise SystemExit(2)
''')
    executable.chmod(0o755)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(cwd), *args], text=True).strip()
    git('init', '-q')
    git('config', 'user.name', 'Fixture')
    git('config', 'user.email', 'fixture@example.invalid')
    git('add', 'scripts/oe')
    git('commit', '-qm', 'CPU-only failing child')
    arm['launch']['expected_git_head'] = git('rev-parse', 'HEAD')
    p.write_text(json.dumps(doc))
    cfg = us.config(p)
    arm = cfg['arms'][0]
    child = us.start(cfg, arm, cfg['lanes'][0])
    assert child['process'].wait(timeout=10) == 2
    child['stream'].close()
    if reap_interrupted:
        us.reap(cfg, {arm['id']: child})
        # Crash after terminal.json commit but before arm-marker commit.
        us.terminal_marker_path(arm).unlink()
    assert not us.terminal_marker_path(arm).exists()
    for _ in range(2):
        result = us.run(cfg, execute=True, max_cycles=2, sleep_fn=lambda _: None)
        row = result['arms'][arm['id']]
        assert row['action'] == 'BLOCKED_NEEDS_HUMAN'
        assert Path(row['failure_receipt']['path']).is_file()
    assert (Path(arm['run_output_root']) / 'calls').read_text().splitlines() == ['called']


def test_orphan_progress_uses_canonical_resume_and_preserves_rollout(tmp_path, monkeypatch):
    from tests.test_unattended_supervisor import _json
    cfg = prepare(tmp_path, monkeypatch)
    arm = cfg['arms'][0]
    root = Path(arm['run_output_root'])
    _json(root / 'execution-state.json', {'status': 'RUNNING', 'next_round': 152})
    before = us.recovery_identity(arm)
    receipt = Path(cfg['state_root']) / 'dispatches' / arm['id'] / 'orphan' / 'intent.json'
    us.atomic(receipt, {'dispatch_id': 'orphan', 'arm_id': arm['id'], 'kind': 'launch',
                       'progress_cwd': arm['launch']['cwd'], 'pre_git_head': 'a' * 40,
                       'pre_recovery_sha256': before['sha256'],
                       'pre_recovery_projection': before['projection']})
    _json(root / 'round-152' / 'state-before.json', {'round_index': 152, 'state_id': 'before'})
    _json(root / 'round-152' / 'ROLLOUT_PHASE_COMPLETE.json',
          {'status': 'PASS', 'round_index': 152, 'state_id_before': 'before'})
    row = us.discover(arm)
    assert row['action'] == 'READY_TO_DISPATCH'
    assert 'rollout' in row['recovery_plan']['must_not_replay']
    # A malformed historical receipt must not be skipped just because a later
    # intent has valid progress.
    other = receipt.parent.parent / 'other' / 'intent.json'
    us.atomic(other, {'arm_id': 'wrong', 'dispatch_id': 'other'})
    assert us.discover(arm)['action'] == 'BLOCKED_NEEDS_HUMAN'
    monkeypatch.setattr(us, 'lock_busy', lambda *_: True)
    assert us.discover(arm)['action'] == 'RUNNING_EXTERNAL'


def test_real_posix_spawn_permission_error_is_not_retried(tmp_path):
    """Real Git + Popen + filesystem boundary; no model or GPU involved."""
    import os
    import subprocess
    if os.name != 'posix': pytest.skip('POSIX executable permission boundary')
    p, doc = _base_config(tmp_path, arm_count=1)
    cwd = Path(doc['arms'][0]['launch']['cwd'])
    executable = cwd/'scripts'/'oe'
    executable.chmod(0o644)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(cwd), *args], text=True).strip()
    git('init', '-q')
    git('config', 'user.name', 'Fixture')
    git('config', 'user.email', 'fixture@example.invalid')
    git('add', 'scripts/oe')
    git('commit', '-qm', 'non-executable canonical fixture')
    doc['arms'][0]['launch']['expected_git_head'] = git('rev-parse', 'HEAD')
    p.write_text(json.dumps(doc))
    cfg = us.config(p)
    for _ in range(2):
        result = us.run(cfg, execute=True, max_cycles=3, sleep_fn=lambda _: None)
        assert result['arms']['rank-0']['action'] == 'BLOCKED_NEEDS_HUMAN'
    events = [json.loads(line) for line in (Path(cfg['state_root'])/'events.jsonl').read_text().splitlines()]
    errors = [e for e in events if e.get('event') == 'DISPATCH_EXCEPTION']
    assert len(errors) == 1
    marker = json.loads(us.terminal_marker_path(cfg['arms'][0]).read_text())
    assert marker['error'].startswith('PermissionError:')
    assert marker['side_effect_status'] == 'UNKNOWN_REQUIRES_READBACK'


def test_materialized_progress_after_dispatch_fault_still_uses_exact_recovery(tmp_path, monkeypatch):
    from tests.test_unattended_supervisor import _json
    cfg = prepare(tmp_path, monkeypatch)
    arm = cfg['arms'][0]
    root = Path(arm['run_output_root'])
    _json(root/'execution-state.json', {'status':'RUNNING','next_round':152,'final_panel_access_count':0})
    def fail(*args): raise OSError('fixture launch failed')
    monkeypatch.setattr(us, 'start', fail)
    us.run(cfg, execute=True, max_cycles=1, sleep_fn=lambda _: None)
    rr = root/'round-152'
    _json(rr/'state-before.json', {'round_index':152,'state_id':'before'})
    _json(rr/'ROLLOUT_PHASE_COMPLETE.json', {'status':'PASS','round_index':152,'state_id_before':'before'})
    projection = us.discover(arm)
    assert projection['action'] == 'READY_TO_DISPATCH'
    assert 'rollout' in projection['recovery_plan']['must_not_replay']


def test_bound_health_resume_does_not_forget_an_unchanged_dispatch_failure(tmp_path, monkeypatch):
    from tests.test_unattended_supervisor import _json, _write_known_health_pause
    cfg = prepare(tmp_path, monkeypatch)
    arm = cfg['arms'][0]
    root = Path(arm['run_output_root'])
    pause_sha = _write_known_health_pause(root)
    _json(root/'execution-state.json', {'status':'PAUSE_FOR_DIAGNOSTIC','next_round':153,
          'final_panel_access_count':0,'carrier_health_pause_sha256':pause_sha})
    monkeypatch.setattr(us, 'bound_health_resolution', lambda arm, detail: (True, {'through_round':152}))
    attempted = []
    def fail(*args):
        attempted.append(1)
        raise PermissionError('same launch failure with valid health resolution')
    monkeypatch.setattr(us, 'start', fail)
    out = us.run(cfg, execute=True, max_cycles=3, sleep_fn=lambda _: None)
    assert attempted == [1]
    assert out['arms'][arm['id']]['action'] == 'BLOCKED_NEEDS_HUMAN'


def test_failed_launch_metadata_is_not_scientific_progress(tmp_path, monkeypatch):
    """A launcher can rewrite state while repeatedly failing the same preflight."""
    from tests.test_unattended_supervisor import _json
    cfg = prepare(tmp_path, monkeypatch)
    arm = cfg['arms'][0]
    state = Path(arm['run_output_root']) / 'execution-state.json'
    _json(state, {'status': 'RUNNING', 'next_round': 152, 'final_panel_access_count': 0})
    attempted = []

    def fail(d, arm, lane):
        attempted.append(arm['id'])
        _json(state, {'status': 'RUNNING', 'next_round': 152,
                      'final_panel_access_count': 0, 'last_preflight_attempt': len(attempted)})
        raise PermissionError('unchanged payload output permission')

    monkeypatch.setattr(us, 'start', fail)
    us.run(cfg, execute=True, max_cycles=3, sleep_fn=lambda _: None)
    us.run(cfg, execute=True, max_cycles=2, sleep_fn=lambda _: None)
    assert attempted == ['rank-0']
    assert us.discover(arm)['action'] == 'BLOCKED_NEEDS_HUMAN'


def test_real_child_rewriting_metadata_then_failing_is_dispatched_once(tmp_path, monkeypatch):
    """Use the real Git/Popen/exit/reap path, including supervisor restart."""
    import subprocess
    import sys
    p, doc = _base_config(tmp_path, arm_count=1)
    cwd = Path(doc['arms'][0]['launch']['cwd'])
    root = Path(doc['arms'][0]['run_output_root'])
    state = root / 'execution-state.json'
    state.write_text(json.dumps({'status': 'RUNNING', 'next_round': 152,
                                 'final_panel_access_count': 0}))
    executable = cwd / 'scripts' / 'oe'
    executable.write_text(f'#!{sys.executable}\n' + '''import json, sys
from pathlib import Path
root = Path(sys.argv[sys.argv.index('--run-output-root') + 1])
count = root / 'fixture-invocations.txt'
with count.open('a') as stream:
    stream.write('invoked\\n')
state = root / 'execution-state.json'
value = json.loads(state.read_text())
value['last_preflight_attempt'] = len(count.read_text().splitlines())
state.write_text(json.dumps(value))
raise SystemExit(2)
''')
    executable.chmod(0o755)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(cwd), *args], text=True).strip()
    git('init', '-q')
    git('config', 'user.name', 'Fixture')
    git('config', 'user.email', 'fixture@example.invalid')
    git('add', 'scripts/oe')
    git('commit', '-qm', 'fixture launcher failing after metadata write')
    doc['arms'][0]['launch']['expected_git_head'] = git('rev-parse', 'HEAD')
    p.write_text(json.dumps(doc))
    cfg = us.config(p)
    spawned = []
    real_start = us.start

    def observe_start(*args):
        child = real_start(*args)
        spawned.append(child['process'])
        return child

    monkeypatch.setattr(us, 'start', observe_start)

    def await_child_exit(_):
        if spawned:
            spawned[-1].wait(timeout=10)

    for _ in range(2):
        result = us.run(cfg, execute=True, max_cycles=6,
                        sleep_fn=await_child_exit)
        assert result['arms']['rank-0']['action'] == 'BLOCKED_NEEDS_HUMAN'
    assert (root / 'fixture-invocations.txt').read_text().splitlines() == ['invoked']


def test_seal_only_child_completion_is_not_blocked_by_terminal_marker(tmp_path, monkeypatch):
    from tests.test_unattended_supervisor import _json
    cfg = prepare(tmp_path, monkeypatch)
    arm = cfg['arms'][0]
    state = Path(arm['run_output_root']) / 'execution-state.json'
    _json(state, {'status': 'RUNNING', 'next_round': 154, 'final_panel_access_count': 0})
    before = {'pre_state_sha256': us.sha(state), 'pre_git_head': 'a' * 40,
              'pre_recovery_sha256': us.recovery_identity(arm)['sha256']}
    us.dispatch_failure(cfg, arm, cfg['lanes'][0], before, OSError('fixture seal acknowledgement lost'))
    _json(state, {'status': 'SEALED', 'next_round': 154, 'final_panel_access_count': 0})
    assert us.discover(arm)['action'] == 'COMPLETE'


def test_old_terminal_without_recovery_identity_cannot_be_cleared_by_metadata(tmp_path, monkeypatch):
    from tests.test_unattended_supervisor import _json
    cfg = prepare(tmp_path, monkeypatch)
    arm = cfg['arms'][0]
    state = Path(arm['run_output_root']) / 'execution-state.json'
    _json(state, {'status': 'RUNNING', 'next_round': 152})
    us.atomic(us.terminal_marker_path(arm), {
        'pre_state_sha256': us.sha(state), 'pre_git_head': 'a' * 40,
        'progress_cwd': arm['launch']['cwd'], 'returncode': 2,
    })
    _json(state, {'status': 'RUNNING', 'next_round': 152, 'heartbeat': 'changed'})
    assert us.discover(arm)['action'] == 'BLOCKED_NEEDS_HUMAN'


@pytest.mark.parametrize('mutate', ['rewrite', 'remove'])
def test_completed_receipt_identity_drift_is_not_new_progress(tmp_path, monkeypatch, mutate):
    from tests.test_unattended_supervisor import _json
    cfg = prepare(tmp_path, monkeypatch)
    arm = cfg['arms'][0]
    root = Path(arm['run_output_root'])
    state = root / 'execution-state.json'
    rollout = root / 'round-152' / 'ROLLOUT_PHASE_COMPLETE.json'
    _json(state, {'status': 'RUNNING', 'next_round': 152, 'final_panel_access_count': 0})
    _json(root / 'round-152' / 'state-before.json', {'round_index': 152, 'state_id': 'before'})
    _json(rollout, {'status': 'PASS', 'round_index': 152, 'state_id_before': 'before'})
    attempted = []

    def fail(*args):
        attempted.append(1)
        if mutate == 'rewrite':
            _json(rollout, {'status': 'PASS', 'round_index': 152, 'state_id_before': 'before',
                            'timestamp': len(attempted)})
        elif rollout.exists():
            rollout.unlink()
        raise OSError('fixture: receipt identity drift during failed launch')

    monkeypatch.setattr(us, 'start', fail)
    us.run(cfg, execute=True, max_cycles=3, sleep_fn=lambda _: None)
    us.run(cfg, execute=True, max_cycles=2, sleep_fn=lambda _: None)
    assert attempted == [1]
    assert us.discover(arm)['action'] == 'BLOCKED_NEEDS_HUMAN'


@pytest.mark.parametrize('after_round', [152, 154])
def test_round_regression_or_unsealed_round_skip_cannot_unlock_retry(tmp_path, monkeypatch, after_round):
    from tests.test_unattended_supervisor import _json
    cfg = prepare(tmp_path, monkeypatch)
    arm = cfg['arms'][0]
    state = Path(arm['run_output_root']) / 'execution-state.json'
    _json(state, {'status': 'RUNNING', 'next_round': 153, 'final_panel_access_count': 0})
    attempted = []

    def fail(*args):
        attempted.append(1)
        _json(state, {'status': 'RUNNING', 'next_round': after_round, 'final_panel_access_count': 0})
        raise OSError('fixture: round router changed without a sealed round')

    monkeypatch.setattr(us, 'start', fail)
    us.run(cfg, execute=True, max_cycles=3, sleep_fn=lambda _: None)
    assert attempted == [1]
    assert us.discover(arm)['action'] == 'BLOCKED_NEEDS_HUMAN'


@pytest.mark.parametrize('matching_intent', [True, False])
def test_legacy_terminal_uses_only_its_bound_dispatch_intent(tmp_path, monkeypatch, matching_intent):
    from tests.test_unattended_supervisor import _json
    from tests.test_execution_runtime_round_transaction_20260916 import write_complete_prefix
    cfg = prepare(tmp_path, monkeypatch)
    arm = cfg['arms'][0]
    pre_recovery = us.recovery_identity(arm)
    before = {'pre_state_sha256': None, 'pre_git_head': 'a' * 40,
              'pre_recovery_sha256': pre_recovery['sha256']}
    failure = us.dispatch_failure(cfg, arm, cfg['lanes'][0], before, OSError('legacy terminal'))
    intent_path = (Path(cfg['state_root']) / 'dispatches' / arm['id']
                   / failure['dispatch_id'] / 'intent.json')
    _json(intent_path, {'pre_git_head': 'a' * 40,
                       'pre_recovery_sha256': pre_recovery['sha256'] if matching_intent else 'b' * 64,
                       'pre_recovery_projection': pre_recovery['projection']})
    root = Path(arm['run_output_root'])
    _json(root / 'execution-state.json', {'status': 'RUNNING', 'next_round': 152})
    write_complete_prefix(root / 'round-152', 'rollout', round_index=152)
    result = us.discover(arm)
    assert result['action'] == ('READY_TO_DISPATCH' if matching_intent else 'BLOCKED_NEEDS_HUMAN')
    if matching_intent:
        assert 'rollout' in result['recovery_plan']['must_not_replay']
