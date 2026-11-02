#!/usr/bin/env python3
"""Read-only Q08 seen-cohort evaluation using the frozen Q17 episode owner."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

PANEL_SHA = '0780dbe68dcc1cafb013a8765f2ecb4b91bfd2a0d92ac3ac9fe4fde7453fccc3'
MATRIX_SHA = '93ad7718a07633e38e15197f43ea21649fac84abc650aa29fad7f3107415f89d'
SEED_ROOT = '/workspace/runs/phase-h1.20-action-admissibility-diagnostic-v1-20260815/seed-roots/base-P1'
RUNTIME_ROOT = '/workspace/maintenance/Q17-VALIDATION-CURVE-202609231600/runtime-ac130148'
CORE_ROOT = '/workspace/maintenance/Q17-VALIDATION-CURVE-202609231600/core-594789'
CORE_SHA = '59478929d884ca794cce2cb49cc6cc16103d6abe'
RUNTIME_SHA = 'ac130148ee08b6462d9728e482a858fe5f514047'
EPISODE_OWNER_ROOT = '/workspace/maintenance/Q17-VALIDATION-CURVE-202609231600/repo'
EPISODE_OWNER_SHA = 'e79383fcbe8b5b8c26b86ccc878bf8363c684ea708127dae0614520348ebd571'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require_file(path: Path, expected: str) -> None:
    if digest(path) != expected:
        raise RuntimeError(f'file identity mismatch: {path}')


def prepare(args):
    require_file(args.panel_manifest, PANEL_SHA)
    require_file(args.checkpoint_matrix, MATRIX_SHA)
    if str(args.seed_root.resolve()) != SEED_ROOT:
        raise RuntimeError('frozen seed-root path mismatch')
    panel = json.loads(args.panel_manifest.read_text())
    matrix = json.loads(args.checkpoint_matrix.read_text())
    if panel['status'] != 'SEEN_PANEL_IDENTITIES_VERIFIED':
        raise RuntimeError('seen identities must be verified')
    if panel['final_panel_access_allowed'] is not False or panel['formal_training_allowed'] is not False:
        raise RuntimeError('read-only development authority required')
    tasks = panel['tasks']
    if len(tasks) != 96 or [t['probe_ordinal'] for t in tasks] != list(range(96)):
        raise RuntimeError('complete ordered 96-task panel required')
    if len({t['task_identity'] for t in tasks}) != 96:
        raise RuntimeError('duplicate seen task identity')
    for cohort in ('EARLY_SEEN', 'MID_SEEN', 'LATE_SEEN'):
        if sum(t['cohort'] == cohort for t in tasks) != 32:
            raise RuntimeError('each frozen seen cohort must contain 32 tasks')
    matches = [c for c in matrix['cells'] if c['lineage'] == args.lineage and c['completed_rounds'] == args.completed_rounds]
    if len(matches) != 1 or matches[0]['status'] != 'PASS':
        raise RuntimeError('checkpoint must have an exact retained payload')
    cell = dict(matches[0])
    require_file(Path(cell['manifest_path']), cell['manifest_sha256'])
    historical_manifest = json.loads(Path(cell['manifest_path']).read_text())
    checkpoint = historical_manifest['checkpoints'][cell['historical_checkpoint_id']]
    if checkpoint['adapter_sha256'] != cell['expected_payload_sha256']:
        raise RuntimeError('historical checkpoint manifest payload identity mismatch')
    for field in ('model', 'runtime_files', 'common_carriers'):
        if historical_manifest[field] != cell[field]:
            raise RuntimeError(f'historical checkpoint manifest drift: {field}')
    cell['state_id'] = checkpoint['state_id']
    require_file(Path(cell['state_path']), cell['state_sha256'])
    adapter = Path(cell['adapter_path'])
    require_file(adapter / 'adapter_model.safetensors', cell['expected_payload_sha256'])
    require_file(adapter / 'adapter_config.json', cell['adapter_config_sha256'])
    root = args.runtime_source_root.resolve()
    if str(root) != RUNTIME_ROOT:
        raise RuntimeError('runtime source checkout identity mismatch')
    if (root / 'EXECUTION_SHA').read_text().strip() != RUNTIME_SHA:
        raise RuntimeError('historical runtime revision mismatch')
    if (Path(CORE_ROOT) / 'EXECUTION_SHA').read_text().strip() != CORE_SHA:
        raise RuntimeError('historical core revision mismatch')
    for path, sha in panel['source']['files'].items():
        relative = Path(path).relative_to(panel['source']['checkout'])
        require_file(root / relative, sha)
    for rel, sha in cell['runtime_files'].items():
        require_file(root / rel, sha)
    for path, key in ((args.memory, 'memory_sha256'), (args.skill, 'skill_sha256'), (args.agent, 'agent_system_sha256')):
        require_file(path, cell['common_carriers'][key])
    if args.model != cell['model']['local_model_path'] or args.revision != cell['model']['revision']:
        raise RuntimeError('model identity mismatch')
    owner = Path(EPISODE_OWNER_ROOT) / 'scripts/openevo_webshop/q17_fixed_checkpoint_probe.py'
    require_file(owner, EPISODE_OWNER_SHA)
    spec = importlib.util.spec_from_file_location('q08_frozen_episode_owner', owner)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.MAX_STEPS != 15 or module.STEP_SEED_STRIDE != 100000:
        raise RuntimeError('episode owner sampling drift')
    return panel, cell, module



class ConsumptionLedger:
    """Observe successful generate lengths without changing kwargs or RNG state."""
    def __init__(self, model, path):
        self.model = model
        self.original = model.generate
        self.out = path.open('x')
        self.counts = {'started_calls': 0, 'completed_calls': 0, 'failed_calls': 0, 'observed_generated_tokens': 0, 'unmeasured_completed_calls': 0}

    def snapshot(self):
        return dict(self.counts)

    def delta(self, before):
        return {k: v-before[k] for k, v in self.counts.items()}

    def install(self):
        self.model.generate = self.generate

    def generate(self, *args, **kwargs):
        self.counts['started_calls'] += 1
        try:
            result = self.original(*args, **kwargs)
        except BaseException as exc:
            self.counts['failed_calls'] += 1
            self.persist({'status': 'FAILED', 'exception_type': type(exc).__name__, 'generated_tokens_unknown': True})
            raise
        self.counts['completed_calls'] += 1
        inputs = kwargs.get('input_ids')
        if inputs is None and args:
            inputs = args[0]
        sequences = getattr(result, 'sequences', result)
        tokens = None
        if inputs is not None and hasattr(inputs, 'shape') and hasattr(sequences, 'shape') and len(sequences.shape) == 2 and sequences.shape[0] == 1:
            tokens = int(sequences.shape[-1]) - int(inputs.shape[-1])
        if tokens is None or tokens < 0:
            self.counts['unmeasured_completed_calls'] += 1
        else:
            self.counts['observed_generated_tokens'] += tokens
        self.persist({'status': 'COMPLETED', 'generated_tokens': tokens})
        return result

    def persist(self, event):
        self.out.write(json.dumps({**event, 'cumulative': self.snapshot()}, sort_keys=True) + '\n')
        self.out.flush()
        os.fsync(self.out.fileno())

    def close(self):
        self.model.generate = self.original
        self.out.close()


def main():
    p = argparse.ArgumentParser()
    for name in ('panel-manifest', 'checkpoint-matrix', 'runtime-source-root', 'seed-root', 'memory', 'skill', 'agent', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--lineage', choices=('ordinary', 'bounded', 'gdr'), required=True)
    p.add_argument('--completed-rounds', type=int, choices=(40, 80, 120, 160), required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--authority-repo', type=Path)
    p.add_argument('--launch-context', type=Path)
    p.add_argument('--cell-claim', type=Path)
    args = p.parse_args()
    panel, cell, owner = prepare(args)
    if args.validate_only:
        guard_identity = None
        if args.authority_repo is not None:
            guard_path = args.authority_repo / 'scripts/openevo_webshop/q08_seen_launch_guard.py'
            if not guard_path.is_file():
                raise RuntimeError('Q08 qualification launch guard is unavailable')
            guard_spec = importlib.util.spec_from_file_location('q08_launch_guard_identity', guard_path)
            guard = importlib.util.module_from_spec(guard_spec)
            guard_spec.loader.exec_module(guard)
            guard_identity = guard.validate_probe_identity(args)
            if not isinstance(guard_identity, dict) or guard_identity.get('status') != 'IDENTITY_PASS':
                raise RuntimeError('Q08 qualification guard identity check failed')
        print(json.dumps({'status': 'IDENTITY_PASS', 'task_count': 96, 'model_calls': 0, 'training_steps': 0, 'final_panel_access': 0, 'launch_guard_identity_validated': guard_identity is not None, 'launch_guard_identity': guard_identity}))
        return 0
    if os.environ.get('CUDA_VISIBLE_DEVICES', '').strip() != args.gpu_uuid:
        raise RuntimeError('GPU UUID mismatch')
    # An independently reviewed authority owner supplies canonical launch/claim checks.
    # The guard is unavailable until its committed preparation is reconciled.
    if any(x is None for x in (args.authority_repo, args.launch_context, args.cell_claim)):
        raise RuntimeError('canonical Q08 authority, launch context and cell claim required')
    guard_path = args.authority_repo / 'scripts/openevo_webshop/q08_seen_launch_guard.py'
    if not guard_path.is_file():
        raise RuntimeError('Q08 canonical launch guard not prepared')
    guard_spec = importlib.util.spec_from_file_location('q08_launch_guard', guard_path)
    guard = importlib.util.module_from_spec(guard_spec)
    guard_spec.loader.exec_module(guard)
    launch_binding = guard.authorize_probe(args)
    # The guard consumes a fixed cell claim; output paths cannot add budget.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path(CORE_ROOT) / 'src'))
    sys.path.insert(0, str(args.runtime_source_root / 'vendor/openevo-webshop/src'))
    modules = owner.load_science_modules(args.runtime_source_root)
    started = time.time()
    rows = []
    with (args.output_dir / 'start-receipt.json').open('x') as out:
        out.write(json.dumps({'status': 'STARTED', 'launch_binding': launch_binding, 'probe_script_sha256': digest(Path(__file__)), 'task_count': 96, 'final_panel_access': 0}, sort_keys=True, indent=2) + '\n')
        out.flush()
        os.fsync(out.fileno())
    runtime = modules['Runtime'](args.model, args.revision, sd_adapter=Path(cell['adapter_path']), memory=args.memory.read_text(), skill_bundle=args.skill.read_text(), agent_system=args.agent.read_text())
    cleanup_errors = []
    interrupted = False
    ledger = ConsumptionLedger(runtime.base._model, args.output_dir / 'generation-consumption.jsonl')
    ledger.install()
    try:
        with (args.output_dir / 'attempts.jsonl').open('x') as out:
            for task in panel['tasks']:
                env = None
                error = None
                fatal = False
                episode = None
                before = ledger.snapshot()
                env_counts = {'started': 0, 'completed': 0, 'failed': 0, 'exception_type': None}
                try:
                    env = modules['WebshopEnv'](modules['WebshopEnvConfig'](seed_root=str(args.seed_root), num_products=1000, seed=int(task['worker_seed'])))
                    original_step = env.step
                    def counted_step(*step_args, **step_kwargs):
                        env_counts['started'] += 1
                        try:
                            result = original_step(*step_args, **step_kwargs)
                        except BaseException as exc:
                            env_counts['failed'] += 1
                            env_counts['exception_type'] = type(exc).__name__
                            raise
                        env_counts['completed'] += 1
                        return result
                    env.step = counted_step
                    episode = owner.run_episode(modules, modules['FixedEnvironmentSeed'](env, int(task['worker_seed'])), runtime, None, task, int(task['paired_generation_seed']))
                    if episode.get('environment_action_schema_violation'):
                        error = 'EnvironmentStepException: reported by frozen episode owner'
                        fatal = env_counts['exception_type'] in ('KeyboardInterrupt', 'SystemExit', 'GeneratorExit')
                except BaseException as exc:
                    fatal = not isinstance(exc, Exception)
                    error = f'{type(exc).__name__}: {exc}'
                finally:
                    if env is not None:
                        try:
                            env.close()
                        except BaseException as exc:
                            fatal = fatal or not isinstance(exc, Exception)
                            error = (error + '; ' if error else '') + f'EnvironmentCloseException: {type(exc).__name__}: {exc}'
                row = {**task, 'checkpoint_id': f"{args.lineage}_r{args.completed_rounds - 1:03d}", 'state_id': cell['state_id'], 'state_file_sha256': cell['state_sha256'], 'adapter_sha256': cell['expected_payload_sha256'], 'instrumentation': 'off', 'engineering_invalid': error is not None, 'error': error, 'episode': episode, 'consumption': ledger.delta(before), 'environment_action_steps': env_counts}
                rows.append(row)
                out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
                out.flush()
                os.fsync(out.fileno())
                print(json.dumps({'completed_tasks': len(rows), 'engineering_invalid': error is not None}), flush=True)
                if fatal:
                    interrupted = True
                    break
    finally:
        try:
            ledger.close()
        except BaseException as exc:
            cleanup_errors.append(f'LedgerClose: {type(exc).__name__}: {exc}')
        finally:
            try:
                runtime.close()
            except BaseException as exc:
                cleanup_errors.append(f'RuntimeClose: {type(exc).__name__}: {exc}')
    valid = [r for r in rows if not r['engineering_invalid']]
    summary = {'schema': 'openevo.q08.seen-checkpoint-probe.v1', 'status': 'PASS' if len(valid) == 96 and not cleanup_errors and not interrupted else 'FAIL', 'cleanup_errors': cleanup_errors, 'interrupted': interrupted, 'q17_scoring_observer_enabled': False, 'consumption_ledger_enabled': True, 'lineage': args.lineage, 'completed_rounds': args.completed_rounds, 'actual_sd_generations': cell['actual_sd_generations'], 'checkpoint': cell, 'panel_manifest_sha256': PANEL_SHA, 'checkpoint_matrix_sha256': MATRIX_SHA, 'probe_script_sha256': digest(Path(__file__)), 'episode_owner_sha256': EPISODE_OWNER_SHA, 'runtime_execution_sha': RUNTIME_SHA, 'core_execution_sha': CORE_SHA, 'image_digest': panel['image'], 'gpu_uuid': args.gpu_uuid, 'sampling': panel['sampling'], 'launch_binding': launch_binding, 'generation_consumption': ledger.snapshot(), 'environment_action_steps': {key: sum(r['environment_action_steps'][key] for r in rows) for key in ('started', 'completed', 'failed')}, 'generation_consumption_sha256': digest(args.output_dir / 'generation-consumption.jsonl'), 'attempted_tasks': len(rows), 'valid_tasks': len(valid), 'engineering_invalid_tasks': len(rows)-len(valid), 'training_steps': 0, 'external_teacher_calls': 0, 'final_panel_access': 0, 'wall_time_seconds': time.time()-started, 'cohort_metrics': {cohort: owner.summarize_rows([r for r in valid if r['cohort'] == cohort], False) for cohort in ('EARLY_SEEN', 'MID_SEEN', 'LATE_SEEN')}, 'attempts_sha256': digest(args.output_dir / 'attempts.jsonl')}
    with (args.output_dir / 'summary.json').open('x') as out:
        out.write(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps({'status': summary['status'], 'valid_tasks': len(valid)}), flush=True)
    return 0 if summary['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
