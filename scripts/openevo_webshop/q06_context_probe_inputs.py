#!/usr/bin/env python3
"""Validate Q06's remaining context inputs without starting inference."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

MANIFEST_REL = Path('docs/evidence/advisor-q06-e2-paired-analysis-20261001/Q06_GDR_CONTEXT_PROBE_MANIFEST.json')
SOURCE_SHA = '0e1ea4bbb62215be71a68f206eabf442c0227f58'
CHECKPOINT = 'historical_gdr_r159_rank128'
MODEL_WEIGHTS_SHA = '69d9120b7c1f7583575487db115360fad14f24b08e9a5dca9712c9992c99cdf5'
MODEL_RECEIPT_SHA = '324ed4f5a6dd5d6a664ad8f4e1cb340e439394d3b7dd7dbe047f313d6815502b'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(authority_repo, source_repo, expected_manifest_sha):
    authority_repo, source_repo = Path(authority_repo), Path(source_repo)
    manifest_path = authority_repo / MANIFEST_REL
    if manifest_path.is_symlink() or sha(manifest_path) != expected_manifest_sha:
        raise ValueError('Q06 context manifest identity drift')
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('schema') != 'openevo.q06.gdr-context-probe-manifest.v1'
            or manifest.get('experiment_id') != 'Q06_TOKEN_CLICK_ENTROPY'
            or manifest.get('formal_probe_enabled') is not False
            or manifest.get('final_panel_access_allowed') is not False
            or manifest.get('runtime_execution_sha') != SOURCE_SHA
            or manifest.get('budget') != {
                'new_episodes': 64, 'fixed_passes': 1, 'adaptive_reruns': 0,
                'external_teacher_calls': 0, 'training_rollouts': 0,
                'optimizer_updates': 0, 'final_panel_access': 0}
            or set(manifest.get('checkpoints', {})) != {CHECKPOINT}
            or manifest.get('sampling') != {
                'max_new_tokens': 512, 'max_steps': 15, 'repetition_penalty': 1.0,
                'step_seed_policy': 'paired_generation_seed + step_index * 100000',
                'temperature': 1.0, 'top_k': 0, 'top_p': 1.0}):
        raise ValueError('Q06 frozen context scope, sampling or budget drift')
    identity_ref = manifest['context_identity']
    identity_path = authority_repo / identity_ref['path']
    if identity_path.is_symlink() or sha(identity_path) != identity_ref['sha256']:
        raise ValueError('Q06 exact context receipt drift')
    identity = json.loads(identity_path.read_text())
    endpoint = manifest['endpoint']
    if (identity['checkpoint_id'] != CHECKPOINT or endpoint['checkpoint_id'] != CHECKPOINT
            or endpoint['source_files'] != identity['files']
            or manifest['checkpoints'][CHECKPOINT] != {
                'adapter_sha256': identity['files']['adapter']['sha256'],
                'state_id': identity['state_id']}):
        raise ValueError('Q06 context endpoint differs from its exact identity receipt')
    for name, ref in identity['files'].items():
        path = Path(ref['path'])
        if (path.is_symlink() or not path.is_file()
                or path.stat().st_size != ref['bytes'] or sha(path) != ref['sha256']):
            raise ValueError('Q06 context source drift: ' + name)
    original = json.loads(Path(identity['files']['e2_manifest']['path']).read_text())
    for key in ('common_carriers', 'model', 'runtime_files', 'seed_root',
                'webshop_dataset', 'openevo_core', 'execution_image', 'sampling'):
        if manifest[key] != original[key]:
            raise ValueError('Q06 context differs from the frozen evaluator inputs: ' + key)
    model_root = Path(manifest['model']['local_model_path'])
    model_receipt_path = model_root / 'verification-receipt.json'
    for path, expected in ((model_root / 'model.safetensors', MODEL_WEIGHTS_SHA),
                           (model_receipt_path, MODEL_RECEIPT_SHA)):
        if path.is_symlink() or not path.is_file() or sha(path) != expected:
            raise ValueError('Q06 frozen merged model payload drift: ' + path.name)
    model_receipt = json.loads(model_receipt_path.read_text())
    if (model_receipt.get('status') != 'PASS'
            or model_receipt.get('base_revision') != manifest['model']['revision']
            or not any(row.get('name') == 'model.safetensors'
                       and row.get('sha256') == MODEL_WEIGHTS_SHA
                       for row in model_receipt.get('merged_files', []))):
        raise ValueError('Q06 frozen merged model verification receipt drift')
    if Path(endpoint['adapter_path']) != Path(identity['files']['adapter']['path']).parent:
        raise ValueError('Q06 adapter directory differs from its sealed payload')
    for name, key in (('memory', 'memory'), ('skill', 'skill'),
                      ('agent_system', 'agent_system')):
        ref = identity['files']['common_' + name]
        if (manifest['common_carriers'][key + '_path'] != ref['path']
                or manifest['common_carriers'][key + '_sha256'] != ref['sha256']):
            raise ValueError('Q06 common carrier differs from its identity receipt: ' + name)
    seed = manifest['seed_root']
    seed_root = Path(seed['path'])
    if len(seed['package_files_sha256']) != seed['package_file_count']:
        raise ValueError('Q06 seed package file count drift')
    for relative, expected in seed['package_files_sha256'].items():
        path = seed_root / relative
        if path.is_symlink() or sha(path) != expected:
            raise ValueError('Q06 seed package payload drift: ' + relative)
    for name, expected in seed['webshop_data_files_sha256'].items():
        path = Path(seed['data_dir']) / name
        if path.is_symlink() or sha(path) != expected:
            raise ValueError('Q06 WebShop input data drift: ' + name)
    prereg = manifest['preregistration']
    blob = subprocess.check_output([
        'git', '-C', str(authority_repo), 'show', prereg['git_sha'] + ':' + prereg['path']])
    if hashlib.sha256(blob).hexdigest() != prereg['sha256']:
        raise ValueError('Q06 frozen preregistration blob drift')
    head = subprocess.check_output(['git', '-C', str(source_repo), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source_repo), 'status', '--porcelain'], text=True).strip()
    if head != SOURCE_SHA or dirty:
        raise ValueError('Q06 pinned probe source must remain clean and exact')
    spec = importlib.util.spec_from_file_location(
        'q06_frozen_probe', source_repo / 'scripts/openevo_webshop/q17_fixed_checkpoint_probe.py')
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    probe.validate_runtime_files(manifest, source_repo)
    panel = source_repo / manifest['task_panel_manifest_path']
    probe.validate_full_panel_manifest(panel, json.loads(panel.read_text()))
    return manifest


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--authority-repo', type=Path, required=True)
    parser.add_argument('--source-repo', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    validate_inputs(args.authority_repo, args.source_repo, args.manifest_sha256)
    print(json.dumps({'status': 'PASS_READ_ONLY_INPUTS', 'checkpoint_id': CHECKPOINT,
                      'panel_tasks': 64, 'new_model_calls': 0,
                      'new_environment_steps': 0, 'launch_authority_granted': False}))


if __name__ == '__main__':
    main()
