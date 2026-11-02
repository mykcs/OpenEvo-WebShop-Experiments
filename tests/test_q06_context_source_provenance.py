"""CPU-only regressions: synthetic inputs are never scientific evidence."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.openevo_webshop import analyze_q06_context_attempts as context
from scripts.openevo_webshop import analyze_q06_paired_attempts as paired
from scripts.openevo_webshop import plot_q06_context_summary as plot


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'docs/evidence/advisor-q06-e2-paired-analysis-20261001'
SOURCE_RECORD = EVIDENCE / 'Q06_GDR_CONTEXT_SOURCE_RECONCILIATION_202610020300.json'


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize('script', ['analyze_q06_context_attempts.py', 'plot_q06_context_summary.py'])
def test_direct_script_help_works_without_pythonpath(script, tmp_path):
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/openevo_webshop' / script), '--help'],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert '--source-record' in result.stdout


@pytest.fixture
def sealed_inputs(tmp_path, monkeypatch):
    # Resampling is irrelevant to provenance; keep synthetic tests cheap.
    monkeypatch.setattr(paired, 'BOOTSTRAPS', 16)
    record = {'schema': context.SOURCE_RECORD_SCHEMA, 'sources': {}}
    args = []
    paths = {}
    for arm in context.EXPECTED:
        rows = [{
            'task_identity': f'synthetic-task-{i}',
            'worker_seed': i,
            'paired_generation_seed': i + 64,
            'engineering_invalid': False,
            'adapter_sha256': context.EXPECTED[arm],
            'note': 'Unicode separators stay inside this JSON string: \u2028\u0085',
            'episode': {'score': i / 64, 'won': False, 'trace': []},
        } for i in range(64)]
        summary = {key: 'synthetic-' + key for key in context.IDENTITIES}
        summary.update({
            'status': 'PASS', 'task_count': 64,
            'external_teacher_calls': 0, 'final_panel_access': 0,
            'checkpoint_manifest_sha256': (
                'edcaaf2b216e3f8713574105e52037675ef3ea091b81892d557dda4af2f9c33c'
                if arm == 'gdr' else 'a' * 64
            ),
        })
        payloads = {
            'attempts': ('\r\n'.join(json.dumps(row, ensure_ascii=False)
                                      for row in rows) + '\r\n').encode(),
            'summary': (json.dumps(summary) + '\n').encode(),
        }
        record['sources'][arm] = {}
        for kind, payload in payloads.items():
            path = tmp_path / f'{arm}-{kind}'
            path.write_bytes(payload)
            paths[(arm, kind)] = path
            record['sources'][arm][kind] = {'recorded_sha256': digest(payload)}
            args += ['--' + arm + ('-summary' if kind == 'summary' else ''), str(path)]
    record_path = tmp_path / 'record.json'
    record_path.write_text(json.dumps(record))
    args += ['--source-record', str(record_path)]
    return args, paths, record_path


def produce(sealed_inputs, tmp_path):
    args, _, _ = sealed_inputs
    output = tmp_path / 'analysis.json'
    context.main(args + ['--output', str(output)])
    return json.loads(output.read_bytes())


def test_analysis_hashes_exact_parsed_bytes_including_crlf_and_unicode(sealed_inputs, tmp_path):
    data = produce(sealed_inputs, tmp_path)
    _, paths, record_path = sealed_inputs
    record, record_sha = context.read_source_record(record_path)
    context.validate_source_provenance(data, record, record_sha)
    assert data['schema'] == 'openevo.q06.three-endpoint-context-analysis.v2'
    for (arm, kind), path in paths.items():
        assert data['sources'][arm][kind + '_sha256'] == digest(path.read_bytes())
        assert data['sources'][arm][kind + '_bytes'] == len(path.read_bytes())
    assert data['endpoints']['gdr']['tasks'] == 64


@pytest.mark.parametrize('kind', ['attempts', 'summary'])
@pytest.mark.parametrize('payload', [b'', b' \n\t', b'{"nonempty": "wrong source"}\n'])
def test_bad_raw_bytes_never_publish_pass(sealed_inputs, tmp_path, kind, payload):
    args, paths, _ = sealed_inputs
    paths[('gdr', kind)].write_bytes(payload)
    output = tmp_path / 'rejected.json'
    with pytest.raises(ValueError, match='Empty source|Source digest mismatch'):
        context.main(args + ['--output', str(output)])
    assert not output.exists()


def test_missing_raw_file_never_publishes_pass(sealed_inputs, tmp_path):
    args, paths, _ = sealed_inputs
    paths[('beta', 'summary')].rename(tmp_path / 'unavailable-summary')
    output = tmp_path / 'rejected.json'
    with pytest.raises(FileNotFoundError):
        context.main(args + ['--output', str(output)])
    assert not output.exists()


@pytest.mark.parametrize('replacement', [b'', b'{"changed": true}\n'])
def test_input_change_after_parse_is_rejected(sealed_inputs, tmp_path, monkeypatch, replacement):
    args, paths, _ = sealed_inputs
    original_ci = context.ci

    def change_source(values, name):
        paths[('bounded', 'attempts')].write_bytes(replacement)
        return original_ci(values, name)

    monkeypatch.setattr(context, 'ci', change_source)
    output = tmp_path / 'rejected.json'
    with pytest.raises(ValueError, match='Empty source|Source digest mismatch'):
        context.main(args + ['--output', str(output)])
    assert not output.exists()


def test_source_record_change_after_parse_is_rejected(sealed_inputs, tmp_path, monkeypatch):
    args, _, record_path = sealed_inputs
    original_ci = context.ci

    def change_record(values, name):
        record_path.write_bytes(record_path.read_bytes() + b'\n')
        return original_ci(values, name)

    monkeypatch.setattr(context, 'ci', change_record)
    output = tmp_path / 'rejected.json'
    with pytest.raises(ValueError, match='Source record changed'):
        context.main(args + ['--output', str(output)])
    assert not output.exists()


@pytest.mark.parametrize('bad_digest', [None, '', context.EMPTY_SHA256, 'a' * 63, 'G' * 64])
def test_invalid_expected_digests_are_rejected(sealed_inputs, bad_digest):
    _, _, record_path = sealed_inputs
    record = json.loads(record_path.read_bytes())
    record['sources']['beta']['summary']['recorded_sha256'] = bad_digest
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='recorded source digest'):
        context.read_source_record(record_path)


@pytest.mark.parametrize('mutation', [
    'empty_digest', 'nonempty_mismatch', 'missing_source', 'zero_size',
    'missing_proof', 'unverified_proof', 'wrong_record', 'blocked_status', 'wrong_schema',
])
def test_plot_rejects_unverified_or_mismatched_analysis_before_writing(
    sealed_inputs, tmp_path, mutation,
):
    data = produce(sealed_inputs, tmp_path)
    if mutation == 'empty_digest':
        data['sources']['gdr']['attempts_sha256'] = context.EMPTY_SHA256
    elif mutation == 'nonempty_mismatch':
        data['sources']['gdr']['summary_sha256'] = 'c' * 64
    elif mutation == 'missing_source':
        del data['sources']['beta']
    elif mutation == 'zero_size':
        data['sources']['bounded']['summary_bytes'] = 0
    elif mutation == 'missing_proof':
        del data['source_provenance']
    elif mutation == 'unverified_proof':
        data['source_provenance']['raw_source_bytes_verified'] = False
    elif mutation == 'wrong_record':
        data['source_provenance']['source_record_sha256'] = 'd' * 64
    elif mutation == 'wrong_schema':
        data['schema'] = 'unrelated.v1'
    else:
        data['status'] = 'BLOCKED_SOURCE_PROVENANCE'
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps(data))
    output_dir = tmp_path / 'plots'
    output_dir.mkdir()
    with pytest.raises(ValueError, match='provenance'):
        plot.main(['--analysis', str(bad), '--source-record', str(sealed_inputs[2]),
                   '--output-directory', str(output_dir)])
    assert list(output_dir.iterdir()) == []


def test_read_only_revalidation_does_not_resample_or_write(sealed_inputs, tmp_path, monkeypatch, capsys):
    args, _, _ = sealed_inputs
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    def forbidden(*args):
        raise AssertionError('source-only audit must not resample')

    monkeypatch.setattr(context, 'ci', forbidden)
    context.main(args + ['--verify-sources-only'])
    audit = json.loads(capsys.readouterr().out)
    assert audit['status'] == 'PASS_SOURCE_REVALIDATION'
    assert audit['source_provenance']['raw_source_bytes_verified'] is True
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_output_remains_create_only(sealed_inputs, tmp_path):
    args, _, _ = sealed_inputs
    output = tmp_path / 'existing.json'
    output.write_bytes(b'preserve concurrent work\n')
    with pytest.raises(FileExistsError):
        context.main(args + ['--output', str(output)])
    assert output.read_bytes() == b'preserve concurrent work\n'


def test_historical_bad_proof_is_preserved_and_superseded_not_silently_repaired():
    record, record_sha = context.read_source_record(SOURCE_RECORD)
    bad_path = EVIDENCE / 'Q06_GDR_CONTEXT_ANALYSIS_20261001.json'
    original = bad_path.read_bytes()
    historical = json.loads(original)
    assert digest(original) == record['supersedes']['analysis_sha256']
    assert record['status'] == 'BLOCKED_SOURCE_PROVENANCE'
    assert record['raw_source_bytes_verified'] is False
    assert record['numerical_estimates_recomputed'] is False
    for arm in context.EXPECTED:
        for kind in ('attempts', 'summary'):
            assert historical['sources'][arm][kind + '_sha256'] == context.EMPTY_SHA256
            assert record['sources'][arm][kind]['raw_bytes_verified'] is False
    with pytest.raises(ValueError, match='provenance'):
        context.validate_source_provenance(historical, record, record_sha)
    for evidence in record['documentary_evidence'].values():
        assert digest((ROOT / evidence['path']).read_bytes()) == evidence['sha256']
    recorded_pair = json.loads((EVIDENCE / 'Q06_E2_PAIRED_ANALYSIS.json').read_bytes())
    for arm in ('bounded', 'beta'):
        for kind in ('attempts', 'summary'):
            key = arm if kind == 'attempts' else arm + '_summary'
            assert record['sources'][arm][kind]['recorded_sha256'] == recorded_pair['input_files'][key]['sha256']
            assert record['sources'][arm][kind]['canonical_path'] == recorded_pair['canonical_server_sources'][arm + '_' + kind]
    narrative = (EVIDENCE / 'Q06_GDR_CONTEXT_ANALYSIS_20261001.md').read_text()
    for kind, filename in (('attempts', 'attempts.jsonl'), ('summary', 'summary.json')):
        assert f'`{filename}` SHA-256: `{record["sources"]["gdr"][kind]["recorded_sha256"]}`' in narrative
    plot_receipt = json.loads((EVIDENCE / 'Q06_GDR_CONTEXT_PLOT_PROVENANCE.json').read_bytes())
    assert plot_receipt['analysis_sha256'] == digest(original)
    assert record['historical_plots']['status'] == 'PRESERVED_NOT_SOURCE_REVALIDATED'
    assert record['historical_plots']['provenance_sha256'] == digest(
        (EVIDENCE / 'Q06_GDR_CONTEXT_PLOT_PROVENANCE.json').read_bytes()
    )
