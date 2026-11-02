#!/usr/bin/env python3
"""Render frozen Q07 evidence; performs no model or environment calls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summaries', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    dest = repo / 'docs/evidence/advisor-q07-task-vector-causal-20261001'
    manifest_path = repo / 'configs/experiment/manifests/advisor-q07-task-vector-causal-20261001.json'
    manifest = json.loads(manifest_path.read_text())
    analysis_path = dest / 'Q07_CAUSAL_ANALYSIS.json'
    analysis = json.loads(analysis_path.read_text())
    tasks = {t['task_identity']: t for t in manifest['tasks']}
    order = sorted(tasks, key=lambda k: tasks[k]['probe_ordinal'])
    assert len(tasks) == 64 and analysis['panel_sha256'] == manifest['panel_sha256']
    ids = ['intact_w159', 'r151_reference'] + [f'{a}_r{r}' for a in ('reverse', 'loo') for r in range(152, 160)]
    scores, rows, sources = {}, [], []
    for cid in ids:
        path = args.summaries / cid / 'summary.json'
        z = json.loads(path.read_text())
        assert z['status'] == 'PASS' and z['task_count'] == 64
        assert z['panel_sha256'] == manifest['panel_sha256'] and z['final_panel_access'] == 0
        assert z['runtime_execution_sha'] == manifest['runtime_execution_sha']
        by_task = {r['task_identity']: r for r in z['rows']}
        assert len(z['rows']) == 64 and set(by_task) == set(tasks)
        scores[cid] = []
        for key in order:
            row, task = by_task[key], tasks[key]
            assert row['engineering_invalid'] is False and row['error'] is None
            assert row['adapter_sha256'] == manifest['checkpoints'][cid]['adapter_sha256']
            for field in ('probe_ordinal', 'paired_generation_seed', 'source_session_index', 'worker_seed'):
                assert row[field] == task[field], (cid, key, field)
            episode = row['episode']
            score = float(episode['score'])
            assert 0 <= score <= 1
            scores[cid].append(score)
            rows.append({'state': cid, 'probe_ordinal': task['probe_ordinal'], 'task_identity': key,
                         'category': task['category'], 'paired_generation_seed': task['paired_generation_seed'],
                         'score': score, 'exact_success': int(episode['won']), 'steps': episode['steps'],
                         'adapter_sha256': row['adapter_sha256'], 'source_summary_sha256': sha(path)})
        sources.append({'state': cid, 'path': f'/workspace/maintenance/ADVISOR-Q07-TASK-VECTOR-CAUSAL-20261001/results/{cid}/summary.json',
                        'sha256': sha(path), 'episodes': 64, 'accounting': 'reused' if cid == 'intact_w159' else 'new'})
    for family, prefix in [('primary_family', 'reverse'), ('secondary_family', 'loo')]:
        for effect in analysis[family]:
            values = scores[f'{prefix}_r{effect["round_index"]}']
            delta = sum(a - b for a, b in zip(values, scores['intact_w159'])) / 64
            assert abs(delta - effect['mean_task_score_delta']) < 1e-12
    matrix = dest / 'Q07_PER_TASK_MATRIX.csv'
    with matrix.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    plt.rcParams.update({'svg.hashsalt': 'q07-frozen-results', 'font.size': 9})
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True, sharex=True)
    for ax, family, title in zip(axes, ('primary_family', 'secondary_family'), ('Reverse vs intact (primary)', 'Remove vs intact (secondary)')):
        for effect in analysis[family]:
            value, interval = effect['mean_task_score_delta'], effect['ci95']
            ax.errorbar(value, effect['round_index'], xerr=[[value - interval[0]], [interval[1] - value]],
                        fmt='o', color='#b23a36' if effect['holm_reject_0_05'] else '#35688b', capsize=3)
        ax.axvline(0, color='grey', linewidth=1)
        ax.set(title=title, xlabel='Mean Task Score difference (95% paired CI)')
        ax.set_yticks(range(152, 160))
        ax.grid(axis='x', alpha=.2)
    axes[0].set_ylabel('Update round')
    fig.suptitle('Q07: 64-task development panel; separate Holm correction per family')
    fig.legend(handles=[Line2D([], [], color='#b23a36', marker='o', linestyle='none', label='Holm-adjusted p <= 0.05'),
                        Line2D([], [], color='#35688b', marker='o', linestyle='none', label='Holm-adjusted p > 0.05')],
               loc='lower center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(dest / 'Q07_CAUSAL_EFFECTS.svg', metadata={'Date': None})
    fig.savefig(dest / 'Q07_CAUSAL_EFFECTS.png', dpi=180, metadata={'Software': 'Q07 frozen evidence renderer'})
    plt.close(fig)
    plotted = ids[2:]
    differences = np.asarray([np.asarray(scores[cid]) - np.asarray(scores['intact_w159']) for cid in plotted])
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(differences, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
    ax.set_yticks(range(len(plotted)), plotted)
    ax.set(xlabel='Frozen task ordinal (0–63)', title='Descriptive per-task score difference vs intact; no task selection')
    fig.colorbar(im, ax=ax, label='Task Score difference')
    fig.tight_layout()
    fig.savefig(dest / 'Q07_PER_TASK_HEATMAP.svg', metadata={'Date': None})
    fig.savefig(dest / 'Q07_PER_TASK_HEATMAP.png', dpi=180, metadata={'Software': 'Q07 frozen evidence renderer'})
    plt.close(fig)
    provenance = {'schema': 'openevo.advisor.q07.derived-deliverables.v1', 'source_summaries': sources,
                  'manifest_sha256': sha(manifest_path), 'analysis_sha256': sha(analysis_path),
                  'renderer_sha256': sha(Path(__file__)), 'matplotlib_version': matplotlib.__version__,
                  'numpy_version': np.__version__,
                  'matrix_rows': len(rows), 'new_episodes': 1088, 'reused_episodes': 64,
                  'additional_model_calls': 0, 'additional_environment_calls': 0, 'final_panel_access': 0,
                  'claim_boundary': 'Development-panel evidence; heatmap descriptive, no controller selection.'}
    (dest / 'Q07_DERIVED_DELIVERABLES.json').write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    files = sorted(p for p in dest.iterdir() if p.is_file() and p.name != 'SHA256SUMS')
    (dest / 'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in files))
    print(json.dumps({'matrix_rows': len(rows), 'source_count': len(sources), 'additional_calls': 0}))


if __name__ == '__main__':
    main()
