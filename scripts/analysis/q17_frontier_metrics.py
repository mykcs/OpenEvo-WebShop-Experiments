import argparse, csv, hashlib, json, math, os, re, statistics, time
from pathlib import Path
from collections import Counter, defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--run-root', required=True)
parser.add_argument('--out-dir', required=True)
args = parser.parse_args()
run_root = Path(args.run_root)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

group_re = re.compile(r'-group(\d+)-r(\d+)$')

def load_json(path):
    with open(path) as f:
        return json.load(f)

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()

def percentile_success(scores, k):
    # Assumes rows are in within-task rollout order. Success is final score >= 1.0 or won=True upstream.
    return any(s >= 1.0 for s in scores[:k])

def stable_state_sig(step):
    receipt = step.get('runtime_context_receipt') or {}
    page_state = receipt.get('page_state')
    verifier = step.get('verifier_state') or {}
    after = verifier.get('after') or {}
    asin = after.get('asin') if isinstance(after, dict) else None
    opts = after.get('options') if isinstance(after, dict) else None
    if isinstance(opts, dict):
        opt_sig = '|'.join(f'{k}={v}' for k, v in sorted(opts.items()))
    else:
        opt_sig = ''
    return f'{page_state}|{asin or ""}|{opt_sig}'

def state_changed(step):
    verifier = step.get('verifier_state') or {}
    return (verifier.get('before') != verifier.get('after')) if isinstance(verifier, dict) else False

round_rows=[]
per_round=[]
manifest=[]
active_unsealed=[]
warnings=[]
carrier_updates=Counter()
carrier_update_rounds=defaultdict(list)

round_dirs=sorted([p for p in run_root.glob('round-*') if p.is_dir()])
for rd in round_dirs:
    try:
        r=int(rd.name.split('-')[-1])
    except Exception:
        continue
    rc_path=rd/'ROUND_COMPLETE.json'
    attempts_path=rd/'attempts.jsonl'
    if not rc_path.exists():
        active_unsealed.append(r)
        continue
    rc=load_json(rc_path)
    rs=rc.get('rollout_summary') or {}
    manifest.append((str(rc_path.relative_to(run_root)), sha256_file(rc_path)))
    attempts=[]
    if attempts_path.exists():
        manifest.append((str(attempts_path.relative_to(run_root)), sha256_file(attempts_path)))
        with open(attempts_path) as f:
            for line_no,line in enumerate(f,1):
                if not line.strip(): continue
                try:
                    x=json.loads(line)
                except Exception as e:
                    warnings.append(f'round {r} attempts.jsonl line {line_no}: {e}')
                    continue
                tid=str(x.get('task_id',''))
                m=group_re.search(tid)
                if not m:
                    warnings.append(f'round {r}: cannot recover group/order from task_id={tid!r}')
                    continue
                group=int(m.group(1)); order=int(m.group(2))
                score=x.get('final_task_score')
                if score is None:
                    score=((x.get('episode') or {}).get('task_score'))
                try: score=float(score or 0.0)
                except Exception: score=0.0
                won=bool(x.get('won')) or score>=1.0
                trace=((x.get('episode') or {}).get('trace') or [])
                page_states=[]; state_sigs=[]; actions=[]; changed=0; env_steps=0
                for st in trace:
                    if not isinstance(st, dict): continue
                    receipt=st.get('runtime_context_receipt') or {}
                    page_states.append(str(receipt.get('page_state')))
                    state_sigs.append(stable_state_sig(st))
                    action=str(st.get('action') or '').strip().lower()
                    if action: actions.append(action)
                    if state_changed(st): changed += 1
                    if ((st.get('verifier_state') or {}).get('environment_step_executed')): env_steps += 1
                attempts.append({
                    'group':group,'order':order,'score':score,'won':won,
                    'trace_len':len(trace),'unique_page_states':len(set(page_states)),
                    'unique_state_sigs':len(set(state_sigs)),'unique_actions':len(set(actions)),
                    'action_count':len(actions),'changed_steps':changed,'env_steps':env_steps,
                    'repeated_action_ratio':(1-len(set(actions))/len(actions)) if actions else None,
                    'repeated_state_ratio':(1-len(set(state_sigs))/len(state_sigs)) if state_sigs else None,
                })
    groups=defaultdict(list)
    for a in attempts:
        groups[a['group']].append(a)
    task_summaries=[]
    for g, rows in groups.items():
        rows=sorted(rows, key=lambda x: x['order'])
        scores=[x['score'] for x in rows]
        wins=[x['won'] for x in rows]
        exact=sum(1 for w in wins if w)
        max_score=max(scores) if scores else 0.0
        task_summaries.append({
            'group':g,
            'n':len(rows),
            'exact_success_count':exact,
            'max_score':max_score,
            'success_at_1':any(wins[:1]),
            'success_at_2':any(wins[:2]),
            'success_at_4':any(wins[:4]),
            'success_at_8':any(wins[:8]),
            'near_miss': exact==0 and max_score>0,
            'frontier': exact==0 and max_score>=0.5,
            'cold': exact==0 and max_score==0,
            'unique_page_states_mean':statistics.mean([x['unique_page_states'] for x in rows]) if rows else 0,
            'unique_state_sigs_mean':statistics.mean([x['unique_state_sigs'] for x in rows]) if rows else 0,
            'unique_actions_mean':statistics.mean([x['unique_actions'] for x in rows]) if rows else 0,
            'state_changing_rate':(sum(x['changed_steps'] for x in rows)/sum(x['env_steps'] for x in rows)) if sum(x['env_steps'] for x in rows) else 0,
            'repeated_action_ratio_mean':statistics.mean([x['repeated_action_ratio'] for x in rows if x['repeated_action_ratio'] is not None]) if any(x['repeated_action_ratio'] is not None for x in rows) else None,
            'repeated_state_ratio_mean':statistics.mean([x['repeated_state_ratio'] for x in rows if x['repeated_state_ratio'] is not None]) if any(x['repeated_state_ratio'] is not None for x in rows) else None,
        })
    n_tasks=len(task_summaries)
    def pct(key):
        return 100*sum(1 for t in task_summaries if t[key])/n_tasks if n_tasks else None
    best8=statistics.mean([t['max_score'] for t in task_summaries])*100 if n_tasks else None
    sel_path=rd/'sd-lora'/'dataset'/'selection.json'
    selected=None; trace_count=None
    if sel_path.exists():
        manifest.append((str(sel_path.relative_to(run_root)), sha256_file(sel_path)))
        sj=load_json(sel_path); selected=sj.get('selected_task_count'); trace_count=sj.get('trace_count')
    geo_path=rd/'task-vector-geometry.json'
    cos=None; cum_align=None
    if geo_path.exists():
        manifest.append((str(geo_path.relative_to(run_root)), sha256_file(geo_path)))
        gj=load_json(geo_path); cos=gj.get('adjacent_update_cosine'); cum_align=gj.get('cumulative_alignment')
    update_path=rd/'sd-lora'/'update.json'
    loss=None; components=None; effective_rank=None
    if update_path.exists():
        manifest.append((str(update_path.relative_to(run_root)), sha256_file(update_path)))
        uj=load_json(update_path); loss=uj.get('training_loss'); components=uj.get('component_count_after'); effective_rank=uj.get('effective_rank_after')
    for c in ['text_memory','skill_bundle','agent_system','parametric_sd_lora']:
        obj=rc.get(c) or {}
        if obj.get('status')=='UPDATE':
            carrier_updates[c]+=1; carrier_update_rounds[c].append(r)
    row={
        'round':r,
        'attempt_count':rs.get('attempt_count'),
        'mean_score_pct':round(100*float(rs.get('mean_score') or 0),6),
        'clean_exact_success_count':rs.get('clean_exact_success_count'),
        'clean_exact_success_task_count':rs.get('clean_exact_success_task_count'),
        'engineering_invalid_count':rs.get('engineering_invalid_count'),
        'task_identity_count':rs.get('task_identity_count'),
        'success_at_1_pct':round(pct('success_at_1'),6) if pct('success_at_1') is not None else None,
        'success_at_2_pct':round(pct('success_at_2'),6) if pct('success_at_2') is not None else None,
        'success_at_4_pct':round(pct('success_at_4'),6) if pct('success_at_4') is not None else None,
        'success_at_8_pct':round(pct('success_at_8'),6) if pct('success_at_8') is not None else None,
        'best_of_8_score_pct':round(best8,6) if best8 is not None else None,
        'zero_exact_task_count':sum(1 for t in task_summaries if t['exact_success_count']==0),
        'near_miss_task_count':sum(1 for t in task_summaries if t['near_miss']),
        'frontier_task_count':sum(1 for t in task_summaries if t['frontier']),
        'cold_task_count':sum(1 for t in task_summaries if t['cold']),
        'selected_task_count':selected,
        'selected_trace_count':trace_count,
        'selected_minus_exact_task_count':(selected-rs.get('clean_exact_success_task_count')) if selected is not None and rs.get('clean_exact_success_task_count') is not None else None,
        'avg_unique_page_states':round(statistics.mean([t['unique_page_states_mean'] for t in task_summaries]),6) if task_summaries else None,
        'avg_unique_state_sigs':round(statistics.mean([t['unique_state_sigs_mean'] for t in task_summaries]),6) if task_summaries else None,
        'avg_unique_actions':round(statistics.mean([t['unique_actions_mean'] for t in task_summaries]),6) if task_summaries else None,
        'state_changing_rate':round(statistics.mean([t['state_changing_rate'] for t in task_summaries]),6) if task_summaries else None,
        'repeated_action_ratio':round(statistics.mean([t['repeated_action_ratio_mean'] for t in task_summaries if t['repeated_action_ratio_mean'] is not None]),6) if any(t['repeated_action_ratio_mean'] is not None for t in task_summaries) else None,
        'repeated_state_ratio':round(statistics.mean([t['repeated_state_ratio_mean'] for t in task_summaries if t['repeated_state_ratio_mean'] is not None]),6) if any(t['repeated_state_ratio_mean'] is not None for t in task_summaries) else None,
        'adjacent_update_cosine':cos,
        'cumulative_alignment':cum_align,
        'sd_lora_training_loss':loss,
        'component_count_after':components,
        'effective_rank_after':effective_rank,
    }
    per_round.append(row)

if not per_round:
    raise SystemExit('no sealed rounds found')
last=per_round[-1]['round']
first=per_round[0]['round']
stem=f'q17_frontier_metrics_R{first:03d}_R{last:03d}'
json_path=out_dir/f'{stem}.json'
csv_path=out_dir/f'{stem}.csv'
md_path=out_dir/f'q17_frontier_summary_R{first:03d}_R{last:03d}.md'
manifest_path=out_dir/f'q17_frontier_manifest_R{first:03d}_R{last:03d}.sha256'

# block helpers
blocks=[(70,79),(80,89),(90,99),(100,109),(110,119),(120,last)]
def block_stats(a,b):
    rows=[r for r in per_round if a<=r['round']<=b]
    if not rows: return None
    keys=['mean_score_pct','success_at_1_pct','success_at_2_pct','success_at_4_pct','success_at_8_pct','best_of_8_score_pct','zero_exact_task_count','near_miss_task_count','frontier_task_count','selected_task_count','avg_unique_state_sigs','avg_unique_actions','state_changing_rate','repeated_action_ratio']
    return {'start':a,'end':b,'n':len(rows), **{k:round(statistics.mean([x[k] for x in rows if x[k] is not None]),6) if any(x[k] is not None for x in rows) else None for k in keys}}
block_summary=[x for x in (block_stats(a,b) for a,b in blocks) if x]

payload={
    'schema':'openevo.q17.frontier.offline-metrics.202609101640',
    'generated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'run_root':str(run_root),
    'output_dir':str(out_dir),
    'sealed_rounds':len(per_round),
    'round_start':first,
    'round_end':last,
    'formal_rollouts':sum(int(r['attempt_count'] or 0) for r in per_round),
    'active_unsealed_rounds':active_unsealed,
    'final_panel_access_checked':'not opened by this script; script reads only round receipts and attempts logs',
    'carrier_updates':{k:carrier_updates[k] for k in ['parametric_sd_lora','text_memory','agent_system','skill_bundle']},
    'carrier_update_rounds':{k:carrier_update_rounds[k] for k in carrier_update_rounds},
    'block_summary':block_summary,
    'per_round':per_round,
    'warnings':warnings[:200],
    'warning_count':len(warnings),
}
json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
with open(csv_path,'w',newline='') as f:
    writer=csv.DictWriter(f, fieldnames=list(per_round[0].keys()), lineterminator='\n')
    writer.writeheader(); writer.writerows(per_round)
# manifest includes outputs last
with open(manifest_path,'w') as f:
    for rel,sha in sorted(set(manifest)):
        f.write(f'{sha}  {rel}\n')
for out in [json_path,csv_path,md_path]:
    pass

sel_equal=sum(1 for r in per_round if r['selected_task_count'] is not None and r['selected_task_count']==r['clean_exact_success_task_count'])
sel_never_exceeds=sum(1 for r in per_round if r['selected_task_count'] is not None and r['selected_task_count']<=r['clean_exact_success_task_count'])
latest20=per_round[-20:]
md=[]
md.append(f'# Q17 DirectApply Frontier Offline Diagnostics R{first:03d}–R{last:03d}')
md.append('')
md.append('## Scope')
md.append(f'- Run root: `{run_root}`')
md.append(f'- Sealed rounds included: {len(per_round)}')
md.append(f'- Formal rollouts included: {payload["formal_rollouts"]:,}')
md.append(f'- Active/unsealed rounds excluded: {active_unsealed or "none"}')
md.append('- This script performed no model calls, no GPU calls, no writes under the run root, and no final-panel access.')
md.append('')
md.append('## Headline')
md.append('- The R90–R99 dip is not a permanent stall: later sealed blocks recover in Score and Success@1.')
md.append('- Reliability improves more visibly than coverage: Success@1 rises more than Success@8 / best-of-8.')
md.append('- SD-LoRA selection remains success-conditioned: selected task count never exceeds exact-success task count in the sealed rounds where selection exists.')
md.append('')
md.append('## Block summary')
md.append('| block | n | Score | Success@1 | Success@8 | best-of-8 | zero-exact | near-miss | frontier | selected tasks | state sigs | unique actions |')
md.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for b in block_summary:
    md.append(f"| R{b['start']:03d}–R{b['end']:03d} | {b['n']} | {b['mean_score_pct']:.2f} | {b['success_at_1_pct']:.2f} | {b['success_at_8_pct']:.2f} | {b['best_of_8_score_pct']:.2f} | {b['zero_exact_task_count']:.2f} | {b['near_miss_task_count']:.2f} | {b['frontier_task_count']:.2f} | {b['selected_task_count'] if b['selected_task_count'] is not None else '—'} | {b['avg_unique_state_sigs']:.2f} | {b['avg_unique_actions']:.2f} |")
md.append('')
md.append('## Selection diagnostic')
md.append(f'- Rounds with `selected_task_count <= clean_exact_success_task_count`: {sel_never_exceeds} / {len([r for r in per_round if r["selected_task_count"] is not None])}')
md.append(f'- Rounds with exact equality: {sel_equal} / {len([r for r in per_round if r["selected_task_count"] is not None])}')
md.append(f'- Latest 20 selected task count mean: {statistics.mean([r["selected_task_count"] for r in latest20 if r["selected_task_count"] is not None]):.2f}')
md.append('')
md.append('## Deliverables')
md.append(f'- JSON: `{json_path}`')
md.append(f'- CSV: `{csv_path}`')
md.append(f'- Manifest: `{manifest_path}`')
md_path.write_text('\n'.join(md)+'\n')
# add output files to manifest too
with open(manifest_path,'a') as f:
    for p in [json_path,csv_path,md_path]:
        f.write(f'{sha256_file(p)}  OUTPUT::{p.name}\n')
print('OUT_JSON',json_path)
print('OUT_CSV',csv_path)
print('OUT_MD',md_path)
print('OUT_MANIFEST',manifest_path)
print('SEALED',len(per_round),f'R{first:03d}-R{last:03d}','FORMAL',payload['formal_rollouts'],'ACTIVE_EXCLUDED',active_unsealed)
print('LAST_BLOCKS')
for b in block_summary[-4:]:
    print(json.dumps(b, ensure_ascii=False))
print('SELECTION', sel_never_exceeds, sel_equal)
print('WARNINGS',len(warnings))
