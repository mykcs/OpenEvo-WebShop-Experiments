"""Frozen before reading fresh outcomes: one matched fixed-checkpoint contrast."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics


def canonical(x):
    return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()


def percentile(values,p):
    x=(len(values)-1)*p;i=int(x);j=min(i+1,len(values)-1)
    return values[i]+(values[j]-values[i])*(x-i)


def analyze(value):
    assert value['status']=='BOTH_ENDPOINTS_VERIFIED_COMPLETE'
    assert value['panel_sha256']=='bcb91ad087ba166b04b75baab1c99f3bf44fb45565aea8ad1b97a33c945ed109'
    rows=value['rows'];assert len(rows)==64 and [r['ordinal'] for r in rows]==list(range(64))
    assert len({r['pair_identity_sha256'] for r in rows})==64
    for r in rows:
        assert 0<=r['r119_score']<=1 and 0<=r['r159_score']<=1
        assert type(r['r119_won']) is bool and type(r['r159_won']) is bool
        assert r['same_task_seed_instruction'] is True
        assert len(r['r119_raw_sha256'])==len(r['r159_raw_sha256'])==64
    delta=[100*(r['r159_score']-r['r119_score']) for r in rows]
    rng=random.Random(2026102821)
    draws=sorted(sum(rng.choices(delta,k=64))/64 for _ in range(50000))
    ci=[percentile(draws,p) for p in (.025,.975)]
    return dict(schema='openevo.q09.fixed-checkpoint-fresh.paired-analysis.v1',
        panel_sha256=value['panel_sha256'],pairs=64,evaluations=128,
        r119_mean_score100=100*statistics.mean(r['r119_score'] for r in rows),
        r159_mean_score100=100*statistics.mean(r['r159_score'] for r in rows),
        r119_exact=sum(r['r119_won'] for r in rows),r159_exact=sum(r['r159_won'] for r in rows),
        r159_minus_r119_score100=statistics.mean(delta),ci95=ci,
        primary_classification='POSITIVE' if ci[0]>0 else 'NEGATIVE' if ci[1]<0 else 'INCONCLUSIVE',
        bootstrap=dict(draws=50000,seed=2026102821,algorithm='Python random.Random.choices paired deltas',interval='percentile95%, linear interpolation (n-1)*p'),
        boundary='Selected fixed checkpoint pair, new task panel; no independent training seed, policy causal comparison or cross-environment claim.',
        outcomes_seen_before_design=False,development_outcome_seen_before_design=True,
        rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    result=analyze(json.loads(a.input.read_text()))
    with a.output.open('x') as f:json.dump(result,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False))
