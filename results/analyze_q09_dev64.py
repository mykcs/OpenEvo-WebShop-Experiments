"""Exploratory paired development analysis, written after outcome exposure."""
import json, random, statistics, hashlib
from pathlib import Path
W=Path(__file__).parent
old=json.loads((W/'q09-r139-measurement.json').read_text())
new=json.loads((W/'q09-dev64-measurement.json').read_text())
a=sorted(old['rows'],key=lambda x:x['values']['probe_ordinal'])
b=sorted(new['rows'],key=lambda x:x['values']['probe_ordinal'])
assert len(a)==len(b)==64
for x,y in zip(a,b):
    for k in ['probe_ordinal','task_identity','paired_generation_seed']:
        assert x['values'][k]==y['values'][k], (k,x,y)
    for v in [x['values'],y['values']]:
        assert v['identity_ok'] and v['trace_complete'] and not v['engineering_invalid']
assert old['summary']['panel_sha256']==new['summary']['panel_sha256']
d=[100*(y['values']['task_score']-x['values']['task_score']) for x,y in zip(a,b)]
rng=random.Random(20261028)
boot=sorted(sum(rng.choices(d,k=64))/64 for _ in range(50000))
def quantile(p):
    i=(len(boot)-1)*p; j=int(i)
    return boot[j]+(boot[min(j+1,len(boot)-1)]-boot[j])*(i-j)
out=dict(status='EXPLORATORY_DEVELOPMENT_ONLY',outcome_exposure='New R159 aggregate was unintentionally read in a separate observer GitHub issue comment at approximately 2026-10-28 18:41 UTC, before this analysis was written. This is not blind preregistration.',comparison='Same stream R159 minus R139; longitudinal checkpoint comparison, not ACTIVE versus FROZEN causal contrast',paired_tasks=64,panel_sha256=new['summary']['panel_sha256'],r139_mean_score_100=old['summary']['mean_task_score']*100,r159_mean_score_100=new['summary']['mean_task_score']*100,difference_score_100=statistics.mean(d),paired_percentile_bootstrap_ci95=[quantile(.025),quantile(.975)],bootstrap_resamples=50000,bootstrap_rng='Python random.Random(20261028), task resampling with replacement',multiplicity='one exploratory longitudinal comparison, unadjusted interval',r139_exact_success=old['summary']['exact_success_count'],r159_exact_success=new['summary']['exact_success_count'],exact_success_difference_descriptive_only=new['summary']['exact_success_count']-old['summary']['exact_success_count'],final_panel_access=0,new_evaluations=64,reused_historical_evaluations=64,limitations=['Previously used development tasks, one training stream, no fresh-task confirmation or independent initialization','Historical R139 raw label is FROZEN but reviewed endpoint reference maps actual checkpoint to ACTIVE; raw labels retained','Historical producer self-hash absent; current code equality cannot repair historical execution provenance','No outcome-based rerun, exclusion, independent-baseline duplication or adaptive panel expansion'])
(W/'q09-dev64-exploratory-analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
