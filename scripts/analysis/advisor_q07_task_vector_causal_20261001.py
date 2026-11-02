#!/usr/bin/env python3
import csv, json, math
from pathlib import Path
import numpy as np

ROOT=Path("/workspace/maintenance/ADVISOR-Q07-TASK-VECTOR-CAUSAL-20261001")
WT=Path("/workspace/tmp/q07-sync-main-20261001")
OUT=ROOT/"results"
M=json.loads((WT/"configs/experiment/manifests/advisor-q07-task-vector-causal-20261001.json").read_text())
DRAWS=100000; BOOT=50000; RS=20261001; BS=20261002

def load(cid):
    z=json.loads((OUT/cid/"summary.json").read_text())
    if z["status"]!="PASS" or z["task_count"]!=64 or z["panel_sha256"]!=M["panel_sha256"]: raise RuntimeError(cid)
    return z

def vec(z):
    d={}
    for r in z["rows"]:
        e=r["episode"]; d[r["task_identity"]]=(float(e["score"]),1.0 if e["won"] else 0.0,1.0 if float(e["score"])>0 else 0.0)
    if len(d)!=64: raise RuntimeError("task identity count")
    return d

def stats(a,b,rng,brng):
    keys=sorted(a)
    x=np.array([a[k][0]-b[k][0] for k in keys],dtype=np.float64)
    obs=float(x.mean()); n=len(x); cnt=0
    left=DRAWS
    while left:
        q=min(5000,left); signs=rng.integers(0,2,size=(q,n),dtype=np.int8)*2-1
        cnt+=int((np.abs((signs*x).mean(axis=1))>=abs(obs)-1e-15).sum()); left-=q
    p=(cnt+1)/(DRAWS+1)
    vals=[]; left=BOOT
    while left:
        q=min(5000,left); idx=brng.integers(0,n,size=(q,n))
        vals.append(x[idx].mean(axis=1)); left-=q
    bb=np.concatenate(vals)
    exact=float(np.mean([a[k][1]-b[k][1] for k in keys]))
    pos=float(np.mean([a[k][2]-b[k][2] for k in keys]))
    return {"mean_task_score_delta":obs,"ci95":[float(np.quantile(bb,.025)),float(np.quantile(bb,.975))],
            "randomization_p":p,"exact_success_delta":exact,"positive_reward_delta":pos,"task_count":n}

def holm(items):
    ps=[x["randomization_p"] for x in items]; order=np.argsort(ps); m=len(ps); prev=0
    adj=[0.0]*m
    for rank,idx in enumerate(order):
        val=min(1.0,(m-rank)*ps[idx]); prev=max(prev,val); adj[idx]=prev
    for i,x in enumerate(items):
        x["holm_adjusted_p"]=adj[i]; x["holm_reject_0_05"]=bool(adj[i]<=.05)

intact=vec(load("intact_w159")); ref=vec(load("r151_reference"))
rng=np.random.default_rng(RS); brng=np.random.default_rng(BS)
reverse=[]; removal=[]; exploratory=[]
for t in range(152,160):
    loo=vec(load(f"loo_r{t}")); rev=vec(load(f"reverse_r{t}"))
    a=stats(rev,intact,rng,brng); a["round_index"]=t; a["contrast"]="reverse_vs_intact"; reverse.append(a)
    b=stats(loo,intact,rng,brng); b["round_index"]=t; b["contrast"]="loo_vs_intact"; removal.append(b)
    # exploratory descriptive reverse vs loo with bootstrap only
    keys=sorted(intact); d=np.array([rev[k][0]-loo[k][0] for k in keys])
    exploratory.append({"round_index":t,"contrast":"reverse_vs_loo","mean_task_score_delta":float(d.mean())})
holm(reverse); holm(removal)
ref_delta=float(np.mean([ref[k][0]-intact[k][0] for k in sorted(intact)]))
result={"schema":"openevo.advisor.q07.task-vector-causal-analysis.v1","status":"PASS",
 "panel_sha256":M["panel_sha256"],"new_episode_count":17*64,"reused_intact_episodes":64,
 "primary_family":reverse,"secondary_family":removal,"exploratory_direction_contrast":exploratory,
 "r151_minus_intact_mean_score":ref_delta,"statistics_contract":M["statistics_contract"],
 "final_panel_access":0,"formal_rollouts_consumed":0,
 "claim_boundary":"development-panel causal interventions; no controller selection or final-panel claim"}
(ROOT/"Q07_CAUSAL_ANALYSIS.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
with (ROOT/"Q07_CAUSAL_EFFECTS.csv").open("w",newline="") as f:
    fields=["family","round_index","mean_task_score_delta","ci_low","ci_high","randomization_p","holm_adjusted_p","holm_reject_0_05","exact_success_delta","positive_reward_delta"]
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for fam,arr in [("reverse",reverse),("removal",removal)]:
        for x in arr:w.writerow({"family":fam,"round_index":x["round_index"],"mean_task_score_delta":x["mean_task_score_delta"],
          "ci_low":x["ci95"][0],"ci_high":x["ci95"][1],"randomization_p":x["randomization_p"],"holm_adjusted_p":x["holm_adjusted_p"],
          "holm_reject_0_05":x["holm_reject_0_05"],"exact_success_delta":x["exact_success_delta"],"positive_reward_delta":x["positive_reward_delta"]})
print(json.dumps({"status":"PASS","primary":reverse,"secondary":removal,"r151_minus_intact":ref_delta},sort_keys=True))
