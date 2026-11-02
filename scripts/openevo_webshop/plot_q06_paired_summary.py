#!/usr/bin/env python3
"""Plot already sealed Q06 estimates; performs no evaluation or resampling."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--output-directory',type=Path,required=True)
    a=p.parse_args();data=json.loads(a.analysis.read_text());root=a.output_directory
    assert data['population']['paired_tasks']==64
    specs=[('Mean Task Score','score units',data['paired_metrics']['task_score']),('Exact success (secondary)','probability units',data['paired_metrics']['exact_success']),('Reasoning-token entropy','nats',data['paired_metrics']['reasoning_entropy_mean_per_token']),('Action-token entropy','nats',data['paired_metrics']['action_entropy_mean_per_token']),('Click-choice entropy / log(choices)','normalized units',data['click_metrics']['normalized_entropy'])]
    fig,axes=plt.subplots(2,3,figsize=(12,6.6));color='#176b87'
    for ax,(title,unit,m) in zip(axes.flat,specs):
        d=m['mean_delta_beta_minus_bounded'];lo,hi=m['ci95']
        ax.axvline(0,color='#777777',lw=1,ls='--');ax.errorbar(d,0,xerr=[[d-lo],[hi-d]],fmt='o',color=color,capsize=5,lw=2)
        ax.set_title(title,fontsize=11);ax.set_yticks([]);ax.set_xlabel(unit);ax.set_ylim(-.8,.8)
        ax.text(.5,.12,f'{d:+.4f}  [{lo:+.4f}, {hi:+.4f}]',transform=ax.transAxes,ha='center',fontsize=9)
        ax.spines[['top','right','left']].set_visible(False)
    ax=axes.flat[5];strata=data['choice_count_strata'];names=['5-8','9-16','17-32','33+'];ys=list(range(len(names)))
    ax.axvline(0,color='#777777',lw=1,ls='--')
    for y,n in zip(ys,names):
        row=strata[n];d=row['mean_h_over_log_n_delta_beta_minus_bounded'];lo,hi=row['ci95_task_cluster']
        ax.errorbar(d,y,xerr=[[d-lo],[hi-d]],fmt='o',color=color,capsize=4)
    ax.set_yticks(ys,[f'{n} choices (n={strata[n]["paired_task_count"]})' for n in names],fontsize=9);ax.invert_yaxis();ax.set_title('Click entropy by choice count',fontsize=11);ax.set_xlabel('normalized units');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Q06: scalar beta minus Bounded | 64 paired development tasks',fontsize=14)
    fig.text(.5,.025,'Points: task-level mean differences; bars: retained 95% bootstrap intervals.\nSecondary metrics are exploratory, without multiplicity correction. Trajectories differ; entropy is descriptive, not causal.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.94))
    outputs=[]
    for ext in ['png','svg']:
        target=root/f'Q06_E2_PAIRED_METRICS.{ext}';assert not target.exists();fig.savefig(target,dpi=180);outputs.append({'path':target.name,'sha256':sha(target)})
    plt.close(fig)
    receipt={'schema':'openevo.q06.paired-plot-provenance.v1','analysis_path':str(a.analysis),'analysis_sha256':sha(a.analysis),'builder_sha256':sha(Path(__file__)),'matplotlib_version':matplotlib.__version__,'outputs':outputs,'new_model_calls':0,'new_environment_steps':0,'final_panel_access':0,'new_bootstrap_resampling':False}
    target=root/'Q06_E2_PLOT_PROVENANCE.json';assert not target.exists();target.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
