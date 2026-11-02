#!/usr/bin/env python3
"""Render retained Q06 context estimates without evaluation or resampling."""
import argparse,hashlib,json,sys
from pathlib import Path
if __package__ in (None, ''):
 sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.openevo_webshop.analyze_q06_context_attempts import read_source_record,validate_source_provenance

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main(argv=None):
 p=argparse.ArgumentParser(__doc__);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--source-record',type=Path,required=True);p.add_argument('--output-directory',type=Path,required=True);a=p.parse_args(argv)
 analysis_bytes=a.analysis.read_bytes();data=json.loads(analysis_bytes);root=a.output_directory
 record,record_sha=read_source_record(a.source_record)
 validate_source_provenance(data,record,record_sha)
 if data['status']!='PASS_SEALED_CONTEXT_ANALYSIS' or data['budget']['valid_episodes']!=192:raise ValueError('Require sealed three-endpoint context analysis')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 names=('bounded','beta');colors=('#176b87','#a84b2a')
 specs=[('Mean Task Score','score units','metrics','score'),('Exact success (descriptive)','probability units','metrics','success'),('Reasoning-token entropy','nats','metrics','reasoning_entropy_mean_per_token'),('Action-token entropy','nats','metrics','action_entropy_mean_per_token'),('Click entropy / log(choices)','normalized units','click','normalized_entropy')]
 fig,axes=plt.subplots(2,3,figsize=(13,7.4))
 for ax,(title,unit,group,key) in zip(axes.flat,specs):
  ax.axvline(0,color='#777',lw=1,ls='--')
  for y,name,color in zip((1,0),names,colors):
   m=data['comparisons'][name][group][key];d=m['mean_delta']
   if d is None:continue
   lo,hi=m['ci95'];ax.errorbar(d,y,xerr=[[d-lo],[hi-d]],fmt='o',capsize=4,color=color)
  ax.set_title(title,fontsize=11);ax.set_yticks([1,0],['GDR − Bounded','GDR − scalar beta'],fontsize=8);ax.set_ylim(-.5,1.5);ax.set_xlabel(unit);ax.spines[['top','right']].set_visible(False)
 ax=axes.flat[5];ax.axvline(0,color='#777',lw=1,ls='--');labels=('5-8','9-16','17-32','33+')
 for index,label in enumerate(labels):
  for offset,name,color in zip((-.13,.13),names,colors):
   m=data['comparisons'][name]['choice_count_strata'][label];d=m['mean_delta']
   if d is None:continue
   lo,hi=m['ci95'];ax.errorbar(d,index+offset,xerr=[[d-lo],[hi-d]],fmt='o',capsize=3,color=color,label='GDR − '+name if index==0 else None)
 ax.set_yticks(range(4),[x+' choices' for x in labels],fontsize=9);ax.invert_yaxis();ax.set_title('Normalized click entropy by choices',fontsize=11);ax.set_xlabel('normalized units');ax.spines[['top','right']].set_visible(False)
 fig.suptitle('Q06 historical GDR ON context | 64 paired development tasks',fontsize=14)
 from matplotlib.lines import Line2D
 fig.legend([Line2D([0],[0],marker='o',color=colors[0],label='GDR − Bounded',linestyle=''),Line2D([0],[0],marker='o',color=colors[1],label='GDR − scalar beta',linestyle='')],['GDR − Bounded','GDR − scalar beta'],loc='lower center',bbox_to_anchor=(.5,.065),ncol=2,frameon=False,fontsize=9)
 fig.text(.5,.012,'Points show paired-task means; bars show retained 95% task-bootstrap intervals. Comparisons are descriptive, without multiplicity adjustment.\nTrajectories differ; this does not establish entropy causality or identify the plotted meeting run.',ha='center',fontsize=8.5)
 fig.tight_layout(rect=(0,.15,1,.93));outputs=[]
 for ext in ('png','svg'):
  target=root/('Q06_CONTEXT_METRICS.'+ext)
  if target.exists():raise FileExistsError(target)
  fig.savefig(target,dpi=180);outputs.append({'path':target.name,'sha256':sha(target)})
 plt.close(fig)
 receipt={'schema':'openevo.q06.context-plot-provenance.v2','analysis_sha256':hashlib.sha256(analysis_bytes).hexdigest(),'source_record_sha256':record_sha,'source_provenance_status':'VERIFIED_RAW_BYTES','builder_sha256':sha(Path(__file__)),'matplotlib_version':matplotlib.__version__,'outputs':outputs,'new_model_calls':0,'new_environment_steps':0,'new_bootstrap_resampling':False,'final_panel_access':0}
 with (root/'Q06_CONTEXT_PLOT_PROVENANCE.json').open('x') as f:json.dump(receipt,f,indent=2,sort_keys=True);f.write('\n')
if __name__=='__main__':main()
