#!/usr/bin/env python3
"""Descriptive three-endpoint Q06 context analysis of sealed rows only."""
from __future__ import annotations
import argparse,hashlib,json,math,os,statistics,sys
from pathlib import Path
if __package__ in (None, ''):
 sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.openevo_webshop.analyze_q06_paired_attempts import summarize_row,state_average,paired_ci,BINS
EXPECTED={'bounded':'1bc2f7f62c1bde68cd4fd34e86c25af7f9663153a9ded631759dda52c87a21a5','beta':'06360601968c23954d33331ba143730192fa1f46b2540998a3c062eab655a650','gdr':'1f789d86ff1a3ba96211eb950f78763f99ed6fb2f0545f11c869ec8e6de3da6c'}
IDENTITIES=('panel_sha256','task_panel_manifest_sha256','task_panel_task_identity_sha256','runtime_execution_sha','probe_script_sha256','prompt_contract_sha256','base_model','base_model_revision')
METRICS=('score','success','steps','generation_tokens','reasoning_entropy_mean_per_token','action_entropy_mean_per_token','reasoning_nll_mean_per_token','action_nll_mean_per_token','reasoning_margin_mean','action_margin_mean','reasoning_generation_p90_mean','action_generation_p90_mean')
CLICK=('entropy','normalized_entropy','hhi','effective_choices','n','top1_probability','top1_top2_probability_margin','selected_probability','selected_rank')
SOURCE_RECORD_SCHEMA = 'openevo.q06.context-source-reconciliation.v1'
EMPTY_SHA256 = hashlib.sha256(b'').hexdigest()


def valid_source_digest(value):
 return (isinstance(value, str) and len(value) == 64
         and all(c in '0123456789abcdef' for c in value)
         and value != EMPTY_SHA256)


def read_source_record(path):
 """Recorded digests are expectations, never proof that raw bytes were read."""
 payload = path.read_bytes()
 record = json.loads(payload)
 if record.get('schema') != SOURCE_RECORD_SCHEMA:
  raise ValueError('Wrong Q06 source record schema')
 for arm in EXPECTED:
  for kind in ('attempts', 'summary'):
   entry = record.get('sources', {}).get(arm, {}).get(kind, {})
   if not valid_source_digest(entry.get('recorded_sha256')):
    raise ValueError(f'Missing/invalid recorded source digest: {arm} {kind}')
 return record, hashlib.sha256(payload).hexdigest()


def read_source_snapshot(path, expected_digest):
 """Hash and parse one byte snapshot; a later path read cannot supply its hash."""
 payload = path.read_bytes()
 if not payload.strip():
  raise ValueError(f'Empty source: {path}')
 digest = hashlib.sha256(payload).hexdigest()
 if digest != expected_digest:
  raise ValueError(f'Source digest mismatch: {path}')
 return payload, digest


def validate_source_provenance(data, source_record, source_record_sha256):
 """Fail closed for legacy, unknown, empty, or mismatched source claims."""
 proof = data.get('source_provenance', {})
 if (data.get('schema') != 'openevo.q06.three-endpoint-context-analysis.v2'
     or data.get('status') != 'PASS_SEALED_CONTEXT_ANALYSIS'
     or proof.get('status') != 'VERIFIED_RAW_BYTES'
     or proof.get('raw_source_bytes_verified') is not True
     or proof.get('method') != 'sha256_and_parse_same_snapshot'
     or not valid_source_digest(source_record_sha256)
     or proof.get('source_record_sha256') != source_record_sha256):
  raise ValueError('Context analysis lacks verified raw-source provenance')
 for arm in EXPECTED:
  source = data.get('sources', {}).get(arm, {})
  for kind in ('attempts', 'summary'):
   digest = source.get(kind + '_sha256')
   expected = source_record.get('sources', {}).get(arm, {}).get(kind, {}).get('recorded_sha256')
   size = source.get(kind + '_bytes')
   if (not valid_source_digest(digest) or digest != expected
       or type(size) is not int or size <= 0):
    raise ValueError(f'Invalid/unbound context source provenance: {arm} {kind}')

def ci(values,name):
 if not values:return {'paired_tasks':0,'mean_delta':None,'ci95':None}
 result=paired_ci(values,name)
 result['mean_delta']=result.pop('mean_delta_beta_minus_bounded')
 return result

def correlation(rows,key):
 pairs=[(r[key],r['score']) for r in rows if r[key] is not None]
 if len(pairs)<2:return None
 x,y=zip(*pairs);xm,ym=statistics.fmean(x),statistics.fmean(y)
 denominator=math.sqrt(sum((a-xm)**2 for a in x)*sum((b-ym)**2 for b in y))
 return sum((a-xm)*(b-ym) for a,b in pairs)/denominator if denominator else None

def main(argv=None):
 p=argparse.ArgumentParser(__doc__)
 for arm in EXPECTED:
  p.add_argument('--'+arm,type=Path,required=True)
  p.add_argument('--'+arm+'-summary',type=Path,required=True)
 p.add_argument('--source-record',type=Path,required=True,help='Append-only record of expected sealed source digests')
 p.add_argument('--verify-sources-only',action='store_true',help='Verify source bytes/identities without analysis or file writes')
 p.add_argument('--output',type=Path);a=p.parse_args(argv)
 if not a.verify_sources_only and a.output is None:p.error('--output is required for analysis')
 record, record_sha = read_source_record(a.source_record)
 snapshots={};raw={};summaries={};sources={}
 for arm in EXPECTED:
  sources[arm]={}
  for kind in ('attempts','summary'):
   path=getattr(a,arm if kind=='attempts' else arm+'_summary')
   payload,digest=read_source_snapshot(path,record['sources'][arm][kind]['recorded_sha256'])
   snapshots[path]=digest
   sources[arm][kind+'_sha256']=digest
   sources[arm][kind+'_bytes']=len(payload)
   if kind=='attempts':
    # JSONL records are physical LF-delimited bytes, not Unicode splitlines.
    raw[arm]=[json.loads(line) for line in payload.split(b'\n') if line.strip()]
   else:summaries[arm]=json.loads(payload)
 for arm in EXPECTED:
  require=len(raw[arm])==64 and summaries[arm].get('status')=='PASS' and summaries[arm].get('task_count')==64
  if not require:raise ValueError('Expected sealed 64-task PASS: '+arm)
  if summaries[arm].get('external_teacher_calls')!=0 or summaries[arm].get('final_panel_access')!=0:raise ValueError('Budget violation: '+arm)
  if any(row.get('engineering_invalid') or row.get('adapter_sha256')!=EXPECTED[arm] for row in raw[arm]):raise ValueError('Invalid/unbound context row: '+arm)
 for key in IDENTITIES:
  if summaries['bounded'].get(key) is None or len({summaries[arm].get(key) for arm in EXPECTED})!=1:raise ValueError('Context shared identity drift: '+key)
 if summaries['gdr']['checkpoint_manifest_sha256']!='edcaaf2b216e3f8713574105e52037675ef3ea091b81892d557dda4af2f9c33c':raise ValueError('Wrong GDR context manifest')
 rows={arm:{r['task_identity']:summarize_row(r) for r in data} for arm,data in raw.items()}
 for arm in EXPECTED:
  for row in rows[arm].values():
   for click in row['click']:
    click['effective_choices']=1.0/click['hhi'] if click['hhi']>0 else None
   row['task_click_normalized_entropy']=state_average(row,'normalized_entropy')
 ids=sorted(rows['bounded'])
 if len(ids)!=64 or any(set(rows[arm])!=set(ids) for arm in EXPECTED):raise ValueError('Duplicate/mismatched task identity')
 for t in ids:
  if len({(rows[arm][t]['worker_seed'],rows[arm][t]['paired_generation_seed']) for arm in EXPECTED})!=1:raise ValueError('Context seed drift: '+t)
 for arm in EXPECTED:sources[arm]['checkpoint_manifest_sha256']=summaries[arm]['checkpoint_manifest_sha256']
 proof={'status':'VERIFIED_RAW_BYTES','raw_source_bytes_verified':True,'method':'sha256_and_parse_same_snapshot','source_record_sha256':record_sha}
 if a.verify_sources_only:
  for path,digest in snapshots.items():read_source_snapshot(path,digest)
  if read_source_record(a.source_record)[1]!=record_sha:raise ValueError('Source record changed during verification')
  print(json.dumps({'status':'PASS_SOURCE_REVALIDATION','sources':sources,'source_provenance':proof,'new_model_calls':0,'new_environment_steps':0,'final_panel_access':0},indent=2,sort_keys=True))
  return
 out={'schema':'openevo.q06.three-endpoint-context-analysis.v2','status':'PASS_SEALED_CONTEXT_ANALYSIS','shared_identity':{k:summaries['bounded'][k] for k in IDENTITIES},'sources':sources,'source_provenance':proof,'comparisons':{},'endpoints':{},'budget':{'valid_episodes':192,'preserved_failed_bounded_slots':64,'total_attempt_slots':256,'new_model_calls_by_analysis':0,'final_panel_access':0},'limitations':['Historical GDR mechanism-family context, without asserting plotted meeting-run identity.','Task-paired trajectories can diverge; this is not a same-state causal contrast.','All context comparisons and correlations are descriptive; no multiplicity-adjusted confirmatory claim.','Reasoning/action p90 is within generation; averaging it does not recover a pooled token p90.']}
 for arm in EXPECTED:
  vals=list(rows[arm].values());out['endpoints'][arm]={'tasks':64,'mean_score':statistics.fmean(r['score'] for r in vals),'exact_successes':sum(r['success'] for r in vals),'generation_tokens':sum(r['generation_tokens'] for r in vals),'click_states':sum(len(r['click']) for r in vals),'sentinel_admissible':sum(r['sentinel_admissible'] for r in vals),'sentinel_selected':sum(r['sentinel_selected'] for r in vals),'descriptive_score_correlations':{k:correlation(vals,k) for k in ('reasoning_entropy_mean_per_token','action_entropy_mean_per_token','task_click_normalized_entropy')}}
 for reference in ('bounded','beta'):
  pairs=[(rows[reference][t],rows['gdr'][t]) for t in ids];result={'direction':'gdr_minus_'+reference,'metrics':{},'click':{},'choice_count_strata':{}}
  for key in METRICS:
   values=[float(g[key])-float(b[key]) for b,g in pairs if b[key] is not None and g[key] is not None];result['metrics'][key]=ci(values,key)
  for key in CLICK:
   values=[float(g)-float(b) for b,g in ((state_average(x,key),state_average(y,key)) for x,y in pairs) if b is not None and g is not None];result['click'][key]=ci(values,'click_'+key)
  for lo,hi,label in BINS:
   values=[float(g)-float(b) for b,g in ((state_average(x,'normalized_entropy',lo,hi),state_average(y,'normalized_entropy',lo,hi)) for x,y in pairs) if b is not None and g is not None];result['choice_count_strata'][label]=ci(values,'stratum_'+label)
  out['comparisons'][reference]=result
 # Retained snapshots define the result. Reject inputs changed during analysis
 # rather than publishing a PASS against a now-different supposedly sealed path.
 for path,digest in snapshots.items():read_source_snapshot(path,digest)
 if read_source_record(a.source_record)[1]!=record_sha:raise ValueError('Source record changed during analysis')
 validate_source_provenance(out,record,record_sha)
 with a.output.open('x') as f:json.dump(out,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
