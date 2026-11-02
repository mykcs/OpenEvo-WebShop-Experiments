#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--amendment',type=Path,required=True);p.add_argument('--arm',choices=('3b','qwen3-1p7b'),required=True);p.add_argument('--execution-git-sha',required=True);a=p.parse_args();m=load(a.amendment)
 schemas={'openevo.ceiling1-stage2-resource-migration.ray-continuation.202609042025','openevo.ceiling1-stage2-resource-migration.ray-v1','openevo.ceiling1-stage2-resource-migration.ray-continuation-v1'}  # legacy-revision-compat
 if m.get('schema') not in schemas or m.get('status') not in {'AUTHORIZED','AUTHORIZED_PENDING_SCIENCE_RECOVERY'}:raise RuntimeError('Ray migration not authorized')
 if m.get('successor_execution_git_sha')!=a.execution_git_sha:raise RuntimeError('Ray execution SHA drift')
 deps=[('ray_backend_config_path','ray_backend_config_sha256'),('ray_pool_ready_path','ray_pool_ready_sha256')]
 if m.get('schema')=='openevo.ceiling1-stage2-resource-migration.ray-v1': deps += [('prior_v7_migration_path','prior_v7_migration_sha256'),('ray_backend_qualification_path','ray_backend_qualification_sha256'),('ray_activation_path','ray_activation_sha256')]  # legacy-revision-compat
 else: deps += [('prior_ray_migration_path','prior_ray_migration_sha256')]
 if m.get('science_amendment_path'):
  deps += [('science_amendment_path','science_amendment_sha256')]
 for key,hkey in deps:
  pth=Path(m[key]);
  if not pth.is_file() or sha(pth)!=m[hkey]:raise RuntimeError(f'Ray migration dependency drift: {key}')
 if m.get('science_amendment_path'):
  sx=load(Path(m['science_amendment_path']))
  if sx.get('schema')!='openevo.ceiling1-stage2-carrier-contract-fail-noop-amendment.v1' or sx.get('status')!='AUTHORIZED' or a.arm not in set(sx.get('applies_to_arms') or []): raise RuntimeError('Ray science amendment invalid')  # legacy-revision-compat
  if sx.get('transition',{}).get('reason')!='NOOP_KEEP_PRIOR_TYPED_CONTRACT_REJECTED_AFTER_REPAIR': raise RuntimeError('Ray science amendment transition drift')
 b=(m.get('boundaries') or {}).get(a.arm) or {};st=load(a.output/'execution-state.json')
 if int(st.get('next_round',-1))!=int(b.get('next_round',-2)) or int(st.get('rollouts_consumed',-1))!=int(b.get('rollouts_consumed',-2)) or st.get('state_id')!=b.get('state_id'):raise RuntimeError('Stage2 state moved after Ray migration receipt')
 if b.get('partial_round') is not None:
  rr=a.output/f"round-{int(b['partial_round']):03d}"
  if sha(rr/'attempts.jsonl')!=b.get('partial_attempts_sha256') or sha(rr/'summary.json')!=b.get('partial_summary_sha256'):raise RuntimeError('partial rollout provenance drift')
 if b.get('carrier_noop_recovery_receipt_path'):
  rp=Path(b['carrier_noop_recovery_receipt_path'])
  if not rp.is_file() or sha(rp)!=b.get('carrier_noop_recovery_receipt_sha256'): raise RuntimeError('carrier-NOOP recovery receipt drift')
  rx=load(rp)
  if rx.get('status')!='PASS' or rx.get('arm')!=a.arm or rx.get('science_amendment_sha256')!=m.get('science_amendment_sha256'): raise RuntimeError('carrier-NOOP recovery receipt invalid')
 print(f'CEILING1_STAGE2_RESOURCE_MIGRATION_RAY_202609042025=PASS {a.arm} {b.get("next_round")} {b.get("rollouts_consumed")}');return 0
if __name__=='__main__':raise SystemExit(main())
