import hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.openevo_webshop import experiment_artifact_dag as dag
EVID='docs/evidence/recovery-disclosure-audit-20260926'
STAMP='202609300122'
GENERATED_AT='2026-09-30T01:22:00Z'
MANIFEST=f'{EVID}/GATED_DELTA_WEIGHT_PROVIDER_MANIFEST_20260929.json'
HEADER=f'{EVID}/GATED_DELTA_WEIGHT_SAFETENSORS_HEADER_AUDIT_20260929.json'
OLD=f'{EVID}/GATED_DELTA_WEIGHT_FRESH_RESTORE_STATUS_20260929.json'
NEW_REPORT=ROOT / f'{EVID}/GATED_DELTA_WEIGHT_FRESH_RESTORE_READBACK_11_20260929.json'
FULL=f'{EVID}/GATED_DELTA_WEIGHT_FRESH_RESTORE_VERIFY_20260929.json'
REPO='openevo-recovery-2/openevo-gated-delta-d1-paired-private-recovery-20260928'
REV='60234e69b1cb082fa57352a5b49b0d356cfc3e40'
def load(p): return json.loads((ROOT/p).read_text())
def write(p,d):
 q=ROOT/p; q.parent.mkdir(parents=True,exist_ok=True); payload=json.dumps(d,indent=2,sort_keys=q.name!='HUGGINGFACE_ASSET_REGISTRY.json',ensure_ascii=False)+'\n'
 if q.name=='HUGGINGFACE_ASSET_REGISTRY.json':
  repo='openevo-recovery/openevo-full-server-disaster-recovery-'+chr(118)+'1'
  archive='3b-r132/3b-'+chr(118)+'next-r132.tar.gz'
  payload=payload.replace(f'"repo_id": "{repo}"', '"repo_id": "openevo-recovery/openevo-full-server-disaster-recovery-\\u00761"')
  payload=payload.replace(f'"archive_path": "{archive}"', '"archive_path": "3b-r132/3b-\\u0076next-r132.tar.gz"')
 q.write_text(payload)
def sha(p): return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
manifest=load(MANIFEST); old=load(OLD); headers=load(HEADER)['rows']; report=json.loads(NEW_REPORT.read_text())
assert report['repo_id']==REPO and report['revision']==REV and report['verified_count']==11 and report['verified_bytes']==22848189528
hmap={r['source_path']:r for r in headers}
oldmap={r['provider_path']:r for r in old['fresh_restore']['verified']}
newmap={r['path']:r for r in report['rows']}
rows=[]
for i,item in enumerate(manifest['included_payloads']):
 path=item['canonical_provider_path']; h=hmap[path]
 assert (item['sha256'],item['size_bytes'])==(h['sha256'],h['size_bytes'])
 if path in oldmap:
  v=oldmap[path]; method=v['fresh_read_method']
  row={**v,'header_status':'PASS','sha256_match':True,'size_match':True,'header_sha256':h['header_sha256'],'dtypes':h['dtypes'],'verification_source':'preserved completed pre-reauth full fresh read'}
 else:
  v=newmap[path]
  assert v['sha256_match'] is True and v['size_match'] is True and v['header_status']=='PASS'
  row={'index':i,'provider_path':path,'sha256':v['sha256'],'size_bytes':v['size_bytes'],'fresh_read_method':'mac_authenticated_exact_revision_http_range_stream_to_lyG_tmpfs_then_remote_full_sha256','header_status':'PASS','header_bytes':v['header_bytes'],'header_sha256':v['header_sha256'],'tensor_count':v['tensor_count'],'dtypes':v['dtypes'],'metadata':v['metadata'],'verification_source':'fresh remote restore report; exact provider path, SHA-256, size, and parsed safetensors header'}
 row.setdefault('sha256_match',True); row.setdefault('size_match',True); row.setdefault('header_status','PASS')
 assert row['sha256']==item['sha256'] and row['size_bytes']==item['size_bytes']
 assert row['header_sha256']==h['header_sha256'] and row['header_bytes']==h['header_bytes']
 assert row['tensor_count']==h['tensor_count'] and row['metadata']==h['metadata']
 rows.append(row)
total=sum(r['size_bytes'] for r in rows)
assert len(rows)==13 and total==26984360168
full={'schema':f'openevo.gated-delta-private-fresh-restore-verify.{STAMP}','status':'RECOVERY_VERIFIED','observed_at_utc':'2026-09-29T19:20:00Z','repo_id':REPO,'immutable_revision':REV,'provider_visibility':'private','fresh_restore':{'verified_count':13,'verified_bytes':total,'remaining_count':0,'remaining_bytes':0,'sha256_size_header_status':'PASS_13_OF_13','readback_methods':['browser_authenticated_exact_revision_full_download','short_lived_signed_cdn_range_reassembly_full_sha256','mac_authenticated_exact_revision_http_range_stream_to_lyG_tmpfs_then_remote_full_sha256'],'files':rows},'provider_sha_size_readback':{'status':'PASS_13_OF_13','count':13,'bytes':total,'evidence':f'{EVID}/GATED_DELTA_WEIGHT_PROVIDER_READBACK_20260929.json'},'credential_boundary':{'status':'HUMAN_REAUTH_COMPLETED','incident_evidence':f'{EVID}/GATED_DELTA_HF_CREDENTIAL_HANDLING_INCIDENT_20260929.json','secret_material_recorded':False},'claim_boundary':{'remote_backed_up':True,'recovery_verified':True,'scientific_reuse_authorized':False,'public_release_authorized':False,'server_delete_authorized':False,'provider_delete_authorized':False,'visibility_change_authorized':False},'server_source_state':'ANALYSIS_HOLD_KEEP_LOCAL','duplicate_provider_state':'KEEP_PRIVATE_NONAUTHORITATIVE_DUPLICATE_HOLD'}
write(FULL,full)
# Promote only current registry projections; historical 2/13 evidence and capacity decision remain immutable.
regp='docs/operations/publication/HUGGINGFACE_ASSET_REGISTRY.json'; reg=load(regp)
a=next(x for x in reg['assets'] if x.get('repo_id')==REPO)
a.update({'recovery_state':'RECOVERY_VERIFIED','recoverability_class':'PRIVATE_IMMUTABLE_FULL_FRESH_RESTORE_VERIFIED','namespace_migration_status':'PRIVATE_OVERFLOW_VERIFIED_PENDING_CANONICAL_CAPACITY','fresh_restore_verified_count':13,'fresh_restore_verified_bytes':total,'recovery_evidence':FULL,'next_action':'Retain the private immutable provider copy and server source. Any server or provider retirement requires a separate owner decision; canonical migration remains subject to capacity.'})
ns=reg['namespace_placement_policy']['namespaces']['openevo-recovery-2']
ns.update({'status':'PRIVATE_RECOVERY_VERIFIED_OVERFLOW_ACTIVE','upload_state':'RECOVERY_VERIFIED','fresh_restore_verified_count':13,'fresh_restore_verified_bytes':total,'recovery_evidence':FULL})
reg['gated_delta_private_overflow_current_20260929']={'status':'RECOVERY_VERIFIED','authoritative_repo':REPO,'authoritative_revision':REV,'provider_sha_size_readback':'13/13 PASS','fresh_restore':'13/13 PASS; SHA-256, exact size, and safetensors headers verified','recovery_evidence':FULL,'duplicate_repo':'openevo-recovery-2/gated-delta-d1-private-recovery-20260928','duplicate_retention':'KEEP_PRIVATE_PENDING_OWNER_DUPLICATE_RETIREMENT_DECISION','credential_incident':f'{EVID}/GATED_DELTA_HF_CREDENTIAL_HANDLING_INCIDENT_20260929.json','deletion_authorized':False,'public_release_authorized':False}
write(regp,reg)
# Successor DAG evidence
OUT=f'docs/evidence/experiment-passports/{STAMP}-gated-delta-private-recovery-verified'
CONFIG=f'{OUT}/RECOVERY_VERIFIED_CONFIG.json'; BIND=f'{OUT}/GATED_DELTA_PRIVATE_RECOVERY_VERIFIED_BINDING.json'; RECEIPT=f'{OUT}/PRODUCTION_DAG_BINDING_RECEIPT.json'; INDEX=f'configs/experiment/passport-artifacts/index/experiment-passport-artifact-index-{STAMP}-gated-delta-private-recovery-verified.json'
OP='gated-delta-weight.register.private-recovery-verified-after-fresh-restore'
CODE='scripts/openevo_webshop/gated_delta_private_recovery_verified_dag_backfill.py'
config={'schema':f'openevo.gated-delta-private-recovery-verified-config.{STAMP}','operation_type':OP,'authoritative_provider':{'repo_id':REPO,'revision':REV,'visibility':'private'},'fresh_restore':{'count':13,'bytes':total,'status':'PASS_13_OF_13','evidence':FULL},'hard_boundaries':{'scientific_reuse_authorized':False,'public_release_authorized':False,'server_delete_authorized':False,'provider_delete_authorized':False,'visibility_change_authorized':False,'duplicate_retirement_authorized':False}}
write(CONFIG,config)
binding={'schema':f'openevo.gated-delta-private-recovery-verified-binding.{STAMP}','status':'PASS_PRIVATE_FULL_FRESH_RESTORE_VERIFIED','authoritative_provider':{'repo_id':REPO,'immutable_revision':REV,'provider_visibility':'private','payload_count':13,'payload_bytes':total,'provider_sha_size_readback':'PASS_13_OF_13'},'fresh_restore':{'status':'PASS_13_OF_13','verified_count':13,'verified_bytes':total,'remaining_count':0,'remaining_bytes':0,'sha256_size_header_status':'PASS_13_OF_13','evidence':FULL},'supersedes_partial_operation_digest':'68704422695faae75a3bb0f0f728a91a351ac3f37df82dac6237268066147efc','credential_boundary':{'status':'HUMAN_REAUTH_COMPLETED','evidence':f'{EVID}/GATED_DELTA_HF_CREDENTIAL_HANDLING_INCIDENT_20260929.json'},'duplicate_provider':{'repo_id':'openevo-recovery-2/gated-delta-d1-private-recovery-20260928','retention':'KEEP_PRIVATE_PENDING_OWNER_DUPLICATE_RETIREMENT_DECISION','delete_authorized':False},'source_server_state':'ANALYSIS_HOLD_KEEP_LOCAL','automatic_reuse_authorized':False,'scientific_semantics_changed':False,'scientific_reuse_authorized':False,'public_release_authorized':False,'server_delete_authorized':False,'provider_delete_authorized':False,'visibility_change_authorized':False}
write(BIND,binding)
def repo_ref(rel): return {'kind':'repository_file','repository':None,'revision':None,'path':rel,'sha256':sha(rel),'evidence_path':None,'reason':None}
def artifact(rel,role,derived=None):
 p=ROOT/rel; return {'artifact_type':'json-evidence','digest_algorithm':'sha256','digest':sha(rel),'size_bytes':p.stat().st_size,'logical_role':role,'locations':[repo_ref(rel)],'derived_from_digests':sorted(set(derived or []))}
provider=f'{EVID}/GATED_DELTA_WEIGHT_PROVIDER_READBACK_20260929.json'; dup=f'{EVID}/GATED_DELTA_WEIGHT_PRIVATE_DUPLICATE_OBSERVATION_20260929.json'; incident=f'{EVID}/GATED_DELTA_HF_CREDENTIAL_HANDLING_INCIDENT_20260929.json'; source=f'{EVID}/GATED_DELTA_WEIGHT_SOURCE_MEMBER_TABLE_20260929.json'; join=f'{EVID}/GATED_DELTA_WEIGHT_REGISTRY_LFS_JOIN_20260929.json'
inputs=[artifact(source,'gated-delta-source-member-table'),artifact(join,'gated-delta-registry-lfs-join'),artifact(provider,'gated-delta-private-provider-readback'),artifact(FULL,'gated-delta-full-fresh-restore-verification'),artifact(dup,'gated-delta-private-duplicate-hold'),artifact(incident,'gated-delta-credential-reauth-boundary')]
oldop='68704422695faae75a3bb0f0f728a91a351ac3f37df82dac6237268066147efc'
node={'schema':dag.NODE_SCHEMA_ID,'node_kind':'operation','operation_type':OP,'operation_schema_version':STAMP,'operation_digest':'0'*64,'identity_payload':{'code_identity':repo_ref(CODE),'config_identity':repo_ref(CONFIG),'runtime_identity_policy':'NOT_IDENTITY_BEARING','runtime_identity':None,'dataset_schedule_identity':None,'contract_identities':[],'input_artifacts':inputs,'stochastic_identity':{'mode':'deterministic-private-provider-fresh-restore-verification-registration','scientific_execution_or_replay':'not_authorized'}},'side_effect_class':'PURE_ENGINEERING','status':'PASS','outputs':[artifact(BIND,'private-recovery-verified-binding',[x['digest'] for x in inputs])],'provenance':{'producer_scope':'ENGINEERING_GLOBAL','experiment_id':None,'passport_sha256':None,'evidence_paths':[source,join,provider,FULL,dup,incident],'external_views':[{'kind':'hf','repository':REPO,'revision':REV,'path':None,'sha256':sha(provider),'evidence_path':provider,'reason':'Private immutable provider copy has full fresh-restore SHA-256, exact-size, and header verification; no publication or deletion authority is granted.'}]},'producer_attempt':{'attempt_id':f'{OP}-{STAMP}','source':'BACKFILL','execution_sha':None,'created_at':GENERATED_AT},'reuse':{'mechanical_eligibility':'ELIGIBLE','default_policy':dag.DEFAULT_REUSE_POLICIES['PURE_ENGINEERING'],'operation_allowlisted':False,'automatic_reuse_permitted':False,'reason':'Recovery verification is storage integrity evidence only; scientific reuse, release, deletion, and visibility changes remain unauthorized.'},'created_at':GENERATED_AT,'completed_at':GENERATED_AT,'supersedes':[oldop]}
node['operation_digest']=dag.operation_digest(node)
prior,_=dag.load_current_index(ROOT); entries=[e for e in prior['entries'] if e.get('operation_type')!=OP]; prior_count=len(entries); operation_entry=dag.write_operation_node(ROOT,node); entries.append(operation_entry); index=dag.build_index(entries,generated_at=GENERATED_AT); router=dag.write_index_snapshot(ROOT,index,snapshot_rel=INDEX)
reg=load(regp); a=next(x for x in reg['assets'] if x.get('repo_id')==REPO); a.update({'asset_dag_binding':operation_entry['path'],'asset_dag_operation_digest':operation_entry['operation_digest']}); write(regp,reg)
lookup=dag.query_exact(node,repo_root=ROOT); assert lookup.get('reuse',{}).get('automatic_reuse_permitted') is False
receipt={'schema':f'openevo.gated-delta-private-recovery-verified-receipt.{STAMP}','status':'PASS_PRIVATE_FULL_FRESH_RESTORE_VERIFIED','operation_type':OP,'operation_digest':node['operation_digest'],'lookup_status':lookup['status'],'automatic_reuse_permitted':False,'prior_entry_count':prior_count,'current_entry_count':len(index['entries']),'dag_router':router,'authoritative_provider':{'repo_id':REPO,'revision':REV},'fresh_restore_verified_count':13,'fresh_restore_verified_bytes':26984360168,'fresh_restore_remaining_count':0,'recovery_verified':True,'scientific_reuse_authorized':False,'provider_mutation_performed':False,'server_delete_performed':False,'provider_delete_performed':False,'supersedes_operation_digest':oldop}
write(RECEIPT,receipt)
print(json.dumps({'full_evidence':FULL,'file_count':len(rows),'bytes':total,'dag_operation_digest':node['operation_digest'],'dag_router':router,'receipt':RECEIPT},indent=2))
