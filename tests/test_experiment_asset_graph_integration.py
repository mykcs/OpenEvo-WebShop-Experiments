import importlib.util, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'docs/evidence/server-audits/2026-09-03/cross-platform-asset-governance-dry-run/ASSET_GRAPH_INTEGRATION_SUMMARY.json'
CATALOG=ROOT/'configs/experiment/asset-registry/public-current-catalog.v1.json'


def load_script(name,path):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def test_public_asset_graph_summary_is_exact_snapshot():
    x=json.loads(PUBLIC.read_text())
    assert x['status']=='PASS' and x['provider_observation_count']==1608
    assert x['provider_counts']=={'server':242,'wandb':1156,'huggingface':27,'kaggle':183}
    assert x['sensitive_hold_count']==3 and x['open_evo_credential_transport_count']==2
    assert x['secret_contents_read'] is False and x['mutation_summary']['scientific_reruns']==0


def test_public_catalog_has_no_private_server_binding():
    x=json.loads(CATALOG.read_text()); assert x['status']=='GENERATED_PUBLIC_SAFE'
    assert x['state_model_policy'].endswith('RESEARCH_ASSET_RECOVERY_AND_DISCLOSURE_POLICY.zh-CN.md')
    for n in x['nodes']:
        assert n['disclosure_state']=='PUBLIC'
        assert all(b['provider']!='server' for b in n['provider_bindings'])
        assert all(not b['provider_id'].startswith('kyo-myk/') for b in n['provider_bindings'])
        assert all(b['disclosure_state']=='PUBLIC' for b in n['provider_bindings'])
        assert all(b['provider_visibility']=='public' for b in n['provider_bindings'])
def test_pr270_stage1_releases_declare_asset_contract():
    base=ROOT/'configs/experiment/campaigns/202609030400/releases'
    for arm in ('qwen25-3b','qwen3-1p7b'):
        r=json.loads((base/f'{arm}-stage1a.json').read_text())
        c=r['asset_contract']; assert c['registry_policy']=='observation-first-canonical-if-proven'
        ids={x['canonical_id'] for x in c['consumes']}
        assert 'openevo--asset--dataset--seed-webshop-v1' in ids
        assert 'openevo--asset--environment--seed-webshop-v1' in ids
        assert 'openevo--asset--runtime--h144-scientific' in ids
        assert c['produces'] and c['recovery'] and c['telemetry']


def test_qwen3_model_and_runtime_assets_are_content_pinned():
    model=json.loads((ROOT/'configs/assets/models/qwen3-1.7b.json').read_text())
    assert model['revision']=='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
    assert model['required_files_sha256']['model-00001-of-00002.safetensors']=='169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5'
    assert model['required_files_sha256']['model-00002-of-00002.safetensors']=='912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9'
    runtime=json.loads((ROOT/'configs/assets/runtimes/h144-scientific-runtime-v1.json').read_text())
    assert runtime['image_digest']=='sha256:d0f8796010899ae41a7af8ae3f44f47b96dea73b297b5af124049b3ae88e2850'


def test_observation_generator_never_assigns_canonical_identity():
    mod=load_script('asset_builder',ROOT/'scripts/governance/build_experiment_asset_registry.py')
    item={'provider':'kaggle','kind':'notebook','provider_id':'mykcs01/example','is_openevo_related':True}
    row=mod.observation(item,'0'*64,set(),1.0)
    assert row['canonical_id'] is None and row['resolution_state']=='unresolved'
    secret=mod.observation(item|{'provider_id':'mykcs01/secret'},'0'*64,{'mykcs01/secret'},1.0)
    assert secret['resolution_state']=='sensitive-hold' and secret['security_class']=='non-scientific-credential-transport'


def test_public_catalog_routes_current_202609030400_stage2_authority():
    x=json.loads(CATALOG.read_text()); by={n['canonical_id']:n for n in x['nodes']}
    assert x['source_current_campaign_manifest_sha256']
    assert x['source_private_graph_role'].startswith('historical provider-observation snapshot')
    assert by['202609030400-ceiling1-autonomous-v1']['scientific_stage']=='stage2'
    assert by['202609030400-ceiling1-autonomous-v1--qwen3-1p7b--stage2-vnext']['scientific_lifecycle']=='running'
    assert by['202609030400-ceiling1-autonomous-v1--qwen25-3b--stage2-vnext']['scientific_lifecycle']=='authorized'
    assert '202609030400-ceiling1-autonomous-v1--text-memory-a2' in by
    assert '202609030400-ceiling1-autonomous-v1--skill-capacity' in by
    assert '202609030400-ceiling1-autonomous-v1--stage2-shared-small-arm' in by
    assert not any('202609021800' in n['canonical_id'] for n in x['nodes'])


def test_public_catalog_prefers_disclosure_over_legacy_visibility():
    mod=load_script('public_catalog',ROOT/'scripts/governance/project_public_asset_catalog.py')
    node={
        'canonical_id':'asset-private','node_level':'asset','display_title_en':'Private Asset',
        'visibility':'public','disclosure_state':'PRIVATE','aliases':[],
        'source_authorities':[],'provider_bindings':[],
    }
    assert mod.public_node(node) is None

    node['canonical_id']='asset-public'
    node['disclosure_state']='PUBLIC'
    node['provider_bindings']=[{
        'provider':'huggingface','provider_id':'example-account/example','binding_role':'model',
    }]
    projected=mod.public_node(node)
    assert projected is not None
    assert projected['disclosure_state']=='PUBLIC'
    assert projected['provider_bindings']==[]

    node['provider_bindings'][0]['provider_visibility']='public'
    node['provider_bindings'][0]['disclosure_state']='PUBLIC'
    projected=mod.public_node(node)
    assert projected['provider_bindings'][0]['provider_id']=='example-account/example'


def test_public_catalog_legacy_visibility_fallback_is_read_compatible():
    mod=load_script('public_catalog_legacy',ROOT/'scripts/governance/project_public_asset_catalog.py')
    node={
        'canonical_id':'legacy-public','node_level':'asset','display_title_en':'Legacy Public',
        'visibility':'public','aliases':[],'source_authorities':[],'provider_bindings':[],
    }
    projected=mod.public_node(node)
    assert projected is not None
    assert projected['disclosure_state']=='PUBLIC'


def test_asset_registry_timestamped_schemas_separate_recovery_disclosure_and_provider_visibility():
    graph_legacy=json.loads((ROOT/'schemas/experiment-asset-graph-node.schema.json').read_text())
    graph_current=json.loads((ROOT/'schemas/experiment-asset-graph-node-202609201821.schema.json').read_text())
    record_legacy=json.loads((ROOT/'schemas/cross-platform-experiment-asset-record.schema.json').read_text())
    record_current=json.loads((ROOT/'schemas/cross-platform-experiment-asset-record-202609201821.schema.json').read_text())
    template=json.loads((ROOT/'templates/cross-platform-experiment-asset-record.example.json').read_text())

    assert graph_legacy['properties']['schema']['const']=='openevo.experiment-asset-graph-node.v1'  # legacy-revision-compat
    assert record_legacy['properties']['schema']['const']=='openevo.cross-platform-experiment-asset-record.v1'  # legacy-revision-compat
    assert graph_current['properties']['schema']['const']=='openevo.experiment-asset-graph-node.202609201821'
    assert record_current['properties']['schema']['const']=='openevo.cross-platform-experiment-asset-record.202609201821'

    for schema in (graph_current, record_current):
        required=set(schema['required'])
        assert {'state_model_policy','recovery_state','recoverability_class','disclosure_state','provider_visibility'} <= required
        assert schema['properties']['state_model_policy']['const'].endswith('RESEARCH_ASSET_RECOVERY_AND_DISCLOSURE_POLICY.zh-CN.md')
        assert 'visibility' not in required
        assert 'visibility' not in schema['properties']

    assert template['schema']=='openevo.cross-platform-experiment-asset-record.202609201821'
    assert template['state_model_policy'].endswith('RESEARCH_ASSET_RECOVERY_AND_DISCLOSURE_POLICY.zh-CN.md')
    assert template['recovery_state']=='RECOVERY_VERIFIED'
    assert template['disclosure_state']=='PUBLIC'
    assert template['provider_visibility']=='public'
    assert 'visibility' not in template
