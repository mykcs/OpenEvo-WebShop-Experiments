#!/usr/bin/env python3
"""Fail-closed end-to-end campaign state machine for Ceiling-1.0.

This layer chooses only the next legally authorized phase action. Scientific
execution remains delegated to phase-frozen launchers / ``oe run-campaign``.
"""
from __future__ import annotations
import argparse, contextlib, dataclasses, fcntl, glob, hashlib, json, os, re, shutil, socket, subprocess, sys, time
from pathlib import Path
from typing import Any

CEILING1_PROGRAM_CONTRACT_V1="openevo.ceiling1-e2e.v1"
CEILING1_PROGRAM_CONTRACT_V2="openevo.ceiling1-e2e.v2"
CEILING1_PROGRAM_CONTRACT=CEILING1_PROGRAM_CONTRACT_V2
CEILING1_PHASE_DAG_V1=[
    ("stage1a_collection",[]),("stage1b_minimax",["stage1a_collection"]),("stage1c_opsd",["stage1b_minimax"]),
    ("stage1d_text_memory",["stage1c_opsd"]),("stage1e_skill_bundle",["stage1d_text_memory"]),("stage1f_agent_system",["stage1e_skill_bundle"]),
    ("stage1g_closeout",["stage1f_agent_system"]),("stage2_vnext",["stage1g_closeout"]),("d1_gdn_diagnostic",["stage2_vnext"]),
    ("final_evaluation",["d1_gdn_diagnostic"]),("final_analysis",["final_evaluation"]),
]
CEILING1_PHASE_DAG_V2=[
    ("stage1a_collection",[]),
    ("stage1a_postcollection_audit",["stage1a_collection"]),
    ("stage1b_minimax",["stage1a_postcollection_audit"]),
    ("stage1b_teacher_evidence",["stage1b_minimax"]),
    ("stage1c_opsd",["stage1b_teacher_evidence"]),
    ("stage1d_text_memory",["stage1c_opsd"]),
    ("stage1e_skill_bundle",["stage1d_text_memory"]),
    ("stage1f_agent_system",["stage1e_skill_bundle"]),
    ("stage1g_closeout",["stage1f_agent_system"]),
    ("stage2_vnext",["stage1g_closeout"]),
    ("stage2_seal",["stage2_vnext"]),
    ("d1_gdn_diagnostic",["stage2_seal"]),
    ("final_evaluation",["d1_gdn_diagnostic"]),
    ("final_analysis",["final_evaluation"]),
    ("cross_arm_reconciliation",["final_analysis"]),
    ("publication_archive_closeout",["cross_arm_reconciliation"]),
]
CEILING1_PHASE_DAG=CEILING1_PHASE_DAG_V2
CEILING1_PHASE_DAGS={CEILING1_PROGRAM_CONTRACT_V1:CEILING1_PHASE_DAG_V1,CEILING1_PROGRAM_CONTRACT_V2:CEILING1_PHASE_DAG_V2}
CEILING1_PROGRAM_CONTRACTS=set(CEILING1_PHASE_DAGS)
CEILING1_ARM_CONTRACT={
    "qwen25-3b":{"model_family":"qwen2.5","model_id":"Qwen/Qwen2.5-3B-Instruct","model_revision":"aa8e72537993ba99e69dfaafa59ed015b17504d1","stage1_thinking":None},
    "qwen3-1p7b":{"model_family":"qwen3","model_id":"Qwen/Qwen3-1.7B","model_revision":"70d244cc86ccca08cf5af4e1e306ecf908b1ad5e","stage1_thinking":False},
}

STATES = {
    "WAITING_DEPENDENCY", "READY", "RUNNING", "RESUME_REQUIRED", "COMPLETE",
    "BLOCKED_SOURCE", "BLOCKED_AUTHORITY", "BLOCKED_RESOURCE", "BLOCKED_SCIENCE",
    "ENGINEERING_RETRYABLE", "MANUAL_REVIEW_REQUIRED", "CAMPAIGN_COMPLETE",
    "HOLD_OPERATOR", "WAITING_LIVE_VERIFICATION", "BLOCKED_UPSTREAM", "PAUSED",
}

class CampaignError(RuntimeError): pass

def campaign_state_root(manifest: dict[str, Any]) -> Path:
    value=manifest.get("control_state_root")
    if not value: raise CampaignError("campaign manifest missing control_state_root")
    return Path(str(value))

def append_transition_journal(manifest: dict[str, Any], record: dict[str, Any]) -> Path:
    root=campaign_state_root(manifest); root.mkdir(parents=True,exist_ok=True)
    path=root/"TRANSITIONS.jsonl"
    payload={"schema":"openevo.autonomous-campaign-transition.v1","campaign_id":manifest["campaign_id"],**record}
    with path.open("a",encoding="utf-8") as f:
        f.write(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n"); f.flush(); os.fsync(f.fileno())
    return path

@contextlib.contextmanager
def campaign_lock(manifest: dict[str, Any]):
    root=campaign_state_root(manifest); root.mkdir(parents=True,exist_ok=True)
    path=root/"campaign.lock"; fd=os.open(path,os.O_CREAT|os.O_RDWR,0o600)
    try:
        try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc: raise CampaignError(f"campaign controller lock is already held: {path}") from exc
        lease={"schema":"openevo.autonomous-campaign-lock.v1","campaign_id":manifest["campaign_id"],"pid":os.getpid(),"host":socket.gethostname(),"acquired_at_epoch":time.time()}
        os.ftruncate(fd,0); os.write(fd,(json.dumps(lease,sort_keys=True)+"\n").encode()); os.fsync(fd)
        yield path
    finally:
        try: fcntl.flock(fd,fcntl.LOCK_UN)
        finally: os.close(fd)

def assert_campaign_execution_authority(manifest:dict[str,Any], resolution:dict[str,Any]) -> None:
    if manifest.get("campaign_id")=="202609030400-ceiling1-autonomous-v1" and resolution.get("kind")!="router":  # legacy-revision-compat
        raise CampaignError("Ceiling Master Plan execution must resolve through the stable hash-bound router")
    if manifest.get("activation_authority") is not True:
        raise CampaignError("master campaign has no execution activation authority")
    if resolution.get("router_execution_activation_authority") is not True:
        raise CampaignError("stable campaign router has no execution activation authority")

def read_transition_journal(manifest:dict[str,Any]) -> tuple[list[dict[str,Any]],str|None]:
    path=campaign_state_root(manifest)/"TRANSITIONS.jsonl"
    if not path.exists(): return [],None
    rows=[]
    try:
        for line_no,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
            if not line.strip(): continue
            row=json.loads(line)
            if not isinstance(row,dict) or row.get("schema")!="openevo.autonomous-campaign-transition.v1": return [],f"transition journal invalid row {line_no}"
            if row.get("campaign_id")!=manifest.get("campaign_id"): return [],f"transition journal campaign mismatch at row {line_no}"
            rows.append(row)
    except Exception as exc: return [],f"transition journal unreadable: {type(exc).__name__}: {exc}"
    return rows,None

def transition_journal_health(manifest:dict[str,Any], status:dict[str,Any]) -> dict[str,Any]:
    rows,problem=read_transition_journal(manifest)
    if problem: return {"status":"BLOCKED","reason":problem,"open_actions":[],"path":str(campaign_state_root(manifest)/'TRANSITIONS.jsonl')}
    open_actions={}
    for row in rows:
        aid=row.get("action_id")
        if not aid: continue
        event=row.get("event")
        if event=="ACTION_STARTED":
            if aid in open_actions: return {"status":"BLOCKED","reason":f"duplicate unclosed ACTION_STARTED {aid}","open_actions":list(open_actions.values())}
            open_actions[aid]=row
        elif event in {"ACTION_FINISHED","ACTION_EXCEPTION","ACTION_RECONCILED"}:
            open_actions.pop(aid,None)
    if not open_actions: return {"status":"PASS","reason":"no orphan transition","open_actions":[],"path":str(campaign_state_root(manifest)/'TRANSITIONS.jsonl')}
    by={(arm,row['phase']):row for arm,items in status.get("arms",{}).items() for row in items}
    recoverable=[]; blocked=[]
    for aid,start in open_actions.items():
        current=by.get((start.get("arm"),start.get("phase")))
        observed=current.get("state") if current else None
        item={"action_id":aid,"arm":start.get("arm"),"phase":start.get("phase"),"state_now":observed}
        if observed in {"RUNNING","RESUME_REQUIRED","ENGINEERING_RETRYABLE","COMPLETE"}: recoverable.append(item)
        else: blocked.append(item)
    if blocked: return {"status":"BLOCKED","reason":"orphan transition has no mechanically provable runtime/receipt successor state","open_actions":recoverable+blocked,"blocked_actions":blocked}
    return {"status":"RECOVERABLE","reason":"orphan transition is explained by current runtime/receipt state","open_actions":recoverable}

def reconcile_recoverable_journal(manifest:dict[str,Any], health:dict[str,Any]) -> None:
    if health.get("status")!="RECOVERABLE": return
    for item in health.get("open_actions") or []:
        append_transition_journal(manifest,{"event":"ACTION_RECONCILED","action_id":item["action_id"],"arm":item.get("arm"),"phase":item.get("phase"),"observed_state":item.get("state_now"),"reconciled_at_epoch":time.time(),"scientific_semantics_changed":False})

def transition_action_id(manifest:dict[str,Any],arm:str,row:dict[str,Any])->str:
    payload={"campaign_id":manifest["campaign_id"],"arm":arm,"phase":row.get("phase"),"state":row.get("state"),"release_path":row.get("release_path"),"next_action":row.get("next_action")}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

@dataclasses.dataclass
class PhaseResult:
    arm: str; phase: str; state: str; detail: str
    progress: dict[str, Any] = dataclasses.field(default_factory=dict)
    next_action: dict[str, Any] | None = None
    release_path: str | None = None
    def as_dict(self) -> dict[str, Any]: return dataclasses.asdict(self)

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def load(path: Path) -> dict[str, Any]:
    value=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value,dict): raise CampaignError(f"JSON object required: {path}")
    return value

def resolve_repository_current_stage2_authority(repo:Path, router:dict[str,Any], manifest:dict[str,Any]) -> dict[str,Any]:
    ref=router.get("repository_current_stage2_runtime_authority")
    if not isinstance(ref,dict): raise CampaignError("stable router has no repository-current Stage2 runtime authority")
    rel=str(ref.get("path") or "")
    path=Path(rel); path=path if path.is_absolute() else repo/path
    if not path.is_file(): raise CampaignError(f"repository-current Stage2 runtime authority missing: {path}")
    expected=str(ref.get("sha256") or ""); observed=sha(path)
    if not expected or observed!=expected: raise CampaignError(f"Stage2 runtime authority SHA drift: {observed} != {expected}")
    authority=load(path)
    expected_schema=str(ref.get("schema") or "")
    if expected_schema and authority.get("schema")!=expected_schema: raise CampaignError("Stage2 runtime authority schema drift")
    expected_status=str(ref.get("status") or "")
    if expected_status and authority.get("status")!=expected_status: raise CampaignError("Stage2 runtime authority status drift")
    manifest_ref=manifest.get("repository_current_stage2_runtime_authority")
    if not isinstance(manifest_ref,dict): raise CampaignError("campaign manifest has no repository-current Stage2 runtime authority")
    if manifest_ref.get("path")!=rel or manifest_ref.get("sha256")!=expected: raise CampaignError("campaign/router Stage2 runtime authority drift")
    bindings=manifest.get("bindings") or {}
    if bindings.get("shared_stage2_runtime_authority_path")!=rel or bindings.get("shared_stage2_runtime_authority_sha256")!=expected:
        raise CampaignError("campaign Stage2 runtime authority binding drift")
    return {"path":str(path),"sha256":observed,"schema":authority.get("schema"),"status":authority.get("status")}

def resolve_manifest(repo: Path, manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path=manifest_path if manifest_path.is_absolute() else repo/manifest_path
    raw=load(path)
    if raw.get("schema")=="openevo.autonomous-campaign.v1":
        return raw,{"kind":"direct","router_path":None,"manifest_path":str(path),"manifest_sha256":sha(path),"binding_role":raw.get("instance_role","campaign_instance")}
    if raw.get("schema")!="openevo.autonomous-campaign-router.v1":
        raise CampaignError(f"unsupported autonomous manifest/router schema: {raw.get('schema')!r}")
    active=raw.get("active_manifest")
    if not isinstance(active,dict): raise CampaignError("autonomous campaign router has no active_manifest binding")
    target=Path(str(active.get("path","")))
    target=target if target.is_absolute() else repo/target
    if not target.is_file(): raise CampaignError(f"router active manifest missing: {target}")
    expected=str(active.get("sha256") or "")
    observed=sha(target)
    if not expected or observed!=expected: raise CampaignError(f"router active manifest SHA drift: {observed} != {expected}")
    manifest=load(target)
    if manifest.get("schema")!="openevo.autonomous-campaign.v1": raise CampaignError("router target is not an autonomous campaign manifest")
    expected_id=active.get("campaign_id")
    if expected_id and manifest.get("campaign_id")!=expected_id: raise CampaignError("router campaign_id binding drift")
    expected_program=active.get("program_contract")
    if expected_program and manifest.get("program_contract")!=expected_program: raise CampaignError("router program_contract binding drift")
    stage2_authority=None
    if manifest.get("program_contract") in CEILING1_PROGRAM_CONTRACTS:
        stage2_authority=resolve_repository_current_stage2_authority(repo,raw,manifest)
    return manifest,{"kind":"router","router_path":str(path),"router_sha256":sha(path),"manifest_path":str(target),"manifest_sha256":observed,"binding_role":active.get("binding_role","active"),"campaign_id":manifest.get("campaign_id"),"router_execution_activation_authority":raw.get("execution_activation_authority") is True,"repository_current_stage2_runtime_authority":stage2_authority}
def dot(value: Any, key: str) -> Any:
    cur=value
    for part in key.split("."):
        if not isinstance(cur,dict) or part not in cur: return None
        cur=cur[part]
    return cur

def expand(value: Any, ctx: dict[str, Any]) -> Any:
    # Scientific phase releases are concrete, hash-reviewable artifacts. Runtime
    # interpolation is deliberately forbidden: literal shell/Python/JSON braces
    # must survive unchanged, and a new arm/lineage gets a new concrete release.
    if isinstance(value,list): return [expand(x,ctx) for x in value]
    if isinstance(value,dict): return {k:expand(v,ctx) for k,v in value.items()}
    return value

def command_output(argv:list[str]) -> str:
    return subprocess.check_output(argv,text=True,stderr=subprocess.STDOUT).strip()

def git_source_status(source:dict[str,Any]) -> tuple[bool,str]:
    checkout=Path(source["checkout"]); expected=str(source["execution_sha"])
    if not checkout.exists(): return False,f"source checkout missing: {checkout}"
    try:
        safe=["git","-c",f"safe.directory={checkout}","-C",str(checkout)]
        head=command_output([*safe,"rev-parse","HEAD"])
        dirty=command_output([*safe,"status","--porcelain"])
    except Exception as exc: return False,f"source git check failed: {type(exc).__name__}: {exc}"
    if head!=expected: return False,f"source HEAD drift: {head} != {expected}"
    if dirty: return False,"source checkout is dirty"
    return True,"source exact/clean"

def docker_state(name:str|None) -> dict[str,Any] | None:
    if not name: return None
    try:
        raw=command_output(["docker","inspect",name,"--format","{{json .State}}"])
        d=json.loads(raw); return d if isinstance(d,dict) else None
    except Exception: return None

def count_jsonl(path:Path, spec:dict[str,Any]) -> tuple[dict[str,Any],str|None]:
    rows=0; good=0; ids=set(); duplicates=0
    id_key=spec.get("identity_key"); true_key=spec.get("required_true_key")
    if not path.exists(): return {"rows":0,"good":0,"exists":False},None
    try:
        with path.open(encoding="utf-8") as f:
            for line_no,line in enumerate(f,1):
                if not line.strip(): continue
                obj=json.loads(line); rows+=1
                if not isinstance(obj,dict): return {},f"non-object JSONL row {line_no}"
                if id_key:
                    ident=dot(obj,id_key)
                    if ident is None: return {},f"missing identity {id_key} at row {line_no}"
                    if ident in ids: duplicates+=1
                    ids.add(ident)
                good += int(true_key is None or dot(obj,true_key) is True)
    except Exception as exc: return {},f"JSONL unreadable: {type(exc).__name__}: {exc}"
    return {"rows":rows,"good":good,"unique_ids":len(ids),"duplicates":duplicates,"exists":True},None
def count_glob_files(spec:dict[str,Any]) -> tuple[dict[str,Any],str|None]:
    pattern=str(spec.get("path_glob") or "")
    files=[Path(x) for x in sorted(glob.glob(pattern))]
    ids=set(); duplicates=0
    rx=re.compile(str(spec.get("identity_regex"))) if spec.get("identity_regex") else None
    for path in files:
        if rx:
            match=rx.search(path.name)
            if not match: return {},f"progress filename identity mismatch: {path.name}"
            ident=match.group(1) if match.groups() else match.group(0)
            if ident in ids: duplicates+=1
            ids.add(ident)
    return {"files":len(files),"good":len(files),"unique_ids":len(ids) if rx else len(files),"duplicates":duplicates,"exists":bool(files)},None

def runtime_states(runtime:dict[str,Any]) -> list[tuple[str,dict[str,Any]|None]]:
    names=[]
    if runtime.get("container_name"): names.append(str(runtime["container_name"]))
    names.extend(str(x) for x in (runtime.get("container_names") or []))
    return [(name,docker_state(name)) for name in names]

def validate_receipt(path:Path, spec:dict[str,Any]) -> tuple[bool,str]:
    if not path.is_file(): return False,"completion receipt absent"
    try: value=load(path)
    except Exception as exc: return False,f"completion receipt unreadable: {exc}"
    for key,expected in (spec.get("required_equal") or {}).items():
        if dot(value,key)!=expected: return False,f"completion receipt {key}={dot(value,key)!r}, expected {expected!r}"
    expected_sha=spec.get("sha256")
    if expected_sha and sha(path)!=expected_sha: return False,"completion receipt SHA drift"
    return True,"completion receipt PASS"

def disk_ok(path:Path,min_free_bytes:int) -> tuple[bool,str]:
    probe=path if path.exists() else path.parent
    while not probe.exists() and probe!=probe.parent: probe=probe.parent
    if not probe.exists(): return False,f"cannot resolve disk root for {path}"
    free=shutil.disk_usage(probe).free
    return free>=min_free_bytes,f"disk free={free} required={min_free_bytes}"

def validate_authority(repo:Path, release:dict[str,Any]) -> tuple[bool,str]:
    auth=release.get("scientific_authority")
    if not isinstance(auth,dict): return False,"scientific authority is absent"
    path=Path(auth.get("path","")); path=path if path.is_absolute() else repo/path
    if not path.is_file(): return False,f"scientific authority missing: {path}"
    expected=auth.get("sha256")
    if expected and sha(path)!=expected: return False,"scientific authority SHA drift"
    try: value=load(path)
    except Exception as exc: return False,f"scientific authority unreadable: {exc}"
    for key,expected_value in (auth.get("required_equal") or {}).items():
        if dot(value,key)!=expected_value: return False,f"scientific authority {key} mismatch"
    return True,"scientific authority exact"

PHASE_BUDGET_KEYS=("formal_task_rollouts_max","external_teacher_requests_max","optimizer_steps_max","local_reflector_requests_max","final_panel_tasks_max")

def validate_asset_contract(value:Any) -> tuple[bool,str]:
    if not isinstance(value,dict): return False,"phase release asset_contract must be an object"
    if value.get("registry_policy")!="observation-first-canonical-if-proven": return False,"asset_contract registry_policy mismatch"
    for bucket in ("consumes","produces","recovery","telemetry","publication"):
        rows=value.get(bucket)
        if not isinstance(rows,list): return False,f"asset_contract.{bucket} must be an array"
        seen=set()
        for idx,row in enumerate(rows):
            if not isinstance(row,dict): return False,f"asset_contract.{bucket}[{idx}] must be an object"
            cid=row.get("canonical_id"); role=row.get("role")
            if not isinstance(cid,str) or not cid: return False,f"asset_contract.{bucket}[{idx}].canonical_id missing"
            if not isinstance(role,str) or not role: return False,f"asset_contract.{bucket}[{idx}].role missing"
            if cid in seen: return False,f"asset_contract.{bucket} duplicate canonical_id {cid}"
            seen.add(cid)
    return True,"asset contract PASS"

def validate_release_structure(release:dict[str,Any]) -> tuple[bool,str]:
    release_mode=str(release.get("release_mode") or "execution")
    if release_mode not in {"execution","adoption"}: return False,"phase release release_mode must be execution or adoption"
    required=("schema","arm","phase","activation_authority","scientific_authority","budget_contract","completion","retry")
    if release_mode=="execution": required=(*required,"execution_source")
    missing=[k for k in required if k not in release]
    if missing: return False,f"phase release missing required core fields: {missing}"
    if release.get("schema")!="openevo.autonomous-phase-release.v1": return False,"phase release schema mismatch"
    if not isinstance(release.get("activation_authority"),bool): return False,"phase release activation_authority must be boolean"
    source=release.get("execution_source")
    if source is not None:
        checkout=source.get("checkout") if isinstance(source,dict) else None; execution_sha=str(source.get("execution_sha") or "") if isinstance(source,dict) else ""
        if not checkout or len(execution_sha)!=40 or any(c not in "0123456789abcdef" for c in execution_sha): return False,"phase release execution_source must pin checkout + 40-hex execution_sha"
    elif release_mode=="execution":
        return False,"execution phase release requires execution_source"
    if release_mode=="adoption":
        provenance=release.get("historical_execution_provenance") or {}
        if provenance.get("status") not in {"PROVEN","PARTIAL_UNKNOWN_NOT_UPGRADED"}:
            return False,"adoption phase release requires explicit historical_execution_provenance status"
    authority=release.get("scientific_authority") or {}; auth_sha=str(authority.get("sha256") or "")
    if not authority.get("path") or len(auth_sha)!=64 or any(c not in "0123456789abcdef" for c in auth_sha): return False,"phase release scientific_authority must pin path + 64-hex sha256"
    if not isinstance(authority.get("required_equal"),dict) or not authority["required_equal"]: return False,"phase release scientific_authority.required_equal must be non-empty"
    budget=release.get("budget_contract")
    if not isinstance(budget,dict): return False,"phase release budget_contract must be an object"
    missing_budget=[k for k in PHASE_BUDGET_KEYS if k not in budget]
    if missing_budget: return False,f"phase release budget_contract missing keys: {missing_budget}"
    for key in PHASE_BUDGET_KEYS:
        value=budget.get(key)
        if not isinstance(value,int) or isinstance(value,bool) or value<0: return False,f"phase release budget_contract.{key} must be a nonnegative integer"
    completion=release.get("completion")
    if not isinstance(completion,dict) or not completion.get("path") or not isinstance(completion.get("required_equal"),dict) or not completion["required_equal"]:
        return False,"phase release completion must pin path + non-empty required_equal"
    retry=release.get("retry")
    if not isinstance(retry,dict) or not isinstance(retry.get("exact_resume"),bool): return False,"phase release retry.exact_resume must be explicit boolean"
    external=release.get("external_executor")
    if external is not None and release.get("activation_authority") is not False:
        return False,"external executor adoption requires activation_authority=false"
    if release.get("activation_authority") is True:
        action=release.get("run_action")
        if not isinstance(action,dict) or action.get("kind")!="command" or not action.get("argv"):
            return False,"activated phase release requires non-empty command run_action"
    progress=release.get("progress")
    if progress is not None:
        if not isinstance(progress,dict): return False,"phase release progress must be an object"
        kind=progress.get("kind")
        if kind=="jsonl":
            for key in ("path","expected_rows","identity_key","required_true_key"):
                if key not in progress: return False,f"phase release progress missing {key}"
            if not isinstance(progress.get("expected_rows"),int) or progress["expected_rows"]<=0: return False,"phase release progress.expected_rows must be positive integer"
        elif kind=="glob_files":
            if not progress.get("path_glob") or not isinstance(progress.get("expected_files"),int) or progress["expected_files"]<=0:
                return False,"glob_files progress requires path_glob + positive expected_files"
        else: return False,f"unsupported phase release progress kind: {kind!r}"
    runtime=release.get("runtime")
    if runtime is not None:
        if not isinstance(runtime,dict): return False,"phase release runtime must be an object"
        names=[]
        if runtime.get("container_name"): names.append(runtime["container_name"])
        names.extend(runtime.get("container_names") or [])
        if len(names)!=len(set(names)): return False,"phase release runtime container identities must be unique"
    external=release.get("external_executor")
    if external is not None:
        if not isinstance(external,dict): return False,"external_executor must be an object"
        if release.get("activation_authority") is not False: return False,"external executor adoption requires activation_authority=false"
        if external.get("read_only_adoption") is not True: return False,"external_executor.read_only_adoption must be true"
        if not external.get("state_path"): return False,"external_executor.state_path missing"
        if not isinstance(external.get("required_equal"),dict) or not external["required_equal"]: return False,"external_executor.required_equal must be non-empty"
        mapping=external.get("status_map")
        if not isinstance(mapping,dict) or not mapping: return False,"external_executor.status_map must be non-empty"
        if any(v not in STATES for v in mapping.values()): return False,"external_executor.status_map contains unsupported controller state"
        missing=str(external.get("missing_state") or "")
        if missing not in STATES: return False,"external_executor.missing_state must be a controller state"
        if not external.get("missing_reason"): return False,"external_executor.missing_reason required"
    return True,"phase release core contract PASS"

def release_invariant(manifest:dict[str,Any],arm:str,phase_id:str,release:dict[str,Any]) -> tuple[bool,str]:
    arm_cfg=manifest["arms"][arm]
    budget=release.get("budget_contract") or {}
    zero_budget_phases={"stage1c_opsd","d1_gdn_diagnostic"}
    if phase_id in zero_budget_phases:
        for key in ("formal_task_rollouts_max","external_teacher_requests_max","final_panel_tasks_max"):
            if budget.get(key)!=0: return False,f"{phase_id} requires budget_contract.{key}=0"
    if phase_id=="stage1c_opsd" and arm_cfg.get("model_family")=="qwen3":
        compat=release.get("model_compat") or {}
        if compat.get("enable_thinking") is not False:
            return False,"Qwen3 OPSD release must explicitly freeze enable_thinking=false"
    if phase_id=="stage2_vnext":
        if budget.get("external_teacher_requests_max")!=0 or budget.get("final_panel_tasks_max")!=0:
            return False,"Stage2-vNext forbids external teacher requests and final-panel tasks"
        if arm_cfg.get("model_family")=="qwen3":
            compat=release.get("model_compat") or {}
            if compat.get("enable_thinking") is not False:
                return False,"Qwen3 Stage2 release must explicitly freeze enable_thinking=false"
        if manifest.get("program_contract")==CEILING1_PROGRAM_CONTRACT_V2:
            successor=release.get("engineering_successor") or {}
            ash=str(successor.get("shared_action_envelope_sha256") or "")
            if successor.get("interface_owned_action_opener") is not True or successor.get("applies_to_both_arms") is not True or len(ash)!=64 or any(c not in "0123456789abcdef" for c in ash):
                return False,"v2 Stage2 release must bind the shared interface-owned action-envelope successor"
    if manifest.get("program_contract") in CEILING1_PROGRAM_CONTRACTS:
        if manifest.get("instance_role")!="migration_fixture_not_current_scientific_authority":
            asset_ok,asset_detail=validate_asset_contract(release.get("asset_contract"))
            if not asset_ok: return False,asset_detail
        zero3=("formal_task_rollouts_max","external_teacher_requests_max","final_panel_tasks_max")
        if phase_id=="stage1a_collection":
            expected={"formal_task_rollouts_max":1440,"external_teacher_requests_max":0,"optimizer_steps_max":0,"local_reflector_requests_max":0,"final_panel_tasks_max":0}
            if any(budget.get(k)!=v for k,v in expected.items()): return False,"Ceiling Stage1A budget must be exactly 1440 rollouts and zero teacher/optimizer/reflector/final"
        elif phase_id=="stage1b_minimax":
            expected={"formal_task_rollouts_max":0,"external_teacher_requests_max":1440,"optimizer_steps_max":0,"local_reflector_requests_max":0,"final_panel_tasks_max":0}
            if any(budget.get(k)!=v for k,v in expected.items()): return False,"Ceiling Stage1B budget must be exactly 1440 external teacher analyses and no other scientific budget"
        elif phase_id=="stage1c_opsd":
            if any(budget.get(k)!=0 for k in zero3) or budget.get("local_reflector_requests_max")!=0 or budget.get("optimizer_steps_max",0)<=0: return False,"Ceiling OPSD requires zero new rollout/teacher/reflector/final budget and positive finite optimizer-step cap"
        elif phase_id in {"stage1d_text_memory","stage1e_skill_bundle","stage1f_agent_system"}:
            if budget.get("formal_task_rollouts_max")!=0 or budget.get("external_teacher_requests_max")!=0 or budget.get("optimizer_steps_max")!=0 or budget.get("final_panel_tasks_max")!=0 or budget.get("local_reflector_requests_max",0)<=0: return False,f"{phase_id} requires local-reflector-only finite budget"
        elif phase_id=="stage1g_closeout":
            if any(budget.get(k)!=0 for k in PHASE_BUDGET_KEYS): return False,"Stage1 closeout must consume zero scientific budget"
        elif phase_id=="stage2_vnext":
            if budget.get("formal_task_rollouts_max")!=20480 or budget.get("external_teacher_requests_max")!=0 or budget.get("final_panel_tasks_max")!=0: return False,"Ceiling Stage2-vNext requires exact 20,480 rollout cap and zero external-teacher/final-panel budget"
        elif phase_id=="d1_gdn_diagnostic":
            if any(budget.get(k)!=0 for k in PHASE_BUDGET_KEYS): return False,"D1 diagnostic must consume zero listed scientific budget"
        elif phase_id=="final_evaluation":
            expected={"formal_task_rollouts_max":0,"external_teacher_requests_max":0,"optimizer_steps_max":0,"local_reflector_requests_max":0,"final_panel_tasks_max":128}
            if any(budget.get(k)!=v for k,v in expected.items()): return False,"Ceiling final evaluation budget must be exactly 128 final-panel tasks and zero training/teacher budget"
        elif phase_id=="final_analysis":
            if any(budget.get(k)!=0 for k in PHASE_BUDGET_KEYS): return False,"final analysis must consume zero scientific budget"
    if phase_id=="final_evaluation":
        gate=release.get("final_panel_gate") or {}
        for name in ("stage2_seal","final_panel_authority"):
            spec=gate.get(name) or {}; path=Path(spec.get("path","")) if spec else None
            if not path: return False,f"final evaluation missing {name} gate"
            ok,detail=validate_receipt(path,spec)
            if not ok: return False,f"final evaluation {name} gate failed: {detail}"
    return True,"release invariants PASS"

def load_release(repo:Path, rel:str|None, ctx:dict[str,Any]) -> tuple[dict[str,Any]|None,Path|None,str|None]:
    if not rel: return None,None,"release slot intentionally empty"
    path=Path(expand(rel,ctx)); path=path if path.is_absolute() else repo/path
    if not path.is_file(): return None,path,f"release file missing: {path}"
    release=expand(load(path),ctx)
    if release.get("schema")!="openevo.autonomous-phase-release.v1": return None,path,"phase release schema mismatch"
    return release,path,None

def phase_context(manifest:dict[str,Any], arm:str) -> dict[str,Any]:
    ctx={"campaign_id":manifest["campaign_id"],"arm":arm}
    ctx.update(manifest.get("bindings") or {})
    arm_cfg=manifest["arms"][arm]
    ctx.update(arm_cfg.get("bindings") or {})
    return ctx

def classify_external_executor(release:dict[str,Any], arm:str, phase_id:str, release_path:str) -> PhaseResult|None:
    spec=release.get("external_executor")
    if not isinstance(spec,dict): return None
    path=Path(str(spec.get("state_path") or ""))
    if not path.is_file():
        return PhaseResult(arm,phase_id,str(spec["missing_state"]),str(spec["missing_reason"]),release_path=release_path)
    try: state=load(path)
    except Exception as exc:
        return PhaseResult(arm,phase_id,"MANUAL_REVIEW_REQUIRED",f"external execution-state unreadable: {type(exc).__name__}: {exc}",release_path=release_path)
    for key,expected in (spec.get("required_equal") or {}).items():
        if dot(state,key)!=expected:
            return PhaseResult(arm,phase_id,"MANUAL_REVIEW_REQUIRED",f"external execution-state {key} drift",release_path=release_path)
    consumed=state.get("rollouts_consumed")
    max_rollouts=(release.get("budget_contract") or {}).get("formal_task_rollouts_max")
    if isinstance(consumed,int) and isinstance(max_rollouts,int) and consumed>max_rollouts:
        return PhaseResult(arm,phase_id,"MANUAL_REVIEW_REQUIRED",f"external rollouts_consumed={consumed} exceeds release budget {max_rollouts}",release_path=release_path)
    raw=str(state.get("status") or "")
    mapped=(spec.get("status_map") or {}).get(raw)
    if not mapped:
        return PhaseResult(arm,phase_id,"MANUAL_REVIEW_REQUIRED",f"external execution-state status {raw!r} is not mapped",release_path=release_path)
    detail=f"read-only adopted external executor state={raw}; Master Plan execution remains disabled"
    progress={k:state.get(k) for k in ("next_round","rollouts_consumed","rounds_complete","state_id","stage2_activation_sha256") if k in state}
    return PhaseResult(arm,phase_id,mapped,detail,progress,release_path=release_path)

def classify_phase(repo:Path,manifest:dict[str,Any],arm:str,phase:dict[str,Any],prior:dict[str,PhaseResult]) -> PhaseResult:
    phase_id=phase["id"]; deps=list(phase.get("depends_on") or [])
    waiting=[d for d in deps if prior.get(d) is None or prior[d].state!="COMPLETE"]
    if waiting:
        return PhaseResult(arm,phase_id,"WAITING_DEPENDENCY",f"waiting for {waiting}")
    ctx=phase_context(manifest,arm); release,path,error=load_release(repo,phase.get("release"),ctx)
    if error:
        gate=phase.get("gate") or {}
        if not phase.get("release") and gate:
            state=str(gate.get("state") or "")
            reason=str(gate.get("reason") or "")
            if state not in STATES or not reason:
                return PhaseResult(arm,phase_id,"BLOCKED_SCIENCE","invalid explicit empty-release gate")
            return PhaseResult(arm,phase_id,state,reason)
        return PhaseResult(arm,phase_id,"BLOCKED_SCIENCE",error,release_path=str(path) if path else None)
    assert release is not None
    core_ok,core_detail=validate_release_structure(release)
    if not core_ok: return PhaseResult(arm,phase_id,"BLOCKED_SCIENCE",core_detail,release_path=str(path))
    inv_ok,inv_detail=release_invariant(manifest,arm,phase_id,release)
    if not inv_ok: return PhaseResult(arm,phase_id,"BLOCKED_SCIENCE",inv_detail,release_path=str(path))
    if release.get("arm") not in (arm,"any") or release.get("phase")!=phase_id:
        return PhaseResult(arm,phase_id,"BLOCKED_AUTHORITY","phase release arm/phase mismatch",release_path=str(path))
    ok,detail=validate_authority(repo,release)
    if not ok: return PhaseResult(arm,phase_id,"BLOCKED_AUTHORITY",detail,release_path=str(path))
    if release.get("execution_source") is not None:
        ok,detail=git_source_status(release["execution_source"])
        if not ok: return PhaseResult(arm,phase_id,"BLOCKED_SOURCE",detail,release_path=str(path))
    for input_spec in release.get("inputs") or []:
        input_path=Path(input_spec["path"]); input_path=input_path if input_path.is_absolute() else repo/input_path
        ok,input_detail=validate_receipt(input_path,input_spec)
        if not ok:
            rec=release.get("input_reconciliation") or {}
            receipt_spec=rec.get("receipt") or {}
            rec_path=Path(receipt_spec.get("path","")) if receipt_spec else None
            rec_ok=False
            if rec_path:
                rec_ok,_=validate_receipt(rec_path,receipt_spec)
            if rec_ok:
                continue
            if rec.get("action"):
                return PhaseResult(arm,phase_id,"ENGINEERING_RETRYABLE",f"input drift is eligible for deterministic reconciliation: {input_detail}",next_action=rec["action"],release_path=str(path))
            return PhaseResult(arm,phase_id,"MANUAL_REVIEW_REQUIRED",f"input authority/provenance drift: {input_detail}",release_path=str(path))
    completion=release.get("completion") or {}; completion_path=Path(completion.get("path","")) if completion else None
    if completion_path:
        complete,c_detail=validate_receipt(completion_path,completion)
        if complete: return PhaseResult(arm,phase_id,"COMPLETE",c_detail,release_path=str(path))
    else: c_detail="no completion receipt contract"
    external=classify_external_executor(release,arm,phase_id,str(path))
    if external is not None: return external
    progress_spec=release.get("progress") or {}; progress={}; expected=None; observed=0; good=0
    kind=progress_spec.get("kind")
    if kind=="jsonl":
        progress_path=Path(progress_spec["path"]); progress,problem=count_jsonl(progress_path,progress_spec); expected=int(progress_spec["expected_rows"]); observed=int(progress.get("rows",0)); good=int(progress.get("good",0))
        if problem: return PhaseResult(arm,phase_id,"MANUAL_REVIEW_REQUIRED",problem,progress,release_path=str(path))
    elif kind=="glob_files":
        progress,problem=count_glob_files(progress_spec); expected=int(progress_spec["expected_files"]); observed=int(progress.get("files",0)); good=int(progress.get("good",0))
        if problem: return PhaseResult(arm,phase_id,"MANUAL_REVIEW_REQUIRED",problem,progress,release_path=str(path))
    if expected is not None and (observed>expected or progress.get("duplicates",0)>0 or progress.get("unique_ids",observed)>expected):
        return PhaseResult(arm,phase_id,"MANUAL_REVIEW_REQUIRED","progress denominator/identity violation",progress,release_path=str(path))
    states=runtime_states(release.get("runtime") or {})
    if states:
        progress["runtime"]=[{"container":name,"present":state is not None,"running":bool(state and state.get("Running") is True),"exit_code":state.get("ExitCode") if state else None,"oom_killed":state.get("OOMKilled") if state else None,"started_at":state.get("StartedAt") if state else None,"finished_at":state.get("FinishedAt") if state else None} for name,state in states]
        running=sum(1 for _,state in states if state and state.get("Running") is True)
        progress["runtime_running_count"]=running; progress["runtime_expected_count"]=len(states)
        if running>0:
            return PhaseResult(arm,phase_id,"RUNNING",f"authorized worker group running {running}/{len(states)}",progress,release_path=str(path))
    if expected is not None and observed==expected and good==expected:
        action=release.get("closeout_action")
        return PhaseResult(arm,phase_id,"ENGINEERING_RETRYABLE","work complete; closeout receipt still absent",progress,action,str(path))
    if observed>0:
        if release.get("retry",{}).get("exact_resume") is True:
            return PhaseResult(arm,phase_id,"RESUME_REQUIRED","legal partial output exists; exact resume required",progress,release.get("resume_action") or release.get("run_action"),str(path))
        return PhaseResult(arm,phase_id,"MANUAL_REVIEW_REQUIRED","partial output exists but release has no exact-resume contract",progress,release_path=str(path))
    min_free=int((release.get("resource_gate") or {}).get("min_free_bytes",0))
    if min_free:
        ok,d_detail=disk_ok(Path((release.get("resource_gate") or {}).get("path","/workspace")),min_free)
        if not ok: return PhaseResult(arm,phase_id,"BLOCKED_RESOURCE",d_detail,progress,release_path=str(path))
    if release.get("activation_authority") is not True:
        return PhaseResult(arm,phase_id,"BLOCKED_AUTHORITY","phase release has no execution activation authority",progress,release_path=str(path))
    return PhaseResult(arm,phase_id,"READY","dependencies/source/authority are ready",progress,release.get("run_action"),str(path))
def validate_manifest(repo:Path,manifest:dict[str,Any]) -> None:
    if manifest.get("schema")!="openevo.autonomous-campaign.v1": raise CampaignError("campaign schema mismatch")
    if not isinstance(manifest.get("arms"),dict) or not manifest["arms"]: raise CampaignError("campaign arms missing")
    if manifest.get("program_contract") in CEILING1_PROGRAM_CONTRACTS:
        if set(manifest["arms"])!=set(CEILING1_ARM_CONTRACT): raise CampaignError("Ceiling program contract requires exactly qwen25-3b and qwen3-1p7b arms")
        expected_dag=CEILING1_PHASE_DAGS[manifest["program_contract"]]
        for arm,expected in CEILING1_ARM_CONTRACT.items():
            cfg=manifest["arms"][arm]
            for key,value in expected.items():
                if cfg.get(key)!=value: raise CampaignError(f"{arm}: Ceiling program contract {key} drift")
            observed=[(x.get("id"),list(x.get("depends_on") or [])) for x in (cfg.get("phases") or [])]
            if observed!=expected_dag: raise CampaignError(f"{arm}: Ceiling program DAG drift")
            if manifest["program_contract"]==CEILING1_PROGRAM_CONTRACT_V2:
                for phase in cfg.get("phases") or []:
                    if not phase.get("release"):
                        gate=phase.get("gate") or {}
                        if gate.get("state") not in STATES or not gate.get("reason"):
                            raise CampaignError(f"{arm}:{phase['id']}: v2 empty release requires explicit gate state+reason")
    for arm,cfg in manifest["arms"].items():
        phases=cfg.get("phases") or []; ids=[x.get("id") for x in phases]
        if None in ids or len(ids)!=len(set(ids)): raise CampaignError(f"{arm}: phase ids invalid")
        seen=set()
        for phase in phases:
            deps=phase.get("depends_on") or []
            if any(d not in seen for d in deps): raise CampaignError(f"{arm}:{phase['id']}: dependency must point backward")
            seen.add(phase["id"])

def evaluate(repo:Path,manifest:dict[str,Any]) -> dict[str,Any]:
    validate_manifest(repo,manifest); arms={}
    for arm,cfg in manifest["arms"].items():
        prior:dict[str,PhaseResult]={}; rows=[]
        for phase in cfg["phases"]:
            result=classify_phase(repo,manifest,arm,phase,prior); prior[phase["id"]]=result; rows.append(result.as_dict())
        arms[arm]=rows
    complete=all(rows and rows[-1]["state"]=="COMPLETE" for rows in arms.values())
    return {"schema":"openevo.autonomous-campaign-status.v1","campaign_id":manifest["campaign_id"],"campaign_complete":complete,"arms":arms,"evaluated_at_epoch":time.time(),"mutation":False}

def first_actionable(status:dict[str,Any]) -> tuple[str,dict[str,Any]]|None:
    priority={"ENGINEERING_RETRYABLE":0,"RESUME_REQUIRED":1,"READY":2}
    choices=[]
    for arm,rows in status["arms"].items():
        for row in rows:
            if row["state"] in priority and row.get("next_action"):
                choices.append((priority[row["state"]],arm,row)); break
            if row["state"] not in {"COMPLETE","RUNNING"}: break
    if not choices: return None
    _,arm,row=sorted(choices,key=lambda x:(x[0],x[1]))[0]; return arm,row

def execute_action(manifest:dict[str,Any],arm:str,row:dict[str,Any],dry_run:bool) -> int:
    action=row.get("next_action") or {}
    if action.get("kind")!="command": raise CampaignError(f"unsupported phase action: {action}")
    if manifest.get("activation_authority") is not True and not dry_run:
        raise CampaignError("master campaign has no execution activation authority")
    argv=[str(x) for x in action.get("argv") or []]
    if not argv: raise CampaignError("empty phase action argv")
    env=os.environ.copy(); env.update({str(k):str(v) for k,v in (action.get("env") or {}).items()})
    cwd=action.get("cwd")
    print(json.dumps({"arm":arm,"phase":row["phase"],"state":row["state"],"argv":argv,"cwd":cwd,"dry_run":dry_run},sort_keys=True))
    if dry_run: return 0
    return subprocess.run(argv,cwd=cwd,env=env,check=False).returncode

def write_status(path:Path|None,value:dict[str,Any]) -> None:
    if not path: return
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name("."+path.name+".tmp")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False)+"\n",encoding="utf-8"); tmp.replace(path)
def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="cmd",required=True)
    for name in ("validate","status","plan","black-box"):
        q=sub.add_parser(name); q.add_argument("--manifest",type=Path,default=Path("configs/experiment/campaigns/current-ceiling1-autonomous.json")); q.add_argument("--repo-root",type=Path); q.add_argument("--output",type=Path)
    run=sub.add_parser("run"); run.add_argument("--manifest",type=Path,default=Path("configs/experiment/campaigns/current-ceiling1-autonomous.json")); run.add_argument("--repo-root",type=Path); run.add_argument("--output",type=Path); run.add_argument("--until-blocked",action="store_true"); run.add_argument("--dry-run",action="store_true")
    a=p.parse_args(argv); repo=(a.repo_root or Path(__file__).resolve().parents[2]).resolve(); manifest,resolution=resolve_manifest(repo,a.manifest)
    if a.cmd=="validate":
        validate_manifest(repo,manifest)
        for arm,cfg in manifest["arms"].items():
            ctx=phase_context(manifest,arm)
            for phase in cfg["phases"]:
                release,path,error=load_release(repo,phase.get("release"),ctx)
                if error and phase.get("release"): raise CampaignError(error)
                if release:
                    ok,detail=validate_release_structure(release)
                    if not ok: raise CampaignError(f"{arm}:{phase['id']}: {detail}")
                    ok,detail=release_invariant(manifest,arm,phase['id'],release)
                    if not ok: raise CampaignError(f"{arm}:{phase['id']}: {detail}")
                    ok,detail=validate_authority(repo,release)
                    if not ok: raise CampaignError(f"{arm}:{phase['id']}: {detail}")
                    if release.get('execution_source') is not None:
                        ok,detail=git_source_status(release['execution_source'])
                        if not ok: raise CampaignError(f"{arm}:{phase['id']}: {detail}")
        result={"schema":"openevo.autonomous-campaign-validation.v1","status":"PASS","campaign_id":manifest["campaign_id"],"manifest_resolution":resolution,"mutation":False}
        write_status(a.output,result); print(json.dumps(result,sort_keys=True)); return 0
    status=evaluate(repo,manifest); status["manifest_resolution"]=resolution; status["transition_journal_health"]=transition_journal_health(manifest,status); write_status(a.output,status)
    if a.cmd in {"status","black-box"}:
        print(json.dumps(status,indent=2,sort_keys=True,ensure_ascii=False)); return 0
    actionable=first_actionable(status)
    if a.cmd=="plan":
        if status["transition_journal_health"].get("status")=="BLOCKED": actionable=None
        result={"campaign_id":manifest["campaign_id"],"manifest_resolution":resolution,"transition_journal_health":status["transition_journal_health"],"actionable":actionable[1] if actionable else None,"arm":actionable[0] if actionable else None,"mutation":False}
        print(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)); return 0
    # run: one transition at a time, re-evaluating all immutable gates after each action.
    if not a.dry_run:
        assert_campaign_execution_authority(manifest,resolution)
    steps=0
    lock_context=contextlib.nullcontext(None) if a.dry_run else campaign_lock(manifest)
    with lock_context:
        while True:
            status=evaluate(repo,manifest); status["manifest_resolution"]=resolution; health=transition_journal_health(manifest,status); status["transition_journal_health"]=health; write_status(a.output,status)
            if health.get("status")=="BLOCKED": raise CampaignError(f"transition journal blocks execution: {health.get('reason')}")
            if not a.dry_run and health.get("status")=="RECOVERABLE":
                reconcile_recoverable_journal(manifest,health)
                health=transition_journal_health(manifest,status); status["transition_journal_health"]=health; write_status(a.output,status)
            if status["campaign_complete"]: print("CAMPAIGN_COMPLETE"); return 0
            actionable=first_actionable(status)
            if not actionable:
                print(json.dumps(status,indent=2,sort_keys=True,ensure_ascii=False)); print("CAMPAIGN_BLOCKED_OR_RUNNING"); return 3
            arm,row=actionable
            action_id=transition_action_id(manifest,arm,row)
            if not a.dry_run:
                append_transition_journal(manifest,{"event":"ACTION_STARTED","action_id":action_id,"arm":arm,"phase":row["phase"],"state_before":row["state"],"release_path":row.get("release_path"),"started_at_epoch":time.time(),"budget_delta_before":{"new_webshop_calls":0,"external_teacher_calls":0,"final_panel_access":0}})
            try:
                rc=execute_action(manifest,arm,row,a.dry_run)
            except Exception as exc:
                if not a.dry_run:
                    append_transition_journal(manifest,{"event":"ACTION_EXCEPTION","action_id":action_id,"arm":arm,"phase":row["phase"],"exception_type":type(exc).__name__,"exception":str(exc),"finished_at_epoch":time.time()})
                raise
            if not a.dry_run:
                append_transition_journal(manifest,{"event":"ACTION_FINISHED","action_id":action_id,"arm":arm,"phase":row["phase"],"exit_code":rc,"finished_at_epoch":time.time()})
            if rc!=0: print(f"CAMPAIGN_ACTION_EXIT={rc}",file=sys.stderr); return rc
            steps+=1
            if a.dry_run or not a.until_blocked: return 0
            if steps>=1000: raise CampaignError("campaign transition safety limit reached")

if __name__=="__main__":
    try: raise SystemExit(main())
    except CampaignError as exc:
        print(f"autonomous-campaign: {exc}",file=sys.stderr); raise SystemExit(2)
