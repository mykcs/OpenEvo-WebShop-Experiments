#!/usr/bin/env python3
"""E3-only zero-formal GPU/storage qualification; never invokes the formal command."""
from __future__ import annotations
import argparse, datetime, hashlib, json, os, shutil, stat, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts import oe_impl
from scripts.openevo_webshop import prelaunch_zero_formal as zf
from scripts.openevo_webshop import post_advisor_e3_guard_202609300248 as guard
from scripts.openevo_webshop.post_advisor_e3_runtime_safety_202609300248 import GPU_UUID
HIST = Path("/workspace/openevo-workspace/qualifications/post-advisor-unanswered-20260929/storage-capacity-e1-e3-20260929.json")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text())
def git(repo,*args): return subprocess.run(["git","-C",str(repo),*args],check=True,capture_output=True,text=True).stdout.strip()
def once(path,doc):
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    tmp=path.with_name(path.name+f".{os.getpid()}.tmp")
    data=(json.dumps(doc,ensure_ascii=False,indent=2,sort_keys=True)+"\n").encode()
    fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o640)
    try:
        with os.fdopen(fd,"wb") as f: f.write(data); f.flush(); os.fsync(f.fileno())
        os.link(tmp,path)
    finally:
        try: tmp.unlink()
        except FileNotFoundError: pass
def attempt(root,stem):
    n=1
    while (root/f"{stem}-attempt-{n:04d}.json").exists(): n+=1
    return root/f"{stem}-attempt-{n:04d}.json"
def owner(indices):
    out=subprocess.run(["gpu-who"],check=True,capture_output=True,text=True).stdout
    expected={f"GPU{i} | idle | -" for i in indices}
    if not expected.issubset({x.strip() for x in out.splitlines()}): raise RuntimeError("E3 GPU owner gate not all-idle")
    return {"observed_at_epoch":time.time(),"selected_indices":indices,"selected_classifications":{str(i):"idle" for i in indices},"gpu_who_stdout":out,"gpu_who_stdout_sha256":hashlib.sha256(out.encode()).hexdigest()}
def storage(rank,campaign,experiment,arm,a_sha,b_sha):
    h=load(HIST); m=h["capacity_method"]
    exact={"historical_max_round_bytes_over_R152_R159":299897856,"conservative_estimate_bytes_per_arm":2399182848,"estimated_total_bytes":9596731392,"explicit_free_space_reserve_bytes":4294967296,"rollouts_per_planned_arm":1024,"rounds_per_planned_arm":8}
    arms=["E1_DIRECTION_PRESERVING_SCALAR_GATE_RANK128",*(f"E3_RANK_CAPACITY_SWEEP_RANK{r}" for r in (8,16,64))]
    if any(m.get(k)!=v for k,v in exact.items()) or m.get("planned_arms")!=arms: raise RuntimeError("frozen storage method drift")
    src=Path(m["source"]); source_bytes=sum(x.stat().st_size for x in src.rglob("*") if x.is_file())
    path=Path("/workspace"); u=shutil.disk_usage(path)
    fs=subprocess.run(["findmnt","-T",str(path),"-no","SOURCE,FSTYPE,TARGET"],check=True,capture_output=True,text=True).stdout.strip().split(maxsplit=2)
    if len(fs)!=3 or not fs[0].startswith("GalaxyFS:") or fs[1]!="fuse.galaxyfs" or fs[2]!="/workspace": raise RuntimeError("workspace is not on expected GalaxyFS mount")
    required=int(m["estimated_total_bytes"])+int(m["explicit_free_space_reserve_bytes"])
    if u.free<required: raise RuntimeError(f"GalaxyFS capacity below frozen plan+reserve: {u.free} < {required}")
    return {"schema":"openevo.post-advisor.storage-capacity-gate.v1","status":"PASS","campaign_id":campaign,"formal_task_calls_consumed":0,"final_panel_access":0,"execution_sha":a_sha,"authority_git_sha":b_sha,"rank":rank,"e3_scope":{"experiment_id":experiment,"rank":rank,"rounds":list(range(152,160)),"formal_rollout_limit":1024,"arm_output_root":arm["output_root"]},"capacity_method":{**m,"historical_receipt_path":str(HIST),"historical_receipt_sha256":sha(HIST),"observed_historical_source_bytes":source_bytes},"filesystem":{"persistent_output_policy":"GalaxyFS","mount":fs[0],"fstype":fs[1],"mount_target":fs[2],"path":str(path),"total_bytes":u.total,"available_bytes":u.free,"required_bytes":required,"unrelated_data_deleted":False},"formal_task_consumed":False,"scientific_budget_consumed":0,"observed_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
def require_uid1001_directory(path, expected):
    path=Path(path)
    if path.is_symlink():
        raise RuntimeError(f"E3 output directory must not be a symlink: {path}")
    try:
        resolved=path.resolve(strict=True); info=path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"E3 output directory must already exist: {path}") from exc
    if resolved!=Path(os.path.abspath(path)) or resolved!=Path(expected).resolve(strict=True):
        raise RuntimeError(f"E3 output directory does not resolve to its exact bound path: {path}")
    if not stat.S_ISDIR(info.st_mode) or info.st_uid!=1001 or stat.S_IMODE(info.st_mode)!=0o750:
        raise RuntimeError(f"E3 output directory must be UID1001 mode 750: {path}")
    return path

def validated_runs_mount(profile):
    configured=Path(str(profile["workspace"]))/str(profile["paths"]["runs"])
    if not configured.is_absolute():
        raise RuntimeError("E3 configured runs path must be absolute")
    try:
        runs_root=configured.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError("E3 configured runs path must already exist") from exc
    if runs_root!=configured or not runs_root.is_dir():
        raise RuntimeError("E3 configured runs path must exist without symlinks")
    mount=subprocess.run(["findmnt","-T",str(runs_root),"-no","SOURCE,FSTYPE,TARGET"],check=True,capture_output=True,text=True).stdout.strip().split(maxsplit=2)
    if len(mount)!=3 or not mount[0].startswith("GalaxyFS:") or mount[1]!="fuse.galaxyfs" or mount[2]!="/workspace":
        raise RuntimeError("E3 configured runs path is not on the exact GalaxyFS mount")
    traverse=subprocess.run(["runuser","-u","experiment-host","--","test","-x",str(runs_root)],check=False,capture_output=True)
    if traverse.returncode!=0:
        raise RuntimeError("UID1001 cannot traverse the configured E3 runs path")
    return runs_root

def run_e3_prelaunch(profile, selected, output_root, attempt_dir, image, container_name, marker_path):
    attempt_number=5
    receipt_path=attempt_dir/f"prelaunch-zero-formal-attempt-{attempt_number:04d}.json"
    if receipt_path.exists(): raise RuntimeError("E3 prelaunch attempt receipt already exists; refusing overwrite")
    payload=zf.payload_command(oe_impl._prelaunch_runtime_python(profile),marker_path,len(selected))
    command=oe_impl.docker_command(profile,selected,image,"disabled",command=payload)
    try: name_index=command.index("--name")+1
    except ValueError as exc: raise RuntimeError("E3 zero-formal Docker command lacks container name") from exc
    command[name_index]=container_name
    started=time.monotonic(); completed=oe_impl._run(command,check=False); wall=max(0.0,time.monotonic()-started)
    parsed=None; parse_error=None
    try: parsed=zf.parse_payload_output(completed.stdout or "")
    except zf.PrelaunchError as exc: parse_error=str(exc)
    receipt=zf.build_prelaunch_receipt(run_id=str(profile["run_id"]),attempt=attempt_number,selected_gpu_uuids=selected,
        image_id=image,command_identity=oe_impl._command_identity(command,image),return_code=int(completed.returncode),
        wall_seconds=wall,payload=parsed,payload_error=parse_error,stderr_tail=completed.stderr or "")
    oe_impl.atomic_json(receipt_path,receipt)
    zf.validate_prelaunch_receipt(receipt,expected_gpu_uuids=selected)
    return {"path":str(receipt_path),"sha256":zf.sha256_file(receipt_path),"receipt_sha256":receipt["receipt_sha256"],"receipt":receipt}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--repo",type=Path,required=True); p.add_argument("--authority-repo",type=Path,required=True); p.add_argument("--authority",type=Path,required=True); p.add_argument("--pending-activation",type=Path,required=True); p.add_argument("--qualification",type=Path,required=True); p.add_argument("--rank",type=int,choices=(8,16,64),required=True); a=p.parse_args()
    if os.geteuid()!=0: raise RuntimeError("E3 prelaunch qualification requires root")
    repo,b,c=a.repo.resolve(),a.authority_repo.resolve(),a.pending_activation.resolve(); ap=a.authority.resolve(); qp=a.qualification.resolve(); rank=a.rank
    ah,bh=git(repo,"rev-parse","HEAD"),git(b,"rev-parse","HEAD")
    if git(repo,"status","--porcelain","--untracked-files=all") or git(b,"status","--porcelain","--untracked-files=all"): raise RuntimeError("A and B must be clean")
    if git(b,"merge-base",ah,bh)!=ah or ah==bh: raise RuntimeError("A/B lineage drift")
    previous=ah
    revisions=git(b,"rev-list","--reverse",f"{ah}..{bh}").splitlines()
    if not revisions: raise RuntimeError("B must be a distinct authority descendant")
    for rev in revisions:
        if git(b,"rev-list","--parents","-n","1",rev).split()!=[rev,previous]: raise RuntimeError("A to B must be linear")
        changed=set(git(b,"diff","--name-only",previous,rev).splitlines())
        if not changed or any(not guard.e3_b_material(name) for name in changed): raise RuntimeError("B contains changes outside E3 authority material")
        previous=rev
    agg=b/guard.E3_AUTHORITY; prep=guard.verify_preparation(repo,agg); aggregate_authority=load(agg); authority=load(ap)
    if set(aggregate_authority.get("source_files") or {})!=set(guard.SOURCE_FILES): raise RuntimeError("E3 A source map incomplete")
    for key,rel in guard.SOURCE_FILES.items():
        if sha(repo/rel)!=aggregate_authority["source_files"][key]: raise RuntimeError(f"E3 A source SHA drift: {key}")
    for path in (agg,ap):
        rel=path.relative_to(b)
        if (b/rel).read_bytes()!=subprocess.run(["git","-C",str(b),"show",f"HEAD:{rel}"],check=True,capture_output=True).stdout: raise RuntimeError("E3 B authority bytes are uncommitted")
    if prep["status"]!="PASS_SOURCE_PREPARATION" or prep["trainer_owned_replay_pending_ranks"] or ap!=b/guard.E3_ARM.format(rank) or authority.get("status")!="AUTHORIZED" or authority.get("execution_git_sha")!=ah or authority.get("managed_gpu_indices")!=[0,1,2,3] or authority.get("rounds")!=list(range(152,160)) or authority.get("formal_rollout_limit")!=1024 or authority.get("treatment_mode")!="BOUNDED_OFF": raise RuntimeError("E3 frozen source/arm authority invalid")
    c_repo=c.parents[3]; ch=git(c_repo,"rev-parse","HEAD")
    c_parent=git(c_repo,"rev-list","--parents","-n","1",ch).split()
    if c_parent!=[ch,bh] or git(c_repo,"status","--porcelain","--untracked-files=all"):
        raise RuntimeError("C2 must be clean and a direct child of exact B")
    if git(c_repo,"diff","--name-only",bh,ch).splitlines()!=[guard.E3_ACTIVATION]:
        raise RuntimeError("C2 may add only the E3 inactive activation scope")
    c_rel=str(c.relative_to(c_repo))
    if c.read_bytes()!=subprocess.run(["git","-C",str(c_repo),"show",f"HEAD:{c_rel}"],check=True,capture_output=True).stdout:
        raise RuntimeError("C2 activation bytes must be committed")
    scope_sha=ch; scope_activation_sha=sha(c)
    draft=load(c); aggregate=load(b/guard.E3_AUTHORITY)
    if (draft.get("schema")!="openevo.post-advisor.e3-rank-capacity-activation.202609300248"
            or draft.get("status")!="PREPARED_NOT_ACTIVE"
            or draft.get("formal_task_consumption_allowed") is not False
            or draft.get("final_panel_access_allowed") is not False
            or draft.get("execution_sha")!=ah or draft.get("authority_git_sha")!=bh
            or draft.get("source_e1_c_sha")!=guard.BASE_C
            or draft.get("campaign_id")!=aggregate.get("campaign_id")
            or draft.get("experiment_id")!=aggregate.get("experiment_id")):
        raise RuntimeError("pending E3 activation schema/identity/status drift")
    if set(draft.get("rank_releases") or {})!={"8","16","64"}:
        raise RuntimeError("pending C must bind exactly the three E3 rank releases")
    expected_release_refs={}
    for r in (8,16,64):
        rel=guard.E3_RELEASE if r==8 else guard.E3_RELEASE_OTHERS.format(r)
        release_path=b/rel
        expected_release_refs[str(r)]={"path":rel,"sha256":sha(release_path)}
        ref=draft["rank_releases"][str(r)]
        if ref!={"path":rel,"sha256":sha(release_path)}:
            raise RuntimeError(f"pending C rank{r} release ref differs from exact B release")
    if draft.get("release")!=expected_release_refs["8"]:
        raise RuntimeError("pending C primary release ref differs from B rank8 release")
    evidence=draft.get("rank_evidence") or {}; attempts=draft.get("prelaunch_attempts") or {}
    if set(evidence)!={"8","16","64"} or set(attempts)!={"8","16","64"}:
        raise RuntimeError("pending C6 must bind rank-specific qualification refs and attempt namespaces")
    for r in (8,16,64):
        arm_doc=load(b/guard.E3_ARM.format(r))
        arm_release=load(b/(guard.E3_RELEASE if r==8 else guard.E3_RELEASE_OTHERS.format(r)))
        profile_ref=arm_release["profiles"][str(r)]; profile_path,arm_profile=guard.checked_ref(b,profile_ref)
        runs_root=validated_runs_mount(arm_profile)
        output_root=Path(arm_doc["output_root"])
        if output_root.is_symlink():
            raise RuntimeError(f"B rank{r} arm output_root must not be a symlink")
        output_root=output_root.resolve(strict=True)
        try:
            output_root.relative_to(runs_root)
        except ValueError as exc:
            raise RuntimeError(f"B rank{r} arm output_root is outside the validated profile runs mount") from exc
        if output_root==runs_root:
            raise RuntimeError(f"B rank{r} output_root must be below the configured runs mount")
        require_uid1001_directory(output_root,output_root)
        expected_parent=output_root/"preflight"/"attempt-0005"
        ref=evidence[str(r)].get("qualification") or {}; qpath=Path(str(ref.get("path") or ""))
        marker_rel=expected_parent.resolve().relative_to(runs_root).as_posix()
        expected_attempt={"attempt":5,"namespace":str(expected_parent),
            "container_name":f"experiment-host-openevo-e3-preflight-rank{r}-attempt-0005",
            "probe_marker_path":f"/workspace/runs/{marker_rel}/.prelaunch-write-probe-0005"}
        if attempts[str(r)]!=expected_attempt:
            raise RuntimeError(f"pending C6 rank{r} prelaunch attempt binding drift")
        expected_q=expected_parent/f"launcher-qualification-{ah[:12]}-{bh[:12]}-attempt-0005.json"
        if qpath!=expected_q:
            raise RuntimeError(f"pending C6 rank{r} qualification path is not the exact attempt-0005 namespace")
        if r==rank and qpath.resolve()!=qp:
            raise RuntimeError("CLI qualification output must match pending C6 rank-specific reference")
        if ref.get("sha256") is not None and (not qpath.is_file() or ref["sha256"]!=sha(qpath)):
            raise RuntimeError(f"pending C6 rank{r} qualification receipt SHA drift")
    router=b/guard.E3_ROUTER.format(rank); rd=load(router)
    if rd.get("status")!="PREPARED_NOT_ACTIVE" or rd.get("formal_task_consumption_allowed") is not False or rd.get("final_panel_access_allowed") is not False or rd.get("rank")!=rank or rd.get("formal_rollout_limit")!=1024: raise RuntimeError("rank router is not inactive")
    release=load(b/(guard.E3_RELEASE if rank==8 else guard.E3_RELEASE_OTHERS.format(rank))); ref=release["profiles"][str(rank)]; pp=b/ref["path"]; profile=load(pp)
    if sha(pp)!=ref["sha256"] or profile.get("source",{}).get("experiment",{}).get("revision")!=ah or profile.get("run_id")!=f"post-advisor-e3-rank{rank}-202609300248" or profile.get("execution",{}).get("mode")!="host-orchestrator" or profile.get("image_id")!="sha256:d0f8796010899ae41a7af8ae3f44f47b96dea73b297b5af124049b3ae88e2850": raise RuntimeError("profile binding drift")
    oe_impl.validate_sources_and_paths(profile); image=oe_impl.resolve_image_identity(profile)
    if image!=profile["image_id"]: raise RuntimeError("image ID drift")
    runs_root=validated_runs_mount(profile)
    out=Path(authority["output_root"])
    if out.is_symlink(): raise RuntimeError("B arm output_root must not be a symlink")
    out=out.resolve(strict=True)
    try:
        out.relative_to(runs_root)
    except ValueError as exc:
        raise RuntimeError("B arm output_root is outside the validated profile runs mount") from exc
    if out==runs_root: raise RuntimeError("B arm output_root must be below the configured runs mount")
    require_uid1001_directory(out,out)
    preflight_root=out/"preflight"; pre=Path(attempts[str(rank)]["namespace"])
    if pre!=preflight_root/"attempt-0005": raise RuntimeError("C6 attempt namespace is not the exact output-root child")
    require_uid1001_directory(preflight_root,out/"preflight")
    require_uid1001_directory(pre,preflight_root/"attempt-0005")
    if (out/"execution-state.json").exists() or list(out.glob("round-*")): raise RuntimeError("arm output/attempt namespace is not clean UID1001 zero-formal root")
    if list(pre.glob("gpu-gate-*.json")) or list(pre.glob("prelaunch-zero-formal-attempt-*.json")):
        raise RuntimeError("E3 attempt-0005 namespace already contains a GPU/prelaunch receipt; refusing overwrite")
    selected=[GPU_UUID[i] for i in (0,1,2,3)]; attempt_binding=attempts[str(rank)]; scope={"execution_sha":ah,"authority_git_sha":bh,"arm_sha256":sha(ap),"rank":rank,"router_sha256":sha(router),"profile_sha256":sha(pp),"run_id":profile["run_id"],"gpu_uuids":selected,"c2_sha":scope_sha,"c2_activation_sha256":scope_activation_sha,"attempt":5,"attempt_namespace":str(pre),"container_name":attempt_binding["container_name"],"probe_marker_path":attempt_binding["probe_marker_path"]}
    sp=attempt(pre,"e3-storage-gate"); sd=storage(rank,load(agg)["campaign_id"],load(agg)["experiment_id"],authority,ah,bh); sd["c2_sha"]=scope_sha; sd["c2_activation_sha256"]=scope_activation_sha; sd["attempt"]=5; once(sp,sd)
    before=owner([0,1,2,3]); gp_profile={**profile,"wait":False,"gpu_gate":{**profile.get("gpu_gate",{}),"interval_seconds":1,"timeout_seconds":0}}; lock=oe_impl.workspace_path(profile,str(profile.get("lock_root",".gpu-locks")))
    with oe_impl.gpu_locks(selected,lock):
        if list(pre.glob("gpu-gate-*.json")):
            raise RuntimeError("E3 attempt-0005 namespace already has a GPU gate receipt under lock; refusing atomic overwrite")
        oe_impl.wait_for_idle(gp_profile,selected,pre); after=owner([0,1,2,3]); result=run_e3_prelaunch(profile,selected,out,pre,image,attempt_binding["container_name"],attempt_binding["probe_marker_path"])
    gpu_path=max(pre.glob("gpu-gate-*.json"),key=lambda x:x.stat().st_mtime_ns); gd=load(gpu_path)
    if gd.get("schema")!="openevo.webshop.gpu-gate.v2" or gd.get("eligible") is not True or gd.get("selected_gpu_uuids")!=selected: raise RuntimeError("standard E3 GPU gate failed")
    zp=Path(result["path"]); zd=load(zp); zf.validate_prelaunch_receipt(zd,expected_gpu_uuids=selected)
    op=attempt(pre,"e3-gpu-owner-gate"); once(op,{"schema":"openevo.post-advisor.e3-gpu-owner-gate.v1","status":"PASS","e3_binding":scope,"before":before,"after":after,"formal_task_consumed":False,"scientific_budget_consumed":0,"model_calls":0,"training_calls":0,"final_panel_access":0})
    zero=zf.build_qualification_receipt(campaign_id=load(agg)["campaign_id"],execution_sha=ah,current_campaign_sha256=sha(router),design_sha256=load(agg)["design_sha256"],activation_sha256=sha(router),profile_sha256=sha(pp),host_router_path=str(router),host_control_head=bh,registry_pointer_path=str(router),registry_pointer_sha256=sha(router),selected_gpu_uuids=selected,prelaunch_receipt=zd,prelaunch_receipt_file_sha256=zf.sha256_file(zp)); zero["e3_binding"]=scope; zero["receipt_sha256"]=zf.canonical_sha256({k:v for k,v in zero.items() if k!="receipt_sha256"})
    zz=attempt(pre,"e3-zero-formal"); once(zz,zero)
    q={"schema":"openevo.post-advisor.e3-rank-launcher-qualification.202609300248","status":"PASS","execution_sha":ah,"authority_git_sha":bh,"authority_sha256":sha(ap),"aggregate_authority_sha256":sha(agg),"persistent_state_rank":rank,"treatment_mode":"BOUNDED_OFF","formal_rollout_limit":1024,"formal_task_consumed":False,"task_bearing_calls":0,"model_calls":0,"training_calls":0,"final_panel_access":0,"e3_binding":scope}
    if qp.parent.resolve()!=pre.resolve() or not qp.name.startswith(f"launcher-qualification-{ah[:12]}-{bh[:12]}-attempt-") or qp.suffix!=".json": raise RuntimeError("qualification receipt path must be append-only under this arm preflight directory")
    once(qp,q)
    binding_path=attempt(pre,"e3-scope-binding")
    receipt_refs={"qualification":{"path":str(qp),"sha256":sha(qp)},"zero_formal":{"path":str(zz),"sha256":sha(zz)},
                  "prelaunch":{"path":str(zp),"sha256":sha(zp)},"gpu_gate":{"path":str(gpu_path),"sha256":sha(gpu_path)},
                  "gpu_owner":{"path":str(op),"sha256":sha(op)},"storage":{"path":str(sp),"sha256":sha(sp)}}
    once(binding_path,{"schema":"openevo.post-advisor.e3-prelaunch-scope-binding.v1","status":"PASS","e3_binding":scope,
                       "receipts":receipt_refs,"formal_task_consumed":False,"scientific_budget_consumed":0,
                       "model_calls":0,"training_calls":0,"final_panel_access":0})
    if (out/"execution-state.json").exists() or list(out.glob("round-*")): raise RuntimeError("prelaunch created formal state")
    print(json.dumps({"status":"PASS_E3_ZERO_FORMAL_PRELAUNCH","rank":rank,"attempt":5,"container_name":attempt_binding["container_name"],"qualification":str(qp),"zero_formal":str(zz),"prelaunch":str(zp),"gpu_gate":str(gpu_path),"gpu_owner":str(op),"storage":str(sp),"scope_binding":str(binding_path),"formal_task_consumed":False,"scientific_budget_consumed":0,"round_or_execution_state_created":False},sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
