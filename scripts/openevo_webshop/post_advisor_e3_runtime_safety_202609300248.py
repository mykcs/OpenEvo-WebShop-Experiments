from __future__ import annotations
"""Fail-closed engineering safety for one formal arm.

Owns GPU/Docker identity, liveness, seed-data, container exactly-once recovery,
and the arm lock. It must not change treatment, task selection, reward logic, or
scientific acceptance semantics.
"""
import fcntl, getpass, json, os, subprocess, time
from contextlib import contextmanager
from pathlib import Path

from scripts.openevo_webshop import ceiling1_stage2_arm_controller as base
from scripts.openevo_webshop import bounded_effective_state_ray_formal_adapter_202609282338 as ray_capacity
from scripts.openevo_webshop.bounded_effective_state_launch_guard_202609282338 import sha as guard_sha

GPU_UUID = {
    0: "GPU-00000000-0000-0000-0000-000000000000",
    1: "GPU-00000000-0000-0000-0000-000000000000",
    2: "GPU-00000000-0000-0000-0000-000000000000",
    3: "GPU-00000000-0000-0000-0000-000000000000",
}


def require_post_advisor_gpu_assignment(managed: list[int]) -> set[str]:
    """Bind an arm's four devices to the canonical launcher allocation."""
    if len(managed) != 4 or len(set(managed)) != 4 or any(index not in GPU_UUID for index in managed):
        raise RuntimeError("post-advisor managed GPUs must be four mapped, unique devices")
    raw = os.environ.get("OPENEVO_ASSIGNED_GPU_UUIDS", "")
    assigned = raw.split(",") if raw else []
    if len(assigned) != 4 or len(set(assigned)) != 4 or any(not uuid.startswith("GPU-") for uuid in assigned):
        raise RuntimeError("formal post-advisor launch requires four assigned GPU UUIDs")
    expected = {GPU_UUID[index] for index in managed}
    if set(assigned) != expected:
        raise RuntimeError("post-advisor managed GPUs differ from oe assigned GPU UUIDs")
    return expected
SCIENCE_IMAGE = "experiment-host/openevo-scientific:h144-runtime-v1"  # legacy-revision-compat: absorbed historical runtime literal from a6f67b
SCIENCE_IMAGE_ID = "sha256:d0f8796010899ae41a7af8ae3f44f47b96dea73b297b5af124049b3ae88e2850"
CARRIER_NOOP_SCHEMA = "openevo.ceiling1-stage2-carrier-contract-fail-noop-amendment.v1"  # legacy-revision-compat: absorbed historical runtime literal from a6f67b
CARRIER_NOOP_REASON = "NOOP_KEEP_PRIOR_TYPED_CONTRACT_REJECTED_AFTER_REPAIR"

def load_json(path: Path):
    return json.loads(path.read_text())

@contextmanager
def exclusive_arm_lock(output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".formal-arm.lock"
    stream = lock_path.open("a+")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"formal arm already has an active controller: {output_root}") from exc
        stream.seek(0); stream.truncate()
        stream.write(json.dumps({"pid": os.getpid(), "output_root": str(output_root)}, sort_keys=True) + "\n")
        stream.flush(); os.fsync(stream.fileno())
        yield
    finally:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()

def validate_carrier_noop_amendment(path: Path, package: dict, repo: Path) -> str:
    ref = package.get("carrier_contract_fail_noop_amendment") or {}
    if ref.get("path") != str(path.relative_to(repo)) or ref.get("sha256") != guard_sha(path):
        raise RuntimeError("carrier contract-fail NOOP amendment READY binding drift")
    doc = load_json(path)
    if doc.get("schema") != CARRIER_NOOP_SCHEMA or doc.get("status") != "AUTHORIZED":
        raise RuntimeError("carrier contract-fail NOOP amendment is not authorized")
    if "qwen3-1p7b" not in set(doc.get("applies_to_arms") or []):
        raise RuntimeError("carrier contract-fail NOOP amendment arm scope drift")
    if set(doc.get("applies_to_carriers") or []) != {"text_memory", "skill_bundle", "agent_system"}:
        raise RuntimeError("carrier contract-fail NOOP amendment carrier scope drift")
    if (doc.get("transition") or {}).get("reason") != CARRIER_NOOP_REASON:
        raise RuntimeError("carrier contract-fail NOOP transition drift")
    unchanged = doc.get("unchanged") or {}
    if int(unchanged.get("max_primary_calls", -1)) != 1 or int(unchanged.get("max_repair_calls", -1)) != 1:
        raise RuntimeError("carrier contract-fail NOOP call budget drift")
    if int(unchanged.get("external_teacher_calls", -1)) != 0 or int(unchanged.get("final_panel_access", -1)) != 0:
        raise RuntimeError("carrier contract-fail NOOP protected boundary drift")
    return guard_sha(path)

def launcher_gpu_env(kind: str, gpus: list[int]) -> dict[str, str]:
    if kind == "rollout":
        if not gpus:
            raise RuntimeError("formal rollout GPU lane is empty")
        return {"CEILING1_STAGE2_GPU_UUIDS": ",".join(GPU_UUID[i] for i in gpus)}
    if kind in {"reflector", "training"}:
        if len(gpus) != 1:
            raise RuntimeError(f"formal {kind} requires exactly one GPU")
        return {"CEILING1_STAGE2_GPU_UUID": GPU_UUID[gpus[0]]}
    if kind == "transition":
        if len(gpus) != 1:
            raise RuntimeError("formal transition requires exactly one GPU")
        return {"BOUNDED_GPU_UUID": GPU_UUID[gpus[0]]}
    raise RuntimeError(f"unsupported formal workload kind for GPU env: {kind}")

def exact_image_gate() -> None:
    q = subprocess.run(["docker", "image", "inspect", SCIENCE_IMAGE, "--format", "{{.Id}}"], text=True, capture_output=True)
    if q.returncode != 0 or q.stdout.strip() != SCIENCE_IMAGE_ID:
        raise RuntimeError("formal science image identity drift")


def fresh_gpu_gate(indices: list[int]) -> dict:
    if getpass.getuser() != "root" or os.geteuid() != 0:
        raise RuntimeError("formal GPU gate requires whoami=root")
    who = subprocess.run(["gpu-who"], text=True, capture_output=True)
    if who.returncode != 0:
        raise RuntimeError("gpu-who unavailable at point of use")
    for index in indices:
        needle = f"GPU{index} | idle | -"
        if needle not in who.stdout:
            raise RuntimeError(f"GPU{index} is not idle by gpu-who")
    q = subprocess.run([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True, capture_output=True, check=True)
    observed = {}
    for line in q.stdout.splitlines():
        idx, uuid, used, util = [part.strip() for part in line.split(",")]
        observed[int(idx)] = {"uuid": uuid, "memory_used_mib": int(used), "utilization": int(util)}
    for index in indices:
        row = observed.get(index)
        if row is None or row["uuid"] != GPU_UUID[index]:
            raise RuntimeError(f"GPU{index} UUID drift")
        if row["memory_used_mib"] > 64 or row["utilization"] != 0:
            raise RuntimeError(f"GPU{index} not idle by nvidia-smi: {row}")
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"], text=True, capture_output=True, check=True)
    target = {GPU_UUID[index] for index in indices}
    if any(line.split(",", 1)[0].strip() in target for line in apps.stdout.splitlines() if line.strip()):
        raise RuntimeError("target GPU has a compute process")
    return {"operator": "root", "gpus": {str(i): observed[i] for i in indices}}


class RuntimeSafetyMixin:
    def validate_seed_root_data(self, r: int) -> dict:
        schedule=load_json(self.wt/'configs/experiment/manifests/h1.44-on-3b-training-schedule-v1.json')  # legacy-revision-compat: absorbed historical runtime literal from a6f67b
        expected=schedule.get('webshop_data_hashes') or {}
        if set(expected)!={'items_shuffle_1000.json','items_ins_v2_1000.json','items_human_ins.json'}: raise RuntimeError('schedule WebShop data hash contract drift')  # legacy-revision-compat: absorbed historical runtime literal from a6f67b
        root=self.a.seed_root.resolve(); observed={}
        for name,digest in expected.items():
            fp=root/name
            if not fp.is_file(): raise RuntimeError(f'WebShop seed-root data missing: {fp}')
            got=base.sha(fp)
            if got!=digest: raise RuntimeError(f'WebShop seed-root data SHA drift: {name}')
            observed[name]=got
        receipt={'schema':'openevo.bounded-gdr.webshop-data-gate.202609152115','status':'PASS','round_index':r,'seed_root':str(root),'webshop_data_hashes':observed,'schedule_sha256':base.sha(self.wt/'configs/experiment/manifests/h1.44-on-3b-training-schedule-v1.json'),'reward_used_for_admission':False,'final_panel_access':0}  # legacy-revision-compat: absorbed historical runtime literal from a6f67b
        gate=self.out/f'round-{r:03d}'/'webshop-data-gate.json'; base.write_once_json(gate,receipt); return receipt

    def run_rollout(self, r: int, rr: Path):
        self.validate_seed_root_data(r)
        return super().run_rollout(r, rr)

    def managed_gpus(self) -> list[int]:
        if getattr(self, "ray_capacity_policy", None) is not None:
            values=list(map(int,self.ray_capacity_policy.get("managed_gpus") or []))
        else:
            values=list(map(int,(self.launch_package.get("execution_policy") or {}).get("managed_gpu_indices") or [4,5,6,7]))
        if len(values)!=4 or len(set(values))!=4:
            raise RuntimeError("formal managed GPU authority must contain four unique GPUs")
        return values

    def fixed_gpus(self, kind: str) -> list[int]:
        if kind == "rollout": return self.managed_gpus()
        if kind in {"reflector", "training"}: return [self.post_gpu_priority()[0]]
        raise RuntimeError(f"unsupported formal workload kind: {kind}")

    def post_gpu_priority(self) -> list[int]:
        arm=self.experiment_arm_id
        key="off" if arm=="BOUNDED_OFF" else "on" if arm=="BOUNDED_EFFECTIVE_STATE_ON" else None
        if key is None: raise RuntimeError("unknown arm for post-rollout GPU priority")
        configured=((self.launch_package.get("execution_policy") or {}).get("post_rollout_gpu_priority") or {}).get(key)
        values=list(map(int,configured or ([7,6,5,4] if key=="off" else [4,5,6,7])))
        if len(values)!=4 or set(values)!=set(self.managed_gpus()): raise RuntimeError("post-rollout GPU priority does not match managed GPU authority")
        return values

    def _select_ray_capacity(self, kind: str, r: int, count: int) -> tuple[list[int], dict]:
        root=self._attempt_root(r,kind)
        selection=root/f"ray-selection-{self.execution_git_sha[:12]}.json"
        gate=ray_capacity.select(self.wt,self.ray_capacity_policy,experiment_arm_id=self.experiment_arm_id,kind=kind,round_index=r,output=selection,active_reflector_kind=getattr(self,'_active_reflector_kind',None),count=count)
        return list(map(int,gate['selected_gpus'])),gate

    def select_post_gpu(self, kind: str, r: int) -> tuple[list[int], dict | None]:
        # An existing attempt keeps its original device; never migrate a live call.
        root = self._attempt_root(r, kind)
        for intent in sorted(root.glob("a*-intent.json")):
            doc = load_json(intent); n = int(doc["attempt_index"])
            if not (root / f"a{n:03d}.json").exists():
                uuids = doc.get("gpu_uuids") or []
                ids = [i for i in self.post_gpu_priority() if GPU_UUID[i] in uuids]
                if len(ids) != 1: raise RuntimeError("post-rollout attempt device drift")
                return ids, None
        if getattr(self,'ray_capacity_policy',None) is not None:
            return self._select_ray_capacity(kind,r,1)
        failures = []
        for i in self.post_gpu_priority():
            try: return [i], fresh_gpu_gate([i])
            except RuntimeError as exc: failures.append(f"GPU{i}: {exc}")
        raise RuntimeError("no permitted idle post-rollout GPU: " + "; ".join(failures))

    def _shared_rollout_gate(self, round_idx: int, gpus: list[int]) -> dict | None:
        path = getattr(self, "shared_rollout_gate_path", None)
        if path is None:
            return None
        doc = load_json(path)
        if doc.get("schema") != "openevo.bounded-gdr.coresident-rollout-gate.20260916" or doc.get("status") != "PASS":
            raise RuntimeError("shared co-resident rollout gate schema/status drift")
        if int(doc.get("round_index", -1)) != round_idx or doc.get("execution_git_sha") != self.execution_git_sha:
            raise RuntimeError("shared co-resident rollout gate round/execution drift")
        if doc.get("ready_package_sha256") != guard_sha(self.ready_package_path) or doc.get("owner_launch_release_sha256") != guard_sha(self.owner_launch_release_path):
            raise RuntimeError("shared co-resident rollout gate authority drift")
        if doc.get("gpu_indices") != gpus or int(doc.get("workers_per_gpu", -1)) != 2:
            raise RuntimeError("shared co-resident rollout gate topology drift")
        if set(doc.get("experiment_arm_ids") or []) != {"BOUNDED_OFF", "BOUNDED_EFFECTIVE_STATE_ON"}:
            raise RuntimeError("shared co-resident rollout gate arm set drift")
        observed_at = float(doc.get("observed_at_unix", 0.0))
        if observed_at <= 0 or time.time() - observed_at > 60.0:
            raise RuntimeError("shared co-resident rollout gate is stale")
        live = doc.get("live_gpu_gate") or {}
        if doc.get("resource_mode") == "RAY_CAPACITY_SHARED_POOL_V1":  # legacy-revision-compat: absorbed historical runtime literal from a6f67b
            policy=getattr(self,'ray_capacity_policy',None)
            if policy is None or live.get('resource_mode')!='RAY_CAPACITY_V1' or live.get('idle_required') is not False:  # legacy-revision-compat: absorbed historical runtime literal from a6f67b
                raise RuntimeError('shared Ray capacity rollout gate policy drift')
            if live.get('resource_policy_sha256')!=policy['_sha256'] or live.get('managed_gpus')!=gpus:
                raise RuntimeError('shared Ray capacity rollout gate identity drift')
            return {**live,"shared_coresidency_gate_path":str(path),"shared_coresidency_gate_sha256":base.sha(path)}
        rows = live.get("gpus") or {}
        for index in gpus:
            row = rows.get(str(index)) or {}
            if row.get("uuid") != GPU_UUID[index] or int(row.get("memory_used_mib", 999999)) > 64 or int(row.get("utilization", -1)) != 0:
                raise RuntimeError(f"shared co-resident rollout gate GPU{index} drift")
        return {**live, "shared_coresidency_gate_path": str(path), "shared_coresidency_gate_sha256": base.sha(path)}

    def _workload_key(self, kind: str) -> str:
        channel = getattr(self, "_active_reflector_kind", None)
        return f"reflector-{channel}" if kind == "reflector" and channel else kind

    def _container_name(self, kind: str, round_idx: int, attempt: int) -> str:
        kind = self._workload_key(kind)
        slug = self.experiment_arm_id.lower().replace("_", "-")
        return f"bounded-{slug}-r{round_idx:03d}-{kind}-a{attempt}"

    def _attempt_root(self, round_idx: int, kind: str) -> Path:
        root = self.out / f"round-{round_idx:03d}" / "container-attempts" / self._workload_key(kind)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _intent_payload(self, kind: str, round_idx: int, attempt: int, container: str, launcher: Path, gpus: list[int]) -> dict:
        return {"schema":"openevo.bounded-gdr.formal-container-launch-intent.202609151945","status":"FROZEN_BEFORE_LAUNCH",
                "experiment_arm_id":self.experiment_arm_id,"treatment_mode":self.treatment_mode,"round_index":round_idx,
                "kind":kind,"attempt_index":attempt,"container_name":container,"execution_git_sha":self.execution_git_sha,
                "ready_package_sha256":guard_sha(self.ready_package_path),"owner_launch_release_sha256":guard_sha(self.owner_launch_release_path),
                "launcher_path":str(launcher.resolve()),"launcher_sha256":base.sha(launcher.resolve()),
                "gpu_uuids":[GPU_UUID[i] for i in gpus],"final_panel_access":0}
    def _validate_intent(self, path: Path, expected: dict) -> dict:
        if not path.is_file(): raise RuntimeError(f"formal container exists without launch intent: {path}")
        got=load_json(path)
        if got != expected: raise RuntimeError(f"formal container launch-intent drift: {path}")
        return got

    def _select_container_attempt(self, kind: str, round_idx: int, launcher: Path, gpus: list[int]) -> tuple[int,str,str]:
        root=self._attempt_root(round_idx,kind)
        for attempt in range(1000):
            container=self._container_name(kind,round_idx,attempt); status=base.dstatus(container)
            terminal=root/f"a{attempt:03d}.json"; intent=root/f"a{attempt:03d}-intent.json"; launch=root/f"a{attempt:03d}-launcher.json"; gate=root/f"a{attempt:03d}-gpu-gate.json"
            expected=self._intent_payload(kind,round_idx,attempt,container,launcher,gpus)
            if terminal.is_file():
                prior=load_json(terminal)
                if status!='absent': raise RuntimeError(f"terminalized formal container still exists: {container}")
                if int(prior.get('container_exit_code',-999))==0: raise RuntimeError(f"successful {kind} attempt already archived; refuse duplicate side effect")
                continue
            if status!='absent':
                self._validate_intent(intent,expected)
                if not gate.is_file(): raise RuntimeError(f"existing formal container lacks GPU gate: {container}")
                return attempt,container,'ADOPT_EXISTING_CONTAINER'
            if intent.is_file():
                self._validate_intent(intent,expected)
                if not launch.is_file(): raise RuntimeError(f"launch intent exists without launcher outcome/container: {container}")
                lr=load_json(launch)
                if int(lr.get('launcher_returncode',-999))!=0 and lr.get('container_status_after_launcher')=='absent':
                    continue
                raise RuntimeError(f"ambiguous prior formal launch attempt requires manual review: {container}")
            if gate.is_file() or launch.is_file(): raise RuntimeError(f"formal attempt metadata is incomplete: {container}")
            return attempt,container,'NEW_CONTAINER'
        raise RuntimeError('container engineering attempt namespace exhausted')

    def _archive_container(self, rr: Path, kind: str, attempt: int, container: str, adopted: bool, rc: int) -> dict:
        root=self._attempt_root(int(rr.name.split('-')[-1]),kind)
        logs=subprocess.run(['docker','logs',container],text=True,capture_output=True); inspect=subprocess.run(['docker','inspect',container],text=True,capture_output=True)
        state={}
        if inspect.returncode==0:
            try: state=(json.loads(inspect.stdout)[0].get('State') or {})
            except Exception: state={'inspect_parse_error':True}
        log_path=root/f"a{attempt:03d}.log"
        if not log_path.exists(): log_path.write_text((logs.stdout or '')+'\nSTDERR\n'+(logs.stderr or ''))
        payload={"schema":"openevo.bounded-gdr.formal-container-attempt.202609151945","status":"PASS" if rc==0 else "ENGINEERING_FAILURE",
                 "experiment_arm_id":self.experiment_arm_id,"treatment_mode":self.treatment_mode,"round_index":int(rr.name.split('-')[-1]),"kind":kind,
                 "attempt_index":attempt,"container_name":container,"adopted_existing_container":adopted,"container_exit_code":rc,
                 "oom_killed":bool(state.get('OOMKilled',False)),"docker_state":state.get('Status'),"execution_git_sha":self.execution_git_sha,
                 "log_path":str(log_path),"log_sha256":base.sha(log_path)}
        base.write_once_json(root/f"a{attempt:03d}.json",payload)
        rm=subprocess.run(['docker','rm',container],text=True,capture_output=True)
        if rm.returncode!=0 and base.dstatus(container)!='absent': raise RuntimeError(f"failed to remove completed formal container {container}")
        return payload

    def _validate_container_inspect_doc(self, doc: dict, kind: str, round_idx: int, gpus: list[int]) -> None:
        if doc.get("Image") != SCIENCE_IMAGE_ID: raise RuntimeError("formal container image ID drift")
        cfg=doc.get("Config") or {}; host=doc.get("HostConfig") or {}
        if cfg.get("User") != "1001:1001": raise RuntimeError("formal container UID/GID drift")
        if host.get("NetworkMode") != "none": raise RuntimeError("formal container network isolation drift")
        observed=set()
        for req in host.get("DeviceRequests") or []: observed.update(req.get("DeviceIDs") or [])
        expected={GPU_UUID[i] for i in gpus}
        if getattr(self.a, "post_advisor_short_fork_authority", None) is not None:
            assigned=require_post_advisor_gpu_assignment(self.managed_gpus())
            if not expected.issubset(assigned):
                raise RuntimeError("post-advisor container GPU request exceeds oe assignment")
        if observed != expected: raise RuntimeError(f"formal container GPU binding drift: expected={sorted(expected)} observed={sorted(observed)}")
        if getattr(self.a, "post_advisor_short_fork_authority", None) is not None:
            tmpfs = (host.get("Tmpfs") or {}).get("/tmp", "")
            if "size=4g" not in tmpfs or "mode=1777" not in tmpfs:
                raise RuntimeError("post-advisor model container lacks qualified 4 GiB /tmp tmpfs")
        mounts={(str(m.get("Source")),str(m.get("Destination")),bool(m.get("RW"))) for m in doc.get("Mounts") or []}
        for required in (str(self.wt),str(self.root)):
            hits=[m for m in mounts if m[0]==required and m[1]==required]
            if len(hits)!=1 or hits[0][2]: raise RuntimeError(f"formal container required read-only mount drift: {required}")
        if not any(m[2] and Path(m[0]).is_relative_to(self.out) for m in mounts): raise RuntimeError("formal container lacks writable arm-output mount")
        cmd=[str(x) for x in (cfg.get("Cmd") or [])]
        def has_pair(flag: str, value: str) -> bool:
            return any(cmd[i]==flag and i+1<len(cmd) and cmd[i+1]==value for i in range(len(cmd)-1))
        if not has_pair("--round-index",str(round_idx)): raise RuntimeError("formal container round identity drift")
        if kind=="rollout":
            if not any(x.endswith("ceiling1_stage2_round_driver.py") for x in cmd): raise RuntimeError("formal rollout entrypoint drift")
            if not has_pair("--state-id",str(self.state["state_id"])) or not has_pair("--experiment-git-sha",self.execution_git_sha): raise RuntimeError("formal rollout state/execution identity drift")
        elif kind=="reflector":
            if not any(x.endswith("ceiling1_stage2_reflector_qwen3.py") for x in cmd): raise RuntimeError("formal reflector entrypoint drift")
            if not has_pair("--state-id",str(self.state["state_id"])): raise RuntimeError("formal reflector state identity drift")
        elif kind=="transition":
            expected_driver = self.wt / "scripts/openevo_webshop/post_advisor_e3_rank_capacity_transition_driver_202609300248.py"
            if cmd.count(str(expected_driver)) != 1: raise RuntimeError("E3 formal transition entrypoint drift")
            authority = getattr(self, "post_advisor_short_fork_authority", None) or {}
            expected_sha = (authority.get("source_files") or {}).get("transition_driver_sha256")
            if not expected_sha or guard_sha(expected_driver) != expected_sha:
                raise RuntimeError("E3 formal transition driver SHA drift")
            if not has_pair("--treatment-mode",self.treatment_mode): raise RuntimeError("formal transition treatment drift")
        else: raise RuntimeError(f"unsupported formal container kind for identity gate: {kind}")

    def _inspect_container_identity(self, container: str, kind: str, round_idx: int, gpus: list[int]) -> dict:
        q=subprocess.run(["docker","inspect",container],text=True,capture_output=True,check=True); docs=json.loads(q.stdout)
        if len(docs)!=1: raise RuntimeError("formal container inspect cardinality drift")
        self._validate_container_inspect_doc(docs[0],kind,round_idx,gpus)
        return docs[0]

    def _prime_reflector_resource_gate(self, r: int, rr: Path, kind: str) -> None:
        self._active_reflector_kind = kind
        ev=load_json(rr/"evidence"/"evidence.json"); gate=(ev.get("gates") or {}).get(kind) or {}
        if not gate.get("eligible"): return
        directory=rr/kind.replace("_","-"); name={"text_memory":"memory.md","skill_bundle":"SKILL.md","agent_system":"agent_system.md"}[kind]
        out=directory/name; receipt=out.with_suffix(out.suffix+".receipt.json"); term=directory/"terminal-receipt.json"; attempt=directory/"model-call-attempt.json"
        if term.exists() or out.exists() or receipt.exists() or attempt.exists(): return
        workload={"text_memory":f"stage2_text_memory_{self.arm}","skill_bundle":f"stage2_skill_bundle_{self.arm}","agent_system":f"stage2_agent_system_{self.arm}"}[kind]
        self._prevalidated_resource_gates[(r,workload)] = self.select_post_gpu("reflector", r)

    def _progress_signature(self, root: Path) -> tuple[int,int,int]:
        count=total=latest=0
        if not root.exists(): return (0,0,0)
        for base_dir,_,names in os.walk(root):
            for name in names:
                fp=Path(base_dir)/name
                try: st=fp.stat()
                except FileNotFoundError: continue
                count+=1; total+=int(st.st_size); latest=max(latest,int(st.st_mtime_ns))
        return (count,total,latest)

    def _gpu_activity(self, gpus: list[int]) -> tuple[bool,bool]:
        q=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,capture_output=True)
        if q.returncode!=0: return (False,False)
        target={GPU_UUID[i] for i in gpus}; seen=set(); active=False
        for line in q.stdout.splitlines():
            parts=[x.strip() for x in line.split(',')]
            if len(parts)!=4 or parts[1] not in target: continue
            seen.add(parts[1]); active=active or int(parts[3])>0  # VRAM residency alone is not forward progress.
        return (seen==target,active)

    def _wait_with_liveness(self, container: str, rr: Path, kind: str, attempt: int, gpus: list[int]) -> int:
        poll=30.0; stall=1800.0
        last_sig=self._progress_signature(rr); last_progress=time.monotonic()
        while True:
            status=base.dstatus(container)
            if status=='exited': return int(subprocess.run(['docker','inspect',container,'--format','{{.State.ExitCode}}'],text=True,capture_output=True,check=True).stdout.strip())
            if status not in {'running','created'}: raise RuntimeError(f'formal container disappeared during liveness wait: {container}={status}')
            sig=self._progress_signature(rr); observed,active=self._gpu_activity(gpus)
            if sig!=last_sig or active: last_progress=time.monotonic(); last_sig=sig
            elif observed and time.monotonic()-last_progress>=stall:
                receipt={"schema":"openevo.bounded-gdr.formal-engineering-stall.202609152145","status":"ENGINEERING_STALL_SUSPECTED","round_index":int(rr.name.split('-')[-1]),"kind":kind,"attempt_index":attempt,"container_name":container,"gpu_indices":gpus,"progress_signature":list(sig),"stall_seconds":stall,"automatic_kill":False,"automatic_retry":False,"final_panel_access":0}
                base.write_once_json(self._attempt_root(int(rr.name.split('-')[-1]),kind)/f'a{attempt:03d}-stall.json',receipt)
                raise RuntimeError(f'engineering stall suspected; container left intact for audited recovery: {container}')
            time.sleep(max(0.01,poll))

    def _run_managed_container(self, kind: str, round_idx: int, launcher: Path, env: dict, gpus: list[int], container_env_key: str='CEILING1_STAGE2_CONTAINER', prevalidated_gate: dict|None=None) -> tuple[str,list[int]]:
        if getattr(self.a, "post_advisor_short_fork_authority", None) is not None:
            assigned=require_post_advisor_gpu_assignment(self.managed_gpus())
            if not {GPU_UUID[i] for i in gpus}.issubset(assigned):
                raise RuntimeError("post-advisor child GPU request exceeds oe assignment")
        if getattr(self,'ray_capacity_policy',None) is not None: ray_capacity.reconcile(self.wt,self.ray_capacity_policy)
        rr=self.out/f"round-{round_idx:03d}"; attempt,container,mode=self._select_container_attempt(kind,round_idx,launcher,gpus); root=self._attempt_root(round_idx,kind)
        expected=self._intent_payload(kind,round_idx,attempt,container,launcher,gpus)
        if mode=='NEW_CONTAINER':
            gate=prevalidated_gate if prevalidated_gate is not None else fresh_gpu_gate(gpus); base.write_once_json(root/f"a{attempt:03d}-gpu-gate.json",gate); base.write_once_json(root/f"a{attempt:03d}-intent.json",expected)
            env2={**os.environ,**env,container_env_key:container,'CEILING1_STAGE2_EXECUTION_GIT_SHA':self.execution_git_sha}
            env2.update(launcher_gpu_env(kind,gpus))
            lease_id=None
            if getattr(self,'ray_capacity_policy',None) is not None and gate.get('schema')=='openevo.bounded-gdr.ray-capacity-selection-gate.20260916':
                lease_id=f"bounded-ray-{self.execution_git_sha[:12]}-{self.experiment_arm_id.lower()}-r{round_idx:03d}-{self._workload_key(kind)}-a{attempt:03d}"
                q=ray_capacity.launch(self.wt,self.ray_capacity_policy,gate,lease_id=lease_id,container=container,launcher=Path(launcher),env=env2)
            else:
                q=subprocess.run([str(launcher)],env=env2,text=True,capture_output=True)
            status=base.dstatus(container)
            launch_receipt={"schema":"openevo.bounded-gdr.formal-launcher-invocation.202609151945","status":"RETURNED","round_index":round_idx,"kind":kind,
                            "attempt_index":attempt,"container_name":container,"launcher_returncode":q.returncode,"container_status_after_launcher":status,
                            "stdout":q.stdout,"stderr":q.stderr,"execution_git_sha":self.execution_git_sha,"ray_capacity_lease_id":lease_id}
            base.write_once_json(root/f"a{attempt:03d}-launcher.json",launch_receipt)
            if status=='absent':
                if q.returncode!=0: raise RuntimeError(q.stderr.strip() or q.stdout.strip() or f"{kind} launcher failed before container creation")
                raise RuntimeError(f"{kind} launcher returned success but no durable container exists")
        status=base.dstatus(container)
        if status=='created': self._inspect_container_identity(container,kind,round_idx,gpus); subprocess.run(['docker','start',container],text=True,capture_output=True,check=True); status='running'
        if status not in {'running','exited'}: raise RuntimeError(f"unsupported existing container state for recovery: {container}={status}")
        self._inspect_container_identity(container,kind,round_idx,gpus)
        rc=self._wait_with_liveness(container,rr,kind,attempt,gpus) if status!='exited' else int(subprocess.run(['docker','inspect',container,'--format','{{.State.ExitCode}}'],text=True,capture_output=True,check=True).stdout.strip())
        self._archive_container(rr,kind,attempt,container,mode!='NEW_CONTAINER',rc)
        if getattr(self,'ray_capacity_policy',None) is not None: ray_capacity.reconcile(self.wt,self.ray_capacity_policy)
        if rc!=0: raise RuntimeError(f"{kind} failed without automatic scientific retry: exit={rc}, receipt={root/f'a{attempt:03d}.json'}")
        return container,gpus

    def launch_retry(self, kind, round_idx, workload, launcher, env, vram=None, max_attempts=1):
        _ = max_attempts  # Base API compatibility only. Formal execution never loops inside one invocation.
        if kind == "reflector":
            self._active_reflector_kind = env["CEILING1_STAGE2_KIND"]
            launcher = self.wt / "scripts/openevo_webshop/launch_bounded_carrier_container_20260916.sh"
        prevalidated=None
        if getattr(self,'ray_capacity_policy',None) is not None:
            if kind=='rollout':
                self._shared_rollout_gate(round_idx,self.managed_gpus())
                gpus,prevalidated=self._select_ray_capacity(kind,round_idx,4)
            elif kind=='reflector':
                selected=self._prevalidated_resource_gates.pop((round_idx,workload),None)
                gpus,prevalidated=selected if selected is not None else self._select_ray_capacity(kind,round_idx,1)
            else:
                gpus,prevalidated=self._select_ray_capacity(kind,round_idx,1)
        else:
            gpus=self.fixed_gpus(kind)
            if kind=="rollout": prevalidated=self._shared_rollout_gate(round_idx,gpus)
            elif kind=="reflector":
                selected=self._prevalidated_resource_gates.pop((round_idx,workload),None)
                gpus,prevalidated=selected if selected is not None else self.select_post_gpu(kind,round_idx)
            else: gpus,prevalidated=self.select_post_gpu(kind,round_idx)
        return self._run_managed_container(kind,round_idx,Path(launcher),env,gpus,prevalidated_gate=prevalidated)
