#!/usr/bin/env python3
"""Q17 R159/R129 only-GPU6/7, fail-closed experimental measurement runner.

Preflight is read-only (CPU). `qualify` loads the two models separately but
consumes zero WebShop episodes. `dev64` runs the already-frozen 64-task
DEVELOPMENT panel on two exact physical GPUs, once per model, and writes only
new measurement receipts. The R159/R129 comparison is diagnostic, not final.

The only GPUs permitted by this source are physical GPU index 6 and 7 with the
pinned live-verified UUIDs; this script never acquires GPU0-5 or runs --gpus all.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from typing import Any

DEFAULT_PLAN = "configs/experiment/metric-campaigns/q17-gpu67-20261012.json"
EXPECTED_SCHEMA = "openevo.q17.gpu67.metric-campaign-plan.v1"
ALLOWED = {
    6: "GPU-00000000-0000-0000-0000-000000000000",
    7: "GPU-00000000-0000-0000-0000-000000000000",
}
SHA = re.compile(r"^[a-f0-9]{64}$")
RUN_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{2,64}$")


class GateError(RuntimeError):
    pass


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_path(p: Path) -> str:
    digest = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(p: Path, expected: str, label: str) -> None:
    if not SHA.fullmatch(expected):
        raise GateError(f"{label}: pinned SHA256 malformed")
    if not p.is_file():
        raise GateError(f"{label}: absent {p}")
    found = sha_path(p)
    if found != expected:
        raise GateError(f"{label}: content SHA256 drift: {found} != {expected}")


def read_json(p: Path) -> dict[str, Any]:
    obj = json.loads(p.read_bytes())
    if not isinstance(obj, dict):
        raise GateError(f"expected JSON object at {p}")
    return obj


def subprocess_checked(args: list[str], timeout_s: int = 30) -> str:
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s, check=False)
    if proc.returncode != 0:
        raise GateError(f"command failed rc={proc.returncode}: {args[0]} {args[1] if len(args)>1 else ''}: {proc.stderr[:240]}")
    return proc.stdout.strip()


def verify_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema") != EXPECTED_SCHEMA:
        raise GateError("unknown metric campaign plan schema")
    devices = plan.get("device_allowlist", [])
    if [(x.get("index"), x.get("uuid"), x.get("arm")) for x in devices] != [
        (6, ALLOWED[6], "r159"), (7, ALLOWED[7], "r129")
    ]:
        raise GateError("GPU scope drift: ONLY physical GPU6 and GPU7 allowed")
    budget = plan["budgets"]
    if budget.get("allowed_gpu_indices") != [6, 7]:
        raise GateError("budget GPU allowlist drift")
    for name, expected in (("frozen_development_tasks_per_arm", 64),
                           ("maximum_new_development_rollouts", 128),
                           ("maximum_development_episode_steps", 15),
                           ("qualifications_per_gpu", 1),
                           ("formal_task_consumption", 0),
                           ("final_panel_access", 0),
                           ("training_updates", 0)):
        if budget.get(name) != expected:
            raise GateError(f"budget drift: {name}")
    if plan["science"]["only_frozen_development_panel"] is not True or plan["science"]["no_new_formal_or_final_tasks"] is not True:
        raise GateError("final/formal protection contract removed")
    if plan["science"]["instrumentation"] != "on" or plan["science"]["q06_context_capture"] is not True:
        raise GateError("frozen instrumentation contract changed")
    if plan["science"]["sampling"] != {
        "temperature": 1.0, "top_p": 1.0, "top_k": 0,
        "do_sample": True, "max_new_tokens": 512, "max_steps": 15,
        "seed_policy": "paired_generation_seed + step_index * 100000",
    }:
        raise GateError("sampling contract drift")
    if not plan["image_id"].startswith("sha256:") or len(plan["image_id"]) != 71:
        raise GateError("Docker image requires full immutable digest")
    s = plan["science"]
    for arm, round_index, updates in (("r159", 159, 160), ("r129", 129, 130)):
        a = plan["arms"][arm]
        if a["round"] != round_index or a["completed_stage2_rounds"] != updates:
            raise GateError(f"{arm} producer-round identity drift")
        if not SHA.fullmatch(a["adapter_sha256"]) or not SHA.fullmatch(a["state_id"]):
            raise GateError(f"{arm} invalid checkpoint identity")
    if s["frozen_panel_sha"] != "4033523c01697576b7f9ce8cddc3a63cf6c1514ef5554e01a8c82b59ee62d478":
        raise GateError("not the original frozen 64-task development panel")


def repo_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise GateError(f"source escapes repo root: {relative}")
    return path


def project_paths(plan: dict[str, Any]) -> dict[str, Path]:
    p = plan["paths"]
    workspace = Path(p["workspace"])
    if str(workspace) != "/workspace":
        raise GateError("unknown workspace authority")
    paths = {k: Path(v) for k, v in p.items() if k not in {
        "frozen_panel_manifest_rel", "probe_script_rel", "qualifier_script_rel"
    }}
    for key in ("q17_root", "runtime_source_root", "core_source_root", "seed_root", "model_root", "output_root", "gpu_lock_root"):
        if workspace not in paths[key].parents:
            raise GateError(f"{key} must be under owner workspace: {paths[key]}")
    if paths["output_root"] in paths["q17_root"].parents or paths["q17_root"] in paths["output_root"].parents:
        raise GateError("output root may not overlap original training state")
    return paths


def carrier_paths(plan: dict[str, Any], p: dict[str, Path]) -> dict[str, Path]:
    out = {}
    for k in ("memory", "skill", "agent"):
        entry = plan["carriers"][k]
        out[k] = p["q17_root"] / ("round-%03d" % entry["round"]) / entry["rel"]
    return out


def adapter_paths(plan: dict[str, Any], p: dict[str, Path]) -> dict[str, Path]:
    out = {}
    for arm in ("r159", "r129"):
        c = plan["arms"][arm]
        out[arm] = p["q17_root"] / ("round-%03d" % c["round"]) / c["adapter_rel"]
    return out


def cpu_source_check(plan: dict[str, Any], repo_root: Path, *, full_hash: bool) -> dict[str, Any]:
    """Immutable scientific identity and local files only: no GPU calls."""
    verify_plan(plan)
    p = project_paths(plan)
    science = plan["science"]
    ref = repo_path(repo_root, plan["paths"]["frozen_panel_manifest_rel"])
    require_sha(ref, science["frozen_manifest_file_sha256"], "frozen development panel")
    frozen = read_json(ref)
    if frozen.get("panel_sha256") != science["frozen_panel_sha"] or len(frozen.get("tasks", [])) != 64:
        raise GateError("frozen development manifest size/hash mismatch")
    if frozen["final_panel_access_allowed"] is not False or frozen["formal_task_consumption_allowed"] is not False:
        raise GateError("frozen development authorizes forbidden final/formal access")
    source_sampling = {
        "max_new_tokens": 512, "max_steps": 15, "repetition_penalty": 1.0,
        "step_seed_policy": "paired_generation_seed + step_index * 100000",
        "temperature": 1.0, "top_k": 0, "top_p": 1.0,
    }
    if frozen["sampling"] != source_sampling:
        raise GateError("frozen original sampling contract mismatch")
    if frozen["runtime_execution_sha"] != science["runtime_execution_sha"]:
        raise GateError("runtime execution identity drift")
    if frozen["openevo_core_sha"] != science["core_execution_sha"]:
        raise GateError("core execution identity drift")
    if frozen["model"]["local_model_path"] != str(p["model_root"]):
        raise GateError("base model path differs from frozen science")
    if frozen["model"]["revision"] != science["base_revision"]:
        raise GateError("model revision differs from frozen science")
    require_sha(repo_path(repo_root, plan["paths"]["probe_script_rel"]), science["science_probe_sha256"], "instrumented WebShop probe")
    require_sha(repo_path(repo_root, plan["paths"]["qualifier_script_rel"]), science["load_qualifier_sha256"], "original zero-rollout model-load qualifier")
    for key, expected in (("runtime_source_root", science["runtime_execution_sha"]),
                          ("core_source_root", science["core_execution_sha"])):
        actual = (p[key] / "EXECUTION_SHA").read_text().strip()
        if actual != expected:
            raise GateError(f"{key} EXECUTION_SHA drift: {actual}")
    for name, sha in frozen["runtime_files"].items():
        require_sha(p["runtime_source_root"] / name, sha, f"scientific runtime {name}")
    if not p["seed_root"].is_dir():
        raise GateError("frozen WebShop seed root missing")
    if not (p["model_root"] / "config.json").is_file():
        raise GateError("Qwen3-1.7B model config missing")
    for kind, path in carrier_paths(plan, p).items():
        require_sha(path, plan["carriers"][kind]["sha256"], f"{kind} carrier")
        if frozen["common_carriers"][{"memory": "memory_sha256", "skill": "skill_sha256", "agent": "agent_system_sha256"}[kind]] != plan["carriers"][kind]["sha256"]:
            raise GateError(f"{kind} carrier differs from frozen original manifest")
    ap = adapter_paths(plan, p)
    model_identities = {}
    for arm, path in ap.items():
        entry = plan["arms"][arm]
        round_root = p["q17_root"] / ("round-%03d" % entry["round"])
        rc = read_json(round_root / "ROUND_COMPLETE.json")
        upd = read_json(round_root / "sd-lora/update.json")
        if rc["round_index"] != upd["round_index"] or rc["round_index"] != entry["round"]:
            raise GateError(f"{arm} round mismatch")
        if rc["state_id_after"] != entry["state_id"] or upd["adapter_digest_after"] != entry["adapter_sha256"]:
            raise GateError(f"{arm} adapter/state identity mismatch")
        if str(path) != upd["adapter_path"]:
            raise GateError(f"{arm} adapter directory differs from sealed producer")
        if arm == "r159":
            pinned = frozen["checkpoints"]["after_160_updates"]
            if pinned["state_id"] != entry["state_id"] or pinned["adapter_sha256"] != entry["adapter_sha256"]:
                raise GateError("R159 differs from pinned canonical frozen panel manifest")
        model_file = path / "adapter_model.safetensors"
        if not model_file.is_file() or model_file.stat().st_size < 100_000:
            raise GateError(f"{arm} adapter weights absent/truncated")
        if full_hash:
            require_sha(model_file, entry["adapter_sha256"], f"{arm} adapter")
        model_identities[arm] = {"round": entry["round"], "adapter_sha256": entry["adapter_sha256"],
                                 "state_id": entry["state_id"], "source_bytes": model_file.stat().st_size,
                                 "full_original_sha_verified_this_pass": bool(full_hash)}
    free = os.statvfs(p["output_root"].parent)
    free_bytes = free.f_bavail * free.f_frsize
    if free_bytes < plan["budgets"]["minimum_free_working_disk_bytes"]:
        raise GateError(f"insufficient working filesystem reserve: {free_bytes} bytes")
    return {"schema": "openevo.q17.gpu67.cpu-science-preflight.v1", "status": "PASS",
            "panel_sha256": science["frozen_panel_sha"], "frozen_manifest_sha256": science["frozen_manifest_file_sha256"],
            "runtime_execution_sha": science["runtime_execution_sha"], "model_producers": model_identities,
            "image_id_requested": plan["image_id"], "owner_working_fs_free_bytes": free_bytes,
            "source_qualification": "FULL_BYTE_SHA" if full_hash else "METADATA_ONLY_PENDING_FULL_ADAPTER_SHA",
            "gpu_calls": 0, "development_rollouts": 0, "final_panel_access": 0,
            "write_operations": 0}


def live_gpu_gate(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Read-only live gate. Global nvidia-smi inventory is NOT GPU compute."""
    text = subprocess_checked(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
                               "--format=csv,noheader,nounits"], 15)
    table = {}
    for line in text.splitlines():
        cols = [s.strip() for s in line.split(",")]
        if len(cols) != 6:
            raise GateError(f"unparsable GPU inventory: {line[:180]}")
        idx, uuid, name, used, capacity, util = cols
        table[int(idx)] = {"index": int(idx), "uuid": uuid, "name": name,
                           "memory_used_mib": int(used), "memory_total_mib": int(capacity),
                           "gpu_util_pct": int(util)}
    active = subprocess_checked(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                 "--format=csv,noheader,nounits"], 15)
    active_uuid = set()
    for line in active.splitlines():
        if not line.strip() or line.startswith("No running processes"):
            continue
        parts = [s.strip() for s in line.split(",")]
        if len(parts) < 2 or not parts[0].startswith("GPU-"):
            raise GateError(f"unparsable GPU compute occupancy: {line[:150]}")
        active_uuid.add(parts[0])
    out = []
    for allowed in plan["device_allowlist"]:
        index = allowed["index"]
        if index not in table:
            raise GateError(f"GPU{index} is not present")
        gpu = table[index]
        if gpu["uuid"] != ALLOWED[index]:
            raise GateError(f"GPU{index} UUID changed: {gpu['uuid']}")
        if "RTX 5090" not in gpu["name"]:
            raise GateError(f"GPU{index} hardware changed")
        if gpu["uuid"] in active_uuid:
            raise GateError(f"GPU{index} has a running compute-app; no takeover")
        if gpu["memory_used_mib"] > plan["budgets"]["maximum_idle_gpu_used_mib"]:
            raise GateError(f"GPU{index} unexpectedly occupied ({gpu['memory_used_mib']} MiB); owner unresolved")
        if gpu["memory_total_mib"] - gpu["memory_used_mib"] < plan["budgets"]["minimum_free_gpu_mib"]:
            raise GateError(f"GPU{index} insufficient free VRAM")
        out.append(gpu)
    return out


def docker_image_gate(plan: dict[str, Any]) -> str:
    actual = subprocess_checked(["docker", "image", "inspect", plan["image_id"],
                                "--format", "{{.Id}}"], 20)
    if actual != plan["image_id"]:
        raise GateError("container image digest drift")
    return actual


def derive_arm_manifest(plan: dict[str, Any], repo_root: Path, arm: str) -> dict[str, Any]:
    source = read_json(repo_path(repo_root, plan["paths"]["frozen_panel_manifest_rel"]))
    entry = plan["arms"][arm]
    if source["checkpoints"]["after_160_updates"]["adapter_sha256"] != plan["arms"]["r159"]["adapter_sha256"]:
        raise GateError("source snapshot changed")
    x = copy.deepcopy(source)
    x["checkpoints"] = {entry["checkpoint_id"]: {
        "produced_by_update_round": entry["round"],
        "update_count": entry["completed_stage2_rounds"],
        "state_id": entry["state_id"],
        "adapter_sha256": entry["adapter_sha256"],
    }}
    x["completed_stage2_rounds"] = entry["completed_stage2_rounds"]
    x["gpu67_campaign_id"] = plan["campaign_id"]
    x["gpu67_arm"] = arm
    x["gpu67_intent"] = "new instrumented development measurement; no formal/final task"
    x["gpu67_original_panel_manifest_sha256"] = plan["science"]["frozen_manifest_file_sha256"]
    return x


def canonical_json(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def create_file_once(path: Path, body: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(body)


def prepare_run_dir(plan: dict[str, Any], repo_root: Path, run_id: str) -> Path:
    if not RUN_ID.fullmatch(run_id):
        raise GateError("run-id must be fixed 3-65 safe characters")
    root = project_paths(plan)["output_root"]
    root.mkdir(mode=0o770, parents=True, exist_ok=True)
    # Host Root is an existing authorized execution path, but payloads are user UID 1001.
    if os.geteuid() == 0:
        os.chown(root, 1001, 1001)
    work = root / run_id
    if not work.exists():
        work.mkdir(mode=0o770)
        if os.geteuid() == 0:
            os.chown(work, 1001, 1001)
    else:
        if not work.is_dir():
            raise GateError("run id collides with a non-directory")
    for arm in ("r159", "r129"):
        dest = work / arm
        dest.mkdir(mode=0o770, exist_ok=True)
        if os.geteuid() == 0:
            os.chown(dest, 1001, 1001)
        target = dest / "science_manifest.json"
        body = canonical_json(derive_arm_manifest(plan, repo_root, arm))
        if target.exists():
            if target.read_bytes() != body:
                raise GateError(f"{arm} science manifest changed while resuming")
        else:
            create_file_once(target, body)
            if os.geteuid() == 0:
                os.chown(target, 1001, 1001)
    return work


@contextlib.contextmanager
def locked_gpu_pair(plan: dict[str, Any]):
    """Exact physical GPU6/7 fcntl file locks; no interaction with GPU0-5."""
    lock_root = project_paths(plan)["gpu_lock_root"]
    if not lock_root.is_dir():
        raise GateError("existing GPU lock root missing; do not create a new parallel lock authority")
    opened = []
    try:
        for physical in (6, 7):
            file = lock_root / (ALLOWED[physical] + ".lock")
            stream = file.open("a+")
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                stream.close()
                raise GateError(f"GPU{physical} existing launcher flock is busy") from exc
            opened.append(stream)
        yield
    finally:
        for f in reversed(opened):
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            f.close()


def container_command(plan: dict[str, Any], repo_root: Path, work: Path,
                      arm: str, *, kind: str) -> list[str]:
    """Build a Docker argv, never shell=True, never a GPU index fallback."""
    p = project_paths(plan)
    which = next(x for x in plan["device_allowlist"] if x["arm"] == arm)
    index, uuid = which["index"], which["uuid"]
    if index not in (6, 7) or ALLOWED[index] != uuid:
        raise GateError("attempt to target non-authorized GPU")
    adir = adapter_paths(plan, p)[arm]
    carriers = carrier_paths(plan, p)
    python_path = ":".join([
        str(p["core_source_root"] / "src"),
        str(p["runtime_source_root"]),
        str(p["runtime_source_root"] / "vendor/openevo-webshop/src"),
        "/work/experiment",
        str(p["runtime_source_root"] / "vendor/openevo-webshop"),
    ])
    argv = ["docker", "run", "--rm", "--restart", "no", "--network", "none",
            "--gpus", "device=" + uuid,
            "--user", "1001:1001", "--cap-drop=ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "1024", "--tmpfs", "/tmp:rw,exec,nosuid,size=8g,mode=1777",
            "--label", "openevo.owner=experiment-host", "--label", "openevo.experiment=q17-r159-gpu67",
            "--label", "openevo.gpu_uuid=" + uuid,
            "--env", "CUDA_VISIBLE_DEVICES=" + uuid,
            "--env", "NVIDIA_VISIBLE_DEVICES=" + uuid,
            "--env", "TRANSFORMERS_OFFLINE=1", "--env", "HF_HUB_OFFLINE=1",
            "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "PYTHONPATH=" + python_path]
    for path in (p["core_source_root"], p["runtime_source_root"], p["seed_root"],
                 p["model_root"], p["q17_root"]):
        argv.extend(["--mount", f"type=bind,source={path},target={path},readonly"])
    argv.extend(["--mount", f"type=bind,source={repo_root},target=/work/experiment,readonly"])
    argv.extend(["--mount", f"type=bind,source={work},target={work}"])
    argv.extend(["--entrypoint", "/usr/local/bin/python3", plan["image_id"]])
    args_common = ["--runtime-source-root", str(p["runtime_source_root"]),
                   "--model", str(p["model_root"]),
                   "--revision", plan["science"]["base_revision"],
                   "--memory", str(carriers["memory"]),
                   "--skill", str(carriers["skill"]),
                   "--agent", str(carriers["agent"])]
    if kind == "qualify":
        outfile = work / arm / "qualification.json"
        argv.extend(["/work/experiment/" + plan["paths"]["qualifier_script_rel"],
                     *args_common, "--adapter", str(adir), "--output", str(outfile)])
    elif kind == "dev64":
        entry = plan["arms"][arm]
        argv.extend(["/work/experiment/" + plan["paths"]["probe_script_rel"],
                     "--manifest", str(work / arm / "science_manifest.json"),
                     "--panel-manifest", "/work/experiment/" + plan["paths"]["frozen_panel_manifest_rel"],
                     "--checkpoint-id", entry["checkpoint_id"],
                     *args_common,
                     "--seed-root", str(p["seed_root"]), "--sd-adapter", str(adir),
                     "--gpu-uuid", uuid, "--output-dir", str(work / arm / "dev64"),
                     "--instrumentation", "on", "--q06-pre-action-diagnostics",
                     "--q06-lineage-id", "q17_" + arm + "_gpu67_development"])
    else:
        raise GateError("unknown Docker command kind")
    # Safety defense in depth: the only device string must be the expected UUID.
    devices = [argv[i + 1] for i, flag in enumerate(argv) if flag == "--gpus"]
    if devices != ["device=" + uuid] or "--gpus all" in " ".join(argv):
        raise GateError("Docker GPU visibility expanded")
    return argv


def require_ready_qualifications(plan: dict[str, Any], work: Path) -> None:
    for arm in ("r159", "r129"):
        out = work / arm / "qualification.json"
        if not out.is_file():
            raise GateError(f"{arm} zero-task GPU load qualification receipt missing")
        j = read_json(out)
        entry = plan["arms"][arm]
        if (j.get("status") != "PASS" or j.get("adapter_sha256") != entry["adapter_sha256"]
                or j.get("generation_calls") != 0 or j.get("final_panel_access") != 0
                or j.get("formal_rollouts_consumed") != 0 or j.get("inference_started") is not False):
            raise GateError(f"{arm} invalid load-only receipt; cannot consume development tasks")


def ensure_new_stage(work: Path, arm: str, name: str) -> None:
    if (work / arm / name).exists():
        raise GateError(f"{arm} {name} already exists; fail closed instead of rerunning/corrupting evidence")


def invoke(plan: dict[str, Any], repo_root: Path, work: Path, arm: str, kind: str, timeout: int) -> dict[str, Any]:
    """One exact project-owned subprocess, never mutating a preexisting container."""
    argv = container_command(plan, repo_root, work, arm, kind=kind)
    prefix = work / arm / kind
    if kind == "dev64":
        (work / arm / "dev64").mkdir(mode=0o770)
        if os.geteuid() == 0:
            os.chown(work / arm / "dev64", 1001, 1001)
    stdout = prefix.with_suffix(".stdout.log")
    stderr = prefix.with_suffix(".stderr.log")
    if stdout.exists() or stderr.exists():
        raise GateError(f"{arm} {kind} logs already exist; cannot replay without new run id")
    with stdout.open("xb") as so, stderr.open("xb") as se:
        started = time.monotonic()
        try:
            cp = subprocess.run(argv, stdout=so, stderr=se, timeout=timeout, check=False)
            rc = cp.returncode
        except subprocess.TimeoutExpired as exc:
            raise GateError(f"{arm} {kind} Docker timed out; inspect exact project-owned container before resume") from exc
    data = {"arm": arm, "kind": kind, "status": "PASS" if rc == 0 else "FAIL",
            "exit_code": rc, "wall_seconds": time.monotonic() - started,
            "stdout_sha256": sha_path(stdout), "stderr_sha256": sha_path(stderr),
            "gpu_uuid": next(x["uuid"] for x in plan["device_allowlist"] if x["arm"] == arm)}
    if rc:
        raise GateError(f"{arm} {kind} exited {rc}, see {stderr} (no retry/overwrite)")
    return data


def verify_dev64_output(plan: dict[str, Any], work: Path, arm: str) -> dict[str, Any]:
    j = read_json(work / arm / "dev64/summary.json")
    if j.get("status") != "PASS" or j.get("instrumentation") != "on":
        raise GateError(f"{arm} instrumented development probe failed")
    if (j.get("task_count") != 64 or j.get("panel_sha256") != plan["science"]["frozen_panel_sha"]
            or j.get("final_panel_access") != 0 or j.get("formal_task_consumption") != 0):
        raise GateError(f"{arm} dev panel identity/counter mismatch")
    if j.get("checkpoint", {}).get("adapter_sha256") != plan["arms"][arm]["adapter_sha256"]:
        raise GateError(f"{arm} model SHA mismatch in science summary")
    rows = j.get("rows", [])
    if len(rows) != 64 or any(row.get("engineering_invalid") for row in rows):
        raise GateError(f"{arm} incomplete/engineering-invalid task receipts")
    if any(any(not step.get("generation_diagnostics") or "q06_pre_action" not in step
                   for step in r["episode"].get("trace", [])) for r in rows):
        raise GateError(f"{arm} missing pre-action / entropy instrumentation")
    return j


def summarize_pair(plan: dict[str, Any], work: Path) -> dict[str, Any]:
    s = {arm: verify_dev64_output(plan, work, arm) for arm in ("r159", "r129")}
    def task_key(r: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(r[k] for k in ("task_identity", "paired_generation_seed", "probe_ordinal",
                                    "source_session_index", "worker_seed"))
    a = {task_key(r): r for r in s["r159"]["rows"]}
    b = {task_key(r): r for r in s["r129"]["rows"]}
    if len(a) != 64 or len(b) != 64 or set(a) != set(b):
        raise GateError("two GPU arms do not share exactly the same task and seed keys")
    same_initial_prefix = 0
    same_per_step_prefix = 0
    paired_steps = 0
    deltas = []
    for k in sorted(a):
        x, y = a[k]["episode"], b[k]["episode"]
        deltas.append(float(x["score"]) - float(y["score"]))
        z1, z2 = x["trace"], y["trace"]
        if not z1 or not z2:
            raise GateError("empty episode trace")
        if z1[0]["q06_pre_action"]["prompt_sha256"] != z2[0]["q06_pre_action"]["prompt_sha256"]:
            raise GateError("task first pre-action prompts changed across models")
        same_initial_prefix += 1
        for st1, st2 in zip(z1, z2):
            paired_steps += 1
            if st1["q06_pre_action"]["prompt_sha256"] == st2["q06_pre_action"]["prompt_sha256"]:
                same_per_step_prefix += 1
    metrics = {}
    for arm, j in s.items():
        m = j["metrics"]
        for key in ("reasoning_generation", "action_generation", "click_choice"):
            if m.get(key) is None:
                raise GateError(f"{arm} missing instrumentation metric {key}")
        metrics[arm] = {
            "model_sha256": plan["arms"][arm]["adapter_sha256"],
            "task_score_0_100": 100 * float(m["mean_score"]),
            "exact_success_rate": m["success_rate"],
            "positive_reward_rate": m["positive_reward_rate"],
            "reasoning_token_entropy_nats": m["reasoning_generation"]["entropy_mean_nats"],
            "action_token_entropy_nats": m["action_generation"]["entropy_mean_nats"],
            "selected_action_mean_nll_nats": m["selected_action_nll_mean_nats"],
            "conditional_click_destination_entropy_nats": m["click_choice"]["mean_entropy_nats"],
            "instrumented_step_count": m["click_choice"]["observed_step_count"],
        }
    return {
        "schema": "openevo.q17.gpu67.instrumented_frozen64_pair.v1", "status": "PASS",
        "frozen_dev_panel_sha256": plan["science"]["frozen_panel_sha"],
        "task_count_per_arm": 64, "run_id": work.name,
        "arms": metrics,
        "paired_dev_mean_score_difference_R159_minus_R129": statistics.mean(deltas),
        "first_step_identical_pre_action_prompts": same_initial_prefix,
        "identical_shared_pre_action_prompts_observed": same_per_step_prefix,
        "paired_on_policy_steps_total": paired_steps,
        "science_limitations": [
            "Re-run development diagnostics only; not original uninstrumented benchmark nor final panel.",
            "After the two policies choose different actions their visited states diverge; equal task ID is not equal pre-action state.",
            "Matched pre-action prompt SHA is context evidence, not a KL or causal experiment by itself.",
            "Raw traces contain task text and remain private on server; this summary omits task strings.",
            "GPU-generated results count as new experiments, not recovery of historical logits.",
        ],
        "execution": {"gpu_indices": [6, 7], "other_gpu_indices_used": [],
                      "formal_task_consumption": 0, "final_panel_access": 0,
                      "training_updates": 0, "source_checkpoint_mutations": 0},
    }


def work_dir(plan: dict[str, Any], run_id: str) -> Path:
    if not RUN_ID.fullmatch(run_id):
        raise GateError("invalid run id")
    return project_paths(plan)["output_root"] / run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "qualify", "dev64", "report", "commands"),
                        default="preflight")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--approve-development-rollouts", action="store_true")
    parser.add_argument("--full-byte-check", action="store_true",
                        help="preflight: hash full 2+ GB adapter weights; always enforced for GPU work")
    args = parser.parse_args(argv)
    repo = args.repo_root.resolve()
    plan_path = (args.plan or (repo / DEFAULT_PLAN)).resolve()
    try:
        plan = read_json(plan_path)
        verify_plan(plan)
        if args.phase == "report":
            result = summarize_pair(plan, work_dir(plan, args.run_id))
            print(json.dumps(result, sort_keys=True, ensure_ascii=False))
            return 0
        static = cpu_source_check(plan, repo, full_hash=(args.full_byte_check or args.phase in ("qualify", "dev64")))
        devices = live_gpu_gate(plan)
        immutable_image = docker_image_gate(plan)
        if args.phase == "preflight":
            print(json.dumps({"status": "READY_FOR_GPU_QUALIFY" if args.full_byte_check else "STATIC_PREFLIGHT_PASS_LARGE_SHA_PENDING",
                              "plan_path": str(plan_path), "science": static, "physical_allowlist": devices,
                              "image_digest": immutable_image, "other_gpu_compute_calls": 0},
                             sort_keys=True, ensure_ascii=False))
            return 0
        if args.phase == "commands":
            if not RUN_ID.fullmatch(args.run_id):
                raise GateError("commands requires --run-id")
            work = work_dir(plan, args.run_id)
            commands = {kind: {arm: container_command(plan, repo, work, arm, kind=kind)
                               for arm in ("r159", "r129")}
                        for kind in ("qualify", "dev64")}
            print(json.dumps({"status": "GPU_6_7_COMMANDS_ONLY_NOT_EXECUTED", "commands": commands,
                              "allowlist": [6, 7], "original_weight_changes": 0},
                             sort_keys=True, ensure_ascii=False))
            return 0
        if not RUN_ID.fullmatch(args.run_id):
            raise GateError("GPU operation requires --run-id")
        if args.phase == "dev64" and not args.approve_development_rollouts:
            raise GateError("dev64 needs --approve-development-rollouts, qualifying alone consumes zero tasks")
        with locked_gpu_pair(plan):
            live_gpu_gate(plan)  # Immediate point-of-use gate; never infer availability from earlier preflight.
            work = prepare_run_dir(plan, repo, args.run_id)
            if args.phase == "qualify":
                for arm in ("r159", "r129"):
                    if (work / arm / "qualification.json").exists():
                        raise GateError(f"{arm} qualification already exists; no implicit repeat")
                    live_gpu_gate(plan)
                    info = invoke(plan, repo, work, arm, "qualify", timeout=900)
                    j = read_json(work / arm / "qualification.json")
                    if j.get("status") != "PASS" or j.get("adapter_sha256") != plan["arms"][arm]["adapter_sha256"]:
                        raise GateError(f"{arm} model load qualifier had incorrect identity")
                    print(json.dumps(info, sort_keys=True), flush=True)
                require_ready_qualifications(plan, work)
                print(json.dumps({"status": "QUALIFICATION_PASS_2_GPU_NO_ROLLOUT", "run_id": args.run_id,
                                  "allowlist": [6, 7], "model_calls": 0, "dev_tasks": 0}))
                return 0
            require_ready_qualifications(plan, work)
            for arm in ("r159", "r129"):
                ensure_new_stage(work, arm, "dev64")
            # Both devices are frozen by kernel locks for the full paired phase.
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(invoke, plan, repo, work, arm, "dev64", 14400)
                           for arm in ("r159", "r129")]
                for future in concurrent.futures.as_completed(futures):
                    print(json.dumps(future.result(), sort_keys=True), flush=True)
            result = summarize_pair(plan, work)
            create_file_once(work / "paired-metrics.json", canonical_json(result))
            print(json.dumps({"status": "PAIRED_DEVELOPMENT64_PASS", "paired_receipt": str(work / "paired-metrics.json"),
                              "new_rollouts": 128, "final_panel_access": 0, "other_gpus_used": []}, sort_keys=True))
            return 0
    except (GateError, KeyError, ValueError, FileNotFoundError, subprocess.TimeoutExpired) as error:
        print(json.dumps({"status": "BLOCKED", "phase": args.phase, "reason": str(error),
                          "gpu_scope": [6, 7], "no_other_gpu_launch": True}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())