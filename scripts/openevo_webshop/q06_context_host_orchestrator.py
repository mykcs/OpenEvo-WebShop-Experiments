#!/usr/bin/env python3
"""Q06's single frozen historical context cell, via canonical Formal Start."""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import hashlib
from pathlib import Path
from scripts.openevo_webshop.q06_context_probe_inputs import validate_inputs, SOURCE_SHA, CHECKPOINT
MANIFEST_REL = 'docs/evidence/advisor-q06-e2-paired-analysis-20261001/Q06_GDR_CONTEXT_PROBE_MANIFEST.json'
MANIFEST_SHA = 'edcaaf2b216e3f8713574105e52037675ef3ea091b81892d557dda4af2f9c33c'
AUTHORITY_REL = 'docs/evidence/advisor-q06-e2-paired-analysis-20261001/Q06_GDR_CONTEXT_IDENTITIES.json'
ENDPOINTS = {CHECKPOINT}
GPU = re.compile(r'^GPU-[0-9a-f-]{36}$')
IMAGE = 'sha256:d0f8796010899ae41a7af8ae3f44f47b96dea73b297b5af124049b3ae88e2850'
def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_digest_identity(value: object) -> str | None:
    """Normalize bare and explicitly prefixed SHA-256 display forms."""
    if not isinstance(value, str):
        return None
    digest = value.removeprefix("sha256:")
    return digest if re.fullmatch(r"[0-9a-f]{64}", digest) else None


def same_sha256_digest(actual: object, declared: object) -> bool:
    """Compare SHA-256 identities only when both operands are valid digests."""
    actual_digest = sha256_digest_identity(actual)
    declared_digest = sha256_digest_identity(declared)
    return (actual_digest is not None and declared_digest is not None and
            actual_digest == declared_digest)


def same_hashed_path_reference(actual: object, expected: object) -> bool:
    """Compare exact paths and validated, prefix-normalized SHA-256 identities."""
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return False
    if set(actual) != set(expected) or actual.get("path") != expected.get("path"):
        return False
    for key in actual:
        if key == "sha256":
            if not same_sha256_digest(actual.get(key), expected.get(key)):
                return False
        elif actual.get(key) != expected.get(key):
            return False
    return "sha256" in actual


def obj(path: Path) -> dict:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def bound(repo: Path, ref: dict) -> Path:
    path = (repo / ref["path"]).resolve()
    require(path == repo or repo in path.parents, "authority path escapes repository")
    require(sha(path) == ref["sha256"], f"authority bytes drift: {path}")
    return path



def route_paths(authority_repo, endpoint, release_arg=None, router_arg=None):
    require(endpoint == CHECKPOINT and release_arg is not None and router_arg is not None,
            'Q06 requires its explicit release and current router')
    paths = tuple(Path(p).resolve() for p in (release_arg, router_arg))
    for path in paths:
        require(authority_repo in path.parents, 'Q06 authority path escapes repository')
    return paths

def formal_gate(authority_repo: Path, endpoint: str, selected_uuid: str,
                receipt_path: Path, output: Path,
                release_arg: Path | None = None,
                router_arg: Path | None = None) -> tuple[dict, dict]:
    require(GPU.fullmatch(selected_uuid) is not None, "invalid selected UUID")
    require(receipt_path.is_file(), "oe Formal Start receipt missing")
    start = obj(receipt_path)
    sys.path.insert(0, str(authority_repo))
    from scripts.openevo_webshop import prelaunch_zero_formal
    require(start.get("schema") == prelaunch_zero_formal.FORMAL_START_SCHEMA and
            start.get("receipt_sha256") == prelaunch_zero_formal.canonical_sha256(
                {k: v for k, v in start.items() if k != "receipt_sha256"}),
            "oe Formal Start receipt identity drift")
    require(start.get("status") == "FORMAL_START_COMMITTED" and
            start.get("formal_start_committed") is True and
            start.get("formal_task_consumption_may_begin") is True,
            "oe Formal Start not committed")
    require(start.get("selected_gpu_uuids") == [selected_uuid] and
            os.getenv("OPENEVO_ASSIGNED_GPU_UUIDS") == selected_uuid and
            os.getenv("OPENEVO_AUTHORIZED_GPU_UUIDS") == selected_uuid,
            "selected GPU identity drift")
    require(start.get("campaign_id") == "20260928-2338-post-advisor-unanswered",
            "campaign identity drift")
    require(start.get("image_id") ==
            "sha256:d0f8796010899ae41a7af8ae3f44f47b96dea73b297b5af124049b3ae88e2850",
            "scientific image drift")
    release_path, router_path = route_paths(
        authority_repo, endpoint, release_arg=release_arg, router_arg=router_arg
    )
    release_rel = str(release_path.relative_to(authority_repo))
    router_rel = str(router_path.relative_to(authority_repo))
    release, router = obj(release_path), obj(router_path)
    require(release.get("status") == "launch-authorized" and
            release.get("formal_task_consumption_allowed") is True and
            router.get("status") == "launch-authorized" and
            router.get("formal_task_consumption_allowed") is True,
            "Q06 release/router not authorized")
    require(router.get("release_receipt_sha256") == sha(release_path),
            "router release hash drift")
    require(selected_uuid in router.get("authorized_gpu_uuids", []),
            "selected UUID not in Q06 router authority")
    require(router.get("endpoint") == endpoint and release.get("endpoint") == endpoint,
            "endpoint routing drift")
    require(release.get("formal_rollout_limit") == 64 and
            router.get("formal_rollout_limit") == 64 and
            release.get("final_panel_access_limit") == 0 and
            router.get("final_panel_access_allowed") is False,
            "arm cap or final lock drift")
    require(not git(authority_repo, "status", "--porcelain", "--untracked-files=all"),
            "authority checkout is dirty")
    for relative in (MANIFEST_REL, AUTHORITY_REL, release_rel, router_rel,
                     release["preregistration"]["path"],
                     release["passport"]["path"],
                     release["path_contract"]["path"],
                     release["host_orchestrator"]["path"],
                     release["execution_readiness"]["profile"]["path"]):
        committed = subprocess.run(
            ["git", "-C", str(authority_repo), "show", f"HEAD:{relative}"],
            check=True, capture_output=True).stdout
        require((authority_repo / relative).read_bytes() == committed,
                f"uncommitted Q06 authority path: {relative}")
    prereg = obj(bound(authority_repo, release["preregistration"]))
    passport = obj(bound(authority_repo, release["passport"]))
    bound(authority_repo, release["path_contract"])
    bound(authority_repo, release["host_orchestrator"])
    profile = obj(bound(authority_repo, release["execution_readiness"]["profile"]))
    require(passport.get("status") == "FROZEN" and
            passport.get("experiment_id") == release.get("experiment_id"),
            "FROZEN Passport identity missing")
    require(release.get("experiment_identity") == {
                "experiment_id": passport["experiment_id"],
                "passport_sha256": passport.get("passport_payload_sha256")},
            "release Passport payload binding drift")
    from scripts.openevo_webshop import experiment_passport_gate
    require(experiment_passport_gate.validate_binding(
                authority_repo, release["experiment_identity"])["status"] == "PASS",
            "Q06 Passport/Registry binding incomplete")
    science = prereg.get("frozen_scientific_contract", {})
    require(science.get("endpoints") == sorted(ENDPOINTS) and
            science.get("panel_tasks_per_endpoint") == 64 and
            science.get("fixed_passes_per_endpoint") == 1 and
            science.get("total_probe_episodes") == 64 and
            science.get("adaptive_reruns") ==
            science.get("training_rollouts") ==
            science.get("teacher_calls") == 0 and
            science.get("final_panel_access_allowed") is False,
            "preregistration science/zero-final drift")
    require(profile.get("run_id") == start.get("run_id") and
            profile.get("execution", {}).get("mode") == "host-orchestrator" and
            profile.get("source", {}).get("experiment", {}).get("revision") == SOURCE_SHA,
            "oe profile/source identity drift")
    command = profile.get("command", [])
    require("--output-dir" in command and
            command[command.index("--output-dir") + 1] == str(output),
            "science output root differs from hash-bound oe profile")
    readiness = start.get("readiness", {})
    profile_ref = release["execution_readiness"]["profile"]
    readiness_profile = readiness.get("profile")
    require(isinstance(readiness_profile, dict) and
            readiness_profile.get("path") ==
                str((authority_repo / profile_ref["path"]).resolve()) and
            same_sha256_digest(profile_ref.get("sha256"),
                               readiness_profile.get("sha256")),
            "oe readiness/profile binding drift")
    ready_ref = readiness.get("ready_receipt", {})
    ready_path = Path(str(ready_ref.get("path", ""))).resolve()
    ready_actual_sha = sha(ready_path) if ready_path.is_file() else None
    require(same_sha256_digest(ready_actual_sha, ready_ref.get("sha256")),
            "oe READY receipt bytes drift")
    ready = obj(ready_path)
    ready_release_ref = ready.get("release_receipt")
    expected_release_ref = {
        "path": str(release_path.resolve()),
        "sha256": sha(release_path),
    }
    require(ready.get("verdict") == "READY" and
            ready.get("profile") == readiness["profile"] and
            same_hashed_path_reference(ready_release_ref, expected_release_ref) and
            same_hashed_path_reference(ready.get("path_contract"),
                                       release.get("path_contract")),
            "oe READY receipt does not bind Q06 release/profile")
    pre_ref = start.get("prelaunch_receipt", {})
    pre_path = Path(str(pre_ref.get("path", ""))).resolve()
    pre_actual_sha = sha(pre_path) if pre_path.is_file() else None
    require(same_sha256_digest(pre_actual_sha, pre_ref.get("sha256")),
            "oe prelaunch receipt bytes drift")
    prelaunch_zero_formal.validate_prelaunch_receipt(
        obj(pre_path), expected_gpu_uuids=[selected_uuid])
    from scripts import oe_impl
    profile["_formal_start_context"] = readiness
    expected_command = oe_impl.host_orchestrator_identity_command(
        profile, [selected_uuid], Path(start["output_root"]), "disabled", receipt_path)
    expected_identity = oe_impl._command_identity(expected_command, start["image_id"])
    require(start.get("command_identity") == expected_identity,
            "oe Formal Start command identity drift")
    return release, start



def build_probe_command(manifest, repo, source, output, gpu_uuid):
    endpoint, carriers = manifest['endpoint'], manifest['common_carriers']
    seed, model = Path(manifest['seed_root']['path']), Path(manifest['model']['local_model_path'])
    mounts = [source, repo, seed, model, Path(endpoint['adapter_path']),
              Path(carriers['memory_path']), Path(carriers['skill_path']), Path(carriers['agent_system_path'])]
    command = ['docker', 'run', '--rm', '--network', 'none', '--user', '1001:1001',
               '--gpus', 'device=' + gpu_uuid]
    for path in dict.fromkeys(mounts):
        require(path.exists() and not path.is_symlink(), 'Q06 mount missing or symlink: ' + str(path))
        command += ['--mount', f'type=bind,src={path},dst={path},readonly']
    command += ['--mount', f'type=bind,src={output},dst={output}',
                '--tmpfs', '/tmp:rw,exec,nosuid,size=1g,mode=1777']
    for value in ['HOME=/tmp', 'TMPDIR=/tmp', 'HF_HOME=/tmp/hf', 'XDG_CACHE_HOME=/tmp/cache',
                  'TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_openevo', 'TRITON_CACHE_DIR=/tmp/triton',
                  'PYTHONDONTWRITEBYTECODE=1', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1',
                  'CUDA_VISIBLE_DEVICES=' + gpu_uuid,
                  f'PYTHONPATH={source}:{source}/vendor/openevo-webshop/src']:
        command += ['--env', value]
    command += ['--entrypoint', 'python3', IMAGE,
                str(source / 'scripts/openevo_webshop/q17_fixed_checkpoint_probe.py'),
                '--manifest', str(repo / MANIFEST_REL),
                '--panel-manifest', str(source / manifest['task_panel_manifest_path']),
                '--checkpoint-id', CHECKPOINT, '--runtime-source-root', str(source),
                '--seed-root', str(seed), '--model', str(model),
                '--revision', manifest['model']['revision'], '--sd-adapter', endpoint['adapter_path'],
                '--memory', carriers['memory_path'], '--skill', carriers['skill_path'],
                '--agent', carriers['agent_system_path'], '--gpu-uuid', gpu_uuid,
                '--output-dir', str(output), '--instrumentation', 'on']
    return command


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--authority-repo', type=Path, required=True)
    p.add_argument('--execution-source', type=Path, required=True)
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--router', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--gpu-uuid')
    p.add_argument('--qualify-only', action='store_true')
    a = p.parse_args()
    repo, source, output = a.authority_repo.resolve(), a.execution_source.resolve(), a.output_dir.absolute()
    manifest = validate_inputs(repo, source, MANIFEST_SHA)
    require(not output.exists() and not output.is_symlink(), 'Q06 pass already consumed or output exists')
    require(output.parent.is_dir() and not output.parent.is_symlink(), 'Q06 output parent must be prepared')
    require(not git(repo, 'status', '--porcelain', '--untracked-files=all'), 'Q06 authority checkout dirty')
    release_path, router_path = route_paths(repo, CHECKPOINT, a.release, a.router)
    release, router = obj(release_path), obj(router_path)
    require(release.get('source_manifest') == {'path': MANIFEST_REL, 'sha256': MANIFEST_SHA},
            'Q06 release must bind the exact context manifest')
    prereg = obj(bound(repo, release['preregistration']))
    science = prereg['frozen_scientific_contract']
    require(science['manifest'] == release['source_manifest'] and science['endpoints'] == [CHECKPOINT]
            and science['model'] == manifest['model'] and science['sampling'] == manifest['sampling']
            and science['common_carriers'] == manifest['common_carriers']
            and science['question_id'] == 'Q06_TOKEN_CLICK_ENTROPY'
            and science['optimizer_updates'] == 0, 'Q06 native contract drift')
    require(science.get('panel_tasks_per_endpoint') == 64
            and science.get('fixed_passes_per_endpoint') == 1
            and science.get('total_probe_episodes') == 64
            and all(science.get(key) == 0 for key in
                    ('adaptive_reruns', 'training_rollouts', 'teacher_calls'))
            and science.get('final_panel_access_allowed') is False,
            'Q06 static budget or final-panel lock drift')
    if a.qualify_only:
        print(json.dumps({'status': 'PASS_Q06_STATIC_INPUTS', 'new_model_calls': 0,
                          'new_environment_steps': 0, 'launch_authority_granted': False}))
        return 0
    require(a.gpu_uuid is not None, 'Q06 selected GPU missing')
    receipt = Path(os.environ.get('OPENEVO_FORMAL_START_RECEIPT', '')).resolve()
    release, start = formal_gate(repo, CHECKPOINT, a.gpu_uuid, receipt, output, a.release, a.router)
    from scripts.openevo_webshop import prelaunch_zero_formal
    qualification_ref = release['zero_formal_qualification']
    qualification_path = Path(qualification_ref['path'])
    require(qualification_path.is_absolute() and not qualification_path.is_symlink()
            and qualification_path.is_file()
            and same_sha256_digest(sha(qualification_path), qualification_ref['sha256']),
            'Q06 externally stored qualification bytes drift')
    qualification = obj(qualification_path)
    require(qualification.get('schema') == 'openevo.q06.zero-formal-qualification.202610011303'
            and qualification.get('receipt_sha256') == prelaunch_zero_formal.canonical_sha256(
                {k: v for k, v in qualification.items() if k != 'receipt_sha256'})
            and qualification.get('formal_task_consumed') is False
            and qualification.get('scientific_state_mutated') is False
            and qualification.get('status') == 'ZERO_FORMAL_QUALIFIED'
            and qualification.get('execution_sha') == SOURCE_SHA
            and qualification.get('campaign_id') == start['campaign_id']
            and qualification.get('selected_gpu_uuids') == [a.gpu_uuid]
            and all(qualification.get(key) == 0 for key in
                    ('formal_task_budget', 'task_bearing_calls', 'model_calls',
                     'training_calls', 'final_panel_access'))
            and same_sha256_digest(qualification.get('candidate_hashes', {}).get('runtime_profile'),
                                   release['execution_readiness']['profile']['sha256']),
            'Q06 qualification is not exact selected-GPU zero-formal PASS')
    qualified_prelaunch_ref = qualification['prelaunch_receipt']
    qualified_prelaunch_path = Path(qualified_prelaunch_ref['path'])
    require(qualified_prelaunch_path.is_file() and not qualified_prelaunch_path.is_symlink()
            and same_sha256_digest(sha(qualified_prelaunch_path), qualified_prelaunch_ref['sha256']),
            'Q06 qualified prelaunch receipt bytes drift')
    qualified_prelaunch = obj(qualified_prelaunch_path)
    prelaunch_zero_formal.validate_prelaunch_receipt(qualified_prelaunch, expected_gpu_uuids=[a.gpu_uuid])
    require(qualified_prelaunch.get('image_id') == IMAGE,
            'Q06 qualified scientific image drift')
    command = build_probe_command(manifest, repo, source, output, a.gpu_uuid)
    # Exclusive mkdir is the one-pass consumption boundary. Every failure preserves it.
    output.mkdir(mode=0o700, exist_ok=False)
    os.chown(output, 1001, 1001)
    claim = {'schema': 'openevo.q06.context-pass-claim.202610011303', 'status': 'CONSUMED',
             'checkpoint_id': CHECKPOINT, 'episode_cap': 64, 'selected_gpu_uuid': a.gpu_uuid,
             'formal_start': {'path': str(receipt), 'sha256': sha(receipt)},
             'command': command, 'model_calls_before_launch': 0}
    with (output / 'context-pass-claim.json').open('x') as stream:
        json.dump(claim, stream, indent=2, sort_keys=True)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    directory_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return subprocess.run(command, check=False).returncode

if __name__ == '__main__':
    raise SystemExit(main())
