#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.openevo_webshop import ceiling1_stage2_arm_controller as base
from scripts.openevo_webshop import bounded_effective_state_recovery_202609282338 as recovery
from scripts.openevo_webshop.post_advisor_e3_guard_202609300248 import (
    verify_preparation, SOURCE_FILES, BASE_C, E3_AUTHORITY, E3_RELEASE,
    E3_RELEASE_OTHERS, E3_ROUTER, e3_b_material,
)
from scripts.openevo_webshop.post_advisor_e3_bounded_short_fork_202609300248 import validate_post_advisor_activation
from scripts.openevo_webshop.post_advisor_e3_runtime_safety_202609300248 import GPU_UUID, require_post_advisor_gpu_assignment, exclusive_arm_lock

R151_STATE = Path("/workspace/runs/bounded-effective-state-gdr-recovery-1e7c4952-20260916/off/round-151/state-after.json")
R151_HEALTH = Path("/workspace/runs/bounded-effective-state-gdr-recovery-1e7c4952-20260916/off/round-151/carrier-health-boundary.json")
ROUNDS = list(range(152, 160))
READY_SHA = "5f9297eaa21c0dd0e385cc5731ec090476cf09464239e58f3c41e8680bad640c"
RELEASE_SHA = "2ea603e403641cec3494aa12bde869130efcca946657b54c6c5a97bf9cbd6caa"
OFF_MODE = "BOUNDED_OFF"
MAG_MODE = "MAGNITUDE_SCALAR_GATE_" + "V" + "1"  # legacy-revision-compat: exact historical frozen identifier
SUPPORTED_PERSISTENT_STATE_RANKS = (8, 16, 64)


def supports_persistent_state_rank(rank: int) -> bool:
    return int(rank) in SUPPORTED_PERSISTENT_STATE_RANKS


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def git_head(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, text=True, capture_output=True,
    ).stdout.strip()


def validate(repo: Path, authority_path: Path, ready_path: Path, release_path: Path):
    authority = load(authority_path)
    if authority.get("schema") != "openevo.post-advisor.e3-rank-short-fork-authority.202609300248":
        raise RuntimeError("short-fork authority schema drift")
    if authority.get("status") != "AUTHORIZED":
        raise RuntimeError("short-fork authority is not authorized")
    head = git_head(repo)
    if authority.get("execution_git_sha") != head:
        raise RuntimeError(f"authority execution SHA drift: {authority.get('execution_git_sha')} != {head}")
    if authority.get("fork_origin") != "R151_AFTER" or authority.get("rounds") != ROUNDS:
        raise RuntimeError("short-fork round/fork scope drift")
    rank = int(authority.get("persistent_state_rank", -1))
    if not supports_persistent_state_rank(rank):
        raise RuntimeError("short-fork rank outside preregistration")
    treatment = authority.get("treatment_mode")
    if treatment != OFF_MODE:
        raise RuntimeError("short-fork treatment outside preregistration")
    if treatment == MAG_MODE and rank != 128:
        raise RuntimeError("magnitude-only advisor treatment must keep rank128")
    managed = list(map(int, authority.get("managed_gpu_indices") or []))
    if len(managed) != 4 or len(set(managed)) != 4 or any(index not in GPU_UUID for index in managed):
        raise RuntimeError("short-fork authority must bind four mapped GPU indices")
    if int(authority.get("final_panel_access_limit", -1)) != 0:
        raise RuntimeError("final-panel boundary drift")
    if authority.get("historical_rank128_control_rerun") is not False:
        raise RuntimeError("historical rank128 rerun boundary drift")
    if sha(ready_path) != READY_SHA or sha(release_path) != RELEASE_SHA:
        raise RuntimeError("historical READY/release content identity drift")

    state = load(R151_STATE)
    health = load(R151_HEALTH)
    if sha(R151_STATE) != authority.get("historical_r151_state_sha256"):
        raise RuntimeError("R151 state SHA drift")
    if state.get("state_id") != authority.get("historical_r151_state_id"):
        raise RuntimeError("R151 state id drift")
    if sha(R151_HEALTH) != authority.get("historical_r151_health_sha256"):
        raise RuntimeError("R151 health SHA drift")
    if health.get("action") != "CONTINUE" or health.get("ledger_status") != "PASS" or health.get("round_index") != 151:
        raise RuntimeError("R151 health is not a legal continuation boundary")

    initial = Path(authority["initial_adapter_path"]).resolve()
    if sha(initial / "adapter_model.safetensors") != authority.get("initial_adapter_sha256"):
        raise RuntimeError("initial adapter SHA drift")
    if int(load(initial / "adapter_config.json").get("r", -1)) != rank:
        raise RuntimeError("initial adapter rank drift")
    if authority.get("engineering_recovery"):
        raise RuntimeError("E3 recovery requires a separate E3-specific authority")
    if Path(authority["output_root"]).resolve().exists():
        p = Path(authority["output_root"]).resolve() / "execution-state.json"
        if not p.is_file():
            leftovers = list(Path(authority["output_root"]).resolve().glob("round-*"))
            if leftovers:
                raise RuntimeError("output has round artifacts without execution-state")
    return authority, load(ready_path)


def seed(authority: dict, authority_path: Path, ready_path: Path, release_path: Path) -> None:
    out = Path(authority["output_root"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "execution-state.json"
    rank = int(authority["persistent_state_rank"])
    treatment = authority["treatment_mode"]
    initial = Path(authority["initial_adapter_path"]).resolve()

    historical = load(R151_STATE)
    sid = base.state_id(
        Path(historical["merged_base_receipt"]),
        initial,
        Path(historical["memory"]),
        Path(historical["skill"]),
        Path(historical["agent_system"]),
        151,
    )
    if state_path.is_file():
        existing = load(state_path)
        checks = {
            "current_sd_adapter": str(initial),
            "bounded_state_rank": rank,
            "bounded_treatment_mode": treatment,
            "post_advisor_short_fork_authority_sha256": (
                recovery.validate_post_advisor_recovery(authority)["predecessor_arm_authority_sha256"]
                if authority.get("engineering_recovery") else sha(authority_path)
            ),
        }
        if authority.get("engineering_recovery"):
            next_round = int(existing.get("next_round", -1))
            if next_round < 152 or next_round > 160 or existing.get("rollouts_consumed") != next_round * 128:
                raise RuntimeError("post-advisor recovery round/rollout control accounting drift")
            if next_round == 152 and existing.get("current_sd_adapter") != str(initial):
                raise RuntimeError("post-advisor recovery inherited adapter drift")
            checks.pop("current_sd_adapter")
        else:
            checks.update(next_round=152, rollouts_consumed=19456)
        for key, value in checks.items():
            if existing.get(key) != value:
                raise RuntimeError(f"existing execution state drift: {key}")
        return

    carrier = load(Path(load(ready_path)["carrier"]["adoption_receipt_path"]))
    historical.update(
        status="RUNNING",
        state_id=sid,
        current_sd_adapter=str(initial),
        current_sd_artifact_id=f"post-advisor-{treatment.lower()}-rank{rank}-r151-after",
        sd_component_count=rank,
        bounded_state_rank=rank,
        bounded_treatment_mode=treatment,
        bounded_logical_history_count=152,
        next_round=152,
        rollouts_consumed=19456,
        post_advisor_fork_origin="R151_AFTER",
        post_advisor_source_state_id=load(R151_STATE)["state_id"],
        post_advisor_initial_adapter_sha256=authority["initial_adapter_sha256"],
        post_advisor_short_fork_authority_sha256=sha(authority_path),
        formal_ready_package_sha256=sha(ready_path),
        owner_launch_release_sha256=sha(release_path),
        carrier_adoption_identity_sha256=carrier["adoption_identity_sha256"],
        final_panel_access_count=0,
        external_teacher_calls=0,
    )
    base.atomic_json(state_path, historical)


def common_cli(repo: Path, authority: dict, authority_path: Path, ready_path: Path, release_path: Path):
    ready = load(ready_path)
    rb = ready["runtime_bindings"]
    carrier = ready["carrier"]
    transition = ready["formal_transition"]
    return [
        sys.executable,
        str(repo / "scripts/openevo_webshop/post_advisor_e3_bounded_short_fork_202609300248.py"),
        "--experiment-arm-id", "BOUNDED_OFF",
        "--treatment-mode", authority["treatment_mode"],
        "--repo", str(repo),
        "--analysis-root", rb["analysis_root"],
        "--seed-root", rb["seed_root"],
        "--output", authority["output_root"],
        "--stage2-activation", rb["stage2_activation"],
        "--capacity-policy", rb["capacity_policy"],
        "--harness-profile", "201.1",
        "--step-seed-policy", "injective-" + "v" + "1",  # legacy-revision-compat: exact historical frozen identifier
        "--step-seed-amendment", str(repo / rb["step_seed_amendment"]),
        "--harness201-carrier-root", rb["harness201_carrier_root"],
        "--carrier-scope-package", str(repo / carrier["scope_package_path"]),
        "--carrier-health-policy", str(repo / carrier["health_policy_path"]),
        "--carrier-contract-identities", str(repo / carrier["identity_path"]),
        "--carrier-adoption-receipt", carrier["adoption_receipt_path"],
        "--carrier-contract-fail-noop-amendment", str(repo / ready["carrier_contract_fail_noop_amendment"]["path"]),
        "--ready-package", str(ready_path),
        "--owner-launch-release", str(release_path),
        "--execution-git-sha", authority["execution_git_sha"],
        "--formal-transition-launcher", str(repo / "scripts/openevo_webshop/launch_post_advisor_e3_rank_capacity_transition_202609300248.sh") ,
        "--empty-history-manifest", str(repo / transition["empty_history_manifest"]),
        "--adapter-config-template", str(repo / transition["adapter_config_template"]),
        "--post-advisor-short-fork-authority", str(authority_path),
        *(["--carrier-health-resolution", authority["carrier_health_resolution"]["path"]]
          if authority.get("carrier_health_resolution") else []),
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--authority", type=Path, required=True)
    p.add_argument("--ready", type=Path)
    p.add_argument("--release", type=Path)
    p.add_argument("--round", type=int)
    p.add_argument("--all", action="store_true")
    p.add_argument("--seal-only", action="store_true")
    p.add_argument("--source-preparation", action="store_true")
    p.add_argument("--qualify-authorized-only", action="store_true")
    args = p.parse_args()
    if args.source_preparation and args.qualify_authorized_only:
        raise SystemExit("choose one E3 preparation or authorized qualification mode")
    if (args.source_preparation or args.qualify_authorized_only) and (args.all or args.round is not None or args.seal_only):
        raise SystemExit("E3 zero-task modes cannot include a formal round or seal")
    if not args.source_preparation and (args.ready is None or args.release is None):
        raise SystemExit("authorized qualification/formal execution requires READY and parent release")

    repo = args.repo.resolve()
    authority_path = args.authority.resolve()
    ready_path = args.ready.resolve() if args.ready else None
    release_path = args.release.resolve() if args.release else None
    if args.source_preparation or args.qualify_authorized_only:
        arm = load(authority_path)
        ref = arm.get("aggregate_authority") or {}
        authority_repo = Path(os.environ.get("OPENEVO_POST_ADVISOR_AUTHORITY_REPO", "")).resolve()
        aggregate_root = authority_repo if args.qualify_authorized_only else repo
        aggregate_path = aggregate_root / str(ref.get("path") or "")
        if not aggregate_path.is_file() or sha(aggregate_path) != ref.get("sha256"):
            raise RuntimeError("E3 aggregate authority drift")
        result = verify_preparation(repo, aggregate_path)
        rank = int(arm.get("persistent_state_rank", -1))
        aggregate_arm = load(aggregate_path).get("arms", {}).get(str(rank)) or {}
        expected_replay = aggregate_arm.get("staged_replay_path") or str(Path(aggregate_arm.get("initial_adapter_path", "")) / "openevo_sd_lora_replay.jsonl")
        if (rank not in (8, 16, 64) or arm.get("initial_adapter_path") != aggregate_arm.get("initial_adapter_path")
                or arm.get("initial_adapter_sha256") != aggregate_arm.get("adapter_model.safetensors_sha256")
                or arm.get("initial_replay_path") != expected_replay
                or arm.get("initial_replay_sha256") != aggregate_arm.get("openevo_sd_lora_replay.jsonl_sha256")
                or arm.get("output_root") != aggregate_arm.get("output_root")
                or arm.get("status") != ("AUTHORIZED" if args.qualify_authorized_only else "PREPARED_NOT_ACTIVE")):
            raise RuntimeError("E3 arm preparation drift")
        if result["status"] != "PASS_SOURCE_PREPARATION" or result["trainer_owned_replay_pending_ranks"]:
            raise RuntimeError("E3 trainer-owned source preparation incomplete")
        if args.source_preparation:
            print(json.dumps({**result, "prepared_rank": rank}, sort_keys=True))
            return 0
        expected = authority_repo / f"configs/experiment/activations/post-advisor-e3-rank{rank}-authority-202609300248.json"
        if authority_path != expected.resolve():
            raise RuntimeError("E3 authorized arm must come from B checkout")
        if (not authority_repo.is_dir() or git_head(authority_repo) == git_head(repo)
                or subprocess.run(["git", "-C", str(authority_repo), "status", "--porcelain", "--untracked-files=all"],
                                  check=True, capture_output=True, text=True).stdout.strip()):
            raise RuntimeError("E3 qualification requires clean committed B authority")
        if (git_head(repo) == BASE_C
                or subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
                                  check=True, capture_output=True, text=True).stdout.strip()):
            raise RuntimeError("E3 qualification requires clean exact execution A")
        ancestor = subprocess.run(["git", "-C", str(authority_repo), "merge-base", git_head(repo), git_head(authority_repo)],
                                  check=True, capture_output=True, text=True).stdout.strip()
        if ancestor != git_head(repo):
            raise RuntimeError("E3 qualification B does not descend from execution A")
        revisions = subprocess.run(["git", "-C", str(authority_repo), "rev-list", "--reverse",
                                    f"{git_head(repo)}..{git_head(authority_repo)}"],
                                   check=True, capture_output=True, text=True).stdout.splitlines()
        if not revisions:
            raise RuntimeError("E3 qualification requires a distinct B authority commit")
        previous = git_head(repo)
        for revision in revisions:
            lineage = subprocess.run(["git", "-C", str(authority_repo), "rev-list", "--parents", "-n", "1", revision],
                                     check=True, capture_output=True, text=True).stdout.split()
            changed = subprocess.run(["git", "-C", str(authority_repo), "diff", "--name-only", previous, revision],
                                     check=True, capture_output=True, text=True).stdout.splitlines()
            if lineage != [revision, previous] or not changed or any(not e3_b_material(name) for name in changed):
                raise RuntimeError("E3 qualification B is not a linear authority-only descendant")
            previous = revision
        aggregate = load(aggregate_path)
        if aggregate_path != authority_repo / E3_AUTHORITY:
            raise RuntimeError("E3 B aggregate authority path drift")
        for path in (aggregate_path, authority_path):
            relative = path.relative_to(authority_repo)
            blob = subprocess.run(["git", "-C", str(authority_repo), "show", f"HEAD:{relative}"],
                                  check=True, capture_output=True).stdout
            if path.read_bytes() != blob:
                raise RuntimeError("E3 qualification requires committed B authority bytes")
        if aggregate.get("execution_git_sha") != git_head(repo) or set(aggregate.get("source_files") or {}) != set(SOURCE_FILES):
            raise RuntimeError("E3 B authority does not bind exact execution source map")
        for key, relative_source in SOURCE_FILES.items():
            source = repo / relative_source
            committed_source = subprocess.run(["git", "-C", str(repo), "show", f"HEAD:{relative_source}"],
                                              check=True, capture_output=True).stdout
            if source.read_bytes() != committed_source or sha(source) != aggregate["source_files"][key]:
                raise RuntimeError(f"E3 B qualification source drift: {key}")
        release_relative = E3_RELEASE if rank == 8 else E3_RELEASE_OTHERS.format(rank)
        release_file = authority_repo / release_relative
        release = load(release_file)
        if (release.get("status") != "launch-authorized" or release.get("formal_task_consumption_allowed") is not True
                or release.get("rank") != rank or release.get("formal_rollout_limit") != 1024
                or release.get("authority") != {"path": E3_AUTHORITY, "sha256": sha(aggregate_path)}):
            raise RuntimeError("E3 B release is not rank-authorized")
        for path in (release_file,):
            relative = path.relative_to(authority_repo)
            blob = subprocess.run(["git", "-C", str(authority_repo), "show", f"HEAD:{relative}"],
                                  check=True, capture_output=True).stdout
            if path.read_bytes() != blob:
                raise RuntimeError("E3 B release bytes are uncommitted")
        router_relative = E3_ROUTER.format(rank)
        router_file = authority_repo / router_relative
        router_blob = subprocess.run(["git", "-C", str(authority_repo), "show", f"HEAD:{router_relative}"],
                                     check=True, capture_output=True).stdout
        if not router_file.is_file() or router_file.read_bytes() != router_blob:
            raise RuntimeError("E3 B rank router bytes are uncommitted")
        router = load(router_file)
        if (router.get("rank") != rank or router.get("formal_rollout_limit") != 1024
                or router.get("campaign_id") != aggregate.get("campaign_id")
                or router.get("status") != "PREPARED_NOT_ACTIVE"
                or router.get("formal_task_consumption_allowed") is not False
                or router.get("final_panel_access_allowed") is not False
                or router.get("release_receipt") != release_relative
                or router.get("release_receipt_sha256") != sha(release_file)
                or router.get("release") != {"path": release_relative, "sha256": sha(release_file)}
                or router.get("experiment_identity") != release.get("experiment_identity")):
            raise RuntimeError("E3 B rank router/release/Passport identity drift")
        refs = {
            "arm": (release.get("arm_authorities") or {}).get(str(rank)) or {},
            "profile": (release.get("profiles") or {}).get(str(rank)) or {},
            "preregistration": release.get("preregistration") or {},
            "passport": release.get("passport") or {},
            "path_contract": release.get("path_contract") or {},
        }
        docs = {}
        for kind, ref_doc in refs.items():
            relative = str(ref_doc.get("path") or "")
            if not relative or relative.startswith("/") or ".." in Path(relative).parts:
                raise RuntimeError(f"E3 B {kind} reference path drift")
            path = authority_repo / relative
            blob = subprocess.run(["git", "-C", str(authority_repo), "show", f"HEAD:{relative}"],
                                  check=True, capture_output=True).stdout
            if not path.is_file() or path.read_bytes() != blob or sha(path) != ref_doc.get("sha256"):
                raise RuntimeError(f"E3 B {kind} reference bytes drift")
            docs[kind] = load(path)
        if (refs["arm"]["path"] != str(authority_path.relative_to(authority_repo))
                or refs["preregistration"]["path"] != "configs/experiment/preregistrations/post-advisor-e3-rank-capacity-preregistration-202609300248.json"
                or refs["path_contract"]["path"] != "configs/experiment/path-contracts/post-advisor-e3-rank-capacity-path-contract-202609300248.json"
                or release.get("execution_readiness", {}).get("profile") != refs["profile"]
                or docs["profile"].get("source", {}).get("experiment", {}).get("revision") != git_head(repo)
                or not docs["preregistration"].get("frozen_scientific_contract")
                or docs["passport"].get("status") != "FROZEN"
                or release.get("passport", {}).get("status") != "FROZEN"
                or release.get("experiment_identity") != {
                    "experiment_id": aggregate["experiment_id"], "passport_sha256": docs["passport"].get("passport_payload_sha256")}):
            raise RuntimeError("E3 B arm/profile/science/Passport binding drift")
        from scripts.openevo_webshop import experiment_passport_gate
        if experiment_passport_gate.validate_binding(authority_repo, release["experiment_identity"]).get("passport_status") != "FROZEN":
            raise RuntimeError("E3 B Passport/Registry binding incomplete")
        relative = authority_path.relative_to(authority_repo)
        committed_arm = subprocess.run(["git", "-C", str(authority_repo), "show", f"HEAD:{relative}"],
                                       check=True, capture_output=True).stdout
        if (committed_arm != authority_path.read_bytes()
                or arm.get("execution_git_sha") != git_head(repo)
                or arm.get("managed_gpu_indices") != [0, 1, 2, 3]
                or arm.get("formal_rollout_limit") != 1024):
            raise RuntimeError("E3 B-qualified arm identity drift")
        validate(repo, authority_path, ready_path, release_path)
        if os.geteuid() != 0:
            raise RuntimeError("E3 qualification receipt requires root to prepare UID1001 run root")
        output = Path(arm["output_root"])
        if not output.exists():
            output.mkdir(mode=0o750, parents=True)
            os.chown(output, 1001, 1001)
        if output.stat().st_uid != 1001 or (output / "execution-state.json").exists() or list(output.glob("round-*")):
            raise RuntimeError("E3 qualification requires unused UID1001 arm output")
        preflight = output / "preflight"
        if not preflight.exists():
            preflight.mkdir(mode=0o750)
            os.chown(preflight, 1001, 1001)
        if preflight.stat().st_uid != 1001:
            raise RuntimeError("E3 preflight owner drift")
        receipt_path = preflight / "launcher-qualification.json"
        receipt = {
            "schema": "openevo.post-advisor.e3-rank-launcher-qualification.202609300248",
            "status": "PASS", "execution_sha": git_head(repo),
            "authority_git_sha": git_head(authority_repo),
            "authority_sha256": sha(authority_path),
            "aggregate_authority_sha256": sha(aggregate_path),
            "persistent_state_rank": rank, "treatment_mode": "BOUNDED_OFF",
            "formal_rollout_limit": 1024, "formal_task_consumed": False,
            "task_bearing_calls": 0, "model_calls": 0, "training_calls": 0,
            "final_panel_access": 0,
        }
        base.write_once_json(receipt_path, receipt)
        print(json.dumps({"receipt_path": str(receipt_path), "receipt_sha256": sha(receipt_path), **receipt}, sort_keys=True))
        return 0
    authority, _ = validate(repo, authority_path, ready_path, release_path)
    require_post_advisor_gpu_assignment(list(map(int, authority["managed_gpu_indices"])))
    validate_post_advisor_activation(repo, authority["execution_git_sha"], authority_path)
    seed(authority, authority_path, ready_path, release_path)
    base_cli = common_cli(repo, authority, authority_path, ready_path, release_path)
    if args.seal_only:
        subprocess.run([*base_cli, "--seal-only"], check=True)
        return 0
    if args.all and authority.get("carrier_health_resolution"):
        live = load(Path(authority["output_root"]) / "execution-state.json")
        next_round = int(live.get("next_round", -1))
        if next_round < 153 or next_round > 160 or live.get("rollouts_consumed") != next_round * 128:
            raise RuntimeError("A9 live round accounting drift")
        recovery.validate_post_advisor_recovery(authority)
        rounds = list(range(next_round, 160))
    else:
        rounds = ROUNDS if args.all else ([args.round] if args.round is not None else [])
    if not rounds:
        raise SystemExit("choose --all, --round, --seal-only, --source-preparation, or --qualify-authorized-only")
    for r in rounds:
        if r not in ROUNDS:
            raise SystemExit(f"round outside 152..159: {r}")
        subprocess.run([*base_cli, "--round-index", str(r)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
