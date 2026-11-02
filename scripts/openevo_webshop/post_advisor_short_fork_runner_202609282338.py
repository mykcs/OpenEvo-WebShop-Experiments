#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.openevo_webshop import ceiling1_stage2_arm_controller as base
from scripts.openevo_webshop import bounded_effective_state_recovery_202609282338 as recovery
from scripts.openevo_webshop.bounded_effective_state_launch_guard_202609282338 import validate_post_advisor_activation
from scripts.openevo_webshop.bounded_effective_state_runtime_safety_202609282338 import GPU_UUID, require_post_advisor_gpu_assignment, exclusive_arm_lock

R151_STATE = Path("/workspace/runs/bounded-effective-state-gdr-recovery-1e7c4952-20260916/off/round-151/state-after.json")
R151_HEALTH = Path("/workspace/runs/bounded-effective-state-gdr-recovery-1e7c4952-20260916/off/round-151/carrier-health-boundary.json")
ROUNDS = list(range(152, 160))
READY_SHA = "5f9297eaa21c0dd0e385cc5731ec090476cf09464239e58f3c41e8680bad640c"
RELEASE_SHA = "2ea603e403641cec3494aa12bde869130efcca946657b54c6c5a97bf9cbd6caa"
OFF_MODE = "BOUNDED_OFF"
MAG_MODE = "MAGNITUDE_SCALAR_GATE_" + "V" + "1"  # legacy-revision-compat: exact historical frozen identifier
SUPPORTED_PERSISTENT_STATE_RANKS = (8, 16, 64, 128)


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
    if authority.get("schema") != "openevo.post-advisor.short-fork-authority.202609282338":
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
    if treatment not in (OFF_MODE, MAG_MODE):
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
    recovery.validate_post_advisor_recovery(authority)
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
        str(repo / "scripts/openevo_webshop/post_advisor_bounded_short_fork_202609282338.py"),
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
        "--formal-transition-launcher", str(repo / "scripts/openevo_webshop/launch_post_advisor_magnitude_transition_20260930.sh") if authority["treatment_mode"] == "MAGNITUDE_SCALAR_GATE_V1" else str(repo / transition["launcher_path"]),
        "--empty-history-manifest", str(repo / transition["empty_history_manifest"]),
        "--adapter-config-template", str(repo / transition["adapter_config_template"]),
        "--post-advisor-short-fork-authority", str(authority_path),
        *(["--carrier-health-resolution", authority["terminal_carrier_health_resolution"]["path"]]
          if authority.get("terminal_carrier_health_resolution") else
          (["--carrier-health-resolution", authority["carrier_health_resolution"]["path"]]
           if authority.get("carrier_health_resolution") else [])),
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--authority", type=Path, required=True)
    p.add_argument("--ready", type=Path, required=True)
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--round", type=int)
    p.add_argument("--all", action="store_true")
    p.add_argument("--seal-only", action="store_true")
    p.add_argument("--qualify-only", action="store_true")
    args = p.parse_args()

    repo = args.repo.resolve()
    authority_path = args.authority.resolve()
    ready_path = args.ready.resolve()
    release_path = args.release.resolve()
    authority, _ = validate(repo, authority_path, ready_path, release_path)
    if args.qualify_only and authority.get("engineering_recovery"):
        doc = recovery.validate_post_advisor_recovery(authority)
        if doc["schema"] == recovery.POST_ADVISOR_A11_RECOVERY_SCHEMA:
            print(json.dumps({"status": "ZERO_FORMAL_QUALIFICATION_PASS", "next_round": 160,
                              "e1_rollouts_consumed": 1024, "remaining_e1_rollouts": 0,
                              "formal_task_consumed": False, "formal_rollout_replay": 0,
                              "final_panel_access": 0}, sort_keys=True))
            return 0
        result = {
            "schema": "openevo.post-advisor.short-fork-qualification.202609282338",
            "status": "PASS",
            "execution_sha": authority["execution_git_sha"],
            "authority_sha256": sha(authority_path),
            "persistent_state_rank": authority["persistent_state_rank"],
            "treatment_mode": authority["treatment_mode"],
            "formal_task_consumed": False,
            "previous_formal_rollouts_consumed": 128,
            "remaining_formal_rollouts": 896,
            "formal_rollout_replay": 0,
            "final_panel_access": 0,
            "recovery_authority_sha256": sha(Path(authority["engineering_recovery"]["path"])),
        }
        receipt_path = Path(doc["qualification_receipt_path"])
        if receipt_path.exists() and load(receipt_path) != result:
            raise RuntimeError("post-advisor recovery qualification receipt drift")
        base.write_once_json(receipt_path, result)
        print(json.dumps({"status": "ZERO_FORMAL_QUALIFICATION_PASS", **result}, sort_keys=True))
        return 0
    a11_seal = authority.get("terminal_carrier_health_resolution") is not None
    if a11_seal and not (args.qualify_only or args.seal_only):
        raise RuntimeError("A11 authorizes seal-only; formal task modes are closed")
    if args.seal_only and a11_seal:
        from scripts.openevo_webshop.bounded_effective_state_launch_guard_202609282338 import validate_post_advisor_terminal_seal_activation
        validate_post_advisor_terminal_seal_activation(repo, authority["execution_git_sha"], authority_path)
    if not args.qualify_only and not a11_seal:
        require_post_advisor_gpu_assignment(list(map(int, authority["managed_gpu_indices"])))
        validate_post_advisor_activation(repo, authority["execution_git_sha"], authority_path)
        if authority.get("engineering_recovery"):
            with exclusive_arm_lock(Path(authority["output_root"]).resolve()):
                recovery.prepare_post_advisor_recovery(authority)
    seed(authority, authority_path, ready_path, release_path)
    base_cli = common_cli(repo, authority, authority_path, ready_path, release_path)

    if args.qualify_only:
        result = subprocess.run([*base_cli, "--qualify-only"], check=True, text=True, capture_output=True)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        if "ZERO_FORMAL_QUALIFICATION_PASS" not in result.stdout:
            raise RuntimeError("short-fork launcher qualification did not report PASS")
        receipt = {
            "schema": "openevo.post-advisor.short-fork-qualification.202609282338",
            "status": "PASS",
            "execution_sha": authority["execution_git_sha"],
            "authority_sha256": sha(authority_path),
            "persistent_state_rank": authority["persistent_state_rank"],
            "treatment_mode": authority["treatment_mode"],
            "formal_task_consumed": False,
            "final_panel_access": 0,
        }
        receipt_path = Path(authority["output_root"]).resolve() / "qualification-receipt.json"
        if receipt_path.exists() and load(receipt_path) != receipt:
            raise RuntimeError("short-fork qualification receipt drift")
        base.atomic_json(receipt_path, receipt)
        return 0
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
        raise SystemExit("choose --all, --round, --seal-only, or --qualify-only")
    for r in rounds:
        if r not in ROUNDS:
            raise SystemExit(f"round outside 152..159: {r}")
        subprocess.run([*base_cli, "--round-index", str(r)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
