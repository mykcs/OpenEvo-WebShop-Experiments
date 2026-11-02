#!/usr/bin/env python3
"""Read-only E3 rank-capacity source and activation gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

RANKS = (8, 16, 64)
ROUNDS = list(range(152, 160))
SOURCE_STATE_ID = "afb7ff3a3e91536576d2661091c0af04411c834b33c9cb7dc4fbaa1eac1bd3e6"
SOURCE_ADAPTER_SHA = "812e64cff1efc55d3eb7e293876c0ab8658734b42486898306e3a338335d784a"
SOURCE_STATE_SHA = "4303d03d3e58ec25807ed835d1c03b3554a8fb92f3f19d7cd00855b54088aba7"
SOURCE_HEALTH_SHA = "e3aba5ad5136d080cd8168cecd21b0b3cc0ab927daab414cccc447c79ee2ecc9"
REPLAY_SHA = "96229160341bb5527f1991ad7d510600e80cc03b041a2d48e4006ce7d47f9abb"
BASE_C = "7c79c8906e37486c2125d466b3c53ebe92ab987a"
DESIGN = "configs/experiment/designs/post-advisor-unanswered-202609282338.json"
E1_ACTIVATION = "configs/experiment/activations/post-advisor-unanswered-202609282338.json"
E1_PREREG = "configs/experiment/preregistrations/post-advisor-e1-beta-direction-preregistration-202609282338.json"
E1_PASSPORT = "configs/experiment/passports/202609282338-post-advisor-e1-beta-direction/passport.json"
E1_CAMPAIGN = "configs/experiment/current-campaign.json"
E3_AUTHORITY = "configs/experiment/activations/post-advisor-e3-rank-capacity-authority-202609300248.json"
E3_PREREG = "configs/experiment/preregistrations/post-advisor-e3-rank-capacity-preregistration-202609300248.json"
E3_RELEASE = "configs/experiment/activations/post-advisor-e3-rank-capacity-release-202609300248.json"
E3_RELEASE_OTHERS = "configs/experiment/activations/post-advisor-e3-rank{}-release-202609300248.json"
E3_ROUTER = "configs/experiment/activations/post-advisor-e3-rank{}-current-campaign-202609300248.json"
E3_ACTIVATION = "configs/experiment/activations/post-advisor-e3-rank-capacity-activation-202609300248.json"
E3_PROFILE = "configs/experiment/profiles/post-advisor-e3-rank{}-profile-202609300248.json"
E3_ARM = "configs/experiment/activations/post-advisor-e3-rank{}-authority-202609300248.json"
SOURCE_FILES = {
    "guard_sha256": "scripts/openevo_webshop/post_advisor_e3_guard_202609300248.py",
    "runner_sha256": "scripts/openevo_webshop/post_advisor_e3_rank_capacity_runner_202609300248.py",
    "controller_sha256": "scripts/openevo_webshop/post_advisor_e3_bounded_short_fork_202609300248.py",
    "runtime_safety_sha256": "scripts/openevo_webshop/post_advisor_e3_runtime_safety_202609300248.py",
    "transition_driver_sha256": "scripts/openevo_webshop/post_advisor_e3_rank_capacity_transition_driver_202609300248.py",
    "shell_sha256": "scripts/openevo_webshop/launch_post_advisor_e3_rank_capacity_transition_202609300248.sh",
    "staging_sha256": "scripts/openevo_webshop/post_advisor_e3_stage_replay_202609300248.py",
    "prelaunch_qualifier_sha256": "scripts/openevo_webshop/post_advisor_e3_prelaunch_qualifier_202609300248.py",
}
HISTORICAL_ROOT = Path("/workspace/runs/bounded-effective-state-gdr-recovery-1e7c4952-20260916/off/round-151")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def e3_b_material(name: str) -> bool:
    return (name.startswith("configs/experiment/activations/post-advisor-e3-")
            and name != E3_ACTIVATION
            or name.startswith("configs/experiment/profiles/post-advisor-e3-")
            or name.startswith("configs/experiment/passports/202609300248-post-advisor-e3-rank-capacity/")
            or name.startswith("configs/experiment/path-contracts/post-advisor-e3-")
            or name == E3_PREREG
            or (name.startswith("configs/experiment/activations/post-advisor-e3-rank") and "-current-campaign-" in name)
            or name == "configs/experiment/current-experiment-registry.json"
            or name.startswith("configs/experiment/experiment-registry-20260930")
            or name.startswith("docs/evidence/post-advisor-unanswered-20260928/E3_"))


def committed(repo: Path, relative: str) -> Path:
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise RuntimeError("E3 authority path outside repository")
    path = repo / relative
    blob = subprocess.run(["git", "-C", str(repo), "show", f"HEAD:{relative}"],
                          check=True, capture_output=True).stdout
    if not path.is_file() or path.read_bytes() != blob:
        raise RuntimeError(f"E3 committed authority bytes drift: {relative}")
    return path


def checked_ref(repo: Path, ref: dict) -> tuple[Path, dict]:
    path = committed(repo, str(ref.get("path") or ""))
    if sha(path) != ref.get("sha256"):
        raise RuntimeError("E3 authority reference SHA drift")
    return path, load(path)


def checked_external(ref: dict) -> dict:
    path = Path(str(ref.get("path") or ""))
    if not path.is_absolute() or not path.is_file() or sha(path) != ref.get("sha256"):
        raise RuntimeError("E3 external receipt identity drift")
    return load(path)


def verify_preparation(repo: Path, authority_path: Path) -> dict:
    """Verify exact frozen E3 inputs without creating an output directory."""
    repo = repo.resolve()
    authority = load(authority_path.resolve())
    if authority.get("schema") != "openevo.post-advisor.e3-rank-capacity-authority.202609300248":
        raise RuntimeError("E3 authority schema drift")
    if authority.get("status") not in ("PREPARED_NOT_ACTIVE", "AUTHORIZED"):
        raise RuntimeError("E3 authority status drift")
    if authority.get("source_e1_c_sha") != BASE_C:
        raise RuntimeError("E3 source C identity drift")
    if sha(repo / DESIGN) != authority.get("design_sha256"):
        raise RuntimeError("E3 frozen design drift")
    design = load(repo / DESIGN)
    cells = [item for item in design.get("experiments", []) if item.get("id") == "E3_RANK_CAPACITY_SWEEP"]
    if len(cells) != 1:
        raise RuntimeError("E3 frozen design cell missing")
    cell = cells[0]
    if [int(item.get("rank", -1)) for item in cell.get("new_arms", [])] != list(RANKS):
        raise RuntimeError("E3 design rank order drift")
    if any(item.get("rounds") != "R152-R159" or item.get("formal_rollouts") != 1024
           for item in cell["new_arms"]):
        raise RuntimeError("E3 design budget drift")
    state_path = HISTORICAL_ROOT / "state-after.json"
    health_path = HISTORICAL_ROOT / "carrier-health-boundary.json"
    adapter_path = HISTORICAL_ROOT / "sd-lora/bounded-state/adapter_model.safetensors"
    if (sha(state_path), sha(health_path), sha(adapter_path)) != (
            SOURCE_STATE_SHA, SOURCE_HEALTH_SHA, SOURCE_ADAPTER_SHA):
        raise RuntimeError("E3 common R151 source bytes drift")
    state = load(state_path)
    health = load(health_path)
    if (state.get("state_id") != SOURCE_STATE_ID or state.get("next_round") != 152
            or state.get("rollouts_consumed") != 152 * 128
            or health.get("action") != "CONTINUE" or health.get("round_index") != 151):
        raise RuntimeError("E3 common R151 state/health boundary drift")
    if authority.get("rounds") != ROUNDS or authority.get("final_panel_access_limit") != 0:
        raise RuntimeError("E3 round/final boundary drift")
    if authority.get("historical_rank128_control_rerun") is not False:
        raise RuntimeError("E3 historical rank128 rerun boundary drift")
    arms = authority.get("arms") or {}
    if sorted(map(int, arms)) != list(RANKS):
        raise RuntimeError("E3 arms must be exactly rank8/16/64")
    outputs: set[str] = set()
    ownership_ready = []
    for rank in RANKS:
        arm = arms[str(rank)]
        if (arm.get("rank") != rank or arm.get("rounds") != ROUNDS
                or arm.get("formal_rollout_limit") != 1024
                or arm.get("treatment_mode") != "BOUNDED_OFF"
                or arm.get("source_state_id") != SOURCE_STATE_ID
                or arm.get("source_adapter_sha256") != SOURCE_ADAPTER_SHA):
            raise RuntimeError(f"E3 rank{rank} arm scientific identity drift")
        output = str(Path(arm["output_root"]).resolve())
        if output in outputs:
            raise RuntimeError("E3 output roots collide")
        outputs.add(output)
        initial = Path(arm["initial_adapter_path"]).resolve()
        receipt = initial / "compression-receipt.json"
        if sha(receipt) != arm.get("compression_receipt_sha256"):
            raise RuntimeError(f"E3 rank{rank} compression receipt drift")
        payload = load(receipt)
        if (payload.get("status") != "PASS" or payload.get("source_state_id") != SOURCE_STATE_ID
                or payload.get("source_adapter_sha256") != SOURCE_ADAPTER_SHA
                or payload.get("target_rank") != rank
                or payload.get("formal_rollouts_consumed") != 0
                or payload.get("final_panel_access") != 0):
            raise RuntimeError(f"E3 rank{rank} compression semantics drift")
        for name, key in (("adapter_model.safetensors", "output_adapter_sha256"),
                          ("adapter_config.json", "output_config_sha256"),
                          ("openevo_sd_lora_replay.jsonl", "output_replay_sha256")):
            path = initial / name
            expected = payload[key]
            if sha(path) != expected or arm.get(name + "_sha256") != expected:
                raise RuntimeError(f"E3 rank{rank} {name} bytes drift")
        if payload["output_replay_sha256"] != REPLAY_SHA:
            raise RuntimeError(f"E3 rank{rank} inherited replay drift")
        if int(load(initial / "adapter_config.json").get("r", -1)) != rank:
            raise RuntimeError(f"E3 rank{rank} adapter config drift")
        replay = initial / "openevo_sd_lora_replay.jsonl"
        stat = replay.stat()
        if rank in (16, 64):
            staged = Path(str(arm.get("staged_replay_path") or ""))
            receipt_path = Path(str(arm.get("staged_replay_receipt_path") or ""))
            expected_root = Path(output) / "preflight"
            if (staged != expected_root / "prior-replay.jsonl"
                    or receipt_path != expected_root / "prior-replay-materialization.json"
                    or not staged.is_file() or not receipt_path.is_file()
                    or sha(receipt_path) != arm.get("staged_replay_receipt_sha256")):
                ownership_ready.append(None)
                continue
            stage = load(receipt_path)
            st = staged.stat()
            if (stage.get("status") != "PASS_ZERO_TASK" or stage.get("rank") != rank
                    or stage.get("source_path") != str(replay)
                    or stage.get("source_sha256") != REPLAY_SHA
                    or stage.get("source_uid") != stat.st_uid
                    or stage.get("source_nlink") != stat.st_nlink
                    or stage.get("source_inode_preserved") is not True
                    or stage.get("target_path") != str(staged)
                    or stage.get("target_sha256") != REPLAY_SHA
                    or stage.get("target_uid") != 1001 or stage.get("target_nlink") != 1
                    or stage.get("formal_rollouts_consumed") != 0
                    or stage.get("final_panel_access") != 0
                    or st.st_uid != 1001 or st.st_nlink != 1 or sha(staged) != REPLAY_SHA):
                raise RuntimeError(f"E3 rank{rank} staged replay receipt drift")
            ownership_ready.append(rank)
        else:
            ownership_ready.append(rank if stat.st_uid == 1001 and stat.st_nlink == 1 else None)
    baseline_ref = authority.get("reusable_baseline") or {}
    baseline = repo / str(baseline_ref.get("path") or "")
    if not baseline.is_file() or sha(baseline) != baseline_ref.get("sha256"):
        raise RuntimeError("E3 rank32/rank128 reusable evidence drift")
    baselines = load(baseline)
    if (baselines.get("status") != "PASS_IDENTITIES_RESOLVED"
            or baselines.get("rank128", {}).get("R151", {}).get("state_id") != SOURCE_STATE_ID
            or not baselines.get("rank128", {}).get("R159")
            or not baselines.get("rank32", {}).get("R159")):
        raise RuntimeError("E3 reusable baseline identity incomplete")
    if authority.get("total_new_rollout_limit") != 3072:
        raise RuntimeError("E3 total rollout budget drift")
    pending = [r for r in RANKS if r not in ownership_ready]
    return {"status": "SOURCE_PREPARED_RUNTIME_BLOCKED" if pending else "PASS_SOURCE_PREPARATION", "ranks": list(RANKS),
            "rounds": ROUNDS, "new_rollout_budget": 3072,
            "trainer_owned_replay_ready_ranks": [x for x in ownership_ready if x is not None],
            "trainer_owned_replay_pending_ranks": pending,
            "formal_task_consumed": False, "final_panel_access": 0}


E3_FORMAL_START_RUNS_ROOT = Path("/workspace/runs/post-advisor-unanswered-20260928")


def validate_e3_formal_start_receipt_location(
    start_path: Path, *, runs_root: Path = E3_FORMAL_START_RUNS_ROOT
) -> None:
    """Mirror the transition consumer path boundary before any E3 task work."""
    candidate = Path(start_path)
    root = Path(runs_root)
    if not candidate.is_absolute() or not root.is_absolute():
        raise RuntimeError("E3 Formal Start path/root must be absolute")
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("E3 Formal Start receipt is outside the transition-visible campaign runs root") from exc
    if not relative.parts:
        raise RuntimeError("E3 Formal Start receipt must be below the campaign runs root")


def verify_formal_activation(repo: Path, authority_path: Path) -> dict:
    preparation = verify_preparation(repo, authority_path)
    authority = load(authority_path)
    if (authority.get("status") != "AUTHORIZED" or preparation["status"] != "PASS_SOURCE_PREPARATION"
            or preparation["trainer_owned_replay_pending_ranks"]):
        raise RuntimeError("E3 formal activation requires complete trainer-owned sources")
    repo = repo.resolve()
    a_sha = git(repo, "rev-parse", "HEAD")
    if git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("E3 execution A is dirty")
    if authority.get("execution_git_sha") != a_sha or a_sha == BASE_C:
        raise RuntimeError("E3 exact execution A revision is unbound")
    if git(repo, "merge-base", BASE_C, a_sha) != BASE_C:
        raise RuntimeError("E3 execution A does not descend from E1 C")
    if set(authority.get("source_files") or {}) != set(SOURCE_FILES):
        raise RuntimeError("E3 source SHA map incomplete")
    for key, relative in SOURCE_FILES.items():
        if sha(committed(repo, relative)) != authority["source_files"][key]:
            raise RuntimeError(f"E3 execution source drift: {key}")
    authority_repo = Path(os.environ.get("OPENEVO_POST_ADVISOR_AUTHORITY_REPO", "")).resolve()
    if not authority_repo.is_dir() or git(authority_repo, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("E3 authority C unavailable or dirty")
    c_sha = git(authority_repo, "rev-parse", "HEAD")
    ancestry = git(authority_repo, "rev-list", "--parents", "-n", "1", c_sha).split()
    if len(ancestry) != 2:
        raise RuntimeError("E3 activation C must have one release B parent")
    b_sha = ancestry[1]
    if git(authority_repo, "merge-base", a_sha, b_sha) != a_sha or b_sha == a_sha:
        raise RuntimeError("E3 release B must descend from exact execution A")
    commits = git(authority_repo, "rev-list", "--reverse", f"{a_sha}..{b_sha}").splitlines()
    previous = a_sha
    for revision in commits:
        if git(authority_repo, "rev-list", "--parents", "-n", "1", revision).split() != [revision, previous]:
            raise RuntimeError("E3 A to B must be linear")
        changed = set(git(authority_repo, "diff", "--name-only", previous, revision).splitlines())
        if not changed or any(not e3_b_material(name) for name in changed):
            raise RuntimeError("E3 B contains source changes")
        previous = revision
    expected_c_changes = {E3_ACTIVATION, *(E3_ROUTER.format(rank) for rank in RANKS)}
    c_revisions = git(authority_repo, "rev-list", "--reverse", f"{b_sha}..{c_sha}").splitlines()
    if len(c_revisions) != 3:
        raise RuntimeError("E3 C lineage must contain inactive C2, evidence-complete C3, and reviewed activation C4")
    previous_c = b_sha
    expected_per_commit = [{E3_ACTIVATION}, {E3_ACTIVATION}, expected_c_changes]
    for revision, expected_files in zip(c_revisions, expected_per_commit):
        if git(authority_repo, "rev-list", "--parents", "-n", "1", revision).split() != [revision, previous_c]:
            raise RuntimeError("E3 C2/C3/C4 lineage is not linear")
        changed = set(git(authority_repo, "diff", "--name-only", previous_c, revision).splitlines())
        if changed != expected_files:
            raise RuntimeError("E3 C2/C3/C4 commit contents drift")
        previous_c = revision
    c2_sha = c_revisions[0]
    c2_bytes = subprocess.run(["git", "-C", str(authority_repo), "show", f"{c2_sha}:{E3_ACTIVATION}"], check=True, capture_output=True).stdout
    c2_activation_sha = hashlib.sha256(c2_bytes).hexdigest()
    c2_doc = json.loads(c2_bytes)
    if (c2_doc.get("status") != "PREPARED_NOT_ACTIVE" or c2_doc.get("execution_sha") != a_sha
            or c2_doc.get("authority_git_sha") != b_sha or c2_doc.get("formal_task_consumption_allowed") is not False
            or c2_doc.get("final_panel_access_allowed") is not False):
        raise RuntimeError("E3 C2 is not the immutable inactive scope")
    c3_bytes = subprocess.run(["git", "-C", str(authority_repo), "show", f"{c_revisions[1]}:{E3_ACTIVATION}"], check=True, capture_output=True).stdout
    c3_doc = json.loads(c3_bytes)
    if (c3_doc.get("status") != "PREPARED_NOT_ACTIVE" or c3_doc.get("pending_evidence")
            or c3_doc.get("formal_task_consumption_allowed") is not False
            or c3_doc.get("final_panel_access_allowed") is not False):
        raise RuntimeError("E3 C3 must contain complete receipts and remain inactive")
    if set(git(authority_repo, "diff", "--name-only", b_sha, c_sha).splitlines()) != expected_c_changes:
        raise RuntimeError("E3 C must change only activation and three rank routers")
    for immutable in (DESIGN, E1_ACTIVATION, E1_PREREG, E1_PASSPORT, E1_CAMPAIGN):
        if git(authority_repo, "show", f"{BASE_C}:{immutable}") != git(authority_repo, "show", f"{c_sha}:{immutable}"):
            raise RuntimeError(f"E3 C changed E1 frozen bytes: {immutable}")
    if authority_path.resolve() != (authority_repo / E3_AUTHORITY).resolve():
        raise RuntimeError("E3 formal authority must come from canonical C checkout")
    committed(authority_repo, E3_AUTHORITY)
    active_arm_path = Path(os.environ.get("OPENEVO_E3_ARM_AUTHORITY", "")).resolve()
    active_rank = next((rank for rank in RANKS if active_arm_path == (authority_repo / E3_ARM.format(rank)).resolve()), None)
    if active_rank is None:
        raise RuntimeError("E3 active arm path is outside C authority")
    active_release_rel = E3_RELEASE if active_rank == 8 else E3_RELEASE_OTHERS.format(active_rank)
    release_path = committed(authority_repo, active_release_rel)
    release = load(release_path)
    activation = load(committed(authority_repo, E3_ACTIVATION))
    if c3_doc.get("rank_evidence") != activation.get("rank_evidence"):
        raise RuntimeError("E3 C4 activation may not add or change evidence after reviewed inactive C3")
    if c3_doc.get("rank_releases") != c2_doc.get("rank_releases") or activation.get("rank_releases") != c3_doc.get("rank_releases"):
        raise RuntimeError("E3 C3/C4 may not add or change rank-release evidence after immutable C2")
    if set(c2_doc.get("rank_releases") or {}) != {str(r) for r in RANKS}:
        raise RuntimeError("E3 C2 rank-release refs incomplete")
    for r in RANKS:
        rel = E3_RELEASE if r == 8 else E3_RELEASE_OTHERS.format(r)
        if c2_doc["rank_releases"].get(str(r)) != {"path": rel, "sha256": sha(authority_repo / rel)}:
            raise RuntimeError(f"E3 C2 rank{r} release path/SHA differs from B")
    c2_evidence = c2_doc.get("rank_evidence") or {}
    if set(c2_evidence) != {str(r) for r in RANKS}:
        raise RuntimeError("E3 C2 qualification refs incomplete")
    for r in RANKS:
        qref = (c2_evidence[str(r)].get("qualification") or {})
        final_qref = (((activation.get("rank_evidence") or {}).get(str(r)) or {}).get("qualification") or {})
        if qref.get("path") != final_qref.get("path"):
            raise RuntimeError(f"E3 C2 rank{r} qualification path changed after scope freeze")
    release_refs = activation.get("rank_releases") or {}
    if set(release_refs) != {str(rank) for rank in RANKS}:
        raise RuntimeError("E3 C does not bind three rank releases")
    for rank in RANKS:
        expected_rel = E3_RELEASE if rank == 8 else E3_RELEASE_OTHERS.format(rank)
        path, doc = checked_ref(authority_repo, release_refs[str(rank)])
        if path != authority_repo / expected_rel or doc.get("rank") != rank or doc.get("formal_rollout_limit") != 1024:
            raise RuntimeError(f"E3 rank{rank} release scope drift")
    active_router_rel = E3_ROUTER.format(active_rank)
    active_router = load(committed(authority_repo, active_router_rel))
    if (Path(os.environ.get("OPENEVO_E3_CURRENT_CAMPAIGN", "")).resolve() != (authority_repo / active_router_rel).resolve()
            or active_router.get("campaign_id") != authority.get("campaign_id")
            or active_router.get("rank") != active_rank
            or active_router.get("status") != "launch-authorized"
            or active_router.get("formal_task_consumption_allowed") is not True
            or active_router.get("final_panel_access_allowed") is not False
            or active_router.get("release_receipt") != active_release_rel
            or active_router.get("release_receipt_sha256") != sha(release_path)
            or active_router.get("experiment_identity") != release.get("experiment_identity")):
        raise RuntimeError("E3 active rank router/release drift")
    if (not str(release.get("schema") or "").startswith("openevo.experiment-")
            or release.get("status") != "launch-authorized" or release.get("formal_task_consumption_allowed") is not True
            or activation.get("status") != "PASS" or activation.get("execution_sha") != a_sha
            or release_refs[str(active_rank)].get("path") != active_release_rel
            or release_refs[str(active_rank)].get("sha256") != sha(release_path)
            or activation.get("final_panel_access_allowed") is not False
            or activation.get("formal_task_consumption_allowed") is not True):
        raise RuntimeError("E3 activation scope drift")
    if release.get("authority", {}).get("path") != E3_AUTHORITY or release["authority"].get("sha256") != sha(authority_path):
        raise RuntimeError("E3 B authority binding drift")
    path_contract_path, _ = checked_ref(authority_repo, release.get("path_contract") or {})
    if path_contract_path.name != "post-advisor-e3-rank-capacity-path-contract-202609300248.json":
        raise RuntimeError("E3 B path contract drift")
    prereg_path, prereg = checked_ref(authority_repo, release.get("preregistration") or {})
    if (prereg_path.name != "post-advisor-e3-rank-capacity-preregistration-202609300248.json"
            or not prereg.get("frozen_scientific_contract")
            or prereg.get("persistent_state_ranks") != list(RANKS)
            or prereg.get("total_new_rollout_limit") != 3072):
        raise RuntimeError("E3 B frozen preregistration drift")
    passport_path, passport = checked_ref(authority_repo, release.get("passport") or {})
    if passport.get("status") != "FROZEN" or release["passport"].get("status") != "FROZEN":
        raise RuntimeError("E3 Passport remains draft")
    if release.get("experiment_identity") != {"experiment_id": authority.get("experiment_id"), "passport_sha256": passport.get("passport_payload_sha256")}:
        raise RuntimeError("E3 release Passport identity drift")
    from scripts.openevo_webshop import experiment_passport_gate
    current = active_router
    identity = current.get("experiment_identity") or {}
    if (identity.get("experiment_id") != authority.get("experiment_id")
            or identity.get("passport_sha256") != passport.get("passport_payload_sha256")
            or experiment_passport_gate.validate_binding(authority_repo, identity).get("passport_status") != "FROZEN"):
        raise RuntimeError("E3 Passport/Registry binding drift")
    if (current.get("phase") != "POST-ADVISOR-E3-FORMAL" or current.get("status") != "launch-authorized"
            or current.get("formal_task_consumption_allowed") is not True
            or current.get("final_panel_access_allowed") is not False
            or current.get("release", {}).get("path") != active_release_rel
            or str(current.get("release", {}).get("sha256", "")).removeprefix("sha256:") != sha(release_path)):
        raise RuntimeError("E3 current campaign route is not formal")
    arm_refs = release.get("arm_authorities") or {}
    profile_refs = release.get("profiles") or {}
    if set(arm_refs) != {str(rank) for rank in RANKS} or set(profile_refs) != set(arm_refs):
        raise RuntimeError("E3 B arm/profile references incomplete")
    bound_arms: dict[int, tuple[Path, dict, Path, dict, list[str]]] = {}
    for rank in RANKS:
        rank_key = str(rank)
        arm_path, arm = checked_ref(authority_repo, arm_refs[rank_key])
        profile_path, profile = checked_ref(authority_repo, profile_refs[rank_key])
        if (arm_path != authority_repo / E3_ARM.format(rank)
                or profile_path != authority_repo / E3_PROFILE.format(rank)
                or arm.get("status") != "AUTHORIZED" or arm.get("execution_git_sha") != a_sha
                or arm.get("persistent_state_rank") != rank or arm.get("formal_rollout_limit") != 1024
                or arm.get("rounds") != ROUNDS or arm.get("treatment_mode") != "BOUNDED_OFF"
                or arm.get("initial_replay_path") != (authority["arms"][rank_key].get("staged_replay_path") or str(Path(authority["arms"][rank_key]["initial_adapter_path"]) / "openevo_sd_lora_replay.jsonl"))
                or arm.get("aggregate_authority", {}).get("sha256") != sha(authority_path)
                or profile.get("source", {}).get("experiment", {}).get("revision") != a_sha
                or profile.get("execution", {}).get("mode") != "host-orchestrator"):
            raise RuntimeError(f"E3 rank{rank} B authority/profile drift")
        if rank == active_rank and release.get("execution_readiness", {}).get("profile") != profile_refs[rank_key]:
            raise RuntimeError("E3 release execution_readiness profile drift")
        command = profile.get("command") or []
        def has_arg(flag: str, value: str) -> bool:
            return any(command[i:i + 2] == [flag, value] for i in range(len(command) - 1))
        if (f"OPENEVO_POST_ADVISOR_AUTHORITY_REPO={authority_repo}" not in command
                or f"OPENEVO_E3_AGGREGATE_AUTHORITY={authority_repo / E3_AUTHORITY}" not in command
                or f"OPENEVO_E3_ARM_AUTHORITY={arm_path}" not in command
                or f"OPENEVO_E3_CURRENT_CAMPAIGN={authority_repo / E3_ROUTER.format(rank)}" not in command
                or not has_arg("--authority", str(arm_path))
                or not has_arg("--repo", str(repo))):
            raise RuntimeError(f"E3 rank{rank} profile authority routing drift")
        managed = arm.get("managed_gpu_indices") or []
        from scripts.openevo_webshop.post_advisor_e3_runtime_safety_202609300248 import GPU_UUID
        if managed != [0, 1, 2, 3]:
            raise RuntimeError(f"E3 rank{rank} GPU identity unbound")
        expected_uuids = [GPU_UUID[index] for index in managed]
        bound_arms[rank] = (arm_path, arm, profile_path, profile, expected_uuids)
        if not set(expected_uuids).issubset(set(current.get("authorized_gpu_uuids") or [])):
            raise RuntimeError(f"E3 rank{rank} GPU pool exceeds campaign authorization")
        evidence = (activation.get("rank_evidence") or {}).get(rank_key) or {}
        if set(evidence) != {"qualification", "zero_formal", "prelaunch", "gpu_gate", "gpu_owner", "storage", "scope_binding"}:
            raise RuntimeError(f"E3 rank{rank} evidence references incomplete")
        qualification = checked_external(evidence["qualification"])
        scope_attempts = c2_doc.get("prelaunch_attempts") or {}
        if set(scope_attempts) != {str(r) for r in RANKS}:
            raise RuntimeError("E3 immutable scope must bind one prelaunch attempt namespace per rank")
        attempt_binding = scope_attempts.get(rank_key) or {}
        expected_qualification_parent = Path(arm["output_root"]) / "preflight" / "attempt-0005"
        runs_root = (Path(profile["workspace"]) / profile["paths"]["runs"]).resolve()
        expected_marker_rel = expected_qualification_parent.resolve().relative_to(runs_root).as_posix()
        expected_attempt_binding = {
            "attempt": 5,
            "namespace": str(expected_qualification_parent),
            "container_name": f"experiment-host-openevo-e3-preflight-rank{rank}-attempt-0005",
            "probe_marker_path": f"/workspace/runs/{expected_marker_rel}/.prelaunch-write-probe-0005",
        }
        if attempt_binding != expected_attempt_binding:
            raise RuntimeError(f"E3 rank{rank} immutable scope attempt binding drift")
        expected_qualification_path = Path(str(evidence["qualification"].get("path") or ""))
        exact_qualification_path = expected_qualification_parent / f"launcher-qualification-{a_sha[:12]}-{b_sha[:12]}-attempt-0005.json"
        if (expected_qualification_path != exact_qualification_path
                or qualification.get("schema") != "openevo.post-advisor.e3-rank-launcher-qualification.202609300248"
                or qualification.get("status") != "PASS" or qualification.get("execution_sha") != a_sha
                or qualification.get("authority_git_sha") != b_sha
                or qualification.get("authority_sha256") != sha(arm_path)
                or qualification.get("aggregate_authority_sha256") != sha(authority_path)
                or qualification.get("persistent_state_rank") != rank
                or qualification.get("treatment_mode") != "BOUNDED_OFF"
                or qualification.get("formal_rollout_limit") != 1024
                or qualification.get("formal_task_consumed") is not False
                or any(qualification.get(key) != 0 for key in ("task_bearing_calls", "model_calls", "training_calls", "final_panel_access"))):
            raise RuntimeError(f"E3 rank{rank} launcher qualification drift")
        zero = checked_external(evidence["zero_formal"])
        prelaunch_path = Path(str(evidence["prelaunch"].get("path") or ""))
        prelaunch = checked_external(evidence["prelaunch"])
        gpu = checked_external(evidence["gpu_gate"])
        storage = checked_external(evidence["storage"])
        owner = checked_external(evidence["gpu_owner"])
        binding_receipt = checked_external(evidence["scope_binding"])
        expected_binding = {"execution_sha": a_sha, "authority_git_sha": b_sha, "arm_sha256": sha(arm_path),
                            "rank": rank, "router_sha256": sha(router_file), "profile_sha256": sha(profile_path),
                            "run_id": profile.get("run_id"), "gpu_uuids": expected_uuids,
                            "c2_sha": c2_sha, "c2_activation_sha256": c2_activation_sha}
        if qualification.get("e3_binding") != expected_binding:
            raise RuntimeError(f"E3 rank{rank} qualification scope binding drift")
        expected_receipts = {key: {"path": str(evidence[key].get("path") or ""), "sha256": evidence[key].get("sha256")}
                             for key in ("qualification", "zero_formal", "prelaunch", "gpu_gate", "gpu_owner", "storage")}
        if (binding_receipt.get("schema") != "openevo.post-advisor.e3-prelaunch-scope-binding.v1"
                or binding_receipt.get("status") != "PASS" or binding_receipt.get("e3_binding") != expected_binding
                or binding_receipt.get("receipts") != expected_receipts
                or binding_receipt.get("formal_task_consumed") is not False
                or any(binding_receipt.get(k) != 0 for k in ("scientific_budget_consumed", "model_calls", "training_calls", "final_panel_access"))):
            raise RuntimeError(f"E3 rank{rank} standard receipt scope binding drift")
        if (zero.get("schema") != "openevo.zero-formal-qualification.202609282338"
                or zero.get("status") != "ZERO_FORMAL_QUALIFIED"
                or zero.get("campaign_id") != authority.get("campaign_id")
                or zero.get("execution_sha") != a_sha
                or zero.get("selected_gpu_uuids") != expected_uuids
                or zero.get("prelaunch_receipt_file_sha256") != "sha256:" + sha(prelaunch_path)
                or zero.get("e3_binding") != expected_binding
                or zero.get("formal_task_consumed") is not False
                or zero.get("scientific_state_mutated") is not False
                or any(zero.get(key) != 0 for key in ("formal_task_budget", "task_bearing_calls", "model_calls", "training_calls", "final_panel_access"))):
            raise RuntimeError(f"E3 rank{rank} zero-formal qualification drift")
        payload = prelaunch.get("payload") or {}
        if (prelaunch.get("schema") != "openevo.prelaunch-zero-formal.202609281010"
                or prelaunch.get("status") != "PASS" or prelaunch.get("selected_gpu_uuids") != expected_uuids
                or prelaunch.get("formal_task_consumed") is not False
                or prelaunch.get("scientific_state_mutated") is not False
                or prelaunch.get("model_calls") != 0 or prelaunch.get("training_calls") != 0
                or prelaunch.get("final_panel_access") != 0 or payload.get("status") != "PASS"
                or payload.get("observed_gpu_count") != len(expected_uuids)):
            raise RuntimeError(f"E3 rank{rank} prelaunch receipt drift")
        if (prelaunch.get("run_id") != profile.get("run_id")
                or gpu.get("schema") != "openevo.webshop.gpu-gate.v2" or gpu.get("eligible") is not True
                or gpu.get("selected_gpu_uuids") != expected_uuids
                or gpu.get("first_failures") or gpu.get("second_failures")
                or not isinstance(gpu.get("first"), list) or not isinstance(gpu.get("second"), list)
                or {row.get("uuid") for row in gpu["first"] if isinstance(row, dict)}.issuperset(expected_uuids) is False
                or {row.get("uuid") for row in gpu["second"] if isinstance(row, dict)}.issuperset(expected_uuids) is False):
            raise RuntimeError(f"E3 rank{rank} GPU gate drift")
        if (owner.get("schema") != "openevo.post-advisor.e3-gpu-owner-gate.v1" or owner.get("status") != "PASS"
                or owner.get("e3_binding") != expected_binding
                or owner.get("formal_task_consumed") is not False
                or any(owner.get(k) != 0 for k in ("scientific_budget_consumed", "model_calls", "training_calls", "final_panel_access"))
                or any((owner.get(k) or {}).get("selected_indices") != [0,1,2,3]
                       or (owner.get(k) or {}).get("selected_classifications") != {str(i):"idle" for i in [0,1,2,3]}
                       for k in ("before", "after"))):
            raise RuntimeError(f"E3 rank{rank} live GPU owner receipt drift")
        capacity = storage.get("capacity_method") or {}
        fs = storage.get("filesystem") or {}
        required = int(capacity.get("estimated_total_bytes", -1)) + int(capacity.get("explicit_free_space_reserve_bytes", -1))
        if (storage.get("schema") != "openevo.post-advisor.storage-capacity-gate.v1"
                or storage.get("status") != "PASS" or storage.get("campaign_id") != authority.get("campaign_id")
                or storage.get("formal_task_calls_consumed") != 0 or storage.get("final_panel_access") != 0
                or storage.get("execution_sha") != a_sha or storage.get("authority_git_sha") != b_sha
                or storage.get("rank") != rank or storage.get("e3_scope", {}).get("rank") != rank
                or storage.get("c2_sha") != c2_sha or storage.get("c2_activation_sha256") != c2_activation_sha
                or storage.get("e3_scope", {}).get("formal_rollout_limit") != 1024
                or fs.get("fstype") != "fuse.galaxyfs" or fs.get("mount_target") != "/workspace"
                or int(fs.get("required_bytes", -1)) != required
                or fs.get("persistent_output_policy") != "GalaxyFS"
                or not str(fs.get("mount") or "").startswith("GalaxyFS:")
                or int(fs.get("available_bytes", -1)) < required or required <= 0
                or int(capacity.get("rollouts_per_planned_arm", -1)) != 1024
                or int(capacity.get("rounds_per_planned_arm", -1)) != 8
                or f"E3_RANK_CAPACITY_SWEEP_RANK{rank}" not in (capacity.get("planned_arms") or [])):
            raise RuntimeError(f"E3 rank{rank} storage capacity drift")
    start_path = Path(os.environ.get("OPENEVO_FORMAL_START_RECEIPT", ""))
    if not start_path.is_file():
        raise RuntimeError("E3 Formal Start receipt missing")
    validate_e3_formal_start_receipt_location(start_path)
    start = load(start_path)
    from scripts.openevo_webshop import prelaunch_zero_formal
    projection = {key: value for key, value in start.items() if key != "receipt_sha256"}
    if (start.get("status") != "FORMAL_START_COMMITTED"
            or start.get("schema") != prelaunch_zero_formal.FORMAL_START_SCHEMA
            or start.get("receipt_sha256") != prelaunch_zero_formal.canonical_sha256(projection)
            or start.get("campaign_id") != authority.get("campaign_id")
            or start.get("formal_start_committed") is not True
            or start.get("formal_task_consumption_may_begin") is not True
            or start.get("formal_task_consumption_before") is not False):
        raise RuntimeError("E3 Formal Start not committed")
    assigned = [value for value in os.environ.get("OPENEVO_ASSIGNED_GPU_UUIDS", "").split(",") if value]
    if not assigned or len(set(assigned)) != len(assigned) or start.get("selected_gpu_uuids") != assigned:
        raise RuntimeError("E3 Formal Start GPU assignment drift")
    matching = [(rank, item) for rank, item in bound_arms.items() if item[0] == active_arm_path]
    if len(matching) != 1:
        raise RuntimeError("E3 Formal Start active arm is not bound in B")
    active_rank, (active_path, active_arm, active_profile_path, active_profile, active_uuids) = matching[0]
    if assigned != active_uuids:
        raise RuntimeError("E3 Formal Start arm GPU assignment drift")
    run_dir = Path(str(start.get("output_root") or "")).resolve()
    if not run_dir.is_dir() or run_dir == Path(active_arm["output_root"]).resolve():
        raise RuntimeError("E3 oe run directory must be distinct from scientific arm root")
    readiness = start.get("readiness") or {}
    readiness_authority = readiness.get("authority") or {}
    if (readiness_authority.get("router_path") != str(authority_repo / active_router_rel)
            or readiness_authority.get("router_release_pointer") != active_release_rel
            or readiness_authority.get("router_status") != "launch-authorized"):
        raise RuntimeError("E3 Formal Start did not use active rank router")
    ready_ref = readiness.get("ready_receipt") or {}
    ready_path = Path(str(ready_ref.get("path") or ""))
    if (not ready_path.is_file() or ready_ref.get("sha256") != "sha256:" + sha(ready_path)
            or load(ready_path).get("verdict") != "READY"):
        raise RuntimeError("E3 oe readiness receipt drift")
    profile_ref = readiness.get("profile") or {}
    if (Path(str(profile_ref.get("path") or "")).resolve() != active_profile_path
            or profile_ref.get("sha256") != "sha256:" + sha(active_profile_path)):
        raise RuntimeError("E3 Formal Start profile differs from active rank B profile")
    manifest = load(run_dir / "launch-receipt.json")
    start_ref = manifest.get("formal_start") or {}
    if (start_ref.get("path") != str(start_path) or start_ref.get("sha256") != "sha256:" + sha(start_path)
            or start_ref.get("receipt_sha256") != start["receipt_sha256"]
            or manifest.get("execution_mode") != "host-orchestrator"
            or manifest.get("profile") != str(active_profile_path)
            or manifest.get("selected_gpu_uuids") != active_uuids):
        raise RuntimeError("E3 launch receipt does not bind Formal Start")
    from scripts import oe_impl
    command = oe_impl.host_orchestrator_identity_command(
        {**active_profile, "_formal_start_context": readiness}, assigned,
        run_dir, manifest["tracking"]["effective_mode"], start_path,
    )
    if (start.get("command_identity") != oe_impl._command_identity(command, start.get("image_id"))
            or start.get("image_id") != manifest.get("image_id")):
        raise RuntimeError("E3 Formal Start command identity drift")
    return preparation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    result = verify_formal_activation(args.repo, args.authority) if args.formal else verify_preparation(args.repo, args.authority)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
