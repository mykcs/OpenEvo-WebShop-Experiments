from __future__ import annotations
import hashlib, json, os, re, subprocess
from pathlib import Path

from scripts.openevo_webshop import experiment_passport_gate
from scripts.openevo_webshop.experiment_registry import load_current_registry, resolve_experiment

READY_SCHEMA = "openevo.bounded-gdr.formal-ready-package.202609151740-v2"  # legacy-revision-compat: absorbed historical runtime literal from a6f67b
LAUNCH_SCHEMA = "openevo.bounded-gdr.owner-launch-release.202609151740-v2"  # legacy-revision-compat: absorbed historical runtime literal from a6f67b
CURRENT_CAMPAIGN = "configs/experiment/current-campaign.json"
POST_ADVISOR_DESIGN = "configs/experiment/designs/post-advisor-unanswered-202609282338.json"
POST_ADVISOR_ACTIVATION = "configs/experiment/activations/post-advisor-unanswered-202609282338.json"
POST_ADVISOR_CAMPAIGN_ID = "20260928-2338-post-advisor-unanswered"
POST_ADVISOR_E1_ID = "202609282338-post-advisor-e1-beta-direction"
AUTHORITY_REPO_ENV = "OPENEVO_POST_ADVISOR_AUTHORITY_REPO"


def _authority_material(path: str) -> bool:
    """Files allowed between the qualified execution commit and release commit."""
    if path == "configs/experiment/current-experiment-registry.json" or re.fullmatch(
        r"configs/experiment/experiment-registry-\d{12}\.json", path
    ):
        return True
    if path == f"configs/experiment/passports/{POST_ADVISOR_E1_ID}/passport.json":
        return True
    if path in {POST_ADVISOR_ACTIVATION, POST_ADVISOR_DESIGN, CURRENT_CAMPAIGN}:
        return False
    if not re.fullmatch(
        r"configs/experiment/(?:(?:activations|releases|profiles|run-profiles|preregistrations|path-contracts)/)?post-advisor-[^/]+\.json",
        path,
    ):
        return False
    name = Path(path).name
    if path in {"configs/experiment/activations/post-advisor-e1-a9-carrier-health-resolution-202609301600.json",
                "configs/experiment/activations/post-advisor-e1-a11-carrier-health-resolution-202609302100.json"}:
        return True
    return any(marker in name for marker in ("authority", "release", "profile", "path-contract", "preregistration"))


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], text=True, capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"post-advisor Git identity failed: {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout.strip()

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def load(path: Path):
    return json.loads(path.read_text())

def _evidence(ref: dict, root: Path, label: str) -> tuple[Path, dict]:
    path = Path(str(ref.get("path") or ""))
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    expected = str(ref.get("sha256") or "").removeprefix("sha256:")
    if not path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", expected) or sha(path) != expected:
        raise RuntimeError(f"post-advisor {label} evidence missing or SHA drift")
    return path, load(path)


def _formal_start(activation: dict, campaign: dict, authority_repo: Path, arm_authority_path: Path) -> None:
    from scripts.openevo_webshop import prelaunch_zero_formal
    from scripts.openevo_webshop.bounded_effective_state_runtime_safety_202609282338 import GPU_UUID
    from scripts import oe_impl
    evidence = activation.get("qualification_evidence")
    required = {"zero_formal", "prelaunch", "gpu_gate", "storage", "launcher_e1", "e1_authority", "release", "profile"}
    if not isinstance(evidence, dict) or not required.issubset(evidence):
        raise RuntimeError("post-advisor activation has no complete SHA-bound qualification evidence")
    docs = {key: _evidence(evidence[key], authority_repo, key) for key in required}
    selected_pool = campaign.get("authorized_gpu_uuids")
    zero = docs["zero_formal"][1]
    prelaunch = docs["prelaunch"][1]
    gpu = docs["gpu_gate"][1]
    storage = docs["storage"][1]
    launcher = docs["launcher_e1"][1]
    arm = docs["e1_authority"][1]
    recovery_ref = arm.get("engineering_recovery")
    if recovery_ref:
        from scripts.openevo_webshop.bounded_effective_state_recovery_202609282338 import validate_post_advisor_recovery
        recovery_path = Path(str(recovery_ref.get("path") or "")).resolve()
        if not recovery_path.is_relative_to(authority_repo):
            raise RuntimeError("post-advisor recovery authority must live in committed B")
        recovery_rel = str(recovery_path.relative_to(authority_repo))
        if not _authority_material(recovery_rel) or _git(authority_repo, "show", f"HEAD:{recovery_rel}") != recovery_path.read_text().strip():
            raise RuntimeError("post-advisor recovery authority is not committed")
        recovery_doc = validate_post_advisor_recovery(arm)
        if recovery_doc.get("schema") in {
            "openevo.post-advisor.e1-engineering-continuation.a9.202609301600",
            "openevo.post-advisor.e1-engineering-continuation.a10.202609301900",
            "openevo.post-advisor.e1-terminal-seal-recovery.a11.202609302100",
        }:
            resolution_ref = arm.get("carrier_health_resolution") or {}
            resolution_path = Path(str(resolution_ref.get("path") or "")).resolve()
            if (not resolution_path.is_file() or not resolution_path.is_relative_to(authority_repo)
                    or sha(resolution_path) != resolution_ref.get("sha256")):
                raise RuntimeError("A9 R152 resolution authority is not SHA-bound")
            resolution_rel = str(resolution_path.relative_to(authority_repo))
            if (not _authority_material(resolution_rel)
                    or _git(authority_repo, "show", f"HEAD:{resolution_rel}") != resolution_path.read_text().strip()):
                raise RuntimeError("A9 R152 resolution must be committed in B")
            if recovery_doc.get("schema") == "openevo.post-advisor.e1-terminal-seal-recovery.a11.202609302100":
                current_ref = arm.get("terminal_carrier_health_resolution") or {}
                current_path = Path(str(current_ref.get("path") or "")).resolve()
                if (not current_path.is_file() or not current_path.is_relative_to(authority_repo)
                        or sha(current_path) != current_ref.get("sha256")):
                    raise RuntimeError("A11 R159 resolution authority is not SHA-bound")
                current_rel = str(current_path.relative_to(authority_repo))
                if (not _authority_material(current_rel)
                        or _git(authority_repo, "show", f"HEAD:{current_rel}") != current_path.read_text().strip()):
                    raise RuntimeError("A11 R159 resolution must be committed in B")
        prereg_path = authority_repo / str((release_ref := load(docs["release"][0])).get("preregistration", {}).get("path") or "")
        if not prereg_path.is_file() or sha(prereg_path) != release_ref.get("preregistration", {}).get("sha256"):
            raise RuntimeError("post-advisor recovery frozen preregistration drift")
        frozen = load(prereg_path).get("frozen_scientific_contract") or {}
        if frozen.get("execution_sha") != recovery_doc["predecessor_execution_git_sha"]:
            raise RuntimeError("post-advisor recovery must preserve the A6 frozen scientific identity")
        snapshot_rel = recovery_doc.get("predecessor_arm_authority_snapshot")
        snapshot = authority_repo / str(snapshot_rel or "")
        if (not isinstance(snapshot_rel, str) or not _authority_material(snapshot_rel)
                or not snapshot.is_file() or sha(snapshot) != recovery_doc["predecessor_arm_authority_sha256"]
                or _git(authority_repo, "show", f"HEAD:{snapshot_rel}") != snapshot.read_text().strip()):
            raise RuntimeError("post-advisor recovery A6 arm authority snapshot drift")
        snapshot_doc = load(snapshot)
        if snapshot_doc.get("execution_git_sha") != recovery_doc["predecessor_execution_git_sha"] or snapshot_doc.get("output_root") != recovery_doc["target_output_root"]:
            raise RuntimeError("post-advisor recovery A6 snapshot identity drift")
        if recovery_doc["successor_execution_git_sha"] != activation["execution_sha"]:
            raise RuntimeError("post-advisor recovery successor activation SHA drift")
        engineering_paths = {
            "scripts/openevo_webshop/bounded_effective_state_recovery_202609282338.py",
            "scripts/openevo_webshop/post_advisor_short_fork_runner_202609282338.py",
            "scripts/openevo_webshop/post_advisor_bounded_short_fork_202609282338.py",
            "scripts/openevo_webshop/launch_post_advisor_magnitude_transition_20260930.sh",
            "scripts/openevo_webshop/bounded_effective_state_launch_guard_202609282338.py",
        }
        if recovery_doc.get("engineering_predecessor_execution_git_sha"):
            predecessor_activation_sha = recovery_doc["engineering_predecessor_activation_git_sha"]
            _git(authority_repo, "merge-base", "--is-ancestor", predecessor_activation_sha,
                 recovery_doc["successor_execution_git_sha"])
            if recovery_doc.get("schema") == "openevo.post-advisor.e1-terminal-seal-recovery.a11.202609302100":
                predecessor_arm_rel = "configs/experiment/activations/post-advisor-e1-a10-recovery-authority-202609301900.json"
            elif recovery_doc.get("schema") == "openevo.post-advisor.e1-engineering-continuation.a10.202609301900":
                predecessor_arm_rel = "configs/experiment/activations/post-advisor-e1-a9-recovery-authority-202609301600.json"
            elif recovery_doc.get("schema") == "openevo.post-advisor.e1-engineering-continuation.a9.202609301600":
                predecessor_arm_rel = "configs/experiment/activations/post-advisor-e1-a8-recovery-authority-202609301200.json"
            else:
                predecessor_arm_rel = "configs/experiment/activations/post-advisor-e1-a7-recovery-authority-202609301000.json"
            predecessor_arm = authority_repo / predecessor_arm_rel
            if (not predecessor_arm.is_file()
                    or sha(predecessor_arm) != recovery_doc.get("engineering_predecessor_arm_authority_sha256")
                    or _git(authority_repo, "show", f"{predecessor_activation_sha}:{predecessor_arm_rel}") != predecessor_arm.read_text().strip()):
                raise RuntimeError("post-advisor engineering predecessor arm authority drift")
            engineering_paths.update({POST_ADVISOR_ACTIVATION, CURRENT_CAMPAIGN})
            changed = set(_git(authority_repo, "diff", "--name-only", predecessor_activation_sha,
                               recovery_doc["successor_execution_git_sha"]).splitlines())
            if not changed or not changed.issubset(engineering_paths):
                raise RuntimeError("post-advisor recovery successor includes non-engineering source changes")
        else:
            changed = set(_git(authority_repo, "diff", "--name-only",
                               recovery_doc["predecessor_execution_git_sha"], recovery_doc["successor_execution_git_sha"]).splitlines())
            if not changed or not changed.issubset(engineering_paths):
                raise RuntimeError("post-advisor recovery A6→A7 includes non-engineering source changes")
    indices = arm.get("managed_gpu_indices") or []
    if len(indices) != 4 or len(set(indices)) != 4 or any(index not in GPU_UUID for index in indices):
        raise RuntimeError("post-advisor E1 managed GPU identity drift")
    expected_arm = [GPU_UUID[index] for index in indices]
    if not isinstance(selected_pool, list) or not set(expected_arm).issubset(set(selected_pool)):
        raise RuntimeError("post-advisor E1 GPUs exceed authorized campaign pool")
    if zero.get("status") != "ZERO_FORMAL_QUALIFIED" or zero.get("execution_sha") != activation["execution_sha"] or zero.get("campaign_id") != POST_ADVISOR_CAMPAIGN_ID or zero.get("selected_gpu_uuids") != expected_arm:
        raise RuntimeError("post-advisor zero-formal qualification identity drift")
    if any(zero.get(key) != 0 for key in ("formal_task_budget", "task_bearing_calls", "model_calls", "training_calls", "final_panel_access")) or zero.get("formal_task_consumed") is not False:
        raise RuntimeError("post-advisor zero-formal qualification consumed protected work")
    if prelaunch.get("status") != "PASS" or prelaunch.get("selected_gpu_uuids") != expected_arm or zero.get("prelaunch_receipt_file_sha256") != "sha256:" + sha(docs["prelaunch"][0]):
        raise RuntimeError("post-advisor prelaunch evidence drift")
    if gpu.get("eligible") is not True or gpu.get("selected_gpu_uuids") != expected_arm or gpu.get("first_failures") or gpu.get("second_failures"):
        raise RuntimeError("post-advisor GPU gate evidence drift")
    if storage.get("status") != "PASS" or storage.get("campaign_id") != POST_ADVISOR_CAMPAIGN_ID or storage.get("final_panel_access") != 0 or storage.get("formal_task_calls_consumed") != 0:
        raise RuntimeError("post-advisor storage evidence drift")
    arm_authority_path = arm_authority_path.resolve()
    if arm_authority_path != docs["e1_authority"][0] or not arm_authority_path.is_relative_to(authority_repo) or arm.get("execution_git_sha") != activation["execution_sha"] or arm.get("persistent_state_rank") != 128 or arm.get("treatment_mode") != "MAGNITUDE_SCALAR_GATE_V1":
        raise RuntimeError("post-advisor E1 arm authority drift")
    arm_rel = str(arm_authority_path.relative_to(authority_repo))
    if not _authority_material(arm_rel) or _git(authority_repo, "show", f"HEAD:{arm_rel}") != arm_authority_path.read_text().strip():
        raise RuntimeError("post-advisor E1 arm authority is not committed in B")
    if launcher.get("status") != "PASS" or launcher.get("execution_sha") != activation["execution_sha"] or launcher.get("authority_sha256") != sha(arm_authority_path) or launcher.get("formal_task_consumed") is not False or launcher.get("final_panel_access") != 0:
        raise RuntimeError("post-advisor E1 launcher qualification drift")
    release_path, release = docs["release"]
    profile_path, profile = docs["profile"]
    if not release_path.is_relative_to(authority_repo) or not profile_path.is_relative_to(authority_repo):
        raise RuntimeError("post-advisor release/profile must be committed authority files")
    for path in (release_path, profile_path):
        rel = str(path.relative_to(authority_repo))
        if not _authority_material(rel) or _git(authority_repo, "show", f"HEAD:{rel}") != path.read_text().strip():
            raise RuntimeError("post-advisor release/profile committed identity drift")
    if profile.get("source", {}).get("experiment", {}).get("revision") != activation["execution_sha"] or profile.get("execution", {}).get("mode") != "host-orchestrator":
        raise RuntimeError("post-advisor profile does not bind execution A and host orchestration")
    if release.get("experiment_identity") != campaign.get("experiment_identity"):
        raise RuntimeError("post-advisor release Passport identity drift")
    if campaign.get("release", {}).get("path") != str(release_path.relative_to(authority_repo)) or str(campaign.get("release", {}).get("sha256", "")).removeprefix("sha256:") != sha(release_path):
        raise RuntimeError("post-advisor campaign release binding drift")
    passport = experiment_passport_gate.validate_binding(authority_repo, campaign.get("experiment_identity") or {})
    if passport.get("passport_status") != "FROZEN" or campaign["experiment_identity"].get("experiment_id") != POST_ADVISOR_E1_ID:
        raise RuntimeError("post-advisor E1 Passport is not frozen")

    start_env = os.environ.get("OPENEVO_FORMAL_START_RECEIPT")
    assigned = [item for item in os.environ.get("OPENEVO_ASSIGNED_GPU_UUIDS", "").split(",") if item]
    if not start_env or not assigned or len(set(assigned)) != len(assigned):
        raise RuntimeError("post-advisor formal launch lacks oe Formal Start assignment")
    start_path = Path(start_env).resolve()
    start = load(start_path)
    projection = {key: value for key, value in start.items() if key != "receipt_sha256"}
    if start.get("schema") != prelaunch_zero_formal.FORMAL_START_SCHEMA or start.get("status") != "FORMAL_START_COMMITTED" or start.get("receipt_sha256") != prelaunch_zero_formal.canonical_sha256(projection):
        raise RuntimeError("post-advisor oe Formal Start receipt invalid")
    if start.get("campaign_id") != POST_ADVISOR_CAMPAIGN_ID or start.get("selected_gpu_uuids") != assigned or assigned != expected_arm or start.get("formal_start_committed") is not True or start.get("formal_task_consumption_may_begin") is not True or start.get("formal_task_consumption_before") is not False:
        raise RuntimeError("post-advisor oe Formal Start identity drift")
    run_dir = Path(str(start.get("output_root") or "")).resolve()
    manifest = load(run_dir / "launch-receipt.json")
    start_ref = manifest.get("formal_start") or {}
    if Path(str(start_ref.get("path") or "")).resolve() != start_path or start_ref.get("sha256") != "sha256:" + sha(start_path) or start_ref.get("receipt_sha256") != start["receipt_sha256"] or manifest.get("execution_mode") != "host-orchestrator" or manifest.get("selected_gpu_uuids") != assigned:
        raise RuntimeError("post-advisor oe launch manifest does not bind Formal Start")
    readiness = start.get("readiness") or {}
    ready_ref = readiness.get("ready_receipt") or {}
    ready_path = Path(str(ready_ref.get("path") or "")).resolve()
    if not ready_path.is_file() or ready_ref.get("sha256") != "sha256:" + sha(ready_path) or load(ready_path).get("verdict") != "READY":
        raise RuntimeError("post-advisor oe readiness receipt drift")
    profile_ref = readiness.get("profile") or {}
    if Path(str(profile_ref.get("path") or "")).resolve() != profile_path or profile_ref.get("sha256") != "sha256:" + sha(profile_path):
        raise RuntimeError("post-advisor oe profile readiness drift")
    expected = oe_impl.host_orchestrator_identity_command({**profile, "_formal_start_context": readiness}, assigned, run_dir, manifest["tracking"]["effective_mode"], start_path)
    if start.get("command_identity") != oe_impl._command_identity(expected, start.get("image_id")) or start.get("image_id") != manifest.get("image_id"):
        raise RuntimeError("post-advisor oe Formal Start command identity drift")


def validate_post_advisor_terminal_seal_activation(repo: Path, execution_sha: str, arm_authority_path: Path) -> dict:
    """A11 CPU-only seal authority; no GPU lease, Formal Start, or task-bearing route."""
    from scripts.openevo_webshop.bounded_effective_state_recovery_202609282338 import (
        POST_ADVISOR_A11_RECOVERY_SCHEMA, validate_post_advisor_recovery,
    )
    repo = repo.resolve()
    if _git(repo, "rev-parse", "HEAD") != execution_sha or _git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("A11 seal requires clean exact execution A")
    authority_root = os.environ.get(AUTHORITY_REPO_ENV)
    if not authority_root:
        raise RuntimeError("A11 seal requires exact authority checkout")
    authority_repo = Path(authority_root).resolve()
    if not authority_repo.is_dir() or _git(authority_repo, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("A11 seal authority checkout must be clean")
    c_sha = _git(authority_repo, "rev-parse", "HEAD")
    parents = _git(authority_repo, "rev-list", "--parents", "-n", "1", c_sha).split()
    if len(parents) != 2:
        raise RuntimeError("A11 activation C must have exactly one parent B")
    b_sha = parents[1]
    commits = _git(authority_repo, "rev-list", "--reverse", f"{execution_sha}..{b_sha}").splitlines()
    if not commits:
        raise RuntimeError("A11 authority B missing")
    _git(authority_repo, "merge-base", "--is-ancestor", "7c79c8906e37486c2125d466b3c53ebe92ab987a", execution_sha)
    previous = execution_sha
    for commit in commits:
        if _git(authority_repo, "rev-list", "--parents", "-n", "1", commit).split() != [commit, previous]:
            raise RuntimeError("A11 source-to-authority chain is not linear")
        changed = set(_git(authority_repo, "diff", "--name-only", previous, commit).splitlines())
        allowed_b = {
            "configs/experiment/activations/post-advisor-e1-a11-carrier-health-resolution-202609302100.json",
            "configs/experiment/activations/post-advisor-e1-a11-terminal-seal-authority-202609302100.json",
            "configs/experiment/activations/post-advisor-e1-a11-terminal-seal-authority-detail-202609302100.json",
            "configs/experiment/activations/post-advisor-e1-a11-terminal-seal-release-202609302100.json",
        }
        if not changed or not changed <= allowed_b:
            raise RuntimeError("A11 B contains non-seal authority changes")
        previous = commit
    if set(_git(authority_repo, "diff", "--name-only", b_sha, c_sha).splitlines()) != {POST_ADVISOR_ACTIVATION, CURRENT_CAMPAIGN}:
        raise RuntimeError("A11 C must change only activation and current-campaign")
    arm_authority_path = arm_authority_path.resolve()
    if not arm_authority_path.is_relative_to(authority_repo):
        raise RuntimeError("A11 arm authority escapes committed B")
    arm_rel = str(arm_authority_path.relative_to(authority_repo))
    if not _authority_material(arm_rel):
        raise RuntimeError("A11 arm authority path is not authority material")
    def committed(path: Path) -> None:
        rel = str(path.relative_to(authority_repo))
        if not path.is_file() or _git(authority_repo, "show", f"HEAD:{rel}") != path.read_text().strip():
            raise RuntimeError(f"A11 authority bytes are not committed: {rel}")
    activation_path = authority_repo / POST_ADVISOR_ACTIVATION
    campaign_path = authority_repo / CURRENT_CAMPAIGN
    design_path = authority_repo / POST_ADVISOR_DESIGN
    for path in (arm_authority_path, activation_path, campaign_path, design_path):
        committed(path)
    arm = load(arm_authority_path)
    activation = load(activation_path)
    campaign = load(campaign_path)
    if (arm.get("formal_task_consumption_allowed") is not False
            or arm.get("final_panel_access_limit") != 0
            or arm.get("historical_rank128_control_rerun") is not False
            or activation.get("schema") != "openevo.post-advisor.e1-terminal-seal-activation.a11.202609302100"
            or activation.get("status") != "SEAL_ONLY_AUTHORIZED"
            or activation.get("execution_sha") != execution_sha
            or activation.get("authority_path") != arm_rel
            or activation.get("authority_sha256") != sha(arm_authority_path)
            or activation.get("campaign_sha256") != sha(campaign_path)
            or activation.get("design_sha256") != sha(design_path)
            or activation.get("design_path") != POST_ADVISOR_DESIGN
            or activation.get("formal_task_consumption_allowed") is not False
            or activation.get("final_panel_access_allowed") is not False):
        raise RuntimeError("A11 seal activation identity or scope drift")
    if (campaign.get("campaign_id") != POST_ADVISOR_CAMPAIGN_ID
            or campaign.get("status") != "seal-only-authorized"
            or campaign.get("phase") != "POST-ADVISOR-E1-SEAL-ONLY"
            or campaign.get("formal_task_consumption_allowed") is not False
            or campaign.get("task_bearing_train_dev_allowed") is not False
            or campaign.get("gpu_allocation_allowed") is not False
            or campaign.get("final_panel_access_allowed") is not False
            or campaign.get("final_unlock_allowed") is not False
            or campaign.get("remaining_e1_rollout_budget") != 0
            or campaign.get("already_consumed_e1_rollouts") != 1024):
        raise RuntimeError("A11 seal-only campaign boundary drift")
    release_ref = campaign.get("release") or {}
    release_path = authority_repo / str(release_ref.get("path") or "")
    if (not release_path.is_relative_to(authority_repo) or not release_path.is_file()
            or sha(release_path) != release_ref.get("sha256") or not _authority_material(str(release_path.relative_to(authority_repo)))):
        raise RuntimeError("A11 release binding drift")
    committed(release_path)
    release = load(release_path)
    if (release.get("status") != "SEAL_ONLY_AUTHORIZED" or release.get("execution_sha") != execution_sha
            or release.get("arm_authority_sha256") != sha(arm_authority_path)
            or release.get("arm_authority") != {"path": arm_rel, "sha256": sha(arm_authority_path)}
            or release.get("engineering_recovery") != {"path": str(Path(arm["engineering_recovery"]["path"]).relative_to(authority_repo)), "sha256": arm["engineering_recovery"]["sha256"]}
            or release.get("terminal_carrier_health_resolution") != {"path": str(Path(arm["terminal_carrier_health_resolution"]["path"]).relative_to(authority_repo)), "sha256": arm["terminal_carrier_health_resolution"]["sha256"]}
            or release.get("formal_task_consumption_allowed") is not False
            or release.get("remaining_rollout_budget") != 0):
        raise RuntimeError("A11 release is not seal-only")
    if campaign.get("experiment_identity") != release.get("experiment_identity"):
        raise RuntimeError("A11 frozen Passport binding drift")
    passport = experiment_passport_gate.validate_binding(authority_repo, campaign["experiment_identity"])
    if passport.get("passport_status") != "FROZEN":
        raise RuntimeError("A11 frozen Passport is unavailable")
    recovery_ref = arm.get("engineering_recovery") or {}
    recovery_path = Path(str(recovery_ref.get("path") or ""))
    resolution_ref = arm.get("terminal_carrier_health_resolution") or {}
    resolution_path = Path(str(resolution_ref.get("path") or ""))
    for path, ref in ((recovery_path, recovery_ref), (resolution_path, resolution_ref)):
        if (not path.is_relative_to(authority_repo) or not _authority_material(str(path.relative_to(authority_repo)))
                or sha(path) != ref.get("sha256")):
            raise RuntimeError("A11 recovery/resolution identity drift")
        committed(path)
    doc = validate_post_advisor_recovery(arm)
    if doc.get("schema") != POST_ADVISOR_A11_RECOVERY_SCHEMA or arm.get("execution_git_sha") != execution_sha:
        raise RuntimeError("A11 seal authority schema/source drift")
    return activation


def validate_post_advisor_activation(repo: Path, execution_sha: str | None = None, arm_authority_path: Path | None = None) -> dict:
    """Verify clean execution A against the separate, committed authority C."""
    repo = repo.resolve()
    head = _git(repo, "rev-parse", "HEAD")
    execution_sha = execution_sha or head
    if execution_sha != head:
        raise RuntimeError("post-advisor activation execution SHA does not match checkout HEAD")
    if _git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("post-advisor activation requires a clean committed checkout")
    authority_root = os.environ.get(AUTHORITY_REPO_ENV)
    if not authority_root:
        raise RuntimeError(f"formal post-advisor launch requires {AUTHORITY_REPO_ENV}")
    authority_repo = Path(authority_root).expanduser().resolve()
    if not authority_repo.is_dir() or _git(authority_repo, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("post-advisor authority checkout must exist and be clean")
    authority_head = _git(authority_repo, "rev-parse", "HEAD")
    parents = _git(authority_repo, "rev-list", "--parents", "-n", "1", authority_head).split()
    if len(parents) != 2:
        raise RuntimeError("post-advisor activation C must have exactly one parent B")
    release_sha = parents[1]
    commits = _git(authority_repo, "rev-list", "--reverse", f"{execution_sha}..{release_sha}").splitlines()
    if not commits:
        raise RuntimeError("release B has no committed authority material after execution A")
    previous = execution_sha
    for commit in commits:
        lineage = _git(authority_repo, "rev-list", "--parents", "-n", "1", commit).split()
        if lineage != [commit, previous]:
            raise RuntimeError("release A..B must be a linear authority-only commit chain")
        changed = set(_git(authority_repo, "diff", "--name-only", previous, commit).splitlines())
        if not changed or any(not _authority_material(path) for path in changed):
            raise RuntimeError("release A..B contains a non-authority commit")
        previous = commit
    release_changes = set(_git(authority_repo, "diff", "--name-only", execution_sha, release_sha).splitlines())
    if not release_changes or any(not _authority_material(path) for path in release_changes):
        raise RuntimeError("release B changed files outside post-advisor authority material")
    activation_changes = set(_git(authority_repo, "diff", "--name-only", release_sha, authority_head).splitlines())
    if activation_changes != {POST_ADVISOR_ACTIVATION, CURRENT_CAMPAIGN}:
        raise RuntimeError("activation C must change only activation and current-campaign")
    design_path = authority_repo / POST_ADVISOR_DESIGN
    activation_path = authority_repo / POST_ADVISOR_ACTIVATION
    campaign_path = authority_repo / CURRENT_CAMPAIGN
    if not all(path.is_file() for path in (design_path, activation_path, campaign_path)):
        raise RuntimeError("post-advisor activation authority files are missing")
    activation = load(activation_path)
    if activation.get("status") != "PASS":
        raise RuntimeError("post-advisor activation is not PASS")
    def committed_bytes(relative_path: str) -> bytes:
        return subprocess.run(["git", "-C", str(authority_repo), "show", f"HEAD:{relative_path}"], text=False, capture_output=True, check=True).stdout
    for relative_path, path in ((POST_ADVISOR_DESIGN, design_path), (POST_ADVISOR_ACTIVATION, activation_path), (CURRENT_CAMPAIGN, campaign_path)):
        if committed_bytes(relative_path) != path.read_bytes():
            raise RuntimeError(f"post-advisor committed authority blob drift: {relative_path}")
    if activation.get("execution_sha") != execution_sha:
        raise RuntimeError("post-advisor activation does not bind qualified execution A")
    if sha(repo / POST_ADVISOR_DESIGN) != sha(design_path):
        raise RuntimeError("authority design differs from qualified execution A")
    design = load(design_path)
    campaign = load(campaign_path)
    if activation.get("schema") != "openevo.post-advisor-unanswered-activation.202609282338" or activation.get("revision") != "202609282338":
        raise RuntimeError("post-advisor activation schema/revision drift")
    if activation.get("design_path") != POST_ADVISOR_DESIGN or activation.get("design_sha256") != sha(design_path):
        raise RuntimeError("post-advisor activation design identity drift")
    if activation.get("campaign_id") != POST_ADVISOR_CAMPAIGN_ID:
        raise RuntimeError("post-advisor activation campaign identity drift")
    if activation.get("formal_task_consumption_allowed") is not True or activation.get("final_panel_access_allowed") is not False:
        raise RuntimeError("post-advisor activation formal/final-panel boundary drift")
    if activation.get("campaign_sha256") != sha(campaign_path):
        raise RuntimeError("post-advisor activation campaign identity drift")
    if campaign.get("campaign_id") != POST_ADVISOR_CAMPAIGN_ID or campaign.get("design") != POST_ADVISOR_DESIGN:
        raise RuntimeError("post-advisor live campaign identity drift")
    if campaign.get("status") != "launch-authorized" or campaign.get("formal_task_consumption_allowed") is not True or campaign.get("task_bearing_train_dev_allowed") is not True:
        raise RuntimeError("post-advisor live campaign remains formal-locked")
    if campaign.get("final_test_status") != "locked" or campaign.get("final_panel_access_allowed") is not False or campaign.get("final_unlock_allowed") is not False:
        raise RuntimeError("post-advisor final-panel lock drift")
    if arm_authority_path is None:
        arm_env = os.environ.get("OPENEVO_POST_ADVISOR_ARM_AUTHORITY")
        arm_authority_path = Path(arm_env) if arm_env else None
    if arm_authority_path is None:
        raise RuntimeError("post-advisor formal launch requires arm authority path")
    _formal_start(activation, campaign, authority_repo, arm_authority_path)
    return activation

def validate_consumer_runtime_checks(package: dict, *, full_hash: bool=False) -> dict:
    consumer=package.get("consumer_runtime_validation") or {}
    if consumer.get("status")!="PASS" or int(consumer.get("formal_rows_consumed",-1))!=0 or int(consumer.get("final_panel_access",-1))!=0:
        raise RuntimeError("READY-v2 consumer runtime validation missing or invalid")  # legacy-revision-compat: absorbed historical runtime literal from a6f67b
    for key,check in (consumer.get("checks") or {}).items():
        kind=check.get("kind")
        if kind=="file":
            target=Path(str(check.get("path") or ""))
            if not target.is_file() or check.get("sha256")!=sha(target): raise RuntimeError(f"consumer runtime binding drift: {key}")
        elif kind=="directory":
            target=Path(str(check.get("path") or ""))
            if not target.is_dir(): raise RuntimeError(f"consumer runtime directory missing: {key}")
        elif kind=="hash_set":
            root=Path(str(check.get("seed_root") or "")); hashes=check.get("hashes") or {}
            if not root.is_dir() or not hashes: raise RuntimeError(f"consumer runtime hash-set missing: {key}")
            for name,expected in hashes.items():
                fp=root/name
                if not fp.is_file() or sha(fp)!=expected: raise RuntimeError(f"consumer runtime hash-set drift: {key}:{name}")
        elif kind=="file_manifest":
            root=Path(str(check.get("root") or "")); verification=Path(str(check.get("verification_receipt_path") or "")); merge=Path(str(check.get("merge_receipt_path") or ""))
            if not root.is_dir() or not verification.is_file() or not merge.is_file(): raise RuntimeError(f"consumer file-manifest root/receipt missing: {key}")
            if sha(verification)!=check.get("verification_receipt_sha256") or sha(merge)!=check.get("merge_receipt_sha256"): raise RuntimeError(f"consumer file-manifest receipt drift: {key}")
            for row in check.get("files") or []:
                fp=root/str(row.get("name") or "")
                if not fp.is_file() or fp.stat().st_size!=int(row.get("bytes",-1)): raise RuntimeError(f"consumer file-manifest file missing/size drift: {key}:{row.get('name')}")
                st=fp.stat(); metadata_same=(st.st_mtime_ns==int(row.get("mtime_ns",-1)) and st.st_ino==int(row.get("inode",-1)))
                if full_hash or not metadata_same:
                    if sha(fp)!=row.get("sha256"): raise RuntimeError(f"consumer file-manifest SHA drift: {key}:{row.get('name')}")
        else:
            raise RuntimeError(f"consumer runtime binding kind invalid: {key}")
    return consumer

def validate_campaign_authority(package: dict, repo: Path) -> dict:
    ref=package.get("campaign_authority") or {}
    campaign_path=repo/str(ref.get("campaign_path") or "")
    if ref.get("campaign_path")!=CURRENT_CAMPAIGN or not campaign_path.is_file() or ref.get("campaign_sha256")!=sha(campaign_path):
        raise RuntimeError("formal campaign authority missing or drifted")
    campaign=load(campaign_path)
    if campaign.get("campaign_id")!=ref.get("campaign_id") or campaign.get("formal_task_consumption_allowed") is not True or campaign.get("gpu_allocation_allowed") is not True:
        raise RuntimeError("formal campaign is not execution-ready")
    lane=package.get("formal_gpu_lane_successor") or {}
    lane_path=repo/str(lane.get("path") or "")
    if not lane_path.is_file() or lane.get("sha256")!=sha(lane_path):
        raise RuntimeError("formal GPU4-7 lane successor missing or drifted")
    lane_doc=load(lane_path)
    if campaign.get("authorized_gpu_uuids")!=lane_doc.get("physical_gpu_uuids") or (campaign.get("resource_successor") or {}).get("sha256")!=lane.get("sha256"):
        raise RuntimeError("formal campaign GPU4-7 lane binding drift")
    binding=campaign.get("experiment_identity") or {}
    if binding.get("experiment_id")!=ref.get("experiment_id") or binding.get("passport_sha256")!=ref.get("passport_sha256"):
        raise RuntimeError("formal campaign Passport binding drift")
    gate=experiment_passport_gate.validate_binding(repo,binding)
    registry,router=load_current_registry(repo); entry=resolve_experiment(registry,binding["experiment_id"])
    passport_path=repo/entry["passport_path"]
    if ref.get("registry_path")!=router["path"] or ref.get("registry_sha256")!=router["sha256"]:
        raise RuntimeError("formal campaign Registry binding drift")
    if ref.get("passport_path")!=entry["passport_path"] or not passport_path.is_file() or ref.get("passport_file_sha256")!=sha(passport_path):
        raise RuntimeError("formal campaign Passport file drift")
    if gate.get("passport_status")!="FROZEN":
        raise RuntimeError("formal campaign Passport is not FROZEN")
    return {"campaign":campaign,"passport_gate":gate,"registry":router,"entry":entry}

def validate_ready(path: Path, repo: Path) -> dict:
    package = load(path)
    if package.get("schema") != READY_SCHEMA or package.get("status") != "FORMAL_READY_AWAITING_OWNER_LAUNCH":
        raise RuntimeError("formal ready package status/schema mismatch")
    boundary = package.get("execution_boundary") or {}
    if boundary.get("formal_task_consumption_allowed") is not False or boundary.get("activation_authority") is not False:
        raise RuntimeError("ready package itself must not authorize formal tasks")
    if int(boundary.get("formal_rows_consumed", -1)) != 0 or int(boundary.get("final_panel_access", -1)) != 0:
        raise RuntimeError("ready package protected counters are not zero")
    for rel, expected in (package.get("repository_file_sha256") or {}).items():
        target = repo / rel
        if not target.is_file() or sha(target) != expected:
            raise RuntimeError(f"ready package repository file drift: {rel}")
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, capture_output=True, check=True).stdout.strip()
    if package.get("execution_git_sha") != head:
        raise RuntimeError("ready package execution checkout HEAD drift")
    for key in ("analysis_prereg", "resource_execution_contract", "formal_gpu_lane_successor", "authority_contract", "carrier_contract_fail_noop_amendment", "step_seed_formal_binding", "gdr_step_trace_qualification"):
        ref = package.get(key) or {}
        target = repo / str(ref.get("path") or "")
        if not target.is_file() or ref.get("sha256") != sha(target):
            raise RuntimeError(f"ready package {key} missing or drifted")
    trace=load(repo / package["gdr_step_trace_qualification"]["path"])
    if trace.get("status")!="PASS_EFFECTIVE_STATE_GDR_STEP_TRACE_EXACT_EQUIVALENCE": raise RuntimeError("GDR step-trace qualification is not PASS")
    tb=trace.get("boundary") or {}
    if int(tb.get("formal_rows_consumed",-1))!=0 or int(tb.get("final_panel_access",-1))!=0: raise RuntimeError("GDR step-trace protected-counter drift")
    from scripts.openevo_webshop.bounded_effective_state_recovery_202609282338 import validate_manifest
    validate_manifest(package)
    pref=package.get("post_rollout_gpu_priority") or {}
    prefpath=repo/str(pref.get("path") or "")
    if not prefpath.is_file() or sha(prefpath)!=pref.get("sha256"): raise RuntimeError("post-rollout GPU priority identity drift")
    d=load(prefpath)
    expected=((package.get("execution_policy") or {}).get("post_rollout_gpu_priority") or {"off":[7,6,5,4],"on":[4,5,6,7]})
    if d.get("off")!=expected.get("off") or d.get("on")!=expected.get("on"): raise RuntimeError("post-rollout GPU priority semantics drift")
    validate_consumer_runtime_checks(package, full_hash=False)
    validate_campaign_authority(package, repo)
    authority = load(repo / package["authority_contract"]["path"])
    if authority.get("status") != "REQUIRED_BINDING_FOR_NEXT_READY_PACKAGE_AND_OWNER_LAUNCH_RELEASE":
        raise RuntimeError("prelaunch authority contract is not active")
    return package

def validate_launch_release(path: Path, ready_path: Path, package: dict, repo: Path) -> dict:
    release = load(path)
    if release.get("schema") != LAUNCH_SCHEMA or release.get("status") != "AUTHORIZED_FORMAL_LAUNCH":
        raise RuntimeError("owner launch release is not authorized")
    if release.get("ready_package_sha256") != sha(ready_path):
        raise RuntimeError("owner launch release ready-package binding drift")
    if release.get("execution_git_sha") != package.get("execution_git_sha"):
        raise RuntimeError("owner launch release execution SHA drift")
    if release.get("formal_task_consumption_allowed") is not True:
        raise RuntimeError("owner launch release does not authorize formal task consumption")
    if int(release.get("final_panel_access", -1)) != 0:
        raise RuntimeError("owner launch release may not open final panel")
    required_sha = {
        "analysis_prereg_sha256": package["analysis_prereg"]["sha256"],
        "resource_execution_contract_sha256": package["resource_execution_contract"]["sha256"],
        "authority_contract_sha256": package["authority_contract"]["sha256"],
        "carrier_contract_fail_noop_amendment_sha256": package["carrier_contract_fail_noop_amendment"]["sha256"],
        "step_seed_formal_binding_sha256": package["step_seed_formal_binding"]["sha256"],
        "gdr_step_trace_qualification_sha256": package["gdr_step_trace_qualification"]["sha256"],
        "coresident_rollout_resource_successor_sha256": package["coresident_rollout_resource_successor"]["sha256"],
        "formal_gpu_lane_successor_sha256": package["formal_gpu_lane_successor"]["sha256"],
        "campaign_sha256": package["campaign_authority"]["campaign_sha256"],
        "passport_sha256": package["campaign_authority"]["passport_sha256"],
        "registry_sha256": package["campaign_authority"]["registry_sha256"],
    }
    for key, expected in required_sha.items():
        if release.get(key) != expected:
            raise RuntimeError(f"owner launch release {key} drift")
    for key in ("campaign_id","experiment_id"):
        if release.get(key)!=package["campaign_authority"][key]:
            raise RuntimeError(f"owner launch release {key} drift")
    authority = load(repo / package["authority_contract"]["path"])
    expected_heads = {
        "control_tower_head": authority["control_tower"]["head"],
        "science_receipt_head": authority["science_receipt"]["head"],
        "binding_head": authority["binding"]["head"],
        "resource_head": authority["resource"]["head"],
        "receipt_only_closeout_sha256": authority["science_receipt"]["receipt_only_closeout_sha256"],
    }
    for key, expected in expected_heads.items():
        if release.get(key) != expected:
            raise RuntimeError(f"owner launch release {key} does not match frozen authority contract")
    if release.get("recovery_amendment_sha256") != (package.get("recovery") or {}).get("sha256"): raise RuntimeError("launch recovery identity drift")
    if release.get("post_rollout_gpu_priority_sha256") != (package.get("post_rollout_gpu_priority") or {}).get("sha256"): raise RuntimeError("launch post-rollout GPU identity drift")
    return release

def validate_launch_pair(repo: Path, ready_path: Path, release_path: Path) -> tuple[dict, dict]:
    package = validate_ready(ready_path, repo)
    release = validate_launch_release(release_path, ready_path, package, repo)
    return package, release
