#!/usr/bin/env python3
"""Fail-closed, one-use authorization for the Q08 seen-cell route."""
from __future__ import annotations
import hashlib, json, os, re, time, stat
import subprocess
import datetime
from pathlib import Path

PROFILE = 'configs/experiment/profiles/post-advisor-q08-seen-retention-profile-202610010742.json'
RELEASE = 'configs/experiment/activations/post-advisor-q08-seen-retention-release-202610010742.json'
ROUTER = 'configs/experiment/activations/post-advisor-q08-seen-retention-router-202610010742.json'
CURRENT = 'configs/experiment/candidates/post-advisor-q08-seen-retention-current-202610010742.json'
PANEL_SHA = "0780dbe68dcc1cafb013a8765f2ecb4b91bfd2a0d92ac3ac9fe4fde7453fccc3"
MATRIX_SHA = "93ad7718a07633e38e15197f43ea21649fac84abc650aa29fad7f3107415f89d"
REUSE_SHA = "45f5107fcd88661ab9f718392195dd28c969a0aa7a02969f352e2d93719c8f2d"
WRAPPER_SHA = "07d01c6b31240dac4a62bddfe54f34ab9a0e636ca893f1d8e5994e95d0853d2c"
CELL_IDS = ("ordinary-r080", "ordinary-r160", "bounded-r040", "bounded-r080", "bounded-r120", "bounded-r160", "gdr-r040", "gdr-r080", "gdr-r120", "gdr-r160")
MISSING = ("ordinary-r040", "ordinary-r120")

class Q08AuthorizationError(RuntimeError):
    pass

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20), b""): h.update(block)
    return h.hexdigest()

def read_json(path: Path) -> dict:
    value=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise Q08AuthorizationError(f"expected JSON object: {path}")
    return value

def expect_file(path: Path, digest: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", str(digest)) or sha256_file(path) != digest:
        raise Q08AuthorizationError(f"SHA-256 mismatch: {path}")

def _normalize_sha(value: object) -> str:
    if not isinstance(value, str):
        raise Q08AuthorizationError("SHA-256 reference must be text")
    normalized=value.removeprefix("sha256:")
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise Q08AuthorizationError("SHA-256 reference is malformed")
    return normalized

RECOVERY_RECEIPT_REL = "configs/experiment/activations/post-advisor-q08-seen-retention-engineering-recovery-202610010931.json"
RECOVERY_CONTROL_REL = "control/engineering-recovery-attempt-0002"
RECOVERY3_RECEIPT_REL = "configs/experiment/activations/post-advisor-q08-seen-retention-engineering-recovery-202610011031.json"
RECOVERY3_CONTROL_REL = "control/engineering-recovery-attempt-0003"
RECOVERY4_RECEIPT_REL = "configs/experiment/activations/post-advisor-q08-seen-retention-engineering-recovery-202610011134.json"
RECOVERY4_CONTROL_REL = "control/engineering-recovery-attempt-0004"
RECOVERY4_PRIOR_AUTHORITY_SHA = "ef2c9f8d2281c29f5031eac8a59f2a24d3050a33"
RECOVERY4_PRIOR_SOURCE_SHA = "448c5c885ee456fc65ef7c28f158fb00a9972d4b"
RECOVERY4_PREDECESSOR_SHA = "2107d41d59d892f1d5020177da860b1c822afb32a8b29b0d8a57c6eb5a3da494"
RECOVERY4_FORMAL_START_BASENAME = "formal-start-attempt-0002.json"
RECOVERY4_ARCHIVE_REL = "control/engineering-recovery-attempt-0004/attempt-0003-archive"
RECOVERY5_RECEIPT_REL = "configs/experiment/activations/post-advisor-q08-seen-retention-engineering-recovery-202610011216.json"
RECOVERY5_PRIOR_AUTHORITY_SHA = "11724803ffad96fab86e135b0122d85b16a83179"
RECOVERY5_PRIOR_SOURCE_SHA = "a402ad6f75bc3afd335e86782dc2cd7f70b4a10a"
RECOVERY5_PREDECESSOR_SHA = "fceed7664d987cca5efb342246404cff56c55fa06660f1957eaba0c02439de48"
RECOVERY5_PRIOR_ADMISSION_SHA = "eff5c191c3db2ef538eaab5eec3ec435642d023d4b1edfbad266ac3fe1856abe"
RECOVERY5_FORMAL_START_BASENAME = "formal-start-attempt-0003.json"
RECOVERY5_ARCHIVE_REL = "control/engineering-recovery-attempt-0005/attempt-0004-archive"
RECOVERY6_RECEIPT_REL = "configs/experiment/activations/post-advisor-q08-seen-retention-engineering-recovery-202610011316.json"
RECOVERY6_CONTROL_REL = "control/engineering-recovery-attempt-0006"
RECOVERY6_PRIOR_AUTHORITY_SHA = "84cb1af709e99876eda8a239d44ec39b4e18574f"
RECOVERY6_PRIOR_SOURCE_SHA = "ccd984028ae193922b93c1e3afc5b17677cbdee7"
RECOVERY6_PREDECESSOR_SHA = "a47d2ea54b1090f035fb82e7d0e69d0dfb14fd4e3b87e341c71dc82284f299b7"
RECOVERY6_PRIOR_ADMISSION_SHA = "476acea012ba5305f1d127d30ced0e8e8f5e1b23a44062ef440e9097032b76b8"
RECOVERY6_FORMAL_START_BASENAME = "formal-start-attempt-0004.json"
RECOVERY6_ARCHIVE_REL = "control/engineering-recovery-attempt-0006/attempt-0005-archive"
RECOVERY7_RECEIPT_REL = "configs/experiment/activations/post-advisor-q08-seen-retention-engineering-recovery-202610011420.json"
RECOVERY7_CONTROL_REL = "control/engineering-recovery-attempt-0007"
RECOVERY7_PRIOR_AUTHORITY_SHA = "f4d9dad2bf5b1ed0454c669890f0a871199cfd7b"
RECOVERY7_PRIOR_SOURCE_SHA = "154f8f72016f129674d2ff82f948d5a382744f4a"
RECOVERY7_PREDECESSOR_SHA = "8fcbae6235b1b04281b7df5bb82ac42c24b00d4694c1393ea907714a6b826f97"
RECOVERY7_FORMAL_START_BASENAME = "formal-start-attempt-0005.json"
RECOVERY7_ARCHIVE_REL = "control/engineering-recovery-attempt-0007/attempt-0006-archive"
RECOVERY8_RECEIPT_REL = "configs/experiment/activations/post-advisor-q08-seen-retention-engineering-recovery-202610021130.json"
RECOVERY8_CONTROL_REL = "control/engineering-recovery-attempt-0008"
RECOVERY8_PRIOR_AUTHORITY_SHA = "65eb2005329f1e510d14d8c940782b9267ce1559"
RECOVERY8_PRIOR_SOURCE_SHA = "f1e52cbe3b16add70bab8452996a5ef00e393c3a"
RECOVERY8_PREDECESSOR_SHA = "7a135c4a8588f8f7920cbf7fc3913cbda9f3871917bbc4ce58dad41af5aa1e2d"
RECOVERY8_FORMAL_START_BASENAME = "formal-start-attempt-0006.json"
RECOVERY8_ARCHIVE_REL = "control/engineering-recovery-attempt-0008/attempt-0007-archive-corrected-02"
RECOVERY9_RECEIPT_REL = "configs/experiment/activations/post-advisor-q08-seen-retention-engineering-recovery-202610021345.json"
RECOVERY9_PREDECESSOR_SHA = "cf20e344210feb721bf4007f935005c6c9b1957df309878435314e6bd4e937cd"
RECOVERY9_PRIOR_AUTHORITY_SHA = "adc0452ee85f5717f6cecc1a1b9bdb022587c264"
RECOVERY9_PRIOR_SOURCE_SHA = "1ddbc6cb4b76a214ad8cb6127d16afd55353cf01"
RECOVERY9_CONTROL_REL = "control/engineering-recovery-attempt-0009"
RECOVERY9_FORMAL_START_BASENAME = "formal-start-attempt-0007.json"
RECOVERY10_RECEIPT_REL = "configs/experiment/activations/post-advisor-q08-seen-retention-engineering-recovery-202610021900.json"
RECOVERY10_CONTROL_REL = "control/engineering-recovery-attempt-0010"
RECOVERY10_PRIOR_AUTHORITY_SHA = "285cc9656cfde8f91a13e97350ec2b435e34ae1c"
RECOVERY10_PRIOR_SOURCE_SHA = "8e23e87850befce550668d583e7fc8310597cf25"
RECOVERY10_PREDECESSOR_SHA = "fc14e0bb9bf6e80bd39293d893cfef8ec59947164f60cfed2121ddb744e72fa9"
RECOVERY10_FORMAL_START_BASENAME = "formal-start-attempt-0008.json"
RECOVERY10_ARCHIVE_REL = "control/engineering-recovery-attempt-0010/attempt-0007-router-path-validation-failure-archive"
RECOVERY11_RECEIPT_REL = "configs/experiment/activations/post-advisor-q08-seen-retention-engineering-recovery-202610022100.json"
RECOVERY11_CONTROL_REL = "control/engineering-recovery-attempt-0011"
RECOVERY11_PRIOR_AUTHORITY_SHA = "454137413aed48cbf600a4bbcb261b71ba30bea1"
RECOVERY11_PRIOR_SOURCE_SHA = "53fe9f717c5d2112e850fc7071b6a6dddea92d3d"
RECOVERY11_PREDECESSOR_SHA = "3dc6aa6620a02a671b50909e2da57f2a2f9d46864fba21c4f6e21f1e756b1fb7"
RECOVERY11_FORMAL_START_BASENAME = "formal-start-attempt-0009.json"
RECOVERY11_ARCHIVE_REL = "control/engineering-recovery-attempt-0011/attempt-0008-probe-import-failure-archive"
RECOVERY12_RECEIPT_REL = "configs/experiment/activations/post-advisor-q08-seen-retention-engineering-recovery-202610022230.json"
RECOVERY12_CONTROL_REL = "control/engineering-recovery-attempt-0012"
RECOVERY12_PRIOR_AUTHORITY_SHA = "1c560def3648a115f405f2316c2641c1c03c2af4"
RECOVERY12_PRIOR_SOURCE_SHA = "368e963b7c3d435e69d57719bdce96381081180c"
RECOVERY12_PREDECESSOR_SHA = "fd2ed48a34dae5ae8ad78801066ed18f6ff2907f542177852ff2caa29a3270f3"
RECOVERY12_FORMAL_START_BASENAME = "formal-start-attempt-0010.json"
RECOVERY12_ARCHIVE_REL = "control/engineering-recovery-attempt-0012/attempt-0009-q26-permission-failure-archive"



def resolve_engineering_recovery_document(repo: Path, release: dict) -> dict | None:
    """Load the committed document named by the release descriptor.

    The descriptor is provenance (path, hash, attempt); continuation semantics
    belong to the authenticated document. This lightweight resolver is safe at
    point of use after admission and deliberately does not rerun admission.
    """
    ref = release.get("engineering_recovery")
    if ref is None:
        return None
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256", "attempt", "prior_authority_git_sha"}:
        raise Q08AuthorizationError("engineering recovery reference has an invalid shape")
    if not isinstance(ref.get("attempt"), int) or ref.get("attempt") < 6:
        return None
    path = bound_path(repo, str(ref["path"]))
    expect_file(path, str(ref["sha256"]))
    committed = subprocess.run(["git", "-C", str(repo), "show", f"HEAD:{ref['path']}"],
                               check=True, capture_output=True).stdout
    raw = path.read_bytes()
    if committed != raw:
        raise Q08AuthorizationError("engineering recovery document differs from committed authority bytes")
    recovery = read_json(path)
    continuation = recovery.get("continuation") or {}
    expected_namespace = f"control/engineering-recovery-attempt-{ref['attempt']:04d}/cells"
    if (recovery.get("attempt") != ref["attempt"]
            or recovery.get("status") != "ZERO_CONSUMPTION_RECOVERY_PASS"
            or continuation.get("context_namespace") != expected_namespace
            or not isinstance(continuation.get("control_attempt"), int)):
        raise Q08AuthorizationError("engineering recovery continuation document is malformed")
    return recovery


def expected_context_namespace(run_root: Path, recovery: dict | None, control_attempt: int) -> Path:
    """Resolve the exact per-attempt namespace from the authenticated document."""
    namespace = Path("control/cells")
    if isinstance(recovery, dict):
        recovery_attempt = int(recovery["attempt"])
        continuation = recovery.get("continuation") or {}
        expected_namespace = f"control/engineering-recovery-attempt-{recovery_attempt:04d}/cells"
        if continuation.get("control_attempt") != control_attempt:
            raise Q08AuthorizationError(f"recovery{recovery_attempt} FormalStart control attempt mismatch")
        if continuation.get("context_namespace") != expected_namespace:
            raise Q08AuthorizationError(f"recovery{recovery_attempt} context namespace is not the exact isolated path")
        namespace = Path(continuation["context_namespace"])
    if namespace.is_absolute() or ".." in namespace.parts:
        raise Q08AuthorizationError("Q08 context namespace escapes the fixed run root")
    return run_root / namespace


def validate_engineering_recovery(repo: Path, profile: dict, release: dict) -> dict | None:
    """Admit exactly one pre-cell control continuation, never a cell resume."""
    ref = release.get("engineering_recovery")
    if ref is None:
        return None
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256", "attempt", "prior_authority_git_sha"}:
        raise Q08AuthorizationError("engineering recovery reference has an invalid shape")
    if ref.get("attempt") == 7:
        if (ref.get("path") != RECOVERY7_RECEIPT_REL
                or ref.get("prior_authority_git_sha") != RECOVERY7_PRIOR_AUTHORITY_SHA):
            raise Q08AuthorizationError("engineering recovery attempt-0007 authority/path drift")
        return _validate_engineering_recovery_attempt7(repo, profile, release, ref)
    if ref.get("attempt") == 8:
        if (ref.get("path") != RECOVERY8_RECEIPT_REL
                or ref.get("prior_authority_git_sha") != RECOVERY8_PRIOR_AUTHORITY_SHA):
            raise Q08AuthorizationError("engineering recovery attempt-0008 authority/path drift")
        return _validate_engineering_recovery_attempt8(repo, profile, release, ref)
    if ref.get("attempt") == 9:
        if (ref.get("path") != RECOVERY9_RECEIPT_REL
                or ref.get("prior_authority_git_sha") != RECOVERY9_PRIOR_AUTHORITY_SHA):
            raise Q08AuthorizationError("engineering recovery attempt-0009 authority/path drift")
        return _validate_engineering_recovery_attempt9(repo, profile, release, ref)
    if ref.get("attempt") == 10:
        if (ref.get("path") != RECOVERY10_RECEIPT_REL
                or ref.get("prior_authority_git_sha") != RECOVERY10_PRIOR_AUTHORITY_SHA):
            raise Q08AuthorizationError("engineering recovery attempt-0010 authority/path drift")
        return _validate_engineering_recovery_attempt10(repo, profile, release, ref)
    if ref.get("attempt") == 12:
        if (ref.get("path") != RECOVERY12_RECEIPT_REL
                or ref.get("prior_authority_git_sha") != RECOVERY12_PRIOR_AUTHORITY_SHA):
            raise Q08AuthorizationError("engineering recovery attempt-0012 authority/path drift")
        return _validate_engineering_recovery_attempt12(repo, profile, release, ref)
    if ref.get("attempt") == 11:
        if (ref.get("path") != RECOVERY11_RECEIPT_REL
                or ref.get("prior_authority_git_sha") != RECOVERY11_PRIOR_AUTHORITY_SHA):
            raise Q08AuthorizationError("engineering recovery attempt-0011 authority/path drift")
        return _validate_engineering_recovery_attempt11(repo, profile, release, ref)
    if ref.get("attempt") == 6:
        if (ref.get("path") != RECOVERY6_RECEIPT_REL
                or ref.get("prior_authority_git_sha") != RECOVERY6_PRIOR_AUTHORITY_SHA):
            raise Q08AuthorizationError("engineering recovery attempt-0006 authority/path drift")
        return _validate_engineering_recovery_attempt6(repo, profile, release, ref)
    if ref.get("attempt") == 5:
        if (ref.get("path") != RECOVERY5_RECEIPT_REL
                or ref.get("prior_authority_git_sha") != RECOVERY5_PRIOR_AUTHORITY_SHA):
            raise Q08AuthorizationError("engineering recovery attempt-0005 authority/path drift")
        return _validate_engineering_recovery_attempt5(repo, profile, release, ref)
    if ref.get("attempt") == 4:
        if (ref.get("path") != RECOVERY4_RECEIPT_REL
                or ref.get("prior_authority_git_sha") != RECOVERY4_PRIOR_AUTHORITY_SHA):
            raise Q08AuthorizationError("engineering recovery attempt-0004 authority/path drift")
        return _validate_engineering_recovery_attempt4(repo, profile, release, ref)
    if ref.get("attempt") == 3:
        if (ref.get("path") != RECOVERY3_RECEIPT_REL
                or ref.get("prior_authority_git_sha") != "9bcecf42ecfd9ad9f3730d35ff0061a73e77e99f"):
            raise Q08AuthorizationError("engineering recovery attempt-0003 authority/path drift")
        return _validate_engineering_recovery_attempt3(repo, profile, release, ref)
    if ref.get("path") != RECOVERY_RECEIPT_REL or ref.get("attempt") != 2:
        raise Q08AuthorizationError("engineering recovery must use the fixed attempt-0002 or approved attempt-0003 receipt")
    prior_authority = ref.get("prior_authority_git_sha")
    if prior_authority != "2376c5c14e99a75b59165ffbe2598d12647ec64e":
        raise Q08AuthorizationError("engineering recovery is not bound to the exact attempt-0001 authority")
    path = bound_path(repo, RECOVERY_RECEIPT_REL)
    expect_file(path, str(ref.get("sha256", "")))
    import subprocess
    committed = subprocess.run(["git", "-C", str(repo), "show", f"HEAD:{RECOVERY_RECEIPT_REL}"], check=True, capture_output=True).stdout
    if committed != path.read_bytes():
        raise Q08AuthorizationError("engineering recovery receipt is not the exact committed authority bytes")
    receipt = read_json(path)
    expected = {
        "schema": "openevo.q08.zero-consumption-engineering-recovery.v1",
        "status": "ZERO_CONSUMPTION_RECOVERY_PASS",
        "campaign_id": profile["campaign_id"], "run_id": profile["run_id"], "attempt": 2,
        "prior_authority_git_sha": prior_authority,
        "prior_profile_sha256": "3a9dc17c1cfb3109d0d94910d6e97ec12c41aca3e592733ef612960fa454a268",
        "prior_release_sha256": "25529694ae6198db38c845a7eeea845ddbb2ade852364e7aec43d77797b8bb86",
        "prior_router_sha256": "eed4e26f7bbd13da1fe4bf0e2cf692c5de09cbb0f250034fa2debb7a3ca76135",
        "prior_operational_source_git_sha": "e1761b3b6b1440258c17ecb180cae2b8e8035f87",
        "scientific_resume": False, "formal_task_consumed": 0, "task_bearing_calls": 0,
        "model_calls": 0, "episodes_consumed": 0, "cell_claims_consumed": 0,
        "final_panel_access": 0, "fresh_seen_episodes_max": 960,
        "reused_historical_episodes": 768, "program_total_maximum_including_reuse": 1920,
        "continuation": {"control_attempt": 2, "formal_start_basename": RECOVERY4_FORMAL_START_BASENAME,
                         "cell_claims_allowed": 10, "cell_resume_allowed": False,
                         "run_id_changed": False, "matrix_changed": False, "panel_changed": False,
                         "budget_changed": False},
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise Q08AuthorizationError(f"engineering recovery mismatch: {key}")
    frozen = profile["frozen_inputs"]
    expected_science = {
        "panel_sha256": PANEL_SHA, "matrix_sha256": MATRIX_SHA, "reuse_audit_sha256": REUSE_SHA,
        "probe_wrapper_sha256": WRAPPER_SHA, "model": frozen["model"],
        "model_revision": frozen["model_revision"], "seed_root": frozen["seed_root"],
        "memory_sha256": frozen["carrier_sha256"]["memory"],
        "skill_sha256": frozen["carrier_sha256"]["skill"],
        "agent_sha256": frozen["carrier_sha256"]["agent"],
    }
    if receipt.get("science_identity") != expected_science:
        raise Q08AuthorizationError("engineering recovery scientific identity changed")
    archive = receipt.get("attempt_0001_archive")
    names = {"launch_receipt": "launch-receipt.json", "gpu_resource_receipt": "gpu-resource-receipt.json",
             "formal_start": "formal-start-attempt-0001.json", "ready_receipt": "ready-receipt-attempt-0001.json"}
    if not isinstance(archive, dict) or set(archive) != set(names):
        raise Q08AuthorizationError("engineering recovery lacks complete attempt-0001 evidence archive")
    run_root = Path(profile["workspace"]) / profile["paths"]["runs"] / profile["run_id"]
    archive_root = run_root / RECOVERY_CONTROL_REL / "attempt-0001-archive"
    runs_root = Path(profile["workspace"]) / profile["paths"]["runs"]
    expected_sources = {
        "launch_receipt": run_root / "launch-receipt.json",
        "gpu_resource_receipt": run_root / "gpu-resource-receipt.json",
        "formal_start": runs_root / f"{profile['run_id']}-readiness-r4/control/formal-start-attempt-0001.json",
        "ready_receipt": runs_root / f"{profile['run_id']}-readiness-r4/control/execution-readiness-65e9b0b18fbd41a088cf70de7d4a4244.json",
    }
    for key, basename in names.items():
        item = archive[key]
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "source_path"}:
            raise Q08AuthorizationError(f"engineering recovery {key} reference is malformed")
        expected_path = archive_root / basename
        if (Path(item["path"]).resolve(strict=True) != expected_path.resolve()
                or Path(item["source_path"]).resolve() != expected_sources[key].resolve()):
            raise Q08AuthorizationError(f"engineering recovery {key} is outside the fixed attempt-0001 provenance")
        expect_file(expected_path, item["sha256"])
    launch = read_json(Path(archive["launch_receipt"]["path"]))
    resource = read_json(Path(archive["gpu_resource_receipt"]["path"]))
    start = read_json(Path(archive["formal_start"]["path"]))
    ready = read_json(Path(archive["ready_receipt"]["path"]))
    inventory = receipt.get("run_root_inventory_before_recovery")
    expected_inventory = sorted([
        {"path": str(run_root / "launch-receipt.json"), "sha256": archive["launch_receipt"]["sha256"]},
        {"path": str(run_root / "gpu-resource-receipt.json"), "sha256": archive["gpu_resource_receipt"]["sha256"]},
    ], key=lambda item: item["path"])
    if inventory != {"files": expected_inventory, "cells_directory_present": False,
                    "cell_claims_directory_present": False, "fixed_output_leaves_present": 0}:
        raise Q08AuthorizationError("attempt-0001 run-root inventory does not prove zero science consumption")
    control_dir = runs_root / f"{profile['run_id']}-readiness-r4" / "control"
    control_inventory = receipt.get("attempt_0001_control_inventory")
    actual_control_files = sorted(
        [{"path": str(item.resolve()), "sha256": sha256_file(item)} for item in control_dir.iterdir()
         if item.is_file() and not item.is_symlink()], key=lambda item: item["path"])
    if (not isinstance(control_inventory, list) or not control_inventory
            or control_inventory != actual_control_files
            or any(Path(item["path"]).parent.resolve() != control_dir.resolve() for item in control_inventory)):
        raise Q08AuthorizationError("attempt-0001 readiness control evidence inventory drifted")
    old_profile = Path("/workspace/control/post-advisor-q08-seen-authority-b-20261001/configs/experiment/profiles/post-advisor-q08-seen-retention-profile-202610010742.json")
    old_release = Path("/workspace/control/post-advisor-q08-seen-authority-b-20261001/configs/experiment/activations/post-advisor-q08-seen-retention-release-202610010742.json")
    old_router = Path("/workspace/control/post-advisor-q08-seen-authority-b-20261001/configs/experiment/activations/post-advisor-q08-seen-retention-router-202610010742.json")
    ready_profile = ready.get("profile") or {}
    ready_release = ready.get("release_receipt") or {}
    ready_authority = ready.get("authority") or {}
    if (ready_profile.get("path") != str(old_profile)
            or _normalize_sha(ready_profile.get("sha256")) != "3a9dc17c1cfb3109d0d94910d6e97ec12c41aca3e592733ef612960fa454a268"
            or ready_release.get("path") != str(old_release)
            or _normalize_sha(ready_release.get("sha256")) != "25529694ae6198db38c845a7eeea845ddbb2ade852364e7aec43d77797b8bb86"
            or ready_authority.get("router_path") != str(old_router)
            or ready_authority.get("campaign_id") != profile["campaign_id"]):
        raise Q08AuthorizationError("attempt-0001 READY does not bind its exact predecessor authority")
    start_ref = launch.get("formal_start") or {}
    resource_ref = launch.get("gpu_resource_receipt") or {}
    resource_accounting = resource_ref.get("accounting") or {}
    selected_gpu = launch.get("selected_gpu_uuids")
    if (launch.get("run_id") != profile["run_id"]
            or launch.get("execution_mode") != "host-orchestrator"
            or not isinstance(selected_gpu, list) or len(selected_gpu) != 1
            or selected_gpu[0] not in profile["gpu_gate"]["authorized_uuids"]
            or launch.get("image_id") != profile["image_id"] or launch.get("exit_code") != 1
            or Path(str(start_ref.get("path", ""))).resolve() != Path(archive["formal_start"]["source_path"]).resolve()
            or _normalize_sha(start_ref.get("sha256")) != archive["formal_start"]["sha256"]
            or Path(str(resource_ref.get("path", ""))).resolve() != Path(archive["gpu_resource_receipt"]["source_path"]).resolve()
            or _normalize_sha(resource_ref.get("sha256")) != archive["gpu_resource_receipt"]["sha256"]
            or resource_accounting.get("status") != "exact" or resource_accounting.get("return_code") != 1
            or resource.get("campaign_id") != profile["campaign_id"]
            or resource.get("status") != "failed" or resource.get("return_code") != 1):
        raise Q08AuthorizationError("attempt-0001 launch/resource evidence is not an exact pre-cell failure")
    ready_ref = (start.get("readiness") or {}).get("ready_receipt") or {}
    if (Path(str(start_ready.get("path", ""))).resolve() != paths["ready"]
            or _normalize_sha(start_ready.get("sha256")) != sha256_file(paths["ready"])
            or Path(str(start_prelaunch.get("path", ""))).resolve() != paths["prelaunch"]
            or _normalize_sha(start_prelaunch.get("sha256")) != sha256_file(paths["prelaunch"])
            or start.get("status") != "FORMAL_START_COMMITTED" or start.get("run_id") != profile["run_id"]
            or start.get("formal_task_consumption_before") is not False
            or Path(str(ready_ref.get("path", ""))).resolve() != Path(archive["ready_receipt"]["source_path"]).resolve()
            or _normalize_sha(ready_ref.get("sha256")) != archive["ready_receipt"]["sha256"]
            or ready.get("verdict") != "READY"):
        raise Q08AuthorizationError("attempt-0001 FormalStart/READY evidence is invalid")
    if receipt.get("failure") != {
        "owner": "q08_seen_host_orchestrator._load_formal_start",
        "message": "READY receipt did not qualify the active exact Q08 router/release",
        "stage": "before-first-cell-claim",
    }:
        raise Q08AuthorizationError("recovery does not bind the exact recorded pre-cell owner failure")
    claims = run_root / "control" / "claims"
    cells = run_root / "cells"
    if claims.exists() or cells.exists():
        raise Q08AuthorizationError("engineering recovery is forbidden after any cell/claim namespace exists")
    recovery_claim = run_root / RECOVERY_CONTROL_REL / "admission-claim.json"
    if recovery_claim.exists() or recovery_claim.is_symlink():
        raise Q08AuthorizationError("the one-use engineering recovery admission was already consumed")
    for cell in CELL_IDS:
        if (run_root / "cells" / cell / "seen-eval").exists():
            raise Q08AuthorizationError("engineering recovery found a consumed fixed output leaf")
    return receipt



def _validate_engineering_recovery_attempt3(repo: Path, profile: dict, release: dict, ref: dict) -> dict:
    path=bound_path(repo,RECOVERY3_RECEIPT_REL)
    expect_file(path,str(ref.get("sha256","")))
    import subprocess
    committed=subprocess.run(["git","-C",str(repo),"show",f"HEAD:{RECOVERY3_RECEIPT_REL}"],check=True,capture_output=True).stdout
    if committed!=path.read_bytes():
        raise Q08AuthorizationError("attempt-0003 recovery receipt is not the committed exact authority bytes")
    receipt=read_json(path)
    predecessor=bound_path(repo,RECOVERY_RECEIPT_REL)
    expect_file(predecessor,"4be163c05ee69fd7f7076bdebde2f6227d0052d6975563abeb19dea89c8560f9")
    predecessor_bytes=subprocess.run(["git","-C",str(repo),"show",f"HEAD:{RECOVERY_RECEIPT_REL}"],check=True,capture_output=True).stdout
    if predecessor_bytes!=predecessor.read_bytes():
        raise Q08AuthorizationError("attempt-0002 predecessor recovery receipt is not committed exact bytes")
    expected={
        "schema":"openevo.q08.zero-consumption-engineering-recovery.v1",
        "status":"ZERO_CONSUMPTION_RECOVERY_PASS",
        "campaign_id":profile["campaign_id"],
        "run_id":profile["run_id"],
        "attempt":3,
        "prior_authority_git_sha":"9bcecf42ecfd9ad9f3730d35ff0061a73e77e99f",
        "prior_operational_source_git_sha":"89e1c90a6e6cad5011e54a31881dbd660c3396bb",
        "predecessor_recovery_sha256":"4be163c05ee69fd7f7076bdebde2f6227d0052d6975563abeb19dea89c8560f9",
        "scientific_resume":False,
        "formal_task_consumed":0,
        "task_bearing_calls":0,
        "model_calls":0,
        "episodes_consumed":0,
        "cell_claims_consumed":0,
        "final_panel_access":0,
        "fresh_seen_episodes_max":960,
        "reused_historical_episodes":768,
        "program_total_maximum_including_reuse":1920,
        "continuation":{"control_attempt":3,"formal_start_basename":"formal-start-attempt-0003.json",
            "cell_claims_allowed":10,"cell_resume_allowed":False,"run_id_changed":False,
            "matrix_changed":False,"panel_changed":False,"budget_changed":False},
        "failure":{"owner":"q08_seen_launch_guard.authorize_probe",
            "message":"PermissionError reading attempt-0002 launch context before the first cell claim",
            "stage":"before-first-cell-claim","cell_id":"ordinary-r080"},
    }
    for key,value in expected.items():
        if receipt.get(key)!=value: raise Q08AuthorizationError(f"attempt-0003 recovery mismatch: {key}")
    frozen=profile["frozen_inputs"]
    science={"panel_sha256":PANEL_SHA,"matrix_sha256":MATRIX_SHA,"reuse_audit_sha256":REUSE_SHA,
        "probe_wrapper_sha256":WRAPPER_SHA,"model":frozen["model"],"model_revision":frozen["model_revision"],
        "seed_root":frozen["seed_root"],"memory_sha256":frozen["carrier_sha256"]["memory"],
        "skill_sha256":frozen["carrier_sha256"]["skill"],"agent_sha256":frozen["carrier_sha256"]["agent"]}
    if receipt.get("science_identity")!=science:
        raise Q08AuthorizationError("attempt-0003 recovery scientific identity changed")
    run_root=Path(profile["workspace"])/profile["paths"]["runs"]/profile["run_id"]
    archive_root=run_root/RECOVERY3_CONTROL_REL/"attempt-0002-archive"
    evidence=receipt.get("attempt_0002_evidence")
    names={"formal_start":"formal-start-attempt-0002.json","ready":"ready-receipt.json",
        "qualification":"qualification.json","launch":"launch-receipt.json",
        "resource":"gpu-resource-receipt.json","launch_context":"launch-context.json",
        "container_inspect":"failed-container-inspect.json","container_stdout":"failed-container-stdout.bin",
        "container_stderr":"failed-container-stderr.bin"}
    if not isinstance(evidence,dict) or set(evidence)!=set(names):
        raise Q08AuthorizationError("attempt-0002 recovery evidence inventory is incomplete")
    for key,basename in names.items():
        item=evidence[key]; expected_path=archive_root/basename
        if not isinstance(item,dict) or set(item)!={"path","sha256"}:
            raise Q08AuthorizationError(f"attempt-0002 recovery evidence {key} reference is malformed")
        if Path(item["path"]).resolve(strict=True)!=expected_path.resolve():
            raise Q08AuthorizationError(f"attempt-0002 recovery evidence {key} path drift")
        expect_file(expected_path,item["sha256"])
    start=read_json(Path(evidence["formal_start"]["path"]))
    ready=read_json(Path(evidence["ready"]["path"]))
    qualification=read_json(Path(evidence["qualification"]["path"]))
    launch=read_json(Path(evidence["launch"]["path"]))
    resource=read_json(Path(evidence["resource"]["path"]))
    context=read_json(Path(evidence["launch_context"]["path"]))
    inspect=json.loads(Path(evidence["container_inspect"]["path"]).read_text())[0]
    stderr=Path(evidence["container_stderr"]["path"]).read_bytes().decode(errors="replace")
    prior_claim=run_root/RECOVERY_CONTROL_REL/"admission-claim.json"
    prior_claim_ref=receipt.get("prior_admission_claim") or {}
    if (prior_claim_ref.get("path")!=str(prior_claim) or prior_claim_ref.get("sha256")!=sha256_file(prior_claim)
            or read_json(prior_claim).get("status")!="CONSUMED_FOR_CONTROL_ATTEMPT_0002"
            or read_json(prior_claim).get("scientific_resume") is not False):
        raise Q08AuthorizationError("attempt-0002 predecessor admission claim is not preserved/hash-bound")
    launch_resource=launch.get("gpu_resource_receipt") or {}
    resource_accounting=launch_resource.get("accounting") or {}
    expected_resource_path=run_root/"gpu-resource-receipt.json"
    if (Path(str(launch_resource.get("path",""))).resolve()!=expected_resource_path.resolve()
            or _normalize_sha(launch_resource.get("sha256"))!=evidence["resource"]["sha256"]
            or resource_accounting.get("status")!="exact" or resource_accounting.get("return_code")!=1
            or resource_accounting.get("pre_gpu_failure") is not False
            or resource_accounting.get("gpu_hours")!=resource.get("gpu_hours")):
        raise Q08AuthorizationError("attempt-0002 launch does not bind its exact failed GPU resource receipt")
    if (start.get("status")!="FORMAL_START_COMMITTED" or start.get("run_id")!=profile["run_id"]
            or start.get("formal_task_consumption_before") is not False
            or start.get("selected_gpu_uuids")!=[receipt.get("selected_gpu_uuid")]
            or start.get("image_id")!=profile["image_id"]
            or ready.get("verdict")!="READY"
            or qualification.get("status")!="ZERO_FORMAL_QUALIFIED"
            or qualification.get("authority_git_sha")!=expected["prior_authority_git_sha"]
            or qualification.get("model_calls")!=0 or qualification.get("task_bearing_calls")!=0
            or launch.get("exit_code")!=1 or launch.get("execution_mode")!="host-orchestrator"
            or resource.get("status")!="failed" or resource.get("return_code")!=1
            or resource.get("campaign_id")!=profile["campaign_id"]
            or resource.get("gpu_count")!=1
            or resource.get("gpu_uuids")!=[receipt.get("selected_gpu_uuid")]
            or resource.get("stage")!="launch"
            or context.get("cell_id")!="ordinary-r080" or context.get("resume_allowed") is not False):
        raise Q08AuthorizationError("attempt-0002 launch/readiness/qualification evidence is not the exact pre-claim failure")
    if (inspect.get("Name")!="/experiment-host-q08-"+profile["run_id"]+"-ordinary-r080"
            or inspect.get("State",{}).get("Status")!="exited"
            or inspect.get("State",{}).get("ExitCode")!=1
            or "PermissionError" not in stderr or "launch-context.json" not in stderr):
        raise Q08AuthorizationError("attempt-0002 retained container does not prove the recorded context permission failure")
    claims=run_root/"control"/"claims"
    if not claims.is_dir() or list(claims.iterdir()):
        raise Q08AuthorizationError("attempt-0002 created a scientific cell claim")
    for cell in CELL_IDS:
        if (run_root/"cells"/cell/"seen-eval").exists():
            raise Q08AuthorizationError("attempt-0002 created a scientific output leaf")
    if (run_root/"control"/"claims"/"ordinary-r080.claim").exists():
        raise Q08AuthorizationError("attempt-0002 one-use cell claim was consumed")
    namespace=run_root/RECOVERY3_CONTROL_REL
    admission=namespace/"admission-claim.json"
    if admission.exists() or admission.is_symlink():
        raise Q08AuthorizationError("attempt-0003 recovery admission was already consumed")
    if not namespace.is_dir() or namespace.stat().st_uid!=1001 or (namespace.stat().st_mode&0o777)!=0o750:
        raise Q08AuthorizationError("attempt-0003 recovery namespace must be UID1001-owned 0750")
    return receipt



def _validate_engineering_recovery_attempt7(repo: Path, profile: dict, release: dict, ref: dict) -> dict:
    """Admit one control-only continuation after Start4 failed before recovery admission."""
    import subprocess
    path = bound_path(repo, RECOVERY7_RECEIPT_REL)
    expect_file(path, str(ref.get("sha256", "")))
    committed = subprocess.run(["git", "-C", str(repo), "show", f"HEAD:{RECOVERY7_RECEIPT_REL}"],
                               check=True, capture_output=True).stdout
    if committed != path.read_bytes():
        raise Q08AuthorizationError("attempt-0007 recovery receipt is not the committed exact authority bytes")
    predecessor = bound_path(repo, RECOVERY6_RECEIPT_REL)
    expect_file(predecessor, RECOVERY7_PREDECESSOR_SHA)
    predecessor_bytes = subprocess.run(["git", "-C", str(repo), "show", f"HEAD:{RECOVERY6_RECEIPT_REL}"],
                                       check=True, capture_output=True).stdout
    if predecessor_bytes != predecessor.read_bytes():
        raise Q08AuthorizationError("attempt-0007 predecessor receipt is not committed exact bytes")
    previous_ref = {"path": RECOVERY6_RECEIPT_REL, "sha256": RECOVERY7_PREDECESSOR_SHA,
                    "attempt": 6, "prior_authority_git_sha": RECOVERY6_PRIOR_AUTHORITY_SHA}
    _validate_engineering_recovery_attempt6(repo, profile, release, previous_ref)
    receipt = read_json(path)
    expected = {
        "schema": "openevo.q08.zero-consumption-engineering-recovery.v1",
        "status": "ZERO_CONSUMPTION_RECOVERY_PASS",
        "campaign_id": profile["campaign_id"], "run_id": profile["run_id"], "attempt": 7,
        "prior_authority_git_sha": RECOVERY7_PRIOR_AUTHORITY_SHA,
        "prior_operational_source_git_sha": RECOVERY7_PRIOR_SOURCE_SHA,
        "predecessor_recovery_sha256": RECOVERY7_PREDECESSOR_SHA,
        "scientific_resume": False, "formal_task_consumed": 0, "task_bearing_calls": 0,
        "model_calls": 0, "episodes_consumed": 0, "cell_claims_consumed": 0,
        "final_panel_access": 0, "fresh_seen_episodes_max": 960,
        "reused_historical_episodes": 768, "program_total_maximum_including_reuse": 1920,
        "continuation": {"control_attempt": 5, "formal_start_basename": RECOVERY7_FORMAL_START_BASENAME,
            "context_namespace": f"{RECOVERY7_CONTROL_REL}/cells", "cell_claims_allowed": 10,
            "cell_resume_allowed": False, "run_id_changed": False, "matrix_changed": False,
            "panel_changed": False, "budget_changed": False},
        "failure": {"owner": "q08_seen_launch_guard._validate_engineering_recovery_attempt6",
            "message": "attempt-0006 Start3 reference graph is incomplete or differs from actual owner readbacks",
            "stage": "before-recovery-admission", "cell_id": "ordinary-r080", "control_attempt": 4},
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise Q08AuthorizationError(f"attempt-0007 recovery mismatch: {key}")
    frozen = profile["frozen_inputs"]
    science = {"panel_sha256": PANEL_SHA, "matrix_sha256": MATRIX_SHA, "reuse_audit_sha256": REUSE_SHA,
        "probe_wrapper_sha256": WRAPPER_SHA, "model": frozen["model"], "model_revision": frozen["model_revision"],
        "seed_root": frozen["seed_root"], "memory_sha256": frozen["carrier_sha256"]["memory"],
        "skill_sha256": frozen["carrier_sha256"]["skill"], "agent_sha256": frozen["carrier_sha256"]["agent"]}
    if receipt.get("science_identity") != science:
        raise Q08AuthorizationError("attempt-0007 recovery scientific identity changed")
    run_root = (Path(profile["workspace"]) / profile["paths"]["runs"] / profile["run_id"]).resolve(strict=True)
    runs_root = (Path(profile["workspace"]) / profile["paths"]["runs"]).resolve(strict=True)
    archive_root = (run_root / RECOVERY7_ARCHIVE_REL).resolve(strict=True)
    evidence = receipt.get("attempt_0006_evidence")
    required = {"formal_start", "ready", "prelaunch", "qualification", "launch", "resource",
                "namespace_inventory", "reference_graph"}
    if not isinstance(evidence, dict) or set(evidence) != required:
        raise Q08AuthorizationError("attempt-0007 evidence inventory is incomplete")
    previous_profile_raw = subprocess.run(["git", "-C", str(repo), "show",
        f"{RECOVERY7_PRIOR_AUTHORITY_SHA}:{PROFILE}"], check=True, capture_output=True).stdout
    previous_profile = json.loads(previous_profile_raw)
    if previous_profile["frozen_inputs"]["operational_source"]["revision"] != RECOVERY7_PRIOR_SOURCE_SHA:
        raise Q08AuthorizationError("attempt-0007 prior source identity differs from failed Start4 authority")
    launch_archive_path = archive_root / "launch-receipt-attempt-0004.json"
    expect_file(launch_archive_path, str((evidence.get("launch") or {}).get("sha256", "")))
    archived_launch = read_json(launch_archive_path)
    prior_recovery = read_json(predecessor)
    prior_continuation = prior_recovery.get("continuation") or {}
    start_path = Path(str(((archived_launch.get("formal_start") or {}).get("path") or ""))).resolve(strict=True)
    if start_path.name != prior_continuation.get("formal_start_basename"):
        raise Q08AuthorizationError("attempt-0007 Start4 path differs from the actual predecessor release continuation")
    start = read_json(start_path)
    ready_ref = (start.get("readiness") or {}).get("ready_receipt") or {}
    prelaunch_ref = start.get("prelaunch_receipt") or {}
    ready_path = Path(str(ready_ref.get("path", ""))).resolve(strict=True)
    prelaunch_path = Path(str(prelaunch_ref.get("path", ""))).resolve(strict=True)
    qualification_path = Path(previous_profile["frozen_inputs"]["qualification_path"]).resolve(strict=True)
    expected_paths = {"formal_start": start_path.resolve(), "ready": ready_path, "prelaunch": prelaunch_path,
        "qualification": qualification_path, "launch": archive_root / "launch-receipt-attempt-0004.json",
        "resource": archive_root / "gpu-resource-receipt-attempt-0004.json",
        "namespace_inventory": archive_root / "namespace-inventory-attempt-0004.json",
        "reference_graph": archive_root / "formal-start-reference-graph-attempt-0004.json"}
    paths = {}
    for key, item in evidence.items():
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise Q08AuthorizationError(f"attempt-0007 evidence reference malformed: {key}")
        actual = Path(str(item["path"])).resolve(strict=True)
        if actual != expected_paths[key] or actual.is_symlink() or not actual.is_file():
            raise Q08AuthorizationError(f"attempt-0007 evidence path differs from its actual owner: {key}")
        expect_file(actual, str(item["sha256"]))
        paths[key] = actual
    if (not archive_root.is_dir() or archive_root.is_symlink()
            or archive_root.stat().st_uid != 0 or archive_root.stat().st_gid != 0
            or stat.S_IMODE(archive_root.stat().st_mode) != 0o555):
        raise Q08AuthorizationError("attempt-0007 evidence archive must be root-owned, immutable and traversable")
    for key in ("launch", "resource", "namespace_inventory", "reference_graph"):
        st = paths[key].stat()
        if st.st_uid != 0 or st.st_gid != 0 or stat.S_IMODE(st.st_mode) != 0o444:
            raise Q08AuthorizationError(f"attempt-0007 archive evidence is not immutable: {key}")
    old_release_bytes = subprocess.run(["git", "-C", str(repo), "show",
        f"{RECOVERY7_PRIOR_AUTHORITY_SHA}:{RELEASE}"], check=True, capture_output=True).stdout
    old_router_bytes = subprocess.run(["git", "-C", str(repo), "show",
        f"{RECOVERY7_PRIOR_AUTHORITY_SHA}:{ROUTER}"], check=True, capture_output=True).stdout
    old_profile_sha = hashlib.sha256(previous_profile_raw).hexdigest()
    old_release_sha = hashlib.sha256(old_release_bytes).hexdigest()
    old_router_sha = hashlib.sha256(old_router_bytes).hexdigest()
    start_authority = (start.get("readiness") or {}).get("authority") or {}
    old_repo = Path(str(start_authority.get("router_path", ""))).resolve(strict=True).parents[3]
    selected_uuids = start.get("selected_gpu_uuids")
    if (not isinstance(selected_uuids, list) or len(selected_uuids) != 1
            or selected_uuids[0] not in profile["gpu_gate"]["authorized_uuids"]):
        raise Q08AuthorizationError("attempt-0007 selected GPU is not derived from the failed Start4 authorization")
    selected = selected_uuids[0]
    if start.get("image_id") != profile["image_id"]:
        raise Q08AuthorizationError("attempt-0007 failed FormalStart image differs from the frozen Q08 runtime")
    old_profile_path = Path(str((start.get("readiness") or {}).get("profile", {}).get("path", ""))).resolve(strict=True)
    old_router_path = Path(str(start_authority.get("router_path", ""))).resolve(strict=True)
    if (old_profile_path != (old_repo / PROFILE).resolve()
            or old_router_path != (old_repo / ROUTER).resolve()
            or old_profile_path.read_bytes() != previous_profile_raw
            or (old_repo / RELEASE).read_bytes() != old_release_bytes
            or old_router_path.read_bytes() != old_router_bytes):
        raise Q08AuthorizationError("attempt-0007 prior profile/release/router paths differ from the failed Start4 authority")
    prior_release = json.loads(old_release_bytes)
    prior_router = json.loads(old_router_bytes)
    prior_recovery_ref = prior_release.get("engineering_recovery") or {}
    if (prior_router.get("release") != {"path": RELEASE, "sha256": old_release_sha}
            or prior_router.get("status") != "launch-authorized"
            or prior_recovery_ref.get("attempt") != 6
            or prior_recovery_ref.get("path") != RECOVERY6_RECEIPT_REL
            or prior_recovery_ref.get("sha256") != RECOVERY7_PREDECESSOR_SHA):
        raise Q08AuthorizationError("attempt-0007 prior release/router chain is not the exact failed recovery6 authority")
    start_ref = (start.get("readiness") or {}).get("ready_receipt") or {}
    start_profile = (start.get("readiness") or {}).get("profile") or {}
    start_prelaunch = start.get("prelaunch_receipt") or {}
    if (start.get("status") != "FORMAL_START_COMMITTED" or start.get("run_id") != profile["run_id"]
            or start.get("campaign_id") != profile["campaign_id"] or start.get("selected_gpu_uuids") != [selected]
            or start.get("formal_task_consumption_before") is not False
            or start.get("scientific_authority_granted_by_receipt") is not False
            or start_ref.get("path") != str(paths["ready"])
            or _normalize_sha(start_ref.get("sha256")) != sha256_file(paths["ready"])
            or start_prelaunch.get("path") != str(paths["prelaunch"])
            or _normalize_sha(start_prelaunch.get("sha256")) != sha256_file(paths["prelaunch"])
            or _normalize_sha(start_profile.get("sha256")) != old_profile_sha
            or Path(str(start_profile.get("path", ""))).name != Path(PROFILE).name):
        raise Q08AuthorizationError("attempt-0007 predecessor FormalStart is not exact zero-task evidence")
    ready = read_json(paths["ready"]); prelaunch = read_json(paths["prelaunch"])
    qualification = read_json(paths["qualification"]); launch = read_json(paths["launch"])
    resource = read_json(paths["resource"]); inventory = read_json(paths["namespace_inventory"])
    if (ready.get("verdict") != "READY"
            or ready.get("profile", {}).get("sha256") != "sha256:" + old_profile_sha
            or ready.get("release_receipt", {}).get("sha256") != "sha256:" + old_release_sha
            or (ready.get("authority") or {}).get("router_status") != "launch-authorized"):
        raise Q08AuthorizationError("attempt-0007 READY does not bind the exact failed authority")
    if (prelaunch.get("status") != "PASS" or prelaunch.get("return_code") != 0
            or prelaunch.get("formal_task_consumed") is not False
            or prelaunch.get("scientific_budget_consumed") != 0 or prelaunch.get("model_calls") != 0
            or prelaunch.get("selected_gpu_uuids") != [selected]):
        raise Q08AuthorizationError("attempt-0007 prelaunch was not a zero-consumption PASS")
    projection = {key: value for key, value in qualification.items() if key != "receipt_sha256"}
    qsha = hashlib.sha256(json.dumps(projection, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    qbinding = qualification.get("q08_binding") or {}
    if (qualification.get("receipt_sha256") != qsha or qualification.get("status") != "ZERO_FORMAL_QUALIFIED"
            or qualification.get("authority_git_sha") != RECOVERY7_PRIOR_AUTHORITY_SHA
            or qualification.get("execution_sha") != RECOVERY7_PRIOR_SOURCE_SHA
            or qualification.get("model_calls") != 0 or qualification.get("task_bearing_calls") != 0
            or qualification.get("selected_gpu_uuids") != [selected]
            or qbinding.get("profile_sha256") != old_profile_sha
            or qbinding.get("release_sha256") != old_release_sha
            or qbinding.get("router_sha256") != old_router_sha
            or qbinding.get("engineering_recovery_sha256") != RECOVERY7_PREDECESSOR_SHA):
        raise Q08AuthorizationError("attempt-0007 qualification is not exact zero-formal evidence")
    start_ref = launch.get("formal_start") or {}; resource_ref = launch.get("gpu_resource_receipt") or {}
    if (launch.get("run_id") != profile["run_id"] or launch.get("execution_mode") != "host-orchestrator"
            or launch.get("exit_code") != 1 or start_ref.get("path") != str(paths["formal_start"])
            or _normalize_sha(start_ref.get("sha256")) != sha256_file(paths["formal_start"])
            or _normalize_sha(resource_ref.get("sha256")) != sha256_file(paths["resource"])):
        raise Q08AuthorizationError("attempt-0007 launch receipt does not bind the exact failed FormalStart")
    if (resource.get("status") != "failed" or resource.get("return_code") != 1
            or resource.get("stage") != "launch" or resource.get("campaign_id") != profile["campaign_id"]
            or resource.get("gpu_uuids") != [selected] or resource.get("gpu_hours", 1) > 0.01):
        raise Q08AuthorizationError("attempt-0007 resource receipt is not the bounded pre-container failure")
    if (inventory.get("source_attempt_files", {}).get("gpu-resource-receipt-attempt-0004.json") != sha256_file(paths["resource"])
            or inventory.get("source_attempt_files", {}).get("launch-receipt-attempt-0004.json") != sha256_file(paths["launch"])):
        raise Q08AuthorizationError("attempt-0007 inventory does not bind exact archived launch/resource bytes")
    if (inventory.get("status") != "PRE_CLAIM_HOST_ORCHESTRATOR_FAILURE"
            or inventory.get("run_id") != profile["run_id"] or inventory.get("campaign_id") != profile["campaign_id"]
            or inventory.get("control_attempt") != 4 or inventory.get("formal_start_sha256") != sha256_file(paths["formal_start"])
            or inventory.get("cell_claims_count") != 0 or inventory.get("seen_eval_leaf_count") != 0
            or inventory.get("recovery_admission_exists") is not False
            or inventory.get("attempt4_context_exists") is not False
            or inventory.get("container", {}).get("inspect_status") != "not_found"
            or inventory.get("container", {}).get("inspect_returncode") != 1
            or inventory.get("container", {}).get("name") != f"experiment-host-q08-{profile['run_id']}-ordinary-r080-a0004"
            or "no such object" not in inventory.get("container", {}).get("inspect_stderr", "").lower()):
        raise Q08AuthorizationError("attempt-0007 namespace inventory does not prove the exact empty pre-container boundary")
    graph = read_json(paths["reference_graph"])
    source_docs = [(name, paths[name]) for name in ("formal_start", "ready", "prelaunch", "qualification", "launch", "resource", "namespace_inventory")]
    expected_source_docs = [{"name": name, "path": str(doc), "sha256": sha256_file(doc)} for name, doc in source_docs]
    computed_edges = []
    resource_alias = (run_root / "gpu-resource-receipt.json").resolve()
    archived_resource_sha = sha256_file(paths["resource"])
    for name, source_doc in source_docs:
        source_value = read_json(source_doc)
        def collect_refs(value, pointer="$"):
            if isinstance(value, dict):
                if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
                    raw_target = Path(value["path"])
                    declared = _normalize_sha(value["sha256"])
                    target = (raw_target if raw_target.is_absolute() else old_repo / raw_target).resolve()
                    if name == "launch" and pointer == "$/gpu_resource_receipt":
                        if (target != resource_alias or declared != archived_resource_sha
                                or inventory.get("source_attempt_files", {}).get("gpu-resource-receipt-attempt-0004.json") != archived_resource_sha):
                            raise Q08AuthorizationError("attempt-0007 resource alias is not backed by its exact immutable archive bytes")
                        actual = archived_resource_sha
                        resolution = {"path": str(paths["resource"]), "sha256": archived_resource_sha}
                    else:
                        target = target.resolve(strict=True)
                        actual = sha256_file(target)
                        resolution = None
                    edge = {"source": str(source_doc), "pointer": pointer, "target": str(target),
                            "declared_sha256": declared, "actual_sha256": actual,
                            "verdict": "MATCH" if declared == actual else "STALE_REFERENCE"}
                    if resolution is not None:
                        edge["immutable_archive_resolution"] = resolution
                    computed_edges.append(edge)
                for key, child in sorted(value.items()):
                    collect_refs(child, pointer + "/" + str(key).replace("~", "~0").replace("/", "~1"))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    collect_refs(child, pointer + "/" + str(index))
        collect_refs(source_value)
    if (graph.get("schema") != "openevo.q08.formal-start-reference-graph.v2"
            or graph.get("status") != "PASS_WITH_ARCHIVED_MUTABLE_RESOURCE_ALIAS"
            or graph.get("source_documents") != expected_source_docs
            or graph.get("edge_count") != len(computed_edges)
            or graph.get("edges") != computed_edges):
        raise Q08AuthorizationError("attempt-0007 Start4 reference graph differs from exact archive-backed owner readbacks")
    claims = run_root / "control" / "claims"
    if claims.exists() and (claims.is_symlink() or any(claims.iterdir())):
        raise Q08AuthorizationError("attempt-0007 found a consumed scientific cell claim")
    for cell in CELL_IDS:
        leaf = run_root / "cells" / cell / "seen-eval"
        if leaf.exists() or leaf.is_symlink():
            raise Q08AuthorizationError("attempt-0007 found a scientific output leaf")
    admission = run_root / RECOVERY7_CONTROL_REL / "admission-claim.json"
    if admission.exists() or admission.is_symlink():
        raise Q08AuthorizationError("attempt-0007 admission was already consumed")
    namespace = run_root / RECOVERY7_CONTROL_REL
    if (not namespace.is_dir() or namespace.is_symlink() or namespace.stat().st_uid != 1001
            or namespace.stat().st_gid != 1001 or stat.S_IMODE(namespace.stat().st_mode) != 0o750):
        raise Q08AuthorizationError("attempt-0007 recovery namespace must be UID/GID1001-owned 0750")
    if (namespace / "cells" / "attempt-0005").exists():
        raise Q08AuthorizationError("attempt-0007 attempt5 namespace already exists before admission")
    return receipt


def _validate_engineering_recovery_attempt8(repo: Path, profile: dict, release: dict, ref: dict) -> dict:
    """Admit one control-only continuation after Start5's qualification-path mismatch."""
    import subprocess

    def git_bytes(revision: str, path: str) -> bytes:
        return subprocess.run(["git", "-C", str(repo), "show", f"{revision}:{path}"],
                              check=True, capture_output=True).stdout

    recovery_path = bound_path(repo, RECOVERY8_RECEIPT_REL)
    expect_file(recovery_path, str(ref.get("sha256", "")))
    if git_bytes("HEAD", RECOVERY8_RECEIPT_REL) != recovery_path.read_bytes():
        raise Q08AuthorizationError("attempt-0008 recovery receipt is not the committed exact authority bytes")

    predecessor_path = bound_path(repo, RECOVERY7_RECEIPT_REL)
    expect_file(predecessor_path, RECOVERY8_PREDECESSOR_SHA)
    if git_bytes("HEAD", RECOVERY7_RECEIPT_REL) != predecessor_path.read_bytes():
        raise Q08AuthorizationError("attempt-0008 predecessor recovery receipt is not committed exact bytes")
    predecessor_ref = {"path": RECOVERY7_RECEIPT_REL, "sha256": RECOVERY8_PREDECESSOR_SHA,
                       "attempt": 7, "prior_authority_git_sha": RECOVERY7_PRIOR_AUTHORITY_SHA}
    _validate_engineering_recovery_attempt7(repo, profile, release, predecessor_ref)

    receipt = read_json(recovery_path)
    expected = {
        "schema": "openevo.q08.zero-consumption-engineering-recovery.v1",
        "status": "ZERO_CONSUMPTION_RECOVERY_PASS",
        "campaign_id": profile["campaign_id"], "run_id": profile["run_id"], "attempt": 8,
        "prior_authority_git_sha": RECOVERY8_PRIOR_AUTHORITY_SHA,
        "prior_operational_source_git_sha": RECOVERY8_PRIOR_SOURCE_SHA,
        "predecessor_recovery_sha256": RECOVERY8_PREDECESSOR_SHA,
        "scientific_resume": False, "formal_task_consumed": 0, "task_bearing_calls": 0,
        "model_calls": 0, "episodes_consumed": 0, "cell_claims_consumed": 0,
        "final_panel_access": 0, "fresh_seen_episodes_max": 960,
        "reused_historical_episodes": 768, "program_total_maximum_including_reuse": 1920,
        "continuation": {"control_attempt": 6, "formal_start_basename": RECOVERY8_FORMAL_START_BASENAME,
            "context_namespace": f"{RECOVERY8_CONTROL_REL}/cells", "cell_claims_allowed": 10,
            "cell_resume_allowed": False, "run_id_changed": False, "matrix_changed": False,
            "panel_changed": False, "budget_changed": False},
        "failure": {"owner": "q08_seen_host_orchestrator._load_formal_start",
            "message": "Q08 qualification path differs from the frozen profile",
            "stage": "before-recovery-admission", "cell_id": "ordinary-r080", "control_attempt": 5},
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise Q08AuthorizationError(f"attempt-0008 recovery mismatch: {key}")

    frozen = profile["frozen_inputs"]
    science = {"panel_sha256": PANEL_SHA, "matrix_sha256": MATRIX_SHA, "reuse_audit_sha256": REUSE_SHA,
        "probe_wrapper_sha256": WRAPPER_SHA, "model": frozen["model"],
        "model_revision": frozen["model_revision"], "seed_root": frozen["seed_root"],
        "memory_sha256": frozen["carrier_sha256"]["memory"],
        "skill_sha256": frozen["carrier_sha256"]["skill"], "agent_sha256": frozen["carrier_sha256"]["agent"]}
    if receipt.get("science_identity") != science:
        raise Q08AuthorizationError("attempt-0008 recovery scientific identity changed")

    previous_profile_raw = git_bytes(RECOVERY8_PRIOR_AUTHORITY_SHA, PROFILE)
    previous_release_raw = git_bytes(RECOVERY8_PRIOR_AUTHORITY_SHA, RELEASE)
    previous_router_raw = git_bytes(RECOVERY8_PRIOR_AUTHORITY_SHA, ROUTER)
    previous_profile = json.loads(previous_profile_raw)
    previous_release = json.loads(previous_release_raw)
    previous_router = json.loads(previous_router_raw)
    previous_recovery = previous_release.get("engineering_recovery") or {}
    if (previous_profile.get("frozen_inputs", {}).get("operational_source", {}).get("revision") != RECOVERY8_PRIOR_SOURCE_SHA
            or previous_recovery.get("attempt") != 7
            or previous_recovery.get("path") != RECOVERY7_RECEIPT_REL
            or previous_recovery.get("sha256") != RECOVERY8_PREDECESSOR_SHA
            or previous_router.get("status") != "launch-authorized"
            or previous_router.get("release") != {"path": RELEASE,
                "sha256": hashlib.sha256(previous_release_raw).hexdigest()}):
        raise Q08AuthorizationError("attempt-0008 predecessor authority chain is not the exact recovery7 state")

    previous_command = previous_profile.get("command") or []
    try:
        qualification_index = previous_command.index("--qualification") + 1
        previous_command_qualification = previous_command[qualification_index]
    except (ValueError, IndexError):
        raise Q08AuthorizationError("attempt-0008 predecessor profile has no exact qualification argument")
    previous_frozen_qualification = previous_profile["frozen_inputs"].get("qualification_path")
    expected_old_qualification = "/workspace/runs/post-advisor-q08-seen-retention-202610010742-qualification-attempt0011/qualification.json"
    expected_frozen_qualification = "/workspace/runs/post-advisor-q08-seen-retention-202610010742-qualification-attempt0012/qualification.json"
    if (previous_command_qualification != expected_old_qualification
            or previous_frozen_qualification != expected_frozen_qualification
            or previous_command_qualification == previous_frozen_qualification):
        raise Q08AuthorizationError("attempt-0008 predecessor does not bind the exact qualification-path drift")

    new_qualification = "/workspace/runs/post-advisor-q08-seen-retention-202610010742-qualification-attempt0014/qualification.json"
    command = profile.get("command") or []
    try:
        current_qualification = command[command.index("--qualification") + 1]
    except (ValueError, IndexError):
        raise Q08AuthorizationError("attempt-0008 successor profile has no qualification argument")
    if (frozen.get("qualification_path") != new_qualification
            or current_qualification != new_qualification
            or frozen.get("operational_source", {}).get("revision") == RECOVERY8_PRIOR_SOURCE_SHA):
        raise Q08AuthorizationError("attempt-0008 successor profile did not bind one fresh qualification and source")
    if (profile.get("run_id") != previous_profile.get("run_id")
            or profile.get("campaign_id") != previous_profile.get("campaign_id")
            or profile.get("image_id") != previous_profile.get("image_id")
            or profile.get("gpu_gate", {}).get("authorized_uuids") != previous_profile.get("gpu_gate", {}).get("authorized_uuids")
            or profile.get("frozen_inputs", {}).get("checkpoint_matrix") != previous_profile.get("frozen_inputs", {}).get("checkpoint_matrix")
            or profile.get("frozen_inputs", {}).get("panel_manifest") != previous_profile.get("frozen_inputs", {}).get("panel_manifest")
            or profile.get("frozen_inputs", {}).get("fresh_reuse_audit") != previous_profile.get("frozen_inputs", {}).get("fresh_reuse_audit")
            or profile.get("cells") != previous_profile.get("cells")
            or profile.get("sampling") != previous_profile.get("sampling")
            or profile.get("final_panel_access_allowed") is not False
            or profile.get("formal_training_allowed") is not False
            or release.get("final_panel_access_limit") != 0
            or release.get("budget") != {"eligible_cells": 10, "episodes_per_cell": 96,
                "fresh_seen_episodes_max": 960, "one_attempt_per_cell": True,
                "program_total_maximum_including_reuse": 1920, "resume_allowed": False,
                "reused_historical_episodes": 768}):
        raise Q08AuthorizationError("attempt-0008 successor changed the frozen run, image, matrix, reuse or panel")

    run_root = (Path(profile["workspace"]) / profile["paths"]["runs"] / profile["run_id"]).resolve(strict=True)
    runs_root = (Path(profile["workspace"]) / profile["paths"]["runs"]).resolve(strict=True)
    archive_root = (run_root / RECOVERY8_ARCHIVE_REL).resolve(strict=True)
    evidence = receipt.get("attempt_0007_evidence")
    required = {"formal_start", "ready", "prelaunch", "qualification", "launch", "resource", "namespace_inventory"}
    if not isinstance(evidence, dict) or set(evidence) != required:
        raise Q08AuthorizationError("attempt-0008 evidence inventory is incomplete")
    start_path = runs_root / f"{profile['run_id']}-readiness-attempt0008/control/formal-start-attempt-0005.json"
    expected_paths = {
        "formal_start": start_path,
        "ready": runs_root / f"{profile['run_id']}-readiness-attempt0008/control/execution-readiness-13ab05e7518a4d489a6af02ded0973e8.json",
        "prelaunch": runs_root / f"{profile['run_id']}-readiness-attempt0008/control/prelaunch-zero-formal-attempt-0005.json",
        "qualification": runs_root / f"{profile['run_id']}-qualification-attempt0012/qualification.json",
        "launch": archive_root / "launch-receipt-attempt-0005.json",
        "resource": archive_root / "gpu-resource-receipt-attempt-0005.json",
        "namespace_inventory": archive_root / "namespace-inventory-attempt-0005.json",
    }
    original_paths = {
        "launch": run_root / "launch-receipt.json",
        "resource": run_root / "gpu-resource-receipt.json",
    }
    paths = {}
    for key, item in evidence.items():
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "source_path"}:
            raise Q08AuthorizationError(f"attempt-0008 evidence reference is malformed: {key}")
        actual = Path(str(item["path"])).resolve(strict=True)
        if actual != expected_paths[key].resolve() or actual.is_symlink() or not actual.is_file():
            raise Q08AuthorizationError(f"attempt-0008 evidence path differs from the exact failed attempt: {key}")
        expect_file(actual, str(item["sha256"]))
        expected_source = original_paths.get(key, actual)
        if Path(str(item["source_path"])).resolve() != expected_source.resolve():
            raise Q08AuthorizationError(f"attempt-0008 source path differs from its exact owner: {key}")
        # The run-root receipt paths are mutable latest-attempt projections. The
        # immutable archive's committed path, recorded SHA, and 0444/0555
        # ownership checks below authenticate the predecessor evidence. Do not
        # compare that historical archive with a live projection that a later
        # authorized FormalStart may have replaced.
        paths[key] = actual

    if (not archive_root.is_dir() or archive_root.is_symlink()
            or archive_root.stat().st_uid != 0 or archive_root.stat().st_gid != 0
            or stat.S_IMODE(archive_root.stat().st_mode) != 0o555):
        raise Q08AuthorizationError("attempt-0008 evidence archive must be root-owned, immutable and traversable")
    for key in ("launch", "resource", "namespace_inventory"):
        st = paths[key].stat()
        if st.st_uid != 0 or st.st_gid != 0 or stat.S_IMODE(st.st_mode) != 0o444:
            raise Q08AuthorizationError(f"attempt-0008 archive evidence is not immutable: {key}")

    start = read_json(paths["formal_start"])
    ready = read_json(paths["ready"])
    prelaunch = read_json(paths["prelaunch"])
    qualification = read_json(paths["qualification"])
    launch = read_json(paths["launch"])
    resource = read_json(paths["resource"])
    inventory = read_json(paths["namespace_inventory"])
    selected = start.get("selected_gpu_uuids")
    if (not isinstance(selected, list) or len(selected) != 1
            or selected[0] not in profile["gpu_gate"]["authorized_uuids"]
            or start.get("status") != "FORMAL_START_COMMITTED"
            or start.get("run_id") != profile["run_id"]
            or start.get("campaign_id") != profile["campaign_id"]
            or start.get("formal_task_consumption_before") is not False
            or start.get("scientific_authority_granted_by_receipt") is not False
            or start.get("image_id") != profile["image_id"]):
        raise Q08AuthorizationError("attempt-0008 predecessor FormalStart is not exact Q08 zero-task evidence")
    if (start.get("readiness", {}).get("profile", {}).get("sha256") != "sha256:" + hashlib.sha256(previous_profile_raw).hexdigest()
            or Path(start.get("readiness", {}).get("profile", {}).get("path", "")).name != Path(PROFILE).name
            or start.get("readiness", {}).get("ready_receipt", {}).get("sha256") != "sha256:" + sha256_file(paths["ready"])
            or start.get("prelaunch_receipt", {}).get("sha256") != "sha256:" + sha256_file(paths["prelaunch"])):
        raise Q08AuthorizationError("attempt-0008 predecessor FormalStart references differ from sealed evidence")
    if (ready.get("verdict") != "READY"
            or ready.get("profile", {}).get("sha256") != "sha256:" + hashlib.sha256(previous_profile_raw).hexdigest()
            or ready.get("release_receipt", {}).get("sha256") != "sha256:" + hashlib.sha256(previous_release_raw).hexdigest()
            or ready.get("authority", {}).get("router_status") != "launch-authorized"):
        raise Q08AuthorizationError("attempt-0008 READY does not bind the exact failed authority")
    if (prelaunch.get("status") != "PASS" or prelaunch.get("return_code") != 0
            or prelaunch.get("formal_task_consumed") is not False
            or prelaunch.get("scientific_budget_consumed") != 0
            or prelaunch.get("model_calls") != 0 or prelaunch.get("training_calls") != 0
            or prelaunch.get("selected_gpu_uuids") != selected):
        raise Q08AuthorizationError("attempt-0008 prelaunch was not a zero-consumption PASS")
    qualification_projection = {key: value for key, value in qualification.items() if key != "receipt_sha256"}
    qualification_sha = hashlib.sha256(json.dumps(qualification_projection, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    if (qualification.get("receipt_sha256") != qualification_sha
            or qualification.get("status") != "ZERO_FORMAL_QUALIFIED"
            or qualification.get("authority_git_sha") != RECOVERY8_PRIOR_AUTHORITY_SHA
            or qualification.get("execution_sha") != RECOVERY8_PRIOR_SOURCE_SHA
            or qualification.get("model_calls") != 0 or qualification.get("task_bearing_calls") != 0
            or qualification.get("selected_gpu_uuids") != selected
            or (qualification.get("q08_binding") or {}).get("profile_sha256") != hashlib.sha256(previous_profile_raw).hexdigest()):
        raise Q08AuthorizationError("attempt-0008 prior qualification does not bind the exact zero-formal profile")
    if (launch.get("run_id") != profile["run_id"] or launch.get("exit_code") != 1
            or launch.get("execution_mode") != "host-orchestrator"
            or (launch.get("formal_start") or {}).get("path") != str(paths["formal_start"])
            or _normalize_sha((launch.get("formal_start") or {}).get("sha256")) != sha256_file(paths["formal_start"])
            or _normalize_sha((launch.get("gpu_resource_receipt") or {}).get("sha256")) != sha256_file(paths["resource"])):
        raise Q08AuthorizationError("attempt-0008 launch receipt does not bind the exact failed FormalStart")
    if (resource.get("status") != "failed" or resource.get("return_code") != 1
            or resource.get("stage") != "launch" or resource.get("campaign_id") != profile["campaign_id"]
            or resource.get("gpu_uuids") != selected or resource.get("gpu_hours", 1) > 0.01):
        raise Q08AuthorizationError("attempt-0008 resource receipt is not the bounded pre-cell failure")
    if (inventory.get("status") != "PRE_CLAIM_HOST_ORCHESTRATOR_FAILURE"
            or inventory.get("run_id") != profile["run_id"] or inventory.get("campaign_id") != profile["campaign_id"]
            or inventory.get("control_attempt") != 5
            or inventory.get("formal_start_sha256") != sha256_file(paths["formal_start"])
            or inventory.get("cell_claims_count") != 0 or inventory.get("seen_eval_leaf_count") != 0
            or inventory.get("formal_task_consumed") != 0 or inventory.get("model_calls") != 0
            or inventory.get("recovery_admission_exists") is not False
            or inventory.get("attempt5_context_exists") is not False
            or inventory.get("container", {}).get("inspect_status") != "not_found"
            or inventory.get("container", {}).get("inspect_returncode") != 1):
        raise Q08AuthorizationError("attempt-0008 namespace inventory does not prove the exact empty pre-admission boundary")

    claims = run_root / "control" / "claims"
    if claims.exists() and (claims.is_symlink() or any(claims.iterdir())):
        raise Q08AuthorizationError("attempt-0008 found a consumed scientific cell claim")
    for cell in CELL_IDS:
        leaf = run_root / "cells" / cell / "seen-eval"
        if leaf.exists() or leaf.is_symlink():
            raise Q08AuthorizationError("attempt-0008 found a scientific output leaf")
    previous_admission = run_root / RECOVERY7_CONTROL_REL / "admission-claim.json"
    if previous_admission.exists() or previous_admission.is_symlink():
        raise Q08AuthorizationError("attempt-0008 predecessor admission was already consumed")
    previous_context = run_root / RECOVERY7_CONTROL_REL / "cells" / "attempt-0005"
    if previous_context.exists() or previous_context.is_symlink():
        raise Q08AuthorizationError("attempt-0008 predecessor cell context already exists")
    admission = run_root / RECOVERY8_CONTROL_REL / "admission-claim.json"
    context = run_root / RECOVERY8_CONTROL_REL / "cells" / "attempt-0006"
    if admission.exists() or admission.is_symlink() or context.exists() or context.is_symlink():
        raise Q08AuthorizationError("attempt-0008 successor admission/context was already consumed")
    return receipt


def _validate_engineering_recovery_attempt9(repo: Path, profile: dict, release: dict, ref: dict) -> dict:
    """Permit Start7 after Start6 failed before admission and consumed nothing."""
    import subprocess

    def git_bytes(revision: str, path: str) -> bytes:
        return subprocess.run(["git", "-C", str(repo), "show", f"{revision}:{path}"],
                              check=True, capture_output=True).stdout

    receipt_path = bound_path(repo, RECOVERY9_RECEIPT_REL)
    expect_file(receipt_path, str(ref.get("sha256", "")))
    raw = receipt_path.read_bytes()
    if git_bytes("HEAD", RECOVERY9_RECEIPT_REL) != raw:
        raise Q08AuthorizationError("attempt-0009 recovery receipt is not committed exact bytes")
    receipt = read_json(receipt_path)

    prior_profile = json.loads(git_bytes(RECOVERY9_PRIOR_AUTHORITY_SHA, PROFILE))
    prior_release = json.loads(git_bytes(RECOVERY9_PRIOR_AUTHORITY_SHA, RELEASE))
    prior_router = json.loads(git_bytes(RECOVERY9_PRIOR_AUTHORITY_SHA, ROUTER))
    prior_recovery = prior_release.get("engineering_recovery") or {}
    if (prior_profile.get("frozen_inputs", {}).get("operational_source", {}).get("revision") != RECOVERY9_PRIOR_SOURCE_SHA
            or prior_recovery.get("attempt") != 8
            or prior_recovery.get("sha256") != RECOVERY9_PREDECESSOR_SHA
            or prior_router.get("status") != "launch-authorized"):
        raise Q08AuthorizationError("attempt-0009 predecessor authority is not the exact recovery8 authority")
    _validate_engineering_recovery_attempt8(repo, prior_profile, prior_release, prior_recovery)

    expected = {
        "schema": "openevo.q08.zero-consumption-engineering-recovery.v1",
        "status": "ZERO_CONSUMPTION_RECOVERY_PASS",
        "campaign_id": profile["campaign_id"], "run_id": profile["run_id"], "attempt": 9,
        "prior_authority_git_sha": RECOVERY9_PRIOR_AUTHORITY_SHA,
        "prior_operational_source_git_sha": RECOVERY9_PRIOR_SOURCE_SHA,
        "predecessor_recovery_sha256": RECOVERY9_PREDECESSOR_SHA,
        "scientific_resume": False, "formal_task_consumed": 0, "task_bearing_calls": 0,
        "model_calls": 0, "episodes_consumed": 0, "cell_claims_consumed": 0,
        "final_panel_access": 0, "fresh_seen_episodes_max": 960,
        "reused_historical_episodes": 768, "program_total_maximum_including_reuse": 1920,
        "continuation": {"control_attempt": 7, "formal_start_basename": RECOVERY9_FORMAL_START_BASENAME,
            "context_namespace": f"{RECOVERY9_CONTROL_REL}/cells", "cell_claims_allowed": 10,
            "cell_resume_allowed": False, "run_id_changed": False, "matrix_changed": False,
            "panel_changed": False, "budget_changed": False},
        "failure": {"owner": "q08_seen_host_orchestrator.execute",
            "message": "Q08 FormalStart attempt differs from the exact recovery continuation",
            "stage": "before-recovery-admission", "cell_id": "ordinary-r080", "control_attempt": 6},
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise Q08AuthorizationError(f"attempt-0009 recovery mismatch: {key}")

    frozen = profile["frozen_inputs"]
    science = {"panel_sha256": PANEL_SHA, "matrix_sha256": MATRIX_SHA, "reuse_audit_sha256": REUSE_SHA,
        "probe_wrapper_sha256": WRAPPER_SHA, "model": frozen["model"],
        "model_revision": frozen["model_revision"], "seed_root": frozen["seed_root"],
        "memory_sha256": frozen["carrier_sha256"]["memory"], "skill_sha256": frozen["carrier_sha256"]["skill"],
        "agent_sha256": frozen["carrier_sha256"]["agent"]}
    if receipt.get("science_identity") != science:
        raise Q08AuthorizationError("attempt-0009 scientific identity changed")
    if (profile.get("campaign_id") != prior_profile.get("campaign_id")
            or profile.get("run_id") != prior_profile.get("run_id")
            or profile.get("image_id") != prior_profile.get("image_id")
            or profile.get("cells") != prior_profile.get("cells")
            or profile.get("sampling") != prior_profile.get("sampling")
            or profile.get("final_panel_access_allowed") is not False
            or profile.get("formal_training_allowed") is not False
            or profile.get("gpu_gate", {}).get("authorized_uuids") != prior_profile.get("gpu_gate", {}).get("authorized_uuids")
            or profile.get("frozen_inputs", {}).get("checkpoint_matrix") != prior_profile.get("frozen_inputs", {}).get("checkpoint_matrix")
            or profile.get("frozen_inputs", {}).get("panel_manifest") != prior_profile.get("frozen_inputs", {}).get("panel_manifest")
            or profile.get("frozen_inputs", {}).get("fresh_reuse_audit") != prior_profile.get("frozen_inputs", {}).get("fresh_reuse_audit")
            or release.get("budget") != prior_release.get("budget")
            or release.get("final_panel_access_limit") != 0):
        raise Q08AuthorizationError("attempt-0009 changed frozen science, GPU pool, or budget")
    prior_qualification = str(prior_profile.get("frozen_inputs", {}).get("qualification_path", ""))
    current_qualification = str(frozen.get("qualification_path", ""))
    prior_match = re.fullmatch(r"(.+-qualification)-attempt(\d{4})/qualification\.json", prior_qualification)
    current_match = re.fullmatch(r"(.+-qualification)-attempt(\d{4})/qualification\.json", current_qualification)
    qualification_attempts_advance = (
        prior_match is not None
        and current_match is not None
        and prior_match.group(1) == current_match.group(1)
        and int(current_match.group(2)) > int(prior_match.group(2))
    )
    if (frozen.get("operational_source", {}).get("revision") != profile.get("source", {}).get("experiment", {}).get("revision")
            or not qualification_attempts_advance):
        raise Q08AuthorizationError("attempt-0009 profile did not bind a fresh sequential qualification for the exact source")
    for attempt in range(int(prior_match.group(2)) + 1, int(current_match.group(2))):
        intermediate_path = Path(f"{prior_match.group(1)}-attempt{attempt:04d}/qualification.json")
        if not intermediate_path.is_file() or intermediate_path.is_symlink():
            raise Q08AuthorizationError("attempt-0009 qualification sequence is missing an immutable intermediate receipt")
        intermediate = read_json(intermediate_path)
        intermediate_binding = intermediate.get("q08_binding") or {}
        intermediate_authority = str(intermediate.get("authority_git_sha", ""))
        committed_profile = json.loads(git_bytes(intermediate_authority, PROFILE))
        committed_release = json.loads(git_bytes(intermediate_authority, RELEASE))
        committed_router = json.loads(git_bytes(intermediate_authority, ROUTER))
        if (intermediate.get("status") != "ZERO_FORMAL_QUALIFIED"
                or intermediate.get("formal_task_budget") != 0
                or intermediate.get("scientific_budget_consumed") != 0
                or intermediate.get("task_bearing_calls") != 0
                or intermediate.get("model_calls") != 0
                or intermediate.get("training_calls") != 0
                or intermediate.get("training_steps") != 0
                or intermediate.get("final_panel_access") != 0
                or intermediate.get("scientific_state_mutated") is not False
                or intermediate_binding.get("engineering_recovery_sha256") != sha256_file(receipt_path)
                or intermediate_binding.get("profile_sha256") != hashlib.sha256(git_bytes(intermediate_authority, PROFILE)).hexdigest()
                or intermediate_binding.get("release_sha256") != hashlib.sha256(git_bytes(intermediate_authority, RELEASE)).hexdigest()
                or intermediate_binding.get("router_sha256") != hashlib.sha256(git_bytes(intermediate_authority, ROUTER)).hexdigest()
                or committed_profile.get("frozen_inputs", {}).get("qualification_path") != str(intermediate_path)
                or committed_profile.get("frozen_inputs", {}).get("operational_source", {}).get("revision") != intermediate_binding.get("operational_source_sha")
                or committed_release.get("status") != "launch-authorized"
                or committed_router.get("status") != "launch-authorized"):
            raise Q08AuthorizationError("attempt-0009 intermediate qualification is not its committed zero-consumption receipt")

    run_root = (Path(profile["workspace"]) / profile["paths"]["runs"] / profile["run_id"]).resolve(strict=True)
    archive = run_root / RECOVERY8_CONTROL_REL / "attempt-0006-descriptor-misread-archive-20261002"
    files = {name: archive / name for name in (
        "formal-start-attempt-0006.json", "launch-receipt.json", "gpu-resource-receipt.json")}
    evidence = receipt.get("attempt_0006_evidence") or {}
    if set(evidence) != set(files):
        raise Q08AuthorizationError("attempt-0009 evidence inventory is incomplete")
    for name, path in files.items():
        item = evidence[name]
        if (not path.is_file() or path.is_symlink() or item.get("path") != str(path)
                or item.get("sha256") != sha256_file(path)):
            raise Q08AuthorizationError(f"attempt-0009 evidence hash/path mismatch: {name}")
    start = read_json(files["formal-start-attempt-0006.json"])
    launch = read_json(files["launch-receipt.json"])
    resource = read_json(files["gpu-resource-receipt.json"])
    if (start.get("status") != "FORMAL_START_COMMITTED"
            or start.get("formal_task_consumption_before") is not False
            or launch.get("exit_code") != 1
            or (launch.get("formal_start") or {}).get("receipt_sha256") != start.get("receipt_sha256")
            or _normalize_sha((launch.get("formal_start") or {}).get("sha256")) != sha256_file(files["formal-start-attempt-0006.json"])
            or resource.get("status") != "failed" or resource.get("stage") != "launch"
            or resource.get("return_code") != 1 or resource.get("gpu_uuids") != start.get("selected_gpu_uuids")):
        raise Q08AuthorizationError("attempt-0009 receipts do not prove the exact pre-admission failure")
    if (run_root / RECOVERY8_CONTROL_REL / "admission-claim.json").exists():
        raise Q08AuthorizationError("attempt-0009 found a prior recovery admission claim")
    claims = run_root / "control" / "claims"
    if claims.exists() and (claims.is_symlink() or any(claims.iterdir())):
        raise Q08AuthorizationError("attempt-0009 found consumed cell claims")
    for cell in CELL_IDS:
        leaf = run_root / "cells" / cell / "seen-eval"
        if leaf.exists() or leaf.is_symlink():
            raise Q08AuthorizationError("attempt-0009 found existing scientific cell output")
    if (run_root / RECOVERY9_CONTROL_REL / "admission-claim.json").exists():
        raise Q08AuthorizationError("attempt-0009 admission was already consumed")
    return receipt



def _validate_engineering_recovery_attempt10(repo: Path, profile: dict, release: dict, ref: dict) -> dict:
    """Permit one control-only retry after Start7 failed before any cell claim."""
    import subprocess
    def git_bytes(revision: str, path: str) -> bytes:
        return subprocess.run(["git", "-C", str(repo), "show", f"{revision}:{path}"], check=True, capture_output=True).stdout
    receipt_path = bound_path(repo, RECOVERY10_RECEIPT_REL)
    expect_file(receipt_path, str(ref.get("sha256", "")))
    raw = receipt_path.read_bytes()
    if git_bytes("HEAD", RECOVERY10_RECEIPT_REL) != raw:
        raise Q08AuthorizationError("attempt-0010 recovery receipt is not committed exact bytes")
    receipt = read_json(receipt_path)
    prior_profile = json.loads(git_bytes(RECOVERY10_PRIOR_AUTHORITY_SHA, PROFILE))
    prior_release = json.loads(git_bytes(RECOVERY10_PRIOR_AUTHORITY_SHA, RELEASE))
    prior_router = json.loads(git_bytes(RECOVERY10_PRIOR_AUTHORITY_SHA, ROUTER))
    prior_recovery = prior_release.get("engineering_recovery") or {}
    if (prior_profile.get("frozen_inputs", {}).get("operational_source", {}).get("revision") != RECOVERY10_PRIOR_SOURCE_SHA
            or prior_recovery.get("attempt") != 9 or prior_recovery.get("path") != RECOVERY9_RECEIPT_REL
            or prior_recovery.get("sha256") != RECOVERY10_PREDECESSOR_SHA or prior_router.get("status") != "launch-authorized"):
        raise Q08AuthorizationError("attempt-0010 predecessor authority is not the exact recovery9 authority")
    _validate_engineering_recovery_attempt9(repo, prior_profile, prior_release, prior_recovery)
    expected = {
        "schema": "openevo.q08.zero-consumption-engineering-recovery.v1", "status": "ZERO_CONSUMPTION_RECOVERY_PASS",
        "campaign_id": profile["campaign_id"], "run_id": profile["run_id"], "attempt": 10,
        "prior_authority_git_sha": RECOVERY10_PRIOR_AUTHORITY_SHA, "prior_operational_source_git_sha": RECOVERY10_PRIOR_SOURCE_SHA,
        "predecessor_recovery_sha256": RECOVERY10_PREDECESSOR_SHA, "scientific_resume": False,
        "formal_task_consumed": 0, "task_bearing_calls": 0, "model_calls": 0, "episodes_consumed": 0,
        "cell_claims_consumed": 0, "final_panel_access": 0, "fresh_seen_episodes_max": 960,
        "reused_historical_episodes": 768, "program_total_maximum_including_reuse": 1920,
        "continuation": {"control_attempt": 8, "formal_start_basename": RECOVERY10_FORMAL_START_BASENAME,
            "context_namespace": f"{RECOVERY10_CONTROL_REL}/cells", "cell_claims_allowed": 10,
            "cell_resume_allowed": False, "run_id_changed": False, "matrix_changed": False,
            "panel_changed": False, "budget_changed": False},
        "failure": {"owner": "q08_seen_host_orchestrator._load_formal_start",
            "message": "READY router_path was compared to activation router instead of current-campaign candidate",
            "stage": "before-first-cell-claim", "cell_id": "ordinary-r080", "control_attempt": 7}}
    for key, value in expected.items():
        if receipt.get(key) != value: raise Q08AuthorizationError(f"attempt-0010 recovery mismatch: {key}")
    frozen = profile["frozen_inputs"]
    science = {"panel_sha256": PANEL_SHA, "matrix_sha256": MATRIX_SHA, "reuse_audit_sha256": REUSE_SHA,
        "probe_wrapper_sha256": WRAPPER_SHA, "model": frozen["model"], "model_revision": frozen["model_revision"],
        "seed_root": frozen["seed_root"], "memory_sha256": frozen["carrier_sha256"]["memory"],
        "skill_sha256": frozen["carrier_sha256"]["skill"], "agent_sha256": frozen["carrier_sha256"]["agent"]}
    if receipt.get("science_identity") != science: raise Q08AuthorizationError("attempt-0010 scientific identity changed")
    if (profile.get("campaign_id") != prior_profile.get("campaign_id") or profile.get("run_id") != prior_profile.get("run_id")
            or profile.get("image_id") != prior_profile.get("image_id") or profile.get("cells") != prior_profile.get("cells")
            or profile.get("sampling") != prior_profile.get("sampling") or profile.get("final_panel_access_allowed") is not False
            or profile.get("formal_training_allowed") is not False
            or profile.get("gpu_gate", {}).get("authorized_uuids") != prior_profile.get("gpu_gate", {}).get("authorized_uuids")
            or frozen.get("checkpoint_matrix") != prior_profile.get("frozen_inputs", {}).get("checkpoint_matrix")
            or frozen.get("panel_manifest") != prior_profile.get("frozen_inputs", {}).get("panel_manifest")
            or frozen.get("fresh_reuse_audit") != prior_profile.get("frozen_inputs", {}).get("fresh_reuse_audit")
            or release.get("budget") != prior_release.get("budget") or release.get("final_panel_access_limit") != 0
            or frozen.get("operational_source", {}).get("revision") != profile.get("source", {}).get("experiment", {}).get("revision")):
        raise Q08AuthorizationError("attempt-0010 changed frozen science, GPU pool, budget, or source binding")
    pm = re.fullmatch(r"(.+-qualification)-attempt(\d{4})/qualification\.json", str(prior_profile["frozen_inputs"]["qualification_path"]))
    cm = re.fullmatch(r"(.+-qualification)-attempt(\d{4})/qualification\.json", str(frozen["qualification_path"]))
    if pm is None or cm is None or pm.group(1) != cm.group(1) or int(cm.group(2)) <= int(pm.group(2)):
        raise Q08AuthorizationError("attempt-0010 did not bind a later sequential qualification")
    for attempt in range(int(pm.group(2))+1, int(cm.group(2))):
        intermediate_path=Path(f"{pm.group(1)}-attempt{attempt:04d}/qualification.json")
        if not intermediate_path.is_file() or intermediate_path.is_symlink():
            raise Q08AuthorizationError("attempt-0010 qualification sequence is missing an immutable intermediate")
        intermediate=read_json(intermediate_path); binding=intermediate.get("q08_binding") or {}
        authority_sha=str(intermediate.get("authority_git_sha", ""))
        committed_profile=json.loads(git_bytes(authority_sha, PROFILE))
        committed_release=json.loads(git_bytes(authority_sha, RELEASE))
        committed_router=json.loads(git_bytes(authority_sha, ROUTER))
        projection={key:value for key,value in intermediate.items() if key != "receipt_sha256"}
        digest=hashlib.sha256(json.dumps(projection,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
        if (intermediate.get("receipt_sha256") != digest or intermediate.get("status") != "ZERO_FORMAL_QUALIFIED"
                or intermediate.get("formal_task_budget") != 0 or intermediate.get("scientific_budget_consumed") != 0
                or intermediate.get("task_bearing_calls") != 0 or intermediate.get("model_calls") != 0
                or intermediate.get("training_calls") != 0 or intermediate.get("training_steps") != 0
                or intermediate.get("final_panel_access") != 0 or intermediate.get("scientific_state_mutated") is not False
                or binding.get("engineering_recovery_sha256") != sha256_file(receipt_path)
                or binding.get("profile_sha256") != hashlib.sha256(git_bytes(authority_sha, PROFILE)).hexdigest()
                or binding.get("release_sha256") != hashlib.sha256(git_bytes(authority_sha, RELEASE)).hexdigest()
                or binding.get("router_sha256") != hashlib.sha256(git_bytes(authority_sha, ROUTER)).hexdigest()
                or committed_profile.get("frozen_inputs", {}).get("qualification_path") != str(intermediate_path)
                or committed_profile.get("frozen_inputs", {}).get("operational_source", {}).get("revision") != binding.get("operational_source_sha")
                or committed_release.get("status") != "launch-authorized" or committed_router.get("status") != "launch-authorized"):
            raise Q08AuthorizationError("attempt-0010 intermediate qualification is not its committed zero-consumption receipt")
    run_root = (Path(profile["workspace"]) / profile["paths"]["runs"] / profile["run_id"]).resolve(strict=True)
    archive = run_root / RECOVERY10_ARCHIVE_REL; evidence = receipt.get("attempt_0007_evidence") or {}
    names = {"formal_start":"formal-start-attempt-0007.json", "ready":"ready-receipt.json",
        "prelaunch":"prelaunch-zero-formal-attempt-0003.json", "launch":"launch-receipt.json", "resource":"gpu-resource-receipt.json"}
    if set(evidence) != set(names): raise Q08AuthorizationError("attempt-0010 evidence inventory is incomplete")
    paths={}
    for key,name in names.items():
        path=archive/name; item=evidence[key]
        if not path.is_file() or path.is_symlink() or item.get("path") != str(path) or item.get("sha256") != sha256_file(path):
            raise Q08AuthorizationError(f"attempt-0010 evidence path/hash mismatch: {key}")
        paths[key]=path
    start=read_json(paths["formal_start"]); ready=read_json(paths["ready"]); prelaunch=read_json(paths["prelaunch"])
    launch=read_json(paths["launch"]); resource=read_json(paths["resource"]); selected=start.get("selected_gpu_uuids")
    ready_ref=(start.get("readiness") or {}).get("ready_receipt") or {}; authority=ready.get("authority") or {}
    original_ready=Path(str(ready_ref.get("path", "")))
    old_repo=Path(str(authority.get("router_path",""))).resolve(strict=True).parents[3]
    if (start.get("status") != "FORMAL_START_COMMITTED" or start.get("formal_task_consumption_before") is not False
            or start.get("formal_task_consumption_may_begin") is not True or start.get("run_id") != profile["run_id"]
            or start.get("campaign_id") != profile["campaign_id"] or (start.get("readiness") or {}).get("formal_start_basename") != "formal-start-attempt-0007.json"
            or not original_ready.is_file() or original_ready.is_symlink()
            or _normalize_sha(ready_ref.get("sha256")) != sha256_file(paths["ready"])
            or sha256_file(original_ready) != sha256_file(paths["ready"])
            or authority.get("router_path") != str(old_repo / CURRENT) or authority.get("router_release_pointer") != RELEASE
            or authority.get("router_status") != "launch-authorized" or ready.get("verdict") != "READY"
            or prelaunch.get("status") != "PASS" or prelaunch.get("scientific_budget_consumed") != 0
            or prelaunch.get("model_calls") != 0 or prelaunch.get("training_calls") != 0
            or launch.get("execution_mode") != "host-orchestrator" or launch.get("exit_code") != 1
            or _normalize_sha((launch.get("formal_start") or {}).get("sha256")) != sha256_file(paths["formal_start"])
            or _normalize_sha((launch.get("gpu_resource_receipt") or {}).get("sha256")) != sha256_file(paths["resource"])
            or resource.get("status") != "failed" or resource.get("stage") != "launch" or resource.get("return_code") != 1
            or resource.get("gpu_uuids") != selected or resource.get("gpu_hours",1) > 0.01
            or not isinstance(selected,list) or len(selected) != 1 or selected[0] not in profile["gpu_gate"]["authorized_uuids"]):
        raise Q08AuthorizationError("attempt-0010 evidence does not prove an exact pre-claim zero-consumption failure")
    claims=run_root/"control"/"claims"
    if claims.exists() and (claims.is_symlink() or any(claims.iterdir())): raise Q08AuthorizationError("attempt-0010 found consumed scientific cell claims")
    for cell in CELL_IDS:
        leaf=run_root/"cells"/cell/"seen-eval"
        if leaf.exists() or leaf.is_symlink(): raise Q08AuthorizationError("attempt-0010 found scientific cell output")
    if (run_root/RECOVERY9_CONTROL_REL/"admission-claim.json").exists(): raise Q08AuthorizationError("attempt-0010 predecessor recovery admission was consumed")
    if (run_root/RECOVERY10_CONTROL_REL/"admission-claim.json").exists(): raise Q08AuthorizationError("attempt-0010 admission was already consumed")
    return receipt


def _validate_engineering_recovery_attempt11(repo: Path, profile: dict, release: dict, ref: dict) -> dict:
    """Permit a new control attempt after attempt8 failed before its first cell claim."""
    def git_bytes(revision: str, path: str) -> bytes:
        return subprocess.run(["git", "-C", str(repo), "show", f"{revision}:{path}"], check=True, capture_output=True).stdout
    receipt_path=bound_path(repo, RECOVERY11_RECEIPT_REL);expect_file(receipt_path,str(ref.get("sha256","")))
    raw=receipt_path.read_bytes()
    if git_bytes("HEAD",RECOVERY11_RECEIPT_REL)!=raw:raise Q08AuthorizationError("attempt-0011 receipt is not committed exact bytes")
    receipt=read_json(receipt_path)
    prior_profile=json.loads(git_bytes(RECOVERY11_PRIOR_AUTHORITY_SHA,PROFILE));prior_release=json.loads(git_bytes(RECOVERY11_PRIOR_AUTHORITY_SHA,RELEASE));prior_router=json.loads(git_bytes(RECOVERY11_PRIOR_AUTHORITY_SHA,ROUTER))
    prior_ref=prior_release.get("engineering_recovery") or {}
    if (prior_profile.get("frozen_inputs",{}).get("operational_source",{}).get("revision")!=RECOVERY11_PRIOR_SOURCE_SHA
            or prior_ref.get("attempt")!=10 or prior_ref.get("path")!=RECOVERY10_RECEIPT_REL
            or prior_ref.get("sha256")!=RECOVERY11_PREDECESSOR_SHA or prior_router.get("status")!="launch-authorized"):
        raise Q08AuthorizationError("attempt-0011 predecessor authority is not the exact recovery10 authority")
    recovery10_path=bound_path(repo,RECOVERY10_RECEIPT_REL);expect_file(recovery10_path,RECOVERY11_PREDECESSOR_SHA)
    recovery10=read_json(recovery10_path)
    if git_bytes("HEAD",RECOVERY10_RECEIPT_REL)!=recovery10_path.read_bytes():raise Q08AuthorizationError("attempt-0010 predecessor receipt changed")
    expected={"schema":"openevo.q08.zero-consumption-engineering-recovery.v1","status":"ZERO_CONSUMPTION_RECOVERY_PASS",
        "campaign_id":profile["campaign_id"],"run_id":profile["run_id"],"attempt":11,
        "prior_authority_git_sha":RECOVERY11_PRIOR_AUTHORITY_SHA,"prior_operational_source_git_sha":RECOVERY11_PRIOR_SOURCE_SHA,
        "predecessor_recovery_sha256":RECOVERY11_PREDECESSOR_SHA,"scientific_resume":False,"formal_task_consumed":0,
        "task_bearing_calls":0,"model_calls":0,"episodes_consumed":0,"cell_claims_consumed":0,"final_panel_access":0,
        "fresh_seen_episodes_max":960,"reused_historical_episodes":768,"program_total_maximum_including_reuse":1920,
        "continuation":{"control_attempt":9,"formal_start_basename":RECOVERY11_FORMAL_START_BASENAME,
            "context_namespace":f"{RECOVERY11_CONTROL_REL}/cells","cell_claims_allowed":10,"cell_resume_allowed":False,
            "run_id_changed":False,"matrix_changed":False,"panel_changed":False,"budget_changed":False},
        "failure":{"owner":"q08_seen_launch_guard.resolve_engineering_recovery_document",
            "message":"NameError: name 'subprocess' is not defined","stage":"before-first-cell-claim","cell_id":"ordinary-r080","control_attempt":8}}
    for key,value in expected.items():
        if receipt.get(key)!=value:raise Q08AuthorizationError(f"attempt-0011 recovery mismatch: {key}")
    frozen=profile["frozen_inputs"]
    science={"panel_sha256":PANEL_SHA,"matrix_sha256":MATRIX_SHA,"reuse_audit_sha256":REUSE_SHA,"probe_wrapper_sha256":WRAPPER_SHA,
        "model":frozen["model"],"model_revision":frozen["model_revision"],"seed_root":frozen["seed_root"],
        "memory_sha256":frozen["carrier_sha256"]["memory"],"skill_sha256":frozen["carrier_sha256"]["skill"],"agent_sha256":frozen["carrier_sha256"]["agent"]}
    if receipt.get("science_identity")!=science:raise Q08AuthorizationError("attempt-0011 scientific identity changed")
    if (profile.get("campaign_id")!=prior_profile.get("campaign_id") or profile.get("run_id")!=prior_profile.get("run_id")
            or profile.get("image_id")!=prior_profile.get("image_id") or profile.get("cells")!=prior_profile.get("cells")
            or profile.get("sampling")!=prior_profile.get("sampling") or profile.get("gpu_gate")!=prior_profile.get("gpu_gate")
            or profile.get("final_panel_access_allowed") is not False or profile.get("formal_training_allowed") is not False
            or frozen.get("checkpoint_matrix")!=prior_profile.get("frozen_inputs",{}).get("checkpoint_matrix")
            or frozen.get("panel_manifest")!=prior_profile.get("frozen_inputs",{}).get("panel_manifest")
            or frozen.get("fresh_reuse_audit")!=prior_profile.get("frozen_inputs",{}).get("fresh_reuse_audit")
            or release.get("budget")!=prior_release.get("budget") or release.get("final_panel_access_limit")!=0
            or frozen.get("operational_source",{}).get("revision")!=profile.get("source",{}).get("experiment",{}).get("revision")):
        raise Q08AuthorizationError("attempt-0011 changed frozen science, GPU pool, budget, or source binding")
    prior_qual=Path(str(prior_profile["frozen_inputs"]["qualification_path"])); q=read_json(prior_qual); qb=q.get("q08_binding") or {}
    if (q.get("status")!="ZERO_FORMAL_QUALIFIED" or q.get("authority_git_sha")!=RECOVERY11_PRIOR_AUTHORITY_SHA
            or q.get("execution_sha")!=RECOVERY11_PRIOR_SOURCE_SHA or q.get("scientific_budget_consumed")!=0
            or q.get("model_calls")!=0 or q.get("task_bearing_calls")!=0 or q.get("training_calls")!=0
            or qb.get("engineering_recovery_sha256")!=RECOVERY11_PREDECESSOR_SHA
            or qb.get("profile_sha256")!=hashlib.sha256(git_bytes(RECOVERY11_PRIOR_AUTHORITY_SHA,PROFILE)).hexdigest()
            or qb.get("release_sha256")!=hashlib.sha256(git_bytes(RECOVERY11_PRIOR_AUTHORITY_SHA,RELEASE)).hexdigest()
            or qb.get("router_sha256")!=hashlib.sha256(git_bytes(RECOVERY11_PRIOR_AUTHORITY_SHA,ROUTER)).hexdigest()):
        raise Q08AuthorizationError("attempt-0011 predecessor qualification is not exact zero-formal evidence")
    run_root=(Path(profile["workspace"])/profile["paths"]["runs"]/profile["run_id"]).resolve(strict=True)
    archive=run_root/RECOVERY11_ARCHIVE_REL;evidence=receipt.get("attempt_0008_evidence") or {}
    names={"formal_start":"formal-start-attempt-0008.json","ready":"ready-receipt.json","prelaunch":"prelaunch-receipt.json",
        "launch":"launch-receipt.json","resource":"gpu-resource-receipt.json","admission":"admission-claim.json",
        "context":"ordinary-r080-launch-context.json","controller_log":"host-orchestrator.log","container_inspect":"container-inspect.json"}
    if set(evidence)!=set(names):raise Q08AuthorizationError("attempt-0011 evidence inventory is incomplete")
    paths={}
    for key,name in names.items():
        path=archive/name;item=evidence[key]
        if not path.is_file() or path.is_symlink() or item.get("path")!=str(path) or item.get("sha256")!=sha256_file(path):
            raise Q08AuthorizationError(f"attempt-0011 evidence path/hash mismatch: {key}")
        paths[key]=path
    start=read_json(paths["formal_start"]);ready=read_json(paths["ready"]);pre=read_json(paths["prelaunch"])
    launch=read_json(paths["launch"]);resource=read_json(paths["resource"]);admission=read_json(paths["admission"])
    context=read_json(paths["context"]);inspect_doc=read_json(paths["container_inspect"])
    ready_ref=(start.get("readiness") or {}).get("ready_receipt") or {};current_profile_sha=hashlib.sha256(git_bytes(RECOVERY11_PRIOR_AUTHORITY_SHA,PROFILE)).hexdigest()
    if (start.get("status")!="FORMAL_START_COMMITTED" or start.get("formal_task_consumption_before") is not False
            or start.get("formal_task_consumption_may_begin") is not True or start.get("run_id")!=profile["run_id"]
            or start.get("campaign_id")!=profile["campaign_id"] or Path(str((start.get("readiness") or {}).get("formal_start_basename",""))).name!="formal-start-attempt-0008.json"
            or _normalize_sha(ready_ref.get("sha256"))!=sha256_file(paths["ready"]) or ready.get("verdict")!="READY"
            or (ready.get("profile") or {}).get("sha256")!="sha256:"+current_profile_sha
            or pre.get("status")!="PASS" or pre.get("scientific_budget_consumed")!=0 or pre.get("model_calls")!=0 or pre.get("training_calls")!=0
            or launch.get("execution_mode")!="host-orchestrator" or launch.get("exit_code")!=1
            or _normalize_sha((launch.get("formal_start") or {}).get("sha256"))!=sha256_file(paths["formal_start"])
            or resource.get("status")!="failed" or resource.get("stage")!="launch" or resource.get("return_code")!=1
            or resource.get("gpu_uuids")!=start.get("selected_gpu_uuids") or resource.get("gpu_hours",1)>0.01
            or admission.get("status")!="CONSUMED_FOR_CONTROL_ATTEMPT_0008"
            or admission.get("recovery_receipt_sha256")!=RECOVERY11_PREDECESSOR_SHA
            or admission.get("formal_start_sha256")!=sha256_file(paths["formal_start"])
            or admission.get("qualification_sha256")!=sha256_file(prior_qual)
            or admission.get("formal_task_consumed_before_claim")!=0 or admission.get("cell_claims_consumed_before_claim")!=0
            or admission.get("episodes_consumed_before_claim")!=0 or admission.get("scientific_resume") is not False
            or context.get("cell_id")!="ordinary-r080" or context.get("control_attempt")!=8 or context.get("resume_allowed") is not False
            or "NameError: name 'subprocess' is not defined" not in paths["controller_log"].read_text()
            or inspect_doc.get("Name")!="/experiment-host-q08-post-advisor-q08-seen-retention-202610010742-ordinary-r080-a0008"
            or inspect_doc.get("State",{}).get("Status")!="exited" or inspect_doc.get("State",{}).get("ExitCode")!=1):
        raise Q08AuthorizationError("attempt-0011 evidence does not prove exact pre-claim zero-consumption failure")
    claims=run_root/"control"/"claims"
    if claims.exists() and (claims.is_symlink() or any(claims.iterdir())):raise Q08AuthorizationError("attempt-0011 found cell claims")
    for cell in CELL_IDS:
        leaf=run_root/"cells"/cell/"seen-eval"
        if leaf.exists() or leaf.is_symlink():raise Q08AuthorizationError("attempt-0011 found scientific output")
    if (run_root/RECOVERY11_CONTROL_REL/"admission-claim.json").exists():raise Q08AuthorizationError("attempt-0011 admission already consumed")
    return receipt

def _validate_engineering_recovery_attempt12(repo: Path, profile: dict, release: dict, ref: dict) -> dict:
    """Permit a new control attempt after attempt9 failed before its first cell claim."""
    def git_bytes(revision: str, path: str) -> bytes:
        return subprocess.run(["git", "-C", str(repo), "show", f"{revision}:{path}"], check=True, capture_output=True).stdout
    receipt_path=bound_path(repo, RECOVERY12_RECEIPT_REL);expect_file(receipt_path,str(ref.get("sha256","")))
    raw=receipt_path.read_bytes()
    if git_bytes("HEAD",RECOVERY12_RECEIPT_REL)!=raw:raise Q08AuthorizationError("attempt-0012 receipt is not committed exact bytes")
    receipt=read_json(receipt_path)
    prior_profile=json.loads(git_bytes(RECOVERY12_PRIOR_AUTHORITY_SHA,PROFILE));prior_release=json.loads(git_bytes(RECOVERY12_PRIOR_AUTHORITY_SHA,RELEASE));prior_router=json.loads(git_bytes(RECOVERY12_PRIOR_AUTHORITY_SHA,ROUTER))
    prior_ref=prior_release.get("engineering_recovery") or {}
    if (prior_profile.get("frozen_inputs",{}).get("operational_source",{}).get("revision")!=RECOVERY12_PRIOR_SOURCE_SHA
            or prior_ref.get("attempt")!=11 or prior_ref.get("path")!=RECOVERY11_RECEIPT_REL
            or prior_ref.get("sha256")!=RECOVERY12_PREDECESSOR_SHA or prior_router.get("status")!="launch-authorized"):
        raise Q08AuthorizationError("attempt-0012 predecessor authority is not the exact recovery11 authority")
    recovery11_path=bound_path(repo,RECOVERY11_RECEIPT_REL);expect_file(recovery11_path,RECOVERY12_PREDECESSOR_SHA)
    recovery11=read_json(recovery11_path)
    if git_bytes("HEAD",RECOVERY11_RECEIPT_REL)!=recovery11_path.read_bytes():raise Q08AuthorizationError("attempt-0011 predecessor receipt changed")
    expected={"schema":"openevo.q08.zero-consumption-engineering-recovery.v1","status":"ZERO_CONSUMPTION_RECOVERY_PASS",
        "campaign_id":profile["campaign_id"],"run_id":profile["run_id"],"attempt":12,
        "prior_authority_git_sha":RECOVERY12_PRIOR_AUTHORITY_SHA,"prior_operational_source_git_sha":RECOVERY12_PRIOR_SOURCE_SHA,
        "predecessor_recovery_sha256":RECOVERY12_PREDECESSOR_SHA,"scientific_resume":False,"formal_task_consumed":0,
        "task_bearing_calls":0,"model_calls":0,"episodes_consumed":0,"cell_claims_consumed":0,"final_panel_access":0,
        "fresh_seen_episodes_max":960,"reused_historical_episodes":768,"program_total_maximum_including_reuse":1920,
        "continuation":{"control_attempt":10,"formal_start_basename":RECOVERY12_FORMAL_START_BASENAME,
            "context_namespace":f"{RECOVERY12_CONTROL_REL}/cells","cell_claims_allowed":10,"cell_resume_allowed":False,
            "run_id_changed":False,"matrix_changed":False,"panel_changed":False,"budget_changed":False},
        "failure":{"owner":"q08_seen_launch_guard.authorize_probe",
            "message":"PermissionError: [Errno 13] Permission denied:","stage":"before-first-cell-claim","cell_id":"ordinary-r080","control_attempt":9}}
    for key,value in expected.items():
        if receipt.get(key)!=value:raise Q08AuthorizationError(f"attempt-0012 recovery mismatch: {key}")
    frozen=profile["frozen_inputs"]
    science={"panel_sha256":PANEL_SHA,"matrix_sha256":MATRIX_SHA,"reuse_audit_sha256":REUSE_SHA,"probe_wrapper_sha256":WRAPPER_SHA,
        "model":frozen["model"],"model_revision":frozen["model_revision"],"seed_root":frozen["seed_root"],
        "memory_sha256":frozen["carrier_sha256"]["memory"],"skill_sha256":frozen["carrier_sha256"]["skill"],"agent_sha256":frozen["carrier_sha256"]["agent"]}
    if receipt.get("science_identity")!=science:raise Q08AuthorizationError("attempt-0012 scientific identity changed")
    if (profile.get("campaign_id")!=prior_profile.get("campaign_id") or profile.get("run_id")!=prior_profile.get("run_id")
            or profile.get("image_id")!=prior_profile.get("image_id") or profile.get("cells")!=prior_profile.get("cells")
            or profile.get("sampling")!=prior_profile.get("sampling") or profile.get("gpu_gate")!=prior_profile.get("gpu_gate")
            or profile.get("final_panel_access_allowed") is not False or profile.get("formal_training_allowed") is not False
            or frozen.get("checkpoint_matrix")!=prior_profile.get("frozen_inputs",{}).get("checkpoint_matrix")
            or frozen.get("panel_manifest")!=prior_profile.get("frozen_inputs",{}).get("panel_manifest")
            or frozen.get("fresh_reuse_audit")!=prior_profile.get("frozen_inputs",{}).get("fresh_reuse_audit")
            or release.get("budget")!=prior_release.get("budget") or release.get("final_panel_access_limit")!=0
            or frozen.get("operational_source",{}).get("revision")!=profile.get("source",{}).get("experiment",{}).get("revision")):
        raise Q08AuthorizationError("attempt-0012 changed frozen science, GPU pool, budget, or source binding")
    prior_qual=Path(str(prior_profile["frozen_inputs"]["qualification_path"])); q=read_json(prior_qual); qb=q.get("q08_binding") or {}
    if (q.get("status")!="ZERO_FORMAL_QUALIFIED" or q.get("authority_git_sha")!=RECOVERY12_PRIOR_AUTHORITY_SHA
            or q.get("execution_sha")!=RECOVERY12_PRIOR_SOURCE_SHA or q.get("scientific_budget_consumed")!=0
            or q.get("model_calls")!=0 or q.get("task_bearing_calls")!=0 or q.get("training_calls")!=0
            or qb.get("engineering_recovery_sha256")!=RECOVERY12_PREDECESSOR_SHA
            or qb.get("profile_sha256")!=hashlib.sha256(git_bytes(RECOVERY12_PRIOR_AUTHORITY_SHA,PROFILE)).hexdigest()
            or qb.get("release_sha256")!=hashlib.sha256(git_bytes(RECOVERY12_PRIOR_AUTHORITY_SHA,RELEASE)).hexdigest()
            or qb.get("router_sha256")!=hashlib.sha256(git_bytes(RECOVERY12_PRIOR_AUTHORITY_SHA,ROUTER)).hexdigest()):
        raise Q08AuthorizationError("attempt-0012 predecessor qualification is not exact zero-formal evidence")
    run_root=(Path(profile["workspace"])/profile["paths"]["runs"]/profile["run_id"]).resolve(strict=True)
    archive=run_root/RECOVERY12_ARCHIVE_REL;evidence=receipt.get("attempt_0009_evidence") or {}
    names={"formal_start":"formal-start-attempt-0009.json","ready":"ready-receipt.json","prelaunch":"prelaunch-receipt.json",
        "launch":"launch-receipt.json","resource":"gpu-resource-receipt.json","admission":"admission-claim.json",
        "context":"ordinary-r080-launch-context.json","controller_log":"host-orchestrator.log","container_inspect":"container-inspect.json"}
    if set(evidence)!=set(names):raise Q08AuthorizationError("attempt-0012 evidence inventory is incomplete")
    paths={}
    for key,name in names.items():
        path=archive/name;item=evidence[key]
        if not path.is_file() or path.is_symlink() or item.get("path")!=str(path) or item.get("sha256")!=sha256_file(path):
            raise Q08AuthorizationError(f"attempt-0012 evidence path/hash mismatch: {key}")
        paths[key]=path
    start=read_json(paths["formal_start"]);ready=read_json(paths["ready"]);pre=read_json(paths["prelaunch"])
    launch=read_json(paths["launch"]);resource=read_json(paths["resource"]);admission=read_json(paths["admission"])
    context=read_json(paths["context"]);inspect_doc=read_json(paths["container_inspect"])
    ready_ref=(start.get("readiness") or {}).get("ready_receipt") or {};current_profile_sha=hashlib.sha256(git_bytes(RECOVERY12_PRIOR_AUTHORITY_SHA,PROFILE)).hexdigest()
    if (start.get("status")!="FORMAL_START_COMMITTED" or start.get("formal_task_consumption_before") is not False
            or start.get("formal_task_consumption_may_begin") is not True or start.get("run_id")!=profile["run_id"]
            or start.get("campaign_id")!=profile["campaign_id"] or Path(str((start.get("readiness") or {}).get("formal_start_basename",""))).name!="formal-start-attempt-0009.json"
            or _normalize_sha(ready_ref.get("sha256"))!=sha256_file(paths["ready"]) or ready.get("verdict")!="READY"
            or (ready.get("profile") or {}).get("sha256")!="sha256:"+current_profile_sha
            or pre.get("status")!="PASS" or pre.get("scientific_budget_consumed")!=0 or pre.get("model_calls")!=0 or pre.get("training_calls")!=0
            or launch.get("execution_mode")!="host-orchestrator" or launch.get("exit_code")!=1
            or _normalize_sha((launch.get("formal_start") or {}).get("sha256"))!=sha256_file(paths["formal_start"])
            or resource.get("status")!="failed" or resource.get("stage")!="launch" or resource.get("return_code")!=1
            or resource.get("gpu_uuids")!=start.get("selected_gpu_uuids") or resource.get("gpu_hours",1)>0.01
            or admission.get("status")!="CONSUMED_FOR_CONTROL_ATTEMPT_0009"
            or admission.get("recovery_receipt_sha256")!=RECOVERY12_PREDECESSOR_SHA
            or admission.get("formal_start_sha256")!=sha256_file(paths["formal_start"])
            or admission.get("qualification_sha256")!=sha256_file(prior_qual)
            or admission.get("formal_task_consumed_before_claim")!=0 or admission.get("cell_claims_consumed_before_claim")!=0
            or admission.get("episodes_consumed_before_claim")!=0 or admission.get("scientific_resume") is not False
            or context.get("cell_id")!="ordinary-r080" or context.get("control_attempt")!=9 or context.get("resume_allowed") is not False
            or "PermissionError: [Errno 13] Permission denied:" not in paths["controller_log"].read_text()
            or str(prior_qual) not in paths["controller_log"].read_text()
            or inspect_doc.get("Name")!="/experiment-host-q08-post-advisor-q08-seen-retention-202610010742-ordinary-r080-a0009"
            or inspect_doc.get("State",{}).get("Status")!="exited" or inspect_doc.get("State",{}).get("ExitCode")!=1):
        raise Q08AuthorizationError("attempt-0012 evidence does not prove exact pre-claim zero-consumption failure")
    claims=run_root/"control"/"claims"
    if claims.exists() and (claims.is_symlink() or any(claims.iterdir())):raise Q08AuthorizationError("attempt-0012 found cell claims")
    for cell in CELL_IDS:
        leaf=run_root/"cells"/cell/"seen-eval"
        if leaf.exists() or leaf.is_symlink():raise Q08AuthorizationError("attempt-0012 found scientific output")
    if (run_root/RECOVERY12_CONTROL_REL/"admission-claim.json").exists():raise Q08AuthorizationError("attempt-0012 admission already consumed")
    return receipt

def _validate_engineering_recovery_attempt6(repo: Path, profile: dict, release: dict, ref: dict) -> dict:
    """Admit one fresh control attempt after the exact attempt3 pre-container path failure."""
    import subprocess
    path = bound_path(repo, RECOVERY6_RECEIPT_REL)
    expect_file(path, str(ref.get("sha256", "")))
    committed = subprocess.run(["git", "-C", str(repo), "show", f"HEAD:{RECOVERY6_RECEIPT_REL}"],
                               check=True, capture_output=True).stdout
    if committed != path.read_bytes():
        raise Q08AuthorizationError("attempt-0006 recovery receipt is not the committed exact authority bytes")
    predecessor = bound_path(repo, RECOVERY5_RECEIPT_REL)
    expect_file(predecessor, RECOVERY6_PREDECESSOR_SHA)
    predecessor_bytes = subprocess.run(["git", "-C", str(repo), "show", f"HEAD:{RECOVERY5_RECEIPT_REL}"],
                                       check=True, capture_output=True).stdout
    if predecessor_bytes != predecessor.read_bytes():
        raise Q08AuthorizationError("attempt-0006 predecessor receipt is not committed exact bytes")
    receipt = read_json(path)
    expected = {
        "schema": "openevo.q08.zero-consumption-engineering-recovery.v1",
        "status": "ZERO_CONSUMPTION_RECOVERY_PASS",
        "campaign_id": profile["campaign_id"], "run_id": profile["run_id"], "attempt": 6,
        "prior_authority_git_sha": RECOVERY6_PRIOR_AUTHORITY_SHA,
        "prior_operational_source_git_sha": RECOVERY6_PRIOR_SOURCE_SHA,
        "predecessor_recovery_sha256": RECOVERY6_PREDECESSOR_SHA,
        "prior_admission_claim_sha256": RECOVERY6_PRIOR_ADMISSION_SHA,
        "scientific_resume": False, "formal_task_consumed": 0, "task_bearing_calls": 0,
        "model_calls": 0, "episodes_consumed": 0, "cell_claims_consumed": 0,
        "final_panel_access": 0, "fresh_seen_episodes_max": 960,
        "reused_historical_episodes": 768, "program_total_maximum_including_reuse": 1920,
        "continuation": {"control_attempt": 4, "formal_start_basename": RECOVERY6_FORMAL_START_BASENAME,
            "context_namespace": f"{RECOVERY6_CONTROL_REL}/cells",
            "cell_claims_allowed": 10, "cell_resume_allowed": False, "run_id_changed": False,
            "matrix_changed": False, "panel_changed": False, "budget_changed": False},
        "failure": {"owner": "q08_seen_host_orchestrator._require_uid1001_dir",
            "message": "attempt-0003 context namespace was root:root 0755; rejected before Q08 container construction",
            "stage": "before-first-cell-container-dispatch", "cell_id": "ordinary-r080"},
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise Q08AuthorizationError(f"attempt-0006 recovery mismatch: {key}")
    frozen = profile["frozen_inputs"]
    science = {"panel_sha256": PANEL_SHA, "matrix_sha256": MATRIX_SHA, "reuse_audit_sha256": REUSE_SHA,
        "probe_wrapper_sha256": WRAPPER_SHA, "model": frozen["model"], "model_revision": frozen["model_revision"],
        "seed_root": frozen["seed_root"], "memory_sha256": frozen["carrier_sha256"]["memory"],
        "skill_sha256": frozen["carrier_sha256"]["skill"], "agent_sha256": frozen["carrier_sha256"]["agent"]}
    if receipt.get("science_identity") != science:
        raise Q08AuthorizationError("attempt-0006 recovery scientific identity changed")
    run_root = (Path(profile["workspace"]) / profile["paths"]["runs"] / profile["run_id"]).resolve(strict=True)
    runs_root = (Path(profile["workspace"]) / profile["paths"]["runs"]).resolve(strict=True)
    evidence_root = (run_root / RECOVERY6_ARCHIVE_REL).resolve(strict=True)
    evidence = receipt.get("attempt_0005_evidence")
    required = {"formal_start", "ready", "prelaunch", "qualification", "launch", "resource",
                "prior_admission", "preexisting_context", "namespace_inventory", "reference_graph"}
    if not isinstance(evidence, dict) or set(evidence) != required:
        raise Q08AuthorizationError("attempt-0006 evidence inventory is incomplete")
    prior_profile_raw = subprocess.run(["git", "-C", str(repo), "show",
        f"{RECOVERY6_PRIOR_AUTHORITY_SHA}:{PROFILE}"], check=True, capture_output=True).stdout
    old_authority_profile = json.loads(prior_profile_raw)
    prior_start_path = runs_root / f"{profile['run_id']}-readiness-attempt0008/control/formal-start-attempt-0003.json"
    if Path(str(evidence["formal_start"].get("path", ""))).resolve(strict=True) != prior_start_path.resolve():
        raise Q08AuthorizationError("attempt-0006 FormalStart evidence path is not the exact attempt3 source")
    prior_start = read_json(prior_start_path)
    ready_ref_from_start = (prior_start.get("readiness") or {}).get("ready_receipt") or {}
    prelaunch_ref_from_start = prior_start.get("prelaunch_receipt") or {}
    if not ready_ref_from_start.get("path") or not prelaunch_ref_from_start.get("path"):
        raise Q08AuthorizationError("attempt-0006 Start3 lacks its required READY/prelaunch references")
    start_ready_path = Path(str(ready_ref_from_start["path"])).resolve(strict=True)
    start_prelaunch_path = Path(str(prelaunch_ref_from_start["path"])).resolve(strict=True)
    qualification_path = Path(str(old_authority_profile["frozen_inputs"]["qualification_path"])).resolve(strict=True)
    expected_roots = {
        "formal_start": prior_start_path.parent,
        "ready": start_ready_path.parent,
        "prelaunch": start_prelaunch_path.parent,
        "qualification": qualification_path.parent,
        "launch": evidence_root, "resource": evidence_root,
        "prior_admission": evidence_root, "preexisting_context": evidence_root,
        "namespace_inventory": evidence_root, "reference_graph": evidence_root,
    }
    expected_names = {
        "formal_start": "formal-start-attempt-0003.json",
        "ready": start_ready_path.name,
        "prelaunch": start_prelaunch_path.name,
        "qualification": qualification_path.name,
        "launch": "launch-receipt-attempt-0003.json",
        "resource": "gpu-resource-receipt-attempt-0003.json",
        "prior_admission": "admission-claim-attempt-0005.json",
        "preexisting_context": "preexisting-context-attempt-0003.json",
        "namespace_inventory": "namespace-inventory-attempt-0003.json",
        "reference_graph": "formal-start-reference-graph-attempt-0003.json",
    }
    paths = {}
    for key, item in evidence.items():
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise Q08AuthorizationError(f"attempt-0006 evidence reference malformed: {key}")
        actual = Path(str(item["path"])).resolve(strict=True)
        if actual.is_symlink() or not actual.is_file() or actual.parent != expected_roots[key].resolve():
            raise Q08AuthorizationError(f"attempt-0006 evidence path outside immutable scope: {key}")
        if key == "ready":
            if not actual.name.startswith("execution-readiness-") or actual.suffix != ".json":
                raise Q08AuthorizationError("attempt-0006 READY reference is not write-once")
        elif actual.name != expected_names[key]:
            raise Q08AuthorizationError(f"attempt-0006 evidence basename drift: {key}")
        expect_file(actual, str(item["sha256"]))
        expected_digests = {
            "formal_start": "f72170cbadb658dbac8af36c9d8e4b05418b70e873172b7b462679533085470e",
            "ready": _normalize_sha(ready_ref_from_start.get("sha256")),
            "prelaunch": "02d356847a0c2de33734f26f841b25b506b3a4e988c385e58ca9adf018c9ee0c",
            "qualification": "29e15820f0abedf37f11856b222f9cf14ec3e39be9c85e9b904d4ed5ed9eb6e4",
            "launch": "9d96abd4f1291ec90ae63458d4e0e338deee28630856160803695293542b72b5",
            "resource": "7c230caee42d50df374a7d48ac8386e1f0c2f8065ac2fea4bb33f268e0783859",
            "prior_admission": RECOVERY6_PRIOR_ADMISSION_SHA,
            "preexisting_context": "313f6050f0610099b46c09a1eb0aff614cced7155b0315cf4322780e09156e86",
            "namespace_inventory": "d9c3b360e010f319b5577409abaede999af468dc7003673768274a8cd85f42fb",
            "reference_graph": "7105807f7239068d43a08f9607ef645c4a56427e12d5f2810b69f6ca9d271dd3",
        }
        if sha256_file(actual) != expected_digests[key]:
            raise Q08AuthorizationError(f"attempt-0006 evidence differs from the immutable reviewed bytes: {key}")
        paths[key] = actual
    start = read_json(paths["formal_start"]); ready = read_json(paths["ready"])
    prelaunch = read_json(paths["prelaunch"]); qualification = read_json(paths["qualification"])
    launch = read_json(paths["launch"]); resource = read_json(paths["resource"])
    prior_admission = read_json(paths["prior_admission"])
    prior_context = read_json(paths["preexisting_context"]); inventory = read_json(paths["namespace_inventory"])
    graph = read_json(paths["reference_graph"])
    prior_authority_root = Path(str((ready.get("authority") or {}).get("router_path", ""))).resolve().parents[3]
    source_docs = [
        ("formal_start", paths["formal_start"]), ("ready", paths["ready"]),
        ("prelaunch", paths["prelaunch"]), ("qualification", paths["qualification"]),
        ("launch", paths["launch"]), ("resource", paths["resource"]),
        ("prior_admission", paths["prior_admission"]), ("preexisting_context", paths["preexisting_context"]),
        ("namespace_inventory", paths["namespace_inventory"]),
    ]
    expected_source_docs = [{"name": name, "path": str(doc), "sha256": sha256_file(doc)} for name, doc in source_docs]
    computed_edges = []
    for name, source_doc in source_docs:
        source_value = read_json(source_doc)
        def collect_refs(value, pointer="$"):
            if isinstance(value, dict):
                if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
                    raw_target = Path(value["path"])
                    declared = _normalize_sha(value["sha256"])
                    if name == "launch" and pointer == "$/gpu_resource_receipt":
                        # Start3 names the mutable run-root alias. Resolve its historical bytes only
                        # through the independently archived, SHA-validated attempt3 resource receipt.
                        alias = (Path(old_authority_profile["workspace"]) / old_authority_profile["paths"]["runs"]
                                 / old_authority_profile["run_id"] / "gpu-resource-receipt.json").resolve()
                        target = (raw_target if raw_target.is_absolute() else prior_authority_root / raw_target).resolve()
                        archived_resource_sha = sha256_file(paths["resource"])
                        if (target != alias or declared != archived_resource_sha
                                or inventory.get("source_attempt_files", {}).get("gpu-resource-receipt-attempt-0003.json") != archived_resource_sha):
                            raise Q08AuthorizationError("attempt-0006 resource alias is not backed by the exact immutable archive bytes")
                        actual = archived_resource_sha
                    else:
                        target = (raw_target if raw_target.is_absolute() else prior_authority_root / raw_target).resolve(strict=True)
                        actual = sha256_file(target)
                    computed_edges.append({"source": str(source_doc), "pointer": pointer, "target": str(target),
                        "declared_sha256": declared, "actual_sha256": actual,
                        "verdict": "MATCH" if declared == actual else "STALE_REFERENCE"})
                for key, child in sorted(value.items()):
                    collect_refs(child, pointer + "/" + str(key).replace("~", "~0").replace("/", "~1"))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    collect_refs(child, pointer + "/" + str(index))
        collect_refs(source_value)
    stale_context_edges = [edge for edge in computed_edges if edge["verdict"] == "STALE_REFERENCE"]
    if (graph.get("schema") != "openevo.q08.formal-start-reference-graph.v1"
            or graph.get("status") != "PASS_WITH_EXPECTED_STALE_CONTEXT_REFERENCE"
            or graph.get("source_documents") != expected_source_docs
            or graph.get("edge_count") != len(computed_edges)
            or graph.get("edges") != computed_edges
            or len(stale_context_edges) != 3
            or any(edge["source"] != str(paths["preexisting_context"]) for edge in stale_context_edges)
            or {edge["pointer"] for edge in stale_context_edges} != {"$/profile", "$/release", "$/router"}):
        raise Q08AuthorizationError("attempt-0006 Start3 reference graph is incomplete or differs from actual owner readbacks")
    selected = "GPU-00000000-0000-0000-0000-000000000000"
    old_profile_sha = hashlib.sha256(prior_profile_raw).hexdigest()
    old_release_sha = hashlib.sha256(subprocess.run(
        ["git", "-C", str(repo), "show", f"{RECOVERY6_PRIOR_AUTHORITY_SHA}:{RELEASE}"],
        check=True, capture_output=True).stdout).hexdigest()
    old_router_sha = hashlib.sha256(subprocess.run(
        ["git", "-C", str(repo), "show", f"{RECOVERY6_PRIOR_AUTHORITY_SHA}:{ROUTER}"],
        check=True, capture_output=True).stdout).hexdigest()
    start_ready = (start.get("readiness") or {}).get("ready_receipt") or {}
    start_profile = (start.get("readiness") or {}).get("profile") or {}
    start_prelaunch = start.get("prelaunch_receipt") or {}
    archived_ready_projection = receipt.get("additional_archived_ready_projection") or {}
    if (archived_ready_projection != {
            "path": str(evidence_root / "ready-receipt-attempt-0003.json"),
            "sha256": "f9faedfd002672e13bd8de6970f7c3fa3783bd7dee1bdbc1acef6c5b01df6509",
            "role": "Preserved extra readiness projection; not referenced by FormalStart attempt3"}):
        raise Q08AuthorizationError("attempt-0006 archive distinguishes the extra readiness projection from the Start3 READY")
    if (inventory.get("source_attempt_files", {}).get("ready-receipt-attempt-0003.json")
            != archived_ready_projection["sha256"]):
        raise Q08AuthorizationError("attempt-0006 inventory does not bind the archived extra READY projection")
    if (start.get("status") != "FORMAL_START_COMMITTED" or start.get("run_id") != profile["run_id"]
            or start.get("campaign_id") != profile["campaign_id"] or start.get("selected_gpu_uuids") != [selected]
            or start.get("formal_task_consumption_before") is not False
            or start.get("scientific_authority_granted_by_receipt") is not False
            or start_ready.get("path") != str(paths["ready"])
            or _normalize_sha(start_ready.get("sha256")) != sha256_file(paths["ready"])
            or start_prelaunch.get("path") != str(paths["prelaunch"])
            or _normalize_sha(start_prelaunch.get("sha256")) != sha256_file(paths["prelaunch"])
            or _normalize_sha(start_profile.get("sha256")) != old_profile_sha
            or Path(str(start_profile.get("path", ""))).name != Path(PROFILE).name):
        raise Q08AuthorizationError("attempt-0006 predecessor FormalStart is not exact zero-task evidence")
    if (ready.get("verdict") != "READY" or ready.get("profile", {}).get("sha256") != "sha256:" + old_profile_sha
            or ready.get("release_receipt", {}).get("sha256") != "sha256:" + old_release_sha
            or (ready.get("authority") or {}).get("router_status") != "launch-authorized"):
        raise Q08AuthorizationError("attempt-0006 READY does not bind the exact predecessor authority")
    if (prelaunch.get("status") != "PASS" or prelaunch.get("formal_task_consumed") is not False
            or prelaunch.get("scientific_budget_consumed") != 0 or prelaunch.get("model_calls") != 0
            or prelaunch.get("training_calls") != 0 or prelaunch.get("selected_gpu_uuids") != [selected]):
        raise Q08AuthorizationError("attempt-0006 prelaunch was not a zero-consumption PASS")
    qprojection = {key: value for key, value in qualification.items() if key != "receipt_sha256"}
    q_expected_receipt = hashlib.sha256(json.dumps(qprojection, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    prior_binding = qualification.get("q08_binding") or {}
    if (qualification.get("receipt_sha256") != q_expected_receipt
            or qualification.get("status") != "ZERO_FORMAL_QUALIFIED"
            or qualification.get("authority_git_sha") != RECOVERY6_PRIOR_AUTHORITY_SHA
            or qualification.get("execution_sha") != RECOVERY6_PRIOR_SOURCE_SHA
            or qualification.get("model_calls") != 0 or qualification.get("task_bearing_calls") != 0
            or qualification.get("selected_gpu_uuids") != [selected]
            or prior_binding.get("profile_sha256") != old_profile_sha
            or prior_binding.get("release_sha256") != old_release_sha
            or prior_binding.get("router_sha256") != old_router_sha
            or prior_binding.get("engineering_recovery_sha256") != RECOVERY6_PREDECESSOR_SHA):
        raise Q08AuthorizationError("attempt-0006 qualification is not exact zero-formal evidence")
    admission_sha = hashlib.sha256(paths["prior_admission"].read_bytes()).hexdigest()
    if (admission_sha != RECOVERY6_PRIOR_ADMISSION_SHA
            or prior_admission.get("status") != "CONSUMED_FOR_CONTROL_ATTEMPT_0005"
            or prior_admission.get("recovery_receipt_sha256") != RECOVERY6_PREDECESSOR_SHA
            or prior_admission.get("formal_start_sha256") != sha256_file(paths["formal_start"])
            or prior_admission.get("qualification_sha256") != sha256_file(paths["qualification"])
            or prior_admission.get("formal_task_consumed_before_claim") != 0
            or prior_admission.get("cell_claims_consumed_before_claim") != 0
            or prior_admission.get("episodes_consumed_before_claim") != 0):
        raise Q08AuthorizationError("attempt-0006 predecessor admission is not exact")
    start_ref = launch.get("formal_start") or {}; resource_ref = launch.get("gpu_resource_receipt") or {}
    if (launch.get("run_id") != profile["run_id"] or launch.get("execution_mode") != "host-orchestrator"
            or launch.get("exit_code") != 1 or start_ref.get("path") != str(paths["formal_start"])
            or _normalize_sha(start_ref.get("sha256")) != sha256_file(paths["formal_start"])
            or _normalize_sha(resource_ref.get("sha256")) != sha256_file(paths["resource"])):
        raise Q08AuthorizationError("attempt-0006 launch receipt does not bind the exact failed FormalStart")
    if (resource.get("status") != "failed" or resource.get("return_code") != 1
            or resource.get("stage") != "launch" or resource.get("campaign_id") != profile["campaign_id"]
            or resource.get("gpu_uuids") != [selected] or resource.get("gpu_hours", 1) > 0.01):
        raise Q08AuthorizationError("attempt-0006 resource receipt is not the bounded pre-container failure")
    if (prior_context.get("run_id") != profile["run_id"] or prior_context.get("cell_id") != "ordinary-r080"
            or prior_context.get("control_attempt") != 3 or prior_context.get("resume_allowed") is not False
            or prior_context.get("selected_gpu_uuids") != [selected]
            or prior_context.get("task_limit") != 96
            or paths["preexisting_context"].name != "preexisting-context-attempt-0003.json"):
        raise Q08AuthorizationError("attempt-0006 stale prior context evidence is inconsistent")
    if (inventory.get("status") != "PRE_CLAIM_HOST_PATH_FAILURE" or inventory.get("control_attempt") != 3
            or inventory.get("attempt_namespace", {}).get("uid") != 0
            or inventory.get("attempt_namespace", {}).get("gid") != 0
            or inventory.get("attempt_namespace", {}).get("mode") != 0o755
            or inventory.get("prior_context_sha256") != hashlib.sha256(paths["preexisting_context"].read_bytes()).hexdigest()
            or inventory.get("cell_claims_count") != 0 or inventory.get("seen_eval_leaf_count") != 0
            or any(inventory.get("seen_eval_leaf_exists", {}).values())
            or inventory.get("namespace_parent", {}).get("uid") != 0
            or inventory.get("namespace_parent", {}).get("gid") != 0
            or inventory.get("namespace_parent", {}).get("mode") != 0o755
            or inventory.get("prior_context_sha256") != "313f6050f0610099b46c09a1eb0aff614cced7155b0315cf4322780e09156e86"
            or inventory.get("container", {}).get("inspect_status") != "not_found"
            or inventory.get("container", {}).get("inspect_returncode") != 1
            or inventory.get("container", {}).get("name") != "experiment-host-q08-post-advisor-q08-seen-retention-202610010742-ordinary-r080-a0003"
            or "no such object" not in inventory.get("container", {}).get("inspect_stderr", "")):
        raise Q08AuthorizationError("attempt-0006 namespace inventory does not prove the exact empty pre-container boundary")
    claims = run_root / "control" / "claims"
    if claims.exists() and (claims.is_symlink() or any(claims.iterdir())):
        raise Q08AuthorizationError("attempt-0006 found a consumed scientific cell claim")
    for cell in CELL_IDS:
        if (run_root / "cells" / cell / "seen-eval").exists():
            raise Q08AuthorizationError("attempt-0006 found a scientific output leaf")
    namespace = run_root / RECOVERY6_CONTROL_REL
    admission_path = namespace / "admission-claim.json"
    if admission_path.exists() or admission_path.is_symlink():
        raise Q08AuthorizationError("attempt-0006 recovery admission was already consumed")
    if (not namespace.is_dir() or namespace.is_symlink() or namespace.stat().st_uid != 1001
            or namespace.stat().st_gid != 1001 or stat.S_IMODE(namespace.stat().st_mode) != 0o750):
        raise Q08AuthorizationError("attempt-0006 recovery namespace must be UID/GID1001-owned 0750")
    if (namespace / "cells" / "attempt-0004").exists():
        raise Q08AuthorizationError("attempt-0006 attempt4 namespace already exists before admission")
    return receipt


def _validate_engineering_recovery_attempt5(repo: Path, profile: dict, release: dict, ref: dict) -> dict:
    """Admit one control-only continuation after an exact pre-claim UID path failure."""
    import subprocess
    path = bound_path(repo, RECOVERY5_RECEIPT_REL)
    expect_file(path, str(ref.get("sha256", "")))
    committed = subprocess.run(["git", "-C", str(repo), "show", f"HEAD:{RECOVERY5_RECEIPT_REL}"], check=True, capture_output=True).stdout
    if committed != path.read_bytes():
        raise Q08AuthorizationError("attempt-0005 recovery receipt is not the committed exact authority bytes")
    predecessor = bound_path(repo, RECOVERY4_RECEIPT_REL)
    expect_file(predecessor, RECOVERY5_PREDECESSOR_SHA)
    predecessor_bytes = subprocess.run(["git", "-C", str(repo), "show", f"HEAD:{RECOVERY4_RECEIPT_REL}"], check=True, capture_output=True).stdout
    if predecessor_bytes != predecessor.read_bytes():
        raise Q08AuthorizationError("attempt-0004 predecessor receipt is not committed exact bytes")
    receipt = read_json(path)
    expected = {
        "schema": "openevo.q08.zero-consumption-engineering-recovery.v1",
        "status": "ZERO_CONSUMPTION_RECOVERY_PASS",
        "campaign_id": profile["campaign_id"], "run_id": profile["run_id"], "attempt": 5,
        "prior_authority_git_sha": RECOVERY5_PRIOR_AUTHORITY_SHA,
        "prior_operational_source_git_sha": RECOVERY5_PRIOR_SOURCE_SHA,
        "predecessor_recovery_sha256": RECOVERY5_PREDECESSOR_SHA,
        "prior_admission_claim_sha256": RECOVERY5_PRIOR_ADMISSION_SHA,
        "scientific_resume": False, "formal_task_consumed": 0, "task_bearing_calls": 0,
        "model_calls": 0, "episodes_consumed": 0, "cell_claims_consumed": 0,
        "final_panel_access": 0, "fresh_seen_episodes_max": 960,
        "reused_historical_episodes": 768, "program_total_maximum_including_reuse": 1920,
        "continuation": {"control_attempt": 3, "formal_start_basename": RECOVERY5_FORMAL_START_BASENAME,
            "cell_claims_allowed": 10, "cell_resume_allowed": False, "run_id_changed": False,
            "matrix_changed": False, "panel_changed": False, "budget_changed": False},
        "failure": {"owner": "q08_seen_launch_guard.authorize_probe",
            "message": "PermissionError reading the exact q7 qualification receipt before the first cell claim",
            "stage": "before-first-cell-claim", "cell_id": "ordinary-r080"},
        "failed_container": {"id": "00d277ca3ca47005f10963fd942f29ace0f5400fa3c3134889eb6d2e0f75b187",
            "name": "experiment-host-q08-post-advisor-q08-seen-retention-202610010742-ordinary-r080-a0002",
            "state": "exited", "exit_code": 1},
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise Q08AuthorizationError(f"attempt-0005 recovery mismatch: {key}")
    frozen = profile["frozen_inputs"]
    science = {"panel_sha256": PANEL_SHA, "matrix_sha256": MATRIX_SHA, "reuse_audit_sha256": REUSE_SHA,
        "probe_wrapper_sha256": WRAPPER_SHA, "model": frozen["model"], "model_revision": frozen["model_revision"],
        "seed_root": frozen["seed_root"], "memory_sha256": frozen["carrier_sha256"]["memory"],
        "skill_sha256": frozen["carrier_sha256"]["skill"], "agent_sha256": frozen["carrier_sha256"]["agent"]}
    if receipt.get("science_identity") != science:
        raise Q08AuthorizationError("attempt-0005 recovery scientific identity changed")
    run_root = (Path(profile["workspace"]) / profile["paths"]["runs"] / profile["run_id"]).resolve(strict=True)
    runs_root = (Path(profile["workspace"]) / profile["paths"]["runs"]).resolve(strict=True)
    evidence = receipt.get("attempt_0004_evidence")
    required = {"formal_start", "ready", "prelaunch", "qualification", "launch", "resource", "context", "prior_admission", "container_inspect", "container_logs"}
    if not isinstance(evidence, dict) or set(evidence) != required:
        raise Q08AuthorizationError("attempt-0005 evidence inventory is incomplete")
    evidence_root = Path(str(receipt.get("evidence_root", ""))).resolve(strict=True)
    if evidence_root != (runs_root / f"{profile['run_id']}-readiness-attempt0008").resolve():
        raise Q08AuthorizationError("attempt-0005 evidence root is outside its fixed readiness scope")
    archive = (run_root / RECOVERY5_ARCHIVE_REL).resolve(strict=True)
    expected_roots = {
        "formal_start": evidence_root / "control", "ready": evidence_root / "control", "prelaunch": evidence_root / "control",
        "qualification": runs_root / f"{profile['run_id']}-qualification-attempt0007",
        "launch": archive, "resource": archive, "context": archive, "prior_admission": archive,
        "container_inspect": archive, "container_logs": archive,
    }
    expected_basenames = {
        "formal_start": "formal-start-attempt-0002.json", "ready": None,
        "prelaunch": "prelaunch-zero-formal-attempt-0002.json",
        "qualification": "qualification.json", "launch": "launch-receipt.json", "resource": "gpu-resource-receipt.json",
        "context": "launch-context-ordinary-r080.json", "prior_admission": "prior-admission-claim-attempt-0004.json",
        "container_inspect": "failed-container-inspect.json", "container_logs": "failed-container-logs.txt",
    }
    paths = {}
    for key in required:
        item = evidence[key]
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise Q08AuthorizationError(f"attempt-0005 evidence reference malformed: {key}")
        raw_path = Path(str(item["path"]))
        actual = raw_path.resolve(strict=True)
        if actual.is_symlink() or not actual.is_file() or actual.parent != expected_roots[key].resolve():
            raise Q08AuthorizationError(f"attempt-0005 evidence path outside immutable scope: {key}")
        if key == "ready":
            if not actual.name.startswith("execution-readiness-"):
                raise Q08AuthorizationError("attempt-0005 READY reference is not a write-once receipt")
        elif actual.name != expected_basenames[key]:
            raise Q08AuthorizationError(f"attempt-0005 evidence basename drift: {key}")
        expect_file(actual, str(item["sha256"]))
        paths[key] = actual
    start = read_json(paths["formal_start"]); ready = read_json(paths["ready"])
    prelaunch = read_json(paths["prelaunch"]); qualification = read_json(paths["qualification"])
    launch = read_json(paths["launch"]); resource = read_json(paths["resource"])
    context = read_json(paths["context"]); admission = read_json(paths["prior_admission"])
    inspect_doc = json.loads(paths["container_inspect"].read_text())
    if not isinstance(inspect_doc, list) or len(inspect_doc) != 1 or not isinstance(inspect_doc[0], dict):
        raise Q08AuthorizationError("attempt-0005 Docker inspect evidence must contain exactly one object")
    inspect = inspect_doc[0]
    selected = receipt.get("selected_gpu_uuid")
    start_ready = (start.get("readiness") or {}).get("ready_receipt") or {}
    start_profile = (start.get("readiness") or {}).get("profile") or {}
    old_profile_raw = subprocess.run(["git", "-C", str(repo), "show", f"{RECOVERY5_PRIOR_AUTHORITY_SHA}:{PROFILE}"], check=True, capture_output=True).stdout
    old_release_raw = subprocess.run(["git", "-C", str(repo), "show", f"{RECOVERY5_PRIOR_AUTHORITY_SHA}:{RELEASE}"], check=True, capture_output=True).stdout
    old_router_raw = subprocess.run(["git", "-C", str(repo), "show", f"{RECOVERY5_PRIOR_AUTHORITY_SHA}:{ROUTER}"], check=True, capture_output=True).stdout
    old_profile_sha = hashlib.sha256(old_profile_raw).hexdigest()
    old_release_sha = hashlib.sha256(old_release_raw).hexdigest()
    old_router_sha = hashlib.sha256(old_router_raw).hexdigest()
    old_profile = json.loads(old_profile_raw); old_release = json.loads(old_release_raw)
    if (start.get("status") != "FORMAL_START_COMMITTED" or start.get("run_id") != profile["run_id"]
            or start.get("campaign_id") != profile["campaign_id"] or start.get("selected_gpu_uuids") != [selected]
            or start.get("formal_task_consumption_before") is not False or start.get("scientific_authority_granted_by_receipt") is not False
            or _normalize_sha(start_ready.get("sha256")) != sha256_file(paths["ready"])
            or Path(str(start_ready.get("path", ""))).resolve() != paths["ready"]
            or _normalize_sha(start_profile.get("sha256")) != old_profile_sha
            or Path(str(start_profile.get("path", ""))).name != Path(PROFILE).name):
        raise Q08AuthorizationError("attempt-0005 FormalStart is not exact zero-cell predecessor evidence")
    if (ready.get("verdict") != "READY" or ready.get("profile", {}).get("sha256") != "sha256:" + old_profile_sha
            or ready.get("release_receipt", {}).get("sha256") != "sha256:" + old_release_sha
            or Path(str(ready.get("profile", {}).get("path", ""))).name != Path(PROFILE).name
            or Path(str(ready.get("release_receipt", {}).get("path", ""))).name != Path(RELEASE).name
            or (ready.get("authority") or {}).get("router_status") != "launch-authorized"):
        raise Q08AuthorizationError("attempt-0005 READY does not bind the exact failed authority")
    prior_binding = qualification.get("q08_binding") or {}
    if (qualification.get("status") != "ZERO_FORMAL_QUALIFIED"
            or qualification.get("authority_git_sha") != RECOVERY5_PRIOR_AUTHORITY_SHA
            or qualification.get("execution_sha") != RECOVERY5_PRIOR_SOURCE_SHA
            or qualification.get("model_calls") != 0 or qualification.get("task_bearing_calls") != 0
            or qualification.get("selected_gpu_uuids") != [selected]
            or prior_binding.get("profile_sha256") != old_profile_sha
            or prior_binding.get("release_sha256") != old_release_sha
            or prior_binding.get("router_sha256") != old_router_sha
            or prior_binding.get("engineering_recovery_sha256") != RECOVERY5_PREDECESSOR_SHA):
        raise Q08AuthorizationError("attempt-0005 qualification is not exact zero-formal evidence")
    if (prelaunch.get("status") != "PASS" or prelaunch.get("formal_task_consumed") is not False
            or prelaunch.get("scientific_budget_consumed") != 0 or prelaunch.get("model_calls") != 0
            or prelaunch.get("training_calls") != 0 or prelaunch.get("selected_gpu_uuids") != [selected]):
        raise Q08AuthorizationError("attempt-0005 prelaunch was not a zero-consumption PASS")
    prior_claim_projection = admission
    if (admission.get("status") != "CONSUMED_FOR_CONTROL_ATTEMPT_0004"
            or admission.get("recovery_receipt_sha256") != RECOVERY5_PREDECESSOR_SHA
            or admission.get("formal_start_sha256") != sha256_file(paths["formal_start"])
            or admission.get("qualification_sha256") != sha256_file(paths["qualification"])
            or admission.get("formal_task_consumed_before_claim") != 0
            or admission.get("cell_claims_consumed_before_claim") != 0
            or admission.get("episodes_consumed_before_claim") != 0):
        raise Q08AuthorizationError("attempt-0005 predecessor admission is not exact zero-consumption evidence")
    start_ref = launch.get("formal_start") or {}; resource_ref = launch.get("gpu_resource_receipt") or {}
    if (launch.get("run_id") != profile["run_id"] or launch.get("execution_mode") != "host-orchestrator"
            or launch.get("exit_code") != 1 or start_ref.get("path") != str(paths["formal_start"])
            or _normalize_sha(start_ref.get("sha256")) != sha256_file(paths["formal_start"])
            or _normalize_sha(resource_ref.get("sha256")) != sha256_file(paths["resource"])):
        raise Q08AuthorizationError("attempt-0005 launch receipt does not bind the failed native FormalStart")
    if (resource.get("status") != "failed" or resource.get("return_code") != 1 or resource.get("stage") != "launch"
            or resource.get("campaign_id") != profile["campaign_id"] or resource.get("gpu_uuids") != [selected]
            or resource.get("gpu_hours") > 0.01):
        raise Q08AuthorizationError("attempt-0005 resource receipt is not the exact bounded pre-claim failure")
    context_ref = context.get("formal_start") or {}; qref = context.get("qualification") or {}
    if (context.get("status") != "FORMAL_START_COMMITTED" or context.get("cell_id") != "ordinary-r080"
            or context.get("run_id") != profile["run_id"] or context.get("campaign_id") != profile["campaign_id"]
            or context.get("task_limit") != 96 or context.get("resume_allowed") is not False
            or _normalize_sha(context_ref.get("sha256")) != sha256_file(paths["formal_start"])
            or Path(str(context_ref.get("path", ""))).resolve() != paths["formal_start"]
            or qref.get("sha256") != sha256_file(paths["qualification"])
            or Path(str(qref.get("path", ""))).resolve() != paths["qualification"]):
        raise Q08AuthorizationError("attempt-0005 launch context does not bind the failed pre-claim execution")
    logs = paths["container_logs"].read_text(errors="replace")
    if ("PermissionError: [Errno 13] Permission denied" not in logs
            or str(Path(old_profile["frozen_inputs"]["qualification_path"])) not in logs):
        raise Q08AuthorizationError("attempt-0005 container logs do not prove the exact qualification read failure")
    state = inspect.get("State") or {}
    config = inspect.get("Config") or {}
    failed_container = receipt["failed_container"]
    if (inspect.get("Id") != failed_container["id"]
            or inspect.get("Name") != "/" + failed_container["name"]
            or state.get("Status") != failed_container["state"]
            or state.get("ExitCode") != failed_container["exit_code"]
            or config.get("User") != "1001:1001"
            or inspect.get("Image") != profile["image_id"]):
        raise Q08AuthorizationError("attempt-0005 failed container id/state/exit identity mismatch")
    claims = run_root / "control" / "claims"
    if claims.exists() and (claims.is_symlink() or any(claims.iterdir())):
        raise Q08AuthorizationError("attempt-0005 found a consumed cell claim")
    for cell in CELL_IDS:
        if (run_root / "cells" / cell / "seen-eval").exists():
            raise Q08AuthorizationError("attempt-0005 found a consumed scientific output leaf")
    if not (run_root / "control/engineering-recovery-attempt-0004/admission-claim.json").is_file():
        raise Q08AuthorizationError("attempt-0005 predecessor admission claim is missing")
    namespace = run_root / "control/engineering-recovery-attempt-0005"
    if (namespace / "admission-claim.json").exists() or (namespace / "admission-claim.json").is_symlink():
        raise Q08AuthorizationError("attempt-0005 recovery admission was already consumed")
    if not namespace.is_dir() or namespace.stat().st_uid != 1001 or (namespace.stat().st_mode & 0o777) != 0o750:
        raise Q08AuthorizationError("attempt-0005 recovery namespace must be UID1001-owned 0750")
    return receipt


def _validate_engineering_recovery_attempt4(repo: Path, profile: dict, release: dict, ref: dict) -> dict:
    """Admit one control continuation after an exact pre-cell FormalStart path failure."""
    path = bound_path(repo, RECOVERY4_RECEIPT_REL)
    expect_file(path, str(ref.get("sha256", "")))
    import subprocess
    committed = subprocess.run(["git", "-C", str(repo), "show", f"HEAD:{RECOVERY4_RECEIPT_REL}"], check=True, capture_output=True).stdout
    if committed != path.read_bytes():
        raise Q08AuthorizationError("attempt-0004 recovery receipt is not the committed exact authority bytes")
    predecessor = bound_path(repo, RECOVERY3_RECEIPT_REL)
    expect_file(predecessor, RECOVERY4_PREDECESSOR_SHA)
    predecessor_bytes = subprocess.run(["git", "-C", str(repo), "show", f"HEAD:{RECOVERY3_RECEIPT_REL}"], check=True, capture_output=True).stdout
    if predecessor_bytes != predecessor.read_bytes():
        raise Q08AuthorizationError("attempt-0003 predecessor recovery receipt is not committed exact bytes")
    _validate_engineering_recovery_attempt3(repo, profile, release, {
        "path": RECOVERY3_RECEIPT_REL,
        "sha256": RECOVERY4_PREDECESSOR_SHA,
        "attempt": 3,
        "prior_authority_git_sha": "9bcecf42ecfd9ad9f3730d35ff0061a73e77e99f",
    })
    receipt = read_json(path)
    expected = {
        "schema": "openevo.q08.zero-consumption-engineering-recovery.v1",
        "status": "ZERO_CONSUMPTION_RECOVERY_PASS",
        "campaign_id": profile["campaign_id"], "run_id": profile["run_id"], "attempt": 4,
        "prior_authority_git_sha": RECOVERY4_PRIOR_AUTHORITY_SHA,
        "prior_operational_source_git_sha": RECOVERY4_PRIOR_SOURCE_SHA,
        "predecessor_recovery_sha256": RECOVERY4_PREDECESSOR_SHA,
        "scientific_resume": False, "formal_task_consumed": 0, "task_bearing_calls": 0,
        "model_calls": 0, "episodes_consumed": 0, "cell_claims_consumed": 0,
        "final_panel_access": 0, "fresh_seen_episodes_max": 960,
        "reused_historical_episodes": 768, "program_total_maximum_including_reuse": 1920,
        "continuation": {"control_attempt": 2, "formal_start_basename": RECOVERY4_FORMAL_START_BASENAME,
            "cell_claims_allowed": 10, "cell_resume_allowed": False, "run_id_changed": False,
            "matrix_changed": False, "panel_changed": False, "budget_changed": False},
        "failure": {"owner": "q08_seen_host_orchestrator._load_formal_start",
            "message": "Q08 engineering recovery permits only its exact canonical FormalStart attempt",
            "stage": "before-first-cell-claim", "cell_id": None},
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise Q08AuthorizationError(f"attempt-0004 recovery mismatch: {key}")
    frozen = profile["frozen_inputs"]
    science = {"panel_sha256": PANEL_SHA, "matrix_sha256": MATRIX_SHA, "reuse_audit_sha256": REUSE_SHA,
        "probe_wrapper_sha256": WRAPPER_SHA, "model": frozen["model"], "model_revision": frozen["model_revision"],
        "seed_root": frozen["seed_root"], "memory_sha256": frozen["carrier_sha256"]["memory"],
        "skill_sha256": frozen["carrier_sha256"]["skill"], "agent_sha256": frozen["carrier_sha256"]["agent"]}
    if receipt.get("science_identity") != science:
        raise Q08AuthorizationError("attempt-0004 recovery scientific identity changed")
    run_root = Path(profile["workspace"]) / profile["paths"]["runs"] / profile["run_id"]
    evidence = receipt.get("attempt_0003_evidence")
    required = {"formal_start", "ready", "prelaunch", "qualification", "launch", "resource"}
    if not isinstance(evidence, dict) or set(evidence) != required:
        raise Q08AuthorizationError("attempt-0004 evidence inventory is incomplete")
    evidence_root = Path(str(receipt.get("evidence_root", ""))).resolve(strict=True)
    runs_root = (Path(profile["workspace"]) / profile["paths"]["runs"]).resolve(strict=True)
    expected_evidence_root = (runs_root / f"{profile['run_id']}-readiness-attempt0008").resolve()
    if evidence_root != expected_evidence_root or evidence_root.is_symlink():
        raise Q08AuthorizationError("attempt-0004 evidence root is outside its fixed readiness scope")
    expected_roots = {
        "formal_start": evidence_root / "control",
        "ready": evidence_root / "control",
        "prelaunch": evidence_root / "control",
        "qualification": runs_root / f"{profile['run_id']}-qualification-attempt0005",
        "launch": run_root / RECOVERY4_ARCHIVE_REL,
        "resource": run_root / RECOVERY4_ARCHIVE_REL,
    }
    expected_basenames = {
        "formal_start": "formal-start-attempt-0001.json",
        "prelaunch": "prelaunch-zero-formal-attempt-0001.json",
        "qualification": "qualification.json",
        "launch": "launch-receipt.json",
        "resource": "gpu-resource-receipt.json",
    }
    expected_paths = {}
    for key in required:
        item = evidence[key]
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise Q08AuthorizationError(f"attempt-0004 evidence reference malformed: {key}")
        actual_path = Path(item["path"]).resolve(strict=True)
        if actual_path.is_symlink() or not actual_path.is_file() or actual_path.parent != expected_roots[key].resolve():
            raise Q08AuthorizationError(f"attempt-0004 evidence path outside immutable scope: {key}")
        if key in expected_basenames and actual_path.name != expected_basenames[key]:
            raise Q08AuthorizationError(f"attempt-0004 evidence basename drift: {key}")
        if key == "ready" and not (actual_path.name.startswith("execution-readiness-") and actual_path.suffix == ".json"):
            raise Q08AuthorizationError("attempt-0004 READY reference is not a write-once receipt")
        expected_paths[key] = actual_path
        expect_file(actual_path, str(item["sha256"]))
    start = read_json(expected_paths["formal_start"])
    ready = read_json(expected_paths["ready"])
    prelaunch = read_json(expected_paths["prelaunch"])
    qualification = read_json(expected_paths["qualification"])
    launch = read_json(expected_paths["launch"])
    resource = read_json(expected_paths["resource"])
    selected = receipt.get("selected_gpu_uuid")
    start_ready = (start.get("readiness") or {}).get("ready_receipt") or {}
    start_prelaunch = start.get("prelaunch_receipt") or {}
    if (start.get("status") != "FORMAL_START_COMMITTED" or start.get("run_id") != profile["run_id"]
            or start.get("campaign_id") != profile["campaign_id"] or start.get("selected_gpu_uuids") != [selected]
            or start.get("formal_task_consumption_before") is not False
            or start.get("scientific_authority_granted_by_receipt") is not False
            or Path(str(start_ready.get("path", ""))).resolve() != expected_paths["ready"].resolve()
            or _normalize_sha(start_ready.get("sha256")) != sha256_file(expected_paths["ready"])
            or Path(str(start_prelaunch.get("path", ""))).resolve() != expected_paths["prelaunch"].resolve()
            or _normalize_sha(start_prelaunch.get("sha256")) != sha256_file(expected_paths["prelaunch"])):
        raise Q08AuthorizationError("attempt-0004 FormalStart is not exact pre-cell evidence")
    if (ready.get("verdict") != "READY" or ready.get("formal_task_consumption_allowed") is not True
            or ready.get("profile", {}).get("sha256") != "sha256:1ea8c5f22b02d18c395d3ec915d12cb587967f70d445bf72bac2304be1cb5a51"
            or ready.get("release_receipt", {}).get("sha256") != "sha256:611c959005a1541c676e95f1abb8eaa566d057f2ed68e7c2277e0dffd6709fdc"):
        raise Q08AuthorizationError("attempt-0004 READY does not bind the exact predecessor authority")
    if (prelaunch.get("status") != "PASS" or prelaunch.get("formal_task_consumed") is not False
            or prelaunch.get("scientific_budget_consumed") != 0 or prelaunch.get("model_calls") != 0
            or prelaunch.get("training_calls") != 0 or prelaunch.get("selected_gpu_uuids") != [selected]):
        raise Q08AuthorizationError("attempt-0004 prelaunch was not a zero-consumption PASS")
    if (qualification.get("status") != "ZERO_FORMAL_QUALIFIED"
            or qualification.get("authority_git_sha") != expected["prior_authority_git_sha"]
            or qualification.get("model_calls") != 0 or qualification.get("task_bearing_calls") != 0
            or qualification.get("selected_gpu_uuids") != [selected]):
        raise Q08AuthorizationError("attempt-0004 qualification was not exact zero-formal evidence")
    start_ref = launch.get("formal_start") or {}
    resource_ref = launch.get("gpu_resource_receipt") or {}
    if (launch.get("run_id") != profile["run_id"] or launch.get("execution_mode") != "host-orchestrator"
            or launch.get("exit_code") != 1 or start_ref.get("path") != str(expected_paths["formal_start"])
            or _normalize_sha(start_ref.get("sha256")) != sha256_file(expected_paths["formal_start"])
            or _normalize_sha(resource_ref.get("sha256")) != sha256_file(expected_paths["resource"])):
        raise Q08AuthorizationError("attempt-0004 launch receipt does not bind the failed native FormalStart")
    if (resource.get("status") != "failed" or resource.get("return_code") != 1
            or resource.get("stage") != "launch" or resource.get("campaign_id") != profile["campaign_id"]
            or resource.get("gpu_uuids") != [selected]
            or selected != "GPU-00000000-0000-0000-0000-000000000000"):
        raise Q08AuthorizationError("attempt-0004 GPU resource receipt is not the exact pre-cell failure")
    claims = run_root / "control" / "claims"
    if claims.exists() and (claims.is_symlink() or any(claims.iterdir())):
        raise Q08AuthorizationError("attempt-0004 found a consumed cell claim")
    for cell in CELL_IDS:
        if (run_root / "cells" / cell / "seen-eval").exists():
            raise Q08AuthorizationError("attempt-0004 found a consumed scientific output leaf")
    if (run_root / RECOVERY3_CONTROL_REL / "admission-claim.json").exists():
        raise Q08AuthorizationError("attempt-0003 recovery admission was unexpectedly consumed")
    namespace = run_root / RECOVERY4_CONTROL_REL
    admission = namespace / "admission-claim.json"
    if admission.exists() or admission.is_symlink():
        raise Q08AuthorizationError("attempt-0004 recovery admission was already consumed")
    if not namespace.is_dir() or namespace.stat().st_uid != 1001 or (namespace.stat().st_mode & 0o777) != 0o750:
        raise Q08AuthorizationError("attempt-0004 recovery namespace must be UID1001-owned 0750")
    return receipt


def bound_path(repo: Path, rel: str) -> Path:
    p=(repo / rel).resolve()
    try: p.relative_to(repo.resolve())
    except ValueError as exc: raise Q08AuthorizationError(f"authority path escapes repo: {rel}") from exc
    return p

def _load_static_chain(repo: Path):
    profile_path=bound_path(repo, PROFILE); release_path=bound_path(repo, RELEASE)
    router_path=bound_path(repo, ROUTER); current_path=bound_path(repo, CURRENT)
    profile=read_json(profile_path); release=read_json(release_path)
    router=read_json(router_path); current=read_json(current_path)
    for key, expected in (("profile", profile_path), ("guard", bound_path(repo, "scripts/openevo_webshop/q08_seen_launch_guard.py")), ("host_orchestrator", bound_path(repo, "scripts/openevo_webshop/q08_seen_host_orchestrator.py")), ("qualifier", bound_path(repo, "scripts/openevo_webshop/q08_seen_prelaunch_qualifier.py"))):
        ref=release.get("artifacts", {}).get(key, {})
        if ref.get("path") != str(expected.relative_to(repo.resolve())):
            raise Q08AuthorizationError(f"release {key} path binding mismatch")
        expect_file(expected, ref.get("sha256", ""))
    if release.get("artifacts", {})["profile"]["sha256"] != sha256_file(profile_path):
        raise Q08AuthorizationError("release profile identity mismatch")
    if router.get("release", {}) != {"path":str(release_path.relative_to(repo.resolve())),"sha256":sha256_file(release_path)}:
        raise Q08AuthorizationError("router release binding mismatch")
    if current.get("router", {}) != {"path":str(router_path.relative_to(repo.resolve())),"sha256":sha256_file(router_path)}:
        raise Q08AuthorizationError("candidate current-campaign router binding mismatch")
    if current.get("profile", {}) != {"path":str(profile_path.relative_to(repo.resolve())),"sha256":sha256_file(profile_path)}:
        raise Q08AuthorizationError("candidate current-campaign profile binding mismatch")
    if current.get("release", {}) != {"path":str(release_path.relative_to(repo.resolve())),"sha256":sha256_file(release_path)}:
        raise Q08AuthorizationError("candidate current-campaign release binding mismatch")
    if release.get("status") != "launch-authorized" or release.get("formal_task_consumption_allowed") is not True:
        raise Q08AuthorizationError("Q08 release is not formally authorized")
    if router.get("status") != "launch-authorized" or router.get("formal_task_consumption_allowed") is not True:
        raise Q08AuthorizationError("Q08 router is not formally authorized")
    if current.get("status") != "launch-authorized" or current.get("formal_task_consumption_allowed") is not True:
        raise Q08AuthorizationError("Q08 candidate route is not formally authorized")
    source_a = profile.get("frozen_inputs", {}).get("operational_source", {})
    source_a_path = Path(str(source_a.get("path", ""))).resolve()
    if not source_a_path.is_dir() or not source_a.get("revision"):
        raise Q08AuthorizationError("Q08 operational source A identity is missing")
    try:
        import subprocess
        a_head = subprocess.run(["git", "-C", str(source_a_path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        a_dirty = subprocess.run(["git", "-C", str(source_a_path), "status", "--porcelain", "--untracked-files=all"], check=True, capture_output=True, text=True).stdout.strip()
        b_head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        b_dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"], check=True, capture_output=True, text=True).stdout.strip()
        ancestor = subprocess.run(["git", "-C", str(repo), "merge-base", a_head, b_head], check=True, capture_output=True, text=True).stdout.strip()
        revisions = subprocess.run(["git", "-C", str(repo), "rev-list", "--reverse", f"{a_head}..{b_head}"], check=True, capture_output=True, text=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Q08AuthorizationError("Q08 A/B Git identity could not be verified") from exc
    if a_dirty or b_dirty or a_head != source_a["revision"] or ancestor != a_head or a_head == b_head or not revisions:
        raise Q08AuthorizationError("Q08 requires clean, distinct B authority descended from exact operational A")
    previous=a_head
    for revision in revisions:
        lineage=subprocess.run(["git","-C",str(repo),"rev-list","--parents","-n","1",revision],check=True,capture_output=True,text=True).stdout.split()
        changed=subprocess.run(["git","-C",str(repo),"diff","--name-only",previous,revision],check=True,capture_output=True,text=True).stdout.splitlines()
        if lineage != [revision,previous] or not changed or any(not name.startswith("configs/experiment/") for name in changed):
            raise Q08AuthorizationError("Q08 B must be a linear authority-only descendant of source A")
        previous=revision
    if profile.get("source", {}).get("experiment") != {"path":str(source_a_path.relative_to(Path(profile["workspace"]).resolve())),"revision":a_head}:
        raise Q08AuthorizationError("Q08 profile source.experiment must point to exact operational source A")
    expected_source = release.get("operational_source")
    if expected_source != {"path": str(source_a_path), "revision": a_head}:
        raise Q08AuthorizationError("Q08 release does not bind exact operational source A")
    for rel in ("scripts/openevo_webshop/q08_seen_checkpoint_probe.py","scripts/openevo_webshop/q08_seen_launch_guard.py","scripts/openevo_webshop/q08_seen_host_orchestrator.py","scripts/openevo_webshop/q08_seen_prelaunch_qualifier.py"):
        if (source_a_path/rel).read_bytes() != (repo/rel).read_bytes():
            raise Q08AuthorizationError(f"Q08 B operational source copy differs from exact A bytes: {rel}")
    if release.get("authority_git_sha") not in (None, b_head):
        raise Q08AuthorizationError("Q08 release authority Git SHA drift")
    if profile.get("identity") != {"uid": 1001, "gid": 1001}:
        raise Q08AuthorizationError("Q08 scientific container identity must be UID/GID 1001:1001")
    if release.get("budget", {}).get("eligible_cells") != 10 or release.get("budget", {}).get("episodes_per_cell") != 96 or release.get("budget", {}).get("fresh_seen_episodes_max") != 960 or release.get("budget", {}).get("reused_historical_episodes") != 768 or release.get("budget", {}).get("program_total_maximum_including_reuse") != 1920:
        raise Q08AuthorizationError("Q08 budget declaration drift")
    return profile, release, router, current

def _cell_id(args) -> str:
    return f"{args.lineage}-r{int(args.completed_rounds):03d}"

def validate_probe_identity(args):
    """Validate the exact Q08 payload in UID1001 validate-only mode; never claim it."""
    repo=Path(args.authority_repo).resolve()
    profile, release, router, current=_load_static_chain(repo)
    if os.getuid()!=1001:
        raise Q08AuthorizationError(f"Q08 probe process must run as UID 1001, got {os.getuid()}")
    panel_ref=profile["frozen_inputs"]["panel_manifest"]
    matrix_ref=profile["frozen_inputs"]["checkpoint_matrix"]
    reuse_ref=profile["frozen_inputs"]["fresh_reuse_audit"]
    panel_path=bound_path(repo,panel_ref["path"]); matrix_path=bound_path(repo,matrix_ref["path"]); reuse_path=bound_path(repo,reuse_ref["path"])
    for path, digest, expected in ((panel_path,panel_ref["sha256"],PANEL_SHA),(matrix_path,matrix_ref["sha256"],MATRIX_SHA),(reuse_path,reuse_ref["sha256"],REUSE_SHA)):
        expect_file(path,expected)
        if digest!=expected: raise Q08AuthorizationError("frozen Q08 evidence reference drift")
    probe_path=bound_path(repo,profile["frozen_inputs"]["probe_script"]["path"])
    if sha256_file(probe_path)!=WRAPPER_SHA: raise Q08AuthorizationError("Q08 probe wrapper identity mismatch")
    panel=read_json(panel_path); matrix=read_json(matrix_path); reuse=read_json(reuse_path)
    if panel.get("status")!="SEEN_PANEL_IDENTITIES_VERIFIED" or len(panel.get("tasks",[]))!=96 or [t.get("probe_ordinal") for t in panel.get("tasks",[])]!=list(range(96)):
        raise Q08AuthorizationError("frozen 96-task panel identity validation failed")
    cells={f"{c.get('lineage')}-r{int(c.get('completed_rounds',0)):03d}":c for c in matrix.get("cells",[])}
    if set(k for k,c in cells.items() if c.get("status")=="PASS")!=set(CELL_IDS) or any(cells[k].get("status")!="MISSING_PAYLOAD_PENDING_EXACT_COPY_SEARCH" for k in MISSING):
        raise Q08AuthorizationError("frozen Q08 cell inventory drift")
    if len(reuse.get("cells",[]))!=12 or sum(len(c.get("rows",[])) for c in reuse["cells"])!=768:
        raise Q08AuthorizationError("historical 768-row reuse inventory drift")
    cell_id=_cell_id(args)
    if cell_id not in CELL_IDS or cell_id in MISSING or int(args.completed_rounds) not in (40,80,120,160):
        raise Q08AuthorizationError(f"Q08 cell is not in the ten-cell frozen panel: {cell_id}")
    cell=cells[cell_id]
    if args.model!=profile["frozen_inputs"]["model"] or args.revision!=profile["frozen_inputs"]["model_revision"]:
        raise Q08AuthorizationError("Q08 model/revision identity mismatch")
    for arg_name,key in (("memory","memory"),("skill","skill"),("agent","agent")):
        path=Path(getattr(args,arg_name)).resolve()
        if path!=Path(profile["frozen_inputs"][key]).resolve(): raise Q08AuthorizationError(f"Q08 carrier path mismatch: {arg_name}")
        expect_file(path,profile["frozen_inputs"]["carrier_sha256"][key])
    if Path(args.panel_manifest).resolve()!=panel_path or Path(args.checkpoint_matrix).resolve()!=matrix_path:
        raise Q08AuthorizationError("Q08 frozen manifest paths mismatch")
    if Path(args.runtime_source_root).resolve()!=Path(profile["frozen_inputs"]["runtime_source_root"]).resolve() or Path(args.seed_root).resolve()!=Path(profile["frozen_inputs"]["seed_root"]).resolve():
        raise Q08AuthorizationError("Q08 runtime or seed-root path mismatch")
    if os.environ.get("CUDA_VISIBLE_DEVICES","").strip()!=args.gpu_uuid or args.gpu_uuid not in profile["gpu_gate"]["authorized_uuids"]:
        raise Q08AuthorizationError("selected GPU is outside the standing pool or visible-device binding")
    run_root=Path(profile["workspace"])/profile["paths"]["runs"]/profile["run_id"]
    expected_output=run_root/"cells"/cell_id/"seen-eval"
    if Path(args.output_dir).resolve()!=expected_output.resolve() or expected_output.exists():
        raise Q08AuthorizationError("Q08 output leaf is not the exact unused fixed cell destination")
    return {"status":"IDENTITY_PASS","cell_id":cell_id,"tasks":96,"adapter_sha256":cell["expected_payload_sha256"],"state_sha256":cell["state_sha256"],"model_calls":0,"training_steps":0,"final_panel_access":0,"claim_consumed":False}

def authorize_probe(args, *, consume_claim: bool = True):
    """Validate exact identities; optionally consume the cell's sole claim atomically."""
    repo=Path(args.authority_repo).resolve()
    validate_probe_identity(args)
    profile, release, router, current=_load_static_chain(repo)
    if os.getuid() != 1001:
        raise Q08AuthorizationError(f"Q08 probe process must run as UID 1001, got {os.getuid()}")
    panel_ref=profile["frozen_inputs"]["panel_manifest"]
    matrix_ref=profile["frozen_inputs"]["checkpoint_matrix"]
    reuse_ref=profile["frozen_inputs"]["fresh_reuse_audit"]
    panel_path=bound_path(repo,panel_ref["path"]); matrix_path=bound_path(repo,matrix_ref["path"]); reuse_path=bound_path(repo,reuse_ref["path"])
    expect_file(panel_path,PANEL_SHA); expect_file(matrix_path,MATRIX_SHA); expect_file(reuse_path,REUSE_SHA)
    if panel_ref["sha256"] != PANEL_SHA or matrix_ref["sha256"] != MATRIX_SHA or reuse_ref["sha256"] != REUSE_SHA:
        raise Q08AuthorizationError("frozen Q08 evidence identity mismatch")
    if sha256_file(bound_path(repo,profile["frozen_inputs"]["probe_script"]["path"])) != WRAPPER_SHA:
        raise Q08AuthorizationError("Q08 probe wrapper identity mismatch")
    panel=read_json(panel_path); matrix=read_json(matrix_path); reuse=read_json(reuse_path)
    if panel.get("status") != "SEEN_PANEL_IDENTITIES_VERIFIED" or len(panel.get("tasks",[])) != 96:
        raise Q08AuthorizationError("frozen 96-task panel validation failed")
    cells={f"{c.get('lineage')}-r{int(c.get('completed_rounds',0)):03d}":c for c in matrix.get("cells",[])}
    if set(k for k,c in cells.items() if c.get("status")=="PASS") != set(CELL_IDS):
        raise Q08AuthorizationError("eligible Q08 cell inventory changed")
    if any(cells[k].get("status") != "MISSING_PAYLOAD_PENDING_EXACT_COPY_SEARCH" for k in MISSING):
        raise Q08AuthorizationError("missing-payload checkpoint status changed")
    if len(reuse.get("cells",[])) != 12 or sum(len(c.get("rows",[])) for c in reuse["cells"]) != 768:
        raise Q08AuthorizationError("historical reuse inventory changed")
    cell_id=_cell_id(args)
    if cell_id in MISSING or cell_id not in CELL_IDS:
        raise Q08AuthorizationError(f"cell is not eligible for this frozen panel: {cell_id}")
    if int(args.completed_rounds) not in (40,80,120,160): raise Q08AuthorizationError("round identity invalid")
    cell=cells[cell_id]
    if args.model != profile["frozen_inputs"]["model"] or args.revision != profile["frozen_inputs"]["model_revision"]:
        raise Q08AuthorizationError("model identity mismatch")
    for arg_name,key in (("memory","memory"),("skill","skill"),("agent","agent")):
        if Path(getattr(args,arg_name)).resolve() != Path(profile["frozen_inputs"][key]).resolve():
            raise Q08AuthorizationError(f"carrier path mismatch: {arg_name}")
        expect_file(Path(getattr(args,arg_name)), profile["frozen_inputs"]["carrier_sha256"][key])
    if Path(args.panel_manifest).resolve()!=panel_path or Path(args.checkpoint_matrix).resolve()!=matrix_path:
        raise Q08AuthorizationError("probe evidence paths differ from frozen profile")
    if Path(args.runtime_source_root).resolve()!=Path(profile["frozen_inputs"]["runtime_source_root"]).resolve() or Path(args.seed_root).resolve()!=Path(profile["frozen_inputs"]["seed_root"]).resolve():
        raise Q08AuthorizationError("runtime or seed-root identity mismatch")
    if os.environ.get("CUDA_VISIBLE_DEVICES","").strip()!=args.gpu_uuid or args.gpu_uuid not in profile["gpu_gate"]["authorized_uuids"]:
        raise Q08AuthorizationError("selected GPU is outside the standing pool or visible-device binding")
    run_root=Path(profile["workspace"])/profile["paths"]["runs"]/profile["run_id"]
    expected_output=run_root/"cells"/cell_id/"seen-eval"
    if Path(args.output_dir).resolve()!=expected_output.resolve(): raise Q08AuthorizationError("output leaf does not match the fixed cell")
    if expected_output.exists(): raise Q08AuthorizationError("fixed output leaf already exists; one-use route forbids resume")
    claims=run_root/"control"/"claims"
    if Path(args.cell_claim).resolve()!=claims/f"{cell_id}.claim": raise Q08AuthorizationError("cell claim path is not the fixed claim for this cell")
    claim=Path(args.cell_claim)
    if not claims.is_dir() or claims.is_symlink() or claims.stat().st_uid != 1001 or (claims.stat().st_mode & 0o022):
        raise Q08AuthorizationError("claim directory must pre-exist and be owned by UID 1001 without group/world write")
    context_path=Path(args.launch_context).resolve()
    ctx=read_json(context_path)
    start_path=Path(str((ctx.get("formal_start") or {}).get("path") or ""))
    match=re.fullmatch(r"formal-start-attempt-(\d{4})\.json",start_path.name)
    if match is None or ctx.get("control_attempt")!=int(match.group(1)):
        raise Q08AuthorizationError("launch context control-attempt identity is invalid")
    recovery_document = resolve_engineering_recovery_document(repo, release)
    expected_context=expected_context_namespace(run_root, recovery_document, int(match.group(1)))/f"attempt-{int(match.group(1)):04d}"/cell_id/"launch-context.json"
    if context_path!=expected_context.resolve() or not context_path.is_file(): raise Q08AuthorizationError("launch context path is not the fixed per-attempt cell context")
    st=context_path.stat(); parent_st=context_path.parent.stat()
    if (st.st_uid!=1001 or st.st_gid!=1001 or (st.st_mode&0o777)!=0o640
            or parent_st.st_uid!=1001 or parent_st.st_gid!=1001 or (parent_st.st_mode&0o777)!=0o750):
        raise Q08AuthorizationError("launch context must be UID1001-owned mode 0640 beneath a UID1001-owned 0750 directory")
    ancestor=context_path.parent
    while ancestor==run_root or run_root in ancestor.parents:
        info=ancestor.lstat()
        if (not ancestor.is_dir() or ancestor.is_symlink() or not (info.st_mode & 0o111)
                or (info.st_mode & 0o022)):
            raise Q08AuthorizationError(f"launch context ancestor is not safely traversable: {ancestor}")
        if ancestor==run_root: break
        ancestor=ancestor.parent
    if (ctx.get("schema")!="openevo.q08.cell-launch-context.v1" or ctx.get("status")!="FORMAL_START_COMMITTED" or ctx.get("cell_id")!=cell_id or ctx.get("run_id")!=profile["run_id"] or ctx.get("campaign_id")!=profile["campaign_id"] or ctx.get("selected_gpu_uuids")!=[args.gpu_uuid] or ctx.get("task_limit")!=96 or ctx.get("final_panel_access")!=0 or ctx.get("resume_allowed") is not False): raise Q08AuthorizationError("launch context is not a committed cell FormalStart")
    for key, expected in (("release",bound_path(repo,RELEASE)),("profile",bound_path(repo,PROFILE)),("router",bound_path(repo,ROUTER))):
        ref=ctx.get(key,{})
        if ref.get("path")!=str(expected.relative_to(repo)) or ref.get("sha256")!=sha256_file(expected):
            raise Q08AuthorizationError(f"launch context {key} chain mismatch")
    qref=ctx.get("qualification",{}); qpath=Path(qref.get("path",""))
    if not qpath.is_absolute() or not qpath.is_file() or sha256_file(qpath)!=qref.get("sha256"):
        raise Q08AuthorizationError("fresh zero-formal qualification receipt mismatch")
    q=read_json(qpath)
    qprojection={k:v for k,v in q.items() if k!="receipt_sha256"}
    if q.get("receipt_sha256") != hashlib.sha256(json.dumps(qprojection,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest():
        raise Q08AuthorizationError("zero-formal qualification receipt self-hash mismatch")
    expected_binding={"operational_source_sha":profile["frozen_inputs"]["operational_source"]["revision"],"authority_git_sha":q.get("authority_git_sha"),"campaign_id":profile["campaign_id"],"run_id":profile["run_id"],"gpu_uuid":args.gpu_uuid,"profile_sha256":sha256_file(bound_path(repo,PROFILE)),"release_sha256":sha256_file(bound_path(repo,RELEASE)),"router_sha256":sha256_file(bound_path(repo,ROUTER)),"panel_sha256":PANEL_SHA,"matrix_sha256":MATRIX_SHA,"reuse_audit_sha256":REUSE_SHA,"image_id":profile["image_id"],"eligible_cells":list(CELL_IDS),"episodes_per_cell":96,"fresh_seen_episodes_max":960,"reused_historical_episodes":768,"program_total_maximum_including_reuse":1920}
    recovery_ref=release.get("engineering_recovery")
    if recovery_ref is not None:
        if not isinstance(recovery_ref,dict): raise Q08AuthorizationError("Q08 engineering recovery reference is malformed")
        recovery_path=bound_path(repo,str(recovery_ref.get("path", "")))
        expect_file(recovery_path,str(recovery_ref.get("sha256", "")))
        expected_binding["engineering_recovery_sha256"]=sha256_file(recovery_path)
    if (q.get("schema")!="openevo.q08.zero-formal-qualification.v1" or q.get("status")!="ZERO_FORMAL_QUALIFIED" or q.get("formal_task_budget")!=0 or q.get("scientific_budget_consumed")!=0 or q.get("model_calls")!=0 or q.get("task_bearing_calls")!=0 or q.get("final_panel_access")!=0 or q.get("selected_gpu_uuids")!=[args.gpu_uuid] or cell_id not in q.get("authorized_cells",[]) or q.get("uid")!=1001 or q.get("gid")!=1001 or q.get("scientific_state_mutated") is not False or q.get("q08_binding")!=expected_binding):
        raise Q08AuthorizationError("fresh selected-GPU qualification is not valid")
    for name in ("prelaunch_receipt", "wrapper_identity_receipt", "gpu_owner_receipt"):
        receipt_ref=q.get(name) or {}; receipt_path=Path(str(receipt_ref.get("path") or ""))
        if (not receipt_path.is_absolute() or not receipt_path.is_file() or receipt_path.is_symlink()
                or receipt_ref.get("sha256")!=sha256_file(receipt_path)):
            raise Q08AuthorizationError(f"Q08 qualification evidence reference is invalid: {name}")
    prelaunch=read_json(Path(q["prelaunch_receipt"]["path"]))
    if (prelaunch.get("status")!="PASS" or prelaunch.get("selected_gpu_uuids")!=[args.gpu_uuid]
            or prelaunch.get("formal_task_consumed") is not False or prelaunch.get("scientific_budget_consumed")!=0
            or prelaunch.get("model_calls")!=0 or prelaunch.get("final_panel_access")!=0):
        raise Q08AuthorizationError("Q08 qualification prelaunch receipt is not a zero-task PASS")
    wrapper=read_json(Path(q["wrapper_identity_receipt"]["path"]))
    if (wrapper.get("status")!="PASS" or wrapper.get("formal_task_consumed") is not False
            or wrapper.get("scientific_budget_consumed")!=0
            or (wrapper.get("result") or {}).get("launch_guard_identity_validated") is not True):
        raise Q08AuthorizationError("Q08 UID1001 identity validation receipt is invalid")
    gpu_owner=read_json(Path(q["gpu_owner_receipt"]["path"]))
    if gpu_owner.get("status")!="PASS" or gpu_owner.get("selected_gpu_uuids")!=[args.gpu_uuid]:
        raise Q08AuthorizationError("Q08 qualification selected-GPU owner receipt is invalid")
    fref=ctx.get("formal_start",{}); fpath=Path(fref.get("path",""))
    if not fpath.is_absolute() or not fpath.is_file() or sha256_file(fpath)!=fref.get("sha256"):
        raise Q08AuthorizationError("FormalStart receipt mismatch")
    fs=read_json(fpath)
    projection={k:v for k,v in fs.items() if k!="receipt_sha256"}
    canonical=hashlib.sha256(json.dumps(projection,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
    if (fs.get("status")!="FORMAL_START_COMMITTED" or fs.get("formal_task_consumption_may_begin") is not True or fs.get("formal_task_consumption_before") is not False or fs.get("run_id")!=profile["run_id"] or fs.get("campaign_id")!=profile["campaign_id"] or fs.get("receipt_sha256")!="sha256:"+canonical or fs.get("selected_gpu_uuids")!=[args.gpu_uuid]):
        raise Q08AuthorizationError("FormalStart does not authorize this Q08 run")
    try:
        qualified_at=datetime.datetime.fromisoformat(str(q["observed_at_utc"]).replace("Z","+00:00")).timestamp()
        formal_started=float(fs["committed_at_epoch"])
        max_age=int(profile.get("gpu_gate",{}).get("qualification_max_age_seconds",900))
    except (KeyError,TypeError,ValueError) as exc:
        raise Q08AuthorizationError("Q08 qualification/FormalStart time binding is malformed") from exc
    if formal_started<qualified_at or formal_started-qualified_at>max_age:
        raise Q08AuthorizationError("Q08 zero-formal qualification was stale at FormalStart")
    try:
        import subprocess
        authority_head=subprocess.run(["git","-C",str(repo),"rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Q08AuthorizationError("Q08 authority checkout identity unavailable at point of use") from exc
    if q.get("authority_git_sha") != authority_head:
        raise Q08AuthorizationError("qualification authority SHA does not match the active Q08 checkout")
    if not consume_claim:
        return {"status":"PRECLAIM_VALIDATION_PASS","cell_id":cell_id,"task_limit":96,
                "output_leaf":str(expected_output),"claim_path":str(claim),"claim_consumed":False}
    projection={"schema":"openevo.q08.cell-claim.v1","campaign_id":profile["campaign_id"],"run_id":profile["run_id"],"cell_id":cell_id,"lineage":args.lineage,"completed_rounds":int(args.completed_rounds),"tasks":96,"output_leaf":str(expected_output),"gpu_uuid":args.gpu_uuid,"panel_sha256":PANEL_SHA,"matrix_sha256":MATRIX_SHA,"reuse_audit_sha256":REUSE_SHA,"adapter_sha256":cell["expected_payload_sha256"],"state_sha256":cell["state_sha256"],"profile_sha256":sha256_file(bound_path(repo,PROFILE)),"release_sha256":sha256_file(bound_path(repo,RELEASE)),"router_sha256":sha256_file(bound_path(repo,ROUTER)),"launch_context_sha256":sha256_file(context_path),"qualification_sha256":qref["sha256"],"formal_start_sha256":fref["sha256"],"uid":os.getuid(),"claimed_at_epoch":time.time(),"scientific_max_episodes":96}
    projection["claim_sha256"]=hashlib.sha256(json.dumps(projection,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    try:
        with claim.open("x",encoding="utf-8") as f:
            json.dump(projection,f,sort_keys=True,indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
        fd=os.open(str(claim.parent),os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    except FileExistsError as exc:
        raise Q08AuthorizationError("the fixed cell claim was already consumed") from exc
    return {"status":"CLAIM_CONSUMED","cell_id":cell_id,"claim_sha256":projection["claim_sha256"],"task_limit":96,"output_leaf":str(expected_output)}
