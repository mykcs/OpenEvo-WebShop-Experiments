"""Receipt-driven unattended scheduler for independent OpenEVO arms.

Engineering orchestration only: real formal work is always delegated to
`scripts/oe run-campaign`, so this module never grants scientific authority.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.openevo_runtime import round_transaction, unattended_gpu_windows
from scripts.openevo_runtime.service_supervision import (
    ServiceSpec,
    render_systemd_unit,
    service_receipt,
)

SCHEMA = "openevo.unattended-orchestration.20261004"
STATUS_SCHEMA = "openevo.unattended-orchestration-status.20261004"
GPU_RE = re.compile(
    r"^GPU-(?:[0-9A-Fa-f]{32}|[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12})$"
)
TOK_RE = re.compile(r"^\$\{([A-Z0-9_]+)\}$")
KNOWN_REASON = "REPAIR_IDENTICAL_INVALID"
KNOWN_TERMINAL = "NOOP_KEEP_PRIOR_TYPED_CONTRACT_REJECTED_AFTER_REPAIR"
MUTATIONS = {
    "raise_token_cap",
    "force_selected",
    "third_repair",
    "extra_reflector_call",
    "change_update_frequency",
    "arm_specific_fallback",
    "change_acceleration_semantics",
    "change_gdr",
    "reward_conditioned_repair",
}


class SupervisorError(RuntimeError):
    pass


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SupervisorError(f"regular JSON file required: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SupervisorError(f"unreadable JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SupervisorError(f"JSON object required: {path}")
    return value


def abspath(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SupervisorError(f"{name} must be a path")
    p = Path(value).expanduser()
    if not p.is_absolute():
        raise SupervisorError(f"{name} must be absolute")
    return str(p.resolve())


def vector(value: Any, name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(x, str) or not x for x in value)
    ):
        raise SupervisorError(f"{name} must be a non-empty string array")
    return value


def config(path: Path) -> dict[str, Any]:
    p = path.expanduser().resolve()
    d = load(p)
    if d.get("schema") != SCHEMA or d.get("scientific_authority") is not False:
        raise SupervisorError(
            "config must use unattended schema and scientific_authority=false"
        )
    d["_path"] = str(p)
    d["_sha256"] = sha(p)
    d["state_root"] = abspath(d.get("state_root"), "state_root")
    poll = d.get("poll_seconds", 30)
    if not isinstance(poll, (int, float)) or isinstance(poll, bool) or poll <= 0:
        raise SupervisorError("poll_seconds must be positive")
    d["poll_seconds"] = float(poll)
    lanes = d.get("lanes")
    arms = d.get("arms")
    maxp = d.get("max_parallel_arms")
    if (
        not isinstance(lanes, list)
        or not lanes
        or type(maxp) is not int
        or maxp < 1
        or maxp > len(lanes)
    ):
        raise SupervisorError("invalid lanes/max_parallel_arms")
    lane_ids = set()
    seen_gpus = set()
    for lane in lanes:
        if not isinstance(lane, dict):
            raise SupervisorError("lane row must be object")
        lid = lane.get("id")
        g = vector(lane.get("gpu_uuids"), f"lane {lid} gpu_uuids")
        normalized_gpus = {x.casefold() for x in g}
        if (
            not isinstance(lid, str)
            or not lid
            or lid in lane_ids
            or len(normalized_gpus) != len(g)
            or any(not GPU_RE.fullmatch(x) for x in g)
        ):
            raise SupervisorError("invalid/duplicate lane identity")
        if seen_gpus.intersection(normalized_gpus):
            raise SupervisorError("GPU lanes must be disjoint")
        lane_ids.add(lid)
        seen_gpus.update(normalized_gpus)
    if not isinstance(arms, list) or not arms:
        raise SupervisorError("arms must be non-empty")
    arm_ids = set()
    roots = set()
    for arm in arms:
        if not isinstance(arm, dict):
            raise SupervisorError("arm row must be object")
        aid = arm.get("id")
        rounds = arm.get("rounds")
        if not isinstance(aid, str) or not aid or aid in arm_ids:
            raise SupervisorError("arm id must be unique")
        arm["run_output_root"] = abspath(
            arm.get("run_output_root"), f"arm {aid} run_output_root"
        )
        if arm["run_output_root"] in roots:
            raise SupervisorError("run roots must be unique")
        if (
            not isinstance(rounds, list)
            or not rounds
            or sorted(set(rounds)) != rounds
            or any(type(x) is not int for x in rounds)
        ):
            raise SupervisorError(f"arm {aid} rounds invalid")
        eligible = vector(arm.get("eligible_lanes"), f"arm {aid} eligible_lanes")
        if any(x not in lane_ids for x in eligible):
            raise SupervisorError(f"arm {aid} references unknown lane")
        launch = arm.get("launch")
        if not isinstance(launch, dict):
            raise SupervisorError(f"arm {aid} launch missing")
        launch["cwd"] = abspath(launch.get("cwd"), f"arm {aid} launch.cwd")
        argv = vector(launch.get("argv"), f"arm {aid} launch.argv")
        canonical_oe = str(Path(launch["cwd"]) / "scripts" / "oe")

        def exact_option(option: str, value: str) -> bool:
            if argv.count(option) != 1:
                return False
            index = argv.index(option)
            return index + 1 < len(argv) and argv[index + 1] == value

        if not (
            argv[0] in {"./scripts/oe", canonical_oe}
            and len(argv) > 2
            and argv[1] == "run-campaign"
            and "--resume" in argv
            and "--wait" in argv
            and exact_option("--gpus", "${GPU_UUIDS}")
            and exact_option("--output-root", "${CONTROL_ROOT}")
            and exact_option("--run-output-root", "${RUN_ROOT}")
        ):
            raise SupervisorError(
                f"arm {aid} must delegate resume to scripts/oe run-campaign with --wait and exact GPU/control/run-root placeholders"
            )
        if "--dry-run" in argv or "--preflight-only" in argv:
            raise SupervisorError("supervisor launch cannot be dry-run/preflight-only")
        env = launch.get("env", {})
        if not isinstance(env, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()
        ):
            raise SupervisorError("launch.env must be string map")
        for key, value in env.items():
            if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key):
                raise SupervisorError(f"launch.env has invalid variable name: {key}")
            if "${" in value and not TOK_RE.fullmatch(value):
                raise SupervisorError(
                    f"launch.env token must occupy whole value: {key}={value}"
                )
        if env.get("OPENEVO_AUTHORIZED_GPU_UUIDS") != "${GPU_UUIDS}":
            raise SupervisorError(
                f"arm {aid} launch.env must bind OPENEVO_AUTHORIZED_GPU_UUIDS to ${{GPU_UUIDS}}"
            )
        expected_head = launch.get("expected_git_head")
        if not isinstance(expected_head, str) or not re.fullmatch(
            r"[0-9a-f]{40}", expected_head
        ):
            raise SupervisorError(
                f"arm {aid} launch.expected_git_head must be exact 40-hex SHA"
            )
        arm["authority_path"] = abspath(
            arm.get("authority_path"), f"arm {aid} authority_path"
        )
        arm["_supervisor_state_root"] = d["state_root"]
        arm_ids.add(aid)
        roots.add(arm["run_output_root"])
    d["arms"] = sorted(arms, key=lambda x: (int(x.get("priority", 100)), x["id"]))
    # An explicit null is NOT "no time-limited grant": it is ambiguous input
    # and must never silently fall back to the legacy unrestricted path.
    if "gpu_handoff_contract" in d:
        window_raw = d["gpu_handoff_contract"]
        if not isinstance(window_raw, dict) or set(window_raw) != {"path", "sha256"}:
            raise SupervisorError("gpu_handoff_contract requires exact {path,sha256}")
        raw_path = window_raw.get("path")
        if not isinstance(raw_path, str) or not Path(raw_path).expanduser().is_absolute():
            raise SupervisorError("gpu_handoff_contract.path must be absolute")
        # Do not resolve away a symlink before the file identity guard sees it.
        handoff_path = Path(raw_path).expanduser()
        expected = window_raw.get("sha256")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise SupervisorError("gpu_handoff_contract.sha256 must be exact hex hash")
        try:
            d["_gpu_window"] = unattended_gpu_windows.read_contract(
                handoff_path,
                lane_gpu_uuids={g for lane in lanes for g in lane["gpu_uuids"]},
                expected_sha256=expected,
            )
        except unattended_gpu_windows.GPUWindowError as exc:
            raise SupervisorError(f"gpu_handoff_contract invalid: {exc}") from exc
    service = d.get("service", {})
    if not isinstance(service, dict):
        raise SupervisorError("service must be object")
    d["service"] = {
        "name": service.get("name", "openevo-unattended-supervisor"),
        "description": service.get("description", "OpenEVO unattended supervisor"),
        "user": service.get("user", "root"),
        "group": service.get("group", "root"),
    }
    return d


def git_clean(cwd: str, expected_head: str | None = None) -> tuple[bool, str]:
    try:
        head = subprocess.check_output(
            ["git", "-c", f"safe.directory={cwd}", "-C", cwd, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-c", f"safe.directory={cwd}", "-C", cwd, "status", "--porcelain"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as exc:
        return False, f"git identity unavailable: {exc.output.strip()}"
    if dirty:
        return False, "launch checkout is dirty"
    if expected_head is not None and head != expected_head:
        return False, f"launch checkout HEAD drift: expected {expected_head} got {head}"
    return True, f"git exact/clean head={head}"


def git_head(cwd: str) -> str:
    try:
        return subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={cwd}",
                "-C",
                cwd,
                "rev-parse",
                "HEAD",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise SupervisorError(
            f"git HEAD unavailable for dispatch progress check: {exc.output.strip()}"
        ) from exc


def state_sha(root: Path) -> str | None:
    path = root / "execution-state.json"
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise SupervisorError(f"execution-state must be a regular file: {path}")
    return sha(path)


def recovery_identity(arm: Mapping[str, Any]) -> dict[str, Any]:
    """Hash the canonical recovery projection for the arm's current round."""
    root = Path(str(arm["run_output_root"]))
    state_path = root / "execution-state.json"
    if state_path.is_file():
        state = load(state_path)
        round_index = state.get("next_round")
        if type(round_index) is not int:
            projection: dict[str, Any] = {
                "state": "INVALID_EXECUTION_STATE",
                "status": state.get("status"),
            }
            round_index = None
        else:
            projection = {"round_index": round_index}
    else:
        round_index = arm["rounds"][0]
        projection = {"round_index": round_index}

    if round_index is not None:
        round_root = root / f"round-{round_index:03d}"
        if not round_root.is_dir() or not any(round_root.iterdir()):
            projection["round_recovery"] = "NO_MATERIALIZED_ROUND"
        else:
            try:
                plan = round_transaction.build_round_recovery_plan(
                    round_root, round_index
                )
            except round_transaction.RoundPlanError as exc:
                projection["round_recovery"] = {
                    "error": str(exc),
                    "integrity": "UNKNOWN",
                }
            else:
                projection["round_recovery"] = {
                    "verdict": plan["verdict"],
                    "recovery_class": plan["recovery_class"],
                    "target_phase": plan["target_phase"],
                    "must_not_replay": plan["must_not_replay"],
                    "required_readbacks_before_resume": plan[
                        "required_readbacks_before_resume"
                    ],
                    "phases": [
                        {
                            key: phase.get(key)
                            for key in (
                                "phase",
                                "status",
                                "receipt_path",
                                "receipt_sha256",
                                "errors",
                            )
                        }
                        for phase in plan["phases"]
                    ],
                }
    canonical = json.dumps(
        projection, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return {
        "round_index": round_index,
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "projection": projection,
    }


def terminal_marker_path(arm: Mapping[str, Any]) -> Path:
    return (
        Path(str(arm["_supervisor_state_root"])) / "arm-terminals" / f"{arm['id']}.json"
    )


def terminal_progress(arm: Mapping[str, Any]) -> dict[str, Any] | None:
    """Reconcile every durable dispatch boundary, including interrupted reap."""
    marker_path = terminal_marker_path(arm)
    records = []
    if marker_path.is_file():
        records.append((marker_path, load(marker_path), False))
    dispatch_root = marker_path.parent.parent / "dispatches" / str(arm["id"])
    if dispatch_root.exists():
        for directory in sorted(dispatch_root.iterdir()):
            terminal = directory / "terminal.json"
            intent = directory / "intent.json"
            path = terminal if terminal.exists() else intent
            if not path.exists():
                continue
            record = load(path)
            if (record.get("arm_id") != arm["id"]
                    or record.get("dispatch_id") != directory.name):
                raise SupervisorError(f"dispatch receipt identity mismatch: {path}")
            records.append((path, record, path == intent))
    for path, record, orphan in records:
        blocked = dispatch_progress(arm, record, marker_path)
        if blocked is not None:
            return {
                **blocked,
                "failure_receipt": {"path": str(path), "sha256": sha(path)},
                **({"side_effect_status": "UNKNOWN_REQUIRES_READBACK",
                    "reason": "unacknowledged dispatch intent; " + blocked["reason"]}
                   if orphan else {}),
            }
    return None


def dispatch_progress(
    arm: Mapping[str, Any], marker: Mapping[str, Any], marker_path: Path,
) -> dict[str, Any] | None:
    """Use canonical recovery progress, not mutable state bytes, to unlock retry.

    A failed launcher may create execution-state or rewrite timestamps, attempt
    counts and diagnostic metadata. None of those proves a frozen round/phase
    advanced. The existing recovery projection binds the next round and phase
    receipts, including completed work that must not be replayed.
    """
    progress_cwd = str(marker.get("progress_cwd") or "")
    if not progress_cwd:
        return {
            "action": "BLOCKED_NEEDS_HUMAN",
            "reason": "prior dispatch terminal lacks progress checkout identity",
        }
    current_head = git_head(progress_cwd)
    pre_head = marker.get("pre_git_head")
    rc = marker.get("returncode")
    kind = marker.get("kind")

    if current_head != pre_head:
        return {
            "action": "BLOCKED_NEEDS_HUMAN",
            "reason": "dispatch checkout HEAD changed without execution-state progress",
            "dispatch_id": marker.get("dispatch_id"),
        }

    pre_recovery_sha = marker.get("pre_recovery_sha256")
    if not isinstance(pre_recovery_sha, str):
        return {
            "action": "BLOCKED_NEEDS_HUMAN",
            "reason": "prior dispatch terminal lacks pre-dispatch recovery identity",
            "dispatch_id": marker.get("dispatch_id"),
        }
    current_recovery = recovery_identity(arm)
    if current_recovery["sha256"] != pre_recovery_sha:
        previous = marker.get("pre_recovery_projection")
        if not isinstance(previous, dict):
            # Older successful spawn intents already recorded this projection.
            # Reuse only the exact intent bound to this terminal, not a nearby run.
            intent_path = (marker_path.parent.parent / "dispatches" / str(arm["id"])
                           / str(marker.get("dispatch_id")) / "intent.json")
            if intent_path.is_file():
                intent = load(intent_path)
                if (intent.get("pre_recovery_sha256") == pre_recovery_sha
                        and intent.get("pre_git_head") == pre_head):
                    previous = intent.get("pre_recovery_projection")
        reason = "prior terminal lacks a bound recovery projection; exact readback required"
        if isinstance(previous, dict):
            # The projection is identity evidence only when it matches the
            # hash captured before dispatch.
            raw = json.dumps(previous, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode("utf-8")
            if hashlib.sha256(raw).hexdigest() != pre_recovery_sha:
                reason = "prior recovery projection identity mismatch"
            else:
                before_round = previous.get("round_index")
                after_round = current_recovery["round_index"]
                if type(before_round) is not int or type(after_round) is not int:
                    reason = "invalid round identity in dispatch recovery"
                elif after_round < before_round:
                    reason = "next_round regressed after dispatch; sealed work cannot be replayed"
                elif after_round > before_round:
                    try:
                        if after_round > arm["rounds"][-1] + 1:
                            raise SupervisorError("next_round beyond frozen rounds")
                        for index in range(before_round, after_round):
                            if index not in arm["rounds"]:
                                raise SupervisorError("round advance outside frozen rounds")
                            sealed = round_transaction.build_round_recovery_plan(
                                Path(str(arm["run_output_root"])) / f"round-{index:03d}", index
                            )
                            if sealed["verdict"] != "COMPLETE_NO_REPLAY":
                                raise SupervisorError(f"round {index} advanced without complete sealed receipts")
                        recovery = previous.get("round_recovery")
                        for phase in recovery.get("phases", []) if isinstance(recovery, dict) else []:
                            if phase.get("status") != round_transaction.COMPLETED:
                                continue
                            path = Path(str(phase.get("receipt_path")))
                            if path.is_symlink() or not path.is_file() or sha(path) != phase.get("receipt_sha256"):
                                raise SupervisorError("completed phase receipt changed or disappeared during round advance")
                    except (SupervisorError, round_transaction.RoundPlanError, OSError) as exc:
                        reason = str(exc)
                    else:
                        return None
                else:
                    def completed(projection: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
                        recovery = projection.get("round_recovery")
                        if not isinstance(recovery, dict):
                            return {}
                        return {phase["phase"]: (phase.get("receipt_path"), phase.get("receipt_sha256"))
                                for phase in recovery.get("phases", [])
                                if phase.get("status") == round_transaction.COMPLETED}

                    before_phases = completed(previous)
                    after_phases = completed(current_recovery["projection"])
                    if any(after_phases.get(name) != identity
                           for name, identity in before_phases.items()):
                        reason = "completed phase receipt changed or disappeared; exact readback required"
                    elif set(after_phases) > set(before_phases):
                        return None
                    else:
                        reason = "recovery bytes changed without a new completed phase"
        return {"action": "BLOCKED_NEEDS_HUMAN", "reason": reason,
                "dispatch_id": marker.get("dispatch_id"), "returncode": rc, "kind": kind}

    return {
        "action": "BLOCKED_NEEDS_HUMAN",
        "reason": "previous supervisor dispatch terminated without observable state/authority progress; readback is required before retry",
        "dispatch_id": marker.get("dispatch_id"),
        "returncode": rc,
        "kind": kind,
    }


def lock_busy(root: Path) -> bool:
    p = root / ".formal-arm.lock"
    if not p.is_file() or p.is_symlink():
        return False
    f = p.open("r+")
    try:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return False
    finally:
        f.close()


def known_health(
    arm: dict[str, Any], state: Mapping[str, Any]
) -> tuple[bool, dict[str, Any]]:
    if (
        state.get("status") != "PAUSE_FOR_DIAGNOSTIC"
        or type(state.get("next_round")) is not int
        or state.get("final_panel_access_count", 0) != 0
    ):
        return False, {"reason": "not a zero-final-panel diagnostic pause"}
    r = state["next_round"] - 1
    if r not in arm["rounds"]:
        return False, {"reason": "pause round outside frozen rounds"}
    root = Path(arm["run_output_root"])
    rr = root / f"round-{r:03d}"
    pause_path = root / "carrier-health-pauses" / f"round-{r:03d}.json"
    ledger_path = rr / "carrier-health-ledger.json"
    boundary_path = rr / "carrier-health-boundary.json"
    complete_path = rr / "ROUND_COMPLETE.json"
    after_path = rr / "state-after.json"
    try:
        ledger = load(ledger_path)
        boundary = load(boundary_path)
        complete = load(complete_path)
        after = load(after_path)
        pause = load(pause_path)
    except SupervisorError as exc:
        return False, {"reason": str(exc)}
    auto = boundary.get("automatic_mutations")
    if not (
        ledger.get("status") == "FAIL"
        and ledger.get("round_index") == r
        and boundary.get("action") == "PAUSE_FOR_DIAGNOSTIC"
        and boundary.get("round_index") == r
        and boundary.get("preserve_completed_transaction") is True
        and boundary.get("block_next_round") is True
        and complete.get("status") == "PASS"
        and complete.get("round_index") == r
        and after.get("next_round") == r + 1
        and pause.get("status") == "PAUSE_FOR_DIAGNOSTIC"
        and pause.get("through_round") == r
        and pause.get("next_round") == r + 1
        and pause.get("formal_transaction_preserved") is True
        and pause.get("automatic_scientific_change") is False
        and isinstance(auto, dict)
        and set(auto) == MUTATIONS
        and all(v is False for v in auto.values())
    ):
        return False, {
            "reason": "health boundary is not known preserved-transaction shape"
        }
    sealed = {
        "health_ledger_sha256": ledger_path,
        "boundary_decision_sha256": boundary_path,
        "round_complete_sha256": complete_path,
        "state_after_sha256": after_path,
    }
    if any(pause.get(key) != sha(path) for key, path in sealed.items()):
        return False, {
            "reason": "health pause does not bind exact sealed transaction hashes"
        }
    observed_pause_sha = sha(pause_path)
    if state.get("carrier_health_pause_sha256") not in {None, observed_pause_sha}:
        return False, {
            "reason": "execution-state pause SHA does not match sealed pause receipt"
        }
    failed = [x for x in ledger.get("rows", []) if x.get("health_status") != "PASS"]
    dirs = {
        "text_memory": "text-memory",
        "skill_bundle": "skill-bundle",
        "agent_system": "agent-system",
    }
    if not failed:
        return False, {"reason": "FAIL ledger has no failed rows"}
    checked = []
    for row in failed:
        carrier = row.get("carrier")
        if row.get("health_reasons") != [KNOWN_REASON] or carrier not in dirs:
            return False, {"reason": f"unknown health failure: {carrier}"}
        term = load(rr / dirs[carrier] / "terminal-receipt.json")
        if not (
            term.get("status") == "NOOP"
            and term.get("reason") == KNOWN_TERMINAL
            and term.get("primary_calls") == 1
            and term.get("repair_calls") == 1
            and term.get("repair_requested") is True
            and term.get("primary_contract_valid") is False
            and term.get("repair_contract_valid") is False
            and term.get("repair_changed_bytes") is False
            and term.get("primary_payload_sha256") == term.get("repair_payload_sha256")
            and term.get("accepted_payload_sha256") == term.get("prior_sha256")
            and term.get("external_teacher_calls") == 0
        ):
            return False, {
                "reason": f"{carrier} is not exact identical-invalid KEEP_PRIOR"
            }
        checked.append(carrier)
    return True, {
        "through_round": r,
        "failed_carriers": checked,
        "pause_sha256": observed_pause_sha,
        "health_ledger_sha256": sha(ledger_path),
        "boundary_decision_sha256": sha(boundary_path),
        "round_complete_sha256": sha(complete_path),
        "state_after_sha256": sha(after_path),
    }


def committed_json(repo: Path, path: Path) -> dict[str, Any]:
    repo = repo.resolve()
    path = path.resolve()
    try:
        rel = path.relative_to(repo).as_posix()
    except ValueError as exc:
        raise SupervisorError("authority artifact is outside launch checkout") from exc
    if path.is_symlink() or not path.is_file():
        raise SupervisorError(f"committed authority artifact missing: {path}")
    try:
        blob = subprocess.check_output(
            ["git", "-c", f"safe.directory={repo}", "-C", str(repo), "show", f"HEAD:{rel}"],
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        raise SupervisorError(
            f"authority artifact is not committed at launch HEAD: {rel}"
        ) from exc
    if blob != path.read_bytes():
        raise SupervisorError(f"authority artifact bytes differ from launch HEAD: {rel}")
    try:
        value = json.loads(blob)
    except Exception as exc:
        raise SupervisorError(f"invalid committed JSON: {rel}") from exc
    if not isinstance(value, dict):
        raise SupervisorError(f"committed JSON object required: {rel}")
    return value


def bound_health_resolution(
    arm: dict[str, Any], detail: Mapping[str, Any]
) -> tuple[bool, dict[str, Any]]:
    authority_text = arm.get("authority_path")
    if not authority_text:
        return False, {"reason": "known pause lacks authority_path binding"}
    repo = Path(arm["launch"]["cwd"]).resolve()
    authority_path = Path(str(authority_text)).resolve()
    try:
        authority = committed_json(repo, authority_path)
    except SupervisorError as exc:
        return False, {"reason": str(exc)}
    ref = authority.get("carrier_health_resolution")
    if not isinstance(ref, dict):
        return False, {"reason": "current authority has no carrier_health_resolution"}
    raw = Path(str(ref.get("path") or ""))
    if raw.is_absolute() or ".." in raw.parts:
        return False, {"reason": "resolution path must be relative and traversal-free"}
    resolution_path = repo / raw
    try:
        receipt = committed_json(repo, resolution_path)
    except SupervisorError as exc:
        return False, {"reason": str(exc)}
    if sha(resolution_path) != ref.get("sha256"):
        return False, {"reason": "authority-bound resolution SHA drift"}
    no_change = {
        key: True
        for key in (
            "formal_budget",
            "repair_call_limit",
            "sampling",
            "scientific_algorithm",
            "token_cap",
        )
    }
    expected_hashes = {
        key: detail[key]
        for key in (
            "pause_sha256",
            "health_ledger_sha256",
            "boundary_decision_sha256",
            "round_complete_sha256",
            "state_after_sha256",
        )
    }
    if not (
        receipt.get("schema") == "openevo.carrier-health-resolution.v1"
        and receipt.get("status") == "RESOLVED"
        and receipt.get("contract_transition") == "SAME_CONTRACT"
        and receipt.get("through_round") == detail["through_round"]
        and receipt.get("output_root") == arm["run_output_root"]
        and receipt.get("external_teacher_calls") == 0
        and receipt.get("final_panel_access") == 0
        and receipt.get("automatic_scientific_change") is False
        and receipt.get("no_change") == no_change
        and all(receipt.get(k) == v for k, v in expected_hashes.items())
    ):
        return False, {"reason": "bound resolution does not match exact preserved pause"}
    return True, {
        "resolution_path": str(resolution_path),
        "resolution_sha256": sha(resolution_path),
        "through_round": detail["through_round"],
    }


def discover(arm: dict[str, Any]) -> dict[str, Any]:
    root = Path(arm["run_output_root"])
    try:
        busy = lock_busy(root)
    except OSError as exc:
        # Unknown writer ownership still blocks the shared lanes conservatively.
        return {"action": "RUNNING_EXTERNAL", "reason": f"arm lock readback failed: {exc}"}
    try:
        return discover_state(arm, root, busy)
    except (SupervisorError, OSError, subprocess.SubprocessError) as exc:
        return {
            "action": "RUNNING_EXTERNAL" if busy else "BLOCKED_NEEDS_HUMAN",
            "reason": f"arm state readback failed: {type(exc).__name__}: {exc}",
            "failure_class": "ENGINEERING_READBACK_FAILURE",
            "next_action": "Repair and read back this arm's exact state/receipt; preserve existing work",
        }


def discover_state(arm: dict[str, Any], root: Path, busy: bool) -> dict[str, Any]:
    ok, reason = git_clean(
        arm["launch"]["cwd"], arm["launch"]["expected_git_head"]
    )
    if not ok:
        return {"action": "BLOCKED_NEEDS_HUMAN", "reason": reason}
    p = root / "execution-state.json"
    if not p.is_file():
        if busy:
            return {"action": "RUNNING_EXTERNAL", "reason": "formal arm lock held before state creation"}
        prior_terminal = terminal_progress(arm)
        if prior_terminal is not None:
            return {**prior_terminal, "next_round": arm["rounds"][0]}
        if root.is_dir() and any(
            x.is_dir() and any(x.iterdir()) for x in root.glob("round-*")
        ):
            return {
                "action": "BLOCKED_NEEDS_HUMAN",
                "reason": "round artifacts exist without execution-state",
            }
        return {
            "action": "READY_TO_DISPATCH",
            "next_round": arm["rounds"][0],
            "reason": "fresh arm",
        }
    s = load(p)
    nr = s.get("next_round")
    if s.get("final_panel_access_count", 0) != 0 or type(nr) is not int:
        return {
            "action": "BLOCKED_NEEDS_HUMAN",
            "reason": "invalid execution-state/final-panel boundary",
        }
    if busy:
        return {
            "action": "RUNNING_EXTERNAL",
            "next_round": nr,
            "reason": "formal arm lock held",
        }
    if s.get("status") == "PAUSE_FOR_DIAGNOSTIC":
        known, detail = known_health(arm, s)
        if not known:
            return {
                "action": "BLOCKED_NEEDS_HUMAN",
                "next_round": nr,
                "health": detail,
                "reason": detail.get("reason"),
            }
        bound, resolution = bound_health_resolution(arm, detail)
        if not bound:
            return {
                "action": "BLOCKED_NEEDS_HUMAN",
                "next_round": nr,
                "health": detail,
                "reason": resolution.get("reason"),
            }
        prior_terminal = terminal_progress(arm)
        if prior_terminal is not None:
            return {**prior_terminal, "next_round": nr, "health": detail}
        return {
            "action": "READY_TO_DISPATCH",
            "next_round": nr,
            "health": detail,
            "bound_resolution": resolution,
            "reason": "known pause has exact committed authority-bound SAME_CONTRACT resolution",
        }
    terminal = arm["rounds"][-1] + 1
    if nr > terminal:
        return {
            "action": "BLOCKED_NEEDS_HUMAN",
            "reason": "next_round beyond frozen rounds",
            "next_round": nr,
        }
    status = s.get("status")
    if nr == terminal and (
        status in {"COMPLETE", "SEALED", "CAMPAIGN_COMPLETE"}
        or (isinstance(status, str) and status.endswith("_SEALED"))
    ):
        return {"action": "COMPLETE", "next_round": nr, "reason": "arm sealed"}
    prior_terminal = terminal_progress(arm)
    if prior_terminal is not None:
        return {**prior_terminal, "next_round": nr}
    if nr == terminal:
        return {
            "action": "READY_TO_DISPATCH",
            "next_round": nr,
            "reason": "seal-only continuation required",
        }
    if nr not in arm["rounds"]:
        return {
            "action": "BLOCKED_NEEDS_HUMAN",
            "reason": "next_round outside frozen rounds",
            "next_round": nr,
        }
    rr = root / f"round-{nr:03d}"
    if rr.is_dir() and any(rr.iterdir()):
        try:
            rp = round_transaction.build_round_recovery_plan(rr, nr)
        except round_transaction.RoundPlanError as exc:
            return {
                "action": "BLOCKED_NEEDS_HUMAN",
                "reason": f"recovery planner failed: {exc}",
                "next_round": nr,
            }
        if rp["verdict"] == "MANUAL_REVIEW_REQUIRED":
            return {
                "action": "BLOCKED_NEEDS_HUMAN",
                "reason": f"recovery requires review: {rp['recovery_class']}",
                "next_round": nr,
                "recovery_plan": rp,
            }
        return {
            "action": "READY_TO_DISPATCH",
            "reason": f"exact resume: {rp['verdict']}",
            "next_round": nr,
            "recovery_plan": rp,
        }
    if s.get("status") not in {
        "RUNNING",
        "READY",
        "READY_FOR_FORMAL",
        "RESUME_REQUIRED",
    }:
        return {
            "action": "BLOCKED_NEEDS_HUMAN",
            "reason": f"unsupported state {s.get('status')!r}",
            "next_round": nr,
        }
    return {
        "action": "READY_TO_DISPATCH",
        "next_round": nr,
        "reason": "next frozen round ready",
    }


def plan(
    d: dict[str, Any], running: Mapping[str, dict[str, Any]] | None = None,
    *, now_utc: float | None = None,
) -> dict[str, Any]:
    running = dict(running or {})
    arms = {a["id"]: {**discover(a), "arm_id": a["id"]} for a in d["arms"]}
    lanes = {
        x["id"]: {"gpu_uuids": x["gpu_uuids"], "assigned_arm": None} for x in d["lanes"]
    }
    for aid, ch in running.items():
        lanes[ch["lane_id"]]["assigned_arm"] = aid
        arms[aid] = {
            **arms[aid],
            "action": "RUNNING",
            "reason": "running under this supervisor",
            "lane_id": ch["lane_id"],
            "dispatch_id": ch["dispatch_id"],
        }
    external = [
        x
        for aid, x in arms.items()
        if x["action"] == "RUNNING_EXTERNAL" and aid not in running
    ]
    assignments = []
    if external:
        for x in arms.values():
            if x["action"] == "READY_TO_DISPATCH":
                x["action"] = "WAIT_RESOURCE"
                x["reason"] = "external controller has unresolved lane ownership"
    else:
        free = [x for x in d["lanes"] if lanes[x["id"]]["assigned_arm"] is None]
        count = len(running)
        for arm in d["arms"]:
            x = arms[arm["id"]]
            if x["action"] != "READY_TO_DISPATCH":
                continue
            if count >= d["max_parallel_arms"]:
                x["action"] = "WAIT_RESOURCE"
                x["reason"] = "max_parallel_arms reached"
                continue
            lane = next((l for l in free if l["id"] in arm["eligible_lanes"]), None)
            if lane is None:
                x["action"] = "WAIT_RESOURCE"
                x["reason"] = "no eligible free lane"
                continue
            lanes[lane["id"]]["assigned_arm"] = arm["id"]
            x["lane_id"] = lane["id"]
            assignments.append(
                {"arm_id": arm["id"], "lane_id": lane["id"], "action": x["action"]}
            )
            free.remove(lane)
            count += 1
    result = {
        "schema": STATUS_SCHEMA,
        "observability_only": True,
        "scientific_authority_granted": False,
        "config_path": d["_path"],
        "config_sha256": d["_sha256"],
        "arms": arms,
        "lanes": lanes,
        "assignments": assignments,
        "blocked_by_external_controller": bool(external),
    }
    if "_gpu_window" in d:
        from datetime import datetime, timezone

        current = time.time() if now_utc is None else now_utc
        if type(current) not in (int, float):
            raise SupervisorError("now_utc must be an epoch timestamp")
        return unattended_gpu_windows.overlay(
            result, d["lanes"], d["_gpu_window"],
            now=datetime.fromtimestamp(current, tz=timezone.utc),
            integrity_error=unattended_gpu_windows.contract_integrity(d["_gpu_window"]),
        )
    return result


def atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    raw = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


def event(d: dict[str, Any], name: str, **fields: Any) -> None:
    root = Path(d["state_root"])
    root.mkdir(parents=True, exist_ok=True)
    p = root / "events.jsonl"
    row = {
        "schema": "openevo.unattended-orchestration-event.20261004",
        "event": name,
        "epoch": time.time(),
        "config_sha256": d["_sha256"],
        "scientific_authority_granted": False,
        **fields,
    }
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


@contextlib.contextmanager
def controller_lock(d: dict[str, Any]):
    root = Path(d["state_root"])
    root.mkdir(parents=True, exist_ok=True)
    p = root / "supervisor.lock"
    fd = os.open(p, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SupervisorError(f"another supervisor holds {p}") from exc
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def render(argv: list[str], values: Mapping[str, str]) -> list[str]:
    out = []
    for raw in argv:
        m = TOK_RE.fullmatch(raw)
        if m:
            if m.group(1) not in values:
                raise SupervisorError(f"unknown token {raw}")
            out.append(values[m.group(1)])
        elif "${" in raw:
            raise SupervisorError(f"token must occupy whole argv element: {raw}")
        else:
            out.append(raw)
    return out


def start(
    d: dict[str, Any],
    arm: dict[str, Any],
    lane: dict[str, Any],
) -> dict[str, Any]:
    if "_gpu_window" in d:
        # The existing run-campaign child has no bounded science-safe exit API.
        # Revalidation of a previous planning snapshot cannot make it safe.
        raise SupervisorError(
            "time-limited GPU grant: no verified bounded-safe-stop controller; "
            "refusing unbounded run-campaign dispatch"
        )
    cwd = arm["launch"]["cwd"]
    clean, reason = git_clean(cwd, arm["launch"]["expected_git_head"])
    if not clean:
        raise SupervisorError(reason)
    did = uuid.uuid4().hex
    dr = Path(d["state_root"]) / "dispatches" / arm["id"] / did
    control = dr / "control"
    control.mkdir(parents=True, mode=0o750)
    log = dr / "launch.log"
    stream = log.open("ab", buffering=0)
    vals = {
        "GPU_UUIDS": ",".join(lane["gpu_uuids"]),
        "CONTROL_ROOT": str(control),
        "ARM_ID": arm["id"],
        "RUN_ROOT": arm["run_output_root"],
    }
    argv = render(arm["launch"]["argv"], vals)
    configured_env = {
        key: render([value], vals)[0]
        for key, value in arm["launch"].get("env", {}).items()
    }
    env = {**os.environ, **configured_env}
    pre_state_sha = state_sha(Path(arm["run_output_root"]))
    pre_recovery = recovery_identity(arm)
    pre_git_head = git_head(cwd)
    if pre_git_head != arm["launch"]["expected_git_head"]:
        stream.close()
        raise SupervisorError(
            "launch checkout HEAD drift immediately before dispatch: "
            f"expected {arm['launch']['expected_git_head']} got {pre_git_head}"
        )
    atomic(
        dr / "intent.json",
        {
            "schema": "openevo.unattended-dispatch-intent.20261004",
            "dispatch_id": did,
            "kind": "launch",
            "arm_id": arm["id"],
            "lane_id": lane["id"],
            "gpu_uuids": lane["gpu_uuids"],
            "cwd": cwd,
            "argv": argv,
            "config_sha256": d["_sha256"],
            "pre_state_sha256": pre_state_sha,
            "pre_recovery_sha256": pre_recovery["sha256"],
            "pre_recovery_projection": pre_recovery["projection"],
            "pre_git_head": pre_git_head,
            "progress_cwd": cwd,
            "scientific_authority_granted_by_supervisor": False,
        },
    )
    try:
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    except OSError:
        stream.close()
        raise
    event(
        d,
        "DISPATCH_STARTED",
        dispatch_id=did,
        kind="launch",
        arm_id=arm["id"],
        lane_id=lane["id"],
        pid=proc.pid,
        config_sha256=d["_sha256"],
        argv=argv,
        gpu_uuids=lane["gpu_uuids"],
        pre_state_sha256=pre_state_sha,
        pre_recovery_sha256=pre_recovery["sha256"],
        pre_git_head=pre_git_head,
    )
    return {
        "arm_id": arm["id"],
        "lane_id": lane["id"],
        "process": proc,
        "stream": stream,
        "dispatch_id": did,
        "kind": "launch",
        "log_path": str(log),
        "pre_state_sha256": pre_state_sha,
        "pre_recovery_sha256": pre_recovery["sha256"],
        "pre_recovery_projection": pre_recovery["projection"],
        "pre_git_head": pre_git_head,
        "progress_cwd": cwd,
        "config_sha256": d["_sha256"],
        "argv": argv,
        "gpu_uuids": lane["gpu_uuids"],
    }


def reap(d: dict[str, Any], children: dict[str, dict[str, Any]]) -> None:
    for aid, ch in list(children.items()):
        rc = ch["process"].poll()
        if rc is None:
            continue
        ch["stream"].close()
        terminal = {
            "schema": "openevo.unattended-dispatch-terminal.20261004",
            "dispatch_id": ch["dispatch_id"],
            "arm_id": aid,
            "lane_id": ch["lane_id"],
            "config_sha256": ch["config_sha256"],
            "argv": ch["argv"],
            "gpu_uuids": ch["gpu_uuids"],
            "kind": ch["kind"],
            "returncode": rc,
            "log_path": ch["log_path"],
            "pre_state_sha256": ch["pre_state_sha256"],
            "pre_recovery_sha256": ch["pre_recovery_sha256"],
            "pre_recovery_projection": ch["pre_recovery_projection"],
            "pre_git_head": ch["pre_git_head"],
            "progress_cwd": ch["progress_cwd"],
            "scientific_authority_granted_by_supervisor": False,
            "finished_at_epoch": time.time(),
        }
        atomic(
            Path(d["state_root"])
            / "dispatches"
            / aid
            / ch["dispatch_id"]
            / "terminal.json",
            terminal,
        )
        atomic(Path(d["state_root"]) / "arm-terminals" / f"{aid}.json", terminal)
        event(
            d,
            "DISPATCH_FINISHED",
            dispatch_id=ch["dispatch_id"],
            arm_id=aid,
            lane_id=ch["lane_id"],
            kind=ch["kind"],
            returncode=rc,
            config_sha256=ch["config_sha256"],
            argv=ch["argv"],
            gpu_uuids=ch["gpu_uuids"],
            pre_state_sha256=ch["pre_state_sha256"],
            pre_recovery_sha256=ch["pre_recovery_sha256"],
            pre_git_head=ch["pre_git_head"],
        )
        del children[aid]



def dispatch_failure(
    d: Mapping[str, Any], arm: Mapping[str, Any], lane: Mapping[str, Any],
    before: Mapping[str, Any], error: Exception,
) -> dict[str, Any]:
    """Persist a dispatch exception before the next poll or supervisor restart.

    An exception can happen after Popen but before acknowledgement, so it never
    proves zero side effects. Existing arm locks and the round recovery planner
    still decide whether an existing writer or materialized progress is present.
    """
    dispatch_id = uuid.uuid4().hex
    receipt = {
        "schema": "openevo.unattended-dispatch-terminal.20261004",
        "dispatch_id": dispatch_id,
        "arm_id": arm["id"],
        "lane_id": lane["id"],
        "kind": "dispatch_exception",
        "returncode": None,
        "config_sha256": d["_sha256"],
        "argv_template": arm["launch"]["argv"],
        "actual_spawn_argv": None,
        "failure_class": "ENGINEERING_DISPATCH_FAILURE",
        "next_action": "Read back existing writer/side effects; repair the affected launch boundary",
        "gpu_uuids": lane["gpu_uuids"],
        "progress_cwd": arm["launch"]["cwd"],
        **before,
        "error": f"{type(error).__name__}: {error}",
        "side_effect_status": "UNKNOWN_REQUIRES_READBACK",
        "scientific_authority_granted_by_supervisor": False,
        "finished_at_epoch": time.time(),
    }
    path = (Path(str(d["state_root"])) / "dispatches" / str(arm["id"])
            / dispatch_id / "terminal.json")
    atomic(path, receipt)
    atomic(terminal_marker_path(arm), receipt)
    return {"path": str(path), "sha256": sha(path), "dispatch_id": dispatch_id}


def run(
    d: dict[str, Any],
    execute: bool,
    once: bool = False,
    max_cycles: int | None = None,
    sleep_fn=time.sleep,
) -> dict[str, Any]:
    children = {}
    cycles = 0
    prior_handoff_state: tuple[tuple[str, str], ...] | None = None
    with controller_lock(d):
        event(d, "SUPERVISOR_STARTED", execute=execute)
        try:
            while True:
                reap(d, children)
                snap = plan(d, children)
                if "gpu_time_window" in snap:
                    current_handoff = tuple(
                        (item["lane_id"], item["classification"])
                        for item in snap["gpu_time_window"]["handoffs"]
                    )
                    if current_handoff != prior_handoff_state:
                        event(
                            d, "GPU_HANDOFF_STATUS_CHANGED",
                            handoffs=snap["gpu_time_window"]["handoffs"],
                        )
                    prior_handoff_state = current_handoff
                atomic(Path(d["state_root"]) / "status.json", snap)
                cycles += 1
                if not execute:
                    return snap
                if (
                    all(x["action"] == "COMPLETE" for x in snap["arms"].values())
                    and not children
                ):
                    event(d, "SUPERVISOR_COMPLETE")
                    return snap
                if "_gpu_window" in d and not children:
                    # With no safe-stop runtime implementation, leave a durable
                    # blocked result instead of running a pointless loop.
                    event(d, "GPU_HANDOFF_REQUIRES_LIVE_RELEASE_READBACK")
                    return snap
                amap = {a["id"]: a for a in d["arms"]}
                lmap = {x["id"]: x for x in d["lanes"]}
                for a in snap["assignments"]:
                    aid = a["arm_id"]
                    if aid in children:
                        continue
                    row = snap["arms"][aid]
                    arm, lane = amap[aid], lmap[a["lane_id"]]
                    before: dict[str, Any] = {}
                    try:
                        pre_recovery = recovery_identity(arm)
                        before = {
                            "pre_state_sha256": state_sha(Path(arm["run_output_root"])),
                            "pre_git_head": git_head(arm["launch"]["cwd"]),
                            "pre_recovery_sha256": pre_recovery["sha256"],
                            "pre_recovery_projection": pre_recovery["projection"],
                        }
                        children[aid] = start(d, arm, lane)
                    except (
                        SupervisorError,
                        OSError,
                        subprocess.SubprocessError,
                    ) as exc:
                        failure = dispatch_failure(d, arm, lane, before, exc)
                        event(
                            d,
                            "DISPATCH_EXCEPTION",
                            arm_id=aid,
                            error=f"{type(exc).__name__}: {exc}",
                            failure_receipt=failure,
                        )
                        row["action"] = "BLOCKED_NEEDS_HUMAN"
                        row["reason"] = f"dispatch failed: {exc}; exact readback required"
                        row["failure_receipt"] = failure
                # Do not leave an old READY-only snapshot after dispatch fails.
                atomic(Path(d["state_root"]) / "status.json", snap)
                if once or (max_cycles is not None and cycles >= max_cycles):
                    return snap
                sleep_fn(d["poll_seconds"])
        finally:
            for ch in children.values():
                ch["stream"].close()
            event(d, "SUPERVISOR_STOPPED", remaining_children=sorted(children))


def render_service(d: dict[str, Any], oe_path: Path) -> dict[str, Any]:
    oe = oe_path.expanduser().resolve()
    if not oe.is_file():
        raise SupervisorError(f"oe missing: {oe}")
    svc = d["service"]
    spec = ServiceSpec(
        service_name=svc["name"],
        description=svc["description"],
        working_directory=oe.parent.parent,
        argv=(str(oe), "supervise-campaign", "--config", d["_path"], "--execute"),
        user=svc["user"],
        group=svc["group"],
        kill_mode="process",
    )
    return {
        "unit": render_systemd_unit(spec),
        "receipt": service_receipt(spec),
        "config_sha256": d["_sha256"],
        "scientific_authority_granted": False,
    }
