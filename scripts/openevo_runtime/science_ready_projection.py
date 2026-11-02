#!/usr/bin/env python3
"""Read-only, source-bound projection of the existing Q04 sole-writer task pool.

This is a *candidate* feed for the existing runtime owner, not a launch/claim
API.  In particular, a manifest's ``mode=queued`` is NOT evidence that an
already-attempted task may be run again.  Frozen scientific launchers, live
resource ownership and the one production writer remain authoritative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from pathlib import Path
from typing import Any

SCHEMA = "openevo.science-ready-projection.20261024"
POOL_SCHEMA = "openevo.advisor-task-pool.v1"
Q09_STATE_SCHEMA = "openevo.ceiling1-stage2-execution-state.v1"
SHA = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
MAX_JSON = 8 * 1024 * 1024
STATES = {"QUEUED", "RUNNING", "COMPLETE", "CONTINUATION_REQUIRED", "DIAGNOSTIC_REQUIRED"}


class ProjectionError(ValueError):
    """A source cannot be trusted, so do not publish a partial READY view."""


def _read(path: Path, *, maximum: int = MAX_JSON) -> bytes:
    if path.is_symlink():
        raise ProjectionError(f"symlink source forbidden: {path}")
    try:
        meta = path.stat()
        if not stat.S_ISREG(meta.st_mode) or meta.st_size > maximum:
            raise ProjectionError(f"not a bounded regular source: {path}")
        with path.open("rb") as handle:
            data = handle.read(maximum + 1)
        if len(data) > maximum:
            raise ProjectionError(f"oversized source: {path}")
        return data
    except OSError as exc:
        raise ProjectionError(f"unreadable source: {path}: {exc}") from exc


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(path: Path) -> tuple[dict[str, Any], str]:
    raw = _read(path)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProjectionError(f"invalid UTF-8 JSON source: {path}") from exc
    if not isinstance(value, dict):
        raise ProjectionError(f"JSON source must be an object: {path}")
    return value, _digest(raw)


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA.fullmatch(value))


def _completed(task: dict[str, Any], state: dict[str, Any]) -> bool:
    if state.get("status") != "COMPLETE" or not _is_sha(state.get("completion_sha256")):
        return False
    completion = task.get("completion")
    if not isinstance(completion, dict) or not isinstance(completion.get("path"), str):
        return False
    try:
        payload, digest = _json(Path(completion["path"]))
    except ProjectionError:
        return False
    return digest == state["completion_sha256"] and all(
        payload.get(key) == wanted for key, wanted in (completion.get("equals") or {}).items()
    )


def _source_files_match(task: dict[str, Any]) -> bool:
    sources = task.get("source_files")
    if not isinstance(sources, dict) or not sources:
        return False
    for name, expected in sources.items():
        if not isinstance(name, str) or not _is_sha(expected):
            return False
        try:
            # Source code can be larger than a JSON receipt, but is still bounded.
            data = _read(Path(name), maximum=32 * 1024 * 1024)
        except ProjectionError:
            return False
        if _digest(data) != expected:
            return False
    return True


def _launcher_identity_match(task: dict[str, Any]) -> bool:
    """Prevent publishing a candidate for an unpinned replacement launcher."""
    argv = task.get("argv")
    sources = task.get("source_files")
    if not isinstance(argv, list) or len(argv) < 2 or not all(isinstance(x, str) for x in argv):
        return False
    # Q04 native launchers are direct Python files, never a shell or opaque -c.
    if Path(argv[0]).name not in {"python", "python3", "python3.11", "python3.12"}:
        return False
    if not argv[1].endswith(".py") or argv[1] not in sources:
        return False
    if "--activation" in argv:
        if argv.count("--activation") != 1 or argv.count("--activation-sha256") != 1:
            return False
        try:
            activation = Path(argv[argv.index("--activation") + 1])
            expected = argv[argv.index("--activation-sha256") + 1]
            if not _is_sha(expected) or _digest(_read(activation)) != expected:
                return False
        except (IndexError, ProjectionError):
            return False
    return True


def _unclaimed_output(task: dict[str, Any]) -> bool:
    raw = task.get("output_root")
    if not isinstance(raw, str) or not raw:
        return False
    p = Path(raw)
    if p.is_symlink():
        return False
    try:
        return not p.exists() or (p.is_dir() and not any(p.iterdir()))
    except OSError:
        return False


def _q10_cpu_blocked(readiness_path: Path, *, expected_sha256: str | None = None) -> tuple[dict[str, Any], dict[str, str]]:
    """Normalize Q10's *actual CPU-only producer* without inventing a release.

    The candidate is deliberately not an activation. A later Q10 activation
    needs a separate adapter for its exact release/profile/claim/lease truth;
    it must never inherit eligibility from this preflight artifact.
    """
    readiness, readiness_sha = _json(readiness_path)
    if expected_sha256 is not None and (not _is_sha(expected_sha256) or readiness_sha != expected_sha256):
        raise ProjectionError("Q10 readiness differs from caller-pinned SHA-256")
    if readiness.get("status") != "CPU_INPUTS_VERIFIED_NOT_FORMAL_READY":
        raise ProjectionError("Q10 requires a new release-specific adapter for changed readiness status")
    identity, panel = readiness.get("identity"), readiness.get("panel")
    if not isinstance(identity, dict) or identity.get("status") != "CPU_IDENTITY_ELIGIBILITY_PASS":
        raise ProjectionError("Q10 identity eligibility is not the known CPU-only state")
    if not isinstance(panel, dict) or any((
        panel.get("status") != "CPU_PANEL_ELIGIBILITY_PASS",
        type(panel.get("final_panel_access")) is not int or panel.get("final_panel_access") != 0,
        type(panel.get("tasks")) is not int or panel.get("tasks") != 128,
    )):
        raise ProjectionError("Q10 panel identity/final access is not the known CPU-only state")
    if any(type(readiness.get(key)) is not int or readiness[key] != 0
           for key in ("formal_task_consumption", "gpu_requests")):
        raise ProjectionError("Q10 CPU-only status conflicts with reported resource/science consumption")
    commit = readiness.get("execution_git_sha")
    if not isinstance(commit, str) or not COMMIT.fullmatch(commit):
        raise ProjectionError("Q10 execution source commit is unbound")
    candidate_ref = readiness.get("candidate")
    if not isinstance(candidate_ref, dict) or not isinstance(candidate_ref.get("path"), str):
        raise ProjectionError("Q10 candidate identity missing")
    if not _is_sha(candidate_ref.get("sha256")):
        raise ProjectionError("Q10 candidate expected SHA-256 missing")
    candidate_path = Path(candidate_ref["path"])
    if not candidate_path.is_absolute():
        raise ProjectionError("Q10 candidate path must be absolute")
    candidate, candidate_sha = _json(candidate_path)
    if candidate_sha != candidate_ref["sha256"]:
        raise ProjectionError("Q10 candidate bytes differ from CPU readiness source")
    if (candidate.get("schema") != "openevo.q10.native-activation.v1"
            or candidate.get("status") != "PREPARED_NOT_ACTIVATED"
            or candidate.get("owner_pr") != 637
            or candidate.get("execution_git_sha") != commit
            or candidate.get("formal_task_consumption_allowed") is not False
            or candidate.get("gpu_allocation_allowed") is not False
            or type(candidate.get("final_panel_access")) is not int
            or candidate["final_panel_access"] != 0):
        raise ProjectionError("Q10 CPU candidate is not a frozen, forbidden-to-start proposal")
    remaining = readiness.get("remaining")
    if (not isinstance(remaining, list) or not remaining or len(remaining) > 64
            or len({item for item in remaining if isinstance(item, str)}) != len(remaining)
            or not all(isinstance(item, str) and item.strip() and len(item) <= 512 for item in remaining)):
        raise ProjectionError("Q10 missing explicit formal activation dependencies")
    if _digest(_read(readiness_path)) != readiness_sha or _digest(_read(candidate_path)) != candidate_sha:
        raise ProjectionError("Q10 producer changed while verifying candidate")
    return ({"producer_status": readiness["status"], "verdict": "BLOCKED",
             "reason": "CANONICAL_FORMAL_RELEASE_UNBOUND", "execution_authority": False,
             "formal_task_consumption": 0, "next_dependencies": remaining},
            {"q10_cpu_readiness_sha256": readiness_sha, "q10_candidate_sha256": candidate_sha})


def _q09_native_status(state_path: Path, *, expected_sha256: str | None = None) -> tuple[dict[str, Any], dict[str, str]]:
    """Observe a Q09 native controller without creating successor/retry authority.

    Q09 already has its own paid-work ledger and partial-round recovery contract.
    An in-progress round or a health pause cannot be converted into a fresh
    queued task by the cross-campaign scheduler. This adapter only makes that
    distinction inspectable alongside the Q04 pool.
    """
    state, state_sha = _json(state_path)
    if expected_sha256 is not None and (not _is_sha(expected_sha256) or state_sha != expected_sha256):
        raise ProjectionError("Q09 state differs from caller-pinned SHA-256")
    if state.get("schema") != Q09_STATE_SCHEMA:
        raise ProjectionError("Q09 native execution-state schema is not recognized")
    if state.get("q09_stream_id") not in {"q09-stream-1", "q09-stream-2"}:
        raise ProjectionError("Q09 stream identity is not recognized")
    for name in ("owner_launch_release_sha256", "formal_ready_package_sha256", "q09_schedule_sha256", "state_id"):
        if not _is_sha(state.get(name)):
            raise ProjectionError(f"Q09 native science identity is missing: {name}")
    if not isinstance(state.get("execution_git_sha"), str) or not COMMIT.fullmatch(state["execution_git_sha"]):
        raise ProjectionError("Q09 frozen execution source commit is unbound")
    next_round = state.get("next_round")
    complete = state.get("rounds_complete")
    consumed = state.get("rollouts_consumed")
    if (type(next_round) is not int or not 120 <= next_round <= 160
            or not isinstance(complete, list) or complete != list(range(next_round))
            or type(consumed) is not int or consumed != 128 * next_round):
        raise ProjectionError("Q09 native completed-round/rollout accounting does not match")
    if (state.get("final_panel_access_count") != 0 or state.get("external_teacher_calls") != 0):
        raise ProjectionError("Q09 protected budget access is outside this projection's known contract")
    status = state.get("status")
    if status == "RUNNING":
        verdict, reason = "STATE_REPORTS_RUNNING", "native controller PID/live owner not independently verified"
    elif status == "PAUSE_FOR_DIAGNOSTIC":
        verdict, reason = "BLOCKED", "native science/health diagnostic requires case-specific successor"
    elif status == "COMPLETE" and next_round == 160:
        verdict, reason = "AWAITING_NATIVE_SEAL", "round-159 and full scientific ledger must be independently sealed"
    else:
        raise ProjectionError("Q09 native status/round combination requires a new exact adapter")
    if _digest(_read(state_path)) != state_sha:
        raise ProjectionError("Q09 producer changed while verifying execution state")
    return ({"producer_status": status, "verdict": verdict, "reason": reason,
             "execution_authority": False, "remaining_rounds_nominal": 160 - next_round,
             "next_round": next_round, "sealed_prefix_count_reported": len(complete),
             "science_recovery_must_use_native_controller": True},
            {"q09_execution_state_sha256": state_sha})


def project_pool(pool_root: Path, *, required_campaigns: tuple[str, ...] = (),
                 q10_cpu_readiness: Path | None = None,
                 q10_cpu_expected_sha256: str | None = None,
                 q09_execution_state: Path | None = None,
                 q09_expected_sha256: str | None = None) -> dict[str, Any]:
    """Inspect one known writer's disk receipts without modifying files.

    No claim can be made that READY coverage is global: this adapter currently
    understands Q04 candidates, Q09 native non-launchable status, and Q10
    CPU-only non-readiness. A new campaign
    must supply its own verified native
    status/activation/budget adapter, not just a historical PR or chat record.
    """
    if q10_cpu_expected_sha256 is not None and q10_cpu_readiness is None:
        raise ProjectionError("Q10 expected SHA requires a Q10 readiness path")
    if q09_expected_sha256 is not None and q09_execution_state is None:
        raise ProjectionError("Q09 expected SHA requires a Q09 execution-state path")
    pool_root = Path(pool_root)
    manifest_path = pool_root / "manifest.json"
    state_path = pool_root / "state.json"
    manifest, manifest_sha = _json(manifest_path)
    state, state_sha = _json(state_path)
    if manifest.get("schema") != POOL_SCHEMA:
        raise ProjectionError("not the known Q04 single-writer pool schema")
    if state.get("manifest_sha256") != manifest_sha:
        raise ProjectionError("live state does not bind these exact manifest bytes")
    tasks = manifest.get("tasks")
    records = state.get("tasks")
    if not isinstance(tasks, list) or not isinstance(records, dict):
        raise ProjectionError("expected manifest tasks[] and state tasks{}")
    by_id: dict[str, dict[str, Any]] = {}
    for row in tasks:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ProjectionError("task identity missing")
        if row["id"] in by_id:
            raise ProjectionError(f"duplicate task identity: {row['id']}")
        by_id[row["id"]] = row
    if set(records) != set(by_id):
        raise ProjectionError("state/manifest task identity set mismatch")
    for task_id, row in records.items():
        if not isinstance(row, dict) or row.get("status") not in STATES:
            raise ProjectionError(f"unknown status for {task_id}")
    # Concurrent writers may rotate the two files during our observation.
    if _digest(_read(manifest_path)) != manifest_sha or _digest(_read(state_path)) != state_sha:
        raise ProjectionError("producer changed while collecting snapshot; retry later")

    rows: list[dict[str, Any]] = []
    for task_id, task in by_id.items():
        seen = records[task_id]
        status = seen["status"]
        verdict = "NOT_LAUNCHABLE"
        reason = status
        if status == "COMPLETE":
            verdict = "SEALED" if _completed(task, seen) else "EVIDENCE_MISMATCH"
            reason = "completion receipt must bind exact bytes and expected fields"
        elif status == "QUEUED":
            dependencies = task.get("depends_on", [])
            if task.get("mode") != "queued" or seen.get("attempted"):
                reason = "MODE_OR_ATTEMPTED"
            elif not isinstance(dependencies, list) or any(d not in by_id for d in dependencies):
                reason = "UNBOUND_DEPENDENCY"
            elif any(not _completed(by_id[d], records[d]) for d in dependencies):
                reason = "DEPENDENCY_NOT_SEALED"
            elif (pool_root / (task_id + "-dispatch.json")).exists():
                reason = "ALREADY_CLAIMED"
            elif not _source_files_match(task):
                reason = "SOURCE_IDENTITY_DRIFT"
            elif not _launcher_identity_match(task):
                reason = "LAUNCHER_OR_ACTIVATION_UNBOUND"
            elif not _unclaimed_output(task):
                reason = "OUTPUT_ALREADY_PRESENT"
            else:
                verdict = "CANDIDATE_FOR_EXISTING_LAUNCHER"
                reason = "scientific launcher + live UUID/VRAM/owner checks still required"
        elif status == "RUNNING":
            verdict = "STATE_REPORTS_RUNNING"
            reason = "live PID/owner has not been independently verified"
        rows.append({"task_id": task_id, "producer_status": status, "verdict": verdict,
                     "reason": reason, "gpu_uuid": task.get("gpu_uuid")})

    # Another task may have completed while verifying source/receipt hashes.
    if _digest(_read(manifest_path)) != manifest_sha or _digest(_read(state_path)) != state_sha:
        raise ProjectionError("producer changed during verification; retry later")
    campaigns: dict[str, dict[str, Any]] = {}
    sources = {"q04_manifest_sha256": manifest_sha, "q04_state_sha256": state_sha}
    observed = {"Q04"}
    if q10_cpu_readiness is not None:
        q10_status, q10_sources = _q10_cpu_blocked(q10_cpu_readiness,
                                                    expected_sha256=q10_cpu_expected_sha256)
        if _digest(_read(manifest_path)) != manifest_sha or _digest(_read(state_path)) != state_sha:
            raise ProjectionError("Q04 producer changed during cross-campaign snapshot")
        campaigns["Q10"] = q10_status
        sources.update(q10_sources)
        observed.add("Q10")
    if q09_execution_state is not None:
        q09_status, q09_sources = _q09_native_status(q09_execution_state,
                                                      expected_sha256=q09_expected_sha256)
        if _digest(_read(manifest_path)) != manifest_sha or _digest(_read(state_path)) != state_sha:
            raise ProjectionError("Q04 producer changed during cross-campaign snapshot")
        campaigns["Q09"] = q09_status
        sources.update(q09_sources)
        observed.add("Q09")
    unobserved = sorted(set(required_campaigns) - observed)
    candidates = [row for row in rows if row["verdict"] == "CANDIDATE_FOR_EXISTING_LAUNCHER"]
    return {
        "schema": SCHEMA, "mutation": False, "execution_authority": False,
        "owner": "openevo-experiment#728 / existing sole writer",
        "sources": sources,
        "coverage": {"observed_campaigns": sorted(observed), "unobserved_campaigns": unobserved,
                     "global_verdict": "READY_SUPPLY_UNVERIFIED_CROSS_CAMPAIGN"},
        "candidate_count": len(candidates), "tasks": rows, "campaigns": campaigns,
        "warning": "CANDIDATE is not READY/permission. Never enqueue, release Holder or run from this projection.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q04-pool-root", type=Path, required=True)
    parser.add_argument("--require-campaign", action="append", default=[])
    parser.add_argument("--q10-cpu-readiness", type=Path,
                        help="optional exact Q10 native CPU-only producer (never launch authority)")
    parser.add_argument("--q10-cpu-readiness-sha256",
                        help="optional caller-pinned exact Q10 readiness bytes (fail closed on drift)")
    parser.add_argument("--q09-execution-state", type=Path,
                        help="optional Q09 native execution-state source (read-only, never launch authority)")
    parser.add_argument("--q09-execution-state-sha256",
                        help="optional caller-pinned exact Q09 state bytes (fail closed on drift)")
    args = parser.parse_args(argv)
    try:
        result = project_pool(args.q04_pool_root, required_campaigns=tuple(args.require_campaign),
                              q10_cpu_readiness=args.q10_cpu_readiness,
                              q10_cpu_expected_sha256=args.q10_cpu_readiness_sha256,
                              q09_execution_state=args.q09_execution_state,
                              q09_expected_sha256=args.q09_execution_state_sha256)
    except ProjectionError as exc:
        parser.exit(2, f"SOURCE_NOT_TRUSTED: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
