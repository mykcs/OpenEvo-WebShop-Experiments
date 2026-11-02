"""Expiring GPU-grant overlay for the existing receipt-driven Supervisor.

This module does not own a GPU, terminate a process, or grant scientific release.
It prevents an unbounded run-campaign child from being mistaken for a safely
time-limited execution. A planned p95 is NEVER an enforceable stop bound.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

SCHEMA = "openevo.unattended-gpu-handoff.202610130955"


class GPUWindowError(ValueError):
    pass


@dataclass(frozen=True)
class Grant:
    gpu_uuid: str
    starts_at: datetime
    handoff_at: datetime
    drain_at: datetime
    authorization_ref: str

    @property
    def cutoff_epoch(self) -> float:
        return self.drain_at.timestamp()


def timestamp(value: Any, field: str, zone: ZoneInfo) -> datetime:
    if not isinstance(value, str) or not value:
        raise GPUWindowError(f"{field}: explicit dated ISO timezone timestamp required")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise GPUWindowError(f"{field}: invalid timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise GPUWindowError(f"{field}: naive timestamp forbidden")
    if parsed.second or parsed.microsecond:
        raise GPUWindowError(f"{field}: timestamp must be minute-aligned")
    if parsed.isoformat() != parsed.astimezone(zone).isoformat():
        raise GPUWindowError(f"{field}: offset/zone mismatch")
    return parsed


def read_contract(
    path: Path, *, lane_gpu_uuids: set[str], expected_sha256: str
) -> dict[str, Any]:
    """Validate a SHA-bound advisory against *exact existing lane* UUIDs."""
    if path.is_symlink() or not path.is_file():
        raise GPUWindowError("handoff contract must be a regular non-symlink file")
    payload=path.read_bytes()
    observed=hashlib.sha256(payload).hexdigest()
    if expected_sha256 != observed:
        raise GPUWindowError("handoff contract SHA-256 mismatch")
    try:
        document=json.loads(payload)
    except (ValueError,UnicodeDecodeError) as exc:
        raise GPUWindowError("handoff contract JSON unreadable") from exc
    if not isinstance(document,dict) or document.get("schema")!=SCHEMA:
        raise GPUWindowError(f"unsupported GPU handoff schema; expected {SCHEMA}")
    if document.get("execution_mode")!="observe-and-block-unbounded":
        raise GPUWindowError("handoff mode must observe-and-block-unbounded")
    try:
        zone=ZoneInfo(document["timezone"])
    except (KeyError,TypeError,ValueError,ZoneInfoNotFoundError) as exc:
        raise GPUWindowError("valid IANA timezone required") from exc
    if not isinstance(document.get("grants"),list) or not document["grants"]:
        raise GPUWindowError("non-empty grants required")
    seen=set()
    grants={}
    for row in document["grants"]:
        if not isinstance(row,dict):
            raise GPUWindowError("grant must be object")
        gpu=row.get("gpu_uuid")
        if not isinstance(gpu,str) or gpu.casefold() in seen:
            raise GPUWindowError("duplicate/nonstring GPU UUID grant")
        seen.add(gpu.casefold())
        begin=timestamp(row.get("starts_at"),"starts_at",zone)
        handoff=timestamp(row.get("handoff_at"),"handoff_at",zone)
        lead=row.get("drain_buffer_seconds")
        if type(lead) is not int or not (60<=lead<=86400):
            raise GPUWindowError("drain_buffer_seconds needs a bounded integer >=60")
        drain=handoff-timedelta(seconds=lead)
        if not begin<drain<handoff:
            raise GPUWindowError("grant has no safe interval before drain")
        auth=row.get("authorization_ref")
        if not isinstance(auth,str) or not auth.strip():
            raise GPUWindowError("authorization_ref required")
        grants[gpu.casefold()]=Grant(gpu,begin,handoff,drain,auth)
    lane_set={x.casefold() for x in lane_gpu_uuids}
    if set(grants)!=lane_set:
        raise GPUWindowError("grants must bind every and only the existing lane GPU UUIDs")
    return {"schema":SCHEMA, "sha256":observed,"path":str(path.resolve()),
            "grants":grants,"timezone":str(zone)}


def contract_integrity(contract: Mapping[str, Any]) -> str | None:
    """Return a bounded failure reason when the source grant changed mid-session."""
    source=Path(str(contract["path"]))
    if source.is_symlink() or not source.is_file():
        return "GPU grant source missing or replaced by symlink"
    try:
        actual=hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError:
        return "GPU grant source cannot be read"
    if actual != contract["sha256"]:
        return "GPU grant source hash drift since Supervisor binding"
    return None


def overlay(
    snapshot: dict[str,Any],
    lanes: list[dict[str,Any]],
    grant_contract: dict[str,Any],
    *, now: datetime, integrity_error: str | None = None,
) -> dict[str,Any]:
    """Never admit an unbounded GPU task on a hard-expiring grant.

    This is a *truthful gate* until the existing run-campaign is upgraded
    with an actual science-safe bounded-unit exit/lease-release handshake.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise GPUWindowError("current time must be timezone-aware")
    assignments=snapshot.get("assignments",[])
    cancel_ids={row["lane_id"] for row in assignments}
    new_assignments=[]
    handoffs=[]
    grants=grant_contract["grants"]
    for lane in lanes:
        lid=lane["id"]
        linked=[grants[g.casefold()] for g in lane["gpu_uuids"]]
        earliest=max(g.starts_at for g in linked)
        cutoff=min(g.drain_at for g in linked)
        deadline=min(g.handoff_at for g in linked)
        # Report each physical grant independently. The *lane* may have no
        # common execution window: later GPU admission must not hide an
        # earlier GPU's already-expired handoff obligation.
        disjoint=earliest >= cutoff
        assigned=snapshot["lanes"][lid]["assigned_arm"]
        # A draft assignment is not an already-running child. Only a RUNNING
        # arm holds this Supervisor's verified lane for release reporting.
        occupied=(
            assigned if assigned is not None
            and snapshot["arms"][assigned]["action"]=="RUNNING" else None
        )
        external=bool(snapshot.get("blocked_by_external_controller")) and not occupied

        per_gpu=[]
        for g in linked:
            if integrity_error is not None:
                card_status="CONTRACT_DRIFT_UNVERIFIED"
            elif external:
                card_status=(
                    "RELEASE_PENDING_EXTERNAL" if now >= g.drain_at
                    else "EXTERNAL_CONTROLLER_UNVERIFIED"
                )
            elif now >= g.handoff_at:
                card_status="RELEASE_PENDING" if occupied else "RELEASE_UNVERIFIED"
            elif now >= g.drain_at:
                card_status="DRAINING" if occupied else "RELEASE_UNVERIFIED"
            elif now < g.starts_at:
                # An already-running owned arm before the grant starts is
                # more urgent than an ordinary future-window notice.
                card_status=(
                    "RUNNING_BEFORE_GRANTED_WINDOW" if occupied
                    else "NOT_YET_GRANTED"
                )
            else:
                card_status="ACTIVE_WINDOW_NO_SAFE_STOP"
            per_gpu.append({
                "gpu_uuid":g.gpu_uuid,
                "authorization_ref":g.authorization_ref,
                "starts_at":g.starts_at.isoformat(),
                "drain_at":g.drain_at.isoformat(),
                "handoff_at":g.handoff_at.isoformat(),
                "classification":card_status,
                "live_release_confirmed":False,
                "current_arm_id":occupied,
            })

        if integrity_error is not None:
            classification="CONTRACT_DRIFT_UNVERIFIED"
            detail=integrity_error+"; no dispatch, new grant source/host readback required"
        elif external:
            classification=(
                "RELEASE_PENDING_EXTERNAL" if now >= cutoff
                else "EXTERNAL_CONTROLLER_UNVERIFIED"
            )
            detail="external controller owns unresolved work; this supervisor cannot observe or release it"
        elif now>=deadline:
            classification="RELEASE_PENDING" if occupied else "RELEASE_UNVERIFIED"
            detail="handoff instant passed; live ZJU owner/lease/holder readback still required"
        elif now>=cutoff:
            classification="DRAINING" if occupied else "RELEASE_UNVERIFIED"
            detail="no new work; inspect current science and release at its permitted boundary"
        elif occupied and now<earliest:
            classification="RUNNING_BEFORE_GRANTED_WINDOW"
            detail=(
                "an existing running arm precedes at least one GPU grant; "
                "verify live owner, grant and science boundary without preempting"
            )
        elif disjoint:
            classification="NO_COMMON_GPU_WINDOW"
            detail="existing multi-GPU lane grants have no shared runnable interval"
        elif now<earliest:
            classification="NOT_YET_GRANTED"
            detail="resource window has not started"
        else:
            classification="ACTIVE_WINDOW_NO_SAFE_STOP"
            detail="canonical run-campaign lacks a verified bounded-safe-stop capability"
        handoffs.append({
            "lane_id":lid,"gpu_uuids":lane["gpu_uuids"],
            "per_gpu":per_gpu,
            "grant_refs":[g.authorization_ref for g in linked],
            "earliest_start_at":earliest.isoformat(),
            "latest_drain_at":cutoff.isoformat(),
            "earliest_handoff_at":deadline.isoformat(),
            "classification":classification,
            "live_release_confirmed":False,
            "reason":detail,
            "current_arm_id":occupied,
        })
        if lid not in cancel_ids:
            continue
        row=next(x for x in assignments if x["lane_id"]==lid)
        aid=row["arm_id"]
        # Remove speculative assignment from future / new starts. Status MUST
        # distinguish this from an active pre-existing child, which we do NOT kill.
        snapshot["lanes"][lid]["assigned_arm"]=None
        state=snapshot["arms"][aid]
        state["action"]="BLOCKED_TIME_WINDOW_NO_SAFE_STOP"
        state["reason"]=detail
        state.pop("lane_id",None)
    snapshot["assignments"]=new_assignments
    snapshot["gpu_time_window"]={
        "schema":SCHEMA,"contract_sha256":grant_contract["sha256"],
        "status":"BLOCKED_CONTRACT_DRIFT" if integrity_error else "OBSERVE_AND_BLOCK_UNBOUNDED",
        "handoffs":handoffs,
        "reminder":"Only host UUID/owner/Ray/Holder live readback can confirm actual release.",
    }
    return snapshot
