#!/usr/bin/env python3
"""CPU-only, deterministic advisory for time-limited GPU grants.

This is a *planning projection*, not a GPU scheduler, lease holder, live
resource check, scientific authorization, or stop/recovery controller.
Existing OpenEVO/Ray/ZJU authorities alone may launch or release workloads.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import combinations
import hashlib
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


CONTRACT_VERSION = "openevo.gpu-window-advice.202610130200"


class PlanError(ValueError):
    """Malformed or unsafe advisory input. No partial execution."""


def require_int(value: Any, name: str, *, minimum: int = 0, maximum: int = 100000) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise PlanError(f"{name} must be integer between {minimum} and {maximum}")
    return value


def parsed_clock(value: Any, name: str, zone: ZoneInfo) -> datetime:
    if not isinstance(value, str):
        raise PlanError(f"{name} requires a dated ISO-8601 timestamp with timezone offset")
    try:
        clock = datetime.fromisoformat(value)
    except ValueError as exc:
        raise PlanError(f"{name}: invalid timestamp: {value}") from exc
    if clock.tzinfo is None or clock.utcoffset() is None:
        raise PlanError(f"{name}: naive clock forbidden; include an explicit offset")
    if clock.second or clock.microsecond:
        raise PlanError(f"{name}: planning timestamps must be minute-aligned")
    local = clock.astimezone(zone)
    if local.isoformat() != clock.isoformat():
        raise PlanError(f"{name}: timezone/offset must match the declared planning timezone")
    return local


def minute(value: int) -> timedelta:
    return timedelta(minutes=value)


@dataclass(frozen=True)
class Window:
    gpu: int
    start: datetime
    handoff: datetime
    buffer_minutes: int
    authorization_ref: str
    handoff_to: str | None

    @property
    def cutoff(self) -> datetime:
        return self.handoff - minute(self.buffer_minutes)


@dataclass(frozen=True)
class Work:
    job_id: str
    priority: int
    gpu_count: int
    allowed_gpus: tuple[int, ...]
    units: int
    unit_p95_minutes: int
    startup_minutes: int
    release_minutes: int
    dependencies: tuple[str, ...]
    ready_for_planning: bool
    allow_gpu_rebind_between_units: bool
    science_ref: str
    estimate_ref: str

    @property
    def duration(self) -> timedelta:
        return minute(self.startup_minutes + self.unit_p95_minutes + self.release_minutes)


@dataclass
class State:
    work: Work
    remaining: int
    next_ready: datetime
    affinity: tuple[int, ...] | None = None
    complete_at: datetime | None = None


def load_contract(blob: dict[str, Any], at: str) -> tuple[str, datetime, list[Window], list[Work], int]:
    if not isinstance(blob, dict) or blob.get("contract_version") != CONTRACT_VERSION:
        raise PlanError(f"requires contract_version={CONTRACT_VERSION}")
    if blob.get("execution_mode") != "simulation-only":
        raise PlanError("execution_mode must be simulation-only: this module never launches a GPU")
    try:
        zone = ZoneInfo(blob["timezone"])
    except (KeyError, TypeError, ValueError, ZoneInfoNotFoundError) as exc:
        raise PlanError("timezone must be a valid IANA zone, e.g., Asia/Singapore") from exc
    reference = parsed_clock(at, "--at", zone)
    window_raw = blob.get("gpu_windows")
    if not isinstance(window_raw, list) or not window_raw:
        raise PlanError("gpu_windows must be a non-empty list of explicit grants")
    windows: list[Window] = []
    for i, row in enumerate(window_raw):
        if not isinstance(row, dict):
            raise PlanError(f"gpu_windows[{i}] must be an object")
        index = require_int(row.get("gpu_index"), "gpu_index", maximum=255)
        start = parsed_clock(row.get("starts_at"), "starts_at", zone)
        handoff = parsed_clock(row.get("handoff_at"), "handoff_at", zone)
        buffer = require_int(row.get("handoff_buffer_minutes"), "handoff_buffer_minutes", minimum=1, maximum=1440)
        auth = row.get("authorization_ref")
        if not isinstance(auth, str) or not auth.strip():
            raise PlanError("authorization_ref must identify the user-provided grant, not a live permission claim")
        handoff_to = row.get("handoff_to")
        if handoff_to is not None and (not isinstance(handoff_to, str) or not handoff_to.strip()):
            raise PlanError("handoff_to, when present, must be a non-empty label")
        if handoff - minute(buffer) <= start:
            raise PlanError(f"gpu {index}: handoff/buffer leaves no safe usable time")
        windows.append(Window(index, start, handoff, buffer, auth, handoff_to))
    by_gpu: dict[int, list[Window]] = {}
    for row in windows:
        by_gpu.setdefault(row.gpu, []).append(row)
    for gpu, entries in by_gpu.items():
        entries.sort(key=lambda x: x.start)
        for prev, later in zip(entries, entries[1:]):
            if prev.handoff > later.start:
                raise PlanError(f"gpu {gpu}: overlapping human grants are ambiguous")
    cap = require_int(blob.get("max_parallel_gpus"), "max_parallel_gpus", minimum=1, maximum=256)
    if cap > len(by_gpu):
        raise PlanError("max_parallel_gpus may not exceed number of explicitly granted GPU indices")

    jobs_raw = blob.get("jobs")
    if not isinstance(jobs_raw, list):
        raise PlanError("jobs must be a list (empty is allowed for handoff-only preview)")
    jobs: list[Work] = []
    known: set[str] = set()
    for n, row in enumerate(jobs_raw):
        if not isinstance(row, dict):
            raise PlanError(f"jobs[{n}] must be an object")
        job_id = row.get("job_id")
        if not isinstance(job_id, str) or not job_id or job_id in known:
            raise PlanError(f"invalid/duplicate job_id: {job_id!r}")
        known.add(job_id)
        prio = require_int(row.get("priority"), "priority (1=highest)", minimum=1, maximum=4)
        count = require_int(row.get("gpu_count"), "gpu_count", minimum=1, maximum=cap)
        subset = row.get("eligible_gpu_indices")
        if not isinstance(subset, list) or not subset:
            raise PlanError("eligible_gpu_indices must be explicit and non-empty")
        indices = [require_int(x, "eligible gpu index", maximum=255) for x in subset]
        if len(set(indices)) != len(indices) or not set(indices).issubset(by_gpu):
            raise PlanError(f"{job_id}: eligible GPU indices must be unique and within declared grants")
        if count > len(indices):
            raise PlanError(f"{job_id}: gpu_count exceeds eligible GPUs")
        dep = row.get("depends_on", [])
        if not isinstance(dep, list) or any(not isinstance(x, str) for x in dep) or len(set(dep)) != len(dep):
            raise PlanError(f"{job_id}: depends_on must contain unique job IDs")
        ready = row.get("ready_for_planning")
        rebind = row.get("allow_gpu_rebind_between_units", False)
        if type(ready) is not bool or type(rebind) is not bool:
            raise PlanError("ready_for_planning and allow_gpu_rebind_between_units must be booleans")
        science = row.get("science_ref")
        est = row.get("estimate_ref")
        if not isinstance(science, str) or not science.strip() or not isinstance(est, str) or not est.strip():
            raise PlanError(f"{job_id}: science_ref and estimate_ref must identify supporting evidence")
        jobs.append(Work(
            job_id,prio,count,tuple(sorted(indices)),
            require_int(row.get("units"), "units",minimum=1,maximum=10000),
            require_int(row.get("unit_p95_minutes"), "unit_p95_minutes",minimum=1,maximum=43200),
            require_int(row.get("startup_minutes"), "startup_minutes",maximum=1440),
            require_int(row.get("release_minutes"), "release_minutes",minimum=1,maximum=1440),
            tuple(dep),ready,rebind,science,est,
        ))
    all_ids = {w.job_id for w in jobs}
    if any(d not in all_ids or d == w.job_id for w in jobs for d in w.dependencies):
        raise PlanError("dependency graph contains an unknown or self-referencing job")
    deps = {w.job_id:set(w.dependencies) for w in jobs}
    discovered: set[str] = set()
    while deps:
        ready_ids = {job_id for job_id,others in deps.items() if others <= discovered}
        if not ready_ids:
            raise PlanError("job dependencies contain a cycle")
        discovered |= ready_ids
        for job_id in ready_ids:
            del deps[job_id]
    return blob["timezone"], reference, sorted(windows,key=lambda x:(x.start,x.gpu)),jobs,cap


def advise(blob: dict[str, Any], *, at: str) -> dict[str, Any]:
    timezone, origin, windows, jobs, parallel_cap = load_contract(blob, at)
    work = {w.job_id: State(w,w.units,origin) for w in jobs}
    windows_for_gpu = {g:[x for x in windows if x.gpu == g] for g in sorted({w.gpu for w in windows})}
    bookings: dict[int, list[tuple[datetime,datetime]]] = {g:[] for g in windows_for_gpu}
    planned: list[dict[str,Any]] = []
    horizon = max(w.handoff for w in windows)
    t = origin

    def window_at(gpu: int, when: datetime, end: datetime) -> Window | None:
        return next((x for x in windows_for_gpu[gpu] if x.start <= when and end <= x.cutoff), None)

    def active_at(when: datetime) -> set[int]:
        return {g for g,rows in bookings.items() if any(s <= when < e for s,e in rows)}

    def is_free(gpu: int, start: datetime, end: datetime) -> bool:
        return (window_at(gpu,start,end) is not None
                and not any(s < end and start < e for s,e in bookings[gpu]))

    def choices(st: State, when: datetime) -> list[tuple[int,...]]:
        task=st.work
        if not task.ready_for_planning or st.remaining <= 0 or when < st.next_ready:
            return []
        if any(work[d].complete_at is None or work[d].complete_at > when for d in task.dependencies):
            return []
        eligible = list(st.affinity if st.affinity is not None else task.allowed_gpus)
        sets=[]
        if st.affinity and not task.allow_gpu_rebind_between_units:
            possible=[st.affinity]
        else:
            possible=list(combinations(eligible,task.gpu_count))
        if len(active_at(when))+task.gpu_count > parallel_cap:
            return []
        for cards in possible:
            if all(is_free(g,when,when+task.duration) for g in cards):
                sets.append(cards)
        def score(cards: tuple[int,...]) -> tuple[float,tuple[int,...]]:
            # Prefer expiring capacity; preserve longer grants for larger future work.
            slack=min((window_at(g,when,when+task.duration).cutoff-(when+task.duration)).total_seconds() for g in cards)
            return (slack,cards)
        return sorted(sets,key=score)

    def earliest_anchor(st: State, now: datetime) -> tuple[datetime,tuple[int,...]] | None:
        task=st.work
        if not task.ready_for_planning or st.remaining <= 0:
            return None
        # A predecessor with a booked finish is a known future release.
        # Reserve its dependent high-priority gang from that release onward,
        # rather than letting lower-priority long work steal a needed card.
        dependency_finishes=[work[d].complete_at for d in task.dependencies]
        if any(finish is None for finish in dependency_finishes):
            return None
        ready_at=max(now,st.next_ready,*dependency_finishes)
        points={ready_at}
        for wd in windows:
            points.add(wd.start)
        for rows in bookings.values():
            for _,end in rows: points.add(end)
        for when in sorted(x for x in points if ready_at <= x < horizon):
            options=choices(st,when)
            if options:
                return when,options[0]
        return None

    while t < horizon and any(st.remaining for st in work.values()):
        while True:
            chosen=None
            for st in sorted(work.values(),key=lambda s:(s.work.priority,s.work.job_id)):
                for cards in choices(st,t):
                    finish=t+st.work.duration
                    # Protect the earliest feasible gang placement of a higher-priority
                    # READY arm. Short jobs may still backfill *before* this reservation.
                    overlap=False
                    for higher in work.values():
                        if higher.work.priority >= st.work.priority or higher.work.job_id==st.work.job_id:
                            continue
                        anchor=earliest_anchor(higher,t)
                        if anchor and t < anchor[0]+higher.work.duration and finish > anchor[0] and set(cards)&set(anchor[1]):
                            overlap=True
                            break
                    if not overlap:
                        chosen=(st,cards)
                        break
                if chosen: break
            if chosen is None: break
            st,cards=chosen
            before=st.work.units-st.remaining
            end=t+st.work.duration
            source=[window_at(g,t,end) for g in cards]
            if any(w is None for w in source):
                raise AssertionError("planner broke availability contract")
            for g in cards:
                bookings[g].append((t,end))
            st.remaining-=1
            st.next_ready=end
            if st.affinity is None and not st.work.allow_gpu_rebind_between_units:
                st.affinity=cards
            if st.remaining==0:
                st.complete_at=end
            planned.append({
                "job_id":st.work.job_id,"unit":before+1,
                "gpu_indices":list(cards),"priority":st.work.priority,
                "planned_start_at":t.isoformat(),
                "planned_science_finish_at":(t+minute(st.work.startup_minutes+st.work.unit_p95_minutes)).isoformat(),
                "planned_released_at":end.isoformat(),
                "authorization_refs":[w.authorization_ref for w in source],
                "handoff_at_by_gpu":{str(w.gpu):w.handoff.isoformat() for w in source},
            })
        change={w.start for w in windows if w.start>t}
        change.update(end for rows in bookings.values() for _,end in rows if end>t)
        change.update(st.next_ready for st in work.values() if st.next_ready>t)
        change.update(w.cutoff for w in windows if w.cutoff>t)
        for st in work.values():
            anchor=earliest_anchor(st,t)
            if anchor and anchor[0]>t:change.add(anchor[0])
        if not change:break
        new_t=min(change)
        if new_t<=t:raise AssertionError("scheduler did not advance")
        t=new_t

    def effective_minutes(start:datetime,end:datetime)->int:
        return max(0,int((end-start).total_seconds()/60))

    # The user's global parallel-card cap limits physically schedulable
    # GPU-minutes; do not report excess candidate cards as avoidable idleness.
    availability = [
        (max(origin,w.start),w.cutoff)
        for w in windows if w.cutoff > max(origin,w.start)
    ]
    total_granted_minutes=sum(effective_minutes(start,end) for start,end in availability)
    ticks=sorted({instant for pair in availability for instant in pair})
    grant_minutes=sum(
        effective_minutes(begin,end)*min(parallel_cap,sum(
            start <= begin < stop for start,stop in availability
        )) for begin,end in zip(ticks,ticks[1:])
    )
    used_minutes=sum(effective_minutes(begin,end) for rows in bookings.values() for begin,end in rows)
    if used_minutes > grant_minutes:
        raise AssertionError("planned GPU-minutes exceed the authorized parallel-card envelope")
    pending=[{
        "job_id":st.work.job_id,
        "units_pending":st.remaining,
        "reason":("NOT_PLANNING_READY" if not st.work.ready_for_planning else
                  "DEPENDENCY_PENDING" if any(work[d].complete_at is None for d in st.work.dependencies) else
                  "WINDOW_OR_SAFE_BOUNDARY_UNAVAILABLE"),
    } for st in sorted(work.values(),key=lambda x:(x.work.priority,x.work.job_id)) if st.remaining]
    return {
        "contract_version":CONTRACT_VERSION,
        "status":"DRY_RUN_ONLY_NOT_AUTHORITY",
        "plan_sha256":"sha256:"+hashlib.sha256(json.dumps(blob,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest(),
        "timezone":timezone,"planning_at":origin.isoformat(),"max_parallel_gpus":parallel_cap,
        "scheduled_units":planned,
        "pending_jobs":pending,
        "handoffs":[{
            "gpu_index":w.gpu,"starts_at":w.start.isoformat(),
            "latest_scheduling_cutoff":w.cutoff.isoformat(),
            "handoff_at":w.handoff.isoformat(),
            "handoff_buffer_minutes":w.buffer_minutes,
            "authorization_ref":w.authorization_ref,
            "handoff_to":w.handoff_to,
        } for w in windows],
        "candidate_grant_gpu_minutes_before_concurrency_cap":total_granted_minutes,
        "projected_usable_gpu_minutes":grant_minutes,
        "projected_busy_gpu_minutes":used_minutes,
        "projected_idle_gpu_minutes":grant_minutes-used_minutes,
        "notes":[
            "Projection only. Every future launch still requires ZJU live index-to-UUID/owner/VRAM/lease readback and exact scientific release.",
            "No background controller or hard handoff enforcement was installed by this planner.",
            "The handoff deadline is an actual release obligation, not permission to terminate other users or non-owned containers.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True,help="read-only planning input JSON")
    parser.add_argument("--at",required=True,help="explicit planning timestamp with date and timezone offset")
    args=parser.parse_args(argv)
    try:
        data=json.loads(args.plan.read_text(encoding="utf-8"))
        decision=advise(data,at=args.at)
    except (PlanError,OSError,json.JSONDecodeError) as exc:
        parser.exit(2, f"INVALID_GPU_WINDOW_PLAN: {exc}\n")
    print(json.dumps(decision,ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
