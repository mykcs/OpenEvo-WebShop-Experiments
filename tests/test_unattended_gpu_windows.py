"""End-to-end CPU-only handoff guard tests for existing supervisor.

No GPU, shell/docker process, Ray lease, user science, or secret is accessed.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path

import pytest

from scripts.openevo_runtime import unattended_supervisor as us
from scripts.openevo_runtime import unattended_gpu_windows as tw


def _file(path:Path,doc:dict)->str:
    path.write_text(json.dumps(doc,sort_keys=True)+"\n",encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(tmp_path:Path)->tuple[Path,dict]:
    checkout=tmp_path/"checkout"
    (checkout/"scripts").mkdir(parents=True)
    (checkout/"scripts/oe").write_text("#!/bin/sh\n")
    u="GPU-00000000-0000-0000-0000-000000000000"
    v="GPU-00000000-0000-0000-0000-000000000000"
    data={
        "schema":us.SCHEMA,
        "scientific_authority":False,
        "state_root":str(tmp_path/"state"),
        "max_parallel_arms":1,
        "poll_seconds":0.01,
        "lanes":[{"id":"alpha","gpu_uuids":[u,v]}],
        "arms":[{
            "id":"alpha-arm",
            "priority":1,
            "run_output_root":str(tmp_path/"run"),
            "authority_path":str(checkout/"authority.json"),
            "rounds":[152,153],
            "eligible_lanes":["alpha"],
            "launch":{
                "cwd":str(checkout),
                "expected_git_head":"a"*40,
                "argv":["./scripts/oe","run-campaign","--release","release.json",
                    "--output-root","${CONTROL_ROOT}","--run-output-root","${RUN_ROOT}",
                    "--gpus","${GPU_UUIDS}","--wait","--resume"],
                "env":{"OPENEVO_AUTHORIZED_GPU_UUIDS":"${GPU_UUIDS}"},
            },
        }],
    }
    return tmp_path/"supervisor.json",data


def _grant(uuid:str,begin="2026-10-14T08:00:00+08:00",end="2026-10-14T11:00:00+08:00",buffer=1200):
    return {
        "gpu_uuid":uuid,
        "starts_at":begin,
        "handoff_at":end,
        "drain_buffer_seconds":buffer,
        "authorization_ref":"SYNTHETIC_USER_GRANT_NOT_REAL",
    }


def _bound(tmp_path:Path,doc:dict,grants:list[dict]|None=None)->Path:
    if grants is None:
        grants=[_grant(g) for g in doc["lanes"][0]["gpu_uuids"]]
    policy=tmp_path/"gpu-handoff.json"
    digest=_file(policy,{
        "schema":tw.SCHEMA,
        "execution_mode":"observe-and-block-unbounded",
        "timezone":"Asia/Singapore",
        "grants":grants,
    })
    doc["gpu_handoff_contract"]={"path":str(policy),"sha256":digest}
    supervisor=tmp_path/"supervisor.json"
    _file(supervisor,doc)
    return supervisor


def _ready(monkeypatch):
    monkeypatch.setattr(us,"discover",
        lambda arm:{"action":"READY_TO_DISPATCH","next_round":152,"reason":"frozen and prepared"})


def _ts(value:str)->float:
    return datetime.fromisoformat(value).timestamp()


def test_legacy_supervisor_is_unchanged_when_time_grant_absent(tmp_path,monkeypatch):
    path,doc=_config(tmp_path)
    _file(path,doc)
    cfg=us.config(path)
    _ready(monkeypatch)
    result=us.plan(cfg)
    assert result["assignments"]==[{"arm_id":"alpha-arm","lane_id":"alpha","action":"READY_TO_DISPATCH"}]
    assert "gpu_time_window" not in result


@pytest.mark.parametrize("ambiguous", [None, False, [], "", {}, {"path":None,"sha256":None}])
def test_explicitly_malformed_contract_never_falls_back_to_unrestricted_gpu(tmp_path,ambiguous):
    path,doc=_config(tmp_path)
    doc["gpu_handoff_contract"]=ambiguous
    _file(path,doc)
    with pytest.raises(us.SupervisorError,match="gpu_handoff_contract"):
        us.config(path)


def test_time_window_does_not_grant_science_even_during_active_interval(tmp_path,monkeypatch):
    path,doc=_config(tmp_path)
    cfg=us.config(_bound(tmp_path,doc))
    _ready(monkeypatch)
    snap=us.plan(cfg,now_utc=_ts("2026-10-14T09:00:00+08:00"))
    assert snap["assignments"]==[]
    assert snap["arms"]["alpha-arm"]["action"]=="BLOCKED_TIME_WINDOW_NO_SAFE_STOP"
    handoff=snap["gpu_time_window"]["handoffs"][0]
    assert handoff["classification"]=="ACTIVE_WINDOW_NO_SAFE_STOP"
    assert handoff["latest_drain_at"]=="2026-10-14T10:40:00+08:00"
    assert handoff["earliest_handoff_at"]=="2026-10-14T11:00:00+08:00"
    assert handoff["live_release_confirmed"] is False


def test_expiry_and_future_windows_do_not_become_unverified_success(tmp_path,monkeypatch):
    path,doc=_config(tmp_path)
    cfg=us.config(_bound(tmp_path,doc))
    _ready(monkeypatch)
    classifications=[
        ( "2026-10-14T07:59:00+08:00","NOT_YET_GRANTED"),
        ( "2026-10-14T10:39:00+08:00","ACTIVE_WINDOW_NO_SAFE_STOP"),
        ( "2026-10-14T10:40:00+08:00","RELEASE_UNVERIFIED"),
        ( "2026-10-14T11:00:00+08:00","RELEASE_UNVERIFIED"),
    ]
    for when,expected in classifications:
        s=us.plan(cfg,now_utc=_ts(when))
        assert s["gpu_time_window"]["handoffs"][0]["classification"]==expected
        assert s["assignments"]==[]
        assert s["gpu_time_window"]["handoffs"][0]["live_release_confirmed"] is False


def test_existing_running_arm_before_grant_is_not_reported_as_waiting(tmp_path,monkeypatch):
    path,doc=_config(tmp_path)
    cfg=us.config(_bound(tmp_path,doc))
    _ready(monkeypatch)
    running={"alpha-arm":{"lane_id":"alpha","dispatch_id":"existing-not-created-by-this-plan"}}
    snap=us.plan(cfg,running,now_utc=_ts("2026-10-14T07:30:00+08:00"))
    handoff=snap["gpu_time_window"]["handoffs"][0]
    assert snap["assignments"]==[]
    assert snap["arms"]["alpha-arm"]["action"]=="RUNNING"
    assert handoff["classification"]=="RUNNING_BEFORE_GRANTED_WINDOW"
    assert handoff["current_arm_id"]=="alpha-arm"
    assert all(x["classification"]=="RUNNING_BEFORE_GRANTED_WINDOW" for x in handoff["per_gpu"])
    assert all(x["live_release_confirmed"] is False for x in handoff["per_gpu"])
    assert handoff["live_release_confirmed"] is False
    assert "without preempting" in handoff["reason"]


def test_partial_multigpu_grant_with_running_arm_flags_future_card(tmp_path,monkeypatch):
    path,doc=_config(tmp_path)
    cards=doc["lanes"][0]["gpu_uuids"]
    grants=[
        _grant(cards[0],end="2026-10-14T15:00:00+08:00"),
        _grant(cards[1],begin="2026-10-14T12:00:00+08:00",
               end="2026-10-14T15:00:00+08:00"),
    ]
    cfg=us.config(_bound(tmp_path,doc,grants))
    _ready(monkeypatch)
    running={"alpha-arm":{"lane_id":"alpha","dispatch_id":"existing"}}
    snap=us.plan(cfg,running,now_utc=_ts("2026-10-14T09:00:00+08:00"))
    handoff=snap["gpu_time_window"]["handoffs"][0]
    assert snap["assignments"]==[]
    assert handoff["classification"]=="RUNNING_BEFORE_GRANTED_WINDOW"
    assert [row["classification"] for row in handoff["per_gpu"]]==[
        "ACTIVE_WINDOW_NO_SAFE_STOP","RUNNING_BEFORE_GRANTED_WINDOW"
    ]
    assert handoff["current_arm_id"]=="alpha-arm"
    assert handoff["live_release_confirmed"] is False


def test_running_child_is_not_killed_or_mistakenly_freed_at_handoff(tmp_path,monkeypatch):
    path,doc=_config(tmp_path)
    cfg=us.config(_bound(tmp_path,doc))
    _ready(monkeypatch)
    active={"alpha-arm":{"lane_id":"alpha","dispatch_id":"already-owned"}}
    for when,expect in [
        ("2026-10-14T10:40:00+08:00","DRAINING"),
        ("2026-10-14T11:00:00+08:00","RELEASE_PENDING"),
    ]:
        s=us.plan(cfg,active,now_utc=_ts(when))
        h=s["gpu_time_window"]["handoffs"][0]
        assert s["arms"]["alpha-arm"]["action"]=="RUNNING"
        assert h["classification"]==expect
        assert h["current_arm_id"]=="alpha-arm"
        assert h["live_release_confirmed"] is False


@pytest.mark.parametrize("when,expected", [
    ("2026-10-14T09:00:00+08:00","EXTERNAL_CONTROLLER_UNVERIFIED"),
    ("2026-10-14T10:40:00+08:00","RELEASE_PENDING_EXTERNAL"),
    ("2026-10-14T11:00:00+08:00","RELEASE_PENDING_EXTERNAL"),
])
def test_external_controller_is_never_a_false_free_handoff(tmp_path,monkeypatch,when,expected):
    path,doc=_config(tmp_path)
    cfg=us.config(_bound(tmp_path,doc))
    monkeypatch.setattr(us,"discover",
        lambda arm:{"action":"RUNNING_EXTERNAL","reason":"existing writer lock held"})
    s=us.plan(cfg,now_utc=_ts(when))
    handoff=s["gpu_time_window"]["handoffs"][0]
    assert s["blocked_by_external_controller"] is True
    assert handoff["classification"]==expected
    assert handoff["live_release_confirmed"] is False
    assert s["assignments"]==[]
    assert "cannot observe or release" in handoff["reason"]


@pytest.mark.parametrize("when,lane_expected,first_expected,second_expected",[
    ("2026-10-14T07:00:00+08:00","NO_COMMON_GPU_WINDOW",
      "NOT_YET_GRANTED","NOT_YET_GRANTED"),
    ("2026-10-14T08:30:00+08:00","NO_COMMON_GPU_WINDOW",
      "ACTIVE_WINDOW_NO_SAFE_STOP","NOT_YET_GRANTED"),
    ("2026-10-14T10:00:00+08:00","RELEASE_UNVERIFIED",
      "RELEASE_UNVERIFIED","NOT_YET_GRANTED"),
    ("2026-10-14T12:15:00+08:00","RELEASE_UNVERIFIED",
      "RELEASE_UNVERIFIED","ACTIVE_WINDOW_NO_SAFE_STOP"),
])
def test_nonoverlapping_multigpu_grants_preserve_each_card_expiry(
    tmp_path,monkeypatch,when,lane_expected,first_expected,second_expected,
):
    path,doc=_config(tmp_path)
    cards=doc["lanes"][0]["gpu_uuids"]
    grants=[
        _grant(cards[0],begin="2026-10-14T08:00:00+08:00",
               end="2026-10-14T09:00:00+08:00",buffer=600),
        _grant(cards[1],begin="2026-10-14T12:00:00+08:00",
               end="2026-10-14T15:00:00+08:00",buffer=600),
    ]
    cfg=us.config(_bound(tmp_path,doc,grants))
    _ready(monkeypatch)
    snap=us.plan(cfg,now_utc=_ts(when))
    row=snap["gpu_time_window"]["handoffs"][0]
    assert snap["assignments"]==[]
    assert row["classification"]==lane_expected
    assert row["per_gpu"][0]["gpu_uuid"]==cards[0]
    assert row["per_gpu"][1]["gpu_uuid"]==cards[1]
    assert [card["classification"] for card in row["per_gpu"]]==[
        first_expected,second_expected,
    ]
    assert all(card["live_release_confirmed"] is False for card in row["per_gpu"])


def test_expiring_first_card_is_not_hidden_by_future_card_with_live_worker(
    tmp_path,monkeypatch,
):
    path,doc=_config(tmp_path)
    cards=doc["lanes"][0]["gpu_uuids"]
    grants=[
        _grant(cards[0],end="2026-10-14T09:00:00+08:00",buffer=600),
        _grant(cards[1],begin="2026-10-14T12:00:00+08:00",
               end="2026-10-14T15:00:00+08:00",buffer=600),
    ]
    cfg=us.config(_bound(tmp_path,doc,grants))
    _ready(monkeypatch)
    running={"alpha-arm":{"lane_id":"alpha","dispatch_id":"verified-existing"}}
    snap=us.plan(cfg,running,now_utc=_ts("2026-10-14T10:00:00+08:00"))
    lane=snap["gpu_time_window"]["handoffs"][0]
    assert lane["classification"]=="RELEASE_PENDING"
    assert lane["per_gpu"][0]["classification"]=="RELEASE_PENDING"
    # Even when the lane is already past another card's release deadline,
    # the future card must reveal that the existing arm is running before grant.
    assert lane["per_gpu"][1]["classification"]=="RUNNING_BEFORE_GRANTED_WINDOW"
    assert lane["current_arm_id"]=="alpha-arm"
    assert lane["live_release_confirmed"] is False
    assert snap["arms"]["alpha-arm"]["action"]=="RUNNING"
    assert not snap["assignments"]


def test_direct_start_mutation_boundary_cannot_bypass_guard(tmp_path,monkeypatch):
    path,doc=_config(tmp_path)
    cfg=us.config(_bound(tmp_path,doc))
    monkeypatch.setattr(us.subprocess,"Popen",
        lambda *a,**kw: pytest.fail("time-limited supervisor launched unbounded GPU work"))
    with pytest.raises(us.SupervisorError,match="bounded-safe-stop"):
        us.start(cfg,cfg["arms"][0],cfg["lanes"][0])
    assert not (Path(cfg["state_root"])/"dispatches").exists()


def test_execute_flag_without_bounded_controller_records_blocker_and_zero_launch(tmp_path,monkeypatch):
    path,doc=_config(tmp_path)
    cfg=us.config(_bound(tmp_path,doc))
    _ready(monkeypatch)
    monkeypatch.setattr(us.subprocess,"Popen",
        lambda *a,**kw: pytest.fail("strict handoff accidentally dispatched"))
    result=us.run(cfg,execute=True,once=True,sleep_fn=lambda _: None)
    assert result["assignments"]==[]
    assert result["arms"]["alpha-arm"]["action"]=="BLOCKED_TIME_WINDOW_NO_SAFE_STOP"
    events=[json.loads(line)["event"] for line in
        (Path(cfg["state_root"])/"events.jsonl").read_text().splitlines()]
    assert "GPU_HANDOFF_REQUIRES_LIVE_RELEASE_READBACK" in events
    assert "DISPATCH_STARTED" not in events


@pytest.mark.parametrize("mutation,reason",[
    ("bad_hash","SHA-256"),
    ("mismatched_uuid","grants must bind"),
    ("no_grant","grants"),
    ("duplicate","duplicate"),
    ("bad_zone","offset/zone"),
    ("naive_time","naive"),
    ("no_buffer","drain_buffer"),
    ("zero_interval","no safe interval"),
    ("unknown_mode","observe-and-block"),
])
def test_invalid_grant_or_binding_never_loads_as_authority(tmp_path,mutation,reason):
    path,doc=_config(tmp_path)
    grants=[_grant(g) for g in doc["lanes"][0]["gpu_uuids"]]
    if mutation=="mismatched_uuid":grants[0]["gpu_uuid"]="GPU-"+("f"*32)
    if mutation=="no_grant":grants=[]
    if mutation=="duplicate":grants[1]["gpu_uuid"]=grants[0]["gpu_uuid"].upper()
    if mutation=="bad_zone":grants[0]["starts_at"]="2026-10-14T00:00:00+00:00"
    if mutation=="naive_time":grants[0]["starts_at"]="2026-10-14T08:00:00"
    if mutation=="no_buffer":grants[0]["drain_buffer_seconds"]=0
    if mutation=="zero_interval":grants[0]["drain_buffer_seconds"]=12000
    policy_path=_bound(tmp_path,doc,grants)
    if mutation=="bad_hash":
        doc["gpu_handoff_contract"]["sha256"]="0"*64
        _file(policy_path,doc)
    if mutation=="unknown_mode":
        payload=Path(doc["gpu_handoff_contract"]["path"])
        policy=json.loads(payload.read_text())
        policy["execution_mode"]="execute"
        doc["gpu_handoff_contract"]["sha256"]=_file(payload,policy)
        _file(policy_path,doc)
    with pytest.raises(us.SupervisorError,match=reason):
        us.config(policy_path)


def test_cannot_silently_widen_existing_supervisor_lane_topology(tmp_path):
    path,doc=_config(tmp_path)
    extra=_grant("GPU-"+("e"*32))
    with pytest.raises(us.SupervisorError,match="grants must bind"):
        us.config(_bound(tmp_path,doc,[_grant(g) for g in doc["lanes"][0]["gpu_uuids"]]+[extra]))


def test_live_grant_source_drift_is_visible_and_never_admits(tmp_path,monkeypatch):
    path,doc=_config(tmp_path)
    cfg=us.config(_bound(tmp_path,doc))
    _ready(monkeypatch)
    contract=Path(cfg["_gpu_window"]["path"])
    content=json.loads(contract.read_text())
    content["grants"][0]["handoff_at"]="2026-10-14T10:00:00+08:00"
    _file(contract,content)
    snap=us.plan(cfg,now_utc=_ts("2026-10-14T09:00:00+08:00"))
    assert snap["assignments"]==[]
    assert snap["gpu_time_window"]["status"]=="BLOCKED_CONTRACT_DRIFT"
    assert snap["gpu_time_window"]["handoffs"][0]["classification"]=="CONTRACT_DRIFT_UNVERIFIED"
    assert snap["gpu_time_window"]["handoffs"][0]["live_release_confirmed"] is False


def test_symlink_contract_is_not_consumed(tmp_path):
    path,doc=_config(tmp_path)
    supervisor=_bound(tmp_path,doc)
    actual=Path(doc["gpu_handoff_contract"]["path"])
    link=tmp_path/"shortcut.json"
    link.symlink_to(actual)
    doc["gpu_handoff_contract"]["path"]=str(link)
    _file(supervisor,doc)
    with pytest.raises(us.SupervisorError,match="non-symlink"):
        us.config(supervisor)


def test_separate_real_host_release_proof_is_still_required(tmp_path,monkeypatch):
    path,doc=_config(tmp_path)
    cfg=us.config(_bound(tmp_path,doc))
    _ready(monkeypatch)
    snap=us.plan(cfg,now_utc=_ts("2026-10-14T11:01:00+08:00"))
    assert snap["gpu_time_window"]["handoffs"][0]["classification"]=="RELEASE_UNVERIFIED"
    assert snap["gpu_time_window"]["handoffs"][0]["live_release_confirmed"] is False
    assert "host UUID/owner/Ray/Holder live readback" in snap["gpu_time_window"]["reminder"]
