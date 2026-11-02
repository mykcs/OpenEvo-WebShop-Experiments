#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any

SCHEMA = "openevo.ceiling1-harness201-3b-successor-capacity-policy.v1"
ALERT_5 = "警报：已经连续 5 轮没改参数，我正在检查是不是又出现了类似以前 gate=8 的假门槛问题。"
HOLD_PREFIXES = (
    "HARNESS201_STAGE2_READINESS_HOLD",
    "READINESS_CIRCUIT_BREAKER_HOLD",
    "STAGED_INTERLOCK_HOLD",
    "PREMATURE_LINEAGE_NO_RESUME",
)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_capacity_policy(path: Path | None, arm: str) -> dict[str, Any] | None:
    if path is None:
        return None
    x = json.loads(path.read_text())
    if x.get("schema") != SCHEMA or x.get("status") != "PREREGISTERED_NO_ACTIVATION_AUTHORITY":
        raise RuntimeError("invalid Harness201 successor capacity policy")
    if x.get("activation_authority") is not False:
        raise RuntimeError("capacity policy must not authorize Stage2")
    if x.get("arm") != arm or arm not in {"3b", "qwen3-1p7b"}:
        raise RuntimeError("Harness201 successor capacity policy arm mismatch")
    if int(x.get("rank", -1)) != 8 or int(x.get("replay_capacity", -1)) != 64:
        raise RuntimeError("successor capacity rank/replay contract drift")
    if x.get("independent_component_stop") is not None:
        raise RuntimeError("successor capacity must not have an independent component stop")
    if int(x.get("effective_rank_limit", -1)) != 4096:
        raise RuntimeError("successor effective-rank limit drift")
    if (x.get("compression") or {}).get("active") is not False:
        raise RuntimeError("compression is not active in Harness201 successor v1")
    core = x.get("openevo_core") or {}
    checkout = Path(str(core.get("checkout_path", "")))
    if not checkout.is_dir() or len(str(core.get("execution_sha", ""))) != 40:
        raise RuntimeError("successor capacity core source is not pinned")
    iso = x.get("isolated_trainer_source") or {}
    if iso.get("python_isolated_mode_required") is not True or iso.get("legacy_installed_package_fallback_forbidden") is not True:
        raise RuntimeError("isolated trainer source contract drift")
    return x

def capacity_decision(component_count: int, policy: dict[str, Any] | None) -> dict[str, Any]:
    if component_count < 0:
        raise ValueError("component_count must be nonnegative")
    if policy is None:
        return {
            "mode": "legacy-upstream-provenance",
            "allow_update": component_count < 64,
            "component_count_before": component_count,
            "candidate_component_count": component_count + 1,
            "candidate_effective_rank": (component_count + 1) * 8,
            "effective_rank_limit": 512,
            "reason": None if component_count < 64 else "NATIVE_SD_LORA_CAPACITY_LIMIT",
        }
    rank = int(policy["rank"])
    limit = int(policy["effective_rank_limit"])
    candidate = (component_count + 1) * rank
    return {
        "mode": "harness201-effective-rank",
        "allow_update": candidate <= limit,
        "component_count_before": component_count,
        "candidate_component_count": component_count + 1,
        "candidate_effective_rank": candidate,
        "effective_rank_limit": limit,
        "reason": None if candidate <= limit else "NATIVE_SD_LORA_EFFECTIVE_RANK_LIMIT",
    }

def blocker_is_nonrecoverable(blocker: object) -> bool:
    text = str(blocker or "")
    return any(marker in text for marker in HOLD_PREFIXES)

def consecutive_noop_diagnostic(output_root: Path, through_round: int) -> dict[str, Any]:
    statuses: list[dict[str, Any]] = []
    streak = 0
    index = through_round
    while index >= 0:
        path = output_root / f"round-{index:03d}" / "sd-lora" / "terminal-receipt.json"
        if not path.is_file():
            break
        status = json.loads(path.read_text()).get("status")
        reason = json.loads(path.read_text()).get("reason")
        statuses.append({"round_index": index, "status": status, "reason": reason})
        if status == "NOOP" and reason in {"NOOP_NO_SUCCESS", "NOOP_NO_QUALIFIED_RECORDS"}:
            streak += 1
            index -= 1
            continue
        break
    statuses.reverse()
    hold = streak >= 10
    alert = streak >= 5
    return {
        "schema": "openevo.ceiling1-sd-lora-noop-window-diagnostic.v2",
        "status": "HOLD" if hold else "ALERT" if alert else "OBSERVE",
        "through_round": through_round,
        "consecutive_noop_no_success": streak,
        "statuses": statuses,
        "warning": ALERT_5 if alert else None,
        "requires_manual_review": hold,
        "pause_before_next_round": hold,
        "modify_training": False,
        "modify_thresholds": False,
        "automatic_science_change": False,
    }


def validate_staged_interlock_round(round_root: Path, capacity_policy_sha256: str | None) -> dict[str, Any]:
    summary=json.loads((round_root/'summary.json').read_text())
    complete=json.loads((round_root/'ROUND_COMPLETE.json').read_text())
    after=json.loads((round_root/'state-after.json').read_text())
    if summary.get('attempt_count')!=128 or summary.get('engineering_invalid_count')!=0:
        raise RuntimeError('staged interlock rollout denominator/readback failed')
    if summary.get('harness_profile')!='2.0.1' or summary.get('step_seed_policy')!='injective-v1':
        raise RuntimeError('staged interlock successor harness/seed readback failed')
    attempts_path=round_root/'attempts.jsonl'
    rows=[json.loads(line) for line in attempts_path.read_text().splitlines() if line.strip()]
    if len(rows)!=128 or any(row.get('engineering_invalid') for row in rows):
        raise RuntimeError('staged interlock attempt identity/engineering gate failed')
    row_prompt_shas={str(row.get('prompt_contract_sha256') or '') for row in rows}
    if len(row_prompt_shas)!=1:
        raise RuntimeError('staged interlock prompt contract row identity drift')
    row_prompt_sha=next(iter(row_prompt_shas))
    if len(row_prompt_sha)!=64:
        raise RuntimeError('staged interlock prompt contract readback failed')
    summary_prompt_sha=str(summary.get('prompt_contract_sha256') or '')
    if summary_prompt_sha and summary_prompt_sha!=row_prompt_sha:
        raise RuntimeError('staged interlock summary/row prompt contract mismatch')
    prompt_sha=summary_prompt_sha or row_prompt_sha
    if any(row.get('harness_profile')!='2.0.1' or row.get('step_seed_policy')!='injective-v1' or row.get('prompt_contract_sha256')!=prompt_sha for row in rows):
        raise RuntimeError('staged interlock per-row harness/seed/prompt readback failed')
    invalid_episodes=sum(bool(row.get('terminated_on_invalid_action')) for row in rows)
    invalid_rate=invalid_episodes/128
    if invalid_rate > 0.159375:
        raise RuntimeError('staged interlock interface invalid-termination regression')
    if complete.get('status')!='PASS' or complete.get('state_id_after')!=after.get('state_id'):
        raise RuntimeError('staged interlock round/state chain failed')
    if int(after.get('final_panel_access_count',-1))!=0 or int(after.get('external_teacher_calls',-1))!=0:
        raise RuntimeError('staged interlock final-panel/teacher boundary failed')
    if capacity_policy_sha256 is not None and after.get('capacity_policy_sha256')!=capacity_policy_sha256:
        raise RuntimeError('staged interlock capacity policy readback failed')
    return {
        'schema':'openevo.ceiling1-harness201-staged-interlock-release.v1',
        'status':'PASS',
        'round_index':int(complete['round_index']),
        'state_id_after':after['state_id'],
        'summary_sha256':sha256(round_root/'summary.json'),
        'round_complete_sha256':sha256(round_root/'ROUND_COMPLETE.json'),
        'state_after_sha256':sha256(round_root/'state-after.json'),
        'capacity_policy_sha256':capacity_policy_sha256,
        'prompt_contract_sha256':prompt_sha,
        'terminated_invalid_episodes':invalid_episodes,
        'terminated_invalid_episode_rate':invalid_rate,
        'terminated_invalid_episode_rate_limit':0.159375,
        'automatic_regression_readback':True,
        'formal_task_consumption_delta':0,
        'final_panel_access_delta':0,
    }

def write_once_release(path: Path, payload: dict[str, Any]) -> None:
    rendered=json.dumps(payload,ensure_ascii=False,sort_keys=True,indent=2)+'\n'
    path.parent.mkdir(parents=True,exist_ok=True)
    try:
        with path.open('x') as f: f.write(rendered)
    except FileExistsError:
        if path.read_text()!=rendered:
            raise RuntimeError('staged interlock release receipt drift')

def validate_manual_hold_release(path: Path | None, state: dict[str, Any]) -> bool:
    if path is None or not path.is_file(): return False
    x=json.loads(path.read_text())
    if x.get('schema')!='openevo.ceiling1-harness201-manual-hold-release.v1' or x.get('status')!='PASS': return False
    if x.get('activation_authority') is not False or x.get('human_approved') is not True: return False
    if int(x.get('next_round',-1))!=int(state.get('next_round',-2)): return False
    if x.get('blocker')!=state.get('blocker'): return False
    return True

def load_step_seed_amendment(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    x=json.loads(path.read_text())
    if x.get('schema')!='openevo.ceiling1-harness201-step-seed-amendment.v1':
        raise RuntimeError('invalid Harness201 step-seed amendment schema')
    if x.get('status')!='QUALIFIED_FOR_SHADOW_ONLY':
        raise RuntimeError('Harness201 step-seed amendment is not qualified')
    p=x.get('candidate_policy') or {}
    if p.get('id')!='injective-step-seed-v1' or int(p.get('base',-1))!=148000000 or int(p.get('step_stride',-1))!=100000:
        raise RuntimeError('Harness201 injective step-seed policy drift')
    if p.get('global_injective_over_frozen_stage2_schedule') is not True or p.get('step0_seed_unchanged') is not True:
        raise RuntimeError('Harness201 step-seed injectivity contract drift')
    q=x.get('qualification_result') or {}
    if q.get('status')!='PASS' or q.get('formal_stage2_authority') is not False or not q.get('sha256'):
        raise RuntimeError('Harness201 step-seed qualification identity missing')
    return x

def load_harness201_carrier_seed(root: Path | None) -> dict[str, Any] | None:
    if root is None:
        return None
    receipt=root/'finalize-receipt.json'
    if not receipt.is_file(): raise RuntimeError('Harness201 carrier finalize receipt missing')
    x=json.loads(receipt.read_text())
    if x.get('schema')!='openevo.ceiling1-harness2.0.1-carrier-finalize-receipt.v2' or x.get('status')!='PASS':
        raise RuntimeError('Harness201 carrier finalize receipt invalid')
    if any(int(x.get(k,-1))!=0 for k in ('formal_task_consumption','final_panel_access','external_teacher_calls','model_calls')):
        raise RuntimeError('Harness201 carrier seed consumed forbidden budget')
    files={'memory':root/'memory.md','skill':root/'skill.md','agent':root/'agent.md'}
    for key,path in files.items():
        if not path.is_file() or sha256(path)!=(x.get('outputs') or {}).get(key,{}).get('sha256'):
            raise RuntimeError(f'Harness201 carrier seed hash mismatch: {key}')
    return {'root':str(root.resolve()),'receipt':str(receipt.resolve()),'receipt_sha256':sha256(receipt),
            'memory':str(files['memory'].resolve()),'skill':str(files['skill'].resolve()),'agent':str(files['agent'].resolve()),
            'memory_sha256':sha256(files['memory']),'skill_sha256':sha256(files['skill']),'agent_sha256':sha256(files['agent'])}
