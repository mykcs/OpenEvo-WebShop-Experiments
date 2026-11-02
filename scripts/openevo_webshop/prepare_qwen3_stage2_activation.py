#!/usr/bin/env python3
"""Build the immutable Qwen3-1.7B Stage2 activation after launch-readiness.

This tool consumes only frozen receipts and qualification artifacts. It makes
no model/API/WebShop/final-panel calls and writes no Stage2 execution state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ARM = "qwen3-1p7b"
EXPECTED_MODEL = "Qwen/Qwen3-1.7B"
EXPECTED_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
EXPECTED_CONTEXT = 40960
EXPECTED_SAMPLING = {
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "repetition_penalty": 1.0,
    "max_new_tokens": 512,
    "max_steps": 15,
}
REQUIRED_WORKLOADS = {
    "stage2_rollout_qwen3-1p7b",
    "stage2_text_memory_qwen3-1p7b",
    "stage2_skill_bundle_qwen3-1p7b",
    "stage2_agent_system_qwen3-1p7b",
    "stage2_sd_lora_qwen3-1p7b",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def atomic_create(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    try:
        with path.open("x") as f:
            f.write(rendered)
    except FileExistsError:
        if path.read_text() != rendered:
            raise RuntimeError(f"immutable activation already exists with different identity: {path}")


def assert_clean_exact(repo: Path, expected_sha: str) -> None:
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()
    if head != expected_sha or dirty:
        raise RuntimeError(f"execution source drift expected={expected_sha} observed={head} dirty={bool(dirty)}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--execution-git-sha", required=True)
    p.add_argument("--carrier-root", type=Path, required=True)
    p.add_argument("--round0-smoke", type=Path, required=True)
    p.add_argument("--sd-vram", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    repo = a.repo.resolve()
    root = a.root.resolve()
    carrier = a.carrier_root.resolve()
    smoke = a.round0_smoke.resolve()
    assert_clean_exact(repo, a.execution_git_sha)

    design_path = repo / "configs/experiment/designs/openevo-ceiling1-stage2-vnext-qwen3-1p7b-v1.json"
    protocol_path = repo / "configs/experiment/openevo-ceiling-1.0-stage2-final-v1.json"
    capacity_path = repo / "configs/experiment/designs/openevo-ceiling1-stage2-vnext-qwen3-1p7b-capacity-v1.json"
    seed_path = repo / "configs/experiment/designs/openevo-ceiling-1.0-harness201-step-seed-amendment-v1.json"
    lease_path = repo / "configs/experiment/openevo-ceiling-1.0-gpu-lease-qwen3-stage2-v1.json"
    for path in (design_path, protocol_path, capacity_path, seed_path, lease_path):
        if not path.is_file():
            raise RuntimeError(f"missing frozen input {path}")

    design = load(design_path)
    if design.get("status") != "FROZEN_PREREG_NEEDS_LAUNCH_READINESS" or design.get("arm") != ARM:
        raise RuntimeError("Qwen3 Stage2 design identity drift")
    if design.get("activation_authority") is not False:
        raise RuntimeError("design itself must not carry activation authority")
    model = design.get("model") or {}
    if model != {
        "id": EXPECTED_MODEL,
        "revision": EXPECTED_REVISION,
        "path": "/workspace/models/qwen3-1.7b",
        "context_window": EXPECTED_CONTEXT,
        "chat_template_enable_thinking": False,
    }:
        raise RuntimeError("Qwen3 model identity drift")
    frozen = design.get("frozen_protocol") or {}
    if int(frozen.get("formal_rollouts", -1)) != 20480 or int(frozen.get("rounds", -1)) != 160 or int(frozen.get("rollouts_per_round", -1)) != 128:
        raise RuntimeError("Qwen3 Stage2 budget drift")
    if frozen.get("sampling") != EXPECTED_SAMPLING or frozen.get("step_seed_policy") != "injective-v1":
        raise RuntimeError("Qwen3 Stage2 sampling/seed drift")
    if int(frozen.get("external_teacher_calls", -1)) != 0 or int(frozen.get("final_panel_access_before_own_stage2_seal", -1)) != 0:
        raise RuntimeError("Qwen3 forbidden budget drift")

    ready_path = root / "PRE_STAGE2_READY_qwen3-1p7b.json"
    ready = load(ready_path)
    if ready.get("status") != "PASS" or ready.get("arm") != ARM or ready.get("stage2_authorized") is not False:
        raise RuntimeError("pre-Stage2 receipt not PASS/closed")
    if (ready.get("model") or {}).get("chat_template_enable_thinking") is not False or int((ready.get("model") or {}).get("context_window", -1)) != EXPECTED_CONTEXT:
        raise RuntimeError("pre-Stage2 Qwen3 readback drift")

    compat_path = root / "stage2-readiness/qwen3-1p7b/STAGE1_PRESTAGE2_COMPAT.json"
    compat = load(compat_path)
    if compat.get("status") != "PASS" or compat.get("arm") != ARM or compat.get("source_ready_sha256") != sha(ready_path):
        raise RuntimeError("Stage2 entry compatibility receipt drift")
    if (compat.get("opsd") or {}).get("adapter_sha256") != (ready.get("initial_stage2_adapter") or {}).get("sha256"):
        raise RuntimeError("Stage2 compatibility OPSD adapter drift")

    base_path = root / "stage2-base/qwen3-1p7b/verification-receipt.json"
    base = load(base_path)
    if base.get("status") != "PASS" or base.get("arm") != ARM:
        raise RuntimeError("merged Stage2 base not PASS")
    if base.get("opsd_adapter_sha256") != (ready.get("initial_stage2_adapter") or {}).get("sha256"):
        raise RuntimeError("merged base OPSD identity drift")
    if int(base.get("formal_task_consumption", -1)) != 0 or int(base.get("final_panel_consumption", -1)) != 0:
        raise RuntimeError("merged base consumed forbidden budget")

    preflight_path = root / "stage2-preflight/qwen3-1p7b/runtime-preflight.json"
    preflight = load(preflight_path)
    if preflight.get("status") != "PASS" or preflight.get("harness_profile") != "201.1":
        raise RuntimeError("Qwen3 initial runtime preflight not PASS")
    if preflight.get("chat_template_enable_thinking") is not False or int(preflight.get("context_window", -1)) != EXPECTED_CONTEXT:
        raise RuntimeError("Qwen3 initial runtime thinking/context drift")
    if int(preflight.get("formal_webshop_calls", -1)) != 0 or int(preflight.get("final_panel_consumption", -1)) != 0:
        raise RuntimeError("initial runtime preflight consumed forbidden budget")

    carrier_receipt = carrier / "finalize-receipt.json"
    cr = load(carrier_receipt)
    if cr.get("schema") != "openevo.ceiling1-harness2.0.1-carrier-finalize-receipt.v2" or cr.get("status") != "PASS":
        raise RuntimeError("Harness201 carrier seed not PASS")
    for key in ("model_calls", "external_teacher_calls", "formal_task_consumption", "final_panel_access"):
        if int(cr.get(key, -1)) != 0:
            raise RuntimeError(f"Harness201 carrier seed consumed forbidden budget: {key}")

    smoke_summary_path = smoke / "summary.json"
    smoke_summary = load(smoke_summary_path)
    if smoke_summary.get("status") != "PASS" or smoke_summary.get("qualification_only") is not True:
        raise RuntimeError("round0 smoke not qualification PASS")
    if int(smoke_summary.get("attempt_count", -1)) != 128 or int(smoke_summary.get("task_identity_count", -1)) != 16 or int(smoke_summary.get("engineering_invalid_count", -1)) != 0:
        raise RuntimeError("round0 smoke denominator/engineering gate failed")
    if smoke_summary.get("harness_profile") != "201.1" or smoke_summary.get("step_seed_policy") != "injective-v1" or smoke_summary.get("sampling") != EXPECTED_SAMPLING:
        raise RuntimeError("round0 smoke protocol drift")
    rows = read_jsonl(smoke / "attempts.jsonl")
    if len(rows) != 128:
        raise RuntimeError("round0 smoke row count drift")
    if any(row.get("experiment_git_sha") != a.execution_git_sha for row in rows):
        raise RuntimeError("round0 smoke execution SHA drift")
    if any(row.get("harness_profile") != "201.1" or row.get("step_seed_policy") != "injective-v1" for row in rows):
        raise RuntimeError("round0 row Harness/seed drift")
    if any(row.get("chat_template_enable_thinking") is not False for row in rows):
        raise RuntimeError("round0 Qwen3 thinking drift")
    if any(any(step.get("runtime_harness") != "201.1" for step in (row.get("episode") or {}).get("trace", [])) for row in rows):
        raise RuntimeError("round0 runtime harness drift")

    vram = load(a.sd_vram.resolve())
    if vram.get("status") != "pass" or vram.get("workload") != "stage2_sd_lora_qwen3-1p7b":
        raise RuntimeError("Qwen3 SD-LoRA VRAM qualification mismatch")
    if int(vram.get("safety_headroom_mib", 0)) < 4096 or int(vram.get("peak_memory_mib", 0)) <= 0:
        raise RuntimeError("Qwen3 SD-LoRA VRAM qualification insufficient")

    capacity = load(capacity_path)
    core = Path((capacity.get("openevo_core") or {}).get("checkout_path", ""))
    core_sha = (capacity.get("openevo_core") or {}).get("execution_sha")
    if not core.is_dir() or not core_sha:
        raise RuntimeError("Qwen3 capacity Core binding missing")
    assert_clean_exact(core, core_sha)
    if int(capacity.get("effective_rank_limit", -1)) != 4096 or int(capacity.get("rank", -1)) != 8 or int(capacity.get("replay_capacity", -1)) != 64:
        raise RuntimeError("Qwen3 capacity algorithm drift")

    lease = load(lease_path)
    workloads = set((lease.get("workloads") or {}).keys())
    if not REQUIRED_WORKLOADS <= workloads:
        raise RuntimeError("Qwen3 lease workloads incomplete")
    rollout = lease["workloads"]["stage2_rollout_qwen3-1p7b"]
    if rollout.get("candidate_gpus") != [4, 5, 6, 7] or int(rollout.get("target_gpus", -1)) != 4 or int(rollout.get("min_gpus", -1)) != 4:
        raise RuntimeError("Qwen3 formal rollout topology drift")

    protocol_sha = sha(protocol_path)
    tm_path = root / "TEXT_MEMORY_COMPLETE_qwen3-1p7b.json"
    sk_path = root / "SKILL_BUNDLE_COMPLETE_qwen3-1p7b.json"
    ag_path = root / "AGENT_SYSTEM_COMPLETE_qwen3-1p7b.json"
    tm, sk, ag = load(tm_path), load(sk_path), load(ag_path)
    for name, marker in (("text", tm), ("skill", sk), ("agent", ag)):
        if marker.get("status") != "pass" or marker.get("scientific_result") is not True:
            raise RuntimeError(f"raw carrier marker not PASS: {name}")

    activation = {
        "schema": "openevo.ceiling1-stage2-activation.v1",
        "status": "authorized",
        "stage2_config_sha256": protocol_sha,
        "authorized_text_memory_schema": tm.get("schema"),
        "authorized_text_memory_split_events": int(tm.get("split_events", 0)),
        "authorized_skill_bundle_schema": sk.get("schema"),
        "authorized_skill_bundle_split_events": int(sk.get("split_events", 0)),
        "authorized_agent_system_schema": ag.get("schema"),
        "authorized_agent_system_split_events": int(ag.get("split_events", 0)),
        "authorized_arms": [ARM],
        "per_arm": {
            ARM: {
                "authorized": True,
                "stage1_complete_path": str(compat_path),
                "stage1_complete_sha256": sha(compat_path),
                "text_memory_complete_path": str(tm_path),
                "text_memory_complete_sha256": sha(tm_path),
                "skill_bundle_complete_path": str(sk_path),
                "skill_bundle_complete_sha256": sha(sk_path),
                "agent_system_complete_path": str(ag_path),
                "agent_system_complete_sha256": sha(ag_path),
            }
        },
        "formal_stage2_rollouts_consumed_at_activation": 0,
        "final_panel_access_count_at_activation": 0,
        "stage2_per_arm_independent": True,
        "final_panel_locked_until_stage2_sealed": True,
        "authorization_basis": "User authorized Qwen3-1.7B Stage2 launch once frozen first-class launch-readiness passes; all listed gates passed without formal Stage2/final-panel/external-teacher consumption.",
        "activation_created_from_source_sha": a.execution_git_sha,
        "evidence": {
            "design": {"path": str(design_path), "sha256": sha(design_path)},
            "prestage2_ready": {"path": str(ready_path), "sha256": sha(ready_path)},
            "stage2_entry_compat": {"path": str(compat_path), "sha256": sha(compat_path)},
            "merged_base_verification": {"path": str(base_path), "sha256": sha(base_path)},
            "initial_runtime_preflight": {"path": str(preflight_path), "sha256": sha(preflight_path)},
            "harness201_carrier_seed": {"path": str(carrier_receipt), "sha256": sha(carrier_receipt)},
            "round0_qualification": {"path": str(smoke_summary_path), "sha256": sha(smoke_summary_path)},
            "sd_lora_vram_qualification": {"path": str(a.sd_vram.resolve()), "sha256": sha(a.sd_vram.resolve())},
            "capacity_policy": {"path": str(capacity_path), "sha256": sha(capacity_path)},
            "step_seed_amendment": {"path": str(seed_path), "sha256": sha(seed_path)},
            "lease_policy": {"path": str(lease_path), "sha256": sha(lease_path)},
        },
    }
    atomic_create(a.output.resolve(), activation)
    print(json.dumps(activation, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
