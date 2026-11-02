#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
from typing import Any

KINDS = {
    "text-memory": ("TEXT_MEMORY_COMPLETE", "memory.md", "memory_sha256"),
    "skill-bundle": ("SKILL_BUNDLE_COMPLETE", "SKILL.md", "skill_sha256"),
    "agent-system": ("AGENT_SYSTEM_COMPLETE", "agent_system.md", "agent_system_sha256"),
}

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)

def load_contract(repo: Path, contract_path: Path | None) -> tuple[dict[str, Any], Path]:
    path = contract_path or repo / "configs/experiment/stage1/202609030400/prestage2_shared.json"
    contract = load(path)
    if contract.get("schema") not in {"openevo.shared-prestage2.v1", "openevo.shared-prestage2.v2"} or contract.get("status") != "FROZEN_METHOD":
        raise RuntimeError("shared pre-Stage2 contract identity mismatch")
    if contract.get("stage2_auto_launch") is not False:
        raise RuntimeError("Stage2 auto-launch must remain disabled")
    return contract, path

def arm_spec(contract: dict[str, Any], arm: str) -> dict[str, Any]:
    if arm not in contract.get("arms", {}):
        raise RuntimeError(f"unknown arm {arm}")
    return contract["arms"][arm]

def verify_opsd_recipe(repo: Path, contract: dict[str, Any]) -> Path:
    recipe = repo / contract["common"]["opsd_recipe"]
    if sha(recipe) != contract["common"]["opsd_recipe_sha256"]:
        raise RuntimeError("OPSD recipe SHA drift")
    return recipe

def metric_indices(path: Path) -> list[int]:
    rows = read_jsonl(path)
    return [int(r["index"]) for r in rows]

def require_exact_indices(path: Path, start: int, stop: int) -> int:
    idx = metric_indices(path)
    expected = list(range(start, stop))
    if idx != expected:
        raise RuntimeError(f"metrics index mismatch {path}: got {idx[:3]}..{idx[-3:] if idx else []}, expected {start}..{stop-1}")
    return len(idx)

def seal_opsd(repo: Path, root: Path, arm: str, contract: dict[str, Any], contract_path: Path) -> dict[str, Any]:
    verify_opsd_recipe(repo, contract)
    d = root / "opsd" / arm
    manifest = load(d / "action-steps.manifest.json")
    if manifest.get("status") != "PASS" or manifest.get("arm") != arm:
        raise RuntimeError("OPSD dataset manifest mismatch")
    n = int(manifest["action_step_count"])
    dataset_sha = manifest["dataset_sha256"]
    first = d / "formal-v1"
    resumed = d / "formal-v2-resume-1024"
    segments: list[dict[str, Any]] = []
    discarded = 0
    if (resumed / "summary.json").is_file():
        checkpoint = first / "checkpoint-001024" / "receipt.json"
        c = load(checkpoint)
        if int(c.get("next_index", -1)) != 1024 or c.get("dataset_sha256") != dataset_sha:
            raise RuntimeError("OPSD checkpoint identity mismatch")
        prefix_idx = metric_indices(first / "train-metrics.jsonl")
        if prefix_idx[:1024] != list(range(1024)):
            raise RuntimeError("OPSD pre-checkpoint metrics prefix mismatch")
        discarded = max(0, len(prefix_idx) - 1024)
        require_exact_indices(resumed / "train-metrics.jsonl", 1024, n)
        s = load(resumed / "summary.json")
        if not (s.get("status") == "PASS" and s.get("scientific_result") is True and s.get("qualification") is False):
            raise RuntimeError("OPSD resumed summary not scientific PASS")
        if s.get("dataset_sha256") != dataset_sha or int(s.get("records_total", -1)) != n or int(s.get("records_consumed", -1)) != n or int(s.get("start_index", -1)) != 1024:
            raise RuntimeError("OPSD resumed accounting mismatch")
        final_dir = resumed / "final-adapter"
        segments = [
            {"start": 0, "stop_exclusive": 1024, "source": str(first), "checkpoint_receipt": str(checkpoint), "checkpoint_receipt_sha256": sha(checkpoint)},
            {"start": 1024, "stop_exclusive": n, "source": str(resumed), "summary": str(resumed / "summary.json"), "summary_sha256": sha(resumed / "summary.json")},
        ]
        summary = s
    else:
        s = load(first / "summary.json")
        if not (s.get("status") == "PASS" and s.get("scientific_result") is True and s.get("qualification") is False):
            raise RuntimeError("OPSD formal summary not scientific PASS")
        if s.get("dataset_sha256") != dataset_sha or int(s.get("records_total", -1)) != n or int(s.get("records_consumed", -1)) != n or int(s.get("start_index", -1)) != 0:
            raise RuntimeError("OPSD formal accounting mismatch")
        require_exact_indices(first / "train-metrics.jsonl", 0, n)
        final_dir = first / "final-adapter"
        segments = [{"start": 0, "stop_exclusive": n, "source": str(first), "summary": str(first / "summary.json"), "summary_sha256": sha(first / "summary.json")}]
        summary = s
    adapter_file = final_dir / "adapter_model.safetensors"
    if not adapter_file.is_file() or sha(adapter_file) != summary.get("final_adapter_sha256"):
        raise RuntimeError("OPSD final adapter mismatch")
    receipt = {
        "schema": "openevo.shared-prestage2.opsd-complete.v1", "status": "PASS", "arm": arm,
        "records_total": n, "optimizer_steps_total": n, "dataset_sha256": dataset_sha,
        "recipe_sha256": summary["recipe_sha256"], "adapter_sha256": summary["final_adapter_sha256"],
        "final_adapter_path": str(final_dir.resolve()), "authoritative_segments": segments,
        "discarded_unsealed_metrics_after_checkpoint": discarded,
        "new_webshop_calls": 0, "new_student_generations": 0, "external_teacher_calls": 0,
        "shared_contract_sha256": sha(contract_path), "completed_at_epoch": time.time(),
    }
    out = root / f"OPSD_COMPLETE_{arm}.json"
    atomic_json(out, receipt)
    return receipt

def seal_carrier(root: Path, arm: str, kind: str, carrier_root: Path, contract_path: Path) -> dict[str, Any]:
    prefix, content_name, sha_key = KINDS[kind]
    mm = load(root / f"MINIMAX_COMPLETE_{arm}.json")
    r = load(carrier_root / "coverage-receipt.json")
    if not (r.get("status") == "pass" and r.get("scientific_result") is True and r.get("qualification") is False):
        raise RuntimeError(f"{kind} is not scientific PASS")
    if r.get("analysis_sha256") != mm.get("analysis_corpus_sha256"):
        raise RuntimeError(f"{kind} analysis corpus mismatch")
    if int(r.get("consumed_rows_total", -1)) != 1440 or int(r.get("unique_source_ids", -1)) != 1440:
        raise RuntimeError(f"{kind} coverage mismatch")
    for key in ("omitted_rows", "duplicates", "unexpected_rows"):
        if r.get(key): raise RuntimeError(f"{kind} has {key}")
    if kind == "text-memory":
        if r.get("adaptive_tree") is not False:
            raise RuntimeError("Text Memory adaptive tree forbidden")
        if r.get("method") != "local_qwen_hierarchical_reflector_a2_depth1_split":
            raise RuntimeError("Text Memory A2 method mismatch")
        if int(r.get("max_split_depth", -1)) != 1 or r.get("recursive_split_allowed") is not False:
            raise RuntimeError("Text Memory A2 split-depth contract mismatch")
        if r.get("split_policy") != "only after full-20 primary and same-20 repair both saturate: fixed first10/last10, then two-child parent merge; no recursive split":
            raise RuntimeError("Text Memory A2 split policy mismatch")
    elif kind == "skill-bundle":
        if r.get("schema") == "openevo.ceiling1-skill-bundle-stage1.capacity-v1":
            if r.get("method") != "local_qwen_hierarchical_skill_bundle_bounded_capacity_v1":
                raise RuntimeError("Skill capacity method mismatch")
            if r.get("split_policy") != "max 20 inputs; capacity-only deterministic frozen-order midpoint bisection; max depth 5; no reward/outcome selection":
                raise RuntimeError("Skill capacity split policy mismatch")
            if int(r.get("max_split_depth", -1)) != 5 or r.get("outcome_based_split") is not False:
                raise RuntimeError("Skill capacity bound/outcome contract mismatch")
        elif r.get("split_policy") != "disabled" or int(r.get("fixed_topology_max_inputs", -1)) != 20:
            raise RuntimeError("Skill fixed20 contract mismatch")
    else:
        if r.get("method") != "local_qwen_hierarchical_agent_system_adaptive_split_v2":
            raise RuntimeError("Agent System capacity-recovery method mismatch")
        if r.get("split_policy") != "max 20 task/aggregate nodes; deterministic frozen-order midpoint bisection on input/output capacity only; no outcome selection":
            raise RuntimeError("Agent System capacity split policy mismatch")
        if int(r.get("max_split_depth", -1)) != 5 or int(r.get("task_group_count", -1)) != 180:
            raise RuntimeError("Agent System bounded recovery contract mismatch")
        tm = load(root / f"TEXT_MEMORY_COMPLETE_{arm}.json")
        sk = load(root / f"SKILL_BUNDLE_COMPLETE_{arm}.json")
        if r.get("memory_sha256") != tm.get("memory_sha256") or r.get("skill_sha256") != sk.get("skill_sha256"):
            raise RuntimeError("Agent upstream carrier mismatch")
    content = carrier_root / content_name
    if not content.is_file() or sha(content) != r.get(sha_key):
        raise RuntimeError(f"{kind} content hash mismatch")
    marker = root / f"{prefix}_{arm}.json"
    x = {**r, "arm": arm, "host_output_root": str(carrier_root.resolve()), "shared_contract_sha256": sha(contract_path), "sealed_at_epoch": time.time()}
    if kind == "text-memory": x["host_memory_path"] = str(content.resolve())
    elif kind == "skill-bundle": x["host_skill_path"] = str(content.resolve())
    else: x["host_agent_system_path"] = str(content.resolve())
    atomic_json(marker, x)
    return x

def seal_ready(root: Path, arm: str, spec: dict[str, Any], contract_path: Path) -> dict[str, Any]:
    mm = load(root / f"MINIMAX_COMPLETE_{arm}.json")
    op = load(root / f"OPSD_COMPLETE_{arm}.json")
    tm = load(root / f"TEXT_MEMORY_COMPLETE_{arm}.json")
    sk = load(root / f"SKILL_BUNDLE_COMPLETE_{arm}.json")
    ag = load(root / f"AGENT_SYSTEM_COMPLETE_{arm}.json")
    rr_path = root / "stage1-carriers" / arm / "runtime-readback.json"
    rr = load(rr_path)
    if mm.get("status") != "PASS" or int(mm.get("parse_ok", -1)) != 1440 or int(mm.get("parse_failures", -1)) != 0:
        raise RuntimeError("MiniMax not closed")
    if op.get("status") != "PASS": raise RuntimeError("OPSD not closed")
    for x in (tm, sk, ag):
        if x.get("status") != "pass" or x.get("scientific_result") is not True:
            raise RuntimeError("carrier not closed")
    if rr.get("status") != "PASS" or "bootstrap" not in set(rr.get("active_adapters") or []):
        raise RuntimeError("runtime readback not PASS")
    for key in ("text_memory", "skill_bundle", "agent_system"):
        item = (rr.get("carrier_readback") or {}).get(key) or {}
        if item.get("requested") is not True or item.get("present_in_decoded_input") is not True:
            raise RuntimeError(f"runtime readback missing {key}")
    out = {
        "schema": "openevo.shared-prestage2.ready.v1", "status": "PASS", "arm": arm,
        "freeze_id": "202609030400", "model": spec,
        "minimax_complete_sha256": sha(root / f"MINIMAX_COMPLETE_{arm}.json"),
        "opsd_complete_sha256": sha(root / f"OPSD_COMPLETE_{arm}.json"),
        "text_memory": {"path": tm["host_memory_path"], "sha256": tm["memory_sha256"]},
        "skill_bundle": {"path": sk["host_skill_path"], "sha256": sk["skill_sha256"]},
        "agent_system": {"path": ag["host_agent_system_path"], "sha256": ag["agent_system_sha256"]},
        "runtime_readback_sha256": sha(rr_path),
        "initial_stage2_adapter": {"path": op["final_adapter_path"], "sha256": op["adapter_sha256"]},
        "shared_contract_sha256": sha(contract_path),
        "stage2_authorized": False,
        "stage2_requires_separate_activation": True,
        "ready_at_epoch": time.time(),
    }
    atomic_json(root / f"PRE_STAGE2_READY_{arm}.json", out)
    return out

def status(root: Path, arm: str) -> dict[str, Any]:
    names = [f"MINIMAX_COMPLETE_{arm}.json", f"OPSD_COMPLETE_{arm}.json", f"TEXT_MEMORY_COMPLETE_{arm}.json", f"SKILL_BUNDLE_COMPLETE_{arm}.json", f"AGENT_SYSTEM_COMPLETE_{arm}.json", f"PRE_STAGE2_READY_{arm}.json"]
    return {"arm": arm, "files": {n: (root / n).is_file() for n in names}, "runtime_readback": (root / "stage1-carriers" / arm / "runtime-readback.json").is_file()}

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--contract", type=Path)
    sp = p.add_subparsers(dest="cmd", required=True)
    for name in ("seal-opsd", "seal-ready", "status"):
        q = sp.add_parser(name); q.add_argument("--arm", required=True)
    q = sp.add_parser("seal-carrier"); q.add_argument("--arm", required=True); q.add_argument("--kind", choices=tuple(KINDS), required=True); q.add_argument("--carrier-root", type=Path, required=True)
    a = p.parse_args(); repo = a.repo.resolve(); root = a.root.resolve(); contract, contract_path = load_contract(repo, a.contract)
    spec = arm_spec(contract, a.arm)
    if a.cmd == "seal-opsd": result = seal_opsd(repo, root, a.arm, contract, contract_path)
    elif a.cmd == "seal-carrier": result = seal_carrier(root, a.arm, a.kind, a.carrier_root.resolve(), contract_path)
    elif a.cmd == "seal-ready": result = seal_ready(root, a.arm, spec, contract_path)
    else: result = status(root, a.arm)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)); return 0

if __name__ == "__main__":
    raise SystemExit(main())
