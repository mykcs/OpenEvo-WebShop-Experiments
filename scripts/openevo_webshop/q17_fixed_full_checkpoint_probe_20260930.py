#!/usr/bin/env python3
"""Run the frozen Q17 panel against an exported full-weight E5 checkpoint.

This uses the existing Q17 episode/harness implementation and loads the model
weights as the model itself. It never converts an FSDP checkpoint into a PEFT
adapter. The input manifest must bind the source FSDP checkpoint, a verified
full-weight export, the pinned base revision, runtime files, and Q17 panel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts.openevo_webshop import q17_fixed_checkpoint_probe as q17


def tree_sha256(root: Path) -> str:
    if not root.is_dir():
        raise ValueError(f"full-weight model path is not a directory: {root}")
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError("full-weight model directory is empty")
    for path in files:
        if path.is_symlink():
            raise ValueError("symlinks are not accepted in full-weight model payload")
        rel = path.relative_to(root).as_posix().encode("utf-8")
        file_digest = q17.sha256_file(path)
        digest.update(len(rel).to_bytes(8, "big"))
        digest.update(rel)
        digest.update(bytes.fromhex(file_digest))
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> str:
    actual = q17.sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA mismatch: {actual} != {expected}")
    return actual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--runtime-source-root", type=Path, required=True)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--source-fsd-checkpoint", type=Path, required=True)
    parser.add_argument("--full-model-export-receipt", type=Path, required=True)
    parser.add_argument("--export-tool", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True,
                        help="verified full-weight HF export, never a PEFT adapter")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instrumentation", choices=("off", "on"), default="on")
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != args.gpu_uuid:
        raise SystemExit("CUDA_VISIBLE_DEVICES mismatch")
    manifest_bytes = args.manifest.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha != args.manifest_sha256:
        raise RuntimeError("E5 milestone manifest SHA mismatch")
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != "openevo.post-advisor-e5-q17-full-checkpoint-milestone.v1":
        raise RuntimeError("unsupported E5 full-checkpoint milestone manifest")
    if manifest.get("final_panel_access_allowed") is not False:
        raise RuntimeError("E5 manifest must keep final-panel access forbidden")
    if manifest.get("formal_task_consumption_allowed") is not False:
        raise RuntimeError("E5 manifest must forbid formal-task consumption")
    panel_manifest = json.loads(args.panel_manifest.read_text())
    panel = q17.validate_full_panel_manifest(args.panel_manifest, panel_manifest)
    if manifest.get("panel_manifest_sha256") != panel["file_sha256"]:
        raise RuntimeError("E5 milestone manifest panel-manifest SHA mismatch")
    if manifest.get("panel_sha256") != panel["panel_sha256"]:
        raise RuntimeError("E5 milestone manifest panel identity mismatch")

    checkpoint = manifest["checkpoints"][args.checkpoint_id]
    if checkpoint.get("update_step") not in (0, 40, 80, 120, 160):
        raise RuntimeError("checkpoint is outside the frozen E5 milestone schedule")
    if checkpoint.get("full_model_path") != str(args.model.resolve()):
        raise RuntimeError("full-weight model path differs from the checkpoint manifest")
    if checkpoint.get("source_fsd_checkpoint_path") != str(args.source_fsd_checkpoint.resolve()):
        raise RuntimeError("source FSDP checkpoint path differs from the checkpoint manifest")
    source_checkpoint_sha = tree_sha256(args.source_fsd_checkpoint)
    if checkpoint.get("source_fsd_checkpoint_sha256") != source_checkpoint_sha:
        raise RuntimeError("source FSDP checkpoint tree SHA mismatch")
    model_tree_sha = tree_sha256(args.model)
    if checkpoint.get("full_model_tree_sha256") != model_tree_sha:
        raise RuntimeError("full-weight model tree SHA mismatch")
    if checkpoint.get("full_model_export_receipt_path") != str(args.full_model_export_receipt.resolve()):
        raise RuntimeError("full-weight export receipt path differs from the checkpoint manifest")
    export_receipt_sha = require_sha(
        args.full_model_export_receipt,
        checkpoint["full_model_export_receipt_sha256"],
        "full-weight export receipt",
    )
    export_receipt = json.loads(args.full_model_export_receipt.read_text())
    if export_receipt.get("status") != "PASS":
        raise RuntimeError("full-weight export receipt is not PASS")
    if export_receipt.get("source_fsd_checkpoint_sha256") != source_checkpoint_sha:
        raise RuntimeError("full-weight export receipt source checkpoint drift")
    if export_receipt.get("full_model_tree_sha256") != model_tree_sha:
        raise RuntimeError("full-weight export receipt output tree drift")
    converter_sha = export_receipt.get("converter_sha256")
    if export_receipt.get("converter_path") != str(args.export_tool.resolve()):
        raise RuntimeError("full-weight export tool path differs from the receipt")
    if not isinstance(converter_sha, str) or q17.sha256_file(args.export_tool) != converter_sha:
        raise RuntimeError("full-weight export tool SHA mismatch")
    if args.revision != manifest["model"]["base_revision"]:
        raise RuntimeError("base-model revision mismatch")
    for rel, expected in manifest["runtime_files"].items():
        require_sha(args.runtime_source_root / rel, expected, f"runtime file {rel}")
    common = manifest["common_carriers"]
    require_sha(args.memory, common["memory_sha256"], "memory")
    require_sha(args.skill, common["skill_sha256"], "skill")
    require_sha(args.agent, common["agent_system_sha256"], "agent system")

    modules = q17.load_science_modules(args.runtime_source_root)
    runtime = modules["Runtime"](
        str(args.model.resolve()), args.revision, sd_adapter=None,
        memory=args.memory.read_text(), skill_bundle=args.skill.read_text(),
        agent_system=args.agent.read_text(),
    )
    observer = None
    if args.instrumentation == "on":
        observer = q17.GenerateObserver(runtime.base._model)
        observer.install()
    rows: list[dict[str, Any]] = []
    started = time.time()
    try:
        for task in panel_manifest["tasks"]:
            raw_env = modules["WebshopEnv"](
                modules["WebshopEnvConfig"](
                    seed_root=str(args.seed_root), num_products=1000,
                    seed=int(task["worker_seed"]),
                )
            )
            error = None
            try:
                episode = q17.run_episode(
                    modules,
                    modules["FixedEnvironmentSeed"](raw_env, int(task["worker_seed"])),
                    runtime, observer, task, int(task["paired_generation_seed"]),
                )
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
                episode = {"trace": [], "steps": 0, "score": 0.0, "won": False,
                           "invalid_terminated": False, "constraint_escape": False,
                           "environment_action_schema_violation": False,
                           "webshop_env_step_calls": 0}
            finally:
                raw_env.close()
            rows.append({
                "task_identity": task["task_identity"],
                "probe_ordinal": int(task["probe_ordinal"]),
                "source_session_index": int(task["source_session_index"]),
                "worker_seed": int(task["worker_seed"]),
                "paired_generation_seed": int(task["paired_generation_seed"]),
                "checkpoint_id": args.checkpoint_id,
                "update_step": int(checkpoint["update_step"]),
                "source_fsd_checkpoint_sha256": checkpoint["source_fsd_checkpoint_sha256"],
                "full_model_tree_sha256": checkpoint["full_model_tree_sha256"],
                "full_model_export_receipt_sha256": export_receipt_sha,
                "instrumentation": args.instrumentation,
                "engineering_invalid": bool(error),
                "error": error,
                "episode": episode,
            })
    finally:
        if observer:
            observer.close()
        runtime.close()

    invalid = any(row["engineering_invalid"] for row in rows)
    summary = {
        "schema": "openevo.post-advisor-e5-q17-full-checkpoint-result.v1",
        "status": "PASS" if len(rows) == q17.FROZEN_PANEL_SIZE and not invalid else "FAIL",
        "checkpoint_id": args.checkpoint_id,
        "update_step": int(checkpoint["update_step"]),
        "source_fsd_checkpoint_sha256": checkpoint["source_fsd_checkpoint_sha256"],
        "full_model_tree_sha256": checkpoint["full_model_tree_sha256"],
        "full_model_export_receipt_sha256": export_receipt_sha,
        "converter_sha256": converter_sha,
        "milestone_manifest_sha256": manifest_sha,
        "probe_script_sha256": q17.sha256_file(Path(__file__).resolve()),
        "task_panel_manifest_sha256": panel["file_sha256"],
        "task_panel_task_identity_sha256": panel["task_identity_sha256"],
        "panel_sha256": panel["panel_sha256"],
        "base_model_revision": args.revision,
        "runtime_execution_sha": manifest["runtime_execution_sha"],
        "prompt_contract_sha256": modules["PROMPT_CONTRACT_SHA256"],
        "instrumentation": args.instrumentation,
        "task_count": len(rows),
        "formal_task_consumption": 0,
        "external_teacher_calls": 0,
        "final_panel_access": 0,
        "model_load_mode": "full-weight model path; sd_adapter=None",
        "metrics": q17.summarize_rows(rows, args.instrumentation == "on"),
        "wall_time_seconds": time.time() - started,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    q17.atomic_json(args.output_dir / "summary.json", summary)
    with (args.output_dir / "attempts.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "status", "checkpoint_id", "update_step", "probe_script_sha256",
        "milestone_manifest_sha256", "task_panel_manifest_sha256", "panel_sha256",
        "formal_task_consumption", "final_panel_access",
    )}, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
