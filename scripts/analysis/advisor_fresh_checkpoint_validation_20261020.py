#!/usr/bin/env python3
"""Fresh, outcome-blind checkpoint comparisons; reuse the original Q07 scorer.

No training, selection on outcomes, final-panel reads, or repeated task attempts.
The environment construction seed is zero; reset randomness remains paired.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis import advisor_q07_fixed_checkpoint_probe_20261001 as probe
from scripts.openevo_webshop import build_seed_source_faithful_panel as catalog

DOMAIN = "openevo.advisor.fresh-checkpoint-validation.20261020.v1"
INVENTORY_SHA = "f863aa68ee4f77e2c04f4c19210cb5aedbd4ec6d654e3282106b44f17d0c9e26"
Q10_PANEL_SHA = "af1546d4d2d628eaa4ed8e5b6647d44cd60c6c7e3c806f1f772b472b5c5a182d"
Q08_SEEN_SHA = "0780dbe68dcc1cafb013a8765f2ecb4b91bfd2a0d92ac3ac9fe4fde7453fccc3"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def select_panel(goals, inventory, q10_panel, excluded_source_sessions=()):
    if len(goals) != 6910:
        raise ValueError("exact construction-zero catalog required")
    excluded = set()
    for group in ("prior_development", "full_training", "final_allowlist"):
        for key in ("instruction_sha256", "goal_sha256"):
            excluded.update(inventory["exclusions"][group][key])
    excluded.update(row["instruction_sha256"] for row in q10_panel["panel"])
    excluded.update(row["goal_sha256"] for row in q10_panel["panel"])
    excluded_sessions = {row["source_session_index"] for row in q10_panel["panel"]}
    excluded_sessions.update(excluded_source_sessions)
    candidates = []
    for session in range(500, 6910):
        text = goals[session]["instruction_text"].strip()
        identity = hashlib.sha256(text.encode()).hexdigest()
        if identity not in excluded and session not in excluded_sessions:
            candidates.append((digest([DOMAIN, session, identity]), session, identity, text))
    rows, seen = [], set()
    for _, session, identity, text in sorted(candidates):
        if identity in seen:
            continue
        seen.add(identity)
        ordinal = len(rows)
        rows.append({"probe_ordinal": ordinal, "task_identity": f"fresh-{session:04d}-{identity[:12]}",
            "source_session_index": session, "worker_seed": 2026102000 + ordinal,
            "paired_generation_seed": 202610200000000 + ordinal * 10000000,
            "instruction": text, "instruction_sha256": identity, "goal_sha256": identity})
        if len(rows) == 64:
            break
    if len(rows) != 64:
        raise ValueError("insufficient genuinely disjoint task identities")
    return rows


def build_panel(args):
    if probe.sha256_file(args.inventory) != INVENTORY_SHA:
        raise ValueError("frozen complete exclusion inventory drift")
    if probe.sha256_file(args.q10_panel) != Q10_PANEL_SHA or probe.sha256_file(args.q08_seen) != Q08_SEEN_SHA:
        raise ValueError("reserved task source bytes drift")
    q10 = json.loads(args.q10_panel.read_text())
    if len(q10["panel"]) != 128 or q10["selection"]["construction_seed"] != 0:
        raise ValueError("Q10 reserved panel identity differs")
    webshop = args.seed_root / "agent_system/environments/env_package/webshop/webshop"
    for rel, expected in q10["selection"]["source_hashes"].items():
        if probe.sha256_file(webshop / rel) != expected:
            raise ValueError(f"catalog source/data drift: {rel}")
    catalog._install_dependency_stubs()
    engine, goal = catalog._setup_seed_imports(args.seed_root, webshop / "data")
    goals = catalog._build_worker_goals(0, engine, goal, webshop / "data", 1000)
    seen = json.loads(args.q08_seen.read_text())
    seen_rows = seen.get("tasks", seen.get("panel"))
    if not isinstance(seen_rows, list) or len(seen_rows) != 96:
        raise ValueError("Q08 exact seen panel required")
    tasks = select_panel(goals, json.loads(args.inventory.read_text()), q10,
        [row["source_session_index"] for row in seen_rows])
    value = {"schema": DOMAIN, "construction_seed": 0, "tasks": tasks,
        "panel_sha256": digest(tasks), "selection_outcomes_seen": False,
        "inventory_sha256": INVENTORY_SHA, "q10_reserved_panel_sha256": probe.sha256_file(args.q10_panel),
        "q08_seen_panel_sha256": Q08_SEEN_SHA,
        "source_hashes": q10["selection"]["source_hashes"],
        "environment_resets": 0, "model_calls": 0, "final_panel_access": 0}
    with args.output.open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": "CPU_PANEL_SELECTED", "tasks": 64, "sha256": probe.sha256_file(args.output)}))


def evaluate(args):
    manifest = json.loads(args.manifest.read_text())
    if manifest["final_panel_access_allowed"] is not False or manifest["optimizer_updates"] != 0:
        raise ValueError("fixed-checkpoint evaluation scope drift")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        raise ValueError("GPU assignment differs")
    if os.environ.get("OPENEVO_RUNTIME_IMAGE_ID") != manifest["image_id"]:
        raise ValueError("pinned runtime image differs")
    checkpoint = manifest["checkpoints"][args.checkpoint_id]
    runtime_root = Path(manifest["runtime_source_root"])
    probe.validate_runtime_files(manifest, runtime_root)
    for path, expected in manifest["read_files"].items():
        probe.validate_payload(Path(path), expected, "model/data/core identity")
    for path, expected in manifest["source_files"].items():
        probe.validate_payload(ROOT / path, expected, "execution source")
    adapter = Path(checkpoint["adapter_path"])
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        probe.validate_payload(adapter / name, checkpoint["files"][name], name)
    carriers = manifest["common_carriers"]
    for kind in ("memory", "skill", "agent"):
        probe.validate_payload(Path(carriers[kind]["path"]), carriers[kind]["sha256"], kind)
    tasks = manifest["tasks"]
    if len(tasks) != 64 or digest(tasks) != manifest["panel_sha256"]:
        raise ValueError("exact prospective task panel differs")
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {"manifest_sha256": probe.sha256_file(args.manifest), "checkpoint_id": args.checkpoint_id,
        "gpu_uuid": args.gpu_uuid, "source_files": manifest["source_files"]}
    # An existing start may include an interrupted model call: never silently replay.
    with (args.output / "START.json").open("x") as stream:
        json.dump(identity, stream, sort_keys=True)
    modules = probe.load_science_modules(runtime_root)
    runtime = modules["Runtime"](manifest["model"]["local_model_path"], manifest["model"]["revision"],
        sd_adapter=adapter, memory=Path(carriers["memory"]["path"]).read_text(),
        skill_bundle=Path(carriers["skill"]["path"]).read_text(),
        agent_system=Path(carriers["agent"]["path"]).read_text())
    rows = []
    try:
        for task in tasks:
            slot = args.output / f"task-{task['probe_ordinal']:03d}"
            slot.mkdir()
            probe.atomic_json(slot / "START.json", task)
            env = modules["WebshopEnv"](modules["WebshopEnvConfig"](
                seed_root=manifest["seed_root"], num_products=1000, seed=0))
            try:
                episode = probe.run_episode(modules, modules["FixedEnvironmentSeed"](env, task["worker_seed"]),
                    runtime, None, task, task["paired_generation_seed"])
                row = {**task, "checkpoint_id": args.checkpoint_id, "engineering_invalid": False, "episode": episode}
                probe.atomic_json(slot / "COMPLETE.json", row)
                rows.append(row)
            except BaseException as exc:
                probe.atomic_json(slot / "ENGINEERING_FAILURE.json", {"error": f"{type(exc).__name__}: {exc}"})
                raise
            finally:
                env.close()
    finally:
        runtime.close()
    result = {"status": "COMPLETE", **identity, "tasks": len(rows), "optimizer_updates": 0,
        "final_panel_access": 0, "metrics": probe.summarize_rows(rows, False), "rows": rows}
    probe.atomic_json(args.output / "COMPLETE.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    panel = commands.add_parser("panel")
    for name in ("seed-root", "inventory", "q10-panel", "q08-seen", "output"):
        panel.add_argument("--" + name, type=Path, required=True)
    run = commands.add_parser("evaluate")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--checkpoint-id", required=True)
    run.add_argument("--gpu-uuid", required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_panel(args) if args.command == "panel" else evaluate(args)


if __name__ == "__main__":
    main()
