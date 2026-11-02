#!/usr/bin/env python3
"""H1.37 adapter-arm proof (closeout requirement #1, MOST IMPORTANT).

The integrity verifier (verify_h1_37_integrity.py) proves the frozen adapter FILE is
still `a60e40d8...` and can recompute the protocol fingerprint. But the adapter is NOT
an input to the protocol fingerprint, so "adapter unchanged" does NOT by itself prove
that every SELF-EVOLUTION cell actually loaded the adapter while every BASE cell ran
with `--adapter none`.

This script closes that gap with INDEPENDENT post-hoc machine evidence already left on
disk, without touching the running/finished experiment. Each cell's W&B offline run
writes `wandb/wandb/offline-run-*/logs/debug.log`, whose protocol config line (the one
containing both `'phase': 'H1'` and `'adapter':`) records the exact `--adapter` value the
runner was launched with. We parse that runner-side recorded config for ALL 128 cells and
assert:

  * every BASE cell           -> config['adapter'] == 'none'
  * every SELF-EVOLUTION cell -> config['adapter'] == the frozen H1.36 adapter container path
  * config['arm']             -> matches the cell's arm label (no arm/adapter swap)
  * config['task_manifest_sha256'] -> the sealed manifest sha (ties adapter proof to the
                                      same sealed panel the integrity verifier fingerprinted)
  * config['cuda_visible_devices'] -> the cell's bound GPU UUID (gpu-before.json), tying the
                                      adapter proof to the exact cell the fingerprint covers

We then prove the SELF target directory is the frozen adapter by hashing the adapter dir
two ways:
  * historical digest: `find | LC_ALL=C sort | xargs sha256sum | sha256sum` -- identical to
    the launcher's pin pipeline; reproduces `a60e40d8...`; embeds absolute host paths.
  * canonical digest: sha256 over a sorted {relative_path: content_sha256} map -- pure
    content hash, path-independent, so the adapter can be moved to another machine and still
    be proven byte-identical.

Stdlib only; runs on the host against the run root without Docker or openevo imports.
"""

from __future__ import annotations

import argparse
import ast
import glob
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ARMS = ("base", "self-evolution")

# Frozen identities (must match the pinned launchers + seal).
EXPECTED_MANIFEST_SHA = "6e55fd5d06a0f2461af2c28643c0aa5d6a15e3048fe7cd182ec9b3b440a0ed07"
EXPECTED_ADAPTER_SHA = "a60e40d866b3ed6aa41873b6ec39c2844adf4b88e58799f3bc2d2d02da3f2489"
EXPECTED_SELF_ADAPTER_CONTAINER = (
    "/workspace/runs/phase-h1.36-strong-base-self-evolution-v1-20260818/artifacts/"
    "h1.36-self-evolution-fit-gpu0-v1/workers/sd-lora-b7e36404ca6320620e3f86e5225699e3/adapter"
)


def _protocol_config(cell_dir: Path) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """Return (config_dict, debug_log_path, error). Picks the protocol config line only."""
    logs = sorted(glob.glob(str(cell_dir / "wandb" / "wandb" / "offline-run-*" / "logs" / "debug.log")))
    if not logs:
        return None, None, "no debug.log"
    for log_path in logs:
        try:
            with open(log_path, encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    # The protocol config dump is the only line carrying both markers; an
                    # earlier short wandb-internal "config:" line exists and must be skipped.
                    if "'phase': 'H1'" in line and "'adapter':" in line and "config: " in line:
                        idx = line.find("config: ")
                        try:
                            return ast.literal_eval(line[idx + len("config: ") :].strip()), log_path, None
                        except (ValueError, SyntaxError) as exc:
                            return None, log_path, f"config parse error: {type(exc).__name__}"
        except OSError as exc:
            return None, log_path, f"read error: {type(exc).__name__}"
    return None, logs[0], "no protocol config line"


def _cell_gpu_uuid(cell_dir: Path) -> str | None:
    before = cell_dir / "gpu-before.json"
    if before.is_file():
        try:
            uuid = json.loads(before.read_text(encoding="utf-8")).get("expected_uuid")
            if isinstance(uuid, str) and uuid:
                return uuid
        except (ValueError, OSError):
            pass
    for lock in cell_dir.glob("GPU-*.lock"):
        name = lock.name[: -len(".lock")]
        return name[len("GPU-") :] if name.startswith("GPU-") else name
    return None


def _historical_adapter_sha(adapter_dir: Path) -> str:
    # Launcher-identical pipeline; embeds absolute host paths (path-dependent by design).
    cmd = f'find "{adapter_dir}" -type f | LC_ALL=C sort | xargs sha256sum | sha256sum'
    out = subprocess.run(["bash", "-c", cmd], check=True, capture_output=True, text=True).stdout
    return out.split()[0] if out.split() else ""


def _canonical_adapter_digest(adapter_dir: Path) -> tuple[str, dict[str, str]]:
    # Path-independent content digest: sha256 over sorted {relative_path: content_sha256}.
    entries: dict[str, str] = {}
    for path in sorted(adapter_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(adapter_dir).as_posix()
            entries[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest(), entries


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def prove(run_root: Path, manifest_path: Path, adapter_dir: Path, output: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: Any) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = [int(v) for v in manifest.get("evaluation_tasks", [])]

    eval_dir = run_root / "evaluation"
    arm_adapter_counts: dict[str, dict[str, int]] = {arm: {} for arm in ARMS}
    mismatches: list[Any] = []
    missing: list[Any] = []
    cross_field_errors: list[Any] = []

    for arm in ARMS:
        for task in tasks:
            cell_dir = eval_dir / f"{arm}-task{task}"
            config, log_path, err = _protocol_config(cell_dir)
            if config is None:
                missing.append({"cell": f"{arm}:{task}", "reason": err, "log": log_path})
                continue
            adapter_value = config.get("adapter")
            key = repr(adapter_value)
            arm_adapter_counts[arm][key] = arm_adapter_counts[arm].get(key, 0) + 1

            # 1) adapter value matches the arm contract.
            if arm == "base":
                if adapter_value != "none":
                    mismatches.append({"cell": f"base:{task}", "adapter": adapter_value, "expected": "none"})
            else:
                if adapter_value != EXPECTED_SELF_ADAPTER_CONTAINER:
                    mismatches.append({"cell": f"self-evolution:{task}", "adapter": adapter_value, "expected": EXPECTED_SELF_ADAPTER_CONTAINER})

            # 2) cross-field consistency ties this adapter proof to the sealed, fingerprinted cell.
            if config.get("arm") != arm:
                cross_field_errors.append({"cell": f"{arm}:{task}", "field": "arm", "value": config.get("arm")})
            if config.get("task_manifest_sha256") != EXPECTED_MANIFEST_SHA:
                cross_field_errors.append({"cell": f"{arm}:{task}", "field": "task_manifest_sha256", "value": config.get("task_manifest_sha256")})
            bound_uuid = _cell_gpu_uuid(cell_dir)
            if bound_uuid and config.get("cuda_visible_devices") != bound_uuid:
                cross_field_errors.append(
                    {"cell": f"{arm}:{task}", "field": "cuda_visible_devices", "config": config.get("cuda_visible_devices"), "bound": bound_uuid}
                )

    record("protocol_config_present_all_cells", not missing, {"missing": missing[:10], "missing_count": len(missing)})
    record(
        "base_cells_adapter_none",
        arm_adapter_counts["base"] == {"'none'": len(tasks)},
        {"base_adapter_values": arm_adapter_counts["base"], "expected_cells": len(tasks)},
    )
    record(
        "self_cells_adapter_frozen_path",
        arm_adapter_counts["self-evolution"] == {repr(EXPECTED_SELF_ADAPTER_CONTAINER): len(tasks)},
        {"self_adapter_values": arm_adapter_counts["self-evolution"], "expected_cells": len(tasks)},
    )
    record("no_arm_adapter_swap", not mismatches, {"mismatches": mismatches[:10]})
    record(
        "adapter_proof_tied_to_sealed_cell",
        not cross_field_errors,
        {"errors": cross_field_errors[:10],
         "proves": "config arm + manifest sha256 + cuda_visible_devices all match the cell, so the recorded adapter value belongs to the exact fingerprinted cell"},
    )

    # --- Prove the SELF target dir IS the frozen adapter -----------------------
    historical = _historical_adapter_sha(adapter_dir) if adapter_dir.is_dir() else ""
    canonical, entries = _canonical_adapter_digest(adapter_dir) if adapter_dir.is_dir() else ("", {})
    record(
        "self_adapter_target_is_frozen_a60e40d8",
        historical == EXPECTED_ADAPTER_SHA,
        {"historical_content_sha256": historical, "expected": EXPECTED_ADAPTER_SHA,
         "note": "the SELF config adapter path maps to this host dir (workspace/runs <-> /workspace/runs); launcher-identical find|sort|xargs sha256sum|sha256sum reproduces the frozen pin"},
    )
    record(
        "canonical_adapter_digest_computed",
        bool(canonical) and len(entries) > 0,
        {"canonical_content_sha256": canonical, "file_count": len(entries),
         "note": "path-independent digest over sorted {relative_path: content_sha256}; survives moving the adapter to another machine",
         "files": entries},
    )

    proof_ok = all(c["ok"] for c in checks)
    result = {
        "schema": "openevo.webshop-h1.37-adapter-arm-proof.v1",
        "experiment": "H1.37-frozen-adapter-fresh-task-eval-v1",
        "adapter_arm_proof_ok": proof_ok,
        "evidence_source": "per-cell wandb offline-run debug.log protocol config line (runner-side, post-hoc, immutable)",
        "expected_self_adapter_container": EXPECTED_SELF_ADAPTER_CONTAINER,
        "adapter_digests": {"historical_launcher_identical": historical, "canonical_path_independent": canonical},
        "checks": checks,
    }
    _atomic_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prove(args.run_root.resolve(), args.manifest.resolve(), args.adapter_dir.resolve(), args.output.resolve())
    print(json.dumps({"adapter_arm_proof_ok": result["adapter_arm_proof_ok"], "failed": [c["check"] for c in result["checks"] if not c["ok"]]}, sort_keys=True))
    return 0 if result["adapter_arm_proof_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
