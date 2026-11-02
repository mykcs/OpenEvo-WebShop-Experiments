#!/usr/bin/env python3
"""Post-run integrity verifier for the H1.37 frozen-adapter fresh-task panel.

This is CLOSEOUT tooling, not part of the frozen evaluation protocol. It re-proves,
independently of reconcile_h1_37.py and independently of W&B (whose offline runs
stayed status=active and never flushed config.yaml), that the resumed 6-GPU run
produced exactly the preregistered 512-attempt panel with zero duplication or
omission and an unchanged scientific contract.

Proof mechanisms, per user-mandated closeout requirements:

1. DUAL-SHA PROVENANCE. The frozen scientific contract (79a5963...) and the resume
   orchestration driver (0fa0a94...) are recorded separately. Attempt metadata keeps
   the frozen contract SHA; this record states which orchestration code ran the
   remaining cells.

2. PREREGISTRATION DEVIATION. The 8->6 GPU scheduling recovery is disclosed as a
   scheduling-only deviation (model/adapter/manifest/parser/seeds/temperature/task
   contract all unchanged); it does not alter the reward comparison semantics.

3. EXACT 512 RECONCILIATION (machine-verified, not asserted): 64 unique task ids;
   exactly 2 arms/task; exactly 4 rollout seeds/arm/task; exactly 512 unique
   (task_id, arm, seed) attempts; no partial cell; per-cell fingerprint stability;
   and -- by recomputing the protocol fingerprint from the frozen constants plus
   each cell's bound GPU UUID -- proof that EVERY attempt used the v3 action parser,
   the sealed manifest sha256, temperature 0.0, prompt contract v2, the frozen model
   revision, and the preregistered generation seeds. Because cuda_visible_devices is
   an input to the fingerprint, a match also proves the UUID GPU binding. A v1/v2
   parser or a swapped manifest would change the fingerprint and fail this check.

4. TEMPORAL / ARM-ORDER CONFOUND (diagnostic only; does NOT change the primary
   analysis). The original 8-GPU launcher completed the first 48 base cells (manifest
   order) before dying at the next wave's GPU gate; the remaining 16 base cells and
   all 64 self-evolution cells ran under the 6-GPU resume. We split base cells into
   original-period versus resumed-period and compare mean score, median wall time,
   and fingerprint validity to detect drift. SELF-EVOLUTION is entirely resumed-period,
   so arm and wall-clock are not counterbalanced; this is disclosed, not adjusted.

Stdlib only (json/hashlib/pathlib/argparse/statistics); runs on the host against the
run root without Docker and without importing openevo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

ARMS = ("base", "self-evolution")

# Frozen H1.37 scientific-contract constants (identical to the pinned launcher).
FROZEN = {
    "phase": "H1",
    "model": "/workspace/models/qwen2.5-7b-instruct",
    "model_revision": "0c32faf81d8f2f83cf79860d2a48111f166348d6dd38cd348576d8b3a0141c1e",
    "environment_seed": 2026,
    "generation_seeds": [2026090200, 2026090201, 2026090202, 2026090203],
    "temperature": 0.0,
    "max_steps": 30,
    "max_new_tokens": 512,
    "prompt_contract": "seed3090.webshop-compact-action-prompt.v2",
    "action_contract": "seed3090.webshop-action-parser.v3-action-command",
    "evaluator": "WebShop task_score/won",
    "parser_retry_seed_offset": 900000,
    "gpu_binding_contract": "single UUID via CUDA_VISIBLE_DEVICES; logical cuda:0",
    "latency_contract": "attempt_wall_time_seconds measured around one closed-loop episode; invalid attempts remain outside speed denominators",
}

# Provenance (requirement #1): keep the two SHAs distinct.
SCIENTIFIC_CONTRACT_SHA = "79a59638a3463e245fb865feb3f14dd248ec51be"
RESUME_ORCHESTRATION_SHA = "0fa0a9431b8fe51ff9864ccf4b22acdd401fd4a7"
EXPECTED_MANIFEST_SHA = "6e55fd5d06a0f2461af2c28643c0aa5d6a15e3048fe7cd182ec9b3b440a0ed07"
EXPECTED_ADAPTER_SHA = "a60e40d866b3ed6aa41873b6ec39c2844adf4b88e58799f3bc2d2d02da3f2489"
# The original launcher completed the first this many base cells (manifest order)
# before the GPU gate stopped it; the resume finished the rest.
ORIGINAL_BASE_CELLS = 48


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def _expected_fingerprint(arm: str, gpu_uuid: str) -> str:
    # task_manifest_sha256 is part of the fingerprinted protocol (see
    # h1_closed_loop_tracked.py); we inject the SEALED manifest sha so that a match
    # also proves the attempt consumed the sealed manifest (a swap would mismatch).
    # The manifest file itself is separately proven to hash to EXPECTED_MANIFEST_SHA
    # by the manifest_sha256_constant check.
    protocol = {**FROZEN, "arm": arm, "cuda_visible_devices": gpu_uuid, "task_manifest_sha256": EXPECTED_MANIFEST_SHA}
    return _digest(protocol)


def _adapter_content_sha(adapter_dir: Path) -> str:
    # Reproduce the launcher's exact adapter pin pipeline so the value is comparable
    # to the sealed frozen_adapter.content_sha256 (a60e40d8...): the launcher hashed
    # `sha256sum` output (which embeds the absolute host paths), not raw file digests.
    # We call the identical shell pipeline rather than approximate it in Python.
    import subprocess

    cmd = f'find "{adapter_dir}" -type f | LC_ALL=C sort | xargs sha256sum | sha256sum'
    out = subprocess.run(["bash", "-c", cmd], check=True, capture_output=True, text=True).stdout
    return out.split()[0] if out.split() else ""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def verify(run_root: Path, manifest_path: Path, seal_path: Path, adapter_dir: Path, output: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: Any) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = [int(v) for v in manifest.get("evaluation_tasks", [])]
    seeds = [int(v) for v in manifest.get("evaluation", {}).get("rollout_seeds", [])]

    # --- A. Manifest identity -------------------------------------------------
    manifest_file_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    seal = json.loads(seal_path.read_text(encoding="utf-8")) if seal_path.is_file() else {}
    seal_sha = seal.get("manifest_file_sha256")
    record(
        "manifest_sha256_constant",
        manifest_file_sha == EXPECTED_MANIFEST_SHA == seal_sha,
        {"manifest_file_sha256": manifest_file_sha, "seal_sha256": seal_sha, "expected": EXPECTED_MANIFEST_SHA},
    )
    record("manifest_64_unique_tasks", len(tasks) == 64 and len(set(tasks)) == 64, {"task_count": len(tasks), "unique": len(set(tasks))})
    record("manifest_4_unique_seeds", seeds == FROZEN["generation_seeds"], {"seeds": seeds})
    record("manifest_arm_contract", manifest.get("evaluation", {}).get("arms") == list(ARMS), {"arms": manifest.get("evaluation", {}).get("arms")})

    # --- B. Exact 512 panel ---------------------------------------------------
    eval_dir = run_root / "evaluation"
    seen: set[tuple[str, int, int]] = set()
    duplicates: list[Any] = []
    partial_cells: list[Any] = []
    per_cell: dict[tuple[str, int], dict[str, Any]] = {}
    for arm in ARMS:
        for task in tasks:
            cell_dir = eval_dir / f"{arm}-task{task}"
            attempts_path = cell_dir / "attempts.jsonl"
            if not attempts_path.is_file():
                partial_cells.append({"cell": f"{arm}:{task}", "reason": "missing attempts.jsonl"})
                continue
            rows = _load_jsonl(attempts_path)
            if len(rows) != len(seeds):
                partial_cells.append({"cell": f"{arm}:{task}", "row_count": len(rows)})
            fingerprints = set()
            gpu_uuid = _cell_gpu_uuid(cell_dir)
            for row in rows:
                key = (str(row.get("arm")), int(row.get("goal_idx", -1)), int(row.get("rollout_seed", -1)))
                if key in seen:
                    duplicates.append(key)
                seen.add(key)
                fp = row.get("protocol_fingerprint")
                if isinstance(fp, str):
                    fingerprints.add(fp)
            per_cell[(arm, task)] = {
                "rows": len(rows),
                "fingerprints": sorted(fingerprints),
                "gpu_uuid": gpu_uuid,
                "mtime": attempts_path.stat().st_mtime,
                "mean_score": (sum(float(r.get("final_task_score") or 0.0) for r in rows) / len(rows)) if rows else None,
                "wall_times": [float(r.get("attempt_wall_time_seconds") or 0.0) for r in rows],
                "task_id_ok": all(r.get("task_id") == f"webshop_{task:06d}" for r in rows),
                "arm_ok": all(r.get("arm") == arm for r in rows),
            }

    expected = {(arm, task, seed) for arm in ARMS for task in tasks for seed in seeds}
    record("exactly_512_unique_attempts", seen == expected and len(seen) == 512, {"unique_attempts": len(seen), "missing": len(expected - seen), "extra": len(seen - expected)})
    record("no_duplicate_attempts", not duplicates, {"duplicates": duplicates[:10]})
    record("no_partial_cells", not partial_cells, {"partial": partial_cells[:10]})
    record("two_arms_per_task", all(any((arm, t, s) in seen for s in seeds) for t in tasks for arm in ARMS), {})
    record("four_seeds_per_arm_task", all(sum((arm, t, s) in seen for s in seeds) == 4 for t in tasks for arm in ARMS), {})
    record("task_id_format_ok", all(c["task_id_ok"] for c in per_cell.values()), {})
    record("cell_arm_label_ok", all(c["arm_ok"] for c in per_cell.values()), {})

    # --- C. Fingerprint recomputation (parser/manifest/temperature/GPU proof) --
    fp_mismatches: list[Any] = []
    unstable_cells: list[Any] = []
    for (arm, task), info in per_cell.items():
        if len(info["fingerprints"]) != 1:
            unstable_cells.append({"cell": f"{arm}:{task}", "fingerprints": info["fingerprints"]})
            continue
        gpu_uuid = info["gpu_uuid"]
        if not gpu_uuid:
            fp_mismatches.append({"cell": f"{arm}:{task}", "reason": "no gpu uuid"})
            continue
        expected_fp = _expected_fingerprint(arm, gpu_uuid)
        if info["fingerprints"][0] != expected_fp:
            fp_mismatches.append({"cell": f"{arm}:{task}", "observed": info["fingerprints"][0], "expected": expected_fp, "gpu_uuid": gpu_uuid})
    record("per_cell_fingerprint_stable", not unstable_cells, {"unstable": unstable_cells[:10]})
    record(
        "fingerprint_matches_frozen_contract",
        not fp_mismatches,
        {"mismatch_count": len(fp_mismatches), "mismatches": fp_mismatches[:10],
         "proves": "every attempt used v3 action parser, sealed manifest sha256, temperature 0.0, prompt v2, frozen model revision, preregistered seeds, and its bound GPU UUID"},
    )

    # --- D. Frozen adapter pin -------------------------------------------------
    adapter_sha = _adapter_content_sha(adapter_dir) if adapter_dir.is_dir() else ""
    seal_adapter_sha = seal.get("frozen_adapter", {}).get("content_sha256")
    record(
        "frozen_adapter_sha256",
        adapter_sha == EXPECTED_ADAPTER_SHA and (not seal_adapter_sha or seal_adapter_sha == EXPECTED_ADAPTER_SHA),
        {"adapter_content_sha256": adapter_sha, "seal_adapter_sha256": seal_adapter_sha, "expected": EXPECTED_ADAPTER_SHA,
         "note": "adapter path is bound by launcher construction (base->none, self-evolution->this dir); it is not an input to the protocol fingerprint and wandb config.yaml was not flushed (offline runs stayed status=active), so per-cell adapter-path plaintext is unavailable. Proof = single frozen adapter dir re-pinned here + arm contract."},
    )

    # --- E. Temporal / arm-order confound (diagnostic only) -------------------
    base_cells = [(t, per_cell[("base", t)]) for t in tasks if ("base", t) in per_cell]
    base_by_order = [t for t in tasks if ("base", t) in per_cell]  # manifest order
    original_tasks = set(base_by_order[:ORIGINAL_BASE_CELLS])
    original = [info for (t, info) in base_cells if t in original_tasks]
    resumed = [info for (t, info) in base_cells if t not in original_tasks]

    def _period_stats(cells: list[dict[str, Any]]) -> dict[str, Any]:
        scores = [c["mean_score"] for c in cells if c["mean_score"] is not None]
        walls = [w for c in cells for w in c["wall_times"]]
        mtimes = [c["mtime"] for c in cells]
        return {
            "cell_count": len(cells),
            "mean_score": (sum(scores) / len(scores)) if scores else None,
            "median_wall_time_seconds": statistics.median(walls) if walls else None,
            "mtime_min": min(mtimes) if mtimes else None,
            "mtime_max": max(mtimes) if mtimes else None,
        }

    temporal = {
        "disclosure": (
            "Arm and wall-clock are NOT counterbalanced: the original 8-GPU launcher completed "
            f"the first {ORIGINAL_BASE_CELLS} base cells (manifest order) before the GPU gate stopped it; "
            "the remaining base cells and ALL self-evolution cells ran later under the 6-GPU resume. "
            "Each episode is independent (frozen model, frozen adapter, frozen environment, temperature 0.0), "
            "so this is a provenance/robustness diagnostic, not a change to the primary analysis."
        ),
        "base_original_period": _period_stats(original),
        "base_resumed_period": _period_stats(resumed),
        "self_evolution_cells_all_resumed_period": True,
    }
    record("temporal_arm_order_diagnostic", True, temporal)  # always recorded; not a pass/fail gate

    # --- Provenance + deviation narrative -------------------------------------
    provenance = {
        "scientific_contract_sha": SCIENTIFIC_CONTRACT_SHA,
        "resume_orchestration_sha": RESUME_ORCHESTRATION_SHA,
        "note": "attempt metadata keeps the frozen scientific-contract SHA; the resume driver that ran the remaining 80 cells is the orchestration SHA. Both are recorded to avoid provenance ambiguity.",
    }
    deviation = {
        "type": "scheduling-only (not scientific-protocol)",
        "gpu_observation_scope": "experiment-host container PID namespace",
        "gpu_ownership_conclusion": "unresolved at the gate; fail-closed scheduling exclusion",
        "summary": (
            f"{ORIGINAL_BASE_CELLS} base cells completed under the original 8-GPU launcher. After two GPUs "
            "(indices 4 and 7) retained ~21 GiB while no owner was visible from the experiment-host PID namespace, "
            "ownership remained unresolved and the resource gate failed closed. The "
            "remaining 80 cells (16 base + 64 self-evolution) were resumed on six identical RTX 5090 GPUs "
            "with unchanged per-episode scientific semantics (model/adapter/manifest/parser/seeds/temperature/"
            "task contract all frozen)."
        ),
    }

    integrity_ok = all(c["ok"] for c in checks if c["check"] != "temporal_arm_order_diagnostic")
    result = {
        "schema": "openevo.webshop-h1.37-integrity.v1",
        "experiment": "H1.37-frozen-adapter-fresh-task-eval-v1",
        "integrity_ok": integrity_ok,
        "provenance": provenance,
        "preregistration_deviation": deviation,
        "checks": checks,
    }
    _atomic_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run_root.resolve(), args.manifest.resolve(), args.seal.resolve(), args.adapter_dir.resolve(), args.output.resolve())
    print(json.dumps({"integrity_ok": result["integrity_ok"], "failed": [c["check"] for c in result["checks"] if not c["ok"]]}, sort_keys=True))
    return 0 if result["integrity_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
