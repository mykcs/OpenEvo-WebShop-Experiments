#!/usr/bin/env python3
"""Zero-model/zero-GPU semantic validation of the source-faithful manifest.

This script runs on the server with the real SEED WebShop runtime. It
instantiates the pinned SEED ``WebshopWorker`` actor for each slot, calls
``worker.reset(session=sampled_session_index)`` through the real
``WebAgentTextEnv``/``SimServer`` path, and checks that every instruction
SHA-256 matches the committed manifest.

Requirements (server only):
    - pinned SEED checkout at 2cf2fadca3c5aba28da68e8e1405182ba8d90e6c
    - WebShop data files (items_shuffle_1000.json, items_ins_v2_1000.json,
      items_human_ins.json)
    - matching Lucene index at ``<index-root>/indexes_1k``
    - the committed semantic manifest

This is catalog inspection, not scientific task consumption: it only reads the
synthetic goal catalog via the real environment reset path. No model, no GPU,
no episode steps, no score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PINNED_SEED_COMMIT = "2cf2fadca3c5aba28da68e8e1405182ba8d90e6c"
PANEL_SIZE = 128
VALIDATION_BASE_SEED = 1000
WORKER_SEED_OFFSET = 1000


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_digest(value: Any) -> str:
    return _sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _hash_data_files(data_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in ("items_shuffle_1000.json", "items_ins_v2_1000.json", "items_human_ins.json"):
        path = data_dir / name
        if not path.is_file():
            raise SystemExit(f"missing WebShop data file: {path}")
        hashes[name] = _sha256_bytes(path.read_bytes())
    return hashes


def _verify_source_tree(seed_root: Path) -> dict[str, Any]:
    """Fail-closed source identity: exact pinned commit and clean worktree."""
    if not (seed_root / ".git").is_dir():
        raise SystemExit(f"SEED source is not a git checkout: {seed_root}")
    result = subprocess.run(
        ["git", "-C", str(seed_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"git rev-parse failed: {result.stderr.strip()}")
    observed_commit = result.stdout.strip()
    if observed_commit != PINNED_SEED_COMMIT:
        raise SystemExit(
            f"SEED checkout commit mismatch: expected {PINNED_SEED_COMMIT}, observed {observed_commit}"
        )
    status_result = subprocess.run(
        ["git", "-C", str(seed_root), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    if status_result.returncode != 0:
        raise SystemExit(f"git status failed: {status_result.stderr.strip()}")
    worktree_clean = status_result.stdout.strip() == ""
    if not worktree_clean:
        raise SystemExit(
            f"SEED worktree is dirty; source bytes may differ from pinned commit {PINNED_SEED_COMMIT}"
        )
    return {
        "expected_commit": PINNED_SEED_COMMIT,
        "observed_commit": observed_commit,
        "worktree_clean": True,
    }


def _compute_expected_session_draw() -> list[int]:
    import numpy as np

    selection = np.random.RandomState(VALIDATION_BASE_SEED).choice(
        np.arange(500), size=PANEL_SIZE, replace=False
    )
    return [int(x) for x in selection]


def _verify_manifest_structure(manifest: dict[str, Any]) -> dict[str, Any]:
    """Structural fail-closed checks independent of any runtime reconstruction."""
    if manifest.get("schema") != "openevo.webshop-seed-source-faithful-panel.v1":
        raise SystemExit("manifest schema mismatch")
    tasks = manifest["tasks"]
    if len(tasks) != PANEL_SIZE:
        raise SystemExit(f"expected {PANEL_SIZE} tasks, got {len(tasks)}")

    expected_draw = _compute_expected_session_draw()
    if len(set(expected_draw)) != PANEL_SIZE:
        raise SystemExit("internal error: expected session draw produced duplicates")

    failures: list[dict[str, Any]] = []
    seen_slots: set[int] = set()
    for task in tasks:
        slot = int(task["slot"])
        worker_seed = int(task["worker_seed"])
        session_index = int(task["sampled_session_index"])
        expected_worker_seed = WORKER_SEED_OFFSET + slot
        expected_session_index = expected_draw[slot]

        checks = []
        if slot != len(seen_slots):
            checks.append(f"slot out of order: expected {len(seen_slots)}, got {slot}")
        if worker_seed != expected_worker_seed:
            checks.append(
                f"worker_seed mismatch: expected {expected_worker_seed}, got {worker_seed}"
            )
        if session_index != expected_session_index:
            checks.append(
                f"sampled_session_index mismatch: expected {expected_session_index}, got {session_index}"
            )
        if slot in seen_slots:
            checks.append(f"duplicate slot {slot}")
        seen_slots.add(slot)

        if checks:
            failures.append({
                "slot": slot,
                "worker_seed": worker_seed,
                "sampled_session_index": session_index,
                "failures": checks,
            })

    if failures:
        raise SystemExit(
            f"manifest structural validation failed for {len(failures)} task(s): {failures[:3]}"
        )
    return {"expected_session_draw": expected_draw, "slot_count": len(seen_slots)}


def _extract_instruction_text(raw_instruction: str) -> str:
    """Strip the "Instruction: " prefix that SimServer HTML renders.

    The manifest records the raw ``goal['instruction_text']`` value, while
    ``WebAgentTextEnv.get_instruction_text()`` returns the same text prefixed
    by the HTML rendering ("Instruction: ...").
    """
    prefix = "Instruction: "
    if not raw_instruction.startswith(prefix):
        raise ValueError(
            f"runtime instruction does not start with expected prefix {prefix!r}: {raw_instruction[:200]!r}"
        )
    return raw_instruction[len(prefix):]


def _experiment_git_sha(experiment_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(experiment_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"experiment git rev-parse failed: {result.stderr.strip()}")
    return result.stdout.strip()


def validate(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve()
    data_root = args.webshop_data.resolve()
    index_root = args.index_root.resolve()
    manifest_path = args.manifest.resolve()
    expected_manifest_file_sha256 = args.expected_manifest_file_sha256
    expected_manifest_content_sha256 = args.expected_manifest_content_sha256
    experiment_root = args.experiment_root.resolve()

    if not expected_manifest_file_sha256:
        raise SystemExit("--expected-manifest-file-sha256 is required")
    if not expected_manifest_content_sha256:
        raise SystemExit("--expected-manifest-content-sha256 is required")

    source_identity = _verify_source_tree(source_root)
    experiment_git_sha = _experiment_git_sha(experiment_root)
    validator_file_sha256 = _sha256_bytes(Path(__file__).read_bytes())

    if not (source_root / "seed" / "analysis.py").is_file():
        raise SystemExit(f"SEED source is incomplete: {source_root}")
    for name in ("items_shuffle_1000.json", "items_ins_v2_1000.json"):
        if not (data_root / name).is_file():
            raise SystemExit(f"WebShop data is incomplete: {data_root / name}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Verify committed manifest identity against frozen external hashes and internal self-consistency.
    manifest_bytes = manifest_path.read_bytes()
    observed_manifest_file_sha256 = _sha256_bytes(manifest_bytes)
    recorded_manifest_content_sha256 = manifest.get("manifest_content_sha256")
    computed_manifest_content_sha256 = _canonical_digest(
        {k: v for k, v in manifest.items() if k != "manifest_content_sha256"}
    )

    if observed_manifest_file_sha256 != expected_manifest_file_sha256:
        raise SystemExit(
            f"manifest file SHA-256 mismatch: expected {expected_manifest_file_sha256}, "
            f"observed {observed_manifest_file_sha256}"
        )
    if computed_manifest_content_sha256 != expected_manifest_content_sha256:
        raise SystemExit(
            f"manifest content SHA-256 mismatch: expected {expected_manifest_content_sha256}, "
            f"computed {computed_manifest_content_sha256}"
        )
    if recorded_manifest_content_sha256 != expected_manifest_content_sha256:
        raise SystemExit(
            f"manifest embedded content SHA-256 mismatch: expected {expected_manifest_content_sha256}, "
            f"recorded {recorded_manifest_content_sha256}"
        )

    # Reject stale/tampered identity before structural reconstruction, whose
    # deterministic sampler may require optional runtime dependencies.
    structure = _verify_manifest_structure(manifest)
    tasks = manifest["tasks"]

    # Verify server data files equal the manifest-frozen hashes (fail-closed).
    data_hashes = _hash_data_files(data_root)
    manifest_data_hashes = manifest.get("benchmark", {}).get("data_files") or {}
    data_hash_mismatches: list[dict[str, str]] = []
    for name, observed_hash in data_hashes.items():
        expected_hash = manifest_data_hashes.get(name)
        if expected_hash is None:
            data_hash_mismatches.append({"file": name, "reason": "missing in manifest"})
        elif observed_hash != expected_hash:
            data_hash_mismatches.append({
                "file": name,
                "expected": expected_hash,
                "observed": observed_hash,
            })
    if data_hash_mismatches:
        raise SystemExit(
            f"server data file hash mismatch(es): {data_hash_mismatches}"
        )

    sys.path.insert(0, str(source_root))
    webshop_root = source_root / "agent_system/environments/env_package/webshop/webshop"
    sys.path.insert(0, str(webshop_root))

    import gym
    import web_agent_site.engine.engine as webshop_engine
    import web_agent_site.envs.web_agent_text_env as text_env_module
    from pyserini.search.lucene import LuceneSearcher

    index_dir = index_root / "indexes_1k"
    if not index_dir.is_dir():
        raise SystemExit(f"WebShop Lucene index is not ready: {index_dir}")

    # Pin data and search paths to the server assets under validation.
    webshop_engine.DEFAULT_FILE_PATH = str(data_root / "items_shuffle_1000.json")
    webshop_engine.DEFAULT_ATTR_PATH = str(data_root / "items_ins_v2_1000.json")
    webshop_engine.HUMAN_ATTR_PATH = str(data_root / "items_human_ins.json")
    text_env_module.init_search_engine = lambda num_products=None: LuceneSearcher(
        str(index_dir)
    )

    # WebshopWorker hosts the real WebAgentTextEnv/SimServer runtime used by SEED.
    from agent_system.environments.env_package.webshop.envs import WebshopWorker

    env_kwargs = {
        "observation_mode": "text",
        "file_path": str(data_root / "items_shuffle_1000.json"),
        "num_products": 1000,
        "human_goals": 0,
    }

    matches = 0
    mismatches: list[dict[str, Any]] = []

    for task in tasks:
        slot = int(task["slot"])
        worker_seed = int(task["worker_seed"])
        session_index = int(task["sampled_session_index"])
        expected_sha256 = str(task["instruction_sha256"])

        worker = WebshopWorker(worker_seed, env_kwargs)
        try:
            worker.reset(session_index)
            raw_instruction = worker.env.get_instruction_text()
            instruction = _extract_instruction_text(raw_instruction)
        finally:
            worker.close()

        observed_sha256 = _sha256_text(instruction)
        if observed_sha256 == expected_sha256:
            matches += 1
        else:
            mismatches.append(
                {
                    "slot": slot,
                    "worker_seed": worker_seed,
                    "session_index": session_index,
                    "expected_sha256": expected_sha256,
                    "observed_sha256": observed_sha256,
                    "observed_instruction_prefix": instruction[:200],
                }
            )

    status = "PASS" if matches == len(tasks) and not mismatches else "FAIL"

    receipt: dict[str, Any] = {
        "schema": "openevo.webshop-seed-source-faithful-runtime-semantic-validation.v1",
        "campaign_id": manifest.get("campaign_id"),
        "status": status,
        "experiment_git_sha": experiment_git_sha,
        "validator": {
            "path": str(Path(__file__).relative_to(experiment_root)),
            "file_sha256": validator_file_sha256,
        },
        "server_receipt_path": str(args.output.resolve()),
        "GPU_used": False,
        "model_loaded": False,
        "formal_task_consumption": 0,
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": observed_manifest_file_sha256,
        "manifest_content_sha256": computed_manifest_content_sha256,
        "source_identity": source_identity,
        "source_pins": {
            "seed_repository": "github.com/jinyangwu/SEED",
            "seed_commit": PINNED_SEED_COMMIT,
            "seed_root_used": str(source_root),
            "data_dir_used": str(data_root),
            "index_root_used": str(index_root),
        },
        "data_files": data_hashes,
        "manifest_structure": {
            "expected_session_draw": structure["expected_session_draw"],
            "slot_count": structure["slot_count"],
        },
        "validation_summary": {
            "total_slots": len(tasks),
            "matches": matches,
            "mismatches": len(mismatches),
            "match_rate": matches / len(tasks) if tasks else 0.0,
        },
        "mismatches": mismatches,
        "created_utc": args.created_utc,
    }
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--webshop-data", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-file-sha256", required=True)
    parser.add_argument("--expected-manifest-content-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--created-utc", required=True)
    args = parser.parse_args()

    receipt = validate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt["validation_summary"], indent=2))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
