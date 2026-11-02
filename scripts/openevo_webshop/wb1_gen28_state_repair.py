#!/usr/bin/env python3
"""Seal and repair the missing WB1 Gen28 native-state update.

This tool never executes WebShop and never allocates a GPU.  `seal` is a
read-only inspection of the already-completed Gen28 run.  `apply` may run the
one CPU-only native MemEvolve update authorized by Amendment 006, but only after
recomputing an identical PASS seal.  The historical update implementation is
materialized byte-for-byte from the actual Gen28 execution commit rather than
silently substituting current code.

Producing state-v28 does not itself adopt it.  `reconcile` creates a local
barrier receipt; repository publication/reconciliation must still advance the
campaign authority before Gen29 is legal.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


SCHEMA_SEAL = "openevo.wb1-gen28-input-seal.v1"
SCHEMA_REPAIR = "openevo.wb1-gen28-deferred-native-update.v1"
SCHEMA_BARRIER = "openevo.wb1-gen28-state-v28-barrier.v1"
CAMPAIGN_ID = "20260821-2341-wb1-seed-aligned-webshop-benchmark"
GENERATION = 28
FORMAL_SEED = 20260821
EXPECTED_EPISODES = 128
EXPECTED_EXACT_SUCCESSES = 2
HISTORICAL_EXECUTION_SHA = "4d9061083cc98bc99244e05346602867f0135c83"
PINNED_OPEN_EVO_SHA = "796e6248a61112a61c880885844d3b0420c0b6ae"
TASK_STREAM_SHA256 = "21f9efc76f36daf67acc70288ccee9cc1db3330c2e942ddd43741ad6bb04d2f5"
STATE_V27_CHECKSUM = "83b06092ea87ce2a3a8a569d493175fbc92a79879e4a11728ff11113456ba132"
STATE_V27_FILE_SHA256 = "135349958a19163cab598bb2595c33c7f22017cdec224f07d11f2f029032685e"
MEMORY_V27_SHA256 = "9efb2716920cfb6ada8eb31bae49b873da4e70145a3b23eb82cb8ca5c580241d"
HISTORICAL_BLOBS = {
    "scripts/openevo_webshop/wb1_native_state.py": "5c6fce27031fba4c7bb0175954c932b2e0e55a74",
    "scripts/openevo_webshop/wb1_contracts.py": "829970bf86cb737c8c7c44369808d87d99fdec2b",
}
COUNTED_BEFORE = 3456
COUNTED_AFTER = 3584
BUDGET_TARGET = 20640
REMAINING_AFTER = BUDGET_TARGET - COUNTED_AFTER


class RepairError(RuntimeError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RepairError(f"required file missing: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RepairError(f"invalid JSON: {path}: {exc}") from exc


def _document_sha(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return _sha256_bytes(_canonical_bytes(payload))


def _write_receipt(path: Path, document: Mapping[str, Any], *, digest_field: str) -> None:
    payload = dict(document)
    payload[digest_field] = _document_sha(payload, digest_field)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _contains_value(value: Any, needle: str) -> bool:
    if value == needle:
        return True
    if isinstance(value, dict):
        return any(_contains_value(item, needle) for item in value.values())
    if isinstance(value, list):
        return any(_contains_value(item, needle) for item in value)
    return False


def _episode_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RepairError(f"episodes line {line_number} is invalid JSON") from exc
        if not isinstance(row, dict):
            raise RepairError(f"episodes line {line_number} is not an object")
        rows.append(row)
    return rows


def build_seal(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    if "generation-28" not in run_root.name:
        raise RepairError(f"refusing non-Gen28 run root: {run_root}")

    attempt = run_root / "attempt-01"
    paths = {
        "episodes": attempt / "episodes.jsonl",
        "summary": attempt / "summary.json",
        "manifest": attempt / "manifest.json",
        "config": attempt / "config.json",
        "state_v27": run_root / "state-v27.json",
        "memory_v27": run_root / "memory-v27.md",
        "task_ids_gen28": run_root / "task-ids-gen28.json",
    }
    identities = {name: _file_identity(path) for name, path in paths.items()}

    if identities["task_ids_gen28"]["sha256"] != TASK_STREAM_SHA256:
        raise RepairError("Gen28 task-stream SHA-256 mismatch")
    if identities["state_v27"]["sha256"] != STATE_V27_FILE_SHA256:
        raise RepairError("state-v27 file SHA-256 mismatch")
    if identities["memory_v27"]["sha256"] != MEMORY_V27_SHA256:
        raise RepairError("memory-v27 SHA-256 mismatch")

    state = _load_json(paths["state_v27"])
    if not isinstance(state, dict):
        raise RepairError("state-v27 must be a JSON object")
    if state.get("state_version") != 27:
        raise RepairError("state-v27 version mismatch")
    if state.get("state_sha256") != STATE_V27_CHECKSUM:
        raise RepairError("state-v27 internal checksum identity mismatch")
    if state.get("adapter_write") or state.get("optimizer_write") or state.get("parameter_update"):
        raise RepairError("state-v27 unexpectedly records parameter mutation")

    rows = _episode_rows(paths["episodes"])
    if len(rows) != EXPECTED_EPISODES:
        raise RepairError(f"expected {EXPECTED_EPISODES} Gen28 episodes; observed {len(rows)}")
    goal_ids: list[int] = []
    successes = 0
    for index, row in enumerate(rows):
        if row.get("valid_completed") is not True:
            raise RepairError(f"Gen28 episode {index} is not valid_completed=true")
        try:
            goal_idx = int(row["goal_idx"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RepairError(f"Gen28 episode {index} has invalid goal_idx") from exc
        if goal_idx < 500:
            raise RepairError(f"final-pool leakage in Gen28 episodes: goal_idx={goal_idx}")
        goal_ids.append(goal_idx)
        successes += int(bool(row.get("won")))
    if successes != EXPECTED_EXACT_SUCCESSES:
        raise RepairError(
            f"Gen28 exact-success mismatch: expected {EXPECTED_EXACT_SUCCESSES}, observed {successes}"
        )
    counts = Counter(goal_ids)
    if len(counts) != 16 or set(counts.values()) != {8}:
        raise RepairError("Gen28 episode stream is not 16 unique train tasks x 8 rollouts")

    identity_documents = [_load_json(paths[name]) for name in ("summary", "manifest", "config")]
    if not any(_contains_value(document, HISTORICAL_EXECUTION_SHA) for document in identity_documents):
        raise RepairError("historical Gen28 execution SHA not found in summary/manifest/config")

    forbidden_existing = [
        run_root / "state-v28.json",
        run_root / "memory-v28.md",
        run_root / "native-state-v28",
        run_root / "state-v28-barrier.json",
    ]
    existing = [str(path) for path in forbidden_existing if path.exists()]
    if existing:
        raise RepairError(f"pre-existing state-v28 artifact(s) require reconciliation, not repair: {existing}")

    seal: dict[str, Any] = {
        "schema": SCHEMA_SEAL,
        "status": "PASS",
        "campaign_id": CAMPAIGN_ID,
        "generation": GENERATION,
        "formal_seed": FORMAL_SEED,
        "run_root": str(run_root),
        "attempt": "attempt-01",
        "historical_execution_sha": HISTORICAL_EXECUTION_SHA,
        "files": identities,
        "episode_contract": {
            "valid_completed": len(rows),
            "exact_successes": successes,
            "unique_tasks": len(counts),
            "rollouts_per_task": 8,
            "goal_idx_min": min(goal_ids),
            "goal_idx_max": max(goal_ids),
            "final_pool_tasks": 0,
        },
        "input_state": {
            "version": 27,
            "state_checksum": STATE_V27_CHECKSUM,
            "state_file_sha256": STATE_V27_FILE_SHA256,
            "memory_sha256": MEMORY_V27_SHA256,
        },
        "task_stream_sha256": TASK_STREAM_SHA256,
        "state_v28_preexisting": False,
        "authorization": "Amendment 006 input-seal gate only; no WebShop/GPU/Gen29/final",
    }
    seal["seal_sha256"] = _document_sha(seal, "seal_sha256")
    return seal


def _assert_seal_matches(seal_path: Path, run_root: Path) -> dict[str, Any]:
    saved = _load_json(seal_path)
    if not isinstance(saved, dict) or saved.get("schema") != SCHEMA_SEAL or saved.get("status") != "PASS":
        raise RepairError("repair requires a PASS Gen28 input seal")
    if saved.get("seal_sha256") != _document_sha(saved, "seal_sha256"):
        raise RepairError("saved input-seal digest mismatch")
    fresh = build_seal(run_root)
    if saved != fresh:
        raise RepairError("Gen28 inputs changed after sealing; repair aborted")
    return saved


def _git_blob_sha(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def _git_show(repo_root: Path, revision_path: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "show", revision_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RepairError(
            f"historical execution object unavailable ({revision_path}); fetch the exact commit before repair: {detail}"
        )
    return result.stdout


def materialize_historical_update(repo_root: Path, destination: Path) -> dict[str, str]:
    recorded: dict[str, str] = {}
    for relative_path, expected_blob in HISTORICAL_BLOBS.items():
        content = _git_show(repo_root, f"{HISTORICAL_EXECUTION_SHA}:{relative_path}")
        observed_blob = _git_blob_sha(content)
        if observed_blob != expected_blob:
            raise RepairError(
                f"historical implementation blob mismatch for {relative_path}: {observed_blob} != {expected_blob}"
            )
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        recorded[relative_path] = observed_blob
    return recorded


def _git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RepairError(f"not a Git checkout: {path}")
    return result.stdout.strip()


def apply_repair(
    *,
    repo_root: Path,
    run_root: Path,
    seal_path: Path,
    open_evo_root: Path,
    output_root: Path,
    codex_binary: str | None = None,
    codex_home: Path | None = None,
) -> dict[str, Any]:
    seal = _assert_seal_matches(seal_path, run_root)
    output_root = output_root.resolve()
    accepted_receipt = output_root / "repair-receipt.json"
    if accepted_receipt.exists():
        raise RepairError(f"exactly-once guard: repair receipt already exists: {accepted_receipt}")
    if (output_root / "native" / "state.json").exists():
        raise RepairError("partial/pre-existing native repair output requires reconciliation before retry")

    open_evo_root = open_evo_root.resolve()
    if _git_head(open_evo_root) != PINNED_OPEN_EVO_SHA:
        raise RepairError("OpenEvo checkout is not at the pinned WB1 commit")

    implementation_root = output_root / "historical-implementation"
    blobs = materialize_historical_update(repo_root.resolve(), implementation_root)
    historical_script = implementation_root / "scripts/openevo_webshop/wb1_native_state.py"
    native_output = output_root / "native"

    env = dict(os.environ)
    pythonpath = [str(implementation_root), str(open_evo_root)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if codex_binary:
        env["WB1_CODEX_BINARY"] = codex_binary
    if codex_home:
        env["WB1_CODEX_HOME"] = str(codex_home.resolve())

    command = [
        sys.executable,
        str(historical_script),
        "--episodes",
        str((run_root / "attempt-01" / "episodes.jsonl").resolve()),
        "--generation",
        "28",
        "--output",
        str(native_output),
        "--prior-state",
        str((run_root / "state-v27.json").resolve()),
        "--reflector-model",
        "gpt-5.5",
        "--harness-mode",
        "direct-codex",
    ]
    result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace")[-4000:]
        raise RepairError(f"deferred native update failed rc={result.returncode}: {detail.strip()}")

    native_receipt_path = native_output / "receipt.json"
    native_state_path = native_output / "state.json"
    native_receipt = _load_json(native_receipt_path)
    state = _load_json(native_state_path)
    if not isinstance(native_receipt, dict) or not isinstance(state, dict):
        raise RepairError("native update did not emit valid receipt/state objects")
    if native_receipt.get("generation") != 28 or native_receipt.get("state_version") != 28:
        raise RepairError("native update generation/state version mismatch")
    if native_receipt.get("open_evo_sha") != PINNED_OPEN_EVO_SHA:
        raise RepairError("native update OpenEvo identity mismatch")
    if native_receipt.get("episodes_sha256") != seal["files"]["episodes"]["sha256"]:
        raise RepairError("native update episode identity does not match input seal")
    if native_receipt.get("harness_mode") != "direct-codex":
        raise RepairError("native update must use direct-codex")
    if native_receipt.get("partial_task_score_included") is not False:
        raise RepairError("native update leaked partial Task Score")
    if state.get("state_version") != 28 or state.get("state_format") != "native-openevo-static-markdown-v1":
        raise RepairError("state-v28 format/version invalid")
    if state.get("adapter_write") or state.get("optimizer_write") or state.get("parameter_update"):
        raise RepairError("state-v28 records forbidden parameter mutation")
    if state.get("partial_task_score_included") is not False:
        raise RepairError("state-v28 records partial Task Score input")

    repair: dict[str, Any] = {
        "schema": SCHEMA_REPAIR,
        "status": "STATE_V28_PRODUCED_PENDING_RECONCILIATION",
        "campaign_id": CAMPAIGN_ID,
        "generation": 28,
        "amendment": "WB1_AMENDMENT_006_GEN28_DEFERRED_NATIVE_STATE_UPDATE_2026-08-27.md",
        "input_seal_sha256": seal["seal_sha256"],
        "input_episodes_sha256": seal["files"]["episodes"]["sha256"],
        "input_state_v27_sha256": STATE_V27_CHECKSUM,
        "historical_execution_sha": HISTORICAL_EXECUTION_SHA,
        "historical_implementation_blobs": blobs,
        "open_evo_sha": PINNED_OPEN_EVO_SHA,
        "native_method": "text_memory_memevolve",
        "harness_mode": "direct-codex",
        "reflector_model": "gpt-5.5",
        "candidate_count": 2,
        "max_records": 100,
        "output": {
            "native_receipt_sha256": _sha256_file(native_receipt_path),
            "state_file_sha256": _sha256_file(native_state_path),
            "state_checksum": state.get("state_sha256"),
            "memory_markdown_sha256": _sha256_bytes(str(state.get("memory_markdown", "")).encode("utf-8")),
            "state_version": state.get("state_version"),
        },
        "scientific_boundary": (
            "Deferred execution of the missing Gen28 native update using sealed historical inputs; "
            "not a deterministic reconstruction and not yet adopted for Gen29."
        ),
        "gpu_authorized": False,
        "new_webshop_task_consumption": 0,
        "gen29_authorized": False,
        "final_unlock_allowed": False,
    }
    _write_receipt(accepted_receipt, repair, digest_field="repair_receipt_sha256")
    return _load_json(accepted_receipt)


def reconcile_repair(*, run_root: Path, seal_path: Path, output_root: Path) -> dict[str, Any]:
    seal = _assert_seal_matches(seal_path, run_root)
    repair_path = output_root / "repair-receipt.json"
    repair = _load_json(repair_path)
    if not isinstance(repair, dict) or repair.get("schema") != SCHEMA_REPAIR:
        raise RepairError("missing/invalid deferred-update repair receipt")
    if repair.get("repair_receipt_sha256") != _document_sha(repair, "repair_receipt_sha256"):
        raise RepairError("repair receipt digest mismatch")
    if repair.get("input_seal_sha256") != seal["seal_sha256"]:
        raise RepairError("repair receipt is not bound to current PASS seal")

    state_path = output_root / "native" / "state.json"
    native_receipt_path = output_root / "native" / "receipt.json"
    if _sha256_file(state_path) != repair["output"]["state_file_sha256"]:
        raise RepairError("state-v28 file changed after repair")
    if _sha256_file(native_receipt_path) != repair["output"]["native_receipt_sha256"]:
        raise RepairError("native receipt changed after repair")
    state = _load_json(state_path)
    if not isinstance(state, dict) or state.get("state_version") != 28:
        raise RepairError("state-v28 is not valid for barrier")
    if state.get("state_sha256") != repair["output"]["state_checksum"]:
        raise RepairError("state-v28 checksum mismatch")
    if _sha256_bytes(str(state.get("memory_markdown", "")).encode("utf-8")) != repair["output"]["memory_markdown_sha256"]:
        raise RepairError("state-v28 memory hash mismatch")

    barrier: dict[str, Any] = {
        "schema": SCHEMA_BARRIER,
        "status": "PASS_STATE_V28_ADOPTABLE_AFTER_REPOSITORY_PUBLICATION",
        "campaign_id": CAMPAIGN_ID,
        "generation": 28,
        "input_seal_sha256": seal["seal_sha256"],
        "repair_receipt_sha256": repair["repair_receipt_sha256"],
        "state_v28": repair["output"],
        "budget_if_adopted": {
            "target": BUDGET_TARGET,
            "counted_before": COUNTED_BEFORE,
            "gen28_credit": 128,
            "counted_after": COUNTED_AFTER,
            "remaining": REMAINING_AFTER,
        },
        "new_webshop_task_consumption": 0,
        "gpu_used": False,
        "final_pool_consumed": 0,
        "gen29_authorized": False,
        "adoption_rule": (
            "Publish this barrier and repair evidence, then update current-campaign in a separate "
            "repository reconciliation. Until that merge, state-v27 remains the program authority."
        ),
    }
    barrier_path = output_root / "state-v28-barrier.json"
    _write_receipt(barrier_path, barrier, digest_field="barrier_sha256")
    return _load_json(barrier_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    seal_p = sub.add_parser("seal")
    seal_p.add_argument("--run-root", type=Path, required=True)
    seal_p.add_argument("--output", type=Path, required=True)

    apply_p = sub.add_parser("apply")
    apply_p.add_argument("--repo-root", type=Path, required=True)
    apply_p.add_argument("--run-root", type=Path, required=True)
    apply_p.add_argument("--seal", type=Path, required=True)
    apply_p.add_argument("--open-evo-root", type=Path, required=True)
    apply_p.add_argument("--output-root", type=Path, required=True)
    apply_p.add_argument("--codex-binary")
    apply_p.add_argument("--codex-home", type=Path)

    rec_p = sub.add_parser("reconcile")
    rec_p.add_argument("--run-root", type=Path, required=True)
    rec_p.add_argument("--seal", type=Path, required=True)
    rec_p.add_argument("--output-root", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "seal":
            document = build_seal(args.run_root)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        elif args.command == "apply":
            document = apply_repair(
                repo_root=args.repo_root,
                run_root=args.run_root,
                seal_path=args.seal,
                open_evo_root=args.open_evo_root,
                output_root=args.output_root,
                codex_binary=args.codex_binary,
                codex_home=args.codex_home,
            )
        else:
            document = reconcile_repair(
                run_root=args.run_root,
                seal_path=args.seal,
                output_root=args.output_root,
            )
    except RepairError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(document, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
