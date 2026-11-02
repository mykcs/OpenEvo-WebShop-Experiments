#!/usr/bin/env python3
"""Resolve and validate the execution-locked M1-A adjacent-state source."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

LEDGER_SCHEMA = "openevo.mechanism1-m1a-accepted-state-ledger.202609062100"
NORM_SCHEMA = "openevo.mechanism1-m1a-adjacent-vector-gram-norm.202609062100"
MANIFEST_SCHEMA = "openevo.mechanism1-m1a-adjacent-source-manifest.202609062100"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def require_hex(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise ValueError(f"{label}: expected lowercase SHA-256")
    return text


def resolve(ledger: dict[str, Any], norm: dict[str, Any]) -> dict[str, Any]:
    if ledger.get("schema") != LEDGER_SCHEMA or ledger.get("terminal_seal") is not True:
        raise ValueError("accepted-state ledger is not terminal sealed")
    states = ledger.get("accepted_states")
    if not isinstance(states, list) or len(states) < 3:
        raise ValueError("M1-A successor requires at least three accepted states")
    normalized = []
    for state in states:
        row = {
            "accepted_transition_index": int(state["accepted_transition_index"]),
            "round_index": int(state["round_index"]),
            "generation_index": int(state["generation_index"]),
            "adapter_sha256": require_hex(state["adapter_sha256"], "adapter_sha256"),
            "state_receipt_path": str(state["state_receipt_path"]),
            "state_receipt_sha256": require_hex(state["state_receipt_sha256"], "state_receipt_sha256"),
        }
        normalized.append(row)
    for before, after in zip(normalized, normalized[1:]):
        if after["accepted_transition_index"] != before["accepted_transition_index"] + 1:
            raise ValueError("accepted-transition ledger is not consecutive")
        if after["round_index"] <= before["round_index"]:
            raise ValueError("accepted-state round indices are not strictly increasing")
        if after["generation_index"] <= before["generation_index"]:
            raise ValueError("accepted-state generation indices are not strictly increasing")
        if after["adapter_sha256"] == before["adapter_sha256"]:
            raise ValueError("accepted-state ledger contains a duplicate adjacent adapter")
    target, previous, final = normalized[-3:]
    if len({target["adapter_sha256"], previous["adapter_sha256"], final["adapter_sha256"]}) != 3:
        raise ValueError("selected M1-A states are not pairwise distinct")
    if norm.get("schema") != NORM_SCHEMA or norm.get("status") != "PASS":
        raise ValueError("Gram-norm receipt is not PASS")
    if norm.get("source_adapter_sha256") != previous["adapter_sha256"]:
        raise ValueError("Gram-norm source adapter mismatch")
    if norm.get("target_adapter_sha256") != final["adapter_sha256"]:
        raise ValueError("Gram-norm target adapter mismatch")
    value = float(norm.get("gram_norm"))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("late adjacent vector Gram norm must be finite and positive")
    return {
        "schema": MANIFEST_SCHEMA,
        "status": "IDENTIFIABLE_EXECUTION_LOCKED",
        "execution_authority": False,
        "formal_task_consumption_allowed": False,
        "selection_used_reward": False,
        "theta_target": target,
        "theta_previous": previous,
        "theta_final": final,
        "source_vector": "theta_final - theta_previous",
        "gram_norm": value,
    }


def verify_bound_files(manifest: dict[str, Any], root: Path) -> None:
    for key in ("theta_target", "theta_previous", "theta_final"):
        state = manifest[key]
        path = root / state["state_receipt_path"]
        if not path.is_file() or sha256(path) != state["state_receipt_sha256"]:
            raise ValueError(f"{key}: state receipt identity mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--gram-norm-receipt", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = resolve(load(args.ledger), load(args.gram_norm_receipt))
    if args.artifact_root is not None:
        verify_bound_files(manifest, args.artifact_root)
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
