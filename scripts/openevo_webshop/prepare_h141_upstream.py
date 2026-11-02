#!/usr/bin/env python3
"""Prepare and attest the run-scoped H1.41 OpenEvo patch worktree.

The pinned clean checkout is never edited. The reset-magnitude arm gets an
explicit worktree at the same upstream SHA plus the preregistered downstream
patch, with a machine-readable receipt before any GPU-bearing command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


CAMPAIGN_ID = "20260821-0141-h141-magnitude-screen"
OPEN_EVO_SHA = "796e6248a61112a61c880885844d3b0420c0b6ae"
PATCH = Path("scripts/openevo_webshop/patches/openevo-sd-lora-reset-magnitudes.patch")
PATCH_SHA = "3b399a38978c1ef8a3df33d9b9784825086ed43e43099ae8cebf909c575392d6"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root: Path, clean_core: Path, patched_core: Path, receipt: Path) -> dict[str, Any]:
    patch = root / PATCH
    if sha256(patch) != PATCH_SHA:
        raise ValueError("H1.41 reset patch digest does not match the preregistration")
    clean_sha = subprocess.check_output(["git", "-C", str(clean_core), "rev-parse", "HEAD"], text=True).strip()
    if clean_sha != OPEN_EVO_SHA:
        raise ValueError(f"clean OpenEvo checkout is not pinned to {OPEN_EVO_SHA}: {clean_sha}")
    clean_status = subprocess.check_output(["git", "-C", str(clean_core), "status", "--porcelain"], text=True)
    if clean_status:
        raise ValueError("clean pinned OpenEvo checkout is dirty")
    if patched_core.exists():
        if not patched_core.is_dir():
            raise ValueError("patched OpenEvo path exists but is not a directory")
        existing_sha = subprocess.check_output(["git", "-C", str(patched_core), "rev-parse", "HEAD"], text=True).strip()
        existing_status = subprocess.check_output(["git", "-C", str(patched_core), "status", "--porcelain"], text=True)
        if existing_sha != OPEN_EVO_SHA or not existing_status:
            raise ValueError("existing H1.41 patched worktree is not the expected applied-patch state")
    else:
        patched_core.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        subprocess.run(["git", "-C", str(clean_core), "worktree", "add", "--detach", str(patched_core), OPEN_EVO_SHA], check=True)
        subprocess.run(["git", "-C", str(patched_core), "apply", "--whitespace=error-all", str(patch)], check=True)
    applied_diff = subprocess.check_output(["git", "-C", str(patched_core), "diff", "--binary", "--", "src/openevo/evolution/parametric/sd_lora_trainer.py"], text=True)
    if not applied_diff:
        raise ValueError("patched worktree has no trainer diff")
    payload = {
        "schema": "openevo.webshop-h1.41-patched-core-receipt.v1",
        "status": "pass",
        "campaign_id": CAMPAIGN_ID,
        "clean_core": str(clean_core),
        "clean_core_sha": clean_sha,
        "patched_core": str(patched_core),
        "patched_core_base_sha": OPEN_EVO_SHA,
        "patch": str(PATCH),
        "patch_sha256": PATCH_SHA,
        "patched_worktree_status": "base-sha-plus-preregistered-diff",
        "trainer_diff_sha256": hashlib.sha256(applied_diff.encode()).hexdigest(),
    }
    receipt.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    receipt.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt.chmod(0o600)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--clean-core", type=Path, required=True)
    parser.add_argument("--patched-core", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve(), args.clean_core.resolve(), args.patched_core.resolve(), args.receipt.resolve()), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
