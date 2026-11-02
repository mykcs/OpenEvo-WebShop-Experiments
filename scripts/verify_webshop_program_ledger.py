#!/usr/bin/env python3
"""Verify the WebShop program ledger after live-router de-duplication.

The original v1 validator is preserved byte-for-byte in
`verify_webshop_program_ledger_legacy.py`.  Its historical experiment and claim
checks remain authoritative.  This v2 facade changes only one control-plane
property: the historical ledger no longer carries a second writable copy of
current campaign state.  Live authority comes directly from
`configs/experiment/current-campaign.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from scripts import verify_webshop_program_ledger_legacy as legacy


LEDGER_ROUTER_PATH = Path("docs/science/webshop/program/WEBSHOP_PROGRAM_LEDGER.v1.json")
LEDGER_PATH = Path("docs/science/webshop/program/WEBSHOP_PROGRAM_LEDGER.history.v1.json")
CURRENT_CAMPAIGN_PATH = legacy.CURRENT_CAMPAIGN_PATH
REPORT_PATH = legacy.REPORT_PATH
EXPECTED_HISTORICAL_BLOB_SHA = "2a11e17fb4cb3f0472b8d7ca2280da2e135f2e7a"
REQUIRED_CURRENT_FIELDS = (
    "campaign_id",
    "phase",
    "status",
    "protocol_status",
    "formal_task_consumption_allowed",
    "gpu_allocation_allowed",
    "task_bearing_train_dev_allowed",
    "final_test_status",
    "final_pool_consumed",
    "final_unlock_allowed",
    "next_gate",
    "latest_event",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise AssertionError(f"expected JSON object: {path}")
    return value


def _inside_git(root: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def _git_blob_sha(root: Path, relative: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "hash-object", str(relative)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"cannot hash tracked ledger snapshot: {relative}")
    return result.stdout.strip()


def _verify_router(root: Path) -> None:
    """Verify the small router in a real checkout; synthetic temp tests may omit it."""

    router_path = root / LEDGER_ROUTER_PATH
    if not router_path.exists():
        if _inside_git(root):
            raise AssertionError(f"ledger router missing: {LEDGER_ROUTER_PATH}")
        return

    router = _load_json(router_path)
    if router.get("schema") != "openevo.webshop-program-ledger-router.v2":
        raise AssertionError("unsupported WebShop ledger router schema")
    if router.get("status") != "ACTIVE_ROUTER":
        raise AssertionError("WebShop ledger router is not active")
    if router.get("historical_inventory") != LEDGER_PATH.as_posix():
        raise AssertionError("ledger router historical_inventory pointer changed")
    if router.get("current_state_authority") != CURRENT_CAMPAIGN_PATH.as_posix():
        raise AssertionError("ledger router current_state_authority pointer changed")
    if router.get("historical_inventory_blob_sha") != EXPECTED_HISTORICAL_BLOB_SHA:
        raise AssertionError("ledger router historical blob pin changed")

    historical_path = root / LEDGER_PATH
    current_path = root / CURRENT_CAMPAIGN_PATH
    if not historical_path.exists() or not current_path.exists():
        raise AssertionError("ledger router target missing")
    if _inside_git(root):
        if _git_blob_sha(root, LEDGER_PATH) != EXPECTED_HISTORICAL_BLOB_SHA:
            raise AssertionError("historical program ledger is not byte-identical to frozen v1")
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", str(LEDGER_ROUTER_PATH), str(LEDGER_PATH)],
            text=True,
            capture_output=True,
            check=False,
        )
        if tracked.returncode != 0:
            raise AssertionError("ledger router/history files are not tracked")


def _verify_current_authority(root: Path) -> dict[str, Any]:
    current = _load_json(root / CURRENT_CAMPAIGN_PATH)
    missing = [field for field in REQUIRED_CURRENT_FIELDS if field not in current]
    if missing:
        raise AssertionError(f"current-campaign live authority missing fields: {missing}")

    for field in (
        "formal_task_consumption_allowed",
        "gpu_allocation_allowed",
        "task_bearing_train_dev_allowed",
        "final_pool_consumed",
        "final_unlock_allowed",
    ):
        if not isinstance(current[field], bool):
            raise AssertionError(f"current-campaign {field} must be boolean")

    latest = current.get("latest_event")
    if not isinstance(latest, dict) or not latest.get("classification"):
        raise AssertionError("current-campaign latest_event needs a classification")

    # Amendment-006 is a CPU-only state-boundary repair, never a hidden task/GPU release.
    if current.get("native_state_repair_allowed"):
        if current["formal_task_consumption_allowed"]:
            raise AssertionError("native-state repair must not authorize formal task consumption")
        if current["task_bearing_train_dev_allowed"]:
            raise AssertionError("native-state repair must not authorize task-bearing train/dev")
        if current["gpu_allocation_allowed"]:
            raise AssertionError("native-state repair must not authorize GPU allocation")
        if current["final_test_status"] != "locked" or current["final_unlock_allowed"]:
            raise AssertionError("native-state repair must keep final locked")
        repair = current.get("gen28_native_state_repair")
        if not isinstance(repair, dict) or repair.get("allowed") is not True:
            raise AssertionError("native-state repair router lacks explicit repair contract")
        if repair.get("webshop_rerun_allowed") is not False:
            raise AssertionError("Gen28 repair must prohibit WebShop rerun")
        if repair.get("new_task_consumption_allowed") is not False:
            raise AssertionError("Gen28 repair must prohibit new task consumption")
        if repair.get("gpu_allowed") is not False or repair.get("gen29_authorized") is not False:
            raise AssertionError("Gen28 repair must prohibit GPU and Gen29")
        if repair.get("deterministic_reconstruction_claim") is not False:
            raise AssertionError("Gen28 deferred MemEvolve update must not claim deterministic reconstruction")

    return current


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    _verify_router(root)
    current = _verify_current_authority(root)

    # Keep every historical inventory/evidence/denominator/report guard from v1.
    # Its old embedded router field is projected transiently from the live owner;
    # the immutable history file remains unchanged and is independently hash-pinned.
    old_ledger_path = legacy.LEDGER_PATH
    old_campaign_path = legacy.CURRENT_CAMPAIGN_PATH
    old_router_fields = legacy.CURRENT_ROUTER_FIELDS
    with tempfile.TemporaryDirectory(prefix="webshop-ledger-history-") as temp_dir:
        historical_ledger_path = Path(temp_dir) / "ledger.json"
        historical_ledger = _load_json(root / LEDGER_PATH)
        historical_ledger["current_state"] = current
        historical_ledger_path.write_text(json.dumps(historical_ledger, sort_keys=True) + "\n", encoding="utf-8")
        try:
            legacy.LEDGER_PATH = historical_ledger_path
            legacy.CURRENT_CAMPAIGN_PATH = old_campaign_path
            result = legacy.verify(root)
        finally:
            legacy.LEDGER_PATH = old_ledger_path
            legacy.CURRENT_CAMPAIGN_PATH = old_campaign_path
            legacy.CURRENT_ROUTER_FIELDS = old_router_fields

    result = dict(result)
    result.update(
        {
            "ledger_router": "pass",
            "historical_inventory": LEDGER_PATH.as_posix(),
            "current_state_authority": CURRENT_CAMPAIGN_PATH.as_posix(),
            "current_campaign": current["campaign_id"],
        }
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
