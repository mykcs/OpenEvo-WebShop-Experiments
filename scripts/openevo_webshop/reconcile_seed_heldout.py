#!/usr/bin/env python3
"""Reconcile SEED held-out comparison run outputs against the frozen panel.

Fail-closed coverage and task-identity reconciliation. Reads the frozen panel
manifest and the per-episode records written by ``run_seed_heldout_eval.py`` for
each (contract, arm) cell, and verifies:

- every scheduled (contract, arm, panel task) cell produced exactly one episode
  record (no missing, no duplicate);
- each episode's goal_idx, instruction hash, and generation seed match the
  frozen manifest (no post-hoc task/seed swap);
- each episode record is complete (task_score, won, trace, steps present).

It does NOT judge scientific quality. It only proves the run covered exactly the
frozen panel with the frozen seeds and identities. Emits a reconciliation report
JSON and exits non-zero on any coverage/identity failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ARMS = ("base", "sd-lora")
CONTRACTS = ("primary", "diagnostic")  # directory names under the run root


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_cell_episodes(cell_dir: Path) -> list[dict[str, Any]]:
    episodes_dir = cell_dir / "episodes"
    if not episodes_dir.is_dir():
        return []
    rows = []
    for path in sorted(episodes_dir.glob("*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001
            rows.append({"_unparseable": str(path), "_error": f"{type(exc).__name__}: {exc}"})
    return rows


def reconcile_contract(manifest: dict[str, Any], run_root: Path, contract: str) -> dict[str, Any]:
    tasks = manifest["tasks"]
    expected = {int(t["goal_idx"]): t for t in tasks}
    contract_report: dict[str, Any] = {"contract": contract, "arms": {}, "ok": True}

    for arm in ARMS:
        cell_dir = run_root / contract / arm
        rows = _load_cell_episodes(cell_dir)
        seen: dict[int, int] = {}
        problems: list[str] = []
        complete = 0
        valid = 0
        for row in rows:
            if "_unparseable" in row:
                problems.append(f"unparseable episode file: {row['_unparseable']}")
                continue
            goal_idx = int(row.get("goal_idx", -1))
            seen[goal_idx] = seen.get(goal_idx, 0) + 1
            exp = expected.get(goal_idx)
            if exp is None:
                problems.append(f"unexpected goal_idx {goal_idx} not in frozen panel")
                continue
            ep = row.get("episode") or {}
            # identity: instruction hash + generation seed must match the manifest
            if row.get("task_id") != exp.get("task_id"):
                problems.append(f"goal_idx {goal_idx}: task_id mismatch")
            if int(row.get("generation_seed", -1)) != int(exp.get("generation_seed", -2)):
                problems.append(f"goal_idx {goal_idx}: generation_seed mismatch (post-hoc seed?)")
            ep_hash = ep.get("task_description_sha256")
            if ep_hash is not None and ep_hash != exp.get("instruction_sha256"):
                problems.append(f"goal_idx {goal_idx}: instruction hash drift")
            # completeness
            if ("task_score" in ep and "won" in ep and "trace" in ep and "steps" in ep) or row.get("runtime_invalid"):
                complete += 1
            else:
                problems.append(f"goal_idx {goal_idx}: incomplete episode record")
            if row.get("valid_episode"):
                valid += 1
        missing = sorted(set(expected) - set(seen))
        duplicates = sorted(g for g, c in seen.items() if c > 1)
        for g in missing:
            problems.append(f"missing episode for goal_idx {g}")
        for g in duplicates:
            problems.append(f"duplicate episodes for goal_idx {g}")
        arm_ok = not problems
        contract_report["arms"][arm] = {
            "expected": len(expected),
            "observed": len(rows),
            "complete": complete,
            "valid": valid,
            "missing": missing,
            "duplicates": duplicates,
            "problems": problems,
            "ok": arm_ok,
        }
        if not arm_ok:
            contract_report["ok"] = False
    return contract_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    args = parser.parse_args()

    manifest = json.loads(args.task_manifest.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "schema": "openevo.webshop-seed-heldout-reconciliation.v1",
        "campaign_id": args.campaign_id,
        "task_manifest_file_sha256": hashlib.sha256(args.task_manifest.read_bytes()).hexdigest(),
        "selected_panel_digest": manifest["selection"]["selected_panel_digest"],
        "panel_size": len(manifest["tasks"]),
        "contracts": {},
        "ok": True,
    }
    for contract in CONTRACTS:
        cell = reconcile_contract(manifest, args.run_root, contract)
        report["contracts"][contract] = cell
        if not cell["ok"]:
            report["ok"] = False
    report["reconciliation_sha256"] = _digest({k: v for k, v in report.items() if k != "reconciliation_sha256"})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": report["ok"],
        "output": str(args.output),
        "reconciliation_sha256": report["reconciliation_sha256"],
        "contracts": {c: {"ok": cell["ok"], **{a: cell["arms"][a]["observed"] for a in ARMS}} for c, cell in report["contracts"].items()},
    }, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
