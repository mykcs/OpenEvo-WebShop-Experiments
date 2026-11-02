#!/usr/bin/env python3
"""Build one Carrier Health v2 ledger for a completed Stage2 round."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.openevo_webshop.carrier_health_v2 import (
    aggregate_runtime_readbacks,
    build_carrier_health_record,
)

CARRIERS = {
    "text_memory": "text-memory",
    "skill_bundle": "skill-bundle",
    "agent_system": "agent-system",
}


def load(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != rendered:
            raise ValueError(f"carrier health ledger immutable file drift: {path}")
        return
    path.write_text(rendered)


def build_round_ledger(round_root: Path, arm: str, identities: dict) -> dict:
    round_root = round_root.resolve()
    state_before = load(round_root / "state-before.json")
    evidence = load(round_root / "evidence" / "evidence.json")
    rows = [json.loads(line) for line in (round_root / "attempts.jsonl").read_text().splitlines() if line.strip()]
    round_index = int(round_root.name.removeprefix("round-"))
    health_rows = []
    for carrier, dirname in CARRIERS.items():
        terminal = load(round_root / dirname / "terminal-receipt.json")
        runtime = aggregate_runtime_readbacks(rows, carrier)
        health_rows.append(build_carrier_health_record(
            round_index=round_index, arm=arm, carrier=carrier,
            state_id_before=state_before.get("state_id"), terminal=terminal, runtime=runtime,
            identities=identities, requested=bool(evidence["gates"][carrier]["eligible"]),
        ))
    return {
        "schema": "openevo.carrier-health-ledger.v2",
        "round_index": round_index, "arm": arm, "rows": health_rows,
        "status": "PASS" if all(row["health_status"] == "PASS" for row in health_rows) else "FAIL",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-root", type=Path, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--identities", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--require-pass", action="store_true")
    args = ap.parse_args()
    payload = build_round_ledger(args.round_root, args.arm, load(args.identities))
    write_json(args.output, payload)
    if args.require_pass and payload["status"] != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
