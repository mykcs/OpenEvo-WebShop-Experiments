#!/usr/bin/env python3
"""Emit the frozen H1.40-G2 stage plan without consuming tasks or GPU time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STAGES = (
    ("qualification", "reuse or requalify the pinned runtime fingerprint", 0),
    ("harvest", "G1 endogenous E2 reservoir, stop at 256 attempts", 256),
    ("acquisition", "G2 versus G1 on eight selected E2 identities", 64),
    ("e1_retention", "G2 versus G1 on eight H1.38B retention identities", 64),
    ("h139_preservation", "G2 versus G1 on 32 H1.39 preservation identities", 256),
    ("t2_fresh", "G0/G1/G2 fresh-task transfer panel", 768),
    ("accounting", "reconcile stage GPU receipts against the 10 GPU-hour cap", 0),
    ("quiescence", "verify no active H1.40 launcher/container remains", 0),
)


def plan(root: Path) -> dict:
    config = json.loads((root / "configs/experiment/h1.40-g2-multigeneration-v1.json").read_text(encoding="utf-8"))
    if config.get("status") != "preregistered-pre-execution" or config.get("formal_task_consumption_allowed") is not False:
        raise ValueError("H1.40 plan requires the preregistered closed state")
    return {
        "schema": "openevo.webshop-h1.40-g2-execution-plan.v1",
        "campaign_id": config["campaign_id"],
        "status": "plan-only",
        "formal_task_consumption_allowed": False,
        "gpu_workload_launched": False,
        "g2_adapter_created": False,
        "t2_panel_evaluated": False,
        "stages": [
            {"name": name, "description": description, "attempts": attempts, "executes_in_preparation": False}
            for name, description, attempts in STAGES
        ],
        "next_gate": "explicit launch-state transition plus current parent-policy live-idle gate",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = plan(args.root.resolve())
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
