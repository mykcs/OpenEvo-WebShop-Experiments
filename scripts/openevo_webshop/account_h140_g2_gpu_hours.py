#!/usr/bin/env python3
"""Fail-closed CPU-only H1.40 campaign resource accounting."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


STAGES = ("qualification_gpu_hours", "harvest_gpu_hours", "engineering_invalid_gpu_hours", "training_gpu_hours", "evaluation_gpu_hours")


def account(receipt: dict[str, Any], cap: float = 10.0) -> dict[str, Any]:
    unknown = [stage for stage in STAGES if receipt.get(stage) is None]
    invalid = [stage for stage in STAGES if receipt.get(stage) is not None and (not isinstance(receipt[stage], (int, float)) or isinstance(receipt[stage], bool) or not math.isfinite(float(receipt[stage])) or float(receipt[stage]) < 0)]
    if invalid:
        return {"schema": "openevo.webshop-h1.40-g2-resource-accounting.v1", "status": "invalid", "invalid_stages": invalid, "cap": cap}
    if unknown:
        return {"schema": "openevo.webshop-h1.40-g2-resource-accounting.v1", "status": "unknown", "unknown_stages": unknown, "campaign_total_gpu_hours": None, "cap": cap, "reason": "missing stage resource receipt; do not infer zero"}
    total = sum(float(receipt[stage]) for stage in STAGES)
    return {"schema": "openevo.webshop-h1.40-g2-resource-accounting.v1", "status": "deviation" if total > cap else "pass", "stage_gpu_hours": {stage: float(receipt[stage]) for stage in STAGES}, "campaign_total_gpu_hours": total, "cap": cap, "within_cap": total <= cap}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = account(json.loads(args.receipt.read_text(encoding="utf-8")))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
