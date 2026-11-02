#!/usr/bin/env python3
"""Reuse the strict paired self-evolution reconciler under the H1.33 identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.openevo_webshop.reconcile_h1_29 import reconcile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cell", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reconcile(args.manifest.resolve(), args.cell, args.output.resolve())
    result["schema"] = "openevo.webshop-h1.33-reconciliation.v1"
    result["experiment"] = "H1.33-positive-replay-transfer-v1"
    result["claim_boundary"] = "H0.1-positive replay fresh-task transfer diagnostic only; no SEED comparison claim"
    args.output.resolve().write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 2 if result["status"] == "invalid" else 0


if __name__ == "__main__":
    raise SystemExit(main())
