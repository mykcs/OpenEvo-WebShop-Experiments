#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.openevo_webshop.elastic_gpu_lane import hardware_exposure_audit


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('receipts', nargs='+', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    rows = [json.loads(path.read_text()) for path in args.receipts]
    receipt = hardware_exposure_audit(rows)
    text = json.dumps(receipt, indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
