#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.openevo_webshop.elastic_gpu_lane import build_qualification_cache


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def gpu_rows() -> dict[int, dict]:
    out = subprocess.check_output(
        [
            'nvidia-smi',
            '--query-gpu=index,uuid,name,memory.total',
            '--format=csv,noheader,nounits',
        ],
        text=True,
    )
    rows = {}
    for line in out.splitlines():
        index, uuid, name, total = [x.strip() for x in line.split(',', 3)]
        rows[int(index)] = {'uuid': uuid, 'name': name, 'total_mib': int(total)}
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--runtime-fingerprint', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    receipt = build_qualification_cache(
        load(args.config),
        gpu_rows(),
        args.runtime_fingerprint,
        time.time(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
