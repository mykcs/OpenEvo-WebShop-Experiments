#!/usr/bin/env python3
"""Canonical ``scripts/oe`` host preflight delegate for H1.41 training."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    command = [
        sys.executable,
        str(ROOT / "scripts/oe"),
        "host",
        "--path-contract",
        str(ROOT / "configs/experiment/h1.41-path-contract-v1.json"),
        "--cwd",
        str(ROOT),
        "--",
        sys.executable,
        str(ROOT / "scripts/openevo_webshop/run_h141_magnitude_training.py"),
        *sys.argv[1:],
    ]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
