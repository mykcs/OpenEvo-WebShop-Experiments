#!/usr/bin/env python3
"""Canonical ``scripts/oe`` host delegate for H1.41 evaluation."""

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
        str(ROOT / "scripts/openevo_webshop/h141_upstream_evaluation.py"),
        *sys.argv[1:],
    ]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
