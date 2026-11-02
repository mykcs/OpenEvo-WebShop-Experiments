#!/usr/bin/env python3
"""Fail-closed tombstone for the superseded generic Ray Stage2 runner."""
from __future__ import annotations
import sys


def main() -> int:
    print(
        "NO-GO: run_ceiling1_stage2_vnext_ray_v1.py is SUPERSEDED_HISTORY_ONLY; "
        "resolve the current hash-bound runtime through scripts/oe campaign.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
