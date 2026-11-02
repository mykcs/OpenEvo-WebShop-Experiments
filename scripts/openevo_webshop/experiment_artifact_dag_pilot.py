#!/usr/bin/env python3
"""Tiny deterministic Passport Artifact DAG pilot; no network/GPU/scientific execution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def deterministic_report(data: bytes) -> dict[str, object]:
    return {
        "input_sha256": hashlib.sha256(data).hexdigest(),
        "line_count": data.count(b"\n"),
        "size_bytes": len(data),
    }


def write_report(input_path: Path, output_path: Path) -> None:
    report = deterministic_report(input_path.read_bytes())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_report(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
