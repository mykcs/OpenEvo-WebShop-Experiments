#!/usr/bin/env python3
"""Create a 64-row WebShop validation carrier from the pinned public 128-row carrier."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SOURCE_SHA = "271bf898898887e2d68494080888cc63c7e6c92db5f573366ca21b773af7f71f"
SOURCE_ROWS = 128
OUTPUT_ROWS = 64


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adapt(source_path: Path, output_path: Path) -> dict[str, object]:
    if source_path.resolve() == output_path.resolve():
        raise ValueError("refusing to overwrite the immutable public 128-row carrier")
    if sha(source_path) != SOURCE_SHA:
        raise ValueError("source WebShop carrier SHA mismatch")
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for this inert carrier adaptation") from exc

    table = parquet.read_table(source_path)
    if table.num_rows != SOURCE_ROWS:
        raise ValueError(f"source carrier must have exactly {SOURCE_ROWS} rows")
    if "answer" not in table.column_names:
        raise ValueError("source Geometry3K carrier schema drifted")
    if "extra_info" not in table.column_names or "prompt" not in table.column_names:
        raise ValueError("source carrier missing the pinned prompt/index fields")
    # Keep the original answer column byte-value semantics untouched; the WebShop
    # runtime consumes only its own environment goals and never reads this column.
    view = table.slice(0, OUTPUT_ROWS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(view, output_path)
    return {
        "schema": "openevo.post-advisor-e5-q17-validation-carrier.v1",
        "status": "INERT_PREPARATION_ONLY",
        "source_path": str(source_path),
        "source_sha256": SOURCE_SHA,
        "source_rows": SOURCE_ROWS,
        "output_path": str(output_path),
        "output_rows": OUTPUT_ROWS,
        "output_sha256": sha(output_path),
        "selected_geometry_carrier_indices": list(range(OUTPUT_ROWS)),
        "geometry_indices_are_webshop_session_ids": False,
        "q17_sessions_are_bound_by": "separate hash-bound Q17 schedule consumed by the generated runtime overlay",
        "answer_column_present_but_not_inspected": True,
        "model_calls": 0,
        "gpu_calls": 0,
        "final_panel_access": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = adapt(args.source, args.output)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
