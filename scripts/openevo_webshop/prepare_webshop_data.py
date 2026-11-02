#!/usr/bin/env python3
"""Validate the three WebShop small-dataset JSON files and optionally manifest them.

This helper intentionally performs no network access. Network transport is owned
by the Mac-side GitHub API relay, while RTX6 only receives already validated
benchmark assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

FILES: dict[str, dict[str, Any]] = {
    "items_shuffle_1000.json": {"min_bytes": 1_000_000, "expect_products": True},
    "items_ins_v2_1000.json": {"min_bytes": 50_000, "expect_products": False},
    "items_human_ins.json": {"min_bytes": 1_000_000, "expect_products": False},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(path: Path, spec: dict[str, Any]) -> tuple[int, str, int, str]:
    size = path.stat().st_size
    if size < int(spec["min_bytes"]):
        raise ValueError(f"file too small: {size} bytes")
    with path.open("r", encoding="utf-8") as fh:
        obj = json.load(fh)
    if not isinstance(obj, (list, dict)) or len(obj) == 0:
        raise ValueError(f"unexpected JSON root: {type(obj).__name__}")
    if spec.get("expect_products") and (not isinstance(obj, list) or len(obj) != 1000):
        raise ValueError(
            f"expected 1000-product list, got {type(obj).__name__} len={len(obj)}"
        )
    return size, sha256(path), len(obj), type(obj).__name__


def validate_all(target: Path, *, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, spec in FILES.items():
        path = target / name
        if not path.is_file():
            raise FileNotFoundError(name)
        size, digest, entries, root_type = validate(path, spec)
        print(
            f"WEBSHOP_DATA_VALID name={name} bytes={size} sha256={digest} "
            f"entries={entries} root={root_type} source={source}",
            flush=True,
        )
        rows.append(
            {
                "name": name,
                "bytes": size,
                "sha256": digest,
                "entries": entries,
                "root_type": root_type,
                "source": source,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--source", default="validated-cache")
    parser.add_argument("target_dir", type=Path)
    args = parser.parse_args()

    if not args.validate_only:
        print(
            "WEBSHOP_DATA_PREPARER_NO_NETWORK: use --validate-only; "
            "transport is handled by fetch_webshop_data_via_github_api.sh",
            file=sys.stderr,
        )
        return 2

    target = args.target_dir.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    try:
        rows = validate_all(target, source=args.source)
    except Exception as exc:  # noqa: BLE001
        print(
            f"WEBSHOP_DATA_VALIDATE_FAILED error={type(exc).__name__}:{exc}",
            file=sys.stderr,
            flush=True,
        )
        return 3

    if args.write_manifest:
        manifest = {
            "format": 4,
            "dataset": "princeton-nlp/WebShop small (1000 products)",
            "files": rows,
            "transport": "pinned GitHub blob API -> Mac cache -> rsync -> RTX6",
            "source": args.source,
        }
        manifest_path = target / "MANIFEST.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"WEBSHOP_DATA_MANIFEST={manifest_path}", flush=True)

    print(f"WEBSHOP_DATA_VALIDATE_OK={target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
