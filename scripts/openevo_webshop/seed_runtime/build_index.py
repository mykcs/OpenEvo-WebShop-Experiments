#!/usr/bin/env python3
"""Build the external SEED WebShop 1K Lucene index in a separate asset root.

The SEED checkout intentionally stays read-only.  The generated resources and
Lucene tree are runtime assets and must remain outside the Git repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    source_probe = source_root / "seed" / "analysis.py"
    if not source_probe.is_file():
        raise SystemExit(f"SEED source is incomplete: {source_probe}")

    product_path = data_root / "items_shuffle_1000.json"
    attr_path = data_root / "items_ins_v2_1000.json"
    human_attr_path = data_root / "items_human_ins.json"
    for path in (product_path, attr_path, human_attr_path):
        if not path.is_file():
            raise SystemExit(f"WebShop data is incomplete: {path}")

    resource_root = output_root / "resources_1k"
    index_root = output_root / "indexes_1k"
    if resource_root.exists() or index_root.exists():
        raise SystemExit(
            f"Refusing to overwrite existing index asset root: {output_root}"
        )
    resource_root.mkdir(parents=True)

    sys.path.insert(0, str(source_root))
    sys.path.insert(
        0,
        str(source_root / "agent_system/environments/env_package/webshop/webshop"),
    )
    from agent_system.environments.env_package.webshop.webshop.web_agent_site.engine import (  # noqa: E501
        engine as webshop_engine,
    )

    webshop_engine.DEFAULT_ATTR_PATH = str(attr_path)
    webshop_engine.HUMAN_ATTR_PATH = str(human_attr_path)

    products, *_ = webshop_engine.load_products(
        filepath=str(product_path), human_goals=True
    )
    documents_path = resource_root / "documents.jsonl"
    with documents_path.open("w", encoding="utf-8") as handle:
        for product in products[:1000]:
            option_texts = []
            for option_name, option_contents in product.get("options", {}).items():
                option_texts.append(f"{option_name}: {', '.join(option_contents)}")
            contents = " ".join(
                [
                    product["Title"],
                    product["Description"],
                    product["BulletPoints"][0],
                    ", and ".join(option_texts),
                ]
            ).lower()
            handle.write(
                json.dumps(
                    {"id": product["asin"], "contents": contents, "product": product},
                    ensure_ascii=False,
                )
                + "\n"
            )

    command = [
        sys.executable,
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(resource_root),
        "--index",
        str(index_root),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        "1",
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]
    subprocess.run(command, check=True)

    manifest = {
        "source_commit": args.source_commit,
        "product_count": len(products[:1000]),
        "data": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (product_path, attr_path, human_attr_path)
        },
        "resources": {
            "path": str(documents_path),
            "bytes": documents_path.stat().st_size,
            "sha256": sha256_file(documents_path),
        },
        "index_path": str(index_root),
        "command": command,
    }
    (output_root / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
