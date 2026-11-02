#!/usr/bin/env python3
"""Fail-closed CPU verifier for the qualification WebShop 1k Lucene index."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


EXPECTED_DOCS = 1000
QUERIES = (
    ("toothpaste", "INDEXES_1K_QUERY_1", (("B09S3NVLCV", 3.793400), ("B00IKXU4OQ", 3.534500), ("B001E77OCU", 3.500700))),
    ("wireless charging iphone", "INDEXES_1K_QUERY_2", (("B09J5HJ8DL", 8.258000), ("B09CPZH548", 7.635600), ("B07THL8PP1", 7.622000))),
    ("men dress shirt", "INDEXES_1K_QUERY_3", (("B09S3BN15C", 6.308700), ("B09QGK5XHZ", 6.038300), ("B09QQP3356", 5.657300))),
    ("slim fit jumpsuits", "INDEXES_1K_QUERY_4", (("B01MSP2B0I", 5.239300), ("B07F2G93BJ", 4.696300), ("B09NNMV9LN", 4.226600))),
    ("quick release iphone accessories", "INDEXES_1K_QUERY_5", (("B08QRHVS6N", 8.404100), ("B01MTYK2T9", 7.110900), ("B01M7U6F8T", 4.756600))),
    ("gym shorts grey", "INDEXES_1K_QUERY_6", (("B07WMMYB6G", 6.446900), ("B096P77XMD", 6.206600), ("B09HX5CD2D", 5.638700))),
    ("dark heather dress shirts", "INDEXES_1K_QUERY_7", (("B07DKGJR74", 7.031300), ("B09RVF4JP1", 6.949700), ("B07XPR3R7N", 6.621900))),
    ("high performance high speed flashes", "INDEXES_1K_QUERY_8", (("B09P57ZKGP", 7.632500), ("B08VH8MZZ6", 6.830100), ("B000EMWBV0", 6.562100))),
    ("women sweaters coffee brown", "INDEXES_1K_QUERY_9", (("B09KRLGSC5", 8.438500), ("B09LHM4WQS", 4.823500), ("B09P5CRVQ6", 4.666400))),
    ("home office furniture rectangular", "INDEXES_1K_QUERY_10", (("B09JC84P1M", 5.049500), ("B07FKGQKZ1", 4.475500), ("B000GLTJ3M", 4.029700))),
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index_tree_identity(index_dir: Path) -> dict[str, Any]:
    rows = [
        {
            "name": path.relative_to(index_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in index_dir.rglob("*")
        if path.is_file()
    ]
    rows.sort(key=lambda row: row["name"])
    if not rows:
        raise ValueError("qualification indexes_1k is empty")
    return {
        "file_count": len(rows),
        "logical_bytes": sum(row["bytes"] for row in rows),
        "tree_sha256": hashlib.sha256(canonical_bytes(rows)).hexdigest(),
    }


def resource_identity(path: Path) -> dict[str, Any]:
    documents = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"resource JSONL contains a blank line at {line_number}")
        try:
            documents.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"resource JSONL is invalid at line {line_number}") from exc
    if len(documents) != EXPECTED_DOCS or any(not isinstance(document, dict) for document in documents):
        raise ValueError("qualification resources must contain exactly 1000 JSON objects")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "document_count": len(documents),
        "canonical_document_fingerprint_sha256": hashlib.sha256(canonical_bytes(documents)).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--fingerprint-out", type=Path)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    if not isinstance(catalog, list) or len(catalog) != EXPECTED_DOCS:
        raise ValueError("qualification 1k catalog must contain exactly 1000 rows")
    ordered_product_ids = [str(row.get("asin", "")) for row in catalog if isinstance(row, dict)]
    if len(ordered_product_ids) != EXPECTED_DOCS or len(set(ordered_product_ids)) != EXPECTED_DOCS or "" in ordered_product_ids:
        raise ValueError("qualification 1k catalog ASIN identity mismatch")
    product_ids = set(ordered_product_ids)

    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(str(args.index.resolve()))
    if int(searcher.num_docs) != EXPECTED_DOCS:
        raise ValueError(f"qualification indexes_1k document count mismatch: {searcher.num_docs}")
    print(f"INDEXES_1K_DOC_COUNT={searcher.num_docs}")
    indexed_product_ids = [str(searcher.doc(index).docid()) for index in range(EXPECTED_DOCS)]
    if indexed_product_ids != ordered_product_ids:
        raise ValueError("qualification indexes_1k indexed document order differs from frozen catalog order")
    if len(set(indexed_product_ids)) != EXPECTED_DOCS or not set(indexed_product_ids) <= product_ids:
        raise ValueError("qualification indexes_1k indexed product identity mismatch")

    query_results = []
    for query, marker, expected_hits in QUERIES:
        hits = searcher.search(query, k=len(expected_hits))
        if len(hits) != len(expected_hits) or any(str(hit.docid) not in product_ids for hit in hits):
            raise ValueError(f"qualification indexes_1k semantic query failed: {query}")
        actual_hits = [(str(hit.docid), round(float(hit.score), 6)) for hit in hits]
        expected_normalized = [(docid, round(score, 6)) for docid, score in expected_hits]
        if actual_hits != expected_normalized:
            raise ValueError(
                f"qualification indexes_1k semantic identity drift: {query}:"
                f" actual={actual_hits} expected={expected_normalized}"
            )
        print(f"{marker}={query}:{hits[0].docid}:{float(hits[0].score):.6f}")
        query_results.append({"query": query, "top_k": [{"docid": docid, "score": score} for docid, score in actual_hits]})
    fingerprint = {
        "schema": "seed3090.webshop-index-semantic-fingerprint.v1",
        "catalog": {
            "path": str(args.catalog.resolve()),
            "bytes": args.catalog.stat().st_size,
            "sha256": sha256_file(args.catalog),
            "document_count": len(ordered_product_ids),
            "ordered_product_ids_sha256": hashlib.sha256(canonical_bytes(ordered_product_ids)).hexdigest(),
        },
        "resources": resource_identity(args.resources),
        "index": {
            **index_tree_identity(args.index.resolve()),
            "document_count": int(searcher.num_docs),
            "ordered_product_ids_sha256": hashlib.sha256(canonical_bytes(indexed_product_ids)).hexdigest(),
        },
        "query_panel": query_results,
    }
    fingerprint["behavior_fingerprint_sha256"] = hashlib.sha256(canonical_bytes({"index": fingerprint["index"], "query_panel": query_results})).hexdigest()
    print(f"INDEXES_1K_BEHAVIOR_FINGERPRINT={fingerprint['behavior_fingerprint_sha256']}")
    if args.fingerprint_out:
        args.fingerprint_out.parent.mkdir(parents=True, exist_ok=True)
        args.fingerprint_out.write_text(json.dumps(fingerprint, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("GPU_WORKLOAD_LAUNCHED=0")
    print("INDEXES_1K_SEMANTIC_VERIFY_PASS=1")
    print("INDEXES_1K_VERIFY_PASS=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
