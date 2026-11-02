#!/usr/bin/env python3
"""Build manifest-bound WebShop parquet inputs for the SEED trainer.

The SEED environment receives the same goal indices through its source
overlay.  These parquet rows carry the corresponding task text and identity
so the trainer's data contract cannot silently fall back to Geometry3K
placeholder rows.  This is a CPU/control-plane operation only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

try:
    from scripts.openevo_webshop.build_h1_30_training_manifest import validate_manifest
    from scripts.openevo_webshop.webshop_task_manifest import canonical_bytes
except ModuleNotFoundError:
    from build_h1_30_training_manifest import validate_manifest  # type: ignore
    from webshop_task_manifest import canonical_bytes  # type: ignore


SCHEMA = "openevo.webshop-h1.30-seed-parquet.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(document)
    observed = document["manifest_sha256"]
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError(f"manifest digest mismatch: expected {expected_sha256}, observed {observed}")
    return document


def build_rows(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    task_rows = manifest["train_tasks"] if split == "train" else manifest["validation_tasks"]
    expected_split = "h1.30-training" if split == "train" else "h1.30-validation"
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for expected_index, task in enumerate(task_rows):
        goal_idx = task["goal_idx"]
        description = str(task["task_description"]).strip()
        if task.get("sample_index") != expected_index or task.get("split") != expected_split:
            raise ValueError(f"manifest sample/split mismatch at {split}[{expected_index}]")
        if goal_idx in seen:
            raise ValueError(f"duplicate goal_idx in {split}: {goal_idx}")
        seen.add(goal_idx)
        description_sha256 = hashlib.sha256(description.encode("utf-8")).hexdigest()
        if task.get("task_description_sha256") != description_sha256:
            raise ValueError(f"task description digest mismatch for goal_idx {goal_idx}")
        rows.append(
            {
                "data_source": "webshop",
                "prompt": [{"role": "user", "content": description}],
                "ability": "agent",
                "task_id": task["task_id"],
                "goal_idx": goal_idx,
                "task_description": description,
                "extra_info": {
                    "split": expected_split,
                    "index": expected_index,
                    "task_id": task["task_id"],
                    "goal_idx": goal_idx,
                    "task_description_sha256": description_sha256,
                },
            }
        )
    return rows


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import datasets
    except ImportError as exc:  # pragma: no cover - exercised in runtime image
        raise RuntimeError("datasets is required to write SEED parquet inputs") from exc
    datasets.Dataset.from_list(rows).to_parquet(str(path))


def build_inputs(manifest_path: Path, output_dir: Path, expected_manifest_sha256: str | None = None) -> dict[str, Any]:
    manifest = load_manifest(manifest_path, expected_manifest_sha256)
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    train_rows = build_rows(manifest, "train")
    validation_rows = build_rows(manifest, "validation")
    train_path = output_dir / "train.parquet"
    validation_path = output_dir / "val.parquet"
    _write_parquet(train_path, train_rows)
    _write_parquet(validation_path, validation_rows)
    receipt = {
        "schema": SCHEMA,
        "status": "prepared-not-authorized",
        "formal_task_consumption_allowed": False,
        "gpu_workload_launched": False,
        "manifest": str(manifest_path.expanduser().resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "train_goal_indices": [row["goal_idx"] for row in train_rows],
        "validation_goal_indices": [row["goal_idx"] for row in validation_rows],
        "row_schema": "verl-agent.RLHFDataset.v1",
        "files": {
            "train": {"path": str(train_path), "rows": len(train_rows), "sha256": sha256_file(train_path)},
            "validation": {"path": str(validation_path), "rows": len(validation_rows), "sha256": sha256_file(validation_path)},
        },
        "rows_sha256": hashlib.sha256(canonical_bytes({"train": train_rows, "validation": validation_rows})).hexdigest(),
    }
    receipt_path = output_dir / "h1.30-seed-parquet-receipt.json"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_dir, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(receipt, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(receipt_path)
    receipt_path.chmod(0o600)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    print(json.dumps(build_inputs(args.manifest, args.output_dir, args.expected_manifest_sha256), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
