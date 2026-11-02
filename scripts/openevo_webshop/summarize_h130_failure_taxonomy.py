#!/usr/bin/env python3
"""Summarize train-only H1.30 capability failures without changing denominators."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        rows.append(row)
    return rows


def classify(row: dict[str, Any]) -> str:
    if row.get("parser_or_fallback_failure"):
        return "parser-or-fallback"
    if row.get("error"):
        return "runtime-error"
    if row.get("qualified_positive"):
        return "qualified-positive"
    episode = row.get("episode") if isinstance(row.get("episode"), dict) else {}
    trace = episode.get("trace") if isinstance(episode.get("trace"), list) else []
    actions = [str(step.get("action") or "").lower() for step in trace if isinstance(step, dict)]
    next_count = sum("next" in action for action in actions)
    if next_count >= 5 or (len(actions) >= 4 and all("next" in action for action in actions[-4:])):
        return "navigation-next-loop"
    if any(actions[index] == actions[index - 1] for index in range(1, len(actions))):
        return "repeated-action-loop"
    if any("back" in action or "prev" in action for action in actions):
        return "backtrack-without-success"
    return "zero-reward-other"


def summarize(paths: list[Path], output: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    source_files = []
    for path in paths:
        resolved = path.expanduser().resolve()
        source_files.append({"path": str(resolved), "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest()})
        rows.extend(_load(resolved))
    tasks = sorted({str(row.get("task_id")) for row in rows})
    result = {
        "schema": "openevo.webshop-h1.30-failure-taxonomy.v1",
        "diagnostic_only": True,
        "attempt_count": len(rows),
        "categories": dict(sorted(collections.Counter(classify(row) for row in rows).items())),
        "task_categories": {
            task: dict(sorted(collections.Counter(classify(row) for row in rows if str(row.get("task_id")) == task).items()))
            for task in tasks
        },
        "qualified_positive_task_ids": sorted({str(row.get("task_id")) for row in rows if classify(row) == "qualified-positive"}),
        "rules": {
            "navigation-next-loop": "at least five actions containing next, or the final four actions all contain next",
            "repeated-action-loop": "at least one pair of consecutive identical actions after navigation classification",
            "backtrack-without-success": "a back/prev action after the previous rules",
            "parser-or-fallback": "parser_or_fallback_failure is true",
            "runtime-error": "error is truthy",
        },
        "source_files": source_files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.attempts, args.output.resolve()), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
