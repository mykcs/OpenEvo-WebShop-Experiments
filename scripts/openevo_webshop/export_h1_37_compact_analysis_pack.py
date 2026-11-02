#!/usr/bin/env python3
"""Derive a secret-free H1.37 analysis pack from server-only episode JSON."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "openevo.h1.37-compact-analysis-pack.v1"


def normalize_action(action: object) -> str:
    """Keep only action class, never benchmark prompt/completion text."""
    value = str(action or "").strip().lower()
    if "buy now" in value or "buy_now" in value:
        return "buy_now"
    if "next page" in value or "next_page" in value:
        return "next_page"
    if value.startswith("search"):
        return "search"
    if value.startswith("click"):
        return "click"
    if value.startswith("back"):
        return "back"
    if value.startswith("prev"):
        return "prev_page"
    return "other"


def compact_row(path: Path, root: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    receipt = data
    episode = data["episode"]
    trace = episode.get("trace", [])
    actions = [normalize_action(item.get("action")) for item in trace]
    parser_failure = bool(receipt.get("parser_or_fallback_failure")) or any(
        bool(item.get("parser_failure")) for item in trace
    )
    valid = bool(receipt.get("identity_ok")) and bool(receipt.get("trace_complete")) and not parser_failure and not receipt.get("error")
    if receipt.get("error"):
        terminal_state = "error"
    elif episode.get("won"):
        terminal_state = "won"
    elif parser_failure:
        terminal_state = "parser_failure"
    else:
        terminal_state = "completed_without_purchase"
    return {
        "task_id": receipt.get("task_id"),
        "arm": data.get("arm"),
        "seed": receipt.get("rollout_seed"),
        "score": receipt.get("final_task_score"),
        "valid": valid,
        "parser_failure": parser_failure,
        "step_count": episode.get("steps"),
        "normalized_action_sequence": ">".join(actions),
        "terminal_state": terminal_state,
        "buy_now": "buy_now" in actions,
        "next_page_count": actions.count("next_page"),
        "protocol_fingerprint": receipt.get("protocol_fingerprint"),
        "wall_time_seconds": data.get("attempt_wall_time_seconds"),
        "source_path": path.relative_to(root).as_posix(),
    }


def exemplars(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    positive = [row for row in rows if row["valid"] and float(row["score"] or 0) > 0][:4]
    negative = [row for row in rows if row["valid"] and float(row["score"] or 0) == 0][:6]
    parser = [row for row in rows if row["parser_failure"]][:1]
    by_key: dict[tuple[object, object], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(row["task_id"], row["seed"])].append(row)
    ties = [pair[0] for pair in by_key.values() if len(pair) == 2 and pair[0]["score"] == pair[1]["score"]][:2]
    return {"positive": positive, "negative": negative, "parser_failure": parser, "paired_score_ties": ties}


def build_pack(source: Path) -> dict[str, Any]:
    source = source.expanduser().resolve()
    rows = [compact_row(path, source) for path in sorted(source.glob("evaluation/*/episodes/*.json"))]
    return {
        "schema": SCHEMA,
        "source_root": str(source),
        "record_count": len(rows),
        "privacy_boundary": "No raw prompts, goals, completions, or full traces; action classes only.",
        "rows": rows,
        "exemplars": exemplars(rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    rendered = json.dumps(build_pack(args.source), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
