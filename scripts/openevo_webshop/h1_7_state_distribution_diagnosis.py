#!/usr/bin/env python3
"""Offline trajectory diagnosis for the H1.5 seen-task and H1.6 held-out panels."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


ACTION_RE = re.compile(r"^(?P<kind>[a-z_]+)\[")
WORD_RE = re.compile(r"\S+")


def _episode_rows(run_root: Path, label: str) -> list[dict]:
    rows = []
    for path in sorted(run_root.glob("cells/compact__*__*/episodes/*.json")):
        cell = path.parts[-3]
        parts = cell.split("__")
        if len(parts) != 3:
            raise ValueError(f"unexpected cell name: {cell}")
        _, partition, arm = parts
        payload = json.loads(path.read_text())
        episode = payload["episode"]
        trace = episode.get("trace") or []
        actions = [str(step.get("action") or "") for step in trace]
        kinds = [(_action_kind(action) or "unknown") for action in actions]
        parser_failures = sum(bool(step.get("parser_failure")) for step in trace)
        action_invalid = sum(not bool(step.get("action_valid")) for step in trace)
        complete = bool(trace) and len(trace) == int(episode.get("steps") or 0)
        valid = not bool(episode.get("error")) and complete and not parser_failures and not action_invalid
        score = float(episode.get("task_score") or 0.0)
        qualified = valid and score > 0.0
        raw_words = sum(len(WORD_RE.findall(str(step.get("raw_completion") or ""))) for step in trace)
        unique_prefixes = len({"|".join(actions[:index]) for index in range(1, min(len(actions), 5) + 1)})
        repeated_transitions = sum(a == b for a, b in zip(actions, actions[1:]))
        rows.append(
            {
                "source": label,
                "cell": cell,
                "partition": partition,
                "arm": arm,
                "path": str(path),
                "task_id": f"webshop_{int(episode['episode']):06d}",
                "score": score,
                "valid": valid,
                "qualified_positive": qualified,
                "steps": len(trace),
                "generation_words": raw_words,
                "first_action": kinds[0] if kinds else "none",
                "action_counts": dict(Counter(kinds)),
                "search_count": kinds.count("search"),
                "click_count": kinds.count("click"),
                "buy_count": kinds.count("buy"),
                "select_count": kinds.count("select"),
                "buy_signal_count": sum("buy now" in action.lower() for action in actions),
                "product_click_count": sum(
                    kind == "click" and target not in {"back to search", "next >", "prev"}
                    for action, kind in zip(actions, kinds)
                    for target in [action.lower().split("[", 1)[-1].rstrip("]")]
                ),
                "back_to_search_count": sum("back to search" in action.lower() for action in actions),
                "next_count": sum("next >" in action.lower() for action in actions),
                "unique_prefixes_5": unique_prefixes,
                "repeated_transitions": repeated_transitions,
                "parser_failures": parser_failures,
                "action_invalid": action_invalid,
            }
        )
    if not rows:
        raise ValueError(f"no episode files found under {run_root}")
    return rows


def _action_kind(action: str) -> str | None:
    match = ACTION_RE.match(action.strip().lower())
    return match.group("kind") if match else None


def _summary(rows: list[dict]) -> dict:
    def avg(key: str, selected: list[dict] | None = None) -> float:
        values = [float(row[key]) for row in (selected or rows)]
        return round(mean(values), 6) if values else 0.0

    first_actions = Counter(row["first_action"] for row in rows)
    return {
        "episodes": len(rows),
        "valid": sum(row["valid"] for row in rows),
        "qualified_positive": sum(row["qualified_positive"] for row in rows),
        "mean_task_score_all": avg("score"),
        "mean_task_score_valid": avg("score", [row for row in rows if row["valid"]]),
        "mean_steps": avg("steps"),
        "mean_generation_words": avg("generation_words"),
        "mean_search_count": avg("search_count"),
        "mean_click_count": avg("click_count"),
        "mean_buy_count": avg("buy_count"),
        "mean_select_count": avg("select_count"),
        "mean_buy_signal_count": avg("buy_signal_count"),
        "mean_product_click_count": avg("product_click_count"),
        "mean_back_to_search_count": avg("back_to_search_count"),
        "mean_next_count": avg("next_count"),
        "mean_unique_prefixes_5": avg("unique_prefixes_5"),
        "mean_repeated_transitions": avg("repeated_transitions"),
        "first_action_counts": dict(sorted(first_actions.items())),
        "task_ids": sorted({row["task_id"] for row in rows}),
    }


def diagnose(h1_5_root: Path, h1_6_root: Path) -> dict:
    rows = _episode_rows(h1_5_root, "h1.5_seen") + _episode_rows(h1_6_root, "h1.6_heldout")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["source"]].append(row)
        groups[f"{row['source']}__{row['arm']}"] .append(row)
        if row["qualified_positive"]:
            groups[f"{row['source']}__qualified_positive"].append(row)
        elif row["valid"]:
            groups[f"{row['source']}__valid_zero"].append(row)
    summaries = {name: _summary(group) for name, group in sorted(groups.items())}
    seen_positive = groups["h1.5_seen__qualified_positive"]
    heldout_valid = groups["h1.6_heldout__valid_zero"]
    return {
        "schema": "seed3090.webshop-h1.7-state-distribution-diagnosis.v1",
        "inputs": {"h1.5_root": str(h1_5_root), "h1.6_root": str(h1_6_root)},
        "scope": {
            "h1.5": "seen four-task compact-v2 fresh-seed panel",
            "h1.6": "eight held-out tasks, six cells, compact-v2",
            "valid_definition": "no runtime error, complete trace, no parser failure, all actions valid",
            "qualified_positive_definition": "valid and task_score > 0",
        },
        "group_summaries": summaries,
        "primary_comparison": {
            "seen_qualified_positive_n": len(seen_positive),
            "heldout_valid_zero_n": len(heldout_valid),
            "seen_positive_minus_heldout_valid_zero": {
                key: round(mean(row[key] for row in seen_positive) - mean(row[key] for row in heldout_valid), 6)
                for key in (
                    "steps", "generation_words", "search_count", "click_count", "buy_count", "select_count",
                    "buy_signal_count", "product_click_count", "back_to_search_count", "next_count",
                    "unique_prefixes_5", "repeated_transitions",
                )
            },
        },
        "interpretation": "Held-out validity without held-out reward is consistent with a state-distribution/action-selection bottleneck; this offline diagnosis does not establish causality or justify D0.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h1-5-root", type=Path, required=True)
    parser.add_argument("--h1-6-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.h1_5_root, args.h1_6_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "groups": len(result["group_summaries"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
