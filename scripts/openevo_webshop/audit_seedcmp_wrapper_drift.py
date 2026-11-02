#!/usr/bin/env python3
"""Audit Track A raw WebShop completions for action-wrapper drift.

This is a read-only evidence audit. It does not re-run inference and it does not
reinterpret task scores. The same parser-compatible projection used by the
formal Track A measurement is re-applied to each saved raw completion so that
model-output wrapper drift is reported separately from fail-closed invalid
steps and from episode-level measurement invalidity.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any

try:
    from scripts.openevo_webshop.webshop_comparison_contract import (
        parse_webshop_action,
        repair_action_command,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scripts.openevo_webshop":
        raise
    from webshop_comparison_contract import (  # type: ignore[no-redef]
        parse_webshop_action,
        repair_action_command,
    )


SCHEMA = "openevo.seedcmp-wrapper-audit.v1"
CAMPAIGN_ID = "20260826-0630-seed-webshop-public-code-reproduction"
CONTRACT = "seed-eval-v2-parser-compat"
EVIDENCE_COMMIT = "f80ae1816384bb7e8e82d193b22644e17f561f19"
DEFAULT_ROOT = Path(
    "docs/evidence/seed-webshop-public-code-reproduction-v1/server/evaluation/parser-compat"
)
CMD_RE = re.compile(r"(?:search|click|select|buy)\[[^\]\n]*\]", re.IGNORECASE)
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _after_think(raw: str) -> str:
    lowered = raw.lower()
    close = lowered.find("</think>")
    if close < 0:
        return raw.strip()
    return raw[close + len("</think>") :].strip()


def wrapper_syntax(raw: str, *, strict_valid: bool) -> str:
    """Classify surface wrapper syntax without changing parser semantics."""

    if strict_valid:
        return "strict_action_xml"
    region = _after_think(raw)
    lowered = region.lower().lstrip()
    if lowered.startswith("[action]"):
        return "square_action"
    if lowered.startswith("[action>"):
        return "square_action_malformed_open"
    if lowered.startswith("<action>") and "</action>" not in lowered:
        return "xml_action_missing_close"
    if lowered.startswith("<action"):
        return "xml_action_malformed"
    if lowered.startswith("=") and CMD_RE.search(lowered[1:]):
        return "equals_prefixed_command"
    if CMD_RE.match(lowered):
        return "command_only"
    if lowered.startswith("<") and CMD_RE.search(lowered):
        return "angle_wrapped_command"
    if CMD_RE.search(lowered):
        return "other_repairable_surface"
    return "no_action_command_surface"


def unrecoverable_reason(raw: str) -> str:
    """Mirror the v3 recovery guard ordering for a failed raw completion."""

    if "<think>" not in raw or "</think>" not in raw:
        return "missing_think_contract"
    if CJK_RE.search(raw):
        return "cjk_contract_rejection"
    if not CMD_RE.search(_after_think(raw)):
        return "no_action_command_after_think"
    return "other_parser_rejection"


def _episode_files(episodes_dir: Path) -> list[Path]:
    return sorted(path for path in episodes_dir.glob("*.json") if path.is_file())


def audit_arm(root: Path, arm: str) -> dict[str, Any]:
    arm_root = root / arm
    episodes_dir = arm_root / "episodes"
    summary_path = arm_root / "summary.json"
    files = _episode_files(episodes_dir)
    if not files:
        raise ValueError(f"no raw episode JSON files found for {arm}: {episodes_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    wrapper_steps: Counter[str] = Counter()
    wrapper_episodes: dict[str, set[str]] = defaultdict(set)
    parser_status_steps: Counter[str] = Counter()
    parser_status_episodes: dict[str, set[str]] = defaultdict(set)
    unrecoverable_reasons: Counter[str] = Counter()
    unrecoverable_reason_episodes: dict[str, set[str]] = defaultdict(set)
    environment_invalid_steps = 0
    recorded_invalid_action_steps = 0
    formal_invalid_episodes = 0
    runtime_invalid_episodes = 0
    total_steps = 0

    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("campaign_id") != CAMPAIGN_ID:
            raise ValueError(f"campaign mismatch in {path}")
        if payload.get("contract") != CONTRACT:
            raise ValueError(f"contract mismatch in {path}")
        if payload.get("arm") != arm:
            raise ValueError(f"arm mismatch in {path}")
        episode = payload.get("episode")
        if not isinstance(episode, dict):
            raise ValueError(f"missing episode object in {path}")
        trace = episode.get("trace")
        if not isinstance(trace, list):
            raise ValueError(f"missing trace array in {path}")

        recorded_invalid_action_steps += int(payload.get("invalid_action_steps", 0) or 0)
        if not bool(payload.get("valid_episode")):
            formal_invalid_episodes += 1
        if bool(payload.get("runtime_invalid")):
            runtime_invalid_episodes += 1

        episode_id = path.name
        for step in trace:
            if not isinstance(step, dict):
                raise ValueError(f"non-object trace step in {path}")
            total_steps += 1
            raw = str(step.get("raw_completion") or "")
            strict_action, strict_valid = parse_webshop_action(raw, require_think=True)
            syntax = wrapper_syntax(raw, strict_valid=strict_valid)
            wrapper_steps[syntax] += 1
            wrapper_episodes[syntax].add(episode_id)

            if strict_valid:
                parser_status = "strict"
                recovered_action = strict_action
            else:
                _repaired_text, recovered_action, repaired = repair_action_command(raw)
                parser_status = "recovered" if repaired else "unrecoverable_fail_closed"
                if not repaired:
                    reason = unrecoverable_reason(raw)
                    unrecoverable_reasons[reason] += 1
                    unrecoverable_reason_episodes[reason].add(episode_id)
            parser_status_steps[parser_status] += 1
            parser_status_episodes[parser_status].add(episode_id)

            recorded_action = str(step.get("action") or "").strip().lower()
            if parser_status != "unrecoverable_fail_closed" and recovered_action != recorded_action:
                raise ValueError(
                    f"recomputed parser action mismatch in {path}: "
                    f"{recovered_action!r} != {recorded_action!r}"
                )
            if not bool(step.get("is_action_valid")):
                environment_invalid_steps += 1

    formal_summary_invalid = int(summary.get("invalid_count", -1))
    if formal_summary_invalid < 0:
        raise ValueError(f"summary missing invalid_count for {arm}")
    if int(summary.get("attempt_count", -1)) != len(files):
        raise ValueError(f"summary/raw episode count mismatch for {arm}")

    known_drift = {
        key: int(wrapper_steps.get(key, 0))
        for key in (
            "square_action",
            "square_action_malformed_open",
            "xml_action_missing_close",
            "xml_action_malformed",
            "equals_prefixed_command",
            "command_only",
            "angle_wrapped_command",
        )
    }
    unknown_wrapper_steps = int(wrapper_steps.get("other_repairable_surface", 0))
    no_command_surface_steps = int(wrapper_steps.get("no_action_command_surface", 0))
    unrecoverable_steps = int(parser_status_steps.get("unrecoverable_fail_closed", 0))
    fail_closed_alignment = (
        unrecoverable_steps == recorded_invalid_action_steps == environment_invalid_steps
    )

    return {
        "arm": arm,
        "episode_count": len(files),
        "step_count": total_steps,
        "wrapper_step_counts": dict(sorted(wrapper_steps.items())),
        "wrapper_episode_counts": {
            key: len(value) for key, value in sorted(wrapper_episodes.items())
        },
        "known_drift_step_counts": known_drift,
        "unknown_wrapper_steps": unknown_wrapper_steps,
        "no_action_command_surface_steps": no_command_surface_steps,
        "parser_step_counts": dict(sorted(parser_status_steps.items())),
        "parser_episode_counts": {
            key: len(value) for key, value in sorted(parser_status_episodes.items())
        },
        "unrecoverable_reason_step_counts": dict(sorted(unrecoverable_reasons.items())),
        "unrecoverable_reason_episode_counts": {
            key: len(value) for key, value in sorted(unrecoverable_reason_episodes.items())
        },
        "recomputed_unrecoverable_fail_closed_episode_count": len(
            parser_status_episodes.get("unrecoverable_fail_closed", set())
        ),
        "recorded_invalid_action_steps": recorded_invalid_action_steps,
        "environment_invalid_steps": environment_invalid_steps,
        "unrecoverable_fail_closed_alignment": fail_closed_alignment,
        "formal_invalid_episode_count_from_raw": formal_invalid_episodes,
        "runtime_invalid_episode_count": runtime_invalid_episodes,
        "formal_summary_invalid_count": formal_summary_invalid,
    }


def build_report(root: Path) -> dict[str, Any]:
    arms = {arm: audit_arm(root, arm) for arm in ("base", "sd-lora")}
    closeout_path = root.parents[2] / "CLOSEOUT.json"
    closeout = json.loads(closeout_path.read_text(encoding="utf-8"))
    closeout_parser_invalid = int(closeout["formal_episode_matrix"]["parser_invalid_total"])

    totals = {
        "episodes": sum(value["episode_count"] for value in arms.values()),
        "steps": sum(value["step_count"] for value in arms.values()),
        "strict_steps": sum(value["parser_step_counts"].get("strict", 0) for value in arms.values()),
        "recovered_steps": sum(value["parser_step_counts"].get("recovered", 0) for value in arms.values()),
        "unrecoverable_fail_closed_steps": sum(
            value["parser_step_counts"].get("unrecoverable_fail_closed", 0)
            for value in arms.values()
        ),
        "recorded_invalid_action_steps": sum(value["recorded_invalid_action_steps"] for value in arms.values()),
        "environment_invalid_steps": sum(value["environment_invalid_steps"] for value in arms.values()),
        "formal_invalid_episode_count_from_raw": sum(
            value["formal_invalid_episode_count_from_raw"] for value in arms.values()
        ),
        "formal_summary_invalid_count": sum(value["formal_summary_invalid_count"] for value in arms.values()),
        "closeout_parser_invalid_total": closeout_parser_invalid,
    }
    drift_observed = totals["recovered_steps"] > 0
    fail_closed_alignment = all(
        value["unrecoverable_fail_closed_alignment"] for value in arms.values()
    )
    measurement_polluted = (
        totals["formal_invalid_episode_count_from_raw"] > 0
        or totals["formal_summary_invalid_count"] > 0
        or closeout_parser_invalid > 0
        or not fail_closed_alignment
    )

    return {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "contract": CONTRACT,
        "source": {
            "evidence_commit": EVIDENCE_COMMIT,
            "root": str(root),
            "method": "saved raw_completion -> strict parser -> v3 action-command recovery",
        },
        "arms": arms,
        "totals": totals,
        "interpretation": {
            "wrapper_drift_observed": drift_observed,
            "unrecoverable_steps_fail_closed": fail_closed_alignment,
            "measurement_pollution_observed": measurement_polluted,
            "claim": (
                "Saved model outputs still exhibit action-wrapper drift. Recoverable "
                "commands are normalized by the frozen v3 compatibility projection; "
                "unrecoverable completions remain invalid environment actions rather than "
                "being silently substituted. The formal 256-episode measurement remains "
                "valid, so wrapper drift, fail-closed agent action failure, and measurement "
                "invalidity are distinct events."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()

    report = build_report(args.root)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if args.check:
        expected = json.loads(args.check.read_text(encoding="utf-8"))
        if expected != report:
            raise SystemExit("wrapper audit receipt does not match recomputed raw evidence")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
