#!/usr/bin/env python3
"""Generic deterministic Harness201 carrier safety projection.

Qualification-only and reward/outcome blind. This makes no model/API calls and
never invents a learned strategy. It keeps only already-conforming safe learned
bullets, drops unsafe/nonconforming bullets, and adds only fixed harness-boundary
placeholders when a section would otherwise be empty. Optional source carriers
are used only to derive quoted task literals that must not survive compilation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from scripts.openevo_webshop.ceiling1_harness2 import (
    ACTION_LIKE_ABSTRACTION_RE,
    DEFAULT_UI_LITERALS,
    EXECUTABLE_ACTION_RE,
    PRODUCT_ID_RE,
    lint_carriers,
)
from scripts.openevo_webshop.ceiling1_harness2_runtime import markdown_sections, required_safe_structure
from scripts.openevo_webshop.ceiling1_harness201_carrier_contract import validate_bundle

STATES = ("Search State", "Results State", "Product State", "Any State")
QUOTED_RE = re.compile(r'''["']([^"']{2,80})["']''')
ABSTRACTED_UI_TERMS = (
    "currently available purchase action",
    "currently available pagination action",
    "currently available return-to-search action",
    "currently available reverse-pagination action",
)
MEMORY_IMPERATIVE_RE = re.compile(
    r"(?i)^- Knowledge:\s*(?:do not|don't|start|apply|read|stop|refine|check|ensure|confirm|verify|return|click|select|buy|purchase)\b"
)
FIXED_AUTHORITY = (
    "- Authority: Current task, current observation, and current Allowed actions are the sole authority for current-state facts and legal actions.",
    "- Authority: Historical carriers may guide strategy only when consistent with the current environment.",
)
FIXED_COORDINATION = (
    "- Coordination: Use Skill for conditional procedures and Memory for reusable knowledge; resolve conflicts in favor of the current environment.",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    tmp.replace(path)


def derive_forbidden(paths: list[Path]) -> set[str]:
    values: set[str] = set()
    for path in paths:
        for value in QUOTED_RE.findall(path.read_text()):
            value = " ".join(value.strip().split()).lower()
            if len(value) >= 2:
                values.add(value)
    return values


def unsafe(text: str, forbidden: set[str]) -> bool:
    low = text.lower()
    return bool(
        EXECUTABLE_ACTION_RE.search(text)
        or PRODUCT_ID_RE.search(text)
        or ACTION_LIKE_ABSTRACTION_RE.search(text)
        or "e.g." in low
        or any(term in low for term in ABSTRACTED_UI_TERMS)
        or any(str(x).lower() in low for x in DEFAULT_UI_LITERALS)
        or any(value in low for value in forbidden)
    )


def bullets(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip().startswith("- ")]


def finalize_memory(text: str, forbidden: set[str]) -> tuple[str, list[dict]]:
    kept: list[str] = []
    trace: list[dict] = []
    for line in bullets(text):
        imperative = bool(MEMORY_IMPERATIVE_RE.search(line))
        keep = line.startswith("- Knowledge:") and not imperative and not unsafe(line, forbidden)
        reason = "strict-safe-declarative-knowledge" if keep else ("imperative-memory-bullet" if imperative else "nonconforming-action-like-or-source-literal")
        trace.append({"source": line, "action": "keep" if keep else "drop", "reason": reason})
        if keep:
            kept.append(line)
    result = "# Memory\n## Reusable Knowledge\n"
    if kept:
        result += "\n".join(kept) + "\n"
    return result, trace


def finalize_skill(text: str, forbidden: set[str]) -> tuple[str, list[dict]]:
    sections = markdown_sections(text)
    trace: list[dict] = []
    out = ["# Skill Bundle"]
    for heading in STATES:
        out.append(f"## {heading}")
        kept: list[str] = []
        for line in bullets(sections.get(heading, "")):
            keep = line.startswith("- When ") and not unsafe(line, forbidden)
            trace.append({"section": heading, "source": line, "action": "keep" if keep else "drop", "reason": "strict-safe-conditional" if keep else "nonconditional-action-like-or-source-literal"})
            if keep:
                kept.append(line)
        out.extend(kept or ["- No additional guidance."])
        out.append("")
    return "\n".join(out).rstrip() + "\n", trace


def finalize_agent(text: str, forbidden: set[str]) -> tuple[str, list[dict]]:
    sections = markdown_sections(text)
    trace: list[dict] = []
    auth: list[str] = []
    coord: list[str] = []
    for heading, prefix, target in (("Authority", "- Authority:", auth), ("Coordination", "- Coordination:", coord)):
        for line in bullets(sections.get(heading, "")):
            keep = line.startswith(prefix) and not unsafe(line, forbidden)
            trace.append({"section": heading, "source": line, "action": "keep" if keep else "drop", "reason": "strict-safe-agent-role" if keep else "nonconforming-procedural-or-source-literal"})
            if keep:
                target.append(line)
    if not auth:
        auth = list(FIXED_AUTHORITY)
        trace.append({"section": "Authority", "action": "fixed-harness-boundary", "source": None})
    if not coord:
        coord = list(FIXED_COORDINATION)
        trace.append({"section": "Coordination", "action": "fixed-harness-boundary", "source": None})
    return "# Agent System\n## Authority\n" + "\n".join(auth) + "\n\n## Coordination\n" + "\n".join(coord) + "\n", trace


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-memory", type=Path)
    p.add_argument("--source-skill", type=Path)
    p.add_argument("--source-agent", type=Path)
    a = p.parse_args()
    source_paths = [x for x in (a.source_memory, a.source_skill, a.source_agent) if x is not None]
    if source_paths and len(source_paths) != 3:
        raise SystemExit("provide all three source carriers or none")
    forbidden = derive_forbidden(source_paths)

    src = a.input.resolve()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    receipt = src / "finalize-receipt.json"
    source = json.loads(receipt.read_text())
    if source.get("schema") != "openevo.ceiling1-harness2-carrier-finalize-receipt.v1":
        raise SystemExit("source Harness2 finalize receipt schema mismatch")
    validation = source.get("validation") or {}
    if (validation.get("lint") or {}).get("status") != "PASS" or validation.get("structural_errors"):
        raise SystemExit("source Harness2 output is not structurally/lint safe enough for deterministic role filtering")

    memory, mt = finalize_memory((src / "memory.md").read_text(), forbidden)
    skill, st = finalize_skill((src / "skill.md").read_text(), forbidden)
    agent, at = finalize_agent((src / "agent.md").read_text(), forbidden)
    (out / "memory.md").write_text(memory)
    (out / "skill.md").write_text(skill)
    (out / "agent.md").write_text(agent)

    strict = lint_carriers(memory=memory, skill=skill, agent=agent, forbid_action_like_abstractions=True)
    structural = required_safe_structure(memory, skill, agent)
    evolution_contract = validate_bundle(memory, skill, agent)
    status = "PASS" if strict.get("status") == "PASS" and not structural and evolution_contract["status"] == "PASS" else "FAIL"
    result = {
        "schema": "openevo.ceiling1-harness2.0.1-carrier-finalize-receipt.v2",
        "status": status,
        "qualification_only": True,
        "model_calls": 0,
        "external_teacher_calls": 0,
        "formal_task_consumption": 0,
        "final_panel_access": 0,
        "source_harness2_finalize_receipt": str(receipt),
        "source_harness2_finalize_receipt_sha256": sha(receipt),
        "source_harness2_status": source.get("status"),
        "source_carriers": [{"path": str(x.resolve()), "sha256": sha(x)} for x in source_paths],
        "source_quoted_literal_count": len(forbidden),
        "transform_policy": {
            "reward_blind": True,
            "outcome_blind": True,
            "semantic_rewrite": False,
            "keep_only_already_conforming_declarative_or_conditional_learned_bullets": True,
            "drop_imperative_nonconforming_action_like_or_source_literal_learned_bullets": True,
            "fixed_harness_boundary_placeholders_only": True,
        },
        "transform_trace": {"memory": mt, "skill": st, "agent": at},
        "outputs": {
            "memory": {"path": str(out / "memory.md"), "sha256": sha(out / "memory.md")},
            "skill": {"path": str(out / "skill.md"), "sha256": sha(out / "skill.md")},
            "agent": {"path": str(out / "agent.md"), "sha256": sha(out / "agent.md")},
        },
        "validation": {"lint": strict, "structural_errors": structural, "evolution_contract": evolution_contract},
    }
    atomic_json(out / "finalize-receipt.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
