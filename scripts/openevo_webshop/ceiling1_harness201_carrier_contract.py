#!/usr/bin/env python3
"""Typed runtime/evolution contract for Harness 2.0.1 carriers.

Pure validation only: no model calls, reward access, or task selection.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable, Iterable

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
PRODUCT_ID_RE = re.compile(r"(?i)\bB0[A-Z0-9]{8}\b")
OPTION_CODE_RE = re.compile(r"(?i)\b[a-z]\d{1,4}(?:[-_][a-z0-9]+)+\b")
EXECUTABLE_ACTION_RE = re.compile(r"(?i)\b(?:search|click|select|buy)\[[^\]\n]+\]")
ACTION_SHAPED_RE = re.compile(
    r"(?i)\b(?:click|buy|search|select|open|browse|navigate|pagination|paginate|"
    r"return[- ]to[- ]search|next_page|next page|previous_page|back to search|buy now)\b|<\s*prev\s*>|next\s*>"
)
UI_LITERALS = ("Next >", "Back to Search", "Buy Now", "< Prev")
CAPS = {"text_memory": 192, "skill_bundle": 160, "agent_system": 96}
SCOPE_MANIFEST_SCHEMA = "openevo.carrier-scope-manifest.v1"


def _nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in str(text).splitlines() if line.strip()]


def _headings(text: str) -> list[str]:
    return [line for line in _nonempty_lines(text) if line.startswith("#")]


def _section_body(lines: list[str], heading: str) -> list[str]:
    try:
        i = lines.index(heading)
    except ValueError:
        return []
    out = []
    for line in lines[i + 1 :]:
        if line.startswith("#"):
            break
        out.append(line)
    return out


def lexical_violations(text: str) -> list[str]:
    violations: list[str] = []
    body_text = "\n".join(line for line in str(text).splitlines() if not line.strip().startswith("#"))
    if CJK_RE.search(body_text):
        violations.append("contains CJK text")
    if PRODUCT_ID_RE.search(body_text):
        violations.append("contains product-id literal")
    if OPTION_CODE_RE.search(body_text):
        violations.append("contains option-code literal")
    if EXECUTABLE_ACTION_RE.search(body_text):
        violations.append("contains executable action")
    if ACTION_SHAPED_RE.search(body_text):
        violations.append("contains action-shaped vocabulary")
    low = body_text.lower()
    for literal in UI_LITERALS:
        if literal.lower() in low:
            violations.append(f"contains UI literal {literal!r}")
    return sorted(set(violations))


def _normalize_scope_text(text: str) -> str:
    from scripts.openevo_webshop.carrier_scope_manifest_v1 import normalize
    return normalize(text)


def _contains_scope_value(text: str, value: str) -> bool:
    haystack = _normalize_scope_text(text)
    needle = _normalize_scope_text(value)
    if not needle:
        return False
    left = r"(?<![a-z0-9])" if needle[0].isalnum() else ""
    right = r"(?![a-z0-9])" if needle[-1].isalnum() else ""
    return re.search(left + re.escape(needle) + right, haystack) is not None


def _scope_units(text: str) -> list[str]:
    body = "\n".join(line for line in str(text).splitlines() if not line.strip().startswith("#"))
    return [unit.strip() for unit in re.split(r"\n+|(?<=[.!?;])\s+", body) if unit.strip()]


def task_local_scope_violations(
    text: str, forbidden_literals: Iterable[str] = (), *,
    typed_value_atoms: Iterable[dict] = (), combination_threshold: int = 2,
) -> list[str]:
    """Reject exact task-local literals and same-task value combinations."""
    body_text = "\n".join(line for line in str(text).splitlines() if not line.strip().startswith("#"))
    violations: list[str] = []
    for literal in forbidden_literals:
        normalized_literal = _normalize_scope_text(literal)
        if normalized_literal and _contains_scope_value(body_text, normalized_literal):
            digest = hashlib.sha256(normalized_literal.encode("utf-8")).hexdigest()
            violations.append(f"contains task-local literal sha256={digest}")

    by_task: dict[str, list[str]] = {}
    for atom in typed_value_atoms:
        if not isinstance(atom, dict):
            continue
        source = str(atom.get("source_instruction_sha256") or "")
        value = _normalize_scope_text(atom.get("value") or "")
        if source and value:
            by_task.setdefault(source, []).append(value)
    threshold = int(combination_threshold)
    if threshold >= 2:
        for unit in _scope_units(text):
            for source, values in by_task.items():
                matched = sorted({value for value in values if _contains_scope_value(unit, value)})
                if len(matched) >= threshold:
                    digest = hashlib.sha256((source + "\0" + "\0".join(matched)).encode("utf-8")).hexdigest()
                    violations.append(f"contains task-local combination sha256={digest}")
    return sorted(set(violations))


def load_scope_manifest(path: Path, *, expected_round_index: int | None = None) -> dict:
    """Load and identity-check the frozen deterministic scope manifest."""
    from scripts.openevo_webshop.carrier_scope_manifest_v1 import POLICY_ID, canonical_sha
    payload = json.loads(path.read_text())
    if payload.get("schema") != SCOPE_MANIFEST_SCHEMA:
        raise ValueError("carrier scope manifest schema mismatch")
    if payload.get("construction_policy_id") != POLICY_ID:
        raise ValueError("carrier scope manifest construction policy mismatch")
    if expected_round_index is not None and int(payload.get("round_index", -1)) != int(expected_round_index):
        raise ValueError("carrier scope manifest round mismatch")
    literals = payload.get("forbidden_literals")
    atoms = payload.get("typed_value_atoms")
    if not isinstance(literals, list) or any(not isinstance(x, str) or not x.strip() for x in literals):
        raise ValueError("carrier scope manifest forbidden_literals invalid")
    if not isinstance(atoms, list) or any(not isinstance(x, dict) for x in atoms):
        raise ValueError("carrier scope manifest typed_value_atoms invalid")
    normalized = [_normalize_scope_text(x) for x in literals]
    if normalized != sorted(set(normalized)):
        raise ValueError("carrier scope manifest literals must be normalized, unique, sorted")
    source_tasks = payload.get("source_tasks")
    if not isinstance(source_tasks, list) or int(payload.get("source_task_count", -1)) != len(source_tasks):
        raise ValueError("carrier scope manifest source task count mismatch")
    if int(payload.get("combination_threshold", -1)) != 2:
        raise ValueError("carrier scope manifest combination threshold mismatch")
    expected_identity = str(payload.get("manifest_identity_sha256") or "")
    identity_payload = {k: v for k, v in payload.items() if k != "manifest_identity_sha256"}
    if expected_identity != canonical_sha(identity_payload):
        raise ValueError("carrier scope manifest identity mismatch")
    if str(payload.get("source_task_digest_sha256") or "") != canonical_sha({"tasks": source_tasks}):
        raise ValueError("carrier scope manifest source task digest mismatch")
    return payload


def task_local_scope_violations_from_manifest(text: str, manifest: dict) -> list[str]:
    return task_local_scope_violations(
        text, manifest.get("forbidden_literals") or (),
        typed_value_atoms=manifest.get("typed_value_atoms") or (),
        combination_threshold=int(manifest.get("combination_threshold", 2)),
    )


def structural_violations(kind: str, text: str) -> list[str]:
    lines = _nonempty_lines(text)
    if not lines:
        return ["empty carrier"]
    errors: list[str] = []
    if kind == "text_memory":
        expected = ["# Memory", "## Reusable Knowledge"]
        if _headings(text) != expected:
            errors.append("memory headings must be exactly # Memory / ## Reusable Knowledge")
        body = _section_body(lines, "## Reusable Knowledge")
        bullets = [x for x in body if x.startswith("- ")]
        if not 1 <= len(bullets) <= 4 or len(body) != len(bullets):
            errors.append("memory must contain 1..4 reusable-knowledge bullets only")
    elif kind == "skill_bundle":
        expected = ["# Skill Bundle", "## Search State", "## Results State", "## Product State", "## Any State"]
        if _headings(text) != expected:
            errors.append("skill headings drift")
        search = _section_body(lines, "## Search State")
        if len(search) != 1 or not search[0].startswith("- "):
            errors.append("Search State must contain exactly one bullet")
        elif not re.match(r"^-\s+(?:When|If)\b", search[0]):
            errors.append("Search State rule must be conditional (When/If)")
        for h in ("## Results State", "## Product State", "## Any State"):
            if _section_body(lines, h) != ["- No additional guidance."]:
                errors.append(f"{h} must remain '- No additional guidance.'")
    elif kind == "agent_system":
        expected = ["# Agent System", "## Authority", "## Coordination"]
        if _headings(text) != expected:
            errors.append("agent headings must be exactly Authority / Coordination")
        authority = _section_body(lines, "## Authority")
        coordination = _section_body(lines, "## Coordination")
        if len(authority) != 1 or not authority[0].startswith("- Authority:"):
            errors.append("Authority must contain exactly one '- Authority:' bullet")
        if len(coordination) != 1 or not coordination[0].startswith("- Coordination:"):
            errors.append("Coordination must contain exactly one '- Coordination:' bullet")
    else:
        errors.append(f"unknown carrier kind {kind}")
    return errors


def validate_carrier(
    kind: str,
    text: str,
    *,
    token_counter: Callable[[str], int] | None = None,
    forbidden_literals: Iterable[str] | None = None,
    scope_manifest: dict | None = None,
) -> dict:
    if forbidden_literals is not None and scope_manifest is not None:
        raise ValueError("supply forbidden_literals or scope_manifest, not both")
    if scope_manifest is not None:
        scope = task_local_scope_violations_from_manifest(text, scope_manifest)
        scope_evaluated = True
    elif forbidden_literals is not None:
        scope = task_local_scope_violations(text, forbidden_literals)
        scope_evaluated = True
    else:
        scope = []
        scope_evaluated = False
    violations = structural_violations(kind, text) + lexical_violations(text) + scope
    tokens = int(token_counter(text)) if token_counter is not None else None
    cap = CAPS.get(kind)
    if tokens is not None and cap is not None and tokens > cap:
        violations.append(f"carrier token count {tokens} exceeds cap {cap}")
    return {
        "status": "PASS" if not violations else "FAIL",
        "kind": kind,
        "tokens": tokens,
        "token_cap": cap,
        "scope_status": "NOT_EVALUATED" if not scope_evaluated else ("PASS" if not scope else "FAIL"),
        "scope_violations": scope,
        "violations": sorted(set(violations)),
    }


def validate_bundle(
    memory: str, skill: str, agent: str, *,
    token_counter: Callable[[str], int] | None = None,
    forbidden_literals: Iterable[str] | None = None,
    scope_manifest: dict | None = None,
) -> dict:
    checks = {
        "text_memory": validate_carrier("text_memory", memory, token_counter=token_counter, forbidden_literals=forbidden_literals, scope_manifest=scope_manifest),
        "skill_bundle": validate_carrier("skill_bundle", skill, token_counter=token_counter, forbidden_literals=forbidden_literals, scope_manifest=scope_manifest),
        "agent_system": validate_carrier("agent_system", agent, token_counter=token_counter, forbidden_literals=forbidden_literals, scope_manifest=scope_manifest),
    }
    return {"status": "PASS" if all(x["status"] == "PASS" for x in checks.values()) else "FAIL", "checks": checks}


def require_evolution_compatible_bundle(
    memory: str, skill: str, agent: str, *, context: str,
    token_counter: Callable[[str], int] | None = None,
    forbidden_literals: Iterable[str] | None = None,
    scope_manifest: dict | None = None,
) -> dict:
    """Require seed and future candidates to share the frozen carrier language.

    This validates existing bytes. Selection, rewriting, generation and historical
    receipt mutation remain outside this function. Token caps are checked when the
    caller supplies its actual tokenizer; structural/lexical failures are always
    sufficient to reject entry.
    """
    result = validate_bundle(memory, skill, agent, token_counter=token_counter, forbidden_literals=forbidden_literals, scope_manifest=scope_manifest)
    if result["status"] != "PASS":
        detail = "; ".join(
            f"{kind}: {', '.join(check['violations'])}"
            for kind, check in result["checks"].items()
            if check["status"] != "PASS"
        )
        raise ValueError(f"Carrier lifecycle contract mismatch ({context}): {detail}")
    return result
