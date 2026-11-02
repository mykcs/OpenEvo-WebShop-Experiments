#!/usr/bin/env python3
"""Create and consume durable cross-phase successor handoff receipts.

The handoff closes one anti-friction gap: stable facts proven by a predecessor
become machine-readable inputs to its successor instead of being rediscovered
from server directories and old launch scripts.

This module is CPU-only. It never authorizes formal task consumption, allocates
a GPU, changes a scientific contract, or treats volatile runtime facts as
reusable. Runtime-qualified facts are exposed only as CACHE_CANDIDATE entries;
runtime_qualification.py remains the authority for current cache reuse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]

SPEC_SCHEMA = "openevo.successor-handoff-spec.v1"
HANDOFF_SCHEMA = "openevo.successor-handoff.v1"
BOOTSTRAP_SCHEMA = "openevo.successor-bootstrap.v1"
ALLOWED_LIFETIMES = {"immutable", "runtime-qualified"}
ALLOWED_KINDS = {"identity", "file", "receipt"}
EXPECTED_LIFETIME_BY_KIND = {
    "identity": "immutable",
    "file": "immutable",
    "receipt": "runtime-qualified",
}


class HandoffError(ValueError):
    """Raised when a handoff cannot be proven safely."""


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_json_bytes(payload))
    temporary.replace(path)


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HandoffError(f"{label} unreadable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise HandoffError(f"{label} must be a JSON object: {path}")
    return payload


def resolve_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root.expanduser().resolve() / path).resolve()


def _validate_expected_sha(expected: Any, *, role: str) -> str | None:
    if expected is None:
        return None
    prefix = "sha256:"
    if (
        not isinstance(expected, str)
        or not expected.startswith(prefix)
        or len(expected) != len(prefix) + 64
        or any(ch not in "0123456789abcdef" for ch in expected[len(prefix):])
    ):
        raise HandoffError(f"{role}: sha256 must be sha256:<64 lowercase hex>")
    return expected


def _validate_pairing(*, role: str, kind: Any, lifetime: Any) -> str | None:
    if kind not in ALLOWED_KINDS or lifetime not in ALLOWED_LIFETIMES:
        return "invalid fact lifetime/kind"
    expected = EXPECTED_LIFETIME_BY_KIND[kind]
    if lifetime != expected:
        return (
            f"invalid kind/lifetime pairing: kind={kind} requires "
            f"lifetime={expected}, observed {lifetime}"
        )
    return None


def _identity_digest(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _capture_file_fact(
    fact: Mapping[str, Any],
    *,
    root: Path,
    role: str,
    kind: str,
    lifetime: str,
) -> dict[str, Any]:
    raw_path = fact.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise HandoffError(f"{role}: {kind} fact requires path")
    path = resolve_path(root, raw_path)
    if not path.is_file():
        raise HandoffError(f"{role}: file is missing: {path}")

    observed_sha = sha256_file(path)
    expected_sha = _validate_expected_sha(fact.get("sha256"), role=role)
    if expected_sha is not None and observed_sha != expected_sha:
        raise HandoffError(
            f"{role}: content mismatch: expected {expected_sha}, observed {observed_sha}"
        )

    captured: dict[str, Any] = {
        "role": role,
        "kind": kind,
        "lifetime": lifetime,
        "path": raw_path,
        "sha256": observed_sha,
    }
    if kind != "receipt":
        return captured

    receipt = load_json(path, label=f"{role} receipt")
    required_schema = fact.get("required_schema")
    if required_schema is not None:
        if not isinstance(required_schema, str) or not required_schema:
            raise HandoffError(f"{role}: required_schema must be a non-empty string")
        if receipt.get("schema") != required_schema:
            raise HandoffError(
                f"{role}: receipt schema mismatch: expected {required_schema}, "
                f"observed {receipt.get('schema')!r}"
            )
        captured["required_schema"] = required_schema

    required_status = fact.get("required_status")
    if required_status is not None:
        if not isinstance(required_status, str) or not required_status:
            raise HandoffError(f"{role}: required_status must be a non-empty string")
        if receipt.get("status") != required_status:
            raise HandoffError(
                f"{role}: receipt status mismatch: expected {required_status}, "
                f"observed {receipt.get('status')!r}"
            )
        captured["required_status"] = required_status
    return captured


def capture(spec_path: Path, *, root: Path = REPO_ROOT) -> dict[str, Any]:
    """Materialize a deterministic handoff receipt from a handoff spec."""
    spec_path = spec_path.expanduser().resolve()
    spec = load_json(spec_path, label="successor handoff spec")
    if spec.get("schema") != SPEC_SCHEMA:
        raise HandoffError(f"spec schema must be {SPEC_SCHEMA}")

    predecessor = spec.get("predecessor")
    if not isinstance(predecessor, dict) or not predecessor:
        raise HandoffError("spec.predecessor must be a non-empty object")
    successor = spec.get("successor")
    if successor is not None and not isinstance(successor, dict):
        raise HandoffError("spec.successor must be an object when present")
    facts = spec.get("facts")
    if not isinstance(facts, list) or not facts:
        raise HandoffError("spec.facts must be a non-empty list")

    seen: set[str] = set()
    captured_facts: list[dict[str, Any]] = []
    for raw_fact in facts:
        if not isinstance(raw_fact, dict):
            raise HandoffError("every spec.facts entry must be an object")
        role = raw_fact.get("role")
        if not isinstance(role, str) or not role.strip():
            raise HandoffError("every fact requires a non-empty string role")
        role = role.strip()
        if role in seen:
            raise HandoffError(f"duplicate fact role: {role}")
        seen.add(role)

        kind = raw_fact.get("kind")
        lifetime = raw_fact.get("lifetime")
        pairing_failure = _validate_pairing(
            role=role, kind=kind, lifetime=lifetime
        )
        if pairing_failure:
            if lifetime not in ALLOWED_LIFETIMES:
                raise HandoffError(
                    f"{role}: lifetime must be one of {sorted(ALLOWED_LIFETIMES)}; "
                    "volatile facts must be rechecked at use and cannot enter a handoff"
                )
            raise HandoffError(f"{role}: {pairing_failure}")

        if kind == "identity":
            if "value" not in raw_fact:
                raise HandoffError(f"{role}: identity fact requires value")
            captured_facts.append(
                {
                    "role": role,
                    "kind": kind,
                    "lifetime": lifetime,
                    "value": raw_fact["value"],
                    "identity_sha256": _identity_digest(raw_fact["value"]),
                }
            )
            continue

        captured_facts.append(
            _capture_file_fact(
                raw_fact,
                root=root,
                role=role,
                kind=kind,
                lifetime=lifetime,
            )
        )

    receipt: dict[str, Any] = {
        "schema": HANDOFF_SCHEMA,
        "status": "PASS",
        "predecessor": predecessor,
        "facts": captured_facts,
        "spec_sha256": sha256_file(spec_path),
        "formal_task_consumption_allowed": False,
        "gpu_allocation_allowed": False,
    }
    if successor:
        receipt["successor"] = successor
    # No wall clock: identical spec + evidence produces byte-identical receipts.
    return receipt


def _validate_handoff_shape(handoff: Mapping[str, Any]) -> list[dict[str, Any]]:
    if handoff.get("schema") != HANDOFF_SCHEMA:
        raise HandoffError(f"handoff schema must be {HANDOFF_SCHEMA}")
    if handoff.get("status") != "PASS":
        raise HandoffError("handoff status must be PASS")
    if handoff.get("formal_task_consumption_allowed") is not False:
        raise HandoffError("handoff must not authorize formal task consumption")
    if handoff.get("gpu_allocation_allowed") is not False:
        raise HandoffError("handoff must not authorize GPU allocation")
    facts = handoff.get("facts")
    if not isinstance(facts, list) or not facts:
        raise HandoffError("handoff facts must be a non-empty list")
    if not all(isinstance(fact, dict) for fact in facts):
        raise HandoffError("handoff facts must contain only objects")
    return facts  # type: ignore[return-value]


def _check_receipt_constraints(fact: Mapping[str, Any], *, path: Path) -> str | None:
    try:
        receipt = load_json(path, label=f"{fact.get('role')} receipt")
    except HandoffError as exc:
        return str(exc)
    required_schema = fact.get("required_schema")
    if required_schema is not None and receipt.get("schema") != required_schema:
        return (
            f"receipt schema changed: expected {required_schema}, "
            f"observed {receipt.get('schema')!r}"
        )
    required_status = fact.get("required_status")
    if required_status is not None and receipt.get("status") != required_status:
        return (
            f"receipt status changed: expected {required_status}, "
            f"observed {receipt.get('status')!r}"
        )
    return None


def bootstrap(
    handoff_path: Path,
    *,
    root: Path = REPO_ROOT,
    required_roles: tuple[str, ...] = (),
    expected_handoff_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate reusable predecessor evidence for successor PREPARE."""
    handoff_path = handoff_path.expanduser().resolve()
    observed_handoff_sha = sha256_file(handoff_path)
    expected_handoff_sha = _validate_expected_sha(
        expected_handoff_sha256, role="successor handoff"
    )
    if (
        expected_handoff_sha is not None
        and observed_handoff_sha != expected_handoff_sha
    ):
        raise HandoffError(
            "successor handoff: content mismatch: "
            f"expected {expected_handoff_sha}, observed {observed_handoff_sha}"
        )

    handoff = load_json(handoff_path, label="successor handoff")
    facts = _validate_handoff_shape(handoff)

    by_role: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    reused_roles: list[str] = []
    cache_candidates: list[str] = []

    for fact in facts:
        role = fact.get("role")
        if not isinstance(role, str) or not role:
            failures.append({"role": "<invalid>", "reason": "fact role is missing"})
            continue
        if role in by_role:
            failures.append({"role": role, "reason": "duplicate role in handoff"})
            continue
        by_role[role] = fact

        lifetime = fact.get("lifetime")
        kind = fact.get("kind")
        pairing_failure = _validate_pairing(
            role=role, kind=kind, lifetime=lifetime
        )
        if pairing_failure:
            failures.append({"role": role, "reason": pairing_failure})
            continue

        if kind == "identity":
            observed_identity = _identity_digest(fact.get("value"))
            if observed_identity != fact.get("identity_sha256"):
                failures.append(
                    {
                        "role": role,
                        "reason": "identity payload no longer matches recorded digest",
                    }
                )
                continue
            checks.append({"role": role, "status": "REUSE", "kind": kind})
            reused_roles.append(role)
            continue

        raw_path = fact.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            failures.append({"role": role, "reason": "path is missing"})
            continue
        path = resolve_path(root, raw_path)
        if not path.is_file():
            failures.append({"role": role, "reason": f"file missing: {path}"})
            continue
        observed_sha = sha256_file(path)
        if observed_sha != fact.get("sha256"):
            failures.append(
                {
                    "role": role,
                    "reason": (
                        f"content changed: expected {fact.get('sha256')}, "
                        f"observed {observed_sha}"
                    ),
                }
            )
            continue

        if kind == "receipt":
            constraint_failure = _check_receipt_constraints(fact, path=path)
            if constraint_failure:
                failures.append({"role": role, "reason": constraint_failure})
                continue
            checks.append(
                {
                    "role": role,
                    "status": "CACHE_CANDIDATE",
                    "kind": kind,
                    "path": raw_path,
                }
            )
            cache_candidates.append(role)
        else:
            checks.append(
                {"role": role, "status": "REUSE", "kind": kind, "path": raw_path}
            )
            reused_roles.append(role)

    missing_required = sorted(set(required_roles) - set(by_role))
    for role in missing_required:
        failures.append({"role": role, "reason": "required role absent from handoff"})

    status = "READY_FOR_SUCCESSOR_PREP" if not failures else "STALE"
    return {
        "schema": BOOTSTRAP_SCHEMA,
        "status": status,
        "handoff_path": str(handoff_path),
        "handoff_sha256": observed_handoff_sha,
        "predecessor": handoff.get("predecessor"),
        "successor": handoff.get("successor"),
        "checks": checks,
        "reused_roles": sorted(reused_roles),
        "runtime_qualification_cache_candidates": sorted(cache_candidates),
        "failures": failures,
        "formal_task_consumption_allowed": False,
        "gpu_allocation_allowed": False,
        "next_step": (
            "continue successor PREPARE; invoke canonical runtime qualification "
            "for any CACHE_CANDIDATE before execution readiness"
            if not failures
            else "repair or recapture predecessor evidence before successor PREPARE"
        ),
    }


def _print_bootstrap_summary(result: Mapping[str, Any]) -> None:
    for check in result.get("checks", []):
        status = str(check.get("status", "UNKNOWN"))
        print(f"[{status:<15}] {check.get('role')}")
    for failure in result.get("failures", []):
        print(f"[STALE          ] {failure.get('role')}: {failure.get('reason')}")
    print(f"[{result['status']}] successor handoff bootstrap")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    capture_parser = sub.add_parser(
        "capture", help="close a predecessor phase into a deterministic handoff receipt"
    )
    capture_parser.add_argument("--spec", type=Path, required=True)
    capture_parser.add_argument("--root", type=Path, default=REPO_ROOT)
    capture_parser.add_argument("--output", type=Path, required=True)

    bootstrap_parser = sub.add_parser(
        "bootstrap", help="reuse predecessor stable facts for successor PREPARE"
    )
    bootstrap_parser.add_argument("--handoff", type=Path, required=True)
    bootstrap_parser.add_argument("--root", type=Path, default=REPO_ROOT)
    bootstrap_parser.add_argument("--require-role", action="append", default=[])
    bootstrap_parser.add_argument("--expected-handoff-sha256")
    bootstrap_parser.add_argument("--output", type=Path)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "capture":
            receipt = capture(args.spec, root=args.root)
            output = args.output.expanduser().resolve()
            atomic_json(output, receipt)
            print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
            print(f"[HANDOFF_PASS] {output} {sha256_file(output)}")
            return 0

        result = bootstrap(
            args.handoff,
            root=args.root,
            required_roles=tuple(args.require_role),
            expected_handoff_sha256=args.expected_handoff_sha256,
        )
        if args.output:
            atomic_json(args.output.expanduser().resolve(), result)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        _print_bootstrap_summary(result)
        return 0 if result["status"] == "READY_FOR_SUCCESSOR_PREP" else 1
    except (OSError, HandoffError) as exc:
        print(f"successor handoff: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
