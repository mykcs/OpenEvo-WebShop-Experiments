#!/usr/bin/env python3
"""Canonical manifest identity hashes.

The repository distinguishes different digests that must never be conflated:

- **materialized manifest file SHA-256**: hash of the exact bytes on disk
  (``materialized_manifest_file_sha256`` / ``manifest_file_sha256``).
- **canonical manifest content SHA-256**: hash of the JSON content with its
  self-referential hash field removed, serialized with sorted keys and compact
  separators (``manifest_content_sha256``). Reformatting the file without
  changing content leaves this digest unchanged.
- **task identity digest**: a campaign-level digest over task identity rows
  (``task_identity_digests``). Its computation is owned by the historical
  manifest builders; new code should bind ``manifest_content_sha256`` (which
  already covers every task row) instead of re-deriving it.

Historical validators such as ``validate_h140_r2_preregistration.py`` pin file
and content hashes with *different* values. New code must reuse these helpers
rather than inferring semantics from a field name.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


# Self-referential hash fields a manifest may carry. The canonical content
# digest is computed with this field removed, matching the historical
# validators (`manifest_sha256` for panels/sets, `ledger_sha256` for ledgers).
SELF_HASH_FIELDS = ("manifest_sha256", "ledger_sha256")


class ManifestIdentityError(ValueError):
    """Raised when a manifest cannot be identified under canonical semantics."""


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON the way historical content hashes were computed."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def canonical_content_sha256(value: Any) -> str:
    """Bare-hex SHA-256 of the canonical JSON serialization of ``value``."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def manifest_file_sha256(path: Path | str) -> str:
    """Bare-hex SHA-256 of the exact manifest file bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest(path: Path | str) -> dict[str, Any]:
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestIdentityError(f"manifest unreadable: {target}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ManifestIdentityError(f"manifest must be a JSON object: {target}")
    return payload


def self_hash_field(payload: dict[str, Any]) -> str | None:
    """Return the self-referential hash field present in ``payload``, if any."""
    for field in SELF_HASH_FIELDS:
        if field in payload:
            return field
    return None


def manifest_content_sha256(path: Path | str) -> str:
    """Bare-hex canonical content SHA-256 of a manifest file.

    The manifest's self-referential hash field is removed before hashing,
    exactly as the historical validators compute ``EXPECTED_CONTENT_HASHES``.
    """
    payload = load_manifest(path)
    field = self_hash_field(payload)
    if field is None:
        raise ManifestIdentityError(
            f"manifest has no self hash field {SELF_HASH_FIELDS}: {path}; "
            "cannot establish canonical content identity"
        )
    stripped = dict(payload)
    stripped.pop(field)
    return canonical_content_sha256(stripped)


def manifest_self_consistent(path: Path | str) -> bool:
    """True when the manifest's embedded self hash matches its content hash."""
    payload = load_manifest(path)
    field = self_hash_field(payload)
    if field is None:
        return False
    stripped = dict(payload)
    recorded = stripped.pop(field)
    return recorded == canonical_content_sha256(stripped)
