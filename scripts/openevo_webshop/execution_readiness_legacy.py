#!/usr/bin/env python3
"""Execution-readiness boundary: PREPARE -> READY_TO_EXECUTE -> EXECUTE.

This module makes campaign execution readiness machine-verifiable. It resolves
a frozen execution release receipt, proves the release is still the *current*
execution authority (not merely a historical receipt), recomputes the identity
of every bound scientific artifact with canonical hash semantics, hash-binds
the execution profile/plan, reuses a cached stable runtime qualification when
the fingerprint is unchanged, rechecks volatile launch facts, and emits an
``openevo.execution-readiness.v1`` READY receipt.

Two authorities are kept deliberately separate:

- **Scientific authority** (this module): is this exact experiment, with these
  exact artifacts, still authorized to consume formal tasks *right now*?
- **Runtime authority** (``runtime_qualification`` + ``runtime_live_gate``):
  can this runtime execute at all? A runtime cache hit never substitutes for
  scientific authority, and a requalification never re-authorizes a campaign.

The module is deliberately not a launcher. It proves readiness; ``scripts/oe``
remains the only canonical entrypoint that may select GPUs and launch cells.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.openevo_webshop import manifest_identity, runtime_live_gate, runtime_qualification  # noqa: E402


LOGGER = logging.getLogger(__name__)

SCHEMA = "openevo.execution-readiness.v1"
RELEASE_SCHEMA_PREFIX = "openevo.experiment-"
CURRENT_CAMPAIGN_PATH = Path("configs/experiment/current-campaign.json")

VERDICT_READY = "READY"
VERDICT_BLOCKED = "BLOCKED"
VERDICT_NOT_PREPARED = "BLOCKED_NOT_PREPARED"
VERDICT_STALE = "STALE"
VERDICT_REPREPARE = "REQUIRES_REPREPARATION"

# The only router status under which formal task consumption may proceed.
EXECUTION_PERMITTING_STATUS = "launch-authorized"

# Roles whose drift means the *scientific identity* changed (reprepare), not
# merely that a volatile receipt aged out (stale).
SCIENTIFIC_IDENTITY_ROLES = {
    "model",
    "tokenizer",
    "adapter",
    "checkpoint",
    "evaluator",
    "prompt",
    "parser",
    "webshop_data",
    "dataset",
    "training_data",
}

# Frozen-contract keys that summarize the scientific identity in receipts.
IDENTITY_SUMMARY_KEYS = (
    "model",
    "model_content_pin",
    "tokenizer",
    "prompt_contract",
    "parser_contract",
    "evaluator",
    "temperature",
    "max_new_tokens",
    "max_steps",
    "training_seed",
    "official_open_evo_stable",
)


class ExecutionReadinessError(Exception):
    """Raised when a readiness check fails in a way that should fail closed."""

    def __init__(self, reason: str, *, verdict: str = VERDICT_NOT_PREPARED) -> None:
        super().__init__(reason)
        self.reason = reason
        self.verdict = verdict


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_sha(value: str | None) -> str | None:
    """Return a SHA-256 digest string with the canonical 'sha256:' prefix."""
    if value is None:
        return None
    if value.startswith("sha256:"):
        return value
    return "sha256:" + value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one evidence object once; later readiness checks cannot replace it."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _resolved(path: Path | str | None, *, root: Path | None = None) -> Path | None:
    if path is None:
        return None
    target = Path(path).expanduser()
    if not target.is_absolute() and root is not None:
        target = root / target
    return target.resolve()


def _runtime_executable_path(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    return Path(path).expanduser().absolute()


def _resolve_artifact_path(path: str, *, root: Path, role: str) -> Path:
    """Resolve a declared artifact path, failing closed on boundary escape.

    Repository-relative paths must stay inside ``root``; absolute paths are
    explicit declarations and are recorded verbatim. A ``..`` escape from a
    repo-relative declaration is never honored.
    """
    raw = Path(path).expanduser()
    target = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    if not raw.is_absolute() and target != root and root not in target.parents:
        raise ExecutionReadinessError(
            f"{role} path escapes the repository/artifact boundary: {path}",
            verdict=VERDICT_BLOCKED,
        )
    return target


def load_release_receipt(path: Path) -> dict[str, Any]:
    """Load and validate an execution release receipt. Fail closed."""
    path = path.expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExecutionReadinessError(f"release receipt unreadable: {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ExecutionReadinessError("release receipt must be a JSON object")

    schema = payload.get("schema") or ""
    if not isinstance(schema, str) or not schema.startswith(RELEASE_SCHEMA_PREFIX):
        raise ExecutionReadinessError(
            f"release receipt schema must start with {RELEASE_SCHEMA_PREFIX!r}, got {schema!r}"
        )

    if payload.get("status") != "launch-authorized":
        raise ExecutionReadinessError(
            f"release receipt status is {payload.get('status')!r}, expected 'launch-authorized'"
        )

    if payload.get("formal_task_consumption_allowed") is not True:
        raise ExecutionReadinessError(
            "release receipt does not allow formal task consumption",
            verdict=VERDICT_NOT_PREPARED,
        )

    campaign_id = payload.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise ExecutionReadinessError("release receipt missing campaign_id")

    prereg = payload.get("preregistration")
    if not isinstance(prereg, dict) or not isinstance(prereg.get("path"), str):
        raise ExecutionReadinessError("release receipt missing preregistration.path")
    if not isinstance(prereg.get("sha256"), str):
        raise ExecutionReadinessError("release receipt missing preregistration.sha256")

    pc = payload.get("path_contract")
    if not isinstance(pc, dict) or not isinstance(pc.get("path"), str):
        raise ExecutionReadinessError("release receipt missing path_contract.path")
    if not isinstance(pc.get("sha256"), str):
        raise ExecutionReadinessError("release receipt missing path_contract.sha256")

    return payload


def resolve_release_path(
    campaign_id: str | None,
    release_path: Path | str | None,
    *,
    current_campaign_path: Path | str | None = None,
    repo_root: Path | str | None = None,
) -> Path:
    """Resolve the release receipt path from explicit input or campaign router.

    Resolution only *finds* the receipt; it grants no authority. Whether the
    receipt is still the current execution authority is decided separately by
    :func:`verify_current_authority`.
    """
    root = Path(repo_root).expanduser().resolve() if repo_root else REPO_ROOT

    if release_path is not None:
        resolved = Path(release_path).expanduser().resolve()
        if not resolved.is_file():
            raise ExecutionReadinessError(f"release receipt missing: {resolved}")
        if resolved != root and root not in resolved.parents:
            raise ExecutionReadinessError(
                f"release receipt is outside the repository/artifact boundary: {resolved}",
                verdict=VERDICT_BLOCKED,
            )
        return resolved

    if campaign_id is None:
        raise ExecutionReadinessError(
            "--release or --campaign is required",
            verdict=VERDICT_NOT_PREPARED,
        )

    router = _load_router(current_campaign_path, root=root)
    if not isinstance(router, dict) or router.get("campaign_id") != campaign_id:
        raise ExecutionReadinessError(
            f"current-campaign router does not match campaign {campaign_id!r}",
            verdict=VERDICT_NOT_PREPARED,
        )

    pointer = _router_release_pointer(router)
    if pointer is None:
        raise ExecutionReadinessError(
            f"current-campaign router has no release/activation path for {campaign_id!r}",
            verdict=VERDICT_NOT_PREPARED,
        )

    resolved = _resolved(pointer, root=root)
    if resolved is None or not resolved.is_file():
        raise ExecutionReadinessError(
            f"resolved release receipt missing: {pointer}",
            verdict=VERDICT_NOT_PREPARED,
        )
    return resolved


def _router_release_pointer(router: dict[str, Any]) -> str | None:
    latest = router.get("latest_event")
    if isinstance(latest, dict):
        for key in ("activation", "release_receipt"):
            value = latest.get(key)
            if isinstance(value, str) and value:
                return value
    for key in ("activation", "release_receipt"):
        value = router.get(key)
        if isinstance(value, dict):
            value = value.get("path")
        if isinstance(value, str) and value:
            return value
    return None


def _load_router(current_campaign_path: Path | str | None, *, root: Path) -> dict[str, Any]:
    cc_path = _resolved(current_campaign_path or CURRENT_CAMPAIGN_PATH, root=root)
    if cc_path is None or not cc_path.is_file():
        raise ExecutionReadinessError(
            f"current-campaign router missing: {cc_path}",
            verdict=VERDICT_BLOCKED,
        )
    try:
        router = json.loads(cc_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExecutionReadinessError(
            f"current-campaign router unreadable: {exc}",
            verdict=VERDICT_BLOCKED,
        ) from exc
    if not isinstance(router, dict):
        raise ExecutionReadinessError(
            "current-campaign router must be a JSON object",
            verdict=VERDICT_BLOCKED,
        )
    return router


def verify_current_authority(
    release: dict[str, Any],
    release_path: Path,
    *,
    current_campaign_path: Path | str | None = None,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Prove the release is still the *current* execution authority.

    A historical release receipt only proves a campaign was once authorized.
    Execution may proceed only when the live current-campaign router still
    routes to the same campaign, still permits formal task consumption, and
    still binds this exact release receipt (path and, when recorded, bytes) as
    its authority pointer. Anything else is ``BLOCKED`` — regardless of
    whether the caller passed ``--campaign`` or an explicit ``--release``.
    """
    root = Path(repo_root).expanduser().resolve() if repo_root else REPO_ROOT
    router = _load_router(current_campaign_path, root=root)
    campaign_id = release["campaign_id"]

    router_campaign = router.get("campaign_id")
    if router_campaign != campaign_id:
        if router.get("replacement_of") == campaign_id:
            reason = (
                f"campaign {campaign_id!r} was superseded by successor "
                f"{router_campaign!r}; the release is historical evidence only"
            )
        else:
            reason = (
                f"current-campaign router points to {router_campaign!r}, "
                f"not {campaign_id!r}"
            )
        raise ExecutionReadinessError(reason, verdict=VERDICT_BLOCKED)

    status = router.get("status")
    if status != EXECUTION_PERMITTING_STATUS:
        raise ExecutionReadinessError(
            f"current campaign closed or not executable: status={status!r}",
            verdict=VERDICT_BLOCKED,
        )

    if router.get("formal_task_consumption_allowed") is not True:
        raise ExecutionReadinessError(
            "current campaign router does not allow formal task consumption",
            verdict=VERDICT_BLOCKED,
        )

    # The router must still bind *this* release as its authority pointer.
    pointer = _router_release_pointer(router)
    if pointer is None:
        raise ExecutionReadinessError(
            "current campaign router does not bind any release receipt as its authority",
            verdict=VERDICT_BLOCKED,
        )
    pointer_path = _resolved(pointer, root=root)
    if pointer_path != release_path.expanduser().resolve():
        raise ExecutionReadinessError(
            f"current campaign router binds a different release ({pointer}); "
            "the supplied release is not the current authority",
            verdict=VERDICT_BLOCKED,
        )

    latest = router.get("latest_event") if isinstance(router.get("latest_event"), dict) else {}
    recorded_sha = _normalize_sha(latest.get("release_receipt_sha256") or router.get("release_receipt_sha256"))
    if recorded_sha is not None and recorded_sha != sha256_file(release_path):
        raise ExecutionReadinessError(
            "current campaign router binds a different release receipt bytes; "
            "the supplied release is superseded",
            verdict=VERDICT_BLOCKED,
        )

    # When the router records its current preregistration, it must agree with
    # the release's preregistration binding.
    router_prereg_path = latest.get("preregistration") or router.get("preregistration")
    if isinstance(router_prereg_path, str) and router_prereg_path != release["preregistration"]["path"]:
        raise ExecutionReadinessError(
            f"current campaign router preregistration {router_prereg_path!r} does not match "
            f"the release preregistration {release['preregistration']['path']!r}",
            verdict=VERDICT_BLOCKED,
        )
    router_prereg_sha = _normalize_sha(latest.get("preregistration_sha256"))
    if router_prereg_sha is not None and router_prereg_sha != _normalize_sha(
        release["preregistration"].get("sha256")
    ):
        raise ExecutionReadinessError(
            "current campaign router preregistration hash does not match the release",
            verdict=VERDICT_BLOCKED,
        )

    return {
        "router_path": str(_resolved(current_campaign_path or CURRENT_CAMPAIGN_PATH, root=root)),
        "router_status": status,
        "router_release_pointer": pointer,
        "campaign_id": campaign_id,
    }


def _artifact_entry(
    role: str,
    path: Path | None,
    recorded_sha: str | None,
    *,
    hash_kind: str = "file",
) -> dict[str, Any] | None:
    if path is None:
        return None
    return {"role": role, "path": str(path), "recorded_sha": recorded_sha, "hash_kind": hash_kind}


def _discover_contract_pins(node: Any, *, root: Path, prefix: str = "") -> list[dict[str, Any]]:
    """Discover ``<name>`` + ``<name>_sha256`` pins inside the frozen contract.

    A dict that carries both a path-like string value and a sibling
    ``*_sha256`` key declares a frozen artifact pin. ``*_content_sha256``
    pins use canonical content-hash semantics; every other pin uses exact
    file bytes. This binds whatever scientific identity the frozen contract
    pins — panels, selection files, patches, adapters — without
    campaign-specific code.
    """
    artifacts: list[dict[str, Any]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if (
                isinstance(key, str)
                and key.endswith("_sha256")
                and isinstance(value, str)
            ):
                base = key[: -len("_sha256")]
                # The path sibling may drop a hash-kind suffix: panels declare
                # `manifest` + `manifest_content_sha256` (or
                # `manifest_file_sha256`).
                sibling = None
                for candidate in (base, base.removesuffix("_content"), base.removesuffix("_file")):
                    value_at = node.get(candidate)
                    if isinstance(value_at, str) and value_at and not value_at.startswith("sha256:"):
                        sibling = value_at
                        break
                if sibling is not None:
                    hash_kind = "content" if base.endswith("_content") else "file"
                    role = f"{prefix}{base}" if prefix else base
                    try:
                        artifacts.append(
                            _artifact_entry(
                                role,
                                _resolve_artifact_path(sibling, root=root, role=role),
                                value,
                                hash_kind=hash_kind,
                            )
                        )
                    except ExecutionReadinessError:
                        raise
        for key, value in node.items():
            if isinstance(value, (dict, list)):
                artifacts.extend(
                    _discover_contract_pins(value, root=root, prefix=f"{prefix}{key}:")
                )
    elif isinstance(node, list):
        for index, value in enumerate(node):
            if isinstance(value, (dict, list)):
                artifacts.extend(
                    _discover_contract_pins(value, root=root, prefix=f"{prefix}{index}:")
                )
    return artifacts


def _collect_declared_artifacts(release: dict[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    """Collect artifact records from the release receipt and preregistration."""
    artifacts: list[dict[str, Any]] = []
    root = repo_root

    prereg_path = _resolve_artifact_path(
        release["preregistration"]["path"], root=root, role="preregistration"
    )
    artifacts.append(
        _artifact_entry("preregistration", prereg_path, release["preregistration"].get("sha256"))
    )

    pc_path = _resolve_artifact_path(
        release["path_contract"]["path"], root=root, role="path_contract"
    )
    artifacts.append(
        _artifact_entry("path_contract", pc_path, release["path_contract"].get("sha256"))
    )

    patch = release.get("upstream_patch")
    if isinstance(patch, dict) and isinstance(patch.get("path"), str):
        patch_path = _resolve_artifact_path(patch["path"], root=root, role="upstream_patch")
        artifacts.append(_artifact_entry("upstream_patch", patch_path, patch.get("sha256")))

    # Optional explicit readiness artifact list in the release receipt. This
    # is where evaluator/prompt/parser implementations, adapter checkpoints,
    # tokenizer assets and data identity files are declared.
    readiness = release.get("execution_readiness")
    if isinstance(readiness, dict):
        for item in readiness.get("frozen_scientific_artifacts") or []:
            if isinstance(item, dict) and isinstance(item.get("path"), str):
                role = item.get("role", "artifact")
                artifacts.append(
                    _artifact_entry(
                        role,
                        _resolve_artifact_path(item["path"], root=root, role=role),
                        item.get("sha256"),
                        hash_kind=item.get("hash_kind", "file"),
                    )
                )

    # Pins declared by the frozen scientific contract itself (evaluation
    # panels with manifest_content_sha256, selection files, patches, ...).
    prereg: dict[str, Any] | None = None
    if prereg_path.is_file():
        try:
            loaded = json.loads(prereg_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                prereg = loaded
        except (OSError, json.JSONDecodeError):
            prereg = None
    if prereg is not None:
        frozen = prereg.get("frozen_scientific_contract")
        if isinstance(frozen, dict):
            artifacts.extend(_discover_contract_pins(frozen, root=root))
        source_artifacts = prereg.get("source_artifacts")
        if isinstance(source_artifacts, dict):
            artifacts.extend(_discover_contract_pins(source_artifacts, root=root, prefix="source:"))

    # Deduplicate by (path, hash_kind), keeping the first declared role.
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, Any]] = []
    for artifact in artifacts:
        if artifact is None:
            continue
        key = (artifact["path"], artifact["hash_kind"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(artifact)
    return unique


def _observed_sha(path: Path, hash_kind: str) -> str:
    if hash_kind == "content":
        return "sha256:" + manifest_identity.manifest_content_sha256(path)
    return sha256_file(path)


def hash_bound_artifacts(release: dict[str, Any], *, repo_root: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """Recompute the identity of every declared artifact.

    Returns a dict with ``missing`` (declared but absent -> not prepared) and
    ``mismatched`` (present but drifted -> stale / reprepare) lists. Hash
    semantics come from the artifact's ``hash_kind``: exact file bytes or
    canonical manifest content, never guessed from a field name.
    """
    root = (repo_root or REPO_ROOT).expanduser().resolve()
    missing: list[dict[str, Any]] = []
    mismatched: list[dict[str, Any]] = []

    for artifact in _collect_declared_artifacts(release, root):
        path = Path(artifact["path"])
        if not path.is_file():
            missing.append(
                {
                    "role": artifact["role"],
                    "path": str(path),
                    "recorded_sha": artifact.get("recorded_sha"),
                    "reason": "missing",
                }
            )
            continue

        try:
            observed = _observed_sha(path, artifact["hash_kind"])
        except manifest_identity.ManifestIdentityError as exc:
            mismatched.append(
                {
                    "role": artifact["role"],
                    "path": str(path),
                    "recorded_sha": _normalize_sha(artifact.get("recorded_sha")),
                    "observed_sha": None,
                    "reason": str(exc),
                }
            )
            continue
        recorded = _normalize_sha(artifact.get("recorded_sha"))
        if recorded is not None and observed != recorded:
            mismatched.append(
                {
                    "role": artifact["role"],
                    "path": str(path),
                    "recorded_sha": recorded,
                    "observed_sha": observed,
                }
            )

    return {"missing": missing, "mismatched": mismatched}


def scientific_identity_document(prereg_path: Path) -> dict[str, Any] | None:
    """Extract the canonical scientific identity document from a preregistration.

    The identity document is the preregistration's frozen scientific contract
    (plus source artifact pins), serialized canonically. Its digest is recorded
    in the READY receipt so the authorized scientific identity is explicit and
    re-checkable at launch time. Runtime drift never changes this document;
    only a re-preregistration does.
    """
    try:
        prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(prereg, dict):
        return None
    frozen = prereg.get("frozen_scientific_contract")
    if not isinstance(frozen, dict) or not frozen:
        return None

    document: dict[str, Any] = {
        "frozen_scientific_contract": frozen,
    }
    if isinstance(prereg.get("source_artifacts"), dict):
        document["source_artifacts"] = prereg["source_artifacts"]

    summary = {key: frozen.get(key) for key in IDENTITY_SUMMARY_KEYS if key in frozen}
    panels = frozen.get("evaluation_panels")
    panel_summary: dict[str, Any] = {}
    if isinstance(panels, dict):
        for name, panel in panels.items():
            if isinstance(panel, dict):
                panel_summary[name] = {
                    key: panel.get(key)
                    for key in ("manifest", "manifest_content_sha256", "rollout_seeds", "attempts_per_arm")
                    if key in panel
                }
    if panel_summary:
        summary["evaluation_panels"] = panel_summary

    return {
        "sha256": "sha256:" + manifest_identity.canonical_content_sha256(document),
        "summary": summary,
    }


def resolve_execution_profile(release: dict[str, Any], *, repo_root: Path) -> dict[str, Any] | None:
    """Return the hash-bound execution profile spec declared by the release.

    The execution profile/plan is a mandatory frozen artifact once declared:
    the release must bind both its path and its exact bytes. A legacy
    bare-string profile reference is not hash-bound and fails closed.
    """
    readiness = release.get("execution_readiness")
    if not isinstance(readiness, dict):
        return None
    profile = readiness.get("profile")
    if profile is None:
        return None
    if isinstance(profile, str):
        raise ExecutionReadinessError(
            "execution_readiness.profile is a bare path without a sha256 binding; "
            "re-release with profile {path, sha256} so the execution plan is frozen",
            verdict=VERDICT_NOT_PREPARED,
        )
    if not isinstance(profile, dict):
        raise ExecutionReadinessError(
            "execution_readiness.profile must be an object with path and sha256",
            verdict=VERDICT_NOT_PREPARED,
        )
    path_value = profile.get("path")
    sha_value = profile.get("sha256")
    if not isinstance(path_value, str) or not isinstance(sha_value, str):
        raise ExecutionReadinessError(
            "execution_readiness.profile requires path and sha256",
            verdict=VERDICT_NOT_PREPARED,
        )
    resolved = _resolve_artifact_path(path_value, root=repo_root, role="profile")
    return {"path": str(resolved), "recorded_sha256": _normalize_sha(sha_value)}


def fingerprint_qualification_defaults(fingerprint_file: Path | str | None) -> dict[str, Any]:
    """Read qualification-input defaults from a runtime fingerprint document.

    The canonical KNOWN_GOOD runtime fingerprint declares the exact inputs a
    normal run needs (route profile, java requirement, runtime python, and the
    model / data / openevo / seed-root paths). Reading them here means a plain
    ``oe run-campaign`` reaches the same runtime-qualification request — and
    therefore the same cache key — without the operator restating each flag.

    Explicit CLI arguments always win over these defaults, and any field the
    fingerprint does not declare falls back to the legacy default in
    ``build_qualification_args``. A missing or malformed fingerprint yields no
    defaults (never raises).
    """
    if fingerprint_file is None:
        return {}
    try:
        data = json.loads(
            Path(fingerprint_file).expanduser().resolve().read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}

    def _node(key: str) -> dict[str, Any]:
        value = data.get(key)
        return value if isinstance(value, dict) else {}

    def _path(key: str) -> str | None:
        value = _node(key).get("path")
        return value if isinstance(value, str) else None

    route = _node("route")
    host_python = _node("host_runtime_python")
    execution_image = _node("execution_image")
    container_required = bool(data.get("container_image_required", False))
    execution_mode = data.get("execution_mode")

    defaults: dict[str, Any] = {
        "profile": route.get("profile"),
        "require_java": route.get("requires_java"),
        "runtime_container": container_required or execution_mode == "container",
        "runtime_python": host_python.get("executable"),
        "source_root": _path("seed_root"),
        "webshop_data": _path("webshop_data"),
        "model": _path("model"),
        "openevo_root": _path("openevo_core"),
        "index_root": _path("index_root"),
        "image": None,
        "expected_image_digest": None,
    }
    if container_required:
        defaults["image"] = execution_image.get("reference")
        defaults["expected_image_digest"] = (
            execution_image.get("id") or execution_image.get("digest")
        )
    return defaults


def build_qualification_args(
    release: dict[str, Any],
    *,
    fingerprint_file: Path,
    cache_root: Path,
    refresh: bool = False,
    repo_root: Path | None = None,
    runtime_python: Path | str | None = None,
    source_root: Path | str | None = None,
    webshop_data: Path | str | None = None,
    model: Path | str | None = None,
    openevo_root: Path | str | None = None,
    index_root: Path | str | None = None,
    image: str | None = None,
    expected_image_digest: str | None = None,
    runtime_container: bool | None = None,
    require_java: bool | None = None,
    allow_dirty: bool = False,
    profile: str | None = None,
) -> argparse.Namespace:
    """Build the Namespace that ``runtime_qualification.qualify`` expects.

    Any input left as ``None`` is filled from the runtime fingerprint document
    (the canonical KNOWN_GOOD runtime contract), then from the legacy default,
    so a normal ``oe run-campaign`` reproduces the cached qualification request
    without restating profile/runtime/asset flags. Explicit values always win.
    """
    defaults = fingerprint_qualification_defaults(fingerprint_file)

    def _pick(value: Any, key: str, legacy: Any) -> Any:
        if value is not None:
            return value
        candidate = defaults.get(key)
        return candidate if candidate is not None else legacy

    return argparse.Namespace(
        fingerprint_file=Path(fingerprint_file).expanduser().resolve(),
        cache_root=Path(cache_root).expanduser().resolve(),
        repo_root=(repo_root or REPO_ROOT).expanduser().resolve(),
        profile=_pick(profile, "profile", "full"),
        source_root=_resolved(_pick(source_root, "source_root", None)),
        webshop_data=_resolved(_pick(webshop_data, "webshop_data", None)),
        model=_resolved(_pick(model, "model", None)),
        openevo_root=_resolved(_pick(openevo_root, "openevo_root", None)),
        openevo_patch_sha=release.get("upstream_patch", {}).get("sha256"),
        runtime_python=_runtime_executable_path(_pick(runtime_python, "runtime_python", None)),
        index_root=_resolved(_pick(index_root, "index_root", None)),
        image=_pick(image, "image", None),
        expected_image_digest=_pick(
            expected_image_digest, "expected_image_digest", None
        ),
        runtime_container=bool(_pick(runtime_container, "runtime_container", False)),
        require_java=bool(_pick(require_java, "require_java", False)),
        allow_dirty=allow_dirty,
        package=[],
        import_name=[],
        refresh=refresh,
        output=None,
    )


def build_live_gate_args(
    qualification_receipt: Path,
    *,
    output_root: Path,
    resume: bool = False,
    runtime_python: Path | str | None = None,
    cache_root: Path | str | None = None,
    artifact_root: Path | str | None = None,
    storage_root: Path | str | None = None,
    experiment_config: Path | str | None = None,
    run_manifest: Path | str | None = None,
    require_cuda: bool = False,
) -> argparse.Namespace:
    """Build the Namespace that ``runtime_live_gate.run_gate`` expects."""
    return argparse.Namespace(
        qualification_receipt=Path(qualification_receipt).expanduser().resolve(),
        runtime_python=_runtime_executable_path(runtime_python),
        cache_root=_resolved(cache_root),
        artifact_root=_resolved(artifact_root),
        storage_root=_resolved(storage_root),
        output_root=Path(output_root).expanduser().resolve(),
        minimum_free_bytes=1_073_741_824,
        require_cuda=require_cuda,
        resume=resume,
        experiment_config=_resolved(experiment_config),
        run_manifest=_resolved(run_manifest),
        output=None,
    )


def _emit_stage(stages: list[dict[str, Any]], label: str, verdict: str, detail: dict[str, Any] | None = None) -> None:
    entry: dict[str, Any] = {"stage": label, "verdict": verdict}
    if detail:
        entry["detail"] = detail
    stages.append(entry)
    detail_str = " ".join(f"{k}={v}" for k, v in (detail or {}).items())
    if detail_str:
        print(f"[{verdict:<20}] {label} {detail_str}")
    else:
        print(f"[{verdict:<20}] {label}")


def _fail(
    result: dict[str, Any],
    stages: list[dict[str, Any]],
    stage: str,
    verdict: str,
    reason: str,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _emit_stage(stages, stage, verdict, {"reason": reason, **(detail or {})})
    result["error"] = reason
    result["verdict"] = verdict
    return result


def evaluate_readiness(args: argparse.Namespace) -> dict[str, Any]:
    """Run the full readiness chain and emit a READY receipt.

    Returns a dict with:
      - verdict: READY, BLOCKED, BLOCKED_NOT_PREPARED, STALE, or
        REQUIRES_REPREPARATION
      - stages: structured stage log (identical to the persisted receipt)
      - release_receipt_path, release_receipt_sha256
      - ready_receipt_path (if written)
      - qualification_receipt_path, live_gate_receipt_path
      - profile: {path, sha256} when the release declares an execution profile
      - error (human-readable failure reason)
    """
    stages: list[dict[str, Any]] = []
    result: dict[str, Any] = {"stages": stages, "verdict": VERDICT_NOT_PREPARED}
    repo_root = Path(getattr(args, "repo_root", None) or REPO_ROOT).expanduser().resolve()

    # Stage 1: resolve the release receipt (no authority is granted here).
    try:
        release_path = resolve_release_path(
            getattr(args, "campaign", None),
            getattr(args, "release", None),
            current_campaign_path=getattr(args, "current_campaign", None),
            repo_root=repo_root,
        )
        release = load_release_receipt(release_path)
        campaign_id = release["campaign_id"]
    except ExecutionReadinessError as exc:
        return _fail(result, stages, "resolve-campaign", exc.verdict, exc.reason)

    result.update(
        {
            "campaign_id": campaign_id,
            "release_receipt_path": str(release_path),
            "release_receipt_sha256": sha256_file(release_path),
        }
    )
    _emit_stage(stages, "resolve-campaign", "PASS", {"campaign_id": campaign_id})

    # Stage 2: current execution authority. A historical launch-authorized
    # release cannot re-authorize a campaign the router has since closed.
    try:
        authority = verify_current_authority(
            release,
            release_path,
            current_campaign_path=getattr(args, "current_campaign", None),
            repo_root=repo_root,
        )
    except ExecutionReadinessError as exc:
        return _fail(result, stages, "current-authority", exc.verdict, exc.reason)
    result["authority"] = authority
    _emit_stage(
        stages,
        "current-authority",
        "PASS",
        {"router_status": authority["router_status"], "campaign_id": campaign_id},
    )

    # Stage 3: frozen scientific artifacts, with canonical hash semantics.
    try:
        check = hash_bound_artifacts(release, repo_root=repo_root)
    except (OSError, ValueError) as exc:
        return _fail(
            result,
            stages,
            "frozen-scientific-artifacts",
            VERDICT_NOT_PREPARED,
            f"artifact hashing failed: {exc}",
        )

    if check["missing"]:
        first = check["missing"][0]
        result["missing_artifacts"] = check["missing"]
        return _fail(
            result,
            stages,
            "frozen-scientific-artifacts",
            VERDICT_NOT_PREPARED,
            f"missing declared scientific artifact: {first['role']} at {first['path']}",
            {"role": first["role"], "path": first["path"]},
        )

    if check["mismatched"]:
        first = check["mismatched"][0]
        verdict = (
            VERDICT_REPREPARE
            if first["role"].split(":")[0] in SCIENTIFIC_IDENTITY_ROLES
            else VERDICT_STALE
        )
        detail = {
            "role": first["role"],
            "path": first["path"],
            "recorded": first.get("recorded_sha", "missing"),
            "observed": first.get("observed_sha", "missing"),
        }
        result["stale_artifacts"] = check["mismatched"]
        return _fail(
            result,
            stages,
            "frozen-scientific-artifacts",
            verdict,
            f"{'scientific identity changed' if verdict == VERDICT_REPREPARE else 'stale artifact'}: "
            f"{first['role']} at {first['path']}",
            detail,
        )

    declared = _collect_declared_artifacts(release, repo_root)
    _emit_stage(
        stages,
        "frozen-scientific-artifacts",
        "PASS",
        {"bound_artifacts": len(declared)},
    )

    # Stage 4: scientific identity document (explicit, re-checkable digest).
    prereg_path = _resolve_artifact_path(
        release["preregistration"]["path"], root=repo_root, role="preregistration"
    )
    identity = scientific_identity_document(prereg_path)
    if identity is not None:
        result["scientific_identity_sha256"] = identity["sha256"]
        _emit_stage(
            stages,
            "scientific-identity",
            "PASS",
            {"identity_sha256": identity["sha256"][:23]},
        )

    # Stage 5: execution profile/plan must be hash-bound and intact.
    profile_spec: dict[str, Any] | None = None
    try:
        profile_spec = resolve_execution_profile(release, repo_root=repo_root)
    except ExecutionReadinessError as exc:
        return _fail(result, stages, "execution-profile", exc.verdict, exc.reason)
    if profile_spec is not None:
        profile_path = Path(profile_spec["path"])
        if not profile_path.is_file():
            return _fail(
                result,
                stages,
                "execution-profile",
                VERDICT_NOT_PREPARED,
                f"declared execution profile is missing: {profile_path}",
            )
        observed_profile_sha = sha256_file(profile_path)
        if observed_profile_sha != profile_spec["recorded_sha256"]:
            return _fail(
                result,
                stages,
                "execution-profile",
                VERDICT_STALE,
                "execution profile bytes changed since the release froze them",
                {
                    "path": str(profile_path),
                    "recorded": profile_spec["recorded_sha256"],
                    "observed": observed_profile_sha,
                },
            )
        profile_spec["sha256"] = observed_profile_sha
        result["profile"] = {"path": str(profile_path), "sha256": observed_profile_sha}
        result["profile_path"] = str(profile_path)
        _emit_stage(
            stages,
            "execution-profile",
            "PASS",
            {"profile_sha256": observed_profile_sha[:23]},
        )

    # Stage 6: runtime qualification (cached when the fingerprint is unchanged).
    fingerprint_file = getattr(args, "fingerprint_file", None)
    cache_root = getattr(args, "cache_root", None)
    if fingerprint_file is None or cache_root is None:
        return _fail(
            result,
            stages,
            "runtime-qualification",
            VERDICT_NOT_PREPARED,
            "--fingerprint-file and --cache-root are required",
        )

    qargs = build_qualification_args(
        release,
        fingerprint_file=Path(fingerprint_file),
        cache_root=Path(cache_root),
        refresh=bool(getattr(args, "refresh_qualification", False)),
        repo_root=repo_root,
        runtime_python=getattr(args, "runtime_python", None),
        source_root=getattr(args, "source_root", None),
        webshop_data=getattr(args, "webshop_data", None),
        model=getattr(args, "model", None),
        openevo_root=getattr(args, "openevo_root", None),
        index_root=getattr(args, "index_root", None),
        image=getattr(args, "image", None),
        expected_image_digest=getattr(args, "expected_image_digest", None),
        runtime_container=getattr(args, "runtime_container", None),
        require_java=getattr(args, "require_java", None),
        allow_dirty=bool(getattr(args, "allow_dirty", False)),
        profile=getattr(args, "profile", None),
    )

    try:
        qualification = runtime_qualification.qualify(qargs)
    except (OSError, ValueError) as exc:
        return _fail(
            result, stages, "runtime-qualification", VERDICT_NOT_PREPARED,
            f"runtime qualification failed: {exc}",
        )

    result["runtime_qualification_decision"] = qualification.get("qualification_decision")
    if qualification.get("status") != "PASS":
        _emit_stage(
            stages,
            "runtime-qualification",
            "FAIL",
            {"failures": qualification.get("failures", [])[:5]},
        )
        result["error"] = "runtime qualification did not pass"
        result["verdict"] = VERDICT_NOT_PREPARED
        return result

    cache_key = qualification.get("cache_key", "")
    fingerprint_sha = qualification.get("request", {}).get("runtime_fingerprint_sha256", "")
    qpath = Path(qualification["cache_path"])
    _emit_stage(
        stages,
        "runtime-qualification",
        "CACHE_HIT" if qualification.get("reused") else "PASS",
        {
            "cache_key": cache_key[:16] if cache_key else "",
            "fingerprint": fingerprint_sha[:23] if fingerprint_sha else "",
            "qualification_decision": qualification.get("qualification_decision"),
        },
    )

    # Stage 7: live volatile gate. Control receipts live in a dedicated
    # control layer so they never collide with formal run output semantics.
    output_root = getattr(args, "output_root", None)
    if output_root is None:
        return _fail(
            result, stages, "live-gate", VERDICT_NOT_PREPARED, "--output-root is required"
        )
    control_dir = Path(output_root).expanduser().resolve() / "control"

    largs = build_live_gate_args(
        qualification_receipt=qpath,
        output_root=control_dir,
        resume=True,  # control receipts are idempotent platform receipts
        runtime_python=getattr(args, "runtime_python", None),
        cache_root=getattr(args, "cache_root", None),
        artifact_root=getattr(args, "artifact_root", None),
        storage_root=getattr(args, "storage_root", None),
        experiment_config=getattr(args, "experiment_config", None),
        run_manifest=getattr(args, "run_manifest", None),
        require_cuda=bool(getattr(args, "require_cuda", False)),
    )

    try:
        live_gate = runtime_live_gate.run_gate(largs)
    except (OSError, ValueError) as exc:
        return _fail(
            result, stages, "live-gate", VERDICT_NOT_PREPARED, f"live gate failed: {exc}"
        )

    if live_gate.get("status") != "PASS":
        _emit_stage(
            stages,
            "live-gate",
            "FAIL",
            {"failures": live_gate.get("failures", [])[:5]},
        )
        result["error"] = "live gate did not pass"
        result["verdict"] = VERDICT_NOT_PREPARED
        return result

    # Formal run output collision is checked against the *formal* run root,
    # never against the control layer.
    run_output_root = getattr(args, "run_output_root", None)
    if run_output_root is not None:
        collision = runtime_live_gate.check_output_collision(
            Path(run_output_root), resume=bool(getattr(args, "resume", False))
        )
        if collision.get("status") == "fail":
            return _fail(
                result,
                stages,
                "live-gate",
                VERDICT_NOT_PREPARED,
                f"formal run output collision: {collision['errors'][0]}",
                {"path": collision.get("path")},
            )

    evidence_id = uuid.uuid4().hex
    live_gate_path = control_dir / f"live-gate-receipt-{evidence_id}.json"
    exclusive_json(live_gate_path, live_gate)
    atomic_json(control_dir / "live-gate-receipt.json", live_gate)
    _emit_stage(
        stages,
        "live-gate",
        "PASS",
        {"receipt_sha256": sha256_file(live_gate_path)[:23]},
    )

    # Stage 8: READY receipt. The stage entry is appended *before* writing so
    # the persisted receipt and the in-memory result carry identical stages.
    frozen_artifacts = [
        {
            "role": a["role"],
            "path": a["path"],
            "sha256": a.get("recorded_sha"),
            "hash_kind": a["hash_kind"],
        }
        for a in declared
    ]

    ready_path = control_dir / f"execution-readiness-{evidence_id}.json"
    _emit_stage(stages, "ready-receipt", "PASS", {"path": str(ready_path)})

    ready_receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign_id": campaign_id,
        "verdict": VERDICT_READY,
        "status_detail": "PASS",
        "release_receipt": {
            "path": str(release_path),
            "sha256": sha256_file(release_path),
        },
        "authority": authority,
        "preregistration": release["preregistration"],
        "path_contract": release["path_contract"],
        "upstream_patch": release.get("upstream_patch"),
        "scientific_identity_sha256": result.get("scientific_identity_sha256"),
        "frozen_scientific_artifacts": frozen_artifacts,
        "runtime_fingerprint_sha256": fingerprint_sha,
        "qualification_cache_key": cache_key,
        "qualification_receipt_path": str(qpath),
        "runtime_qualification_decision": qualification.get("qualification_decision"),
        "qualification_receipt_sha256": sha256_file(qpath),
        "live_gate_receipt_path": str(live_gate_path),
        "live_gate_receipt_sha256": sha256_file(live_gate_path),
        "ready_at_epoch": time.time(),
        "stages": stages,
        "formal_task_consumption_allowed": True,
    }

    if profile_spec is not None:
        ready_receipt["profile"] = {"path": str(profile_path), "sha256": profile_spec["sha256"]}

    exclusive_json(ready_path, ready_receipt)
    atomic_json(control_dir / "execution-readiness.json", ready_receipt)

    result.update(
        {
            "verdict": VERDICT_READY,
            "ready_receipt_path": str(ready_path),
            "qualification_receipt_path": str(qpath),
            "live_gate_receipt_path": str(live_gate_path),
            "qualification_reused": bool(qualification.get("reused")),
        }
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, help="path to execution release receipt")
    parser.add_argument("--campaign", help="campaign id (resolves via current-campaign.json)")
    parser.add_argument(
        "--current-campaign",
        type=Path,
        default=CURRENT_CAMPAIGN_PATH,
        help="path to current-campaign router",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--fingerprint-file", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--run-output-root",
        type=Path,
        help="formal run output root to collision-check (distinct from the control layer)",
    )
    parser.add_argument("--runtime-python", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--webshop-data", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--openevo-root", type=Path)
    parser.add_argument("--index-root", type=Path)
    parser.add_argument("--image")
    parser.add_argument("--expected-image-digest")
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--storage-root", type=Path)
    parser.add_argument("--experiment-config", type=Path)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--profile", default="full")
    parser.add_argument("--runtime-container", action="store_true")
    parser.add_argument("--require-java", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--refresh-qualification", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = evaluate_readiness(args)
    verdict = result.get("verdict", VERDICT_NOT_PREPARED)
    print(f"EXECUTION_READINESS: {verdict}")
    if verdict != VERDICT_READY:
        return 2
    if args.dry_run:
        print("DRY_RUN: ready but not launching")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
