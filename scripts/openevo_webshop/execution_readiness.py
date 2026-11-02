#!/usr/bin/env python3
"""Fail-closed facade for execution readiness.

The historical implementation is preserved in ``execution_readiness_legacy`` so
this facade can harden the campaign boundary without duplicating the runtime and
artifact machinery that already passed the round-2 test suite.

The key rule is fail-fast: a campaign that lacks current authority or explicit
scientific identity is rejected immediately; a missing hash-bound execution
profile is rejected at the profile stage, still *before* runtime qualification.
Stable runtime facts remain cacheable; scientific authority is never inferred
from a runtime cache hit.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any


_WRAPPER_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_WRAPPER_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_WRAPPER_REPO_ROOT))

from scripts.openevo_webshop import execution_readiness_legacy as _legacy  # noqa: E402
from scripts.openevo_webshop import experiment_passport_gate  # noqa: E402
from scripts.validate_run_manifest import validate as validate_run_manifest  # noqa: E402


# Re-export the existing public surface so callers/tests keep the same API.
for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_legacy, _name)


_LEGACY_RESOLVE_RELEASE_PATH = _legacy.resolve_release_path
_LEGACY_VERIFY_CURRENT_AUTHORITY = _legacy.verify_current_authority
_LEGACY_RESOLVE_EXECUTION_PROFILE = _legacy.resolve_execution_profile
_LEGACY_SCIENTIFIC_IDENTITY_DOCUMENT = _legacy.scientific_identity_document
_LEGACY_EVALUATE_READINESS = _legacy.evaluate_readiness
_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")


def _require_router_release_sha256(router: dict[str, Any]) -> str:
    """Return the router's bound release digest or fail closed.

    Current authority is a binding to *bytes*, not merely to a pathname. A
    missing/malformed digest therefore cannot authorize formal task use.
    """
    latest = router.get("latest_event") if isinstance(router.get("latest_event"), dict) else {}
    value = latest.get("release_receipt_sha256") or router.get("release_receipt_sha256")
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ExecutionReadinessError(
            "current campaign router must bind release_receipt_sha256 before execution",
            verdict=VERDICT_BLOCKED,
        )
    normalized = _legacy._normalize_sha(value)
    assert normalized is not None
    return normalized


def _safe_router_pointer(router: dict[str, Any], *, root: Path) -> str | None:
    """Require the router release pointer to resolve *inside* the repo root."""
    pointer = _legacy._router_release_pointer(router)
    if pointer is None:
        return None
    raw = Path(pointer).expanduser()
    target = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    if target != root and root not in target.parents:
        raise ExecutionReadinessError(
            f"current-campaign release_receipt path escapes the repository/artifact boundary: {pointer}",
            verdict=VERDICT_BLOCKED,
        )
    return pointer


def resolve_release_path(
    campaign_id: str | None,
    release_path: Path | str | None,
    *,
    current_campaign_path: Path | str | None = None,
    repo_root: Path | str | None = None,
) -> Path:
    """Resolve a release while refusing router pointers that escape the repo."""
    root = Path(repo_root).expanduser().resolve() if repo_root else REPO_ROOT
    if release_path is None and campaign_id is not None:
        router = _legacy._load_router(current_campaign_path, root=root)
        _safe_router_pointer(router, root=root)

    resolved = _LEGACY_RESOLVE_RELEASE_PATH(
        campaign_id,
        release_path,
        current_campaign_path=current_campaign_path,
        repo_root=root,
    )
    if resolved != root and root not in resolved.parents:
        raise ExecutionReadinessError(
            f"release receipt is outside the repository/artifact boundary: {resolved}",
            verdict=VERDICT_BLOCKED,
        )
    return resolved


def verify_current_authority(
    release: dict[str, Any],
    release_path: Path,
    *,
    current_campaign_path: Path | str | None = None,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Require a repo-contained pointer and exact router-bound release bytes."""
    root = Path(repo_root).expanduser().resolve() if repo_root else REPO_ROOT
    router = _legacy._load_router(current_campaign_path, root=root)
    _safe_router_pointer(router, root=root)
    recorded_sha = _require_router_release_sha256(router)
    observed_sha = _legacy.sha256_file(release_path.expanduser().resolve())
    if recorded_sha != observed_sha:
        raise ExecutionReadinessError(
            "current campaign router binds different release receipt bytes; "
            "the supplied release is not the current authority",
            verdict=VERDICT_BLOCKED,
        )
    return _LEGACY_VERIFY_CURRENT_AUTHORITY(
        release,
        release_path,
        current_campaign_path=current_campaign_path,
        repo_root=root,
    )


def require_scientific_identity_document(prereg_path: Path) -> dict[str, Any]:
    """Require a non-empty frozen scientific contract before execution."""
    identity = _LEGACY_SCIENTIFIC_IDENTITY_DOCUMENT(prereg_path)
    if identity is None:
        raise ExecutionReadinessError(
            "preregistration lacks a non-empty frozen_scientific_contract; "
            "scientific identity must be explicit before execution",
            verdict=VERDICT_NOT_PREPARED,
        )
    return identity


def resolve_execution_profile(release: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    """Require a hash-bound execution profile; absence is not READY."""
    readiness = release.get("execution_readiness")
    if not isinstance(readiness, dict) or readiness.get("profile") is None:
        raise ExecutionReadinessError(
            "release receipt lacks mandatory hash-bound execution_readiness.profile {path, sha256}; "
            "prepare and freeze the execution plan before runtime qualification",
            verdict=VERDICT_NOT_PREPARED,
        )
    profile = _LEGACY_RESOLVE_EXECUTION_PROFILE(release, repo_root=repo_root)
    if profile is None:
        raise ExecutionReadinessError(
            "release receipt lacks a hash-bound execution profile",
            verdict=VERDICT_NOT_PREPARED,
        )
    return profile




def validate_run_manifest_passport_link(
    run_manifest: Path | str | None, passport_gate: dict[str, Any], *, repo_root: Path
) -> dict[str, Any]:
    """Bind a supplied Run Manifest to the exact Passport launch identity.

    The Run Manifest validator owns shape/pair semantics. This readiness layer
    owns the cross-artifact equality: a Passport-bound formal release may only
    carry a manifest with the same canonical experiment ID and Passport hash.
    """
    if run_manifest is None:
        return {"status": "NOT_REQUESTED", "reason": "no run manifest supplied to readiness"}
    path = Path(run_manifest).expanduser().resolve()
    errors = validate_run_manifest(path)
    if errors:
        raise ExecutionReadinessError(
            "Run Manifest contract failed before runtime qualification: " + errors[0],
            verdict=VERDICT_NOT_PREPARED,
        )
    document = _legacy.json.loads(path.read_text(encoding="utf-8"))
    experiment = document["experiment"]
    manifest_id = experiment.get("experiment_id")
    manifest_passport = experiment.get("passport_sha256")
    if passport_gate.get("status") == "PASS":
        if manifest_id != passport_gate.get("experiment_id"):
            raise ExecutionReadinessError(
                "Run Manifest experiment_id does not match the release/Passport binding",
                verdict=VERDICT_NOT_PREPARED,
            )
        if manifest_passport != passport_gate.get("passport_sha256"):
            raise ExecutionReadinessError(
                "Run Manifest passport_sha256 does not match the release/Registry binding",
                verdict=VERDICT_NOT_PREPARED,
            )
        return {
            "status": "PASS", "path": str(path),
            "experiment_id": manifest_id, "passport_sha256": manifest_passport,
            "reason": "Run Manifest and FROZEN Passport launch identity match",
        }
    if manifest_id is None and manifest_passport is None:
        return {
            "status": "UNBOUND_MIGRATION", "path": str(path),
            "experiment_id": None, "passport_sha256": None,
            "reason": "legacy/unregistered manifest retains explicit null cross-link slots",
        }
    try:
        verified = experiment_passport_gate.validate_binding(
            repo_root, {"experiment_id": manifest_id, "passport_sha256": manifest_passport}
        )
    except experiment_passport_gate.PassportGateError as exc:
        raise ExecutionReadinessError(
            f"Run Manifest Passport cross-link is invalid: {exc}", verdict=VERDICT_NOT_PREPARED
        ) from exc
    return {
        "status": "PASS", "path": str(path),
        "experiment_id": verified["experiment_id"],
        "passport_sha256": verified["passport_sha256"],
        "reason": "legacy release carries an independently verified Passport-bound manifest",
    }


def _fail_fast_contract(args: Any) -> dict[str, Any] | None:
    """Cheap authority/scientific-identity checks before runtime work.

    Profile validation stays in the canonical Stage 5 so artifact drift keeps
    its existing Stage-3 classification, but Stage 5 is now mandatory and
    therefore still blocks before runtime qualification.
    """
    stages: list[dict[str, Any]] = []
    result: dict[str, Any] = {"stages": stages, "verdict": VERDICT_NOT_PREPARED}
    root = Path(getattr(args, "repo_root", None) or REPO_ROOT).expanduser().resolve()

    try:
        release_path = resolve_release_path(
            getattr(args, "campaign", None),
            getattr(args, "release", None),
            current_campaign_path=getattr(args, "current_campaign", None),
            repo_root=root,
        )
        release = _legacy.load_release_receipt(release_path)
    except ExecutionReadinessError as exc:
        return _legacy._fail(result, stages, "resolve-campaign", exc.verdict, exc.reason)

    try:
        verify_current_authority(
            release,
            release_path,
            current_campaign_path=getattr(args, "current_campaign", None),
            repo_root=root,
        )
    except ExecutionReadinessError as exc:
        return _legacy._fail(result, stages, "current-authority", exc.verdict, exc.reason)

    passport_gate = experiment_passport_gate.evaluate_release_gate(release, root)
    if passport_gate["status"] == "FAIL":
        return _legacy._fail(
            result, stages, "experiment-passport", VERDICT_NOT_PREPARED,
            str(passport_gate.get("reason") or "FROZEN Experiment Passport gate failed"),
        )
    try:
        validate_run_manifest_passport_link(
            getattr(args, "run_manifest", None), passport_gate, repo_root=root
        )
    except ExecutionReadinessError as exc:
        return _legacy._fail(
            result, stages, "run-manifest-passport-link", exc.verdict, exc.reason
        )

    prereg_path = _legacy._resolve_artifact_path(
        release["preregistration"]["path"], root=root, role="preregistration"
    )
    # Preserve the legacy STALE classification if the preregistration bytes
    # themselves drifted; Stage 3 will report that before any runtime work.
    recorded_prereg = _legacy._normalize_sha(release["preregistration"].get("sha256"))
    if prereg_path.is_file() and _legacy.sha256_file(prereg_path) == recorded_prereg:
        try:
            require_scientific_identity_document(prereg_path)
        except ExecutionReadinessError as exc:
            return _legacy._fail(result, stages, "scientific-identity", exc.verdict, exc.reason)
    return None


def _passport_gate_for_args(args: Any) -> dict[str, Any]:
    root = Path(getattr(args, "repo_root", None) or REPO_ROOT).expanduser().resolve()
    release_path = resolve_release_path(
        getattr(args, "campaign", None), getattr(args, "release", None),
        current_campaign_path=getattr(args, "current_campaign", None), repo_root=root,
    )
    release = _legacy.load_release_receipt(release_path)
    return experiment_passport_gate.evaluate_release_gate(release, root)


def evaluate_readiness(args: Any) -> dict[str, Any]:
    """Fail fast on non-executable science, then run the full canonical chain."""
    failure = _fail_fast_contract(args)
    if failure is not None:
        return failure
    gate = _passport_gate_for_args(args)
    root = Path(getattr(args, "repo_root", None) or REPO_ROOT).expanduser().resolve()
    manifest_link = validate_run_manifest_passport_link(
        getattr(args, "run_manifest", None), gate, repo_root=root
    )
    result = _LEGACY_EVALUATE_READINESS(args)
    result["experiment_passport"] = gate
    result["run_manifest_passport_link"] = manifest_link
    ready_path = result.get("ready_receipt_path")
    if result.get("verdict") == VERDICT_READY and isinstance(ready_path, str):
        path = Path(ready_path)
        if path.is_file():
            persisted = _legacy.json.loads(path.read_text(encoding="utf-8"))
            persisted["experiment_passport"] = gate
            persisted["run_manifest_passport_link"] = manifest_link
            _legacy.atomic_json(path, persisted)
    return result


# Harden the globals used by the preserved evaluator. Its function object
# resolves these names in the legacy module at call time, so the full receipt
# path receives the same fail-closed semantics as this facade.
_legacy.resolve_release_path = resolve_release_path
_legacy.verify_current_authority = verify_current_authority
_legacy.resolve_execution_profile = resolve_execution_profile


def main(argv: list[str] | None = None) -> int:
    args = _legacy.parse_args(argv)
    result = evaluate_readiness(args)
    verdict = result.get("verdict", VERDICT_NOT_PREPARED)
    print(f"EXECUTION_READINESS: {verdict}")
    if verdict != VERDICT_READY:
        return 2
    if args.dry_run:
        print("DRY_RUN: ready but not launching")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
