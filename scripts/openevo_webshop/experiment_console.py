#!/usr/bin/env python3
"""Read-only status console for the major OpenEvo experiment lineages.

The console deliberately separates durable repository authority from volatile
server observations.  It never launches, resumes, stops, edits, or migrates a
scientific run.  Execution remains owned by the current campaign/release and
its exact launcher/recovery authority.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = ROOT / "configs" / "experiment" / "experiment-console-202609110009.json"
SCHEMA = "openevo.experiment-console-index.202609110009"
SOURCE_TYPES = {"static", "repo_json", "server_json", "release_external_executor"}
LIVE_FIELDS = (
    "next_round",
    "rollouts_consumed",
    "sd_generation_count",
    "sd_component_count",
    "vnext_candidate_accept_count",  # legacy-revision-compat: existing live-state field
    "vnext_candidate_reject_count",  # legacy-revision-compat: existing live-state field
    "vnext_sd_candidate_generation_count",  # legacy-revision-compat: existing live-state field
    "external_teacher_calls",
    "final_panel_access_count",
    "blocker",
    "blocked_at_epoch",
    "twin_control_git_sha",
    "twin_runtime_semantics",
    "current_sd_artifact_id",
)


class ConsoleError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConsoleError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConsoleError(f"JSON object required: {path}")
    return value


def _repo_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ConsoleError(f"repository path must be relative: {value}")
    resolved = (root / path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ConsoleError(f"repository path escapes root: {value}")
    return resolved


def _path_get(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _mismatches(value: Mapping[str, Any], expected: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, wanted in expected.items():
        observed = _path_get(value, key)
        if observed != wanted:
            out.append({"field": key, "expected": wanted, "observed": observed})
    return out


def _extract_live(value: Mapping[str, Any]) -> dict[str, Any]:
    metrics = {key: value.get(key) for key in LIVE_FIELDS if value.get(key) is not None}
    rounds = value.get("rounds_complete")
    if isinstance(rounds, list):
        metrics["rounds_complete"] = len(rounds)
        if rounds:
            metrics["last_completed_round"] = rounds[-1]
    return metrics


def validate_catalog(root: Path, catalog: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if catalog.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    if catalog.get("role") != "operator-observability-only; never scientific or execution authority":
        errors.append("catalog role must remain explicitly read-only/non-authoritative")

    runtimes = catalog.get("runtime_assets")
    if not isinstance(runtimes, Mapping):
        errors.append("runtime_assets must be an object")
        runtimes = {}
    for runtime_id, spec in runtimes.items():
        if not isinstance(spec, Mapping):
            errors.append(f"runtime {runtime_id}: object required")
            continue
        rel = spec.get("asset_path")
        if not isinstance(rel, str):
            errors.append(f"runtime {runtime_id}: asset_path required")
            continue
        try:
            asset_path = _repo_path(root, rel)
            asset = _read_json(asset_path)
        except ConsoleError as exc:
            errors.append(str(exc))
            continue
        wanted = spec.get("expected_image_digest")
        if wanted and asset.get("image_digest") != wanted:
            errors.append(f"runtime {runtime_id}: image digest mismatch")

    rows = catalog.get("experiments")
    if not isinstance(rows, list) or not rows:
        errors.append("experiments must be a non-empty array")
        return errors
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            errors.append("experiment row must be an object")
            continue
        experiment_id = row.get("id")
        if not isinstance(experiment_id, str) or not experiment_id:
            errors.append("experiment id required")
            continue
        if experiment_id in seen:
            errors.append(f"duplicate experiment id: {experiment_id}")
        seen.add(experiment_id)
        for key in ("display_name", "model", "treatment", "what_changed", "what_stays_fixed", "state_source", "runtime", "resume", "authority_paths"):
            if key not in row:
                errors.append(f"{experiment_id}: missing {key}")
        source = row.get("state_source")
        if not isinstance(source, Mapping) or source.get("type") not in SOURCE_TYPES:
            errors.append(f"{experiment_id}: unsupported state_source")
        runtime = row.get("runtime")
        if isinstance(runtime, Mapping) and runtime.get("asset") and runtime.get("asset") not in runtimes:
            errors.append(f"{experiment_id}: unknown runtime asset {runtime.get('asset')}")
        for rel in row.get("authority_paths") or []:
            if not isinstance(rel, str):
                errors.append(f"{experiment_id}: authority path must be a string")
                continue
            try:
                path = _repo_path(root, rel)
            except ConsoleError as exc:
                errors.append(str(exc))
                continue
            if not path.exists():
                errors.append(f"{experiment_id}: authority path missing: {rel}")
    return errors


def _observe_repo_json(root: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    evidence: list[str] = []
    mismatches: list[dict[str, Any]] = []
    for document in source.get("documents") or []:
        path = _repo_path(root, str(document["path"]))
        payload = _read_json(path)
        evidence.append(str(path.relative_to(root)))
        mismatches.extend(_mismatches(payload, document.get("required_equal") or {}))
        for name, dotted in (document.get("metrics") or {}).items():
            metrics[name] = _path_get(payload, str(dotted))
    if mismatches:
        return {
            "effective_status": "IDENTITY_MISMATCH",
            "observation_scope": "repository",
            "identity_trust": "FAIL",
            "identity_mismatches": mismatches,
            "metrics": metrics,
            "evidence": evidence,
        }
    return {
        "effective_status": source.get("status", "OBSERVED"),
        "observation_scope": "repository",
        "identity_trust": "PASS",
        "metrics": metrics,
        "evidence": evidence,
    }


def _server_unavailable(path: Path, note: str | None = None) -> dict[str, Any]:
    server_root_visible = Path("/workspace").exists()
    status = "STATE_MISSING_ON_SERVER" if server_root_visible else "LIVE_UNAVAILABLE_OFF_SERVER"
    return {
        "effective_status": status,
        "observation_scope": "server" if server_root_visible else "off-server",
        "identity_trust": "NOT_OBSERVED",
        "state_path": str(path),
        "note": note or "No live state was read; durable authority is unchanged.",
        "metrics": {},
    }


def _observe_server_json(source: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(source.get("path", "")))
    if not path.is_absolute():
        raise ConsoleError(f"server_json path must be absolute: {path}")
    if not path.exists():
        return _server_unavailable(path)
    payload = _read_json(path)
    mismatch = _mismatches(payload, source.get("required_equal") or {})
    if mismatch:
        return {
            "effective_status": "IDENTITY_MISMATCH",
            "observation_scope": "server",
            "identity_trust": "FAIL",
            "state_path": str(path),
            "identity_mismatches": mismatch,
            "metrics": _extract_live(payload),
        }
    return {
        "effective_status": str(payload.get("status") or "OBSERVED"),
        "observation_scope": "server",
        "identity_trust": "PASS",
        "state_path": str(path),
        "metrics": _extract_live(payload),
    }


def _observe_release_external_executor(root: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    release_path = _repo_path(root, str(source.get("release_path", "")))
    release = _read_json(release_path)
    external = release.get("external_executor")
    if not isinstance(external, Mapping):
        return {
            "effective_status": "NO_EXTERNAL_EXECUTOR_CONTRACT",
            "observation_scope": "repository",
            "identity_trust": "FAIL",
            "release_path": str(release_path.relative_to(root)),
            "metrics": {},
        }
    state_path = Path(str(external.get("state_path", "")))
    if not state_path.is_absolute():
        raise ConsoleError(f"external executor state path must be absolute: {state_path}")
    if not state_path.exists():
        missing = _server_unavailable(state_path, str(external.get("missing_reason") or "external executor state is absent"))
        if missing["observation_scope"] == "server":
            missing["effective_status"] = str(external.get("missing_state") or "STATE_MISSING_ON_SERVER")
        missing["release_status"] = release.get("status")
        missing["release_path"] = str(release_path.relative_to(root))
        return missing

    payload = _read_json(state_path)
    mismatch = _mismatches(payload, external.get("required_equal") or {})
    if mismatch:
        return {
            "effective_status": "IDENTITY_MISMATCH",
            "observation_scope": "server",
            "identity_trust": "FAIL",
            "release_status": release.get("status"),
            "release_path": str(release_path.relative_to(root)),
            "state_path": str(state_path),
            "identity_mismatches": mismatch,
            "metrics": _extract_live(payload),
        }
    status_map = external.get("status_map") or {}
    raw_status = str(payload.get("status") or "OBSERVED")
    return {
        "effective_status": str(status_map.get(raw_status, raw_status)),
        "raw_live_status": raw_status,
        "observation_scope": "server",
        "identity_trust": "PASS",
        "release_status": release.get("status"),
        "release_path": str(release_path.relative_to(root)),
        "state_path": str(state_path),
        "metrics": _extract_live(payload),
    }


def observe(root: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    source = row["state_source"]
    source_type = source["type"]
    if source_type == "static":
        return {
            "effective_status": source.get("status", "STATIC"),
            "observation_scope": "repository-index",
            "identity_trust": "INDEX_ONLY",
            "note": source.get("note"),
            "metrics": {},
        }
    if source_type == "repo_json":
        return _observe_repo_json(root, source)
    if source_type == "server_json":
        return _observe_server_json(source)
    if source_type == "release_external_executor":
        return _observe_release_external_executor(root, source)
    raise ConsoleError(f"unsupported source type: {source_type}")


def runtime_snapshot(root: Path, catalog: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    runtime = dict(row.get("runtime") or {})
    asset_id = runtime.get("asset")
    if not asset_id:
        return runtime
    spec = catalog["runtime_assets"][asset_id]
    asset_path = _repo_path(root, spec["asset_path"])
    asset = _read_json(asset_path)
    runtime.update({
        "asset_id": asset_id,
        "asset_path": spec["asset_path"],
        "image_digest": asset.get("image_digest"),
        "local_tag": asset.get("local_tag"),
        "hf_archive": asset.get("hf_archive"),
    })
    return runtime


def build_snapshot(root: Path, catalog: Mapping[str, Any]) -> dict[str, Any]:
    errors = validate_catalog(root, catalog)
    if errors:
        raise ConsoleError("catalog validation failed: " + "; ".join(errors))
    rows = []
    from scripts.openevo_webshop.experiment_registry import RegistryError, console_static_overlay
    for legacy_item in catalog["experiments"]:
        item = dict(legacy_item)
        static_source = "legacy-console-compat"
        canonical_id = None
        passport_sha = None
        try:
            overlay = console_static_overlay(root, str(item["id"]))
        except RegistryError as exc:
            raise ConsoleError(f"Registry static identity invalid: {exc}") from exc
        if overlay is not None:
            canonical_id = overlay.pop("canonical_experiment_id")
            passport_sha = overlay.pop("passport_sha256")
            item.update(overlay)
            static_source = "experiment-registry/passport"
        rows.append({
            "id": item["id"],
            "canonical_experiment_id": canonical_id,
            "passport_sha256": passport_sha,
            "static_identity_source": static_source,
            "display_name": item["display_name"],
            "model": item["model"],
            "treatment": item["treatment"],
            "what_changed": list(item["what_changed"]),
            "what_stays_fixed": list(item["what_stays_fixed"]),
            "observation": observe(root, item),
            "runtime": runtime_snapshot(root, catalog, item),
            "resume": dict(item["resume"]),
            "authority_paths": list(item["authority_paths"]),
        })
    return {
        "schema": "openevo.experiment-console-snapshot.202609110009",
        "catalog_role": catalog["role"],
        "observed_at_epoch": time.time(),
        "mutation": False,
        "experiments": rows,
    }


def _progress(metrics: Mapping[str, Any]) -> str:
    bits: list[str] = []
    rounds = metrics.get("rounds_complete", metrics.get("rounds"))
    if rounds is not None:
        bits.append(f"{rounds} rounds")
    rollouts = metrics.get("rollouts_consumed", metrics.get("rollouts"))
    if rollouts is not None:
        bits.append(f"{rollouts} rollouts")
    updates = metrics.get("accepted_parametric_updates", metrics.get("vnext_candidate_accept_count"))  # legacy-revision-compat
    if updates is not None:
        bits.append(f"{updates} updates")
    rejects = metrics.get("vnext_candidate_reject_count")  # legacy-revision-compat
    if rejects is not None:
        bits.append(f"{rejects} rejects")
    return ", ".join(bits) if bits else "-"


def _short_model(model: str) -> str:
    return model.split("/")[-1]


def render_status(snapshot: Mapping[str, Any]) -> str:
    rows = snapshot["experiments"]
    widths = {"id": 34, "model": 23, "status": 25, "resume": 29}
    header = (
        f"{'EXPERIMENT':<{widths['id']}} {'MODEL':<{widths['model']}} "
        f"{'STATUS':<{widths['status']}} {'RESUME':<{widths['resume']}} PROGRESS"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        status = str(row["observation"]["effective_status"])
        resume = str(row["resume"].get("policy", "-"))
        lines.append(
            f"{row['id']:<{widths['id']}} {_short_model(row['model']):<{widths['model']}} "
            f"{status:<{widths['status']}} {resume:<{widths['resume']}} "
            f"{_progress(row['observation'].get('metrics') or {})}"
        )
    lines.extend([
        "",
        "Read-only view: repository authority and live server state are kept separate.",
        "Use `./scripts/oe experiments show <id>` for what changed, what stayed fixed, runtime identity, and authority paths.",
    ])
    return "\n".join(lines)


def _find(snapshot: Mapping[str, Any], experiment_id: str) -> Mapping[str, Any]:
    for row in snapshot["experiments"]:
        if row["id"] == experiment_id:
            return row
    choices = ", ".join(row["id"] for row in snapshot["experiments"])
    raise ConsoleError(f"unknown experiment id {experiment_id!r}; choose one of: {choices}")


def render_show(row: Mapping[str, Any]) -> str:
    observation = row["observation"]
    runtime = row["runtime"]
    lines = [
        f"Experiment: {row['display_name']} ({row['id']})",
        f"Canonical ID: {row.get('canonical_experiment_id') or 'legacy console only'}",
        f"Static identity source: {row.get('static_identity_source')}",
        f"Passport: {row.get('passport_sha256') or 'not yet backfilled'}",
        f"Model: {row['model']}",
        f"Treatment: {row['treatment']}",
        f"Status now: {observation['effective_status']} [{observation.get('observation_scope')}]",
        f"Identity check: {observation.get('identity_trust')}",
        f"Progress: {_progress(observation.get('metrics') or {})}",
        "",
        "What changed:",
    ]
    lines.extend(f"- {item}" for item in row["what_changed"])
    lines.append("What stayed fixed:")
    lines.extend(f"- {item}" for item in row["what_stays_fixed"])
    lines.extend(["", "Runtime:"])
    if runtime.get("image_digest"):
        lines.append(f"- image digest: {runtime['image_digest']}")
    if runtime.get("local_tag"):
        lines.append(f"- local tag (not authority by itself): {runtime['local_tag']}")
    if runtime.get("hf_archive"):
        lines.append(f"- archive: {runtime['hf_archive']}")
    if runtime.get("note"):
        lines.append(f"- note: {runtime['note']}")
    if runtime.get("kind"):
        lines.append(f"- kind: {runtime['kind']}")
    metrics = observation.get("metrics") or {}
    if metrics.get("blocker"):
        lines.extend(["", f"Blocker: {metrics['blocker']}"])
    if observation.get("identity_mismatches"):
        lines.extend(["", "Identity mismatches:"])
        lines.extend(
            f"- {x['field']}: expected={x['expected']!r}, observed={x['observed']!r}"
            for x in observation["identity_mismatches"]
        )
    if observation.get("note"):
        lines.extend(["", f"Observation note: {observation['note']}"])
    lines.extend([
        "",
        f"Resume policy: {row['resume'].get('policy')}",
        f"Next action: {row['resume'].get('next_action')}",
        "Authority / evidence entrypoints:",
    ])
    lines.extend(f"- {path}" for path in row["authority_paths"])
    lines.append("")
    lines.append("This console does not grant execution authority and cannot resume the run itself.")
    return "\n".join(lines)


def render_resume_plan(row: Mapping[str, Any]) -> str:
    observation = row["observation"]
    return "\n".join([
        f"Experiment: {row['id']}",
        f"Observed status: {observation['effective_status']}",
        f"Identity check: {observation.get('identity_trust')}",
        f"Resume policy: {row['resume'].get('policy')}",
        f"Safe next action: {row['resume'].get('next_action')}",
        "Execution action: NONE (read-only plan). Re-read current authority immediately before any real resume.",
    ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--catalog", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status")
    status.add_argument("--json", action="store_true")
    show = sub.add_parser("show")
    show.add_argument("experiment_id")
    show.add_argument("--json", action="store_true")
    plan = sub.add_parser("resume-plan")
    plan.add_argument("experiment_id")
    plan.add_argument("--json", action="store_true")
    passport = sub.add_parser("passport")
    passport.add_argument("experiment_id")
    passport.add_argument("--json", action="store_true")
    verify = sub.add_parser("verify")
    verify.add_argument("experiment_id")
    verify.add_argument("--json", action="store_true")
    revive = sub.add_parser("revive-plan")
    revive.add_argument("experiment_id")
    revive.add_argument("--json", action="store_true")
    sub.add_parser("check")
    args = parser.parse_args(argv)

    root = args.repo_root.expanduser().resolve()
    catalog_path = args.catalog.expanduser().resolve() if args.catalog else root / "configs" / "experiment" / "experiment-console-202609110009.json"
    try:
        if args.command == "passport":
            from scripts.openevo_webshop.experiment_registry import RegistryError, build_passport_projection, render_passport_projection
            try:
                view = build_passport_projection(root, args.experiment_id)
            except RegistryError as exc:
                raise ConsoleError(str(exc)) from exc
            print(json.dumps(view, indent=2, sort_keys=True, ensure_ascii=False) if args.json else render_passport_projection(view))
            return 0
        if args.command == "verify":
            from scripts.openevo_webshop.experiment_verify import VerifyError, render_verification, verify_experiment
            try:
                report = verify_experiment(root, args.experiment_id)
            except VerifyError as exc:
                raise ConsoleError(str(exc)) from exc
            print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) if args.json else render_verification(report))
            return 0 if report["identity"] == "PASS" else 2
        if args.command == "revive-plan":
            from scripts.openevo_webshop.experiment_revive_plan import RevivePlanError, build_revive_plan, render_revive_plan
            try:
                recovery = build_revive_plan(root, args.experiment_id)
            except RevivePlanError as exc:
                raise ConsoleError(str(exc)) from exc
            print(json.dumps(recovery, indent=2, sort_keys=True, ensure_ascii=False) if args.json else render_revive_plan(recovery))
            return 0
        catalog = _read_json(catalog_path)
        errors = validate_catalog(root, catalog)
        if args.command == "check":
            if errors:
                print(json.dumps({"status": "FAIL", "mutation": False, "errors": errors}, indent=2, ensure_ascii=False))
                return 2
            print(json.dumps({"status": "PASS", "mutation": False, "experiment_count": len(catalog["experiments"])}, sort_keys=True))
            return 0
        if errors:
            raise ConsoleError("catalog validation failed: " + "; ".join(errors))
        snapshot = build_snapshot(root, catalog)
        if args.command == "status":
            print(json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False) if args.json else render_status(snapshot))
            return 0
        row = _find(snapshot, args.experiment_id)
        if args.command == "show":
            print(json.dumps(row, indent=2, sort_keys=True, ensure_ascii=False) if args.json else render_show(row))
            return 0
        if args.command == "resume-plan":
            print(json.dumps({"mutation": False, "experiment": row}, indent=2, sort_keys=True, ensure_ascii=False) if args.json else render_resume_plan(row))
            return 0
    except ConsoleError as exc:
        print(f"experiment-console: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
