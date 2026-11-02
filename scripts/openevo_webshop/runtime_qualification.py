#!/usr/bin/env python3
"""Cache stable runtime qualification by an immutable fingerprint document.

This is deliberately narrower than the launch/live gate. Expensive facts that
should be stable for one runtime identity (packages, imports, tokenizer/model
assets, source identities, JDK and image identity) may be reused only when the
fingerprint document, qualification inputs, and qualification implementation are
byte-for-byte equivalent. Volatile facts such as GPU occupancy, CUDA visibility,
storage headroom, output collisions, W&B connectivity, and formal release state
are intentionally excluded and must be checked at use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.openevo_webshop import environment_doctor as envdoc  # noqa: E402


SCHEMA = "openevo.runtime-qualification.v1"
CACHE_SCHEMA = "openevo.runtime-qualification-cache.v1"
ENVIRONMENT_CACHE_SCHEMA = "openevo.runtime-environment-check-cache.v1"
# Dependency declarations can change inside a source checkout even when the
# installed Python distributions have not changed yet. Re-check the base on
# dependency-manifest drift, never simply because HEAD moved.
DEPENDENCY_MANIFESTS = (
    "pyproject.toml", "setup.cfg", "setup.py", "requirements.txt",
    "requirements.in", "requirements.lock.txt", "uv.lock", "poetry.lock",
    "Pipfile.lock", "environment.yml", "conda-lock.yml",
)
# Only these known source/asset fields may change without changing the base runtime.
# Unknown fingerprint fields remain part of the base key, failing closed.
SOURCE_ASSET_FINGERPRINT_FIELDS = frozenset({
    "openevo_core", "seed_root", "model", "webshop_data", "index_root",
})
RUNTIME_IDENTITY_PROBE = """
import importlib.metadata as metadata
import json
import sys

rows = []
for distribution in metadata.distributions():
    name = distribution.metadata.get("Name") or ""
    rows.append({"name": name.lower().replace("_", "-"), "version": distribution.version,
                 "requires": sorted(distribution.requires or [])})
print(json.dumps({
    "executable": sys.executable,
    "version": sys.version,
    "prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "distributions": sorted(rows, key=lambda row: (row["name"], row["version"])),
}, sort_keys=True))
"""


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolved(path: Path | None) -> str | None:
    return str(path.expanduser().resolve()) if path is not None else None


def executable_path(path: Path | None) -> str | None:
    return str(path.expanduser().absolute()) if path is not None else None


def _file_identity(path: Path, *, hash_limit_bytes: int = 64 << 20) -> dict[str, Any]:
    requested = path.expanduser()
    try:
        stat = requested.stat()
    except OSError:
        return {"path": str(requested), "present": False}
    identity: dict[str, Any] = {
        "path": str(requested),
        "resolved_path": str(requested.resolve()),
        "present": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }
    if requested.is_file() and stat.st_size <= hash_limit_bytes:
        identity["sha256"] = sha256_file(requested)
    return identity


def _command(argv: list[str], *, timeout: float = 60.0) -> subprocess.CompletedProcess[bytes] | None:
    try:
        return subprocess.run(argv, check=False, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None


def _unavailable_runtime_identity(executable: Path) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "requested_executable": _file_identity(executable, hash_limit_bytes=1 << 20),
    }


def _parse_runtime_probe(stdout: bytes) -> tuple[dict[str, Any], list[dict[str, str]]] | None:
    try:
        payload = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    distributions = payload.pop("distributions", None)
    if (
        not isinstance(distributions, list)
        or not isinstance(payload.get("prefix"), str)
        or not all(
            isinstance(row, dict)
            and isinstance(row.get("name"), str)
            and isinstance(row.get("version"), str)
            for row in distributions
        )
    ):
        return None
    return payload, distributions


def _runtime_identity(args: argparse.Namespace) -> dict[str, Any]:
    executable = args.runtime_python.expanduser() if args.runtime_python is not None else Path(sys.executable)
    probe = _command([str(executable), "-c", RUNTIME_IDENTITY_PROBE])
    if probe is None or probe.returncode != 0:
        return _unavailable_runtime_identity(executable)
    parsed = _parse_runtime_probe(probe.stdout)
    if parsed is None:
        return _unavailable_runtime_identity(executable)
    payload, distributions = parsed
    encoded = json.dumps(distributions, sort_keys=True, separators=(",", ":")).encode("utf-8")
    prefix = Path(payload["prefix"])
    return {
        "status": "observed",
        **payload,
        "requested_executable": _file_identity(executable, hash_limit_bytes=1 << 20),
        "pyvenv_cfg": _file_identity(prefix / "pyvenv.cfg", hash_limit_bytes=1 << 20),
        "distributions_count": len(distributions),
        "distributions_sha256": sha256_bytes(encoded),
    }


def _git_identity(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    requested = path.expanduser().resolve()
    root_probe = _command(["git", "-C", str(requested), "rev-parse", "--show-toplevel"])
    if root_probe is None or root_probe.returncode != 0:
        return {"path": str(requested), "git": False, "present": requested.exists()}
    root = Path(root_probe.stdout.decode("utf-8").strip()).resolve()
    head = _command(["git", "-C", str(root), "rev-parse", "HEAD"])
    status = _command(["git", "-C", str(root), "status", "--porcelain=v1", "-z", "--untracked-files=all"])
    diff = _command(["git", "-C", str(root), "diff", "--binary", "HEAD", "--"], timeout=120.0)
    if head is None or status is None or diff is None:
        return {"path": str(requested), "root": str(root), "git": True, "status": "unavailable"}
    status_bytes = status.stdout
    return {
        "path": str(requested),
        "root": str(root),
        "git": True,
        "head": head.stdout.decode("utf-8").strip() if head.returncode == 0 else None,
        "dirty": bool(status_bytes),
        "status_sha256": sha256_bytes(status_bytes),
        "tracked_diff_sha256": sha256_bytes(diff.stdout),
        "untracked": _untracked_identities(root, status_bytes),
    }


def _untracked_identities(root: Path, status: bytes) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    for entry in status.split(b"\0"):
        if not entry.startswith(b"?? "):
            continue
        relative = entry[3:].decode("utf-8", errors="replace")
        identity = _file_identity(root / relative)
        identity["relative_path"] = relative
        identities.append(identity)
    return identities


def _declared_asset_identity(path: Path | None, *, kind: str) -> dict[str, Any] | None:
    if path is None:
        return None
    requested = path.expanduser()
    root = requested.resolve()
    if not root.is_dir():
        return {"kind": kind, "path": str(requested), "resolved_path": str(root), "present": False}
    if kind == "model":
        descriptors = (
            "MANIFEST.json",
            "manifest.json",
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "tokenizer.model",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "added_tokens.json",
        )
        payloads = sorted(root.glob("*.safetensors"))
    elif kind == "webshop_data":
        descriptors = (
            "MANIFEST.json",
            "manifest.json",
            "items_shuffle_1000.json",
            "items_ins_v2_1000.json",
            "items_human_ins.json",
        )
        payloads = []
    else:
        descriptors = ("MANIFEST.json", "manifest.json", "build_manifest.json")
        payloads = []
    return {
        "kind": kind,
        "path": str(requested),
        "resolved_path": str(root),
        "present": True,
        "descriptors": {name: _file_identity(root / name) for name in descriptors},
        "payloads": [_file_identity(payload, hash_limit_bytes=0) for payload in payloads],
    }


def collect_live_identity(args: argparse.Namespace) -> dict[str, Any]:
    """Collect cheap, deterministic facts that must remain stable for cache reuse."""
    return {
        "schema": "openevo.runtime-live-identity.v1",
        "runtime_python": _runtime_identity(args),
        "docker_image": envdoc.check_docker(
            args.image, args.expected_image_digest,
            runtime_container=args.runtime_container,
        ),
        "git": {
            "experiment": _git_identity(args.repo_root),
            "source": _git_identity(args.source_root),
            "openevo": _git_identity(args.openevo_root),
        },
        "assets": {
            "model": _declared_asset_identity(args.model, kind="model"),
            "webshop_data": _declared_asset_identity(args.webshop_data, kind="webshop_data"),
            "index_root": _declared_asset_identity(args.index_root, kind="index_root"),
        },
    }


def implementation_identity() -> dict[str, str]:
    doctor_path = Path(envdoc.__file__).resolve()
    return {
        "qualification_sha256": sha256_file(Path(__file__).resolve()),
        "environment_doctor_sha256": sha256_file(doctor_path),
    }


def request_document(args: argparse.Namespace) -> dict[str, Any]:
    fingerprint_file = args.fingerprint_file.expanduser().resolve()
    if not fingerprint_file.is_file():
        raise ValueError(f"runtime fingerprint document is missing: {fingerprint_file}")
    route = envdoc.ROUTE_PROFILES[args.profile]
    return {
        "schema": CACHE_SCHEMA,
        "runtime_fingerprint_document": resolved(fingerprint_file),
        "runtime_fingerprint_sha256": sha256_file(fingerprint_file),
        "implementation": implementation_identity(),
        "profile": args.profile,
        "repo_root": resolved(args.repo_root),
        "route": {key: (list(value) if isinstance(value, tuple) else value) for key, value in route.items()},
        "source_root": resolved(args.source_root),
        "webshop_data": resolved(args.webshop_data),
        "model": resolved(args.model),
        "openevo_root": resolved(args.openevo_root),
        "openevo_patch_sha": args.openevo_patch_sha,
        "runtime_python": executable_path(args.runtime_python),
        "index_root": resolved(args.index_root),
        "image": args.image,
        "expected_image_digest": args.expected_image_digest,
        "runtime_container": bool(args.runtime_container),
        "require_java": bool(args.require_java),
        "allow_dirty": bool(args.allow_dirty),
        "packages": list(args.package or route["packages"]),
        "imports": list(args.import_name or route["imports"]),
        "live_identity": collect_live_identity(args),
    }


def cache_key(request: dict[str, Any]) -> str:
    encoded = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reusable_receipt(path: Path, request: dict[str, Any]) -> dict[str, Any] | None:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(receipt, dict):
        return None
    if receipt.get("schema") != SCHEMA or receipt.get("status") != "PASS":
        return None
    if receipt.get("request") != request:
        return None
    return receipt


def dependency_manifest_identity(request: dict[str, Any]) -> dict[str, dict[str, str]] | None:
    """Track dependency declarations by content, not checkout HEAD or timestamps."""
    result: dict[str, dict[str, str]] = {}
    for role, key in (("experiment", "repo_root"), ("source", "source_root"),
                      ("openevo", "openevo_root")):
        root_value = request.get(key)
        if not root_value:
            continue
        root = Path(root_value)
        manifests: dict[str, str] = {}
        for name in DEPENDENCY_MANIFESTS:
            path = root / name
            try:
                if path.is_file():
                    manifests[name] = sha256_file(path)
            except OSError:
                return None
        result[role] = manifests
    return result


def environment_cache_request(request: dict[str, Any]) -> dict[str, Any] | None:
    """Identify ONLY the base environment, independent of experiment/source SHA.

    The full qualification continues to bind git, source imports, model/tokenizer
    and data. This lower cache contains no source/asset PASS claims.
    """
    runtime = request["live_identity"].get("runtime_python")
    docker = request["live_identity"].get("docker_image")
    if (not isinstance(runtime, dict) or runtime.get("status") != "observed"
            or not isinstance(runtime.get("distributions_sha256"), str)
            or not isinstance(docker, dict)
            or docker.get("status") not in ("pass", "skip")
            or (request["image"] and not request["runtime_container"]
                and (docker.get("status") != "pass" or not docker.get("image_id")))):
        return None
    try:
        fingerprint = json.loads(Path(request["runtime_fingerprint_document"]).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(fingerprint, dict):
        return None
    dependency_manifests = dependency_manifest_identity(request)
    if dependency_manifests is None:
        return None

    # The existing fingerprint is a combined scientific/runtime document.
    # Keep every unknown field in the environment key; exclude only proven
    # source and data/model identifiers, which full qualification revalidates.
    base = {key: value for key, value in fingerprint.items()
            if key not in SOURCE_ASSET_FINGERPRINT_FIELDS}
    return {
        "schema": ENVIRONMENT_CACHE_SCHEMA,
        "base_fingerprint_sha256": sha256_bytes(
            json.dumps(base, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
        "implementation": request["implementation"],
        "profile": request["profile"],
        "runtime_python": request["runtime_python"],
        "observed_runtime_python": runtime,
        "packages": request["packages"],
        "dependency_manifests": dependency_manifests,
        "pip_check_required": bool(request["route"]["require_pip_check"]),
        "image": request["image"],
        "expected_image_digest": request["expected_image_digest"],
        "runtime_container": request["runtime_container"],
        "observed_docker_image": docker,
    }


def environment_checks_to_cache(checks: dict[str, Any], *, pip_required: bool) -> dict[str, Any] | None:
    required = ("runtime_python", "packages", "pip_check") if pip_required else (
        "runtime_python", "packages",
    )
    selected: dict[str, Any] = {}
    for name in required:
        value = checks.get(name)
        if not isinstance(value, dict) or value.get("status") != "pass" or value.get("errors") != []:
            return None
        selected[name] = value
    return selected


def reusable_environment_checks(path: Path, request: dict[str, Any]) -> dict[str, Any] | None:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if (not isinstance(receipt, dict)
            or receipt.get("schema") != ENVIRONMENT_CACHE_SCHEMA
            or receipt.get("status") != "PASS"
            or receipt.get("request") != request):
        return None
    checks = receipt.get("checks")
    if not isinstance(checks, dict):
        return None
    selected = environment_checks_to_cache(checks, pip_required=request["pip_check_required"])
    if selected is None or set(selected) != set(checks):
        return None
    proof = sha256_bytes(json.dumps(
        {"request": request, "checks": checks}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8"))
    return selected if proof == receipt.get("content_sha256") else None


def perform_checks(
    args: argparse.Namespace, *, environment_checks: dict[str, Any] | None = None
) -> dict[str, Any]:
    route = envdoc.ROUTE_PROFILES[args.profile]
    runtime_python = executable_path(args.runtime_python)
    cached_environment = environment_checks or {}
    assets_required = bool(route["require_webshop_assets"] or route["require_model_assets"])
    missing_assets = [
        name
        for name, required, value in (
            ("source root", assets_required, args.source_root),
            ("WebShop data", route["require_webshop_assets"], args.webshop_data),
            ("model", route["require_model_assets"], args.model),
        )
        if required and value is None
    ]
    assets = (
        {"status": "fail", "errors": ["profile requires " + ", ".join(missing_assets)]}
        if missing_assets
        else envdoc.check_runtime_assets(
            args.source_root,
            args.webshop_data,
            args.model,
            args.index_root,
            require_seed_entry=route["require_seed_entry"],
        )
        if assets_required
        else {"status": "skip", "errors": [], "reason": "profile does not require scientific assets"}
    )
    model_completeness = (
        envdoc.check_model_completeness(args.model)
        if route["require_model_assets"] and args.model is not None
        else {"status": "fail", "errors": ["profile requires a model snapshot"]}
        if route["require_model_assets"]
        else {"status": "skip", "errors": [], "reason": "profile does not require a model snapshot"}
    )
    checks: dict[str, Any] = {
        "runtime_python": cached_environment.get("runtime_python")
        or envdoc.check_python_executable(args.runtime_python),
        "git": envdoc.check_git(args.repo_root.expanduser().resolve(), allow_dirty=args.allow_dirty),
        "assets": assets,
        "model_completeness": model_completeness,
        "packages": cached_environment.get("packages")
        or envdoc.check_packages(tuple(args.package or route["packages"]), python_executable=runtime_python),
        "imports": envdoc.check_imports(tuple(args.import_name or route["imports"]), python_executable=runtime_python),
        "external_runtime_imports": envdoc.check_external_runtime_imports(
            args.source_root or args.repo_root,
            openevo_root=args.openevo_root,
            enabled=True,
            python_executable=runtime_python,
            profile=args.profile,
        ) if route["require_external_runtime"] else {"status": "skip", "errors": [], "reason": "profile does not require OpenEvo runtime imports"},
        "external_source_identity": envdoc.check_external_source_identity(
            args.source_root or args.repo_root,
            openevo_root=args.openevo_root,
            enabled=True,
            profile=args.profile,
            expected_openevo_patch_sha=args.openevo_patch_sha,
        ) if route["require_external_runtime"] else {"status": "skip", "errors": [], "reason": "profile does not require OpenEvo source identity"},
        "tokenizer": envdoc.check_tokenizer(args.model, enabled=True, python_executable=runtime_python) if route["require_tokenizer"] and args.model is not None else {"status": "fail", "errors": ["profile requires a tokenizer model"]} if route["require_tokenizer"] else {"status": "skip", "errors": [], "reason": "profile does not require a tokenizer"},
        "pip_check": (cached_environment.get("pip_check")
            or envdoc.check_pip(python_executable=runtime_python))
        if route["require_pip_check"] else {"status": "skip", "errors": [], "reason": "profile does not require pip consistency"},
        "java": envdoc.check_java(args.require_java),
        "docker": envdoc.check_docker(
            args.image,
            args.expected_image_digest,
            runtime_container=args.runtime_container,
        ),
    }
    failures = [
        f"{name}: {error}"
        for name, value in checks.items()
        for error in value.get("errors", [])
    ]
    return {"checks": checks, "failures": failures}


def changed_request_fields(previous: Any, current: Any, prefix: str = "") -> list[str]:
    """Describe identity changes without printing sensitive or bulky values."""
    if isinstance(previous, dict) and isinstance(current, dict):
        return [field for key in sorted(set(previous) | set(current))
                for field in changed_request_fields(previous.get(key), current.get(key),
                                                    f"{prefix}.{key}" if prefix else key)]
    return [prefix] if previous != current else []


def qualification_decision(cache_path: Path, request: dict[str, Any], *, refresh: bool) -> dict[str, Any]:
    previous = None
    for path in cache_path.parent.glob("*.json"):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(doc, dict) or doc.get("schema") != SCHEMA or doc.get("status") != "PASS":
            continue
        if not isinstance(doc.get("qualified_at_epoch"), (int, float)):
            continue
        prior = doc.get("request")
        if not isinstance(prior, dict) or any(prior.get(k) != request.get(k) for k in ("profile", "repo_root")):
            continue
        if previous is None or doc.get("qualified_at_epoch", 0) > previous.get("qualified_at_epoch", 0):
            previous = doc
    return {
        "reason": "explicit_refresh_requested" if refresh else "no_matching_pass_receipt",
        "previous_cache_key": previous.get("cache_key") if previous else None,
        "changed_request_fields": changed_request_fields(previous["request"], request) if previous else [],
        "initial_qualification": previous is None,
        "environment_mutation_allowed": False,
    }


def qualify(args: argparse.Namespace) -> dict[str, Any]:
    request = request_document(args)
    key = cache_key(request)
    cache_path = args.cache_root.expanduser().resolve() / f"{key}.json"
    docker_identity = request["live_identity"].get("docker_image") or {}
    image_failed = docker_identity.get("status") == "fail"
    if not args.refresh and not image_failed:
        cached = reusable_receipt(cache_path, request)
        if cached is not None:
            return {
                **cached,
                "reused": True,
                "cache_path": str(cache_path),
                "observed_at_epoch": time.time(),
                "qualification_decision": {"reason": "exact_request_match", "changed_request_fields": [],
                                           "environment_mutation_allowed": False},
            }

    decision = qualification_decision(cache_path, request, refresh=bool(args.refresh))
    base_request = environment_cache_request(request)
    base_key = cache_key(base_request) if base_request is not None else None
    base_path = (args.cache_root.expanduser().resolve() / "environment" / f"{base_key}.json"
                 if base_key is not None else None)
    cached_base = (reusable_environment_checks(base_path, base_request)
                   if base_path is not None and base_request is not None
                   and not args.refresh and not image_failed else None)
    environment_info: dict[str, Any] = {
        "status": "REUSED" if cached_base is not None else "NOT_CACHED",
        "cache_key": base_key,
        "cache_path": str(base_path) if base_path is not None else None,
        "checks_reused": sorted(cached_base) if cached_base is not None else [],
    }

    if image_failed:
        decision["reason"] = "live_image_identity_failed"
        result = {"checks": {"docker": docker_identity},
                  "failures": [f"docker: {error}" for error in docker_identity.get("errors", [])]
                              or ["docker: live image identity check failed"]}
    else:
        result = perform_checks(args, environment_checks=cached_base)

    # A new source PASS never inherits an earlier source PASS. Only share the
    # three bounded base-environment checks after this full qualification passes.
    if not result["failures"] and base_path is not None and base_request is not None and cached_base is None:
        stable = environment_checks_to_cache(
            result["checks"], pip_required=base_request["pip_check_required"]
        )
        if stable is not None:
            proof = sha256_bytes(json.dumps(
                {"request": base_request, "checks": stable},
                sort_keys=True, separators=(",", ":")
            ).encode("utf-8"))
            atomic_json(base_path, {
                "schema": ENVIRONMENT_CACHE_SCHEMA,
                "status": "PASS",
                "request": base_request,
                "checks": stable,
                "content_sha256": proof,
            })
            environment_info["status"] = "BUILT"
    receipt = {
        "environment_cache": environment_info,
        "schema": SCHEMA,
        "status": "PASS" if not result["failures"] else "FAIL",
        "request": request,
        "cache_key": key,
        "cache_path": str(cache_path),
        "qualified_at_epoch": time.time(),
        "observed_at_epoch": time.time(),
        "reused": False,
        "qualification_decision": decision,
        "checks": result["checks"],
        "failures": result["failures"],
        "formal_task_consumption_allowed": False,
    }
    atomic_json(cache_path, receipt)
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fingerprint-file", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--webshop-data", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--profile", choices=tuple(envdoc.ROUTE_PROFILES), default="full")
    parser.add_argument("--runtime-python", type=Path)
    parser.add_argument("--openevo-root", type=Path)
    parser.add_argument("--openevo-patch-sha")
    parser.add_argument("--index-root", type=Path)
    parser.add_argument("--image")
    parser.add_argument("--expected-image-digest")
    parser.add_argument("--runtime-container", action="store_true")
    parser.add_argument("--require-java", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--package", action="append")
    parser.add_argument("--import-name", action="append")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        receipt = qualify(args)
    except (OSError, ValueError) as exc:
        print(f"runtime qualification: {exc}", file=sys.stderr)
        return 2
    if args.output:
        atomic_json(args.output.expanduser().resolve(), receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    print("RUNTIME_QUALIFICATION: " + ("REUSED" if receipt.get("reused") else receipt["status"]))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
