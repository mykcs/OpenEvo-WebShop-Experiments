#!/usr/bin/env python3
"""Fail-closed environment doctor for a reproducible WebShop run.

The doctor is deliberately independent of W&B and scientific scoring.  It
checks the execution identity, required assets, Python dependency graph,
optional Docker image, CUDA visibility, and the requested W&B mode.  A W&B
failure may produce an explicit offline fallback, but it is never hidden.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGES = (
    "torch",
    "transformers",
    "vllm",
    "flashinfer-python",
    "flashinfer-cubin",
    "flashinfer-jit-cache",
    "ray",
    "hydra-core",
    "datasets",
    "wandb",
)
DEFAULT_IMPORTS = ("torch", "transformers", "vllm", "flashinfer.comm", "wandb")
OPENEVO_PACKAGES = ("torch", "transformers", "peft", "wandb")
OPENEVO_IMPORTS = ("torch", "transformers", "peft", "wandb")
ROUTE_PROFILES = {
    "platform": {
        "packages": ("torch",),
        "imports": ("torch",),
        "require_seed_entry": False,
        "require_webshop_assets": False,
        "require_model_assets": False,
        "require_external_runtime": False,
        "require_tokenizer": False,
        "require_pip_check": True,
    },
    "full": {
        "packages": DEFAULT_PACKAGES,
        "imports": DEFAULT_IMPORTS,
        "require_seed_entry": True,
        "require_webshop_assets": True,
        "require_model_assets": True,
        "require_external_runtime": True,
        "require_tokenizer": True,
        "require_pip_check": True,
    },
    "seed": {
        "packages": DEFAULT_PACKAGES,
        "imports": DEFAULT_IMPORTS,
        "require_seed_entry": True,
        "require_webshop_assets": True,
        "require_model_assets": True,
        "require_external_runtime": True,
        "require_tokenizer": True,
        "require_pip_check": True,
    },
    "openevo": {
        "packages": OPENEVO_PACKAGES,
        "imports": OPENEVO_IMPORTS,
        "require_seed_entry": False,
        "require_webshop_assets": True,
        "require_model_assets": True,
        "require_external_runtime": True,
        "require_tokenizer": True,
        "require_pip_check": True,
    },
}
DEFAULT_MIN_FREE_BYTES = 1 << 30


def run(argv: list[str], *, timeout: float = 20.0) -> dict[str, Any]:
    try:
        result = subprocess.run(
            argv,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": str(exc)}
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_python_executable(path: Path | None) -> dict[str, Any]:
    """Record and validate the interpreter that will run the experiment."""
    executable = str(path.expanduser()) if path is not None else sys.executable
    result = run(
        [
            executable,
            "-c",
            "import json, sys; print(json.dumps({'executable': sys.executable, 'version': sys.version, 'prefix': sys.prefix}))",
        ],
        timeout=20.0,
    )
    if not result["ok"]:
        return {
            "status": "fail",
            "requested": executable,
            "errors": [result["stderr"] or "runtime Python could not be executed"],
        }
    try:
        identity = json.loads(result["stdout"])
    except json.JSONDecodeError:
        return {
            "status": "fail",
            "requested": executable,
            "errors": ["runtime Python identity was not valid JSON"],
        }
    return {"status": "pass", **identity, "requested": executable, "errors": []}


def check_git(repo: Path, *, allow_dirty: bool) -> dict[str, Any]:
    head = run(["git", "-C", str(repo), "rev-parse", "HEAD"])
    dirty = run(["git", "-C", str(repo), "status", "--porcelain"])
    errors: list[str] = []
    if not head["ok"]:
        errors.append(f"git HEAD unavailable: {head['stderr'] or head['stdout']}")
    if dirty["stdout"] and not allow_dirty:
        errors.append("experiment checkout is dirty; use a clean immutable checkout")
    return {
        "status": "pass" if not errors else "fail",
        "head": head["stdout"] or None,
        "dirty": bool(dirty["stdout"]),
        "errors": errors,
    }


def check_paths(paths: dict[str, Path]) -> dict[str, Any]:
    errors: list[str] = []
    observed: dict[str, Any] = {}
    for name, path in paths.items():
        exists = path.exists()
        readable = os.access(path, os.R_OK) if exists else False
        observed[name] = {"path": str(path), "exists": exists, "readable": readable}
        if not exists or not readable:
            errors.append(f"{name} is missing or unreadable: {path}")
    return {"status": "pass" if not errors else "fail", "assets": observed, "errors": errors}


def check_runtime_assets(
    source_root: Path,
    webshop_data: Path,
    model: Path,
    index_root: Path | None = None,
    *,
    require_seed_entry: bool = True,
) -> dict[str, Any]:
    required = {
        "webshop_catalog": webshop_data / "items_shuffle_1000.json",
        "webshop_attributes": webshop_data / "items_ins_v2_1000.json",
        "webshop_human_attributes": webshop_data / "items_human_ins.json",
        "model_config": model / "config.json",
    }
    if require_seed_entry:
        required["seed_entry"] = source_root / "seed" / "analysis.py"
    shards = sorted(model.glob("*.safetensors"))
    if not shards:
        return {"status": "fail", "assets": {}, "errors": [f"model has no safetensors shards: {model}"]}
    required.update({path.name: path for path in shards})
    if index_root is not None:
        required.update(
            {
                "index_directory": index_root / "indexes_1k",
                "index_documents": index_root / "resources_1k" / "documents.jsonl",
                "index_build_manifest": index_root / "build_manifest.json",
            }
        )
    return check_paths(required)


def check_model_completeness(model: Path) -> dict[str, Any]:
    """Check the local snapshot before a loader can silently hit the network."""
    model = model.expanduser().resolve()
    errors: list[str] = []
    required = {"config.json": model / "config.json", "tokenizer_config.json": model / "tokenizer_config.json"}
    observed: dict[str, Any] = {}
    for name, path in required.items():
        present = path.is_file() and path.stat().st_size > 0 if path.exists() else False
        observed[name] = {"path": str(path), "present": present}
        if not present:
            errors.append(f"model snapshot is missing a non-empty {name}: {path}")

    shards = sorted(model.glob("*.safetensors"))
    observed["safetensors_shards"] = [
        {"path": str(path), "bytes": path.stat().st_size} for path in shards
    ]
    if not shards:
        errors.append(f"model snapshot has no safetensors shards: {model}")
    elif any(path.stat().st_size == 0 for path in shards):
        errors.append("model snapshot contains an empty safetensors shard")

    tokenizer_variants = (
        (model / "tokenizer.json",),
        (model / "tokenizer.model",),
        (model / "vocab.json", model / "merges.txt"),
    )
    tokenizer_files = next(
        (variant for variant in tokenizer_variants if all(path.is_file() and path.stat().st_size > 0 for path in variant)),
        None,
    )
    observed["tokenizer_files"] = [str(path) for path in tokenizer_files] if tokenizer_files else []
    if tokenizer_files is None:
        errors.append("model snapshot has no complete tokenizer payload (tokenizer.json, tokenizer.model, or vocab.json+merges.txt)")
    return {
        "status": "pass" if not errors else "fail",
        "path": str(model),
        "assets": observed,
        "errors": errors,
    }


def check_write_target(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "skip", "errors": [], "reason": "no artifact root requested"}
    target = path.expanduser().resolve()
    parent = target if target.exists() else target.parent
    writable = parent.is_dir() and os.access(parent, os.W_OK)
    errors = [] if writable else [f"artifact target parent is not writable: {parent}"]
    return {
        "status": "pass" if writable else "fail",
        "target": str(target),
        "parent": str(parent),
        "exists": target.exists(),
        "writable": writable,
        "errors": errors,
    }


def check_cache(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "skip", "errors": [], "reason": "no cache root requested"}
    cache = path.expanduser().resolve()
    if not cache.exists():
        return {
            "status": "warn",
            "path": str(cache),
            "exists": False,
            "errors": [],
            "warning": "persistent cache root is absent; a bootstrap may download assets",
        }
    readable = os.access(cache, os.R_OK)
    return {
        "status": "pass" if readable else "fail",
        "path": str(cache),
        "exists": True,
        "readable": readable,
        "errors": [] if readable else [f"cache root is unreadable: {cache}"],
    }


def check_storage(path: Path | None, minimum_free_bytes: int) -> dict[str, Any]:
    """Fail before allocation when the artifact filesystem cannot hold a run."""
    if path is None:
        return {"status": "skip", "errors": [], "reason": "no storage root requested"}
    root = path.expanduser().resolve()
    probe = root if root.exists() else root.parent
    if not probe.is_dir():
        return {
            "status": "fail",
            "path": str(root),
            "errors": [f"storage probe parent is missing: {probe}"],
        }
    usage = shutil.disk_usage(probe)
    errors: list[str] = []
    if usage.free < minimum_free_bytes:
        errors.append(
            f"storage free bytes {usage.free} is below required {minimum_free_bytes} on {probe}"
        )
    return {
        "status": "pass" if not errors else "fail",
        "path": str(root),
        "probe": str(probe),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "minimum_free_bytes": minimum_free_bytes,
        "capacity_percent": round((usage.used / usage.total) * 100, 2) if usage.total else None,
        "errors": errors,
    }


def check_nvidia_smi(require_cuda: bool) -> dict[str, Any]:
    if not require_cuda:
        return {"status": "skip", "errors": [], "reason": "CUDA not required"}
    result = run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        timeout=20.0,
    )
    if not result["ok"]:
        return {"status": "fail", "errors": [result["stderr"] or "nvidia-smi failed"]}
    rows = [line for line in result["stdout"].splitlines() if line.strip()]
    return {
        "status": "pass" if rows else "fail",
        "gpu_count": len(rows),
        "rows": rows,
        "errors": [] if rows else ["nvidia-smi returned no GPUs"],
    }


def check_java(require_java: bool) -> dict[str, Any]:
    if not require_java:
        return {"status": "skip", "errors": [], "reason": "JDK not required"}
    missing = [name for name in ("java", "javac") if shutil.which(name) is None]
    versions: dict[str, str] = {}
    for name in ("java", "javac"):
        if name in missing:
            continue
        result = run([name, "-version"], timeout=10.0)
        versions[name] = result["stderr"] or result["stdout"]
    errors = [f"required JDK executable is unavailable: {name}" for name in missing]
    return {
        "status": "pass" if not errors else "fail",
        "required": True,
        "versions": versions,
        "errors": errors,
    }


def check_imports(names: tuple[str, ...], *, python_executable: str | None = None) -> dict[str, Any]:
    if python_executable is not None:
        code = (
            "import importlib, json, sys; "
            "names=json.loads(sys.argv[1]); loaded={}; errors=[]; "
            "\nfor name in names:\n"
            "  try:\n    module=importlib.import_module(name); loaded[name]=str(getattr(module, '__file__', 'loaded'))\n"
            "  except Exception as exc: errors.append(f'import {name}: {type(exc).__name__}: {exc}')\n"
            "print(json.dumps({'loaded': loaded, 'errors': errors}))"
        )
        result = run([python_executable, "-c", code, json.dumps(names)], timeout=120.0)
        if not result["ok"]:
            return {"status": "fail", "loaded": {}, "errors": [result["stderr"] or "runtime import probe failed"]}
        try:
            payload = json.loads(result["stdout"])
        except json.JSONDecodeError:
            return {"status": "fail", "loaded": {}, "errors": ["runtime import probe was not valid JSON"]}
        return {
            "status": "pass" if not payload["errors"] else "fail",
            "loaded": payload["loaded"],
            "errors": payload["errors"],
        }
    errors: list[str] = []
    loaded: dict[str, str] = {}
    for name in names:
        try:
            module = importlib.import_module(name)
            loaded[name] = str(getattr(module, "__file__", "loaded"))
        except Exception as exc:  # import failures are the diagnostic output
            errors.append(f"import {name}: {type(exc).__name__}: {exc}")
    return {"status": "pass" if not errors else "fail", "loaded": loaded, "errors": errors}


def check_external_runtime_imports(
    source_root: Path,
    *,
    openevo_root: Path | None = None,
    enabled: bool = False,
    python_executable: str | None = None,
    profile: str = "full",
) -> dict[str, Any]:
    """Import external runtime packages in a clean subprocess with explicit paths."""
    if not enabled:
        return {"status": "skip", "errors": [], "reason": "external runtime import probe not requested"}
    source_root = source_root.expanduser().resolve()
    paths = [str(source_root), str(source_root / "agent_system/environments/env_package/webshop/webshop")]
    modules = []
    if profile in {"full", "seed"}:
        modules.extend(
            (
                "seed",
                "verl",
                "web_agent_site.utils",
                "web_agent_site.engine",
                "agent_system.environments.env_package.webshop.envs",
                "pyserini.search.lucene",
            )
        )
    if profile in {"full", "openevo"}:
        if openevo_root is None:
            return {
                "status": "fail",
                "paths": paths,
                "modules": modules,
                "errors": ["OpenEvo root was not supplied; OpenEvo import remains unverified"],
            }
        openevo_root = openevo_root.expanduser().resolve()
        openevo_paths = [str(openevo_root)]
        if (openevo_root / "src").is_dir():
            openevo_paths.insert(0, str(openevo_root / "src"))
        paths = [*openevo_paths, *paths]
        modules.extend(("openevo", "openevo.evolution.framework.contracts"))
    code = (
        "import importlib, json, sys; "
        "paths=json.loads(sys.argv[1]); modules=json.loads(sys.argv[2]); "
        "[sys.path.insert(0, path) for path in reversed(paths)]; "
        "[importlib.import_module(name) for name in modules]; print('external-runtime-imports-pass')"
    )
    result = run(
        [python_executable or sys.executable, "-c", code, json.dumps(paths), json.dumps(modules)],
        timeout=60.0,
    )
    return {
        "status": "pass" if result["ok"] else "fail",
        "paths": paths,
        "modules": modules,
        "output": result["stdout"] or result["stderr"],
        "errors": [] if result["ok"] else [result["stderr"] or "external runtime import probe failed"],
    }


def check_external_source_identity(
    source_root: Path,
    *,
    openevo_root: Path | None = None,
    enabled: bool = False,
    profile: str = "full",
    expected_openevo_patch_sha: str | None = None,
) -> dict[str, Any]:
    """Require clean Git identities for external code used by a formal route."""
    if not enabled:
        return {"status": "skip", "errors": [], "reason": "external source identity probe not requested"}
    checks = {}
    if profile in {"full", "seed"}:
        checks["seed_source"] = check_git(source_root.expanduser().resolve(), allow_dirty=False)
    if profile in {"full", "openevo"}:
        if openevo_root is None:
            return {
                "status": "fail",
                "checks": checks,
                "errors": ["OpenEvo root was not supplied; upstream SHA cannot be recorded"],
            }
        openevo_root = openevo_root.expanduser().resolve()
        if profile == "openevo" and expected_openevo_patch_sha:
            patch_files = (
                "src/openevo/evolution/parametric/sd_lora_trainer.py",
                "src/openevo/evolution/parametric/trainer_service.py",
            )
            identity = check_git(openevo_root, allow_dirty=True)
            status = run(["git", "-C", str(openevo_root), "status", "--porcelain", "--untracked-files=all"])
            dirty_paths = [line[2:].lstrip() for line in status["stdout"].splitlines() if len(line) >= 3]
            allowed_paths = set(patch_files) | {"desktop/.env.openevo-desktop"}
            errors = list(identity.get("errors", []))
            unexpected = sorted(set(dirty_paths) - allowed_paths)
            if unexpected:
                errors.append("unexpected dirty OpenEvo paths: " + ", ".join(unexpected))
            try:
                diff_process = subprocess.run(
                    ["git", "-C", str(openevo_root), "diff", "--", *patch_files],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=20.0,
                )
            except (OSError, subprocess.TimeoutExpired):
                diff_process = None
            observed_patch_sha = (
                hashlib.sha256(diff_process.stdout.encode("utf-8")).hexdigest()
                if diff_process is not None and diff_process.returncode == 0
                else None
            )
            if observed_patch_sha != expected_openevo_patch_sha:
                errors.append(
                    f"OpenEvo patch digest mismatch: expected {expected_openevo_patch_sha}, observed {observed_patch_sha}"
                )
            checks["openevo_core"] = {
                **identity,
                "status": "pass" if not errors else "fail",
                "patch_sha256": observed_patch_sha,
                "expected_patch_sha256": expected_openevo_patch_sha,
                "patch_verified": observed_patch_sha == expected_openevo_patch_sha and not unexpected,
                "errors": errors,
            }
        else:
            checks["openevo_core"] = check_git(openevo_root, allow_dirty=False)
    errors = [f"{name}: {error}" for name, value in checks.items() for error in value.get("errors", [])]
    return {"status": "pass" if not errors else "fail", "checks": checks, "errors": errors}


def check_tokenizer(
    model: Path,
    *,
    enabled: bool = False,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Load the tokenizer locally so a run cannot silently download a revision."""
    if not enabled:
        return {"status": "skip", "errors": [], "reason": "tokenizer load probe not requested"}
    code = (
        "from transformers import AutoTokenizer; "
        "tokenizer=AutoTokenizer.from_pretrained(" + repr(str(model.expanduser().resolve())) + ", "
        "local_files_only=True, use_fast=True); "
        "print({'class': type(tokenizer).__name__, 'vocab_size': len(tokenizer)})"
    )
    result = run([python_executable or sys.executable, "-c", code], timeout=120.0)
    return {
        "status": "pass" if result["ok"] else "fail",
        "model": str(model.expanduser().resolve()),
        "output": result["stdout"] or result["stderr"],
        "errors": [] if result["ok"] else [result["stderr"] or "local tokenizer probe failed"],
    }


def check_experiment_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "skip", "errors": [], "reason": "no experiment config requested"}
    path = path.expanduser().resolve()
    errors: list[str] = []
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "fail", "path": str(path), "errors": [f"experiment config is unreadable: {exc}"]}
    if not isinstance(document, dict):
        errors.append("experiment config must be a JSON object")
    else:
        for key in ("schema", "status", "formal_task_consumption_allowed"):
            if key not in document:
                errors.append(f"experiment config is missing required field: {key}")
        if "formal_task_consumption_allowed" in document and not isinstance(document["formal_task_consumption_allowed"], bool):
            errors.append("formal_task_consumption_allowed must be boolean")
    return {"status": "pass" if not errors else "fail", "path": str(path), "errors": errors}


def check_run_manifest(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "skip", "errors": [], "reason": "no run manifest requested"}
    path = path.expanduser().resolve()
    try:
        from scripts.validate_run_manifest import validate

        errors = validate(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors = [f"run manifest validation could not run: {exc}"]
    return {"status": "pass" if not errors else "fail", "path": str(path), "errors": errors}


def check_packages(names: tuple[str, ...], *, python_executable: str | None = None) -> dict[str, Any]:
    if python_executable is not None:
        code = (
            "import importlib.metadata, json, sys; names=json.loads(sys.argv[1]); "
            "versions={name: (importlib.metadata.version(name) if name in {dist.metadata['Name'] for dist in importlib.metadata.distributions()} else None) for name in names}; "
            "print(json.dumps(versions))"
        )
        result = run([python_executable, "-c", code, json.dumps(names)], timeout=60.0)
        if not result["ok"]:
            return {"status": "fail", "versions": {}, "errors": [result["stderr"] or "runtime package probe failed"]}
        try:
            versions = json.loads(result["stdout"])
        except json.JSONDecodeError:
            return {"status": "fail", "versions": {}, "errors": ["runtime package probe was not valid JSON"]}
    else:
        versions = {name: package_version(name) for name in names}
    errors = [f"package is not installed: {name}" for name, version in versions.items() if version is None]
    return {"status": "pass" if not errors else "fail", "versions": versions, "errors": errors}


def check_pip(*, python_executable: str | None = None) -> dict[str, Any]:
    result = run([python_executable or sys.executable, "-m", "pip", "check"], timeout=60.0)
    output = "\n".join(value for value in (result["stdout"], result["stderr"]) if value)
    return {
        "status": "pass" if result["ok"] else "fail",
        "returncode": result["returncode"],
        "output": output[:12000],
        "errors": [] if result["ok"] else ["python -m pip check is not green"],
    }


def check_cuda(require_cuda: bool, *, python_executable: str | None = None) -> dict[str, Any]:
    result = run(
        [
            python_executable or sys.executable,
            "-c",
            "import torch; print({'available': bool(torch.cuda.is_available()), 'count': int(torch.cuda.device_count()), 'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})",
        ],
        timeout=30.0,
    )
    if not result["ok"]:
        return {"status": "fail" if require_cuda else "warn", "output": result, "errors": [result["stderr"] or "torch CUDA probe failed"]}
    available = "'available': True" in result["stdout"] or '"available": True' in result["stdout"]
    errors = ["CUDA is required but torch.cuda.is_available() is false"] if require_cuda and not available else []
    return {"status": "pass" if not errors else "fail", "required": require_cuda, "available": available, "output": result["stdout"], "errors": errors}


def check_docker(
    image: str | None,
    expected_digest: str | None,
    *,
    runtime_container: bool = False,
) -> dict[str, Any]:
    if not image:
        return {"status": "skip", "errors": [], "reason": "no image requested"}
    if runtime_container:
        return {
            "status": "skip",
            "image": image,
            "expected_digest": expected_digest,
            "errors": [],
            "reason": "image identity must be checked by the host control plane",
        }
    if not shutil.which("docker"):
        return {"status": "fail", "errors": ["docker CLI is not installed"]}
    result = run(["docker", "image", "inspect", image, "--format", "{{.Id}}"], timeout=20.0)
    if not result["ok"]:
        return {"status": "fail", "errors": [result["stderr"] or f"Docker image is unavailable: {image}"]}
    errors = []
    if expected_digest and result["stdout"] != expected_digest:
        errors.append(f"Docker image identity mismatch: expected {expected_digest}, observed {result['stdout']}")
    return {"status": "pass" if not errors else "fail", "image": image, "image_id": result["stdout"], "expected_digest": expected_digest, "errors": errors}


def check_wandb(mode: str, *, fallback: bool, python_executable: str | None = None) -> dict[str, Any]:
    if mode in {"disabled", "offline"}:
        return {"status": "pass", "requested": mode, "effective": mode, "errors": []}
    if not os.environ.get("WANDB_API_KEY"):
        error = "WANDB_API_KEY is absent; online initialization cannot be attempted"
        if fallback:
            return {"status": "warn", "requested": mode, "effective": "offline", "fallback_reason": error, "errors": []}
        return {"status": "fail", "requested": mode, "effective": None, "errors": [error]}
    result = run(
        [python_executable or sys.executable, "-c", "import wandb; api=wandb.Api(timeout=10); api.viewer; print('wandb-online-pass')"],
        timeout=20.0,
    )
    if result["ok"]:
        return {"status": "pass", "requested": mode, "effective": "online", "probe": result["stdout"], "errors": []}
    error = result["stderr"] or "W&B online probe failed"
    if fallback:
        return {"status": "warn", "requested": mode, "effective": "offline", "fallback_reason": error[:1000], "errors": []}
    return {"status": "fail", "requested": mode, "effective": None, "errors": [error[:1000]]}


def doctor(args: argparse.Namespace) -> dict[str, Any]:
    profile = getattr(args, "profile", "full")
    if profile not in ROUTE_PROFILES:
        raise ValueError(f"unsupported doctor profile: {profile}")
    route = ROUTE_PROFILES[profile]
    runtime_python = str(args.runtime_python.expanduser()) if getattr(args, "runtime_python", None) else None
    checks: dict[str, Any] = {
        "profile": {"status": "pass", "name": profile, "errors": []},
        "runtime_python": check_python_executable(getattr(args, "runtime_python", None)),
        "git": check_git(args.repo_root, allow_dirty=args.allow_dirty),
        "assets": check_runtime_assets(
            args.source_root,
            args.webshop_data,
            args.model,
            getattr(args, "index_root", None),
            require_seed_entry=route["require_seed_entry"],
        ),
        "model_completeness": check_model_completeness(args.model),
        "packages": check_packages(tuple(args.package or route["packages"]), python_executable=runtime_python),
        "imports": check_imports(tuple(args.import_name or route["imports"]), python_executable=runtime_python),
        "external_runtime_imports": check_external_runtime_imports(
            args.source_root,
            openevo_root=getattr(args, "openevo_root", None),
            enabled=getattr(args, "check_runtime_imports", False),
            python_executable=runtime_python,
            profile=profile,
        ),
        "external_source_identity": check_external_source_identity(
            args.source_root,
            openevo_root=getattr(args, "openevo_root", None),
            enabled=getattr(args, "check_runtime_imports", False),
            profile=profile,
            expected_openevo_patch_sha=getattr(args, "openevo_patch_sha", None),
        ),
        "tokenizer": check_tokenizer(args.model, enabled=getattr(args, "probe_tokenizer", False), python_executable=runtime_python),
        "pip_check": check_pip(python_executable=runtime_python),
        "cuda": check_cuda(args.require_cuda, python_executable=runtime_python),
        "nvidia_smi": check_nvidia_smi(args.require_cuda),
        "java": check_java(getattr(args, "require_java", False)),
        "docker": check_docker(
            args.image,
            args.expected_image_digest,
            runtime_container=getattr(args, "runtime_container", False),
        ),
        "wandb": check_wandb(args.wandb_mode, fallback=args.wandb_fallback, python_executable=runtime_python),
        "cache": check_cache(getattr(args, "cache_root", None)),
        "artifacts": check_write_target(getattr(args, "artifact_root", None)),
        "storage": check_storage(
            getattr(args, "storage_root", None) or getattr(args, "artifact_root", None),
            getattr(args, "minimum_free_bytes", DEFAULT_MIN_FREE_BYTES),
        ),
        "experiment_config": check_experiment_config(getattr(args, "experiment_config", None)),
        "run_manifest": check_run_manifest(getattr(args, "run_manifest", None)),
    }
    failures = [f"{name}: {error}" for name, value in checks.items() for error in value.get("errors", [])]
    return {
        "schema": "openevo.environment-doctor.v1",
        "profile": profile,
        "status": "PASS" if not failures else "FAIL",
        "python": sys.executable,
        "checks": checks,
        "failures": failures,
        "formal_task_consumption_allowed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--webshop-data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=tuple(ROUTE_PROFILES),
        default="full",
        help="runtime route to validate: full/seed checks SEED+OpenEvo; openevo checks the OpenEvo image only",
    )
    parser.add_argument(
        "--runtime-python",
        type=Path,
        help="interpreter used by the experiment; all package/import/CUDA probes run here",
    )
    parser.add_argument("--openevo-root", type=Path)
    parser.add_argument("--openevo-patch-sha", help="expected diff digest for the two approved patched OpenEvo trainer files")
    parser.add_argument("--index-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--storage-root", type=Path)
    parser.add_argument("--minimum-free-bytes", type=int, default=DEFAULT_MIN_FREE_BYTES)
    parser.add_argument("--image")
    parser.add_argument("--expected-image-digest")
    parser.add_argument(
        "--runtime-container",
        action="store_true",
        help="skip host Docker CLI inspection; image identity is checked before container launch",
    )
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-java", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--check-runtime-imports", action="store_true")
    parser.add_argument("--probe-tokenizer", action="store_true")
    parser.add_argument("--experiment-config", type=Path)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--package", action="append")
    parser.add_argument("--import-name", action="append")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = doctor(args)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"ENVIRONMENT: {result['status']}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
