#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA = "openevo.formal-launch-stable-gate.202609161210"
PASS_STATUS = "PASS_STABLE_FORMAL_LAUNCH_PATH"
DEFAULT_WORKER = "scripts.openevo_webshop.ceiling1_stage2_rollout_worker_qwen3"
DEFAULT_SAFETY = "scripts/openevo_webshop/bounded_effective_state_runtime_safety_202609282338.py"
REQUIRED_GPU_ENV_TOKENS = (
    "CEILING1_STAGE2_GPU_UUIDS",
    "CEILING1_STAGE2_GPU_UUID",
    "BOUNDED_GPU_UUID",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def require_clean(repo: Path) -> None:
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True)
    if dirty.strip():
        raise RuntimeError("execution checkout is dirty")


def module_path(repo: Path, module: str) -> Path:
    return repo / (module.replace(".", "/") + ".py")


def local_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names if alias.name.startswith("scripts.openevo_webshop."))
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("scripts.openevo_webshop."):
            result.add(node.module)
    return result


def dependency_closure(repo: Path, root_module: str) -> dict:
    queue = [root_module]
    seen: set[str] = set()
    missing: list[str] = []
    files: dict[str, str] = {}
    while queue:
        module = queue.pop()
        if module in seen:
            continue
        seen.add(module)
        path = module_path(repo, module)
        if not path.is_file():
            missing.append(module)
            continue
        files[str(path.relative_to(repo))] = sha256(path)
        queue.extend(sorted(local_imports(path) - seen))
    if missing:
        raise RuntimeError(f"execution checkout runtime dependency closure missing modules: {sorted(missing)}")
    return {"root_module": root_module, "module_count": len(seen), "files": dict(sorted(files.items()))}


def launcher_env_bridge(repo: Path, relpath: str) -> dict:
    path = repo / relpath
    if not path.is_file():
        raise RuntimeError(f"formal runtime safety source missing: {relpath}")
    text = path.read_text(encoding="utf-8")
    missing = [token for token in REQUIRED_GPU_ENV_TOKENS if token not in text]
    if missing:
        raise RuntimeError(f"formal launcher GPU env bridge missing tokens: {missing}")
    return {"path": relpath, "sha256": sha256(path), "required_tokens": list(REQUIRED_GPU_ENV_TOKENS)}


def image_id(image: str) -> str:
    q = subprocess.run(["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True, capture_output=True)
    if q.returncode != 0 or not q.stdout.strip():
        raise RuntimeError(f"formal science image unavailable: {image}")
    return q.stdout.strip()


def production_import_smoke(repo: Path, core: Path, image: str, worker_module: str,
                            *, seed_root: Path | None = None) -> dict:
    py_path = f"{core}/src:{repo}:{repo}/vendor/openevo-webshop/src"
    probe = f"import {worker_module}; print('FORMAL_IMPORT_SMOKE_PASS')"
    if seed_root is not None:
        probe += (
            "; from openevo_webshop.env_compat import WebshopEnv, WebshopEnvConfig"
            f"; env=WebshopEnv(WebshopEnvConfig(seed_root={str(seed_root)!r},num_products=1000,seed=0))"
            "; env._build(); print('WEBSHOP_CONSTRUCTION_PASS')"
        )
    cmd = [
        "docker", "run", "--rm", "--network", "none", "--user", "1001:1001",
        "--read-only", "--tmpfs", "/tmp:rw,nosuid,size=1g", "--pull", "never",
        "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "HOME=/tmp",
        "-e", f"PYTHONPATH={py_path}",
        "-v", f"{core}:{core}:ro", "-v", f"{repo}:{repo}:ro",
    ]
    if seed_root is not None:
        cmd += ["-v", f"{seed_root}:{seed_root}:ro"]
    cmd += ["--entrypoint", "/usr/local/bin/python3", image, "-c", probe]
    q = subprocess.run(cmd, text=True, capture_output=True)
    if q.returncode != 0 or "FORMAL_IMPORT_SMOKE_PASS" not in q.stdout or (
        seed_root is not None and "WEBSHOP_CONSTRUCTION_PASS" not in q.stdout
    ):
        detail = (q.stderr or q.stdout).strip()
        raise RuntimeError(f"formal science-image worker import smoke failed: {detail}")
    return {"status": "PASS", "pythonpath": py_path, "network": "none", "user": "1001:1001",
            "command": cmd, "real_boundary": "WebshopEnv._build" if seed_root is not None else "worker_import",
            "formal_task_consumption_allowed": False, "gpu_requested": False}


def webshop_boundary_identity(repo: Path, core: Path, seed_root: Path) -> dict:
    from scripts.openevo_webshop.runtime_qualification import _git_identity
    core_identity = _git_identity(core)
    if not core_identity or core_identity.get("git") is not True or not core_identity.get("head") or core_identity.get("status") == "unavailable":
        raise RuntimeError("WebShop core source identity unavailable")
    webshop = seed_root / "agent_system/environments/env_package/webshop/webshop"
    required = [webshop / "data" / name for name in
                ("items_shuffle_1000.json", "items_ins_v2_1000.json", "items_human_ins.json")]
    index = webshop / "search_engine/indexes_1k"
    if not index.is_dir() or not any(index.iterdir()):
        raise RuntimeError("WebShop construction index missing")
    files = required + sorted(webshop.rglob("*.py")) + sorted(p for p in index.rglob("*") if p.is_file())
    files.append(repo / "vendor/openevo-webshop/src/openevo_webshop/env_compat.py")
    return {"core": core_identity, "seed_root": str(seed_root),
            "files": {str(path): sha256(path) for path in files}}


def build_receipt(repo: Path, core: Path, image: str, worker_module: str, safety_relpath: str,
                  *, seed_root: Path | None = None) -> dict:
    started = time.monotonic()
    require_clean(repo)
    head = git_head(repo)
    observed_image = image_id(image)
    closure = dependency_closure(repo, worker_module)
    bridge = launcher_env_bridge(repo, safety_relpath)
    boundary = webshop_boundary_identity(repo, core, seed_root) if seed_root is not None else None
    smoke = production_import_smoke(repo, core, observed_image, worker_module, seed_root=seed_root) if seed_root is not None else production_import_smoke(repo, core, observed_image, worker_module)
    if boundary is not None and webshop_boundary_identity(repo, core, seed_root) != boundary:
        raise RuntimeError("WebShop boundary changed during qualification")
    elapsed = time.monotonic() - started
    return {
        "schema": SCHEMA,
        "status": PASS_STATUS,
        "execution_git_sha": head,
        "science_image": image,
        "science_image_id": observed_image,
        "worker_module": worker_module,
        "dependency_closure": closure,
        "launcher_gpu_env_bridge": bridge,
        "science_image_import_smoke": smoke,
        "webshop_boundary_identity": boundary,
        "stable_gate_elapsed_seconds": round(elapsed, 3),
        "owner_hot_path_contract": {
            "stable_checks_must_not_repeat_after_owner_start": True,
            "allowed_after_owner_start": [
                "fresh_gpu_owner_uuid_vram_util_process_gate",
                "fresh_storage_gate",
                "ready_and_release_identity_check",
                "mint_owner_launch_release",
                "start_formal_runner",
            ],
            "workload_start_target_seconds": 30,
            "workload_start_witness": "formal rollout container plus target-GPU load or first formal task artifact",
            "runner_pid_alone_is_not_success": True,
        },
    }


def validate_receipt(repo: Path, receipt_path: Path) -> dict:
    doc = json.loads(receipt_path.read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA or doc.get("status") != PASS_STATUS:
        raise RuntimeError("stable launch-path receipt is not PASS")
    require_clean(repo)
    if doc.get("execution_git_sha") != git_head(repo):
        raise RuntimeError("stable launch-path receipt execution SHA drift")
    for relpath, expected in (doc.get("dependency_closure") or {}).get("files", {}).items():
        path = repo / relpath
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"stable launch-path dependency drift: {relpath}")
    bridge = doc.get("launcher_gpu_env_bridge") or {}
    path = repo / str(bridge.get("path") or "")
    if not path.is_file() or sha256(path) != bridge.get("sha256"):
        raise RuntimeError("stable launch-path GPU env bridge drift")
    if image_id(str(doc.get("science_image") or "")) != doc.get("science_image_id"):
        raise RuntimeError("stable launch-path science image drift")
    boundary = doc.get("webshop_boundary_identity")
    if boundary is not None:
        current = webshop_boundary_identity(repo, Path(boundary["core"]["path"]), Path(boundary["seed_root"]))
        if current != boundary:
            raise RuntimeError("stable launch-path WebShop boundary drift")
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execution-repo", type=Path, required=True)
    ap.add_argument("--core-repo", type=Path, required=True)
    ap.add_argument("--science-image", required=True)
    ap.add_argument("--worker-module", default=DEFAULT_WORKER)
    ap.add_argument("--runtime-safety", default=DEFAULT_SAFETY)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--webshop-seed-root", type=Path, help="engineering-only real WebShop construction; no task reset, model or GPU")
    ap.add_argument("--validate-receipt", type=Path)
    args = ap.parse_args()
    repo = args.execution_repo.resolve()
    if args.validate_receipt:
        doc = validate_receipt(repo, args.validate_receipt.resolve())
        print(json.dumps({"status": doc["status"], "execution_git_sha": doc["execution_git_sha"]}, sort_keys=True))
        return 0
    if args.output is None:
        raise SystemExit("--output is required when building a stable gate receipt")
    doc = build_receipt(repo, args.core_repo.resolve(), args.science_image, args.worker_module, args.runtime_safety,
                        seed_root=args.webshop_seed_root.resolve() if args.webshop_seed_root else None)
    rendered = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_text(encoding="utf-8") != rendered:
        raise RuntimeError("stable launch-path receipt immutable drift")
    if not args.output.exists():
        args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps({"status": doc["status"], "execution_git_sha": doc["execution_git_sha"], "receipt_sha256": sha256(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
