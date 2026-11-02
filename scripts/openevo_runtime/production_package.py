"""Build the exact prospective Execution Runtime package identity without launching work."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.openevo_webshop.experiment_passport import canonical_bytes

SCHEMA = "openevo.execution-runtime-package.202609171852"
REQ = "requirements-execution-runtime-202609171852.txt"
BASE_FILES = (
    ".python-version", REQ,
    "configs/experiment/bounded-effective-state-qualified-topologies-202609170845.json",
    "schemas/execution-generation-202609170115.schema.json",
    "schemas/operation-intent-202609170352.schema.json",
    "schemas/operation-terminal-202609170352.schema.json",
    "schemas/exact-reuse-candidate-202609170352.schema.json",
    "schemas/runtime-event-202609171058.schema.json",
    "schemas/runtime-correlation-202609171101.schema.json",
    "schemas/resource-envelope-202609170600.schema.json",
    "schemas/resource-request-202609170600.schema.json",
    "schemas/resource-lease-202609170600.schema.json",
    "schemas/lease-release-202609170600.schema.json",
    "schemas/resource-blocked-202609170600.schema.json",
    "schemas/post-channel-dag-202609170650.schema.json",
    "schemas/process-supervision-contract-202609171050.schema.json",
    "schemas/execution-runtime-package-202609171852.schema.json",
    "scripts/openevo_webshop/experiment_passport.py",
    "scripts/openevo_webshop/experiment_registry.py",
    "scripts/openevo_webshop/experiment_artifact_dag.py",
    "scripts/openevo_webshop/ceiling1_ray_placement_backend_202609042025.py",
)

class ProductionPackageError(RuntimeError):
    pass

def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _module_file(module: str) -> str | None:
    if not module.startswith("scripts."):
        return None
    candidate = ROOT / (module.replace(".", "/") + ".py")
    if candidate.is_file():
        return str(candidate.relative_to(ROOT))
    return None


def _static_python_dependencies(relative: str) -> set[str]:
    """Return repo-local code plus static schema dependencies."""
    path = ROOT / relative
    if path.suffix != ".py":
        return set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        raise ProductionPackageError(
            f"cannot parse package dependency source: {relative}"
        ) from exc
    dependencies: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            direct = _module_file(node.module)
            if direct:
                dependencies.add(direct)
            elif node.module.startswith("scripts."):
                for alias in node.names:
                    child = _module_file(f"{node.module}.{alias.name}")
                    if child:
                        dependencies.add(child)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                direct = _module_file(alias.name)
                if direct:
                    dependencies.add(direct)

        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if (
                isinstance(node.left, ast.Name)
                and node.left.id == "ROOT"
                and isinstance(node.right, ast.Constant)
                and isinstance(node.right.value, str)
                and node.right.value.startswith("schemas/")
            ):
                schema = ROOT / node.right.value
                if not schema.is_file():
                    raise ProductionPackageError(
                        f"static package schema dependency missing: {node.right.value}"
                    )
                dependencies.add(node.right.value)
    return dependencies


def _runtime_files() -> list[str]:
    pending = {
        str(path.relative_to(ROOT))
        for path in sorted((ROOT / "scripts/openevo_runtime").glob("*.py"))
    }
    pending.update(BASE_FILES)
    resolved: set[str] = set()
    while pending:
        relative = min(pending)
        pending.remove(relative)
        if relative in resolved:
            continue
        path = ROOT / relative
        if not path.is_file():
            raise ProductionPackageError(f"package file missing: {relative}")
        resolved.add(relative)
        pending.update(_static_python_dependencies(relative) - resolved)
    return sorted(resolved)
def _temporal_pin() -> str:
    text = (ROOT / REQ).read_text(encoding="utf-8")
    m = re.search(r"(?m)^temporalio==([^\s#]+)$", text)
    if not m:
        raise ProductionPackageError("exact temporalio pin missing")
    return m.group(1)
def build_production_package() -> dict[str, Any]:
    rows = []
    for relative in _runtime_files():
        path = ROOT / relative
        if not path.is_file():
            raise ProductionPackageError(f"package file missing: {relative}")
        data = path.read_bytes()
        rows.append({"path": relative, "sha256": _sha(data), "size_bytes": len(data)})
    projection = {
        "schema": SCHEMA,
        "python_version": (ROOT / ".python-version").read_text(encoding="utf-8").strip(),
        "temporalio_version": _temporal_pin(),
        "files": rows,
        "scientific_authority": False,
        "formal_launch_authority": False,
        "gpu_authority": False,
        "final_panel_authority": False,
    }
    return {**projection, "package_identity": _sha(canonical_bytes(projection))}
def validate_production_package(package: Mapping[str, Any]) -> None:
    expected = build_production_package()
    if dict(package) != expected:
        raise ProductionPackageError("production package manifest does not match exact repository bytes")
def _clean_head() -> str:
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True)
    if dirty.strip():
        raise ProductionPackageError("repository must be clean before exact package readback")
    return head
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    head = _clean_head()
    if head != args.expected_git_sha:
        raise ProductionPackageError("expected Git SHA does not match clean checkout")
    package = build_production_package()
    validate_production_package(package)
    result = {"source_git_sha": head, "package": package, "zero_formal": True, "gpu_requests": 0}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_bytes(result) + b"\n")
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
