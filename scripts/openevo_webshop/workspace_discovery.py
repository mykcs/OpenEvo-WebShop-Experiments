#!/usr/bin/env python3
"""Server workspace discovery for OpenEvo fast-start entrypoint.

This module is dependency-free and read-only. It tells a fresh Agent/session:
- whether the current checkout is the canonical control plane,
- where the canonical control/runtime checkouts live,
- and what the next legal command is.

It never installs packages, rebuilds runtimes, or does GPU work.
"""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from typing import Any


DEFAULT_REGISTRY = Path("/workspace/control/openevo-experiment/configs/platform/server-registry.json")
FALLBACK_CONTROL_GUESSES = (
    "/workspace/control/openevo-experiment",
)


class WorkspaceDiscoveryError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_registry(starting_path: Path | None = None) -> Path | None:
    """Look upward from the current checkout for a registry, then at fallbacks."""
    if starting_path is not None:
        probe = starting_path.resolve()
        for _ in range(4):
            candidate = probe / "configs" / "platform" / "server-registry.json"
            if candidate.is_file():
                return candidate
            candidate = probe / "server-registry.json"
            if candidate.is_file():
                return candidate
            if probe.parent == probe:
                break
            probe = probe.parent
    for guess in FALLBACK_CONTROL_GUESSES:
        candidate = Path(guess) / "configs" / "platform" / "server-registry.json"
        if candidate.is_file():
            return candidate
    return None


def _canonical_path(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser().resolve()


def classify_checkout(
    current_checkout: Path | str,
    *,
    registry_path: Path | str | None = None,
    env_override: str | None = None,
) -> dict[str, Any]:
    """Classify ``current_checkout`` against the canonical server registry.

    Returns a machine-readable dict with ``is_canonical_control``,
    ``is_canonical_runtime``, and the next legal command.
    """
    current = Path(current_checkout).expanduser().resolve()

    if env_override:
        canonical_control = _canonical_path(env_override)
        canonical_runtime = None
        registry_file = None
    elif registry_path is not None:
        registry_file = Path(registry_path).expanduser().resolve()
        if not registry_file.is_file():
            raise WorkspaceDiscoveryError(f"registry not found: {registry_file}")
        registry = _load_json(registry_file)
        canonical_control = _canonical_path(registry.get("canonical_control_checkout"))
        canonical_runtime = _canonical_path(registry.get("canonical_runtime_checkout"))
    else:
        registry_file = _find_registry(current)
        if registry_file is None or not registry_file.is_file():
            return {
                "status": "NO_REGISTRY",
                "current_path": str(current),
                "hostname": socket.gethostname(),
                "canonical_control_checkout": None,
                "canonical_runtime_checkout": None,
                "is_canonical_control": False,
                "is_canonical_runtime": False,
                "is_non_canonical": True,
                "registry_path": str(DEFAULT_REGISTRY),
                "next_action": "NO_REGISTRY",
                "next_command": None,
                "error": "Cannot find server-workspace registry. Ask the operator to create configs/platform/server-registry.json in the canonical control checkout.",
            }
        registry = _load_json(registry_file)
        canonical_control = _canonical_path(registry.get("canonical_control_checkout"))
        canonical_runtime = _canonical_path(registry.get("canonical_runtime_checkout"))

    is_control = canonical_control is not None and current == canonical_control
    is_runtime = canonical_runtime is not None and current == canonical_runtime

    result: dict[str, Any] = {
        "status": "OK" if (is_control or is_runtime) else "NON_CANONICAL_CHECKOUT",
        "current_path": str(current),
        "hostname": socket.gethostname(),
        "canonical_control_checkout": str(canonical_control) if canonical_control else None,
        "canonical_runtime_checkout": str(canonical_runtime) if canonical_runtime else None,
        "is_canonical_control": is_control,
        "is_canonical_runtime": is_runtime,
        "is_non_canonical": not (is_control or is_runtime),
        "registry_path": str(registry_file) if registry_file else None,
    }

    if is_control:
        result["next_action"] = "OK"
        result["next_command"] = "./scripts/oe doctor"
    elif is_runtime:
        result["next_action"] = "OK_RUNTIME_CHECKOUT"
        result["next_command"] = f"cd {canonical_control} && ./scripts/oe doctor"
    elif canonical_control is not None:
        result["next_action"] = "NON_CANONICAL_EXECUTION_CHECKOUT"
        result["next_command"] = f"cd {canonical_control} && ./scripts/oe doctor"
    else:
        result["next_action"] = "NO_CANONICAL_CONTROL_DECLARED"
        result["next_command"] = None

    return result


def discover_workspace(
    *,
    current_checkout: Path | str | None = None,
    registry_path: Path | str | None = None,
) -> dict[str, Any]:
    """Return machine-readable workspace discovery result (legacy flat API)."""
    current = Path(current_checkout or Path.cwd()).expanduser().resolve()
    result = classify_checkout(current, registry_path=registry_path)

    flat: dict[str, Any] = {
        "status": result["status"],
        "registry_path": result.get("registry_path"),
        "host": result["hostname"],
        "current_checkout": result["current_path"],
        "canonical_control_checkout": result["canonical_control_checkout"],
        "canonical_runtime_checkout": result["canonical_runtime_checkout"],
        "is_canonical_control": result["is_canonical_control"],
    }

    if result["is_canonical_control"]:
        flat["next_command"] = "./scripts/oe doctor"
        flat["message"] = "This is the canonical control checkout. Run './scripts/oe doctor' for fast-start status."
    else:
        canonical_control = result["canonical_control_checkout"]
        if canonical_control:
            flat["next_command"] = f"cd {canonical_control} && ./scripts/oe doctor"
            flat["message"] = (
                f"NON_CANONICAL_EXECUTION_CHECKOUT: current={current}\n"
                f"canonical_control_checkout={canonical_control}\n"
                f"Next legal command: cd {canonical_control} && ./scripts/oe doctor"
            )
        else:
            flat["next_command"] = None
            flat["message"] = result.get("error", "NO_CANONICAL_CONTROL_DECLARED")

    return flat


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--registry", type=Path, help="override registry path")
    args = parser.parse_args(argv)

    result = discover_workspace(registry_path=args.registry)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(result.get("message", result["status"]))

    return 0 if result["status"] == "OK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
