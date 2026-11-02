#!/usr/bin/env python3
"""Fail-closed authority gate for formal M1-A controlled replay.

The activation release is effective only from a clean checkout of freshly fetched
current main whose router points to the exact release and frozen preparation chain.
This script never runs WebShop and never allocates a GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

RELEASE_REL = Path("configs/experiment/activations/openevo-mechanism1-m1a-adjacent-execution-release-202609062300.json")
ROUTER_REL = Path("configs/experiment/campaigns/current-mechanism1-autonomous.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_static_chain(repo_root: Path) -> dict[str, Any]:
    release_path = repo_root / RELEASE_REL
    router_path = repo_root / ROUTER_REL
    release = read_json(release_path)
    router = read_json(router_path)
    lineage = router["m1a_lineage"]
    if release.get("status") != "READY_CONDITIONAL_CURRENT_MAIN":
        raise RuntimeError("activation release status drift")
    if release.get("activation_authority") is not True or release.get("formal_task_consumption_allowed") is not True:
        raise RuntimeError("activation release does not request formal authority")
    if release.get("scope") != "M1-A adjacent-distinct-state successor only":
        raise RuntimeError("activation scope drift")
    if int(release.get("formal_episode_cap", -1)) != 576:
        raise RuntimeError("activation budget drift")
    if lineage.get("activation_authority") is not True or lineage.get("formal_task_consumption_allowed") is not True:
        raise RuntimeError("router does not bind M1-A activation authority")
    if lineage.get("activation_release") != str(RELEASE_REL):
        raise RuntimeError("router activation release path drift")
    if lineage.get("activation_release_sha256") != sha256(release_path):
        raise RuntimeError("router activation release hash drift")
    gate = release["authority_gate"]
    gate_path = repo_root / gate["path"]
    if gate_path.resolve() != Path(__file__).resolve():
        raise RuntimeError("authority gate path drift")
    if sha256(gate_path) != gate["sha256"] or lineage.get("authority_gate_sha256") != gate["sha256"]:
        raise RuntimeError("authority gate hash drift")
    for name, row in release["frozen_preparation"].items():
        path = repo_root / row["path"]
        if sha256(path) != row["sha256"]:
            raise RuntimeError(f"frozen preparation hash drift: {name}")
    prep = read_json(repo_root / release["frozen_preparation"]["preparation_receipt"]["path"])
    qual = read_json(repo_root / release["frozen_preparation"]["runtime_qualification"]["path"])
    probe = read_json(repo_root / release["frozen_preparation"]["probe_manifest"]["path"])
    if prep.get("status") != "PASS_ACTIVATION_REQUIRED" or prep.get("execution_authority") is not False:
        raise RuntimeError("preparation receipt no longer has the expected pre-activation status")
    if qual.get("status") != "PASS" or qual.get("formal_webshop_tasks_consumed") != 0 or qual.get("final_panel_access") != 0:
        raise RuntimeError("runtime qualification accounting drift")
    if probe.get("cell_count") != 9 or probe.get("episodes_per_cell") != 64 or probe.get("closed_loop_episode_cap") != 576:
        raise RuntimeError("frozen probe matrix drift")
    if probe["probe_panel"].get("r27_to_r49_overlap_count") != 0:
        raise RuntimeError("probe/update-task exclusion drift")
    if probe["probe_panel"].get("ceiling_final_panel_goal_overlap_count") != 0 or probe["probe_panel"].get("ceiling_final_panel_instruction_hash_overlap_count") != 0:
        raise RuntimeError("Ceiling final-panel exclusion drift")
    return {"release_sha256": sha256(release_path), "gate_sha256": sha256(gate_path), "formal_episode_cap": 576}


def git_output(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo_root), *args], text=True).strip()


def evaluate_authority(repo_root: Path, current_head: str, current_main_head: str, working_tree_clean: bool = True) -> dict[str, Any]:
    static = verify_static_chain(repo_root)
    if not working_tree_clean:
        return {"status": "NOT_AUTHORIZED_DIRTY_WORKTREE", "authorized": False, "head": current_head, "current_main": current_main_head, **static}
    if current_head != current_main_head:
        return {"status": "NOT_AUTHORIZED_NOT_CURRENT_MAIN", "authorized": False, "head": current_head, "current_main": current_main_head, **static}
    return {"status": "AUTHORIZED_CURRENT_MAIN", "authorized": True, "head": current_head, "current_main": current_main_head, **static}


def live_authority(repo_root: Path, fetch_main: bool = True) -> dict[str, Any]:
    if fetch_main:
        subprocess.check_call(["git", "-C", str(repo_root), "fetch", "--quiet", "origin", "main"])
    head = git_output(repo_root, "rev-parse", "HEAD")
    main = git_output(repo_root, "rev-parse", "refs/remotes/origin/main")
    clean = not bool(git_output(repo_root, "status", "--porcelain", "--untracked-files=normal"))
    return evaluate_authority(repo_root, head, main, working_tree_clean=clean)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--no-fetch", action="store_true", help="testing only; formal execution must fresh-fetch current main")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = live_authority(args.repo_root.resolve(), fetch_main=not args.no_fetch)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(result["status"])
    return 0 if result["authorized"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
