#!/usr/bin/env python3
"""Validate and print dormant advisor work packages; never launch experiments.

A successful check establishes planning consistency only. It is neither a
runtime qualification nor an activation or scientific-completion receipt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA = "openevo.advisor-work-package.v1"
OWNERS = {"Q02": 629, "Q03": 630, "Q04": 631, "Q05": 632,
          "Q06": 633, "Q08": 635, "Q09": 636, "Q10": 637}
ROLES = {"audit", "primary", "confirmation", "conditional"}
COSTS = ("training_rollouts", "training_slots", "evaluation_episodes")


class PlanningError(ValueError):
    pass


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PlanningError(f"{name} must be an object")
    return value


def _strings(value: Any, name: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(x, str) or not x.strip() for x in value):
        raise PlanningError(f"{name} must be a list of nonempty strings")
    if nonempty and not value:
        raise PlanningError(f"{name} must not be empty")
    return value


def validate_package(package: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    if not isinstance(package, dict) or package.get("schema") != SCHEMA:
        raise PlanningError(f"schema must be {SCHEMA}")
    q = package.get("question_id")
    if not isinstance(q, str) or q not in OWNERS or type(package.get("owner_pr")) is not int or package["owner_pr"] != OWNERS[q]:
        raise PlanningError("question_id and owner_pr must identify the existing owner")
    if package.get("status") != "PLANNED_NOT_ACTIVATED":
        raise PlanningError("work packages must remain PLANNED_NOT_ACTIVATED")
    authority = _object(package.get("execution_authority"), "execution_authority")
    for key in ("formal_task_consumption_allowed", "gpu_allocation_allowed", "final_panel_access_allowed"):
        if authority.get(key) is not False:
            raise PlanningError(f"{key} must be explicitly false")
    if type(authority.get("final_panel_access")) is not int or authority["final_panel_access"] != 0:
        raise PlanningError("final_panel_access must be integer zero")
    plan = package.get("plan")
    if not isinstance(plan, str) or not plan or Path(plan).is_absolute() or ".." in Path(plan).parts:
        raise PlanningError("plan must be a repository-relative path without traversal")
    if root is not None:
        root = root.resolve()
        target = (root / plan).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise PlanningError("plan resolves outside the repository") from exc
        if not target.is_file():
            raise PlanningError(f"plan does not exist in this checkout: {plan}")
    science = _object(package.get("science"), "science")
    for key in ("estimand", "claim_boundary", "primary_metric"):
        if not isinstance(science.get(key), str) or not science[key].strip():
            raise PlanningError(f"science.{key} must name the scientific scope")
    threshold = science.get("practical_effect_threshold")
    if type(threshold) not in (int, float) or not (0 <= threshold < float("inf")):
        raise PlanningError("practical_effect_threshold must be a finite nonnegative number")
    deps = _object(package.get("dependencies"), "dependencies")
    for key in ("hard", "selection", "historical", "existing_execution_gates"):
        _strings(deps.get(key), f"dependencies.{key}")
    if q in deps["hard"]:
        raise PlanningError("a package cannot depend on itself")
    stages = package.get("stages")
    if not isinstance(stages, list) or not stages:
        raise PlanningError("stages must name at least one concrete stage")
    ids: set[str] = set()
    rows = []
    for stage in stages:
        if not isinstance(stage, dict):
            raise PlanningError("stage must be an object")
        sid = stage.get("id")
        if not isinstance(sid, str) or not sid.strip() or sid in ids:
            raise PlanningError("stage identifiers must be nonempty and unique")
        ids.add(sid)
        if stage.get("role") not in ROLES:
            raise PlanningError(f"stage {sid}: unknown role")
        if not isinstance(stage.get("trigger"), str) or not stage["trigger"].strip():
            raise PlanningError(f"stage {sid}: explicit prospective trigger required")
        _strings(stage.get("freeze_required"), f"stage {sid}.freeze_required")
        budget = _object(stage.get("budget"), f"stage {sid}.budget")
        for key in COSTS:
            if type(budget.get(key)) is not int or budget[key] < 0:
                raise PlanningError(f"stage {sid}: {key} must be a nonnegative integer")
        if budget["training_rollouts"] and budget["training_slots"]:
            raise PlanningError(f"stage {sid}: rollout and SEED-slot units must not be double counted")
        rows.append({"id": sid, "role": stage["role"], "budget": {k: budget[k] for k in COSTS},
                     "trigger": stage["trigger"], "freeze_required": stage["freeze_required"]})
    _strings(package.get("acceptance"), "acceptance", nonempty=True)
    luna = _object(package.get("luna"), "luna")
    for key in ("reading_order", "implementation_steps", "cpu_commands", "stop_conditions"):
        _strings(luna.get(key), f"luna.{key}", nonempty=True)
    _strings(luna.get("remaining_work"), "luna.remaining_work")
    return {"question_id": q, "owner_pr": OWNERS[q], "plan": plan, "stages": rows,
            "execution_authority": authority}


def validate_graph(packages: list[dict[str, Any]]) -> None:
    mapping = {p["question_id"]: p for p in packages}
    if len(mapping) != len(packages):
        raise PlanningError("multiple packages claim the same question")
    active: set[str] = set()
    finished: set[str] = set()

    def visit(q: str) -> None:
        if q in active:
            raise PlanningError(f"hard-dependency cycle at {q}")
        if q in finished:
            return
        active.add(q)
        for dep in mapping[q]["dependencies"]["hard"]:
            if dep in mapping:
                visit(dep)
        active.remove(q)
        finished.add(q)

    for q in mapping:
        visit(q)


def handoff(package: dict[str, Any]) -> str:
    lines = [f"{package['question_id']} / PR #{package['owner_pr']}",
             "Planning-only entry. Resolve live execution authority before scientific calls.",
             f"Plan: {package['plan']}", "", "Read in order:"]
    lines.extend(f"{i}. {x}" for i, x in enumerate(package["luna"]["reading_order"], 1))
    lines.extend(["", "Implement and verify:"])
    lines.extend(f"{i}. {x}" for i, x in enumerate(package["luna"]["implementation_steps"], 1))
    lines.extend(["", "Remaining work:"])
    lines.extend(f"- {x}" for x in package["luna"]["remaining_work"])
    lines.extend(["", "CPU commands:", *package["luna"]["cpu_commands"], "", "Stop conditions:"])
    lines.extend(f"- {x}" for x in package["luna"]["stop_conditions"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "budget", "handoff"))
    parser.add_argument("--package", type=Path, action="append", required=True)
    parser.add_argument("--repo-root", type=Path,
                        help="Otherwise infer each root from configs/experiment/advisor-questions")
    args = parser.parse_args()
    try:
        packages = []
        summaries = []
        for path in args.package:
            package = json.loads(path.read_text(encoding="utf-8"))
            if args.repo_root is not None:
                root = args.repo_root
            elif path.resolve().parent.parts[-3:] == ("configs", "experiment", "advisor-questions"):
                root = path.resolve().parents[3]
            else:
                raise PlanningError("noncanonical package location requires --repo-root")
            summaries.append(validate_package(package, root))
            packages.append(package)
        validate_graph(packages)
        if args.mode == "handoff":
            for package in packages:
                print(handoff(package), end="")
        else:
            print(json.dumps({"status": "PLANNING_VALIDATION_PASS", "scientific_completion": False,
                              "execution_authority_granted": False, "packages": summaries,
                              "budget_note": "Stages may be conditional or mutually exclusive; do not sum alternatives as launch authority."},
                             ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (PlanningError, ValueError, OSError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "PLANNING_VALIDATION_FAIL", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
