import copy
import importlib.util
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "scripts/analysis/advisor_work_package.py"
SPEC = importlib.util.spec_from_file_location("advisor_work_package", SOURCE)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def fixture(q="Q02"):
    return {
        "schema": module.SCHEMA, "question_id": q, "owner_pr": module.OWNERS[q],
        "status": "PLANNED_NOT_ACTIVATED", "plan": "docs/science/plan.md",
        "execution_authority": {"formal_task_consumption_allowed": False, "gpu_allocation_allowed": False,
                                "final_panel_access_allowed": False, "final_panel_access": 0},
        "science": {"estimand": "fixed-input effect", "claim_boundary": "one frozen stream",
                    "primary_metric": "Task Score", "practical_effect_threshold": .02},
        "dependencies": {"hard": [], "selection": [], "historical": [], "existing_execution_gates": []},
        "stages": [{"id": "A", "role": "primary", "trigger": "exact identity audit passes",
                    "budget": {"training_rollouts": 0, "training_slots": 0, "evaluation_episodes": 64},
                    "freeze_required": ["model revision"]}],
        "acceptance": ["sealed outcomes and statistics"],
        "luna": {"reading_order": ["AGENTS.md"], "implementation_steps": ["audit identity"],
                 "cpu_commands": ["python3 -m unittest"], "remaining_work": ["runtime qualification"],
                 "stop_conditions": ["final-panel overlap"]},
    }


class WorkPackageTests(unittest.TestCase):
    def test_validation_does_not_grant_execution(self):
        result = module.validate_package(fixture())
        self.assertIs(result["execution_authority"]["gpu_allocation_allowed"], False)

    def test_rejects_permission_and_false_like_values(self):
        for key, value in (("gpu_allocation_allowed", True), ("gpu_allocation_allowed", 0),
                           ("final_panel_access", False), ("final_panel_access", 1)):
            p = fixture()
            p["execution_authority"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(module.PlanningError):
                module.validate_package(p)

    def test_rejects_unbounded_or_double_counted_budget(self):
        for changes in ({"training_slots": -1}, {"training_rollouts": True},
                        {"training_slots": 128, "training_rollouts": 128}):
            p = fixture()
            p["stages"][0]["budget"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(module.PlanningError):
                module.validate_package(p)

    def test_rejects_wrong_owner_and_escape(self):
        for changes in ({"owner_pr": 630}, {"plan": "../outside.md"}, {"plan": "/tmp/outside.md"}):
            p = fixture()
            p.update(changes)
            with self.subTest(changes=changes), self.assertRaises(module.PlanningError):
                module.validate_package(p)

    def test_requires_prospective_trigger(self):
        p = fixture()
        p["stages"][0]["trigger"] = ""
        with self.assertRaises(module.PlanningError):
            module.validate_package(p)

    def test_malformed_nested_objects_fail_as_planning_errors(self):
        for key in ("execution_authority", "science", "dependencies", "luna"):
            for invalid in (None, [], "false"):
                p = fixture()
                p[key] = invalid
                with self.subTest(key=key, invalid=invalid), self.assertRaises(module.PlanningError):
                    module.validate_package(p)
        p = fixture()
        p["stages"][0]["budget"] = None
        with self.assertRaises(module.PlanningError):
            module.validate_package(p)
        p = fixture()
        p["question_id"] = []
        with self.assertRaises(module.PlanningError):
            module.validate_package(p)

    def test_rejects_duplicate_owner_and_dependency_cycles(self):
        a, b = fixture(), fixture("Q03")
        with self.assertRaises(module.PlanningError):
            module.validate_graph([a, copy.deepcopy(a)])
        a["dependencies"]["hard"] = ["Q03"]
        b["dependencies"]["hard"] = ["Q02"]
        with self.assertRaises(module.PlanningError):
            module.validate_graph([a, b])

    def test_unmerged_dependency_is_not_an_invented_cycle(self):
        p = fixture()
        p["dependencies"]["hard"] = ["Q01"]
        module.validate_graph([p])


if __name__ == "__main__":
    unittest.main()
