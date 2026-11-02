from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.ci import plan_cpu_gate


class CpuCiRelevanceTests(unittest.TestCase):
    def test_root_readme_and_prose_docs_use_docs_mode(self) -> None:
        self.assertEqual(
            plan_cpu_gate.classify_changed_paths(
                ["README.md", "docs/infrastructure/ci/README.md"]
            ),
            "docs",
        )

    def test_scientific_evidence_markdown_forces_full(self) -> None:
        self.assertEqual(
            plan_cpu_gate.classify_changed_paths(
                ["docs/evidence/remote-runs/example/REPORT.md"]
            ),
            "full",
        )

    def test_normative_markdown_forces_full(self) -> None:
        paths = (
            "docs/experiment-tracking/OPEN_EVO_CEILING_1.0_END_TO_END_MASTER_PLAN.md",
            "docs/infrastructure/ci/CI_POLICY.md",
            "docs/science/webshop/program/OPEN_EVO_MECHANISM_1.0_MASTER_PLAN.md",
            "docs/operations/execution/SHARED_ARM_EXECUTION_RULES.md",
            "docs/operations/execution/FRICTION_REGISTER.md",
            "docs/operations/execution/EXECUTION_READINESS.md",
            "docs/science/webshop/benchmark/WB1_AMENDMENT_006_GEN28_DEFERRED_NATIVE_STATE_UPDATE_2026-08-27.md",
        )
        for path in paths:
            with self.subTest(path=path):
                self.assertEqual(plan_cpu_gate.classify_changed_paths([path]), "full")

    def test_machine_readable_docs_force_full(self) -> None:
        self.assertEqual(
            plan_cpu_gate.classify_changed_paths(["docs/DOCS_INDEX.v1.json"]),
            "full",
        )

    def test_agent_authority_files_force_full(self) -> None:
        for path in ("AGENTS.md", "CLAUDE.md"):
            with self.subTest(path=path):
                self.assertEqual(plan_cpu_gate.classify_changed_paths([path]), "full")

    def test_code_or_mixed_change_forces_full(self) -> None:
        cases = (
            ["scripts/ci/preflight.py"],
            ["configs/experiment/current-campaign.json"],
            ["tests/test_ci_contract.py"],
            ["docs/README.md", "scripts/ci/preflight.py"],
            ["vendor/openevo-webshop/README.md"],
            [],
        )
        for paths in cases:
            with self.subTest(paths=paths):
                self.assertEqual(plan_cpu_gate.classify_changed_paths(paths), "full")

    def test_historical_pr_replay_keeps_scope_semantics(self) -> None:
        fixtures = {
            313: (["docs/infrastructure/ci/CIRCLECI_CLOUD_CUTOVER_RETROSPECTIVE_2026-09-05.md", "docs/infrastructure/ci/CI_POLICY.md", "docs/infrastructure/ci/README.md", "docs/operations/execution/FRICTION_REGISTER.md"], "full"),
            314: (["AGENTS.md", "docs/operations/execution/FRICTION_REGISTER.md"], "full"),
            312: (["docs/evidence/server-audits/2026-09-05/RECLAIM_EXECUTION_CLOSEOUT.md", "docs/evidence/server-audits/2026-09-05/reclaim-execution-receipt-20260905.json"], "full"),
            311: (["AGENTS.md", "docs/infrastructure/ci/CI_POLICY.md", "docs/operations/execution/SHARED_ARM_EXECUTION_RULES.md"], "full"),
        }
        for pr, (paths, expected) in fixtures.items():
            with self.subTest(pr=pr):
                self.assertEqual(plan_cpu_gate.classify_changed_paths(paths), expected)

    def test_git_base_diff_includes_both_rename_paths(self) -> None:
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            subprocess.run(["git","init","-q",root],check=True)
            subprocess.run(["git","-C",root,"config","user.email","ci@example.invalid"],check=True)
            subprocess.run(["git","-C",root,"config","user.name","CI"],check=True)
            (root/"scripts").mkdir(); (root/"scripts/x.py").write_text("x=1\n")
            subprocess.run(["git","-C",root,"add","."],check=True); subprocess.run(["git","-C",root,"commit","-qm","base"],check=True)
            base=subprocess.check_output(["git","-C",root,"rev-parse","HEAD"],text=True).strip()
            (root/"docs").mkdir(); subprocess.run(["git","-C",root,"mv","scripts/x.py","docs/x.md"],check=True)
            subprocess.run(["git","-C",root,"commit","-qam","rename"],check=True)
            changed=plan_cpu_gate.changed_paths_from_git_base(base,root)
            self.assertEqual(changed,["docs/x.md","scripts/x.py"])
            self.assertEqual(plan_cpu_gate.classify_changed_paths(changed),"full")

    def test_documentation_test_discovery_is_nonempty_and_grounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tests = root / "tests"
            tests.mkdir()
            (tests / "test_documentation_links.py").write_text("# docs/README.md\n")
            (tests / "test_unrelated.py").write_text("def test_x(): pass\n")
            selected = plan_cpu_gate.documentation_test_paths(root)
            self.assertEqual(selected, ["tests/test_documentation_links.py"])

    def test_documentation_test_discovery_includes_real_contracts(self) -> None:
        selected = set(plan_cpu_gate.documentation_test_paths())
        self.assertIn("tests/test_documentation_links.py", selected)
        self.assertIn("tests/test_documentation_state_hygiene.py", selected)
        self.assertIn("tests/test_ci_contract.py", selected)
        self.assertGreaterEqual(len(selected), 10)


if __name__ == "__main__":
    unittest.main()
