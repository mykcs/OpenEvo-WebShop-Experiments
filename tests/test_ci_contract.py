from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.ci import preflight


class CiContractTests(unittest.TestCase):
    def test_current_repository_preflight_is_clean(self) -> None:
        self.assertEqual(preflight.run_checks(), [])

    def test_preflight_runs_timestamp_guard_instead_of_only_describing_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root / "scripts"
            scripts.mkdir(parents=True)
            shutil.copy2(
                preflight.ROOT / "scripts/validate_timestamp_revision_naming.py",
                scripts / "validate_timestamp_revision_naming.py",
            )
            subprocess.run(["git", "init", "-b", "main"], cwd=root, check=True, capture_output=True)
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(
                ["git", "-c", "user.name=CI Test", "-c", "user.email=ci@example.invalid", "commit", "-m", "baseline"],
                cwd=root,
                check=True,
                capture_output=True,
            )
            subprocess.run(["git", "switch", "-c", "feature"], cwd=root, check=True, capture_output=True)
            bad_name = "new-protocol-" + "v" + "1.json"
            bad = root / "configs/experiment/designs" / bad_name
            bad.parent.mkdir(parents=True)
            bad.write_text("{}\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(
                ["git", "-c", "user.name=CI Test", "-c", "user.email=ci@example.invalid", "commit", "-m", "bad revision"],
                cwd=root,
                check=True,
                capture_output=True,
            )

            errors = preflight.check_timestamp_revision_naming(root)
            self.assertTrue(any(bad_name in error for error in errors), errors)

    def test_unpinned_ci_dependency_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "requirements-ci.txt").write_text(
                "pytest>=8\nbuild==1.2.2.post1\n",
                encoding="utf-8",
            )
            errors = preflight.check_ci_requirements(root)
            self.assertTrue(any("pytest>=8" in error for error in errors), errors)

    @staticmethod
    def _write_ci_contract(
        root: Path,
        *,
        test_command: str = '"$PYTHON_BIN" -m pytest -q -n 2 tests',
        automatic_gpu: bool = False,
    ) -> None:
        ci_dir = root / "scripts/ci"
        ci_dir.mkdir(parents=True)
        (ci_dir / "cloudflare-cpu-ci.sh").write_text(
            f"""\
#!/usr/bin/env bash
"$PYTHON_BIN" scripts/ci/preflight.py
python -m pip install --requirement requirements-ci.txt
{test_command}
"$PYTHON_BIN" -m build
python -c "main(['contract-test'])"
test ! -e /dev/nvidia0
echo CLOUDFLARE_API_TOKEN
echo /proc/[0-9]*/environ
""",
            encoding="utf-8",
        )
        (ci_dir / "cloudflare-pages-build.sh").write_text(
            """#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="$ROOT/.cloudflare-ci-pages"
rm -rf "$OUTPUT_DIR"
bash "$ROOT/scripts/ci/cloudflare-cpu-ci.sh"
mkdir -p "$OUTPUT_DIR"
printf '%s\n' 'OpenEvo CPU CI passed.' > "$OUTPUT_DIR/index.html"
""",
            encoding="utf-8",
        )
        (root / ".python-version").write_text("3.13.3\n", encoding="utf-8")

        workflow = root / ".github/workflows/gpu-smoke.yml"
        workflow.parent.mkdir(parents=True)
        auto_trigger = "  push:\n    branches: [main]\n" if automatic_gpu else ""
        workflow.write_text(
            f"""\
on:
{auto_trigger}  workflow_dispatch:
    inputs:
      authorized_gpu_uuid:
        required: true
permissions:
  contents: read
jobs:
  gpu-smoke:
    if: github.event_name == 'workflow_dispatch'
    timeout-minutes: 10
    runs-on: [self-hosted, linux, x64, gpu-5090]
    env:
      AUTHORIZED_GPU_UUID: ${{{{ inputs.authorized_gpu_uuid }}}}
    steps:
      - uses: actions/checkout@{preflight.ACTIONS_CHECKOUT_SHA}
        with:
          persist-credentials: false
      - run: nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
      - run: echo "compute apps; smoke not started"
""",
            encoding="utf-8",
        )

        circleci = root / ".circleci/config.yml"
        circleci.parent.mkdir(parents=True, exist_ok=True)
        circleci.write_text(
            """version: 2.1
jobs:
  cpu_ci:
    docker:
      - image: cimg/python:3.13.3@sha256:a00bfa949271c5519c63d623c6c5ab61fa0868f88ffa3a66b0983bf7df376f3d
    resource_class: small
    steps:
      - run:
          command: |
            test ! -e /dev/nvidia0
            test ! -e /var/run/docker.sock
            git fetch --no-tags origin main
            base_ref=origin/main
            python3 scripts/validate_timestamp_revision_naming.py --base-ref "$base_ref"
            mode="$(python3 scripts/ci/plan_cpu_gate.py --base-ref origin/main)"
            echo "export OPENEVO_CPU_CI_MODE=$mode" >> "$BASH_ENV"
            bash scripts/ci/cloudflare-cpu-ci.sh --mode "$OPENEVO_CPU_CI_MODE"
            /home/circleci/.cache/pip
            {{ checksum "requirements-ci.txt" }}
workflows:
  cloud_cpu_ci:
    jobs:
      - cpu_ci
""",
            encoding="utf-8",
        )

        revision = root / ".github/workflows/revision-naming-202609041833.yml"
        revision.parent.mkdir(parents=True, exist_ok=True)
        revision.write_text(
            f"""name: Revision naming timestamp guard (manual fallback)
on:
  workflow_dispatch:
permissions:
  contents: read
jobs:
  timestamp-revision-naming:
    if: github.event_name == 'workflow_dispatch' && github.actor == 'mykcs'
    runs-on: [self-hosted, openevo-mac-ci]
    steps:
      - uses: actions/checkout@{preflight.ACTIONS_CHECKOUT_SHA}
        with:
          fetch-depth: 0
          persist-credentials: false
      - run: python3 scripts/validate_timestamp_revision_naming.py --base-ref HEAD^
""",
            encoding="utf-8",
        )

        gate = root / ".github/workflows/base-owned-integration-gate-202609060057.yml"
        gate.write_text(
            f"""name: Base-owned integration gate 202609060057
on:
  pull_request_target:
    types: [labeled]
permissions:
  contents: read
  pull-requests: read
jobs:
  base-owned-integration-gate-202609060057:
    if: github.event.label.name == 'ci-legacy-gate' && github.event.pull_request.draft == false
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@{preflight.ACTIONS_CHECKOUT_SHA}
        with:
          ref: ${{{{ github.event.pull_request.base.sha }}}}
          persist-credentials: false
      - run: python3 scripts/ci/base_owned_integration_gate_202609060057.py
      - uses: actions/checkout@{preflight.ACTIONS_CHECKOUT_SHA}
        with:
          ref: ${{{{ github.event.pull_request.head.sha }}}}
          persist-credentials: false
      - run: |
          docker run --network none --read-only --cpus 2 --memory 4g --pids-limit 1024 --cap-drop ALL --security-opt no-new-privileges:true \
            --tmpfs /tmp:rw,exec,nosuid,nodev,size=3g,uid=1001,gid=1002 image \
            sh -c 'test ! -e /var/run/docker.sock; PIP_NO_INDEX=1 python3 -m pytest -q -n 2 tests; python3 -m build --no-isolation --wheel .; echo contract-test'
""",
            encoding="utf-8",
        )

        mac_text = f"""name: Mac CPU fallback qualification
on:
  workflow_dispatch:
permissions:
  contents: read
jobs:
  qualify:
    if: github.event_name == 'workflow_dispatch' && github.actor == 'mykcs'
    runs-on: [self-hosted, openevo-mac-ci]
    steps:
      - uses: actions/checkout@{preflight.ACTIONS_CHECKOUT_SHA}
        with:
          fetch-depth: 1
          persist-credentials: false
      - run: |
          test ! -e /dev/nvidia0
          test ! -e /var/run/docker.sock
      - run: bash scripts/ci/cloudflare-cpu-ci.sh
"""
        fallback_path = root / ".github/workflows/mac-cpu-fallback-qualification.yml"
        template_path = root / ".github/runner/openevo-mac/automatic-cpu-ci.workflow.yml"
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        template_path.parent.mkdir(parents=True, exist_ok=True)
        fallback_path.write_text(
            mac_text.replace("fetch-depth: 1", "fetch-depth: 0"),
            encoding="utf-8",
        )
        template_path.write_text(mac_text, encoding="utf-8")

    def test_mutable_action_tag_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_ci_contract(root)
            workflow = root / ".github/workflows/gpu-smoke.yml"
            workflow.write_text(
                workflow.read_text(encoding="utf-8").replace(
                    preflight.ACTIONS_CHECKOUT_SHA, "v7"
                ),
                encoding="utf-8",
            )
            errors = preflight.check_workflow_contract(root)
            self.assertTrue(
                any("full-length commit SHA" in error for error in errors),
                errors,
            )

    def test_unreviewed_checkout_sha_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_ci_contract(root)
            workflow = root / ".github/workflows/gpu-smoke.yml"
            workflow.write_text(
                workflow.read_text(encoding="utf-8").replace(
                    preflight.ACTIONS_CHECKOUT_SHA, "0" * 40
                ),
                encoding="utf-8",
            )
            errors = preflight.check_workflow_contract(root)
            self.assertTrue(
                any("pinned to reviewed" in error for error in errors),
                errors,
            )

    def test_legacy_automatic_mac_cpu_gate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_ci_contract(root)
            legacy = root / ".github/workflows/mac-cpu-ci.yml"
            legacy.write_text("name: legacy automatic Mac CI\n", encoding="utf-8")
            errors = preflight.check_workflow_contract(root)
            self.assertTrue(
                any("must stay retired" in error for error in errors),
                errors,
            )

    def test_missing_circleci_cpu_gate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_ci_contract(root)
            (root / ".circleci/config.yml").unlink()
            errors = preflight.check_workflow_contract(root)
            self.assertTrue(
                any(".circleci/config.yml" in error for error in errors),
                errors,
            )

    def test_manual_mac_fallback_requires_full_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_ci_contract(root)
            workflow = root / ".github/workflows/mac-cpu-fallback-qualification.yml"
            workflow.write_text(
                workflow.read_text(encoding="utf-8").replace(
                    "fetch-depth: 0", "fetch-depth: 1"
                ),
                encoding="utf-8",
            )
            errors = preflight.check_workflow_contract(root)
            self.assertTrue(
                any("must fetch full history" in error for error in errors),
                errors,
            )

    def test_scientific_runner_label_in_mac_fallback_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_ci_contract(root)
            workflow = root / ".github/workflows/mac-cpu-fallback-qualification.yml"
            workflow.write_text(
                workflow.read_text(encoding="utf-8").replace(
                    "runs-on: [self-hosted, openevo-mac-ci]",
                    "runs-on: [self-hosted, linux, x64, gpu-5090]",
                ),
                encoding="utf-8",
            )
            errors = preflight.check_workflow_contract(root)
            self.assertTrue(
                any("manual Mac fallback must not target scientific-server runner labels" in error for error in errors),
                errors,
            )

    def test_revision_naming_fallback_must_be_manual_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_ci_contract(root)
            workflow = root / ".github/workflows/revision-naming-202609041833.yml"
            workflow.write_text(
                workflow.read_text(encoding="utf-8").replace(
                    "  workflow_dispatch:\n",
                    "  workflow_dispatch:\n  push:\n",
                ),
                encoding="utf-8",
            )
            errors = preflight.check_workflow_contract(root)
            self.assertTrue(
                any("revision naming fallback must remain manual-only" in error for error in errors),
                errors,
            )

    def test_unittest_only_cloudflare_gate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_ci_contract(
                root,
                test_command="python3 -m unittest discover -s tests",
            )
            errors = preflight.check_workflow_contract(root)
            self.assertTrue(
                any("complete pytest collection" in error for error in errors),
                errors,
            )
            self.assertTrue(
                any("unittest-only discovery" in error for error in errors),
                errors,
            )

    def test_automatic_gpu_smoke_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_ci_contract(root, automatic_gpu=True)
            errors = preflight.check_workflow_contract(root)
            self.assertTrue(
                any("manual-only" in error and "push/PR triggers" in error for error in errors),
                errors,
            )

    def test_self_hosted_cpu_route_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_ci_contract(root)
            legacy = root / ".github/workflows/legacy-cpu.yml"
            legacy.write_text(
                "jobs:\n"
                "  cpu-check:\n"
                "    runs-on: [self-hosted, linux, x64, cpu-only]\n",
                encoding="utf-8",
            )
            errors = preflight.check_workflow_contract(root)
            self.assertTrue(
                any("must not route automatic CPU CI" in error for error in errors),
                errors,
            )


class CiRedundantBudgetTests(unittest.TestCase):
    """Keep redundant work conditional without weakening the trusted full gate."""

    def assert_budgeted_config(self, text: str) -> None:
        lines = [line.rstrip() for line in text.splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
        parameter_index = lines.index("parameters:")
        self.assertEqual(lines[parameter_index:parameter_index + 4], [
            "parameters:", "  run_redundant_pr_ci:",
            "    type: boolean", "    default: false",
        ])
        workflow_index = lines.index("workflows:")
        self.assertEqual(lines[workflow_index:], [
            "workflows:", "  manual_cpu_ci:",
            "    when: << pipeline.parameters.run_redundant_pr_ci >>",
            "    jobs:", "      - cpu_ci",
        ])

    @staticmethod
    def current_config() -> str:
        return (preflight.ROOT / ".circleci/config.yml").read_text(encoding="utf-8")

    def test_explicit_opt_in_is_the_only_circleci_route(self) -> None:
        self.assert_budgeted_config(self.current_config())

    def test_redundant_opt_in_cannot_silently_default_true(self) -> None:
        with self.assertRaises(AssertionError):
            self.assert_budgeted_config(self.current_config().replace("default: false", "default: true"))

    def test_missing_pre_allocation_condition_is_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            self.assert_budgeted_config(
                self.current_config().replace(
                    "    when: << pipeline.parameters.run_redundant_pr_ci >>\n", ""
                )
            )

    def test_automatic_main_route_is_rejected(self) -> None:
        injected = self.current_config().replace(
            "    when: << pipeline.parameters.run_redundant_pr_ci >>",
            "    when:\n      or:\n        - equal: [main, << pipeline.git.branch >>]\n        - << pipeline.parameters.run_redundant_pr_ci >>",
        )
        with self.assertRaises(AssertionError):
            self.assert_budgeted_config(injected)

    def test_second_unconditional_workflow_is_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            self.assert_budgeted_config(self.current_config() + "\n  duplicate:\n    jobs:\n      - cpu_ci\n")



if __name__ == "__main__":
    unittest.main()
