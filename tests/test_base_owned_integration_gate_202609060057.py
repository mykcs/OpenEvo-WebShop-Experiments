from __future__ import annotations

from pathlib import Path

from scripts.ci import base_owned_integration_gate_202609060057 as gate

ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github/workflows/base-owned-integration-gate-202609060057.yml"


def test_static_contract_matches_current_circleci() -> None:
    assert gate.validate_circleci_config((ROOT / ".circleci/config.yml").read_text()) == []
    assert gate.validate_cpu_gate((ROOT / "scripts/ci/cloudflare-cpu-ci.sh").read_text()) == []


def test_trusted_execution_substrate_is_base_locked() -> None:
    assert gate.BASE_LOCKED == {".python-version", "requirements-ci.txt"}
    assert "base-owned-integration-gate-202609060057.yml" in " ".join(sorted(gate.SELF_PROTECTED))
    assert "base_owned_integration_gate_202609060057.py" in " ".join(sorted(gate.SELF_PROTECTED))


def test_workflow_uses_distinct_app_bound_context_and_no_job_container() -> None:
    text = WORKFLOW.read_text()
    assert "name: base-owned-integration-gate-202609060057" in text
    assert "pull_request_target:" in text
    assert "types: [labeled]" in text
    assert "github.event.label.name == 'ci-legacy-gate'" in text
    assert "opened, synchronize" not in text
    assert "ready_for_review" not in text
    assert "runs-on: ubuntu-24.04" in text
    assert "container:" not in text
    assert "statuses: read" not in text
    assert "self-hosted" not in text


def test_candidate_runs_only_in_nested_locked_sandbox() -> None:
    text = WORKFLOW.read_text()
    assert "--network none" in text
    assert "--tmpfs /tmp:rw,exec,nosuid,nodev,size=3g,uid=1001,gid=1002" in text
    assert "--tmpfs /tmp:rw,noexec" not in text
    assert "--read-only" in text
    assert "--cap-drop ALL" in text
    assert "--security-opt no-new-privileges:true" in text
    assert "--pids-limit 1024" in text
    assert "--cpus 2" in text
    assert "--memory 4g" in text
    assert "/var/run/docker.sock" in text
    assert "type=bind,src=\"$GITHUB_WORKSPACE/candidate\",dst=/workspace" in text
    assert "type=bind,src=\"$RUNNER_TEMP/wheelhouse\",dst=/wheelhouse,readonly" in text


def test_candidate_gate_is_offline_and_fixed_by_trusted_workflow() -> None:
    text = WORKFLOW.read_text()
    assert "PIP_NO_INDEX=1" in text
    assert "PIP_FIND_LINKS=/wheelhouse" in text
    assert "python3 scripts/ci/preflight.py" in text
    assert "python3 -m pytest -q -n 2 tests" in text
    assert "python3 -m build --no-isolation --wheel" in text
    assert "contract-test" in text
    assert "cloudflare-cpu-ci.sh --mode full" not in text
