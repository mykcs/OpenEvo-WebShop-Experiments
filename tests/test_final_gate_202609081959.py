from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/base-owned-final-gate-202609081959.yml"
VERIFIER = ROOT / "scripts/ci/base_owned_final_gate_202609081959.py"


def test_final_gate_is_label_only_and_trusted_base_owned() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "pull_request_target:" in text
    assert "types: [labeled]" in text
    assert "opened, synchronize" not in text
    assert "github.event.label.name == 'ci-final-gate'" in text
    assert "github.event.pull_request.draft == false" in text
    assert "runs-on: ubuntu-24.04" in text
    assert "persist-credentials: false" in text
    assert "--network none" in text
    assert "--read-only" in text
    assert "--cap-drop ALL" in text
    assert "--security-opt no-new-privileges:true" in text
    assert "PIP_NO_INDEX=1" in text
    assert "python3 -m pytest -q -n 2 tests" in text
    assert "python3 scripts/ci/base_owned_final_gate_202609081959.py" in text


def test_final_gate_verifier_self_protects_successor_only() -> None:
    text = VERIFIER.read_text(encoding="utf-8")
    assert '".github/workflows/base-owned-final-gate-202609081959.yml"' in text
    assert '"scripts/ci/base_owned_final_gate_202609081959.py"' in text
    assert '".github/workflows/base-owned-integration-gate-202609060057.yml"' not in text
    assert 'event.get("action") != "labeled"' in text
    assert '!= "ci-final-gate"' in text
