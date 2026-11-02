from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/"scripts/ci/buildkite_final_provider_contract_20261006.json"
GATE=ROOT/"scripts/ci/buildkite_final_gate_20261006.py"
RENDER=ROOT/"scripts/ci/render_buildkite_final_pipeline_20261006.py"

def load(path: Path, name: str):
    spec=importlib.util.spec_from_file_location(name,path)
    assert spec and spec.loader
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def test_contract_pins_gate_and_boundaries():
    c=json.loads(CONTRACT.read_text())
    assert c["integration_branch"]=="main"
    assert c["final_alias_branch"]=="ci-final-buildkite"
    assert c["queue"]=="linux-small"
    assert c["boundaries"]=={
        "gpu":False,
        "scientific_execution":False,
        "release_publication":False,
        "automatic_branch_builds":False,
        "automatic_pr_builds":False,
        "ruleset_mutation_before_qualification":False,
    }
    assert hashlib.sha256(GATE.read_bytes()).hexdigest()==c["gate_script"]["sha256"]

def test_renderer_is_single_linux_final_and_hash_pinned():
    c=json.loads(CONTRACT.read_text())
    text=load(RENDER,"render_bk_exp").render()
    assert "queue: linux-small" in text
    assert "queue: linux-medium" not in text
    assert c["gate_script"]["sha256"] in text
    assert c["gate_script"]["path"] in text
    for forbidden in ("macos-","nvidia","webshop formal","wandb","cuda_visible_devices"):
        assert forbidden not in text.lower()

def test_gate_preserves_no_network_no_socket_no_gpu_contract():
    text=GATE.read_text().lower()
    for required in (
        "--network",
        "none",
        "/dev/nvidia0",
        "/var/run/docker.sock",
        "--read-only",
        "--cap-drop",
        "no-new-privileges:true",
        "python3 scripts/ci/preflight.py",
        "python3 -m pytest -q -n 2 tests",
        "python3 -m build",
    ):
        assert required in text
    for forbidden in ("nvidia-smi", "wandb login", 'run(["ssh"', 'run(["scp"'):
        assert forbidden not in text
