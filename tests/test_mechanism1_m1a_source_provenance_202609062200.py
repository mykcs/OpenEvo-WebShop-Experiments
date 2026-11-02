import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "docs/evidence/server-runs/2026-09-06/mechanism1-m1a-adjacent-source/source-provenance-seal-202609062200.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_provenance_is_hash_closed_and_execution_locked():
    manifest = load(PROVENANCE)
    assert manifest["status"] == "SOURCE_PROVENANCE_SEALED_EXECUTION_LOCKED"
    assert manifest["execution_authority"] is False
    assert manifest["formal_task_consumption_allowed"] is False
    assert manifest["selection_used_reward"] is False

    for binding in manifest["successor_source"].values():
        bound = ROOT / binding["path"]
        assert bound.is_file()
        assert len(binding["git_blob_sha"]) == 40

    seal = manifest["terminal_seal"]
    assert sha256(ROOT / seal["path"]) == seal["sha256"]
    assert seal["status"] == "PASS"
    assert seal["parameter_updates"] == 7

    capsule = manifest["source_capsule"]
    assert capsule["remote_status"] == "REMOTE_VERIFIED_EXACT"
    assert len(capsule["archive_sha256"]) == 64
    assert len(capsule["manifest_sha256"]) == 64

    store = manifest["component_store"]
    assert store["status"] == "REMOTE_VERIFIED_EXACT"
    assert len(store["head"]["sha256"]) == 64
    assert len(store["guard"]["sha256"]) == 64
    selected = store["selected_snapshots"]
    assert list(selected) == ["theta_target_R14", "theta_previous_R27", "theta_final_R49"]
    assert len({row["adapter_sha256"] for row in selected.values()}) == 3

    gate = manifest["activation_boundary"]
    assert gate["probe_manifest_frozen"] is False
    assert gate["runtime_qualification_passed"] is False
    assert gate["activation_release_present"] is False
