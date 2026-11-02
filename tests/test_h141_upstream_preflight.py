import hashlib
import json
import subprocess
from pathlib import Path

from scripts.openevo_webshop.prepare_h141_upstream import OPEN_EVO_SHA, PATCH_SHA, run
from scripts.openevo_webshop.h141_upstream_evaluation import host_adapter_path, live_idle_authorized


def test_h141_patch_digest_matches_contract():
    root = Path(__file__).parents[1]
    patch = root / "scripts/openevo_webshop/patches/openevo-sd-lora-reset-magnitudes.patch"
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == PATCH_SHA


def test_h141_preflight_rejects_dirty_clean_core(tmp_path):
    clean = tmp_path / "clean"
    subprocess.run(["git", "init", "-q", str(clean)], check=True)
    subprocess.run(["git", "-C", str(clean), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(clean), "config", "user.name", "test"], check=True)
    (clean / "README").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(clean), "add", "README"], check=True)
    subprocess.run(["git", "-C", str(clean), "commit", "-qm", "fixture"], check=True)
    dirty = clean / "README"
    dirty.write_text("changed\n", encoding="utf-8")
    try:
        run(Path(__file__).parents[1], clean, tmp_path / "patched", tmp_path / "receipt.json")
    except ValueError as exc:
        assert "pinned" in str(exc) or "dirty" in str(exc)
    else:
        raise AssertionError("dirty pinned checkout must fail closed")


def test_h141_evaluation_translates_container_adapter_uri_to_host_run_root(tmp_path):
    uri = "file:///workspace/runs/phase-h141-magnitude-screen-20260821/training/C0_exact_h140/artifacts/adapter"
    assert host_adapter_path(uri, tmp_path) == tmp_path / "training/C0_exact_h140/artifacts/adapter"


def test_h141_evaluation_remaps_only_to_authorized_live_idle_uuids(monkeypatch, tmp_path):
    class Gate:
        def __init__(self, returncode):
            self.returncode = returncode

    monkeypatch.setattr(
        "scripts.openevo_webshop.h141_upstream_evaluation.run_gate",
        lambda root, uuid, scratch: Gate(0 if uuid in {"u1", "u3"} else 1),
    )
    assert live_idle_authorized(tmp_path, ["u0", "u1", "u2", "u3"], tmp_path / "scratch") == ["u1", "u3"]
