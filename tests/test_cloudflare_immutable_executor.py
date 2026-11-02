from __future__ import annotations

import io
import tarfile
import tempfile
from pathlib import Path

import pytest

from scripts.ci import cloudflare_immutable_executor_202609081840 as executor


def add_tar_file(archive: tarfile.TarFile, name: str, data: bytes = b"") -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(data))


def test_runtime_identity_is_exactly_pinned() -> None:
    assert executor.IMAGE_INDEX_DIGEST == "sha256:a00bfa949271c5519c63d623c6c5ab61fa0868f88ffa3a66b0983bf7df376f3d"
    assert executor.AMD64_MANIFEST_DIGEST == "sha256:f4acb365df4fc4210cf0ede2eda93e4b8ae2280356b2f30f1e3fb4c2b6315b57"
    assert executor.EXPECTED_PYTHON == "3.13.3"


def test_safe_layer_path_rejects_escape() -> None:
    with pytest.raises(RuntimeError, match="unsafe OCI layer path"):
        executor.safe_member_path("../../host")
    with pytest.raises(RuntimeError, match="unsafe OCI layer path"):
        executor.safe_member_path("/absolute")


def test_whiteout_removes_lower_file_before_extract() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "root"
        root.mkdir()
        (root / "old.txt").write_text("old", encoding="utf-8")
        layer = Path(tmp) / "layer.tgz"
        with tarfile.open(layer, "w:gz") as archive:
            add_tar_file(archive, ".wh.old.txt")
            add_tar_file(archive, "new.txt", b"new")
        executor.extract_layer(layer, root)
        assert not (root / "old.txt").exists()
        assert (root / "new.txt").read_text(encoding="utf-8") == "new"


def test_opaque_whiteout_clears_lower_directory() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "root"
        directory = root / "etc/example"
        directory.mkdir(parents=True)
        (directory / "lower.txt").write_text("lower", encoding="utf-8")
        layer = Path(tmp) / "layer.tgz"
        with tarfile.open(layer, "w:gz") as archive:
            add_tar_file(archive, "etc/example/.wh..wh..opq")
            add_tar_file(archive, "etc/example/upper.txt", b"upper")
        executor.extract_layer(layer, root)
        assert not (directory / "lower.txt").exists()
        assert (directory / "upper.txt").read_text(encoding="utf-8") == "upper"


def test_inner_full_gate_uses_fixed_commands_and_readonly_probe() -> None:
    text = executor.inner_script("full")
    for fragment in (
        "IMMUTABLE_RUNTIME_ISOLATION=PASS",
        "python3 scripts/ci/preflight.py",
        "python3 -m pytest -q -n 2 tests",
        "python3 -m build --no-isolation --wheel",
        "contract-test",
        "touch /workspace/.cloudflare-write-probe",
        "isolated_connect_ex",
        "NoNewPrivs",
        "CapEff",
    ):
        assert fragment in text
    assert "cloudflare-cpu-ci.sh" not in text


def test_namespace_entry_mounts_candidate_and_wheelhouse_readonly() -> None:
    text = executor.enter_script()
    for fragment in (
        'mount -o remount,bind,ro "$ROOTFS"',
        'mount -o remount,bind,ro "$ROOTFS/workspace"',
        'mount -o remount,bind,ro "$ROOTFS/wheelhouse"',
        "mount -t proc",
        "setpriv",
        "--no-new-privs",
        "--bounding-set=-all",
        'mount --bind "/dev/$name" "$ROOTFS/dev/$name"',
    ):
        assert fragment in text
    assert "mount --rbind /dev" not in text
