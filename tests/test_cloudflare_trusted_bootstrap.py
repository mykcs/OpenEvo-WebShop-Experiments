from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from scripts.ci import cloudflare_trusted_bootstrap_202609081711 as bootstrap

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    ".python-version",
    "requirements-ci.txt",
    ".circleci/config.yml",
    ".github/workflows/base-owned-integration-gate-202609060057.yml",
    "scripts/ci/base_owned_integration_gate_202609060057.py",
    "scripts/ci/cloudflare-cpu-ci.sh",
    "scripts/ci/cloudflare_trusted_bootstrap_202609081711.py",
    "scripts/ci/cloudflare_immutable_executor_202609081840.py",
)


def run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def make_repo(tmp: str) -> tuple[Path, str, str]:
    root = Path(tmp)
    for relative in REQUIRED_FILES:
        source = ROOT / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (root / "normal.txt").write_text("base\n", encoding="utf-8")

    run_git(root, "init", "-b", "main")
    run_git(root, "config", "user.name", "CI Test")
    run_git(root, "config", "user.email", "ci@example.invalid")
    run_git(root, "remote", "add", "origin", "https://github.com/mykcs/openevo-experiment.git")
    run_git(root, "add", ".")
    run_git(root, "commit", "-m", "base")
    main_sha = run_git(root, "rev-parse", "HEAD")

    run_git(root, "checkout", "-b", "candidate")
    (root / "normal.txt").write_text("candidate\n", encoding="utf-8")
    run_git(root, "add", "normal.txt")
    run_git(root, "commit", "-m", "candidate")
    head_sha = run_git(root, "rev-parse", "HEAD")
    return root, main_sha, head_sha


def set_cloudflare_identity(monkeypatch: pytest.MonkeyPatch, head_sha: str) -> None:
    monkeypatch.setenv("CF_PAGES", "1")
    monkeypatch.setenv("CF_PAGES_BRANCH", "candidate")
    monkeypatch.setenv("CF_PAGES_COMMIT_SHA", head_sha)
    for name in ("CLOUDFLARE_API_TOKEN", "CLOUDFLARE_API_KEY", "CF_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def test_trusted_bootstrap_accepts_fresh_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, main_sha, head_sha = make_repo(tmp)
        set_cloudflare_identity(monkeypatch, head_sha)
        receipt = bootstrap.attest(root, "main", main_sha)
        assert receipt["status"] == "PASS"
        assert receipt["head"] == head_sha
        assert receipt["current_main"] == main_sha
        assert receipt["candidate_code_executed"] is False
        assert receipt["candidate_sandbox_authorized"] is False


def test_trusted_bootstrap_rejects_provider_head_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, main_sha, head_sha = make_repo(tmp)
        set_cloudflare_identity(monkeypatch, "0" * 40)
        with pytest.raises(RuntimeError, match="provider head mismatch"):
            bootstrap.attest(root, "main", main_sha)


def test_trusted_bootstrap_rejects_self_modification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, main_sha, _ = make_repo(tmp)
        target = root / "scripts/ci/cloudflare_trusted_bootstrap_202609081711.py"
        target.write_text("raise SystemExit(0)\n", encoding="utf-8")
        run_git(root, "add", str(target.relative_to(root)))
        run_git(root, "commit", "-m", "attack verifier")
        head_sha = run_git(root, "rev-parse", "HEAD")
        set_cloudflare_identity(monkeypatch, head_sha)
        with pytest.raises(RuntimeError, match="trusted/base-owned path drift"):
            bootstrap.attest(root, "main", main_sha)


def test_trusted_bootstrap_rejects_locked_dependency_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, main_sha, _ = make_repo(tmp)
        target = root / "requirements-ci.txt"
        target.write_text(target.read_text(encoding="utf-8") + "evil-package==1.0\n", encoding="utf-8")
        run_git(root, "add", "requirements-ci.txt")
        run_git(root, "commit", "-m", "attack requirements")
        head_sha = run_git(root, "rev-parse", "HEAD")
        set_cloudflare_identity(monkeypatch, head_sha)
        with pytest.raises(RuntimeError, match="trusted/base-owned path drift"):
            bootstrap.attest(root, "main", main_sha)


def test_trusted_bootstrap_rejects_candidate_behind_current_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, _, head_sha = make_repo(tmp)
        run_git(root, "checkout", "main")
        (root / "main-only.txt").write_text("new main\n", encoding="utf-8")
        run_git(root, "add", "main-only.txt")
        run_git(root, "commit", "-m", "advance main")
        current_main = run_git(root, "rev-parse", "HEAD")
        run_git(root, "checkout", "candidate")
        set_cloudflare_identity(monkeypatch, head_sha)
        with pytest.raises(RuntimeError, match="behind or detached"):
            bootstrap.attest(root, "main", current_main)


def test_trusted_bootstrap_rejects_cloudflare_deploy_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, main_sha, head_sha = make_repo(tmp)
        set_cloudflare_identity(monkeypatch, head_sha)
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "red-team-probe")
        with pytest.raises(RuntimeError, match="deploy credential exposed"):
            bootstrap.attest(root, "main", main_sha)


def test_trusted_bootstrap_rejects_immutable_executor_modification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, main_sha, _ = make_repo(tmp)
        target = root / "scripts/ci/cloudflare_immutable_executor_202609081840.py"
        target.write_text("raise SystemExit(0)\n", encoding="utf-8")
        run_git(root, "add", str(target.relative_to(root)))
        run_git(root, "commit", "-m", "attack immutable executor")
        head_sha = run_git(root, "rev-parse", "HEAD")
        set_cloudflare_identity(monkeypatch, head_sha)
        with pytest.raises(RuntimeError, match="trusted/base-owned path drift"):
            bootstrap.attest(root, "main", main_sha)
