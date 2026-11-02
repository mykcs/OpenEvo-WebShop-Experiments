from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from scripts.ci import workers_builds_trusted_entry_20260928 as entry

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    ".python-version",
    "requirements-ci.txt",
    ".circleci/config.yml",
    ".github/workflows/base-owned-final-gate-202609081959.yml",
    "scripts/ci/base_owned_final_gate_202609081959.py",
    "scripts/ci/cloudflare-cpu-ci.sh",
    "scripts/ci/cloudflare_trusted_bootstrap_202609081711.py",
    "scripts/ci/cloudflare_immutable_executor_202609081840.py",
    "scripts/ci/workers_builds_trusted_entry_20260928.py",
    "scripts/ci/workers_builds_immutable_executor_20260928.py",
    "scripts/ci/workers_builds_provider_contract_20260928.json",
)


def run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
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
    run_git(
        root,
        "remote",
        "add",
        "origin",
        "https://github.com/mykcs/openevo-experiment.git",
    )
    run_git(root, "add", ".")
    run_git(root, "commit", "-m", "base")
    main_sha = run_git(root, "rev-parse", "HEAD")

    run_git(root, "checkout", "-b", "candidate")
    (root / "normal.txt").write_text("candidate\n", encoding="utf-8")
    run_git(root, "add", "normal.txt")
    run_git(root, "commit", "-m", "candidate")
    head_sha = run_git(root, "rev-parse", "HEAD")
    return root, main_sha, head_sha


def set_workers_identity(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    head_sha: str,
    branch: str = "ci-final",
) -> None:
    main_sha = run_git(root, "rev-parse", "main")
    actual_head = run_git(root, "rev-parse", "HEAD")
    tree = run_git(root, "rev-parse", "HEAD^{tree}")
    merge_sha = run_git(root, "commit-tree", tree, "-p", main_sha, "-p", actual_head, "-m", "synthetic merge")
    run_git(root, "update-ref", "refs/remotes/origin/main", main_sha)
    run_git(root, "update-ref", "refs/remotes/origin/ci-final-pr-123-head", actual_head)
    run_git(root, "update-ref", "refs/remotes/origin/ci-final-pr-123-merge", merge_sha)
    monkeypatch.setattr(entry, "provider_git_token", lambda _root: "test-only")
    monkeypatch.setattr(entry, "discover_live_pr_number", lambda _root, _head: "123")
    monkeypatch.setattr(entry, "fetch_live_pr_refs", lambda _root, _number, token=None: None)
    monkeypatch.setenv("WORKERS_CI", "1")
    monkeypatch.setenv("WORKERS_CI_BRANCH", branch)
    monkeypatch.setenv("WORKERS_CI_COMMIT_SHA", head_sha)
    monkeypatch.setenv("WORKERS_CI_BUILD_UUID", "build-test-uuid")


def fake_executor(root: Path, trusted_main: str, mode: str, _work: Path) -> dict[str, object]:
    return {
        "status": "PASS",
        "candidate_environment": "env-i allowlist only",
        "candidate": run_git(root, "rev-parse", "HEAD"),
        "trusted_main": trusted_main,
        "mode": mode,
    }


def test_attest_accepts_fresh_exact_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, main_sha, head_sha = make_repo(tmp)
        set_workers_identity(monkeypatch, root, head_sha)
        monkeypatch.setattr(entry, "run_executor", fake_executor)

        receipt = entry.attest(root, "main", main_sha, "runtime-probe")

        assert receipt["status"] == "PASS"
        assert receipt["provider"] == "Cloudflare Workers Builds"
        assert receipt["head"] == head_sha
        assert receipt["current_main"] == main_sha
        assert receipt["qualification_pr"] == 123
        assert receipt["candidate_code_executed"] is False


def test_attest_rejects_provider_head_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, main_sha, _ = make_repo(tmp)
        set_workers_identity(monkeypatch, root, "0" * 40)
        monkeypatch.setattr(entry, "run_executor", fake_executor)

        with pytest.raises(RuntimeError, match="provider head mismatch"):
            entry.attest(root, "main", main_sha, "runtime-probe")


def test_attest_rejects_invalid_final_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, main_sha, head_sha = make_repo(tmp)
        set_workers_identity(monkeypatch, root, head_sha, "research/unbound")
        monkeypatch.setattr(entry, "run_executor", fake_executor)

        with pytest.raises(RuntimeError, match="authorized final-candidate branch"):
            entry.attest(root, "main", main_sha, "runtime-probe")


def test_attest_rejects_stable_trigger_without_matching_live_pr_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, main_sha, head_sha = make_repo(tmp)
        set_workers_identity(monkeypatch, root, head_sha)
        run_git(root, "update-ref", "refs/remotes/origin/ci-final-pr-123-head", main_sha)
        monkeypatch.setattr(entry, "run_executor", fake_executor)

        with pytest.raises(RuntimeError, match="live PR #123 head mismatch"):
            entry.attest(root, "main", main_sha, "runtime-probe")


def test_pr_discovery_requires_one_open_mergeable_head() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, main_sha, head_sha = make_repo(str(Path(tmp) / "candidate"))
        remote = Path(tmp) / "remote.git"
        run_git(root, "init", "--bare", str(remote))
        run_git(root, "remote", "set-url", "origin", str(remote))
        run_git(root, "push", "origin", "HEAD:refs/pull/123/head")

        with pytest.raises(RuntimeError, match="open mergeable GitHub PR: found 0"):
            entry.discover_live_pr_number(root, head_sha)
        tree = run_git(root, "rev-parse", "HEAD^{tree}")
        merge_sha = run_git(root, "commit-tree", tree, "-p", main_sha, "-p", head_sha, "-m", "merge")
        run_git(root, "push", "origin", f"{merge_sha}:refs/pull/123/merge")
        assert entry.discover_live_pr_number(root, head_sha) == "123"
        with pytest.raises(RuntimeError, match="open mergeable GitHub PR: found 0"):
            entry.discover_live_pr_number(root, main_sha)

        run_git(root, "push", "origin", "HEAD:refs/pull/124/head")
        assert entry.discover_live_pr_number(root, head_sha) == "123"
        run_git(root, "push", "origin", f"{merge_sha}:refs/pull/124/merge")
        with pytest.raises(RuntimeError, match="open mergeable GitHub PR: found 2"):
            entry.discover_live_pr_number(root, head_sha)


def test_provider_git_credential_is_only_used_by_trusted_entry() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, _, _ = make_repo(tmp)
        run_git(
            root,
            "remote", "set-url", "origin",
            "https://x-access-token:test-only@github.com/mykcs/openevo-experiment.git",
        )
        assert entry.provider_git_token(root) == "test-only"


def test_authenticated_refresh_keeps_credential_out_of_argv_and_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, _, _ = make_repo(tmp)
        captured: dict[str, object] = {}
        monkeypatch.setenv("GIT_TRACE_CURL", str(root / "git-trace.log"))

        def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            captured["args"] = args
            captured["env"] = kwargs.get("env")
            return subprocess.CompletedProcess(args, 1, "", "secret-in-git-stderr")

        monkeypatch.setattr(entry.subprocess, "run", fake_run)
        with pytest.raises(RuntimeError, match="cannot refresh current main") as error:
            entry.fetch_live_pr_refs(root, "123", token="test-only")
        assert "test-only" not in str(captured["args"])
        assert "secret-in-git-stderr" not in str(error.value)
        environment = captured["env"]
        assert isinstance(environment, dict)
        assert environment["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraheader"
        assert environment["GIT_CONFIG_VALUE_0"].startswith("AUTHORIZATION: basic ")
        assert "GIT_TRACE_CURL" not in environment


def test_attest_rejects_main_moving_during_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, main_sha, head_sha = make_repo(tmp)
        set_workers_identity(monkeypatch, root, head_sha)
        def moving_executor(candidate: Path, trusted_main: str, mode: str, work: Path) -> dict[str, object]:
            result = fake_executor(candidate, trusted_main, mode, work)
            run_git(root, "update-ref", "refs/remotes/origin/main", head_sha)
            return result
        monkeypatch.setattr(entry, "run_executor", moving_executor)
        with pytest.raises(RuntimeError, match="current main moved"):
            entry.attest(root, "main", main_sha, "runtime-probe")


def test_attest_rejects_head_moving_during_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, main_sha, head_sha = make_repo(tmp)
        set_workers_identity(monkeypatch, root, head_sha)

        def moving_executor(candidate: Path, trusted_main: str, mode: str, work: Path) -> dict[str, object]:
            result = fake_executor(candidate, trusted_main, mode, work)
            run_git(root, "update-ref", "refs/remotes/origin/ci-final-pr-123-head", main_sha)
            return result

        monkeypatch.setattr(entry, "run_executor", moving_executor)
        with pytest.raises(RuntimeError, match="live PR #123 head mismatch"):
            entry.attest(root, "main", main_sha, "runtime-probe")


def test_attest_rejects_merge_ref_disappearing_during_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, main_sha, head_sha = make_repo(tmp)
        set_workers_identity(monkeypatch, root, head_sha)

        def closing_executor(candidate: Path, trusted_main: str, mode: str, work: Path) -> dict[str, object]:
            result = fake_executor(candidate, trusted_main, mode, work)
            run_git(root, "update-ref", "-d", "refs/remotes/origin/ci-final-pr-123-merge")
            return result

        monkeypatch.setattr(entry, "run_executor", closing_executor)
        with pytest.raises(RuntimeError, match="ci-final-pr-123-merge"):
            entry.attest(root, "main", main_sha, "runtime-probe")


def test_attest_rejects_merge_ref_with_wrong_base(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, main_sha, head_sha = make_repo(tmp)
        set_workers_identity(monkeypatch, root, head_sha)
        tree = run_git(root, "rev-parse", "HEAD^{tree}")
        wrong_merge = run_git(root, "commit-tree", tree, "-p", head_sha, "-p", head_sha, "-m", "wrong base")
        run_git(root, "update-ref", "refs/remotes/origin/ci-final-pr-123-merge", wrong_merge)
        monkeypatch.setattr(entry, "run_executor", fake_executor)
        with pytest.raises(RuntimeError, match="does not bind current main and exact head"):
            entry.attest(root, "main", main_sha, "runtime-probe")


def test_attest_rejects_successor_self_modification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, main_sha, _ = make_repo(tmp)
        target = root / "scripts/ci/workers_builds_trusted_entry_20260928.py"
        target.write_text("raise SystemExit(0)\n", encoding="utf-8")
        run_git(root, "add", str(target.relative_to(root)))
        run_git(root, "commit", "-m", "attack trusted entry")
        head_sha = run_git(root, "rev-parse", "HEAD")
        set_workers_identity(monkeypatch, root, head_sha)
        monkeypatch.setattr(entry, "run_executor", fake_executor)

        with pytest.raises(RuntimeError, match="trusted/base-owned path drift"):
            entry.attest(root, "main", main_sha, "runtime-probe")


def test_attest_rejects_successor_executor_modification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, main_sha, _ = make_repo(tmp)
        target = root / "scripts/ci/workers_builds_immutable_executor_20260928.py"
        target.write_text("raise SystemExit(0)\n", encoding="utf-8")
        run_git(root, "add", str(target.relative_to(root)))
        run_git(root, "commit", "-m", "attack executor")
        head_sha = run_git(root, "rev-parse", "HEAD")
        set_workers_identity(monkeypatch, root, head_sha)
        monkeypatch.setattr(entry, "run_executor", fake_executor)

        with pytest.raises(RuntimeError, match="trusted/base-owned path drift"):
            entry.attest(root, "main", main_sha, "runtime-probe")


def test_attest_rejects_locked_dependency_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, main_sha, _ = make_repo(tmp)
        target = root / "requirements-ci.txt"
        target.write_text(
            target.read_text(encoding="utf-8") + "evil-package==1.0\n",
            encoding="utf-8",
        )
        run_git(root, "add", "requirements-ci.txt")
        run_git(root, "commit", "-m", "attack requirements")
        head_sha = run_git(root, "rev-parse", "HEAD")
        set_workers_identity(monkeypatch, root, head_sha)
        monkeypatch.setattr(entry, "run_executor", fake_executor)

        with pytest.raises(RuntimeError, match="trusted/base-owned path drift"):
            entry.attest(root, "main", main_sha, "runtime-probe")


def test_attest_rejects_provider_contract_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, main_sha, _ = make_repo(tmp)
        target = root / "scripts/ci/workers_builds_provider_contract_20260928.json"
        target.write_text('{"build_command":"true"}\n', encoding="utf-8")
        run_git(root, "add", str(target.relative_to(root)))
        run_git(root, "commit", "-m", "attack provider contract")
        head_sha = run_git(root, "rev-parse", "HEAD")
        set_workers_identity(monkeypatch, root, head_sha)
        monkeypatch.setattr(entry, "run_executor", fake_executor)

        with pytest.raises(RuntimeError, match="trusted/base-owned path drift"):
            entry.attest(root, "main", main_sha, "runtime-probe")


def test_attest_rejects_candidate_behind_current_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, _, head_sha = make_repo(tmp)
        run_git(root, "checkout", "main")
        (root / "main-only.txt").write_text("advance\n", encoding="utf-8")
        run_git(root, "add", "main-only.txt")
        run_git(root, "commit", "-m", "advance main")
        current_main = run_git(root, "rev-parse", "HEAD")
        run_git(root, "checkout", "candidate")
        set_workers_identity(monkeypatch, root, head_sha)
        monkeypatch.setattr(entry, "run_executor", fake_executor)

        with pytest.raises(RuntimeError, match="behind or detached"):
            entry.attest(root, "main", current_main, "runtime-probe")


def test_attest_requires_workers_builds_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, main_sha, _ = make_repo(tmp)
        monkeypatch.delenv("WORKERS_CI", raising=False)
        monkeypatch.delenv("WORKERS_CI_BRANCH", raising=False)
        monkeypatch.delenv("WORKERS_CI_COMMIT_SHA", raising=False)
        monkeypatch.delenv("WORKERS_CI_BUILD_UUID", raising=False)
        monkeypatch.setattr(entry, "run_executor", fake_executor)

        with pytest.raises(RuntimeError, match="WORKERS_CI=1"):
            entry.attest(root, "main", main_sha, "runtime-probe")


def test_git_checkout_credentials_are_removed_before_candidate_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, main_sha, head_sha = make_repo(tmp)
        run_git(
            root,
            "config",
            "--local",
            "http.https://github.com/.extraheader",
            "AUTHORIZATION: basic redacted",
        )
        set_workers_identity(monkeypatch, root, head_sha)
        monkeypatch.setattr(entry, "run_executor", fake_executor)

        receipt = entry.attest(root, "main", main_sha, "runtime-probe")

        assert receipt["status"] == "PASS"
        remaining = subprocess.run(
            [
                "git",
                "config",
                "--local",
                "--get-regexp",
                r"http\..*extraheader",
            ],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        assert remaining.stdout.strip() == ""


def test_embedded_provider_origin_credentials_are_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, main_sha, head_sha = make_repo(tmp)
        run_git(
            root,
            "remote",
            "set-url",
            "origin",
            "https://x-access-token:placeholder@example.invalid/mykcs/openevo-experiment.git",
        )
        set_workers_identity(monkeypatch, root, head_sha)
        monkeypatch.setattr(entry, "run_executor", fake_executor)

        with pytest.raises(RuntimeError, match="expected GitHub repository"):
            entry.attest(root, "main", main_sha, "runtime-probe")

        run_git(
            root,
            "remote",
            "set-url",
            "origin",
            "https://x-access-token:placeholder@github.com/mykcs/openevo-experiment.git",
        )
        run_git(
            root,
            "config",
            "--local",
            "remote.origin.pushurl",
            "https://x-access-token:placeholder@github.com/mykcs/openevo-experiment.git",
        )
        run_git(
            root,
            "config",
            "--local",
            "remote.other.url",
            "https://x-access-token:placeholder@github.com/mykcs/other.git",
        )
        (root / ".git/logs/HEAD").write_text(
            "clone from https://x-access-token:placeholder@github.com/mykcs/openevo-experiment.git\n",
            encoding="utf-8",
        )
        (root / ".git/FETCH_HEAD").write_text(
            "fetch from https://x-access-token:placeholder@github.com\n",
            encoding="utf-8",
        )
        fetch_tokens: list[str | None] = []

        def observe_fetch(_root: Path, _number: str, token: str | None = None) -> None:
            fetch_tokens.append(token)
            config = (root / ".git/config").read_text(encoding="utf-8")
            assert ("placeholder" in config) == (token is None)

        def observe_executor(candidate: Path, trusted_main: str, mode: str, work: Path) -> dict[str, object]:
            assert fetch_tokens == [None]
            assert "placeholder" not in (root / ".git/config").read_text(encoding="utf-8")
            assert (root / ".git/logs/HEAD").read_bytes() == b""
            assert (root / ".git/FETCH_HEAD").read_bytes() == b""
            return fake_executor(candidate, trusted_main, mode, work)

        monkeypatch.setattr(entry, "fetch_live_pr_refs", observe_fetch)
        monkeypatch.setattr(entry, "run_executor", observe_executor)
        receipt = entry.attest(root, "main", main_sha, "runtime-probe")

        assert receipt["status"] == "PASS"
        assert fetch_tokens == [None, "test-only"]
        assert run_git(root, "remote", "get-url", "origin") == (
            "https://github.com/mykcs/openevo-experiment.git"
        )
        assert "placeholder" not in (root / ".git/config").read_text(encoding="utf-8")
        assert (root / ".git/logs/HEAD").read_bytes() == b""
        assert (root / ".git/FETCH_HEAD").read_bytes() == b""
        assert run_git(root, "remote") == "origin"


def test_executor_accepts_preparation_output_before_trusted_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, main_sha, head_sha = make_repo(str(Path(tmp) / "repo"))
        work = Path(tmp) / "work"
        work.mkdir()
        expected = {
            "status": "PASS",
            "candidate": head_sha,
            "trusted_main": main_sha,
            "mode": "runtime-probe",
        }
        source = (
            "import json\n"
            "print('Collecting pinned CI dependencies')\n"
            "print('WORKERS_BUILDS_IMMUTABLE_EXECUTOR=PASS')\n"
            f"print({json.dumps(expected)!r})\n"
        )
        monkeypatch.setattr(entry, "trusted_text", lambda *_args: source)

        assert entry.run_executor(root, main_sha, "runtime-probe", work) == expected
