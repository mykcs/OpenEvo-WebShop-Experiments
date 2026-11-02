from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.ci import workers_builds_immutable_executor_20260928 as executor


class FakePredecessor:
    @staticmethod
    def inner_script(mode: str) -> str:
        assert mode in {"runtime-probe", "full"}
        return (
            "#!/bin/bash\n"
            "set -euo pipefail\n"
            "echo IMMUTABLE_RUNTIME_ISOLATION=PASS\n"
        )


def test_enter_script_uses_empty_environment_allowlist() -> None:
    text = executor.enter_script()
    for fragment in (
        "/usr/bin/env -i",
        "PATH=",
        "HOME=/tmp/home",
        "CI=true",
        "PIP_NO_INDEX=1",
        "/usr/bin/setpriv",
        "--no-new-privs",
        "--bounding-set=-all",
    ):
        assert fragment in text


def test_inner_script_rejects_sensitive_environment_names() -> None:
    text = executor.inner_script(FakePredecessor(), "runtime-probe")
    assert "python3 /wheelhouse/credential_probe.py" in text
    assert text.index("python3 /wheelhouse/credential_probe.py") < text.index(
        "IMMUTABLE_RUNTIME_ISOLATION=PASS"
    )
    source = executor.credential_probe_source()
    for fragment in (
        "candidate_env_sensitive_names",
        "TOKEN",
        "SECRET",
        "CLOUDFLARE",
        "WORKERS_CI",
        "cmdline",
        "environ",
        "fd",
        "git_metadata",
        "credential_store",
        "CANDIDATE_PROC_ENV=PASS",
        "CANDIDATE_CREDENTIAL_SURFACES=PASS",
    ):
        assert fragment in source


def probe_function():
    namespace = {"__name__": "credential_probe_test"}
    exec(executor.credential_probe_source(), namespace)
    return namespace["probe"]


def test_credential_probe_checks_proc_and_git_without_reporting_values() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        proc = root / "proc"
        process = proc / "1"
        (process / "fd").mkdir(parents=True)
        (process / "environ").write_bytes(b"PATH=/usr/bin\0CLOUDFLARE_API_TOKEN=private-value\0")
        (process / "cmdline").write_bytes(b"python3\0--api-key=private-value\0")
        (process / "fd" / "7").symlink_to("/tmp/provider-secret")
        workspace = root / "workspace"
        (workspace / ".git").mkdir(parents=True)
        (workspace / ".git/config").write_text(
            "url = https://x-access-token:private-value@github.com/mykcs/openevo-experiment.git\n",
            encoding="utf-8",
        )
        home = root / "home"
        home.mkdir()
        (home / ".netrc").write_text("machine github.com password private-value\n", encoding="utf-8")

        findings = probe_function()(proc, workspace, home, {b"PATH": b"/usr/bin"})

        assert findings == [
            "credential_store",
            "git_metadata",
            "proc_cmdline",
            "proc_environ",
            "proc_fd",
        ]
        assert "private-value" not in repr(findings)


def test_credential_probe_accepts_clean_candidate_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        proc = root / "proc"
        process = proc / "1"
        (process / "fd").mkdir(parents=True)
        (process / "environ").write_bytes(b"PATH=/usr/bin\0HOME=/tmp/home\0")
        (process / "cmdline").write_bytes(b"/bin/bash\0/wheelhouse/inner.sh\0")
        workspace = root / "workspace"
        (workspace / ".git").mkdir(parents=True)
        (workspace / ".git/config").write_text(
            "[remote \"origin\"]\nurl = https://github.com/mykcs/openevo-experiment.git\n",
            encoding="utf-8",
        )
        home = root / "home"
        home.mkdir()

        original_exists = Path.exists

        def exists_with_unreadable_root(path: Path) -> bool:
            if str(path).startswith("/workspace/"):
                raise PermissionError("unreadable root home")
            return original_exists(path)

        monkeypatch.setattr(Path, "exists", exists_with_unreadable_root)

        assert probe_function()(proc, workspace, home, {b"PATH": b"/usr/bin"}) == []


def test_full_gate_counts_are_structured_and_bounded() -> None:
    output = "1398 passed, 1 skipped in 65.23s (0:01:05)\ncontract checks passed: 5\n"
    assert executor.full_gate_counts(output) == {
        "pytest": {"passed": 1398, "skipped": 1, "warnings": 0},
        "installed_contract_checks": 5,
    }
    with pytest.raises(RuntimeError, match="pytest counts violate"):
        executor.full_gate_counts(output.replace("1 skipped", "2 skipped"))
    with pytest.raises(RuntimeError, match="contract count"):
        executor.full_gate_counts(output.replace("contract checks passed: 5", "contract checks passed: 4"))
    with pytest.raises(RuntimeError, match="summary is missing or ambiguous"):
        executor.full_gate_counts(output + "1398 passed, 1 skipped in 65.23s (0:01:05)\n")


def test_run_isolated_does_not_inherit_os_environment() -> None:
    source = inspect.getsource(executor.run_isolated)
    assert "os.environ.copy()" not in source
    assert '"PATH": "/usr/bin:/bin"' in source
    assert '"ROOTFS": str(rootfs)' in source
    assert '"CANDIDATE": str(candidate)' in source
    assert '"WHEELHOUSE": str(wheelhouse)' in source


def test_successor_loads_predecessor_from_trusted_main() -> None:
    source = inspect.getsource(executor.load_predecessor)
    assert "trusted_git_text" in source
    assert "PREDECESSOR_PATH" in source
    assert "candidate / PREDECESSOR_PATH" not in source


def test_isolated_output_is_captured_before_trusted_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        rootfs = work / "rootfs"
        (rootfs / "usr/bin").mkdir(parents=True)
        (rootfs / "usr/bin/env").touch()
        wheelhouse = work / "wheelhouse"
        wheelhouse.mkdir()
        seen: dict[str, object] = {}

        def fake_run(*_args, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(stdout="probe output\nCANDIDATE_PROC_ENV=PASS\nCANDIDATE_CREDENTIAL_SURFACES=PASS\nIMMUTABLE_RUNTIME_ISOLATION=PASS\n")

        monkeypatch.setattr(executor.subprocess, "run", fake_run)
        receipt = executor.run_isolated(
            FakePredecessor(), rootfs, work, wheelhouse, work, "runtime-probe"
        )

        assert seen["capture_output"] is True
        assert seen["text"] is True
        assert receipt["isolation"] == "PASS"
        assert receipt["proc_environment"] == "PASS"
        assert receipt["credential_surfaces"] == "PASS"
        assert receipt["full_gate"] == "NOT_RUN"


def test_isolated_output_without_marker_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        rootfs = work / "rootfs"
        (rootfs / "usr/bin").mkdir(parents=True)
        (rootfs / "usr/bin/env").touch()
        wheelhouse = work / "wheelhouse"
        wheelhouse.mkdir()
        monkeypatch.setattr(
            executor.subprocess,
            "run",
            lambda *_args, **_kwargs: SimpleNamespace(stdout="probe output\n"),
        )

        with pytest.raises(RuntimeError, match="isolated runtime PASS marker missing"):
            executor.run_isolated(
                FakePredecessor(), rootfs, work, wheelhouse, work, "runtime-probe"
            )


def test_full_isolated_receipt_reports_counts_without_candidate_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        rootfs = work / "rootfs"
        (rootfs / "usr/bin").mkdir(parents=True)
        (rootfs / "usr/bin/env").touch()
        wheelhouse = work / "wheelhouse"
        wheelhouse.mkdir()
        output = (
            "candidate diagnostic content\n"
            "CANDIDATE_PROC_ENV=PASS\n"
            "CANDIDATE_CREDENTIAL_SURFACES=PASS\n"
            "IMMUTABLE_RUNTIME_ISOLATION=PASS\n"
            "1398 passed, 1 skipped in 65.23s (0:01:05)\n"
            "contract checks passed: 5\n"
            "IMMUTABLE_RUNTIME_FULL_GATE=PASS\n"
        )
        monkeypatch.setattr(
            executor.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(stdout=output)
        )

        receipt = executor.run_isolated(
            FakePredecessor(), rootfs, work, wheelhouse, work, "full"
        )

        assert receipt["pytest"] == {"passed": 1398, "skipped": 1, "warnings": 0}
        assert receipt["installed_contract_checks"] == 5
        assert receipt["credential_surfaces"] == "PASS"
        assert "candidate diagnostic content" not in repr(receipt)
