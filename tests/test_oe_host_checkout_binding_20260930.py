from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts import oe_impl


def test_host_rejects_a_contract_for_another_checkout_before_delegation(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    another = tmp_path / "another"
    checkout.mkdir()
    another.mkdir()
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"paths": {"experiment": {
        "host_path": str(another),
        "container_path": "/workspace/src/openevo-experiment",
        "expected_type": "dir", "access": "ro",
    }}}))
    marker = tmp_path / "delegated"
    result = oe_impl.main([
        "host", "--path-contract", str(contract), "--cwd", str(checkout),
        "--", "/bin/sh", "-c", f"touch {marker}",
    ])
    assert result == 2
    assert not marker.exists()


def test_host_accepts_the_exact_contract_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"paths": {"experiment": {
        "host_path": str(checkout),
        "container_path": "/workspace/src/openevo-experiment",
        "expected_type": "dir", "access": "ro",
    }}}))
    result = oe_impl.main([
        "host", "--path-contract", str(contract), "--cwd", str(checkout),
        "--", sys.executable, "-c", "pass",
    ])
    assert result == 0


def test_host_reports_a_missing_contract_as_a_cli_error(tmp_path: Path) -> None:
    import contextlib
    import io

    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        result = oe_impl.main([
            "host", "--path-contract", str(tmp_path / "missing.json"),
            "--cwd", str(tmp_path), "--", sys.executable, "-c", "pass",
        ])
    assert result == 2
    assert "oe: host path contract cannot be read:" in stderr.getvalue()
    assert "Traceback" not in stderr.getvalue()


def test_host_reports_invalid_json_as_a_cli_error(tmp_path: Path) -> None:
    import contextlib
    import io

    contract = tmp_path / "contract.json"
    contract.write_text("{bad json")
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        result = oe_impl.main([
            "host", "--path-contract", str(contract),
            "--cwd", str(tmp_path), "--", sys.executable, "-c", "pass",
        ])
    assert result == 2
    assert "oe: host path contract cannot be read:" in stderr.getvalue()
    assert "Traceback" not in stderr.getvalue()


def test_host_reports_missing_checkout_as_a_cli_error(tmp_path: Path) -> None:
    import contextlib
    import io

    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"paths": {"experiment": {
        "host_path": str(tmp_path),
        "container_path": "/workspace/src/openevo-experiment",
        "expected_type": "dir", "access": "ro",
    }}}))
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        result = oe_impl.main([
            "host", "--path-contract", str(contract),
            "--cwd", str(tmp_path / "absent"), "--", sys.executable, "-c", "pass",
        ])
    assert result == 2
    assert "oe: host checkout path cannot be resolved:" in stderr.getvalue()
    assert "Traceback" not in stderr.getvalue()


def test_host_rejects_a_non_object_paths_field(tmp_path: Path) -> None:
    import contextlib
    import io

    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"paths": ["unexpected"]}))
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        result = oe_impl.main([
            "host", "--path-contract", str(contract),
            "--cwd", str(tmp_path), "--", sys.executable, "-c", "pass",
        ])
    assert result == 2
    assert "oe: host path contract must contain named paths" in stderr.getvalue()
    assert "Traceback" not in stderr.getvalue()


def test_host_reports_invalid_utf8_as_a_cli_error(tmp_path: Path) -> None:
    import contextlib
    import io

    contract = tmp_path / "contract.json"
    contract.write_bytes(b"\xff")
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        result = oe_impl.main([
            "host", "--path-contract", str(contract),
            "--cwd", str(tmp_path), "--", sys.executable, "-c", "pass",
        ])
    assert result == 2
    assert "oe: host path contract cannot be read:" in stderr.getvalue()
    assert "Traceback" not in stderr.getvalue()
