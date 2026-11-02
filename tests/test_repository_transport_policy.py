from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_PREFIXES = (ROOT / "scripts", ROOT / ".github")
SELF_HTTPS_URL = "https://github.com/" + "mykcs/openevo-experiment.git"


def _active_text_files() -> list[Path]:
    files: list[Path] = []
    for prefix in ACTIVE_PREFIXES:
        if not prefix.exists():
            continue
        for path in prefix.rglob("*"):
            if path.is_file():
                files.append(path)
    return files


def test_active_launch_code_does_not_fresh_clone_self_over_https() -> None:
    """Server launch paths must reuse checkout/worktree or scoped SSH transport.

    The ZJU server has a repository-scoped SSH credential for this repository.
    A fresh clone over the canonical HTTPS URL can incorrectly prompt for GitHub
    credentials and has previously been misdiagnosed as missing repository access.
    Keep that transport out of active launch/automation code.
    """

    offenders: list[str] = []
    for path in _active_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if SELF_HTTPS_URL in text and "git clone" in text:
            offenders.append(path.relative_to(ROOT).as_posix())

    assert not offenders, (
        "Do not hard-code a fresh HTTPS self-clone in active automation. "
        "Reuse the existing checkout/object store via git worktree, or use the "
        "repository-scoped SSH transport documented by mykcs/zju-server. "
        f"Offenders: {offenders}"
    )
