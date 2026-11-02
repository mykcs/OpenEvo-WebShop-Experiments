from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from scripts.ci import preflight


def test_preflight_rejects_pages_wrapper_that_swallows_cpu_gate_failure() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ci_dir = root / "scripts/ci"
        ci_dir.mkdir(parents=True)
        (ci_dir / "cloudflare-cpu-ci.sh").write_text(
            "#!/usr/bin/env bash\nexit 23\n",
            encoding="utf-8",
        )
        (ci_dir / "cloudflare-pages-build.sh").write_text(
            """#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="$ROOT/.cloudflare-ci-pages"
{ bash "$ROOT/scripts/ci/cloudflare-cpu-ci.sh"; } > "$OUTPUT_DIR/index.html" 2>&1 || true
# OpenEvo CPU CI passed.
""",
            encoding="utf-8",
        )
        errors = preflight.check_workflow_contract(root)
        assert any("must not suppress failures" in error for error in errors), errors
        assert any("top-level command" in error for error in errors), errors


def test_pages_wrapper_binds_cloudflare_checkout_and_restores_main_ref() -> None:
    text = (preflight.ROOT / "scripts/ci/cloudflare-pages-build.sh").read_text(encoding="utf-8")
    for fragment in (
        "CF_PAGES_BRANCH",
        "CF_PAGES_COMMIT_SHA",
        "git rev-parse HEAD",
        "git rev-parse --is-shallow-repository",
        "git fetch --no-tags --unshallow origin",
        "main:refs/remotes/origin/main",
        "git merge-base origin/main HEAD",
    ):
        assert fragment in text, fragment


def test_pages_wrapper_propagates_gate_failure_and_removes_stale_marker() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ci_dir = root / "scripts/ci"
        ci_dir.mkdir(parents=True)
        source_wrapper = preflight.ROOT / "scripts/ci/cloudflare-pages-build.sh"
        (ci_dir / "cloudflare-pages-build.sh").write_text(
            source_wrapper.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (ci_dir / "cloudflare-cpu-ci.sh").write_text(
            "#!/usr/bin/env bash\nexit 23\n",
            encoding="utf-8",
        )
        output_dir = root / ".cloudflare-ci-pages"
        output_dir.mkdir()
        (output_dir / "index.html").write_text("stale green marker\n", encoding="utf-8")

        env = os.environ.copy()
        for name in ("CF_PAGES", "CF_PAGES_BRANCH", "CF_PAGES_COMMIT_SHA"):
            env.pop(name, None)
        result = subprocess.run(
            ["bash", str(ci_dir / "cloudflare-pages-build.sh")],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 23, result
        assert not output_dir.exists(), (
            "a failed CPU gate must not leave a stale deployable PASS marker"
        )
