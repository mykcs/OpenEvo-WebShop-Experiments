from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SOURCE = ROOT / "vendor/openevo-webshop/src/openevo_webshop"


class WebshopPackagingContractTests(unittest.TestCase):
    def test_installed_source_import_and_contract_test_do_not_need_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            install_root = Path(tmp) / "site-packages"
            shutil.copytree(PACKAGE_SOURCE, install_root / "openevo_webshop")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(install_root)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import openevo_webshop; "
                        "from openevo_webshop.cli import main; "
                        "raise SystemExit(main(['contract-test']))"
                    ),
                ],
                cwd=tmp,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("contract checks passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
