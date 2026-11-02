from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts/openevo_webshop/run_mechanism1_m1d_seed_stage1_202609060100.py"


def load_driver():
    spec = importlib.util.spec_from_file_location("m1d_driver_argparse_test", DRIVER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_upstream_seed_flag_does_not_abbreviate_seed_root():
    driver = load_driver()
    argv = [
        str(DRIVER),
        "--seed-root", "/seed",
        "--source-git-root", "/seed",
        "--manifest", "/manifest.json",
        "--mode", "formal",
        "--output-dir", "/formal-output",
        "--seed", "2026",
        "--num-tasks", "180",
    ]
    with mock.patch.object(sys, "argv", argv), \
         mock.patch.object(driver, "load_manifest", return_value={}), \
         mock.patch.object(driver, "verify_source", return_value={"source_commit": driver.SOURCE_COMMIT}) as verify, \
         mock.patch.object(driver, "run_formal", return_value={"status": "PASS"}) as run_formal:
        assert driver.main() == 0
    verify.assert_called_once_with(Path("/seed"), Path("/seed"))
    remainder = run_formal.call_args.args[2]
    assert remainder == [
        "--output-dir", "/formal-output",
        "--seed", "2026",
        "--num-tasks", "180",
    ]


def test_driver_parser_explicitly_disables_option_abbreviation():
    source = DRIVER.read_text(encoding="utf-8")
    assert "ArgumentParser(allow_abbrev=False)" in source
