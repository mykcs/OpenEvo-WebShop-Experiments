from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.openevo_webshop import experiment_registry as registry

ROOT = Path(__file__).resolve().parents[1]
OE = ROOT / "scripts" / "oe"
ALIAS = "qwen3-1p7b-directapply-no-gdr"
CANONICAL = "202609082022-q17-directapply-no-gdr"
BOUNDED = "202609160255-bounded-effective-state-gdr"


def run_passport(identity: str, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(OE), "experiments", "passport", identity, *extra],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )


def test_passport_human_output_is_one_screen_identity_card() -> None:
    result = run_passport(ALIAS)
    assert result.returncode == 0, result.stderr
    for phrase in (
        "Canonical ID: 202609082022-q17-directapply-no-gdr",
        "Purpose:", "What changed:", "What must stay fixed:",
        "Model: Qwen/Qwen3-1.7B @ 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "Budget: 160 rounds × 128 = 20480 rollouts; horizon=15",
        "Resume policy: EXACT_RECEIPT_LINEAGE_ONLY",
        "Read-only: this command never authorizes or starts continuation.",
    ):
        assert phrase in result.stdout


def test_passport_json_is_stable_read_only_projection() -> None:
    alias_result = run_passport(ALIAS, "--json")
    canonical_result = run_passport(CANONICAL, "--json")
    assert alias_result.returncode == canonical_result.returncode == 0
    alias_view = json.loads(alias_result.stdout)
    canonical_view = json.loads(canonical_result.stdout)
    assert alias_view == canonical_view
    assert alias_view["experiment_id"] == CANONICAL
    assert alias_view["mutation"] is False
    assert alias_view["passport_sha256"] == "155e8885d2af2d4ee8b281f75cfec5c8b3af18b2c42b63f440da2af3e00805b6"
    assert alias_view["protocol_fingerprint"]
    assert alias_view["current"]["activity_status"] in {
        "ACTIVE_GPU", "ACTIVE_CPU_OR_IO", "IDLE_AT_BOUNDARY", "NOT_OBSERVED", "UNKNOWN"
    }


def test_off_server_visibility_is_not_identity_failure() -> None:
    entry = registry.resolve_experiment(registry.load_current_registry(ROOT)[0], ALIAS)
    live_path = Path(str(entry["state_source"]["path"]))
    if live_path.exists():
        return
    view = json.loads(run_passport(ALIAS, "--json").stdout)
    assert view["current"] == {
        "lifecycle_status": "LIVE_NOT_OBSERVED",
        "activity_status": "NOT_OBSERVED",
        "identity_trust": "NOT_OBSERVED",
        "semantic_progress": {},
    }


def test_unknown_passport_alias_fails_closed() -> None:
    result = run_passport("not-a-real-experiment")
    assert result.returncode == 2
    assert "unknown experiment identity or alias" in result.stderr


def test_registry_router_is_minimal_and_hash_bound() -> None:
    reg, router = registry.load_current_registry(ROOT)
    assert set(router) == {"path", "sha256"}
    assert registry.validate_registry(reg, ROOT) == []
    entries = {entry["experiment_id"]: entry for entry in reg["entries"]}
    assert {CANONICAL, BOUNDED} <= set(entries)
    assert entries[CANONICAL]["lifecycle_class"] == "COMPLETE_SEALED"
    assert entries[BOUNDED]["lifecycle_class"] == "COMPLETE_SEALED"
    assert len(entries) == len(reg["entries"])
