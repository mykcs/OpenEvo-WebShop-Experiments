from __future__ import annotations
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "configs/experiment/campaigns/202609111730/authority-snapshots/020b6b2fccaf3e924f3e22299c45467cf0544f3f5e8d9cbd5df6f7ab839003d7.json"
DRIVER = ROOT / "scripts/openevo_webshop/ceiling1_stage2_current_arm_driver_202609042025.sh"
EXPECTED = "020b6b2fccaf3e924f3e22299c45467cf0544f3f5e8d9cbd5df6f7ab839003d7"

def test_frozen_step_seed_asset_is_in_execution_tree() -> None:
    assert ASSET.is_file()
    assert hashlib.sha256(ASSET.read_bytes()).hexdigest() == EXPECTED

def test_current_arm_driver_consumes_tree_bound_step_seed_asset() -> None:
    text = DRIVER.read_text()
    assert 'STEP_SEED="$REPO/configs/experiment/campaigns/202609111730/authority-snapshots/020b6b2fccaf3e924f3e22299c45467cf0544f3f5e8d9cbd5df6f7ab839003d7.json"' in text
    assert '--step-seed-amendment "$STEP_SEED"' in text
