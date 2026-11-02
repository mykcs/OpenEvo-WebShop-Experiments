from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
STARTUP_SOP = REPO_ROOT / "docs" / "experiment-tracking" / "STARTUP_SOP_5090.md"
CASE_STUDY = "OPEN_EVO_PLATFORM_ANTI_FRICTION_CASE_STUDY_2026-08-21.md"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_readme_routes_agents_to_anti_friction_case_and_reuses_stable_pass() -> None:
    text = _text(README)
    assert CASE_STUDY in text
    assert "Reuse an exact matching PASS runtime qualification" in text
    assert "runtime projection of the parent standing pool" in text
    assert "Run CPU/contract gates and a bounded single-GPU gate before preregistered multi-GPU cells." not in text
    assert "set the UUIDs allocated by\n`zju-server`" not in text


def test_startup_sop_uses_scripts_oe_as_current_control_plane() -> None:
    text = _text(STARTUP_SOP)
    assert "当前 canonical launcher 是 `scripts/oe`" in text
    assert "fingerprint unchanged → reuse exact PASS" in text
    assert "runtime candidate projection" in text
    assert "established eight-device RTX 5090 pool" in text
    assert "verified monitoring-only `nvitop` handle" in text
    assert "`profile.command` 是完整 executable argv" in text


def test_startup_sop_does_not_restore_superseded_gate_semantics() -> None:
    text = _text(STARTUP_SOP)
    forbidden = (
        "新科学线必须先提交新 preregistration、更新 current-campaign，并从父仓 standing pool 登记精确 UUID 子集",
        "先完成并记录 CPU/单卡 gate",
        "本次 run/campaign 已授权的 UUID 集合",
        "startup_gate_5090.sh` 只检查本次",
    )
    for phrase in forbidden:
        assert phrase not in text


def test_legacy_startup_gate_is_compatibility_only_in_active_sop() -> None:
    text = _text(STARTUP_SOP)
    assert "历史 `scripts/openevo_webshop/startup_gate_5090.sh`" in text
    assert "不再是 future campaign 的首选 control plane" in text
