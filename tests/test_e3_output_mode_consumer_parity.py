from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_science_output_writer_matches_e3_qualifier_mode_contract() -> None:
    writer = (ROOT / "scripts/openevo_webshop/ensure_ceiling1_science_output_writable.sh").read_text()
    qualifier = (ROOT / "scripts/openevo_webshop/post_advisor_e3_prelaunch_qualifier_202609300248.py").read_text()
    assert 'install -d -o 1001 -g 1001 -m 750 "$OUT"' in writer
    assert 'install -d -o 1001 -g 1001 -m 770 "$OUT"' not in writer
    assert 'stat.S_IMODE(info.st_mode)!=0o750' in qualifier
