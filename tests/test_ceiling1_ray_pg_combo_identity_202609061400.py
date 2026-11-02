import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "scripts/openevo_webshop/ceiling1_ray_placement_backend_202609042025.py"
spec = importlib.util.spec_from_file_location("ceiling1_pg_combo", BACKEND)
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


def test_pg_name_is_deterministic_for_same_physical_combo():
    out = ROOT / "tmp-selection.json"
    a = backend.pg_name(out, "mechanism1_seed_stage1_3b", 1, (3,))
    b = backend.pg_name(out, "mechanism1_seed_stage1_3b", 1, (3,))
    assert a == b
    assert a.startswith("ce1-")


def test_pg_name_changes_when_physical_combo_changes():
    out = ROOT / "tmp-selection.json"
    a = backend.pg_name(out, "mechanism1_seed_stage1_3b", 1, (3,))
    b = backend.pg_name(out, "mechanism1_seed_stage1_3b", 1, (4,))
    assert a != b


def test_pg_name_changes_with_combo_order_for_explicit_identity():
    out = ROOT / "tmp-selection.json"
    a = backend.pg_name(out, "stage2_rollout_3b", 2, (4, 5))
    b = backend.pg_name(out, "stage2_rollout_3b", 2, (5, 4))
    assert a != b
