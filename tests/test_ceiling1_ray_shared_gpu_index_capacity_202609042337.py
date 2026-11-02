from pathlib import Path
POOL=Path(__file__).resolve().parents[1]/'scripts/openevo_webshop/ceiling1_ray_gpu_pool_202609042025.py'
def test_gpu_identity_capacity_tracks_process_slots():
    text=POOL.read_text()
    assert "f'gpu_index_{idx}': slots" in text
    assert "f'gpu_index_{idx}': 1" not in text
def test_real_exclusivity_is_not_gpu_identity():
    text=POOL.read_text()
    assert "process_slot': slots" in text
    assert "arm_3b_slot': same" in text
    assert "arm_q17_slot': same" in text
