from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/openevo_webshop/launch_h130_seed_route_smoke.sh"
)


def test_h130_seed_smoke_keeps_formal_consumption_fail_closed():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "formal_task_consumption_allowed=false" in text
    assert "trainer.total_training_steps=1" in text
    assert "trainer.save_freq=-1" in text
    assert "--tmpfs /tmp:rw,exec,nosuid,size=4g" in text


def test_h130_seed_smoke_binds_resources_and_runtime_identity():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "ray_init.num_cpus=32" in text
    assert "OMP_NUM_THREADS=1" in text
    assert "--pids-limit 16384" in text
    assert "search_engine/indexes_1k" in text
    assert "+actor_rollout_ref.model.override_config._attn_implementation=eager" in text
    assert "--gpus \"\\\"device=${GPU_SPEC}\\\"\"" in text


def test_h130_seed_smoke_validates_manifest_and_parquet_hashes():
    text = SCRIPT.read_text(encoding="utf-8")

    assert '.schema == "openevo.webshop-h1.30-training-manifest.v1"' in text
    assert '.training_manifest_sha256 == $sha' in text
    assert '.manifest_sha256 == $sha and .formal_task_consumption_allowed == false' in text
