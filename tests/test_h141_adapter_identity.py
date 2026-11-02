from pathlib import Path

from scripts.openevo_webshop.compute_adapter_tensor_digest import _loaded_parameter_name
from scripts.openevo_webshop.h141_upstream_evaluation import adapter_identity


def test_safetensors_lora_names_match_inference_parameter_names():
    assert _loaded_parameter_name("base_model.model.layer.lora_A.weight") == "base_model.model.layer.lora_A.default.weight"
    assert _loaded_parameter_name("base_model.model.layer.lora_B.weight") == "base_model.model.layer.lora_B.default.weight"


def test_adapter_identity_reads_state_digest_and_recomputes_loaded_digest(monkeypatch, tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "openevo_sd_lora_state.safetensors").write_bytes(b"state")
    state_sha = __import__("hashlib").sha256(b"state").hexdigest()
    (adapter / "openevo_sd_lora_state.json").write_text('{"state_weights_sha256": "' + state_sha + '"}\n', encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")

    class Result:
        returncode = 0
        stdout = '{"loaded_tensor_digest": "' + ('a' * 64) + '", "tensor_count": 2}\n'
        stderr = ""

    monkeypatch.setattr("scripts.openevo_webshop.h141_upstream_evaluation.subprocess.run", lambda *args, **kwargs: Result())
    identity = adapter_identity(adapter)
    assert identity["state_weights_sha256"] == state_sha
    assert identity["tensor_digest"] == "a" * 64
    assert identity["tensor_count"] == 2
