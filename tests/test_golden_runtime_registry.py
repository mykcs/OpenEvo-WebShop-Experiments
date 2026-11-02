from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.openevo_webshop import golden_runtime_registry as registry


def _experiment(model: dict[str, str | None]) -> dict:
    return {"provenance": {key: value for key, value in model.items() if value is not None}}


def test_runtime_family_is_model_and_dataset_independent():
    runtime = registry.load_runtime()
    assert "model" not in runtime
    assert "dataset" not in runtime
    assert "assets" not in runtime
    assert runtime["asset_contract"] == {
        "model": "external-model-asset-required",
        "dataset": "external-dataset-asset-required",
    }


def test_registry_contains_exact_3b_and_7b_bindings():
    result = registry.validate_all()
    assert result["status"] == "PASS", result
    assert result["model_assets"] == ["qwen2.5-3b-instruct", "qwen2.5-7b-instruct"]


def test_h144_3b_resolves_only_to_3b_asset():
    asset = registry.load_model("qwen2.5-3b-instruct")
    result = registry.resolve_experiment(
        _experiment(
            {
                "base_model": asset["hf_id"],
                "base_model_revision": asset["revision"],
                "base_model_content_pin": asset["content_pin"],
            }
        )
    )
    assert result["model_asset"] == "qwen2.5-3b-instruct"
    assert result["model_hf_id"] == "Qwen/Qwen2.5-3B-Instruct"


def test_h143_7b_resolves_by_content_identity_without_invented_revision():
    asset = registry.load_model("qwen2.5-7b-instruct")
    assert asset["revision"] is None
    result = registry.resolve_experiment(
        _experiment(
            {
                "base_model": asset["hf_id"],
                "base_model_content_pin": asset["content_pin"],
            }
        )
    )
    assert result["model_asset"] == "qwen2.5-7b-instruct"
    assert result["model_hf_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert result["model_revision"] is None


def test_3b_cannot_silently_fallback_to_7b():
    bindings = json.loads(registry.BINDINGS_PATH.read_text(encoding="utf-8"))
    broken = copy.deepcopy(bindings)
    three = next(rule for rule in broken["rules"] if rule["match"]["base_model_hf_id"].endswith("3B-Instruct"))
    three["model_asset"] = "qwen2.5-7b-instruct"
    original_read = registry._read
    with patch.object(registry, "_read", side_effect=lambda path: broken if path == registry.BINDINGS_PATH else original_read(path)):
        result = registry.validate_all()
    assert result["status"] == "FAIL"
    assert any("binding/model HF identity mismatch" in error for error in result["errors"])


def test_revision_mismatch_fails_closed():
    asset = registry.load_model("qwen2.5-3b-instruct")
    experiment = _experiment(
        {
            "base_model": asset["hf_id"],
            "base_model_revision": "not-the-frozen-revision",
            "base_model_content_pin": asset["content_pin"],
        }
    )
    with pytest.raises(registry.RegistryError, match="model revision mismatch"):
        registry.resolve_experiment(experiment)


def test_declared_revision_fails_when_asset_has_no_authoritative_revision():
    asset = registry.load_model("qwen2.5-7b-instruct")
    experiment = _experiment(
        {
            "base_model": asset["hf_id"],
            "base_model_revision": "invented-revision",
            "base_model_content_pin": asset["content_pin"],
        }
    )
    with pytest.raises(registry.RegistryError, match="model revision mismatch"):
        registry.resolve_experiment(experiment)


def test_content_pin_mismatch_fails_closed():
    asset = registry.load_model("qwen2.5-3b-instruct")
    experiment = _experiment(
        {
            "base_model": asset["hf_id"],
            "base_model_revision": asset["revision"],
            "base_model_content_pin": "0" * 64,
        }
    )
    with pytest.raises(registry.RegistryError, match="model content pin mismatch"):
        registry.resolve_experiment(experiment)


def test_unknown_model_has_no_fallback():
    with pytest.raises(registry.RegistryError, match="exactly one binding"):
        registry.resolve_experiment(_experiment({"base_model": "Example/Unknown-Model"}))


def test_placeholder_revision_is_rejected(tmp_path, monkeypatch):
    asset = copy.deepcopy(registry.load_model("qwen2.5-7b-instruct"))
    asset["revision"] = "unknown-placeholder"
    path = tmp_path / "qwen2.5-7b-instruct.json"
    path.write_text(json.dumps(asset), encoding="utf-8")
    monkeypatch.setattr(registry, "MODEL_ROOT", tmp_path)
    with pytest.raises(registry.RegistryError, match="non-authoritative revision placeholder"):
        registry.load_model("qwen2.5-7b-instruct")
