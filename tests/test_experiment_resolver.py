from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.openevo_webshop import experiment_resolver as resolver
from scripts.openevo_webshop import golden_runtime_registry as registry


ROOT = Path(__file__).resolve().parents[1]
ZERO_SHA = "0" * 64
ONE_SHA = "1" * 64
TWO_SHA = "2" * 64


def _experiment(asset_id: str) -> dict:
    model = registry.load_model(asset_id)
    dataset = registry.load_dataset("seed-webshop-v1")
    runtime = registry.load_runtime()
    runtime_evidence = registry._read(ROOT / runtime["runtime_evidence"]["path"])
    provenance = {
        "upstream_openevo_sha": runtime_evidence["sources"]["openevo_revision"],
        "seed_repository": "https://github.com/jinyangwu/SEED",
        "seed_commit": "2cf2fadca3c5aba28da68e8e1405182ba8d90e6c",
        "base_model": model["hf_id"],
        "base_model_content_pin": model["content_pin"],
    }
    if model.get("revision") is not None:
        provenance["base_model_revision"] = model["revision"]
    return {
        "schema": "openevo.experiment-test-fast-path.v1",
        "experiment": f"TEST-{asset_id}",
        "campaign_id": f"test-{asset_id}",
        "status": "preregistered-pre-execution",
        "provenance": provenance,
        "benchmark": {
            "name": "WebShop small / SEED-compatible heldout-cleaned stream",
            "environment": "WebAgentTextEnv-v0",
            "num_products": 1000,
            "human_goals": False,
            "data_files": {
                "items_shuffle_1000.json": dataset["required_files_sha256"]["items_shuffle_1000.json"],
                "items_ins_v2_1000.json": dataset["required_files_sha256"]["items_ins_v2_1000.json"],
            },
        },
        "interface_contract": {
            "prompt_contract": "seed3090.webshop-compact-action-prompt.v2",
            "action_parser_contract": "seed3090.webshop-action-parser.v3-action-command",
        },
        "execution_contract": {
            "rollout_runner": {"path": "scripts/openevo_webshop/test_rollouts.py", "sha256": ONE_SHA},
            "native_trainer_entrypoint": {"path": "scripts/openevo_webshop/test_trainer.py", "sha256": TWO_SHA},
        },
        "schedule": {
            "path": "configs/experiment/manifests/test-schedule.json",
            "content_sha256": ONE_SHA,
            "file_sha256": TWO_SHA,
        },
    }


def _write(tmp_path: Path, payload: dict, name: str = "experiment.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def test_3b_formal_identity_resolves_to_3b(tmp_path):
    payload = _experiment("qwen2.5-3b-instruct")
    model = registry.load_model("qwen2.5-3b-instruct")
    payload["provenance"]["model_files_sha256"] = {
        key: model["required_files_sha256"][key]
        for key in ("config.json", "model.safetensors.index.json", "tokenizer_config.json")
    }
    result = resolver.resolve(_write(tmp_path, payload))
    assert result["status"] == "RESOLVED"
    assert result["model"]["asset_id"] == "qwen2.5-3b-instruct"
    assert result["dataset"]["asset_id"] == "seed-webshop-v1"
    assert result["model"]["declared_file_hashes_checked"] == payload["provenance"]["model_files_sha256"]
    assert result["authority_boundary"]["execution_authorized"] is False


def test_7b_formal_identity_resolves_without_invented_revision(tmp_path):
    payload = _experiment("qwen2.5-7b-instruct")
    result = resolver.resolve(_write(tmp_path, payload))
    assert result["model"]["asset_id"] == "qwen2.5-7b-instruct"
    assert result["model"]["revision"] is None


def test_resolution_digest_is_independent_of_absolute_config_path(tmp_path):
    payload = _experiment("qwen2.5-3b-instruct")
    first = resolver.resolve(_write(tmp_path, payload, "a.json"))
    second_dir = tmp_path / "nested"
    second_dir.mkdir()
    second = resolver.resolve(_write(second_dir, payload, "b.json"))
    assert first["experiment"]["config_path"] != second["experiment"]["config_path"]
    assert first["experiment"]["config_sha256"] == second["experiment"]["config_sha256"]
    assert first["resolution_sha256"] == second["resolution_sha256"]


def test_missing_content_pin_fails_closed(tmp_path):
    payload = _experiment("qwen2.5-3b-instruct")
    del payload["provenance"]["base_model_content_pin"]
    with pytest.raises(resolver.ExperimentResolutionError, match="base_model_content_pin"):
        resolver.resolve(_write(tmp_path, payload))


def test_wrong_content_pin_fails_closed(tmp_path):
    payload = _experiment("qwen2.5-3b-instruct")
    payload["provenance"]["base_model_content_pin"] = ZERO_SHA
    with pytest.raises(resolver.ExperimentResolutionError, match="content pin mismatch"):
        resolver.resolve(_write(tmp_path, payload))


def test_wrong_declared_model_file_hash_fails_closed(tmp_path):
    payload = _experiment("qwen2.5-3b-instruct")
    payload["provenance"]["model_files_sha256"] = {"config.json": ZERO_SHA}
    with pytest.raises(resolver.ExperimentResolutionError, match="model file hash mismatch"):
        resolver.resolve(_write(tmp_path, payload))


def test_unknown_declared_model_file_fails_closed(tmp_path):
    payload = _experiment("qwen2.5-3b-instruct")
    payload["provenance"]["model_files_sha256"] = {"mystery.bin": ZERO_SHA}
    with pytest.raises(resolver.ExperimentResolutionError, match="unknown model file"):
        resolver.resolve(_write(tmp_path, payload))


def test_wrong_dataset_hash_fails_closed(tmp_path):
    payload = _experiment("qwen2.5-3b-instruct")
    payload["benchmark"]["data_files"]["items_shuffle_1000.json"] = ZERO_SHA
    with pytest.raises(resolver.ExperimentResolutionError, match="dataset file hash mismatch"):
        resolver.resolve(_write(tmp_path, payload))


def test_missing_required_dataset_file_fails_closed(tmp_path):
    payload = _experiment("qwen2.5-3b-instruct")
    del payload["benchmark"]["data_files"]["items_ins_v2_1000.json"]
    with pytest.raises(resolver.ExperimentResolutionError, match="missing required WebShop data files"):
        resolver.resolve(_write(tmp_path, payload))


def test_unknown_dataset_file_fails_closed(tmp_path):
    payload = _experiment("qwen2.5-3b-instruct")
    payload["benchmark"]["data_files"]["unknown.json"] = ZERO_SHA
    with pytest.raises(resolver.ExperimentResolutionError, match="unknown dataset file"):
        resolver.resolve(_write(tmp_path, payload))


def test_wrong_openevo_revision_fails_closed(tmp_path):
    payload = _experiment("qwen2.5-3b-instruct")
    payload["provenance"]["upstream_openevo_sha"] = "f" * 40
    with pytest.raises(resolver.ExperimentResolutionError, match="OpenEvo revision mismatch"):
        resolver.resolve(_write(tmp_path, payload))


def test_wrong_seed_commit_fails_closed(tmp_path):
    payload = _experiment("qwen2.5-3b-instruct")
    payload["provenance"]["seed_commit"] = "f" * 40
    with pytest.raises(resolver.ExperimentResolutionError, match="SEED commit mismatch"):
        resolver.resolve(_write(tmp_path, payload))


def test_wrong_seed_repository_fails_closed(tmp_path):
    payload = _experiment("qwen2.5-3b-instruct")
    payload["provenance"]["seed_repository"] = "https://example.invalid/not-seed"
    with pytest.raises(resolver.ExperimentResolutionError, match="SEED repository mismatch"):
        resolver.resolve(_write(tmp_path, payload))


def test_wrong_webshop_environment_and_size_fail_closed(tmp_path):
    payload = _experiment("qwen2.5-3b-instruct")
    payload["benchmark"]["environment"] = "OtherEnv-v0"
    with pytest.raises(resolver.ExperimentResolutionError, match="unsupported benchmark environment"):
        resolver.resolve(_write(tmp_path, payload))
    payload = _experiment("qwen2.5-3b-instruct")
    payload["benchmark"]["num_products"] = 10000
    with pytest.raises(resolver.ExperimentResolutionError, match="unsupported WebShop product count"):
        resolver.resolve(_write(tmp_path, payload, "size.json"))


def test_declared_7b_revision_cannot_be_claimed_without_asset_proof(tmp_path):
    payload = _experiment("qwen2.5-7b-instruct")
    payload["provenance"]["base_model_revision"] = "invented-upstream-revision"
    with pytest.raises(resolver.ExperimentResolutionError, match="model revision mismatch"):
        resolver.resolve(_write(tmp_path, payload))


def test_invalid_science_contract_hash_fails_closed(tmp_path):
    payload = _experiment("qwen2.5-3b-instruct")
    payload["execution_contract"]["rollout_runner"]["sha256"] = "not-a-sha"
    with pytest.raises(resolver.ExperimentResolutionError, match="rollout_runner.sha256"):
        resolver.resolve(_write(tmp_path, payload))


def test_standalone_resolver_cli_is_read_only_and_returns_json(tmp_path):
    path = _write(tmp_path, _experiment("qwen2.5-3b-instruct"))
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/openevo_webshop/experiment_resolver.py"), "--experiment", str(path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    result = json.loads(completed.stdout)
    assert result["status"] == "RESOLVED"
    assert result["authority_boundary"]["formal_task_consumption_allowed"] is False


def test_oe_resolve_routes_to_same_resolver(tmp_path):
    path = _write(tmp_path, _experiment("qwen2.5-7b-instruct"))
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/oe"), "resolve", "--experiment", str(path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    result = json.loads(completed.stdout)
    assert result["status"] == "RESOLVED"
    assert result["model"]["asset_id"] == "qwen2.5-7b-instruct"
    assert result["authority_boundary"]["execution_authorized"] is False
