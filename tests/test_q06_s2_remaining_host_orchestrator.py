from __future__ import annotations

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from scripts.openevo_webshop import q06_s2_remaining_host_orchestrator as host


def contract(cell_id="gdr-r80"):
    mechanism, rounds, checkpoint_id = host.ALLOWED_CELLS[cell_id]
    return {
        "schema": host.CONTRACT_SCHEMA, "status": "LAUNCH_AUTHORIZED",
        "append_only": True, "question_id": "Q06_TOKEN_CLICK_ENTROPY", "stage": "S2",
        "cell": {"cell_id": cell_id, "mechanism": mechanism, "completed_stage2_rounds": rounds,
                 "checkpoint_id": checkpoint_id, "lineage_id": "exact-gdr-training-run"},
        "execution_constraints": {**host.SCIENCE_LIMITS, "max_scorer_forward_calls": 100000,
                                  "max_scorer_input_tokens": 1000000,
                                  "max_persistent_output_bytes": 1073741824},
        "resource_limit_provenance": {"source": "frozen-scorer-cost-receipt", "classification": "protective_estimate"},
    }


def load_fixture(repo, value):
    path = repo / "contract.json"
    path.write_text(json.dumps(value))
    return host.validate_contract(repo, {"path": path.name, "sha256": host.sha256_file(path)})


def test_allowlist_only_contains_unconsumed_existing_weight_cells():
    assert len(host.ALLOWED_CELLS) == 7
    assert not {"bounded-r120", "bounded-r160", "gdr-r160", "ordinary-r40", "ordinary-r120"} & host.ALLOWED_CELLS.keys()
    with TemporaryDirectory() as temporary:
        repo = Path(temporary)
        for cell_id in host.ALLOWED_CELLS:
            assert load_fixture(repo, contract(cell_id))[1]["cell"]["cell_id"] == cell_id
        value = contract()
        value["cell"]["cell_id"] = "gdr-r160"
        with pytest.raises(host.GateError, match="already consumed"):
            load_fixture(repo, value)


@pytest.mark.parametrize("field,value", [("episodes", 63), ("passes", 2), ("optimizer_updates", 1),
                                         ("teacher_calls", 1), ("final_panel_access_allowed", True)])
def test_science_budget_rejects_drift(field, value):
    with TemporaryDirectory() as temporary:
        document = contract()
        document["execution_constraints"][field] = value
        with pytest.raises(host.GateError, match="science budget drift"):
            load_fixture(Path(temporary), document)


def test_resource_estimates_require_source_and_do_not_require_wall_stop():
    with TemporaryDirectory() as temporary:
        repo = Path(temporary)
        value = contract()
        assert "max_gpu_hours" not in load_fixture(repo, value)[1]["execution_constraints"]
        del value["resource_limit_provenance"]
        with pytest.raises(host.GateError, match="lack provenance"):
            load_fixture(repo, value)


def test_command_runs_new_observer_with_frozen_science_import_priority(monkeypatch):
    with TemporaryDirectory() as temporary:
        repo = Path(temporary)
        monkeypatch.setattr(host, "READ_ROOT", repo)
        producer, science, adapter, seed, model, output = [repo / name for name in (
            "producer", "science", "adapter", "seed", "model", "output")]
        for path in (producer, science, adapter, seed, model, output):
            path.mkdir()
        carriers = {}
        for key in ("agent_system", "memory", "skill"):
            path = repo / (key + ".md")
            path.write_text(key)
            carriers[key + "_path"] = str(path)
        manifest = {"model": {"local_model_path": str(model), "revision": "fixed-model"},
                    "common_carriers": carriers, "endpoint": {"adapter_path": str(adapter)},
                    "seed_root": {"path": str(seed)}, "task_panel_manifest_path": "fixed-panel.json"}
        original = copy.deepcopy(manifest)
        command = host.build_probe_command(repo, producer, science, repo / "manifest.json", manifest,
                                            {"probe_guard": {"path": "guard.py"}}, contract(), output, "GPU-fixed")
        assert command[command.index("--probe-script") + 1] == str(producer / host.PROBE_PATH)
        assert command[command.index("--runtime-source-root") + 1] == str(science)
        assert f"PYTHONPATH={science}:{repo}:{producer}:{science}/vendor/openevo-webshop/src" in command
        assert "--q06-pre-action-diagnostics" in command
        assert command[command.index("--q06-lineage-id") + 1] == "exact-gdr-training-run"
        assert "--task-limit" not in command
        assert manifest == original
        assert command.count("--mount") == 2
        assert f"type=bind,src={repo},dst={repo},readonly" in command


def test_ordinary_adapter_path_is_bound_by_actual_update_not_fabricated_summary():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        adapter = root / "adapter_model.safetensors"
        adapter.write_bytes(b"ordinary payload")
        adapter_sha = host.sha256_file(adapter)
        summary = root / "summary.json"
        summary.write_text(json.dumps({"checkpoint_id": "after_080_updates", "checkpoint": {
            "state_id": "frozen-ordinary-state", "adapter_sha256": adapter_sha, "update_count": 80}}))
        update = root / "update.json"
        update.write_text(json.dumps({"adapter_path": str(root), "adapter_digest_after": "sha256:" + adapter_sha}))
        value = contract("ordinary-r80")
        value["cell"].update({"source_summary_path": str(summary), "source_summary_sha256": host.sha256_file(summary),
                              "actual_parameter_update_events": 80, "source_update_path": str(update),
                              "source_update_sha256": host.sha256_file(update)})
        value["known_shared_identity"] = {"adapter": {"adapter_sha256": adapter_sha,
                                                        "size_bytes": adapter.stat().st_size}}
        manifest = {"endpoint": {"adapter_path": str(root)}, "source_update_path": str(update),
                    "source_update_sha256": host.sha256_file(update)}
        checkpoint, actual = host.resolve_checkpoint_adapter(value, manifest)
        assert actual == adapter
        assert "adapter_path" not in checkpoint
        manifest["endpoint"]["adapter_path"] = str(root / "invented")
        with pytest.raises(host.GateError, match="update adapter identity drift"):
            host.resolve_checkpoint_adapter(value, manifest)


def test_all_adapter_files_and_configuration_are_hash_bound():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        config = root / "adapter_config.json"
        config.write_text('{"r":128,"lora_alpha":256}')
        endpoint = {"adapter_path": str(root), "source_files": {"adapter_config": {
            "path": str(config), "sha256": host.sha256_file(config), "bytes": config.stat().st_size}}}
        host.validate_endpoint_source_files(endpoint)
        config.write_text('{"r":64,"lora_alpha":256}')
        with pytest.raises(host.GateError, match="byte identity drift"):
            host.validate_endpoint_source_files(endpoint)
        endpoint["source_files"].clear()
        with pytest.raises(host.GateError, match="inventory missing"):
            host.validate_endpoint_source_files(endpoint)
