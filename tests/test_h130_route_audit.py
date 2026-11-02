from scripts.openevo_webshop.audit_h130_routes import audit


def _config():
    return {
        "frozen_comparison_contract": {
            "model_revision": "model-rev",
            "prompt_contract": "prompt-v1",
            "action_contract": "action-v1",
            "evaluator": "shared-evaluator-v1",
        }
    }


def _route(arm, *, status="audited", manifest="manifest-hash"):
    route = {
        "arm": arm,
        "status": status,
        "method": arm.lower(),
        "evaluation": {
            "model_revision": "model-rev",
            "prompt_contract": "prompt-v1",
            "action_contract": "action-v1",
            "evaluator": "shared-evaluator-v1",
            "task_manifest_sha256": manifest,
        },
    }
    if arm == "OPEN_EVO":
        route["adapter"] = {
            "adapter_identity_digest": "adapter-id",
            "state_weights_sha256": "state-hash",
            "peft_weights_sha256": "peft-hash",
            "source_training_manifest_sha256": "train-hash",
            "fresh_reload": {"status": "pass"},
        }
    else:
        route["checkpoint"] = {
            "checkpoint_sha256": "checkpoint-hash",
            "reload_receipt_sha256": "reload-hash",
            "source_revision": "seed-rev",
            "reload_status": "pass",
        }
    return route


def test_route_audit_passes_only_with_both_fresh_routes():
    result = audit(_config(), {"manifest_sha256": "manifest-hash"}, [_route("OPEN_EVO"), _route("SEED")])
    assert result["status"] == "pass"
    assert result["release_gate"] == "pass"
    assert result["formal_task_consumption_allowed"] is False


def test_route_audit_blocks_stale_open_evo_and_missing_seed_reload():
    open_evo = _route("OPEN_EVO", status="candidate-not-authorized")
    seed = _route("SEED")
    seed["checkpoint"]["reload_status"] = "missing"
    result = audit(_config(), {"manifest_sha256": "manifest-hash"}, [open_evo, seed])
    assert result["status"] == "blocked"
    assert any("OPEN_EVO route status" in item for item in result["failures"])
    assert any("SEED checkpoint reload status" in item for item in result["failures"])


def test_route_audit_blocks_manifest_mismatch():
    result = audit(_config(), {"manifest_sha256": "manifest-hash"}, [_route("OPEN_EVO", manifest="other"), _route("SEED")])
    assert result["status"] == "blocked"
    assert any("task_manifest_sha256" in item for item in result["failures"])
