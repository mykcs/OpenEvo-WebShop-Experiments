import copy
import importlib.util
from pathlib import Path

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("m1a", HERE.parent / "scripts/openevo_webshop/resolve_mechanism1_m1a_source_202609062100.py")
m1a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m1a)


def state(i):
    return {
        "accepted_transition_index": i,
        "round_index": i * 2,
        "generation_index": i * 3,
        "adapter_sha256": f"{i:064x}",
        "state_receipt_path": f"s{i}.json",
        "state_receipt_sha256": f"{i + 10:064x}",
    }


def inputs():
    ledger = {"schema": m1a.LEDGER_SCHEMA, "terminal_seal": True,
              "accepted_states": [state(i) for i in range(1, 8)]}
    norm = {"schema": m1a.NORM_SCHEMA, "status": "PASS", "gram_norm": 2.0,
            "source_adapter_sha256": ledger["accepted_states"][-2]["adapter_sha256"],
            "target_adapter_sha256": ledger["accepted_states"][-1]["adapter_sha256"]}
    return ledger, norm


def test_resolves_last_three_and_remains_locked():
    ledger, norm = inputs()
    result = m1a.resolve(ledger, norm)
    assert result["theta_target"]["accepted_transition_index"] == 5
    assert result["theta_previous"]["accepted_transition_index"] == 6
    assert result["theta_final"]["accepted_transition_index"] == 7
    assert result["execution_authority"] is False
    assert result["formal_task_consumption_allowed"] is False


def test_rejects_duplicate_adapter():
    ledger, norm = inputs()
    ledger["accepted_states"][-1]["adapter_sha256"] = ledger["accepted_states"][-2]["adapter_sha256"]
    norm["target_adapter_sha256"] = ledger["accepted_states"][-1]["adapter_sha256"]
    try:
        m1a.resolve(ledger, norm)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate adapter was accepted")


def test_rejects_nonconsecutive_or_nonpositive_norm():
    ledger, norm = inputs()
    broken = copy.deepcopy(ledger)
    broken["accepted_states"][-1]["accepted_transition_index"] += 1
    try:
        m1a.resolve(broken, norm)
    except ValueError as exc:
        assert "consecutive" in str(exc)
    else:
        raise AssertionError("nonconsecutive ledger was accepted")
    norm["gram_norm"] = 0
    try:
        m1a.resolve(ledger, norm)
    except ValueError as exc:
        assert "finite and positive" in str(exc)
    else:
        raise AssertionError("zero vector was accepted")
