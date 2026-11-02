from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "docs/evidence/wb1/phase-2/amendment006-gen28-state-v28"
CURRENT = ROOT / "configs/experiment/current-campaign.json"
INDEX = ROOT / "docs/evidence/wb1/results-index.v1.json"

STATE_CHECKSUM = "8fd4df88c667436cb5d5f9c5ea2ac6206c748a47199bc2b3a48417b0a59211b7"
STATE_FILE_SHA = "9c3dc189891c71851bf17f0fa347544216f2805727cd616a92e5848ba7ed2403"
MEMORY_SHA = "64b8febd6abd8326ccd9c1b08cd1c9be11c3132c9544fc47dcf91ec2dbd9701b"
REPAIR_SHA = "013c98881653a14830ebcfb482bd28345376a03bd8f7c1aaad0308491b055472"
BARRIER_SHA = "76005090736fca36e7955b378b798da68768130e764f7b940d7e0abbbe048ba9"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gen28_repair_barrier_and_state_identity_agree() -> None:
    repair = load(EV / "repair-receipt.json")
    barrier = load(EV / "state-v28-barrier.json")
    native = load(EV / "native/receipt.json")
    state = load(EV / "native/state.json")
    rec = load(EV / "RECONCILIATION.json")

    assert repair["repair_receipt_sha256"] == REPAIR_SHA
    assert barrier["barrier_sha256"] == BARRIER_SHA
    assert repair["new_webshop_task_consumption"] == 0
    assert repair["gpu_authorized"] is False
    assert barrier["new_webshop_task_consumption"] == 0
    assert barrier["gpu_used"] is False
    assert barrier["final_pool_consumed"] == 0
    assert barrier["gen29_authorized"] is False

    assert state["state_version"] == 28
    assert state["state_sha256"] == STATE_CHECKSUM
    assert sha(EV / "native/state.json") == STATE_FILE_SHA
    assert native["state_sha256"] == STATE_CHECKSUM
    assert native["partial_task_score_included"] is False
    assert native["open_evo_sha"] == "796e6248a61112a61c880885844d3b0420c0b6ae"
    assert state["adapter_write"] is False
    assert state["optimizer_write"] is False
    assert state["parameter_update"] is False
    assert barrier["state_v28"]["state_checksum"] == STATE_CHECKSUM
    assert barrier["state_v28"]["state_file_sha256"] == STATE_FILE_SHA
    assert barrier["state_v28"]["memory_markdown_sha256"] == MEMORY_SHA
    assert rec["classification"] == "GEN28_STATE_V28_BARRIER_PASS_ADOPTED"
    assert rec["status"] == "PASS"


def test_cross_host_seals_bind_identical_scientific_inputs() -> None:
    transfer = load(EV / "cross-host-transfer-receipt.json")
    server = load(EV / "server-input-seal.json")
    local = load(EV / "local-input-seal.json")

    assert transfer["status"] == "PASS"
    assert transfer["server_seal_sha256"] == server["seal_sha256"]
    assert transfer["local_seal_sha256"] == local["seal_sha256"]
    assert transfer["episodes_sha256"] == server["files"]["episodes"]["sha256"] == local["files"]["episodes"]["sha256"]
    assert transfer["task_stream_sha256"] == server["task_stream_sha256"] == local["task_stream_sha256"]
    assert server["episode_contract"]["valid_completed"] == 128
    assert local["episode_contract"]["valid_completed"] == 128
    assert server["episode_contract"]["exact_successes"] == 2
    assert local["episode_contract"]["exact_successes"] == 2


def test_pre_method_failures_did_not_consume_scientific_update() -> None:
    for attempt in (1, 2):
        receipt = load(EV / f"environment-attempt-0{attempt}.json")
        assert receipt["status"] == "INVALID_PRE_METHOD_ENVIRONMENT"
        assert receipt["native_state_written"] is False
        assert receipt["native_receipt_written"] is False
        assert receipt["repair_receipt_written"] is False
        assert receipt["codex_invoked"] is False
        assert receipt["webshop_executed"] is False
        assert receipt["gpu_used"] is False


def test_current_campaign_owner_and_immutable_e1_seal_remain_valid() -> None:
    from scripts import verify_webshop_program_ledger
    from scripts.analysis import validate_post_advisor_e1_evidence_202610010045 as e1_bundle
    from scripts.openevo_webshop import experiment_registry

    current = load(CURRENT)
    assert current["campaign_id"] == "20260928-2338-post-advisor-unanswered"

    # Reuse the live owner to validate current routing; do not pin this campaign
    # to an E1 phase/status as the same campaign advances through later cells.
    assert verify_webshop_program_ledger.verify(ROOT)["current_campaign"] == current["campaign_id"]

    identity = current["experiment_identity"]
    registry, _router = experiment_registry.load_current_registry(ROOT)
    entries = [row for row in registry["entries"] if row["experiment_id"] == identity["experiment_id"]]
    assert len(entries) == 1
    assert entries[0]["passport_sha256"] == identity["passport_sha256"]
    experiment_registry.load_bound_passport(ROOT, entries[0])

    # The live release pointer must remain repository-confined and hash-bound.
    release = current["release"]
    release_path = Path(release["path"])
    assert not release_path.is_absolute() and ".." not in release_path.parts
    release_file = (ROOT / release_path).resolve()
    assert release_file.is_relative_to(ROOT.resolve()) and release_file.is_file()
    assert sha(release_file) == release["sha256"]
    assert current["release_receipt"] == release["path"]
    assert current["release_receipt_sha256"] == release["sha256"]

    # Preserve the locked final-panel boundary in every campaign phase.
    assert current["final_test_status"] == "locked"
    assert current["final_panel_access_allowed"] is False
    assert current["final_unlock_allowed"] is False
    assert current["final_pool_consumed"] is False
    latest = current["latest_event"]
    assert latest["final_panel_access"] == 0
    assert latest["final_unlock_allowed"] is False
    assert latest["final_pool_consumed"] is False

    # Validate the immutable E1 seal itself, independent of current phase.
    evidence = e1_bundle.validate(ROOT / e1_bundle.BUNDLE)
    assert evidence["status"] == "PASS_EVIDENCE_BUNDLE_INTEGRITY"
    assert evidence["execution_authority"] is False
    assert evidence["scientific_efficacy_assessed"] is False

def test_results_index_advances_atomically_to_gen28_v28() -> None:
    index = load(INDEX)
    coverage = index["coverage"]
    assert coverage["last_valid_generation"] == 28
    assert coverage["valid_generation_count"] == 28
    assert coverage["valid_completed_episodes"] == 3584
    assert coverage["success_count"] == 9
    assert coverage["latest_native_state"]["state_version"] == 28
    assert coverage["latest_native_state"]["state_sha256"] == STATE_CHECKSUM
    gen28 = [r for r in index["results"] if r["generation"] == 28]
    assert len(gen28) == 1
    assert gen28[0]["valid_completed_episodes"] == 128
    assert gen28[0]["success_count"] == 2
    assert gen28[0]["state_update"]["state_version"] == 28
