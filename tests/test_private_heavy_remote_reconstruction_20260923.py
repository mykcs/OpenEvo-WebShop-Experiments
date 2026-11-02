import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVID = ROOT / "docs/evidence/huggingface-asset-classification-20260923"
AGG = EVID / "PRIVATE_HEAVY_RECONSTRUCTION_ACCEPTANCE_202609230029.json"
LATE = EVID / "PRIVATE_HEAVY_LATE_REMOTE_RECONSTRUCTION_PROBE_202609230029.json"
HIST = EVID / "PRIVATE_HEAVY_HISTORICAL_REMOTE_RECONSTRUCTION_PROBE_202609230029.json"
DOC = ROOT / "docs/operations/publication/PRIVATE_HEAVY_REMOTE_RECONSTRUCTION_READBACK_20260923.md"
PLAN = ROOT / "docs/operations/publication/STAGE2_CHECKPOINT_ARCHIVE_DECOMPOSITION_PLAN_20260921.md"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_late_remote_probe_is_exact_but_bounded():
    x = load(LATE)
    assert x["status"] == "PASS"
    assert x["provider"]["repo_id"] == "openevo-recovery/openevo-ceiling1-v2-archive"  # legacy-revision-compat: historical HF repository identity
    assert x["provider"]["revision"] == "6849828d27d56f11423b0b417303564f4d36b2f6"
    assert set(map(int, x["rounds"])) == {144, 146}
    assert len(x["tensor_results"]) == 8
    assert all(r["max_abs_error"] == 0.0 and r["array_equal"] for r in x["tensor_results"])
    assert x["transport"]["http_range"] is True
    assert x["transport"]["bytes"] == 155_523_656
    assert any("not full 224-tensor" in s for s in x["limitations"])
    assert any("Does not authorize provider deletion" in s for s in x["limitations"])


def test_historical_remote_probe_spans_early_mid_late_and_is_exact():
    x = load(HIST)
    assert x["status"] == "PASS"
    assert x["target_provider"]["repo_id"] == "kyo-myk/openevo-ceiling-1.0-stage2-checkpoint-archive"
    assert x["target_provider"]["revision"] == "6b8d54a37703dc0325159cbfbfd7656a9f0aa304"
    assert x["component_provider"]["repo_id"] == "openevo-recovery/openevo-ceiling1-v2-archive"  # legacy-revision-compat: historical HF repository identity
    assert set(map(int, x["rounds"])) == {0, 50, 137}
    assert [x["rounds"][str(r)]["n"] for r in (0, 50, 137)] == [1, 45, 132]
    assert len(x["tensor_results"]) == 12
    assert all(r["max_abs_error"] == 0.0 and r["array_equal"] for r in x["tensor_results"])
    assert x["transport"]["bytes"] == 115_304_496


def test_aggregate_keeps_reconstruction_and_deletion_separate():
    x = load(AGG)
    assert x["status"] == "PASS_CURRENT_KEEP_RECONSTRUCTIBLE_BASIS_WITH_FRESH_REMOTE_TENSOR_READBACK"
    assert x["current_keep_reconstructible_logical_bytes"] == 46_994_469_425
    assert x["fresh_remote_tensor_readback_bytes"] == 46_929_725_624
    assert x["existing_non_tensor_reconstruction_binding_bytes"] == 64_743_801
    assert x["remote_probe_transport"]["total_bytes_read"] == 270_828_152
    assert x["server_local_recovery_source_check"]["observed_present"] is False
    assert x["serializer_byte_intent_review"]["lineage_state"] == "HISTORICAL_COMPLETE"
    assert x["serializer_byte_intent_review"]["resume_policy"] == "NO_RESUME_HISTORICAL"
    assert x["retention_conclusion"]["classification_change"] == "NONE"
    assert x["retention_conclusion"]["delete_authorized"] is False
    late = next(c for c in x["classes"] if c["class_id"] == "expanded_late_7b_r142_to_r146")
    assert "not present as same-path objects" in late["remote_snapshot_nuance"]


def test_human_docs_preserve_scope_and_route_to_successor_evidence():
    doc = DOC.read_text(encoding="utf-8")
    assert "PRIVATE ≠" not in doc  # this page should be a scoped readback, not redefine the top-level standard
    assert "46,929,725,624 B" in doc
    assert "12 / 12 tensors" in doc
    assert "8 / 8 tensors" in doc
    assert "max_abs_error = 0.0" in doc
    assert "没有授权删除" in doc
    assert "snapshot JSON" in doc
    plan = PLAN.read_text(encoding="utf-8")
    assert "PRIVATE_HEAVY_REMOTE_RECONSTRUCTION_READBACK_20260923.md" in plan
    assert "46,929,725,624 B" in plan
    assert "purpose-built PUBLIC repo published" in plan
