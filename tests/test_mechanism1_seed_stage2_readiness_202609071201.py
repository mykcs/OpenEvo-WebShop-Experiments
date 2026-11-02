from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTER = ROOT / "configs/experiment/campaigns/current-mechanism1-autonomous.json"
DESIGN = ROOT / "configs/experiment/designs/openevo-mechanism1-seed-stage2-minimax-sft-successor-202609070053.json"
MINI = ROOT / "docs/evidence/mechanism1/m1d-minimax-closeout-202609071201.json"
DATA = ROOT / "docs/evidence/mechanism1/m1d-stage2-sft-data-ready-202609071201.json"
RUNTIME = ROOT / "docs/evidence/mechanism1/m1d-stage2-runtime-qualification-202609071201.json"
RESOURCE = ROOT / "configs/experiment/amendments/openevo-mechanism1-seed-stage2-gpu01234567-resource-202609071201.json"
PREP = ROOT / "configs/experiment/receipts/openevo-mechanism1-seed-stage2-sft-preparation-202609071201.json"
CURRENT = ROOT / "docs/science/webshop/program/OPEN_EVO_MECHANISM_1.0_CURRENT.md"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_m1d_is_closed_after_exact_minimax_seal():
    router = load(ROUTER)
    m1d = router["m1d_readiness"]
    mini = load(MINI)
    assert mini["status"] == "PASS"
    assert mini["records"] == mini["parse_ok"] == mini["unique_trajectory_ids"] == 1440
    assert mini["parse_failures"] == 0
    assert mini["minimax_output_sha256"] == "6bc91622eeb2e4a5ef5887a6fab5595cd0ed298aeca46690eba7bb3648bcee92"
    assert m1d["activation_authority"] is False
    assert m1d["formal_task_consumption_allowed"] is False
    assert m1d["minimax_closeout_evidence_sha256"] == sha(MINI)


def test_sft_data_and_runtime_are_sealed_without_task_consumption():
    data = load(DATA)
    runtime = load(RUNTIME)
    assert data["status"] == "PASS"
    assert (data["candidate_rows"], data["sft_records"], data["train_rows"], data["validation_rows"]) == (1440, 1440, 1296, 144)
    assert data["new_teacher_calls"] == 0 and data["new_webshop_calls"] == 0
    assert runtime["status"] == "PASS" and runtime["qualification_only"] is True
    assert runtime["formal_stage2_episodes"] == runtime["webshop_actions"] == runtime["optimizer_steps"] == 0
    assert runtime["python"] == "3.10.12"
    assert runtime["packages"]["torch"] == "2.13.0+cu130"
    assert runtime["packages"]["vllm"] == "0.27.1"
    assert runtime["packages"]["ray"] == "2.49.2"


def test_resource_amendment_is_prospective_eight_gpu_and_no_reshard():
    doc = load(RESOURCE)
    assert doc["status"] == "PASS_PROSPECTIVE_NO_TRAINING_CONSUMED"
    assert doc["scientific_algorithm_contract_changed"] is False
    assert doc["resource_execution_contract_changed"] is True
    assert doc["n_gpus_per_node"] == 8
    assert doc["no_mid_run_reshard"] is True
    assert [row["index"] for row in doc["gpus"]] == list(range(8))
    assert len({row["uuid"] for row in doc["gpus"]}) == 8
    assert doc["required_free_mib_each_at_launch"] == 30000
    assert "Do not delete, prune, stop, rename" in doc["docker_safety"]


def test_offline_sft_is_authorized_while_stage2_stays_locked():
    router = load(ROUTER)
    succ = router["m1d_stage2_successor"]
    prep = load(PREP)
    assert succ["offline_sft_preparation_authority"] is True
    assert succ["activation_authority"] is False
    assert succ["formal_task_consumption_allowed"] is False
    assert succ["status"] == "SFT_PREPARATION_AUTHORIZED_STAGE2_LOCKED"
    assert succ["minimax_closeout_evidence_sha256"] == sha(MINI)
    assert succ["sft_data_evidence_sha256"] == sha(DATA)
    assert succ["runtime_qualification_evidence_sha256"] == sha(RUNTIME)
    assert succ["resource_amendment_sha256"] == sha(RESOURCE)
    assert succ["sft_preparation_receipt_sha256"] == sha(PREP)
    assert prep["offline_sft_preparation_authority"] is True
    assert prep["stage2_activation_authority"] is False
    assert prep["formal_task_consumption_allowed"] is False
    assert prep["sft_contract"]["nproc_per_node"] == 8
    assert prep["sft_contract"]["epochs"] == 3


def test_original_successor_design_remains_inactive_and_current_doc_projects_machine_state():
    design = load(DESIGN)
    assert design["activation_authority"] is False
    assert design["formal_task_consumption_allowed"] is False
    text = CURRENT.read_text(encoding="utf-8")
    assert "M1-D itself is closed" in text
    assert "Offline SFT preparation is authorized" in text
    assert "Stage2 WebShop task consumption remains locked" in text
    assert "GPU0-7" in text
