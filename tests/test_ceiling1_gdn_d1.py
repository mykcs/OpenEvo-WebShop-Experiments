from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "openevo_webshop" / "ceiling1_gdn_d1.py"
PREREG = ROOT / "docs/evidence/q17-final-closeout-20260912/frozen-inputs/af4be7750be48ca6c1b5c5d7f56ce56288c12123c2475e96fc7cf8ec225b98dd.json"  # legacy-revision-compat: exact executed prereg bytes
spec = importlib.util.spec_from_file_location("ceiling1_gdn_d1", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def prereg():
    return json.loads(PREREG.read_text())


def geometry(checks=None):
    return {
        "status": "PASS",
        "final_panel_access": 0,
        "external_teacher_calls_delta": 0,
        "new_environment_calls_delta": 0,
        "mechanism_checks": checks or {
            "telescoping_relative_error": True,
            "late_window_median_residual_energy": True,
            "task_vector_rank95_fraction": True,
            "primary_K64_parameter_relative_error": True,
        },
    }


def candidate(action=1.0, parse=1.0, js=1e-6, logit=0.01, nll=0.001):
    return {
        "greedy_action_exact_agreement_vs_control": action,
        "greedy_action_parse_rate": parse,
        "new_parse_invalid_rows": [],
        "top1_token_agreement_vs_control": 1.0,
        "next_token_js_mean": js,
        "centered_next_logit_relative_l2_mean": logit,
        "teacher_forced_target_nll_delta_vs_control": nll,
    }


def function(k64=None, k80=None, k96=None):
    results = {"CONTROL": candidate()}
    if k64 is not None:
        results["K64"] = k64
    if k80 is not None:
        results["K80"] = k80
    if k96 is not None:
        results["K96"] = k96
    return {
        "status": "PASS_MEASUREMENT_COMPLETE",
        "control_action_parse_rate": 1.0,
        "execution": {
            "training": False,
            "environment_calls": 0,
            "external_teacher_calls": 0,
            "final_panel_access": 0,
        },
        "results": results,
    }


def test_prereg_primary_and_qwen3_contract_are_frozen():
    p = prereg()
    assert p["geometry"]["bounded_candidates"]["primary"] == 64
    assert p["geometry"]["bounded_candidates"]["secondary"] == [80, 96]
    assert p["activation"]["minimum_accepted_parametric_updates_for_primary_test"] == 96
    assert p["probe"]["generation"]["arm_chat_template_policy"]["qwen3-1p7b"] == "explicit enable_thinking=false"
    assert p["classification"]["D1_PRIMARY_FAIL_SECONDARY_PRESERVATION_ONLY"]


def test_probe_selection_is_deterministic_and_task_balanced():
    rows = []
    for task in range(20):
        for step in (2, 0, 1):
            rows.append({
                "task_id": f"task-{task:02d}",
                "source_schedule_position": 100 + task,
                "step_idx": step,
                "trajectory_id": f"traj-{task}",
                "prompt_messages": [{"role": "user", "content": "x"}],
                "response_messages": [{"role": "assistant", "content": "y"}],
            })
    selected = mod.select_probe_rows(list(reversed(rows)), tasks=16, per_task=2)
    assert len(selected) == 32
    assert len({x["task_id"] for x in selected}) == 16
    assert [x["task_id"] for x in selected[::2]] == [f"task-{i:02d}" for i in range(16)]
    assert all(selected[i]["step_idx"] == 0 and selected[i + 1]["step_idx"] == 1 for i in range(0, 32, 2))


def test_primary_pass_requires_mechanism_and_k64_function_preservation():
    result = mod.classify_payload(prereg(), geometry(), function(k64=candidate(), k80=candidate(), k96=candidate()))
    assert result["classification"] == "D1_PRIMARY_PASS"
    assert result["primary_pass"] is True


def test_secondary_cannot_rescue_primary_claim():
    result = mod.classify_payload(
        prereg(),
        geometry(),
        function(k64=candidate(action=31 / 32), k80=candidate(), k96=candidate()),
    )
    assert result["classification"] == "D1_PRIMARY_FAIL_SECONDARY_PRESERVATION_ONLY"
    assert result["primary_pass"] is False


def test_mechanism_failure_precedes_function_result():
    checks = geometry()["mechanism_checks"]
    checks["late_window_median_residual_energy"] = False
    result = mod.classify_payload(prereg(), geometry(checks), function(k64=candidate()))
    assert result["classification"] == "D1_FAIL_MECHANISM_NOT_REPLICATED"


def test_insufficient_updates_is_not_scientific_failure():
    result = mod.classify_payload(
        prereg(),
        {"status": "D1_INSUFFICIENT_PARAMETER_UPDATES_NOT_A_FAILURE"},
        None,
    )
    assert result["classification"] == "D1_INSUFFICIENT_PARAMETER_UPDATES_NOT_A_FAILURE"
    assert result["primary_pass"] is False


def test_provenance_violation_fails_closed():
    g = geometry()
    g["final_panel_access"] = 1
    result = mod.classify_payload(prereg(), g, function(k64=candidate()))
    assert result["classification"] == "D1_INVALID_PROVENANCE"
