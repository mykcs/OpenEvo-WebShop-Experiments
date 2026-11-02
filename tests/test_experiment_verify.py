from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

from scripts.openevo_webshop import experiment_verify as ev
from scripts.openevo_webshop.experiment_passport import load_passport

ROOT = Path(__file__).resolve().parents[1]
Q17 = ROOT / "configs/experiment/passports/202609082022-q17-directapply-no-gdr/passport.json"


def _q17() -> dict:
    return load_passport(Q17, root=ROOT)


def test_q17_offserver_verify_is_identity_pass_not_live_failure() -> None:
    report = ev.verify_experiment(ROOT, "qwen3-1p7b-directapply-no-gdr")
    assert report["identity"] == "PASS"
    assert report["layers"]["PASSPORT"]["status"] == "PASS"
    assert report["layers"]["REGISTRY"]["status"] == "PASS"
    assert report["layers"]["STATIC_ASSETS"]["status"] == "PASS"
    assert report["layers"]["LIVE_STATE"]["status"] == "NOT_OBSERVED"
    assert report["layers"]["EXEC_LINEAGE"]["status"] == "NOT_APPLICABLE"
    assert report["overall"] == "MANUAL_REVIEW_REQUIRED"
    assert report["safe_to_consider_continuation"] == "MANUAL_REVIEW"


def test_legacy_no_resume_is_never_consider_continuation() -> None:
    report = ev.verify_experiment(ROOT, "qwen25-3b-legacy-stage2")
    assert report["identity"] == "PASS"
    assert report["resume_policy"] == "NO_RESUME"
    assert report["safe_to_consider_continuation"] == "NO"


def test_wrong_model_revision_fails_static_identity() -> None:
    passport = copy.deepcopy(_q17())
    passport["scientific_identity"]["model"]["model_revision"] = "f" * 40
    result = ev._verify_static_assets(ROOT, passport)
    assert result["status"] == "FAIL"
    assert any(d["location"] == "semantic-invariant" and d["status"] == "FAIL" for d in result["details"])


def test_wrong_runtime_digest_fails_static_identity() -> None:
    passport = copy.deepcopy(_q17())
    passport["initial_execution_identity"]["runtime_image_digest"] = "sha256:" + "0" * 64
    result = ev._verify_static_assets(ROOT, passport)
    assert result["status"] == "FAIL"
    assert any(d["location"] == "runtime-image" and d["status"] == "FAIL" for d in result["details"])


def test_wrong_seed_identity_fails_static_identity() -> None:
    passport = copy.deepcopy(_q17())
    passport["scientific_identity"]["seed_policy"]["schedule"]["sha256"] = "0" * 64
    result = ev._verify_static_assets(ROOT, passport)
    assert result["status"] == "FAIL"


def test_wrong_successor_receipt_fails_lineage() -> None:
    initial = "1" * 40
    current = "3" * 40
    wrong = [
        {"previous": initial, "successor": "2" * 40, "science_changed": False},
        {"previous": "9" * 40, "successor": current, "science_changed": False},
    ]
    result = ev.verify_execution_lineage_graph(initial, current, wrong)
    assert result["status"] == "FAIL"


def test_science_changing_successor_is_not_accepted() -> None:
    initial = "1" * 40
    current = "2" * 40
    result = ev.verify_execution_lineage_graph(
        initial,
        current,
        [{"previous": initial, "successor": current, "science_changed": True}],
    )
    assert result["status"] == "FAIL"


def test_verify_only_spawns_git_offserver(monkeypatch) -> None:
    calls: list[list[str]] = []
    original = ev.subprocess.run

    def wrapped(cmd, *args, **kwargs):
        calls.append([str(x) for x in cmd])
        return original(cmd, *args, **kwargs)

    monkeypatch.setattr(ev.subprocess, "run", wrapped)
    report = ev.verify_experiment(ROOT, "qwen3-1p7b-directapply-no-gdr")
    assert report["identity"] == "PASS"
    assert calls
    assert {call[0] for call in calls} == {"git"}


def test_oe_verify_cli_human_and_json() -> None:
    human = subprocess.run(
        [sys.executable, str(ROOT / "scripts/oe"), "experiments", "verify", "qwen3-1p7b-directapply-no-gdr"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert human.returncode == 0, human.stderr
    assert "IDENTITY: PASS" in human.stdout
    assert "SAFE_TO_CONSIDER_CONTINUATION: MANUAL_REVIEW" in human.stdout
    machine = subprocess.run(
        [sys.executable, str(ROOT / "scripts/oe"), "experiments", "verify", "qwen3-1p7b-directapply-no-gdr", "--json"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert machine.returncode == 0, machine.stderr
    payload = json.loads(machine.stdout)
    assert payload["mutation"] is False
    assert payload["layers"]["LIVE_STATE"]["status"] == "NOT_OBSERVED"


def test_composite_recovery_requires_exact_same_science_binding(tmp_path: Path) -> None:
    initial, intermediate, current = "1" * 40, "2" * 40, "3" * 40
    maintenance = tmp_path / "maintenance"; bundle = maintenance / "bundle"
    bundle.mkdir(parents=True)
    receipt = tmp_path / "recovery.json"
    receipt.write_text(json.dumps({"status":"AUTHORIZED_RETRY","scientific_semantics_changed":False,
                                   "formal_rollout_replay":0,"previous_control_sha":initial,
                                   "successor_control_sha":intermediate})+"\n")
    binding = maintenance / "EXACT_HEAD_BINDING.json"
    binding.write_text(json.dumps({"status":"PASS","scientific_semantics_changed":False,
                                   "previous_control_sha":initial,"successor_control_sha":current})+"\n")
    run = tmp_path / "run"; (run / "round-000").mkdir(parents=True); (run / "round-001").mkdir(parents=True)
    (run / "round-000/state-after.json").write_text(json.dumps({"twin_control_git_sha":initial})+"\n")
    (run / "round-001/state-after.json").write_text(json.dumps({"twin_control_git_sha":current})+"\n")
    passport = {"initial_execution_identity":{"initial_experiment_sha":initial,
                                                "initial_run_root":str(run),
                                                "activation":{"path":str(bundle / "activation.json")}}}
    live = {"twin_control_git_sha":current,"rounds_complete":[0,1],"controller_recoveries":[{
        "previous_control_sha":initial,"successor_control_sha":current,
        "scientific_semantics_changed":False,"formal_rollout_replay":0,
        "recovery_receipt":str(receipt),"recovery_receipt_sha256":ev._sha256_path(receipt),
        "recovery_kind":"COMPOSITE_TEST"}]}
    result = ev._verify_execution_lineage(ROOT, passport, live)
    assert result["status"] == "PASS"
    binding.unlink()
    result = ev._verify_execution_lineage(ROOT, passport, live)
    assert result["status"] == "FAIL"
    assert "exact same-science binding" in result["reason"]


def test_sealed_registered_run_is_identity_pass_but_not_resumable(monkeypatch) -> None:
    monkeypatch.setattr(ev, "_verify_live_state", lambda entry, passport: (
        ev._layer("PASS", "sealed", lifecycle_status="STAGE2_SEALED"),
        {"status":"STAGE2_SEALED","twin_control_git_sha":passport["initial_execution_identity"]["initial_experiment_sha"]}))
    report = ev.verify_experiment(ROOT, "qwen3-1p7b-directapply-no-gdr")
    assert report["identity"] == "PASS"
    assert report["overall"] == "PASS"
    assert report["safe_to_consider_continuation"] == "NO"


def test_receipt_pairs_do_not_cross_nested_recoveries() -> None:
    doc = {
        "controller_recoveries": [
            {"previous_control_sha":"1"*40, "successor_control_sha":"2"*40},
            {"previous_control_sha":"3"*40, "successor_control_sha":"4"*40},
        ]
    }
    pairs = set(ev._receipt_pairs(doc))
    assert pairs == {("1"*40, "2"*40), ("3"*40, "4"*40)}
    assert ("1"*40, "4"*40) not in pairs


def test_sealed_boundary_can_bind_authorized_successor_without_git_object(tmp_path: Path) -> None:
    initial, successor = "1" * 40, "2" * 40
    run = tmp_path / "run"; (run / "round-000").mkdir(parents=True); (run / "round-001").mkdir(parents=True)
    shared = {"state_id":"a"*64,"rollouts_consumed":128,"next_round":1,
              "external_teacher_calls":0,"final_panel_access_count":0}
    (run/"round-000/state-after.json").write_text(json.dumps({**shared,"twin_control_git_sha":initial})+"\n")
    (run/"round-001/state-before.json").write_text(json.dumps({**shared,"twin_control_git_sha":successor})+"\n")
    passport={"initial_execution_identity":{"initial_run_root":str(run)},"budget_contract":{"round_limit":2}}
    proof=ev._sealed_boundary_predecessor(passport,successor,{initial})
    assert proof is not None
    assert proof["previous"] == initial and proof["successor"] == successor
    assert "sealed boundary" in proof["proof"]


def test_actual_execution_sequence_requires_every_used_transition() -> None:
    initial, middle, current = "1"*40, "2"*40, "3"*40
    controls = [initial, middle, current]
    good = [
        {"previous":initial,"successor":middle,"science_changed":False,"evidence":"a"},
        {"previous":middle,"successor":current,"science_changed":False,"evidence":"b"},
        {"previous":initial,"successor":current,"science_changed":False,"evidence":"shortcut"},
    ]
    result = ev.verify_actual_execution_sequence(initial,current,controls,good)
    assert result["status"] == "PASS" and result["transition_count"] == 2
    missing = ev.verify_actual_execution_sequence(initial,current,controls,[good[-1]])
    assert missing["status"] == "FAIL"
    assert missing["previous"] == initial and missing["successor"] == middle


def test_verifier_subprocess_surface_is_git_only() -> None:
    import ast, inspect
    tree = ast.parse(inspect.getsource(ev))
    commands = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "run" and isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess":
                commands.append(node.args[0])
    assert commands
    for command in commands:
        assert isinstance(command, ast.List)
        first = command.elts[0]
        assert isinstance(first, ast.Constant) and first.value == "git"
