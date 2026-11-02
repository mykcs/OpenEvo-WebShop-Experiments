import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/experiment/manifests/openevo-mechanism1-m1a-adjacent-probe-202609062200.json"
PROVENANCE = ROOT / "docs/evidence/server-runs/2026-09-06/mechanism1-m1a-adjacent-source/source-provenance-binding-202609062200.json"


def qform(x, gram):
    return sum(x[i] * gram[i][j] * x[j] for i in range(len(x)) for j in range(len(x)))


def test_preparation_remains_execution_locked():
    m = json.loads(MANIFEST.read_text())
    p = json.loads(PROVENANCE.read_text())
    assert m["status"] == "FROZEN_PREPARATION_EXECUTION_LOCKED"
    assert m["execution_authority"] is False
    assert m["formal_task_consumption_allowed"] is False
    assert p["execution_authority"] is False
    assert p["formal_task_consumption_allowed"] is False


def test_probe_panel_and_matrix_are_exactly_frozen():
    m = json.loads(MANIFEST.read_text())
    tasks = m["probe_panel"]["tasks"]
    assert len(tasks) == 64
    assert len({x["source_session_index"] for x in tasks}) == 64
    assert m["probe_panel"]["r27_to_r49_overlap_count"] == 0
    assert m["probe_panel"]["ceiling_final_panel_goal_overlap_count"] == 0
    assert m["probe_panel"]["ceiling_final_panel_instruction_hash_overlap_count"] == 0
    assert len(m["cells"]) == 9
    assert m["closed_loop_episode_cap"] == 576
    assert [x["probe_ordinal"] for x in tasks] == list(range(64))
    assert all(tasks[i]["paired_generation_seed"] < tasks[i + 1]["paired_generation_seed"] for i in range(63))


def test_random_controls_match_tau_in_exact_gram_metric():
    m = json.loads(MANIFEST.read_text())
    gram = m["parameter_basis"]["gram_matrix"]
    tau_norm = m["states"]["tau_gram_norm"]
    assert tau_norm > 0
    tau = m["states"]["tau_adjacent_late_coefficients"]
    assert math.isclose(math.sqrt(qform(tau, gram)), tau_norm, abs_tol=1e-12)
    assert [x["seed"] for x in m["random_directions"]] == [2026090501, 2026090502, 2026090503]
    for row in m["random_directions"]:
        assert math.isclose(math.sqrt(qform(row["direction_coefficients"], gram)), tau_norm, abs_tol=1e-12)


def test_provenance_binds_portable_source_identities():
    p = json.loads(PROVENANCE.read_text())
    assert p["source_capsule"]["archive_sha256"] == "add9eb6411c96d8d97924079b9e309a1ee898dbb3f4b125833caa56432ff61a7"
    assert p["source_capsule"]["manifest_sha256"] == "5865a0f0d63ec141f88023b6a3c3bbd34b20adbad19dfb91a81a059c7f93c8e8"
    assert p["terminal_seal"]["sha256"] == "8dba5235b884d1d01e62999fab6dbe18a28078c7cfd54cace5dc9b5a935d462c"
    assert p["component_store_immutable_snapshot"]["head"]["sha256"] == "4482cda6e7616aa3f91dd5b78d6ecf0a56abf38c50fb4d8cec672a2b817c75a4"
    assert p["component_store_immutable_snapshot"]["guard"]["sha256"] == "0f3ee306d28feff12d53702ce8b8c17781f6c6ab94fedabf5725870160fec63d"


def test_zero_task_runtime_qualification_and_preparation_receipt_remain_locked():
    q = json.loads((ROOT / "docs/evidence/server-runs/2026-09-06/mechanism1-m1a-activation-preparation/runtime-qualification-202609062200.json").read_text())
    r = json.loads((ROOT / "configs/experiment/receipts/openevo-mechanism1-m1a-activation-preparation-202609062200.json").read_text())
    assert q["status"] == "PASS" and q["cell_count"] == 9
    assert q["formal_webshop_tasks_consumed"] == 0
    assert q["formal_webshop_episodes_consumed"] == 0
    assert q["final_panel_access"] == 0 and q["external_teacher_calls"] == 0
    assert q["canonical_remote_provenance_seal_sha256"] == "a2948fb280d49c484f1864041d7ed9820c4b1bc03e028dcf381b8180979b6e6e"
    assert q["component_store_head_sha256"] == "4482cda6e7616aa3f91dd5b78d6ecf0a56abf38c50fb4d8cec672a2b817c75a4"
    assert q["execution_authority"] is False and q["formal_task_consumption_allowed"] is False
    assert r["status"] == "PASS_ACTIVATION_REQUIRED"
    assert r["execution_authority"] is False and r["formal_task_consumption_allowed"] is False


def test_task_selection_recomputes_from_frozen_schedule_without_outcome_choice():
    m = json.loads(MANIFEST.read_text())
    schedule = json.loads((ROOT / m["source_schedule"]["path"]).read_text())
    panel = json.loads((ROOT / m["ceiling_final_panel_exclusion"]["path"]).read_text())
    excluded_sessions = {int(x.split(":", 2)[1]) for x in m["r27_to_r49_update_exclusion"]["bound_task_identities"]}
    final_sessions = {int(x["goal_idx"]) for x in panel["tasks"]}
    final_hashes = {x["instruction_sha256"] for x in panel["tasks"]}
    expected = []
    seen = set()
    for round_row in schedule["stage2"]["rounds"]:
        for group in round_row["groups"]:
            sid = int(group["source_session_index"])
            if sid in seen:
                continue
            seen.add(sid)
            if sid in excluded_sessions or sid in final_sessions or group["instruction_sha256"] in final_hashes:
                continue
            expected.append(sid)
            if len(expected) == 64:
                break
        if len(expected) == 64:
            break
    assert expected == [x["source_session_index"] for x in m["probe_panel"]["tasks"]]


def test_preparation_hash_chain_is_closed():
    import hashlib

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    m_path = MANIFEST
    q_path = ROOT / "docs/evidence/server-runs/2026-09-06/mechanism1-m1a-activation-preparation/runtime-qualification-202609062200.json"
    r_path = ROOT / "configs/experiment/receipts/openevo-mechanism1-m1a-activation-preparation-202609062200.json"
    router_path = ROOT / "configs/experiment/campaigns/current-mechanism1-autonomous.json"
    q = json.loads(q_path.read_text())
    r = json.loads(r_path.read_text())
    router = json.loads(router_path.read_text())
    assert q["manifest_sha256"] == sha(m_path)
    assert q["experiment_git_sha"] == r["runtime_qualified_git_sha"]
    assert r["probe_manifest"]["sha256"] == sha(m_path)
    assert r["runtime_qualification"]["sha256"] == sha(q_path)
    seal_path = ROOT / r["canonical_remote_provenance_seal"]["path"]
    assert r["canonical_remote_provenance_seal"]["sha256"] == sha(seal_path)
    assert router["m1a_lineage"]["canonical_remote_provenance_seal_sha256"] == sha(seal_path)
    assert router["m1a_lineage"]["probe_manifest_sha256"] == sha(m_path)
    assert router["m1a_lineage"]["runtime_qualification_sha256"] == sha(q_path)
    assert router["m1a_lineage"]["activation_preparation_receipt_sha256"] == sha(r_path)


def test_canonical_remote_provenance_and_guard_roles_are_distinct_and_bound():
    import hashlib

    p = json.loads(PROVENANCE.read_text())
    binding = p["canonical_remote_provenance_seal"]
    seal_path = ROOT / binding["path"]
    assert hashlib.sha256(seal_path.read_bytes()).hexdigest() == "a2948fb280d49c484f1864041d7ed9820c4b1bc03e028dcf381b8180979b6e6e"
    assert binding["sha256"] == hashlib.sha256(seal_path.read_bytes()).hexdigest()
    roles = p["guard_identity_roles"]
    assert roles["shared_component_store_head_sha256"] == "4482cda6e7616aa3f91dd5b78d6ecf0a56abf38c50fb4d8cec672a2b817c75a4"
    assert roles["remote_archive_guard"]["sha256"] == "7365f732accb8392248d3dba0d051a95fee70e65bf865bb5589663fc4a57969e"
    assert roles["source_capsule_guard"]["sha256"] == "0f3ee306d28feff12d53702ce8b8c17781f6c6ab94fedabf5725870160fec63d"
    assert roles["qualification_live_guard"]["sha256"] == "50a8f4fb0eb25466e221068f232f1ce0027dcb10a82567d5ae5991d6a5ca6651"
    assert len({roles["remote_archive_guard"]["sha256"], roles["source_capsule_guard"]["sha256"], roles["qualification_live_guard"]["sha256"]}) == 3
