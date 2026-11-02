"""Strict, zero-replay import of sealed predecessor rollout evidence."""
from pathlib import Path
import hashlib, json

SCHEMA = 'openevo.bounded-gdr.round0-rollout-reuse-recovery.20260916'
SEALED_REUSE_SCHEMA = 'openevo.bounded-gdr.sealed-rollout-reuse-recovery.v2'  # legacy-revision-compat: absorbed historical runtime literal from a6f67b
CONTROL_REBIND_KEYS = {'formal_ready_package_sha256', 'owner_launch_release_sha256', 'carrier_adoption_identity_sha256'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_manifest(package):
    ref = package.get('recovery')
    if not ref: return None
    path = Path(ref['amendment_path'])
    if not path.is_file() or sha(path) != ref['sha256']:
        raise RuntimeError('recovery amendment missing or hash drift')
    doc = json.loads(path.read_text())
    if doc.get('status') != 'AUTHORIZED' or doc.get('schema') not in {SCHEMA, SEALED_REUSE_SCHEMA}:
        raise RuntimeError('recovery schema/status drift')
    if doc.get('successor_execution_git_sha') != package['execution_git_sha']:
        raise RuntimeError('recovery successor SHA drift')
    if doc.get('target_pair_root') != package['pair_output_root']:
        raise RuntimeError('recovery target root drift')
    if doc.get('formal_rollout_replay') != 0 or doc.get('final_panel_access_delta') != 0 or doc.get('scientific_contract_changed') is not False:
        raise RuntimeError('recovery protected boundary drift')
    if doc['schema'] == SCHEMA:
        if doc.get('historical_rollout_reuse') != {'0': [doc['predecessor_execution_git_sha']]}:
            raise RuntimeError('recovery producer allowlist must be exactly Round0')
    else:
        reuse = doc.get('reused_rollouts')
        if not isinstance(reuse, dict) or not reuse:
            raise RuntimeError('sealed-rollout recovery reuse map missing')
        for arm, rounds in reuse.items():
            if arm not in {'off','on'} or not isinstance(rounds, dict) or not rounds:
                raise RuntimeError('sealed-rollout recovery arm map invalid')
            for round_key, item in rounds.items():
                if str(int(round_key)) != round_key or int(round_key) < 0 or not isinstance(item, dict):
                    raise RuntimeError('sealed-rollout recovery round identity invalid')
                if item.get('attempt_count') != 128 or not item.get('producer_execution_git_sha'):
                    raise RuntimeError('sealed-rollout recovery denominator/producer drift')
                files = item.get('files')
                if not isinstance(files, dict):
                    raise RuntimeError('sealed-rollout recovery file map missing')
                required = {'state-before.json','summary.json','attempts.jsonl','ROLLOUT_PHASE_COMPLETE.json','carrier-scope-round-binding.json'}
                if not required.issubset(files):
                    raise RuntimeError('sealed-rollout recovery manifest incomplete')
        target_root = Path(doc['target_pair_root'])
        successor = doc['successor_execution_git_sha']
        for arm, rounds in reuse.items():
            for round_key, item in rounds.items():
                r = int(round_key)
                phase_path = target_root / arm / f'round-{r:03d}' / 'ROLLOUT_PHASE_COMPLETE.json'
                if not phase_path.is_file():
                    raise RuntimeError('sealed-rollout recovery declared round is not sealed in target root')
                phase = json.loads(phase_path.read_text())
                if phase.get('status') != 'PASS' or int(phase.get('round_index', -1)) != r:
                    raise RuntimeError('sealed-rollout recovery declared phase identity drift')
                if phase.get('execution_git_sha') != item.get('producer_execution_git_sha'):
                    raise RuntimeError('sealed-rollout recovery declared producer identity drift')
        for arm in ('off', 'on'):
            arm_root = target_root / arm
            if not arm_root.is_dir():
                continue
            for rr in sorted(arm_root.glob('round-*')):
                phase_path = rr / 'ROLLOUT_PHASE_COMPLETE.json'
                if not phase_path.is_file():
                    continue
                try:
                    r = int(rr.name.removeprefix('round-'))
                except ValueError as exc:
                    raise RuntimeError('sealed-rollout recovery target round identity invalid') from exc
                phase = json.loads(phase_path.read_text())
                producer = phase.get('execution_git_sha')
                if producer and producer != successor:
                    item = ((reuse.get(arm) or {}).get(str(r)))
                    if not isinstance(item, dict) or item.get('producer_execution_git_sha') != producer:
                        raise RuntimeError('sealed-rollout recovery missing historical producer coverage')
    return doc

def _sealed_reuse_item(doc, arm, r):
    if not doc or doc.get('schema') != SEALED_REUSE_SCHEMA:
        return None
    return ((doc.get('reused_rollouts') or {}).get(arm) or {}).get(str(r))

def validate_import(package, arm, r, rr):
    doc = validate_manifest(package)
    if not doc:
        return {package['execution_git_sha']}
    if doc['schema'] == SCHEMA:
        if r != 0: return {package['execution_git_sha']}
        item = doc['arms'][arm]
        source = Path(doc['source_pair_root']) / arm / 'round-000'
        targets = lambda rel: (source / rel, Path(rr) / rel)
        producer = doc['predecessor_execution_git_sha']
    else:
        item = _sealed_reuse_item(doc, arm, r)
        if item is None:
            return {package['execution_git_sha']}
        targets = lambda rel: (Path(rr) / rel,)
        producer = item['producer_execution_git_sha']
    if item['attempt_count'] != 128: raise RuntimeError('recovery denominator drift')
    files = item['files']
    for name in ('state-before.json', 'summary.json', 'attempts.jsonl', 'ROLLOUT_PHASE_COMPLETE.json', 'carrier-scope-round-binding.json'):
        if name not in files: raise RuntimeError('recovery manifest incomplete')
    for name, digest in files.items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts: raise RuntimeError('unsafe recovery path')
        for fp in targets(rel):
            if not fp.is_file() or sha(fp) != digest: raise RuntimeError(f'recovery payload drift: {fp}')
    phase = json.loads((Path(rr)/'ROLLOUT_PHASE_COMPLETE.json').read_text())
    if phase.get('status') != 'PASS' or int(phase.get('round_index', -1)) != r or phase.get('execution_git_sha') != producer:
        raise RuntimeError('recovery rollout phase producer identity drift')
    rows = [json.loads(line) for line in (Path(rr)/'attempts.jsonl').read_text().splitlines() if line.strip()]
    if len(rows) != 128 or {row.get('experiment_git_sha') for row in rows} != {producer}:
        raise RuntimeError('recovery attempt producer identity drift')
    return {producer}

def _science_projection_without_control_rebind(state, base):
    return {k: v for k, v in base.round_science_state(state).items() if k not in CONTROL_REBIND_KEYS}

def _partial_round_has_science_side_effects(rr: Path) -> bool:
    for name in ("attempts.jsonl", "summary.json", "ROLLOUT_PHASE_COMPLETE.json"):
        if (rr / name).exists():
            return True
    episodes = rr / "episodes"
    if episodes.is_dir() and any(episodes.iterdir()):
        return True
    attempts = rr / "container-attempts" / "rollout"
    if attempts.is_dir():
        for fp in attempts.iterdir():
            if fp.name.startswith("ray-selection-"):
                continue
            return True
    return False

def _control_rebind_receipt_path(rr: Path, state: dict) -> tuple[Path, dict, str]:
    successor_control = {k: state.get(k) for k in sorted(CONTROL_REBIND_KEYS)}
    if not all(successor_control.values()):
        raise RuntimeError("partial round successor control identity missing")
    control_binding_sha256 = hashlib.sha256(
        json.dumps(successor_control, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return rr / f"PARTIAL_ROUND_CONTROL_REBIND-{control_binding_sha256[:12]}.json", successor_control, control_binding_sha256

def _science_projection_sha256(state: dict, base) -> str:
    return hashlib.sha256(
        json.dumps(
            _science_projection_without_control_rebind(state, base),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

def _validate_existing_partial_control_rebind(rr: Path, arm: str, r: int, before: Path, state: dict, base) -> bool:
    receipt_path, successor_control, control_binding_sha256 = _control_rebind_receipt_path(rr, state)
    if not receipt_path.is_file():
        return False
    receipt = base.load(receipt_path)
    expected = {
        "schema": "openevo.bounded-gdr.partial-round-control-rebind-acceptance.20260918",
        "status": "PASS_CONTROL_ONLY",
        "arm": arm,
        "round_index": r,
        "state_before_sha256": sha(before),
        "science_projection_sha256": _science_projection_sha256(state, base),
        "control_binding_sha256": control_binding_sha256,
        "successor_control_values": successor_control,
        "state_before_rewritten": False,
        "scientific_state_mutated": False,
        "formal_rollout_replay": 0,
        "additional_model_calls": 0,
        "final_panel_access": 0,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"partial round control-rebind receipt drift: {key}")
    prior = base.load(before)
    if receipt.get("preserved_control_values") != {k: prior.get(k) for k in sorted(CONTROL_REBIND_KEYS)}:
        raise RuntimeError("partial round control-rebind preserved control identity drift")
    if _science_projection_without_control_rebind(prior, base) != _science_projection_without_control_rebind(state, base):
        raise RuntimeError("partial round control-rebind science projection drift")
    return True

def ensure_before(package, arm, r, rr, state, base):
    doc = validate_manifest(package) if package.get("recovery") else None
    reused = bool(doc and ((doc.get("schema") == SCHEMA and r == 0) or _sealed_reuse_item(doc, arm, r) is not None))
    before = rr / "state-before.json"
    if reused and before.is_file():
        validate_import(package, arm, r, rr)
        prior = base.load(before)
        if _science_projection_without_control_rebind(prior, base) != _science_projection_without_control_rebind(state, base):
            raise RuntimeError("recovery scientific state-before drift")
        return
    if before.is_file():
        if _validate_existing_partial_control_rebind(rr, arm, r, before, state, base):
            return
        prior = base.load(before)
        same_science = (
            _science_projection_without_control_rebind(prior, base)
            == _science_projection_without_control_rebind(state, base)
        )
        control_changed = any(prior.get(k) != state.get(k) for k in CONTROL_REBIND_KEYS)
        if same_science and control_changed and not _partial_round_has_science_side_effects(rr):
            receipt_path, successor_control, control_binding_sha256 = _control_rebind_receipt_path(rr, state)
            receipt = {
                "schema": "openevo.bounded-gdr.partial-round-control-rebind-acceptance.20260918",
                "status": "PASS_CONTROL_ONLY",
                "arm": arm,
                "round_index": r,
                "state_before_sha256": sha(before),
                "science_projection_sha256": _science_projection_sha256(state, base),
                "control_binding_sha256": control_binding_sha256,
                "preserved_control_values": {k: prior.get(k) for k in sorted(CONTROL_REBIND_KEYS)},
                "successor_control_values": successor_control,
                "state_before_rewritten": False,
                "scientific_state_mutated": False,
                "formal_rollout_replay": 0,
                "additional_model_calls": 0,
                "final_panel_access": 0,
            }
            base.write_once_json(receipt_path, receipt)
            return
    base.ensure_round_state_before(before, state)
from pathlib import Path
import hashlib, json, os

POST_ADVISOR_RECOVERY_SCHEMA = "openevo.post-advisor.e1-engineering-continuation.202609301000"
POST_ADVISOR_A8_RECOVERY_SCHEMA = "openevo.post-advisor.e1-engineering-continuation.a8.202609301200"
POST_ADVISOR_A7_EXECUTION_SHA = "556939a584462805ec2d2f0493b4ff215563768a"
POST_ADVISOR_A7_ACTIVATION_SHA = "a37cf149cacc0423cc9da93f602d73cb68d59fd5"
POST_ADVISOR_A9_RECOVERY_SCHEMA = "openevo.post-advisor.e1-engineering-continuation.a9.202609301600"
POST_ADVISOR_A10_RECOVERY_SCHEMA = "openevo.post-advisor.e1-engineering-continuation.a10.202609301900"
POST_ADVISOR_A9_EXECUTION_SHA = "00fdd0092a39e28ba2271c72bc5e113e30a0837e"
POST_ADVISOR_A9_ACTIVATION_SHA = "d3f8b26e9ece60872c6a2fca02b3525c9b26f38a"
POST_ADVISOR_A8_EXECUTION_SHA = "31a669ab9af34fdf5733849ea8ccfb7af37e7427"
POST_ADVISOR_A8_ACTIVATION_SHA = "a67580323e44ae82945e6ceb20da220a887cef4a"
POST_ADVISOR_A11_RECOVERY_SCHEMA = "openevo.post-advisor.e1-terminal-seal-recovery.a11.202609302100"
POST_ADVISOR_A10_EXECUTION_SHA = "3181cb7440d48633685d9c675a1f4684e37500ce"
POST_ADVISOR_A10_ACTIVATION_SHA = "7c79c8906e37486c2125d466b3c53ebe92ab987a"
POST_ADVISOR_R159_PAUSE_SHA = "600eca8272bfafaf946936699b6ffd91ab0233a17e5dd681e2bf0c1d1afb04a1"
POST_ADVISOR_R152_PAUSE_SHA = "27a99e987a256e2658a903f8bf3753a905672a4f76e4ae84fa14f9733bed04e6"

def _post_advisor_recovery_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def validate_a9_resolution(authority: dict, doc: dict) -> dict:
    from scripts.openevo_webshop.carrier_health_failure_policy_v1 import validate_resolution_receipt
    from scripts.openevo_webshop import ceiling1_stage2_arm_controller as base
    ref = authority.get("carrier_health_resolution") or {}
    path = Path(str(ref.get("path") or ""))
    if (not path.is_file() or path != Path(str(doc.get("resolution_receipt_path")))
            or _post_advisor_recovery_sha(path) != ref.get("sha256")
            or ref.get("sha256") != doc.get("resolution_receipt_sha256")):
        raise RuntimeError("A9 resolution path/SHA drift")
    receipt = json.loads(path.read_text())
    policy_path = Path(str(doc.get("carrier_health_policy_path") or ""))
    pause_path = Path(str(doc.get("pause_receipt_path") or ""))
    if (not policy_path.is_file() or _post_advisor_recovery_sha(policy_path) != doc.get("carrier_health_policy_sha256")
            or not pause_path.is_file() or _post_advisor_recovery_sha(pause_path) != POST_ADVISOR_R152_PAUSE_SHA):
        raise RuntimeError("A9 carrier policy/pause bytes drift")
    policy = base.load_carrier_health_policy(policy_path)
    validate_resolution_receipt(policy, receipt, expected_arm="qwen3-1p7b", expected_round=152,
                                expected_pause_receipt_sha256=POST_ADVISOR_R152_PAUSE_SHA)
    pause = json.loads(pause_path.read_text())
    expected = {"experiment_arm_id": "BOUNDED_OFF", "model_arm": "qwen3-1p7b",
                "policy_sha256": doc["carrier_health_policy_sha256"],
                "health_ledger_sha256": pause["health_ledger_sha256"],
                "boundary_decision_sha256": pause["boundary_decision_sha256"],
                "round_complete_sha256": pause["round_complete_sha256"],
                "state_after_sha256": pause["state_after_sha256"],
                "owner_authorization": "Owner explicitly authorized same-contract R152 diagnostic resolution and R153 continuation in this conversation."}
    if (pause.get("arm") != "BOUNDED_OFF" or pause.get("next_round") != 153
            or pause.get("formal_transaction_preserved") is not True
            or any(receipt.get(key) != value for key, value in expected.items())):
        raise RuntimeError("A9 resolution dual-arm/diagnostic/owner authorization drift")
    return receipt

def _validate_post_advisor_a9(authority: dict, doc: dict, path: Path, *, check_live_state: bool) -> dict:
    from scripts.openevo_webshop import ceiling1_stage2_arm_controller as base
    is_a10 = doc.get("schema") == POST_ADVISOR_A10_RECOVERY_SCHEMA
    expected_predecessor = POST_ADVISOR_A9_EXECUTION_SHA if is_a10 else POST_ADVISOR_A8_EXECUTION_SHA
    expected_activation = POST_ADVISOR_A9_ACTIVATION_SHA if is_a10 else POST_ADVISOR_A8_ACTIVATION_SHA
    if (doc.get("status") != "AUTHORIZED_ENGINEERING_CONTINUATION"
            or doc.get("successor_execution_git_sha") != authority.get("execution_git_sha")
            or doc.get("target_output_root") != authority.get("output_root")
            or doc.get("predecessor_execution_git_sha") != "6b4ff4503adc9494c949f654521f09ee52ec28b4"
            or doc.get("engineering_predecessor_execution_git_sha") != expected_predecessor
            or doc.get("engineering_predecessor_activation_git_sha") != expected_activation
            or doc.get("transition_output_parent_name") != "a8-transition-output"
            or doc.get("consumed_rollouts_before_recovery") != 128
            or doc.get("remaining_rollout_budget") != 896
            or doc.get("formal_rollout_replay") != 0
            or doc.get("final_panel_access_delta") != 0
            or doc.get("external_teacher_calls_delta") != 0
            or doc.get("scientific_contract_changed") is not False):
        raise RuntimeError("A9 execution lineage, budget, or frozen contract drift")
    previous = Path(str(doc.get("engineering_predecessor_recovery_path") or ""))
    if not previous.is_file() or _post_advisor_recovery_sha(previous) != doc.get("engineering_predecessor_recovery_sha256"):
        raise RuntimeError("A9 A8 authority drift")
    previous_doc = json.loads(previous.read_text())
    if (previous_doc.get("schema") != (POST_ADVISOR_A9_RECOVERY_SCHEMA if is_a10 else POST_ADVISOR_A8_RECOVERY_SCHEMA)
            or previous_doc.get("successor_execution_git_sha") != expected_predecessor):
        raise RuntimeError("post-advisor engineering predecessor semantics drift")
    for key in ("predecessor_arm_authority_sha256", "predecessor_arm_authority_snapshot",
                "predecessor_replay_sha256", "predecessor_state_id", "predecessor_formal_start_path",
                "predecessor_formal_start_sha256"):
        if doc.get(key) != previous_doc.get(key):
            raise RuntimeError(f"A9 inherited A6 provenance drift: {key}")
    predecessor_names = ("engineering_predecessor_formal_start", "engineering_predecessor_launch_receipt",
                         "engineering_predecessor_formal_log") if is_a10 else (
                             "engineering_predecessor_formal_start", "engineering_predecessor_handoff")
    for name in predecessor_names:
        fp = Path(str(doc.get(name + "_path") or ""))
        if not fp.is_file() or _post_advisor_recovery_sha(fp) != doc.get(name + "_sha256"):
            raise RuntimeError(f"A9 A8 {name} drift")
    start = json.loads(Path(doc["engineering_predecessor_formal_start_path"]).read_text())
    if start.get("status") != "FORMAL_START_COMMITTED" or start.get("formal_task_consumption_may_begin") is not True:
        raise RuntimeError("post-advisor predecessor Formal Start semantics drift")
    if is_a10:
        failed = json.loads(Path(doc["engineering_predecessor_launch_receipt_path"]).read_text())
        failed_log = Path(doc["engineering_predecessor_formal_log_path"]).read_text()
        start_path = Path(doc["engineering_predecessor_formal_start_path"])
        launch_path = Path(doc["engineering_predecessor_launch_receipt_path"])
        log_path = Path(doc["engineering_predecessor_formal_log_path"])
        start_ref = failed.get("formal_start") or {}
        if (start_ref.get("path") != str(start_path)
                or start_ref.get("sha256") != "sha256:" + doc["engineering_predecessor_formal_start_sha256"]
                or start_ref.get("receipt_sha256") != start.get("receipt_sha256")
                or start.get("output_root") != str(launch_path.parent)
                or log_path != start_path.parent.parent / "formal-launch.log"
                or launch_path.name != "launch-receipt.json"
                or failed.get("run_id") != "post-advisor-e1-beta-direction-a9-20260930"):
            raise RuntimeError("A10 A9 Formal Start/failed launch path cross-binding drift")
        if (failed.get("exit_code") != 1 or "release A..B contains a non-authority commit" not in failed_log
                or doc.get("engineering_predecessor_handoff_absent_path") != str(Path(doc["target_output_root"]) / f"RECOVERY_HANDOFF-{POST_ADVISOR_A9_EXECUTION_SHA[:12]}.json")
                or os.path.lexists(doc["engineering_predecessor_handoff_absent_path"])):
            raise RuntimeError("A10 A9 pre-R153 activation failure provenance drift")
        if not (Path(doc["target_output_root"]) / f"RECOVERY_HANDOFF-{doc['successor_execution_git_sha'][:12]}.json").is_file():
            r153 = Path(doc["target_output_root"]) / "round-153"
            if doc.get("engineering_predecessor_r153_absent_path") != str(r153) or os.path.lexists(r153):
                raise RuntimeError("A10 predecessor must have no R153 output before handoff")
    else:
        handoff = json.loads(Path(doc["engineering_predecessor_handoff_path"]).read_text())
        if handoff.get("status") != "PASS" or handoff.get("successor_execution_git_sha") != POST_ADVISOR_A8_EXECUTION_SHA:
            raise RuntimeError("A9 A8 handoff semantics drift")
    root = Path(doc["target_output_root"]).resolve(); rr = root / "round-152"
    files = doc.get("sealed_r152_files")
    if not isinstance(files, dict) or len(files) != 223:
        raise RuntimeError("A9 R152 exact 223-file closure missing")
    required = {"ROUND_COMPLETE.json", "state-after.json", "carrier-health-ledger.json",
                "carrier-health-boundary.json", "summary.json", "attempts.jsonl",
                "container-attempts/transition/a003.json", "a8-transition-output/bounded-state/bounded_recurrence_receipt.json"}
    if not required.issubset(files):
        raise RuntimeError("A9 R152 sealed science/diagnostic evidence incomplete")
    observed = {str(fp.relative_to(rr)) for fp in rr.rglob("*") if fp.is_file()}
    if check_live_state and observed != set(files):
        raise RuntimeError("A9 R152 exact sealed file closure drift")
    for name, digest in files.items():
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or not (rr / rel).is_file() or _post_advisor_recovery_sha(rr / rel) != digest:
            raise RuntimeError(f"A9 sealed R152 file drift: {name}")
    pause_path = root / "carrier-health-pauses/round-152.json"
    if doc.get("pause_receipt_path") != str(pause_path):
        raise RuntimeError("A9 pause path drift")
    pause = json.loads(pause_path.read_text())
    for name, key in (("ROUND_COMPLETE.json", "round_complete_sha256"), ("state-after.json", "state_after_sha256"),
                      ("carrier-health-ledger.json", "health_ledger_sha256"),
                      ("carrier-health-boundary.json", "boundary_decision_sha256")):
        if files[name] != pause.get(key):
            raise RuntimeError("A9 R152 pause evidence SHA drift")
    complete = json.loads((rr / "ROUND_COMPLETE.json").read_text())
    after = json.loads((rr / "state-after.json").read_text())
    ledger = json.loads((rr / "carrier-health-ledger.json").read_text())
    boundary = json.loads((rr / "carrier-health-boundary.json").read_text())
    transition = json.loads((rr / "container-attempts/transition/a003.json").read_text())
    if (complete.get("execution_git_sha") != POST_ADVISOR_A8_EXECUTION_SHA
            or complete.get("experiment_arm_id") != "BOUNDED_OFF"
            or after.get("next_round") != 153 or after.get("rollouts_consumed") != 19584
            or after.get("final_panel_access_count") != 0 or after.get("external_teacher_calls") != 0
            or ledger.get("arm") != "BOUNDED_OFF" or ledger.get("round_index") != 152
            or boundary.get("action") != "PAUSE_FOR_DIAGNOSTIC" or boundary.get("ledger_status") != "FAIL"
            or transition.get("status") != "PASS"):
        raise RuntimeError("A9 R152 complete/diagnostic semantics drift")
    before = base.load(rr / "state-before.json")
    base.validate_rollout_summary(base.load(rr / "summary.json"), 152, before["state_id"])
    producers = base.validate_attempt_rows(rr, 152, before["state_id"], {doc["predecessor_execution_git_sha"]})
    if producers != {doc["predecessor_execution_git_sha"]}:
        raise RuntimeError("A9 R152 producer allowlist drift")
    validate_a9_resolution(authority, doc)
    handoff_path = root / f"RECOVERY_HANDOFF-{doc['successor_execution_git_sha'][:12]}.json"
    expected_handoff = {"schema": ("openevo.post-advisor.e1-r152-complete-a10-handoff.202609301900" if is_a10
                                   else "openevo.post-advisor.e1-r152-complete-a9-handoff.202609301600"),
                        "status": "PASS", "round_index": 152, "next_round": 153,
                        "producer_execution_git_sha": doc["predecessor_execution_git_sha"],
                        "completed_by_execution_git_sha": POST_ADVISOR_A8_EXECUTION_SHA,
                        "successor_execution_git_sha": doc["successor_execution_git_sha"],
                        "round_complete_sha256": files["ROUND_COMPLETE.json"],
                        "pause_receipt_sha256": POST_ADVISOR_R152_PAUSE_SHA,
                        "resolution_receipt_sha256": doc["resolution_receipt_sha256"],
                        "recovery_authority_sha256": _post_advisor_recovery_sha(path),
                        "formal_rollout_replay": 0, "final_panel_access": 0}
    if is_a10:
        expected_handoff.update(
            engineering_predecessor_execution_git_sha=POST_ADVISOR_A9_EXECUTION_SHA,
            engineering_predecessor_formal_start_sha256=doc["engineering_predecessor_formal_start_sha256"],
            engineering_predecessor_launch_receipt_sha256=doc["engineering_predecessor_launch_receipt_sha256"],
            engineering_predecessor_formal_log_sha256=doc["engineering_predecessor_formal_log_sha256"],
        )
    if handoff_path.is_file() and json.loads(handoff_path.read_text()) != expected_handoff:
        raise RuntimeError("A9 completed-round handoff drift")
    if check_live_state:
        live_path = root / "execution-state.json"
        if not live_path.is_file(): raise RuntimeError("A9 live state missing")
        live = json.loads(live_path.read_text())
        if not handoff_path.is_file():
            if (_post_advisor_recovery_sha(live_path) != doc.get("engineering_predecessor_paused_state_sha256")
                    or live.get("status") != "PAUSE_FOR_DIAGNOSTIC" or live.get("next_round") != 153
                    or live.get("carrier_health_pause_sha256") != POST_ADVISOR_R152_PAUSE_SHA):
                raise RuntimeError("A9 pre-handoff A8 paused state drift")
        else:
            if (live.get("next_round", -1) < 153 or live.get("next_round", 161) > 160
                    or live.get("rollouts_consumed") != live["next_round"] * 128
                    or live.get("final_panel_access_count") != 0 or live.get("external_teacher_calls") != 0):
                raise RuntimeError("A9 live continuation accounting drift")
    return doc

def validate_a11_terminal_seal(authority: dict, doc: dict, *, check_live_state: bool = True) -> dict:
    """Read-only, exact predecessor and terminal-health check for the seal-only successor."""
    from scripts.openevo_webshop import ceiling1_stage2_arm_controller as base
    from scripts.openevo_webshop.carrier_health_failure_policy_v1 import validate_resolution_receipt
    root = Path(str(doc.get("target_output_root") or ""))
    old_path = Path(str(doc.get("a10_arm_authority_path") or ""))
    old_sha = str(doc.get("a10_arm_authority_sha256") or "")
    if (doc.get("schema") != POST_ADVISOR_A11_RECOVERY_SCHEMA or doc.get("status") != "AUTHORIZED_SEAL_ONLY"
            or doc.get("successor_execution_git_sha") != authority.get("execution_git_sha")
            or doc.get("predecessor_execution_git_sha") != "6b4ff4503adc9494c949f654521f09ee52ec28b4"
            or doc.get("engineering_predecessor_execution_git_sha") != POST_ADVISOR_A10_EXECUTION_SHA
            or doc.get("engineering_predecessor_activation_git_sha") != POST_ADVISOR_A10_ACTIVATION_SHA
            or str(root) != authority.get("output_root") or doc.get("formal_rollout_replay") != 0
            or doc.get("remaining_rollout_budget") != 0 or doc.get("already_consumed_rollouts") != 1024
            or doc.get("final_panel_access") != 0 or doc.get("external_teacher_calls") != 0
            or doc.get("scientific_contract_changed") is not False):
        raise RuntimeError("A11 seal-only lineage/budget drift")
    if (not old_path.is_file() or _post_advisor_recovery_sha(old_path) != old_sha
            or old_path.name != "post-advisor-e1-a10-recovery-authority-202609301900.json"):
        raise RuntimeError("A11 exact A10 predecessor authority drift")
    old_authority = json.loads(old_path.read_text())
    if old_authority.get("execution_git_sha") != POST_ADVISOR_A10_EXECUTION_SHA or old_authority.get("output_root") != str(root):
        raise RuntimeError("A11 A10 predecessor identity drift")
    validate_post_advisor_recovery(old_authority, check_live_state=check_live_state)
    old_recovery = json.loads(Path(old_authority["engineering_recovery"]["path"]).read_text())
    validate_a9_resolution(old_authority, old_recovery)
    pause_path = root / "carrier-health-pauses/round-159.json"
    pause = json.loads(pause_path.read_text())
    if (_post_advisor_recovery_sha(pause_path) != POST_ADVISOR_R159_PAUSE_SHA
            or pause.get("status") != "PAUSE_FOR_DIAGNOSTIC" or pause.get("arm") != "BOUNDED_OFF"
            or pause.get("through_round") != 159 or pause.get("next_round") != 160
            or pause.get("formal_transaction_preserved") is not True):
        raise RuntimeError("A11 R159 pause identity drift")
    evidence = doc.get("round_evidence")
    if not isinstance(evidence, dict) or set(evidence) != {str(r) for r in range(152, 160)}:
        raise RuntimeError("A11 eight-round evidence manifest missing")
    expected_state_id = None
    for r in range(152, 160):
        rr = root / f"round-{r:03d}"
        files = evidence[str(r)]
        if not isinstance(files, dict) or set(files) != {"state-before.json", "summary.json", "attempts.jsonl", "ROUND_COMPLETE.json", "state-after.json", "sd-lora/terminal-receipt.json"}:
            raise RuntimeError(f"A11 R{r} evidence file closure drift")
        for rel, digest in files.items():
            fp = rr / rel
            if not fp.is_file() or _post_advisor_recovery_sha(fp) != digest:
                raise RuntimeError(f"A11 R{r} evidence SHA drift: {rel}")
        before = json.loads((rr / "state-before.json").read_text())
        after = json.loads((rr / "state-after.json").read_text())
        complete = json.loads((rr / "ROUND_COMPLETE.json").read_text())
        if (expected_state_id is not None and before.get("state_id") != expected_state_id):
            raise RuntimeError(f"A11 R{r} state chain drift")
        expected_state_id = after.get("state_id")
        if (complete.get("status") != "PASS" or complete.get("round_index") != r
                or complete.get("experiment_arm_id") != "BOUNDED_OFF"
                or complete.get("treatment_mode") != "MAGNITUDE_SCALAR_GATE_V1"
                or after.get("next_round") != r + 1 or after.get("rollouts_consumed") != (r + 1) * 128
                or after.get("final_panel_access_count") != 0 or after.get("external_teacher_calls") != 0):
            raise RuntimeError(f"A11 R{r} sealed transaction drift")
        base.validate_rollout_summary(base.load(rr / "summary.json"), r, before["state_id"])
        allowed = {doc["predecessor_execution_git_sha"] if r == 152 else POST_ADVISOR_A10_EXECUTION_SHA}
        producers = base.validate_attempt_rows(rr, r, before["state_id"], allowed)
        if producers != allowed:
            raise RuntimeError(f"A11 R{r} producer provenance drift")
        episode_shas = (doc.get("episode_shas") or {}).get(str(r))
        observed_episodes = {str(fp.relative_to(rr)): _post_advisor_recovery_sha(fp)
                             for fp in rr.glob("shard-*/episodes/*.json")}
        if (not isinstance(episode_shas, dict) or len(episode_shas) != 128
                or observed_episodes != episode_shas):
            raise RuntimeError(f"A11 R{r} exact 128-episode evidence drift")
    terminal_files = doc.get("terminal_files")
    expected_terminal = {"carrier-health-pauses/round-159.json": POST_ADVISOR_R159_PAUSE_SHA,
                         "round-159/carrier-health-ledger.json": pause["health_ledger_sha256"],
                         "round-159/carrier-health-boundary.json": pause["boundary_decision_sha256"],
                         "round-159/ROUND_COMPLETE.json": pause["round_complete_sha256"],
                         "round-159/state-after.json": pause["state_after_sha256"]}
    if terminal_files != expected_terminal:
        raise RuntimeError("A11 terminal evidence manifest drift")
    for rel, digest in terminal_files.items():
        if _post_advisor_recovery_sha(root / rel) != digest:
            raise RuntimeError(f"A11 terminal evidence SHA drift: {rel}")
    ledger = json.loads((root / "round-159/carrier-health-ledger.json").read_text())
    boundary = json.loads((root / "round-159/carrier-health-boundary.json").read_text())
    if ledger.get("status") != "FAIL" or boundary.get("action") != "PAUSE_FOR_DIAGNOSTIC":
        raise RuntimeError("A11 R159 health FAIL must remain truthful")
    resolution_ref = authority.get("terminal_carrier_health_resolution") or {}
    resolution_path = Path(str(resolution_ref.get("path") or ""))
    if (not resolution_path.is_file() or resolution_ref.get("sha256") != _post_advisor_recovery_sha(resolution_path)
            or resolution_ref.get("path") != doc.get("resolution_receipt_path")
            or resolution_ref.get("sha256") != doc.get("resolution_receipt_sha256")):
        raise RuntimeError("A11 terminal resolution path/SHA drift")
    resolution = json.loads(resolution_path.read_text())
    policy_path = Path(str(old_recovery["carrier_health_policy_path"]))
    if (not policy_path.is_file() or _post_advisor_recovery_sha(policy_path) != old_recovery["carrier_health_policy_sha256"]):
        raise RuntimeError("A11 inherited carrier policy drift")
    policy = base.load_carrier_health_policy(policy_path)
    validate_resolution_receipt(policy, resolution, expected_arm="qwen3-1p7b", expected_round=159,
                                expected_pause_receipt_sha256=POST_ADVISOR_R159_PAUSE_SHA)
    expected_resolution = {"experiment_arm_id": "BOUNDED_OFF", "model_arm": "qwen3-1p7b",
                           "policy_sha256": old_recovery["carrier_health_policy_sha256"],
                           "health_ledger_sha256": pause["health_ledger_sha256"],
                           "boundary_decision_sha256": pause["boundary_decision_sha256"],
                           "round_complete_sha256": pause["round_complete_sha256"],
                           "state_after_sha256": pause["state_after_sha256"],
                           "remaining_rollout_budget": 0, "external_teacher_calls": 0,
                           "final_panel_access": 0,
                           "owner_authorization": "Owner authorized same-contract R159 diagnostic resolution and seal-only completion in this conversation."}
    if any(resolution.get(k) != v for k, v in expected_resolution.items()):
        raise RuntimeError("A11 diagnostic/owner/science boundary drift")
    if resolution.get("no_change") != {"formal_budget": True, "repair_call_limit": True,
                                       "sampling": True, "scientific_algorithm": True, "token_cap": True}:
        raise RuntimeError("A11 forbidden scientific change detected")
    if check_live_state:
        live_path = root / "execution-state.json"
        live = json.loads(live_path.read_text())
        if (_post_advisor_recovery_sha(live_path) != doc.get("predecessor_paused_state_sha256")
                or live.get("status") != "PAUSE_FOR_DIAGNOSTIC" or live.get("next_round") != 160
                or live.get("rollouts_consumed") != 20480 or live.get("carrier_health_pause_sha256") != POST_ADVISOR_R159_PAUSE_SHA
                or live.get("final_panel_access_count") != 0 or live.get("external_teacher_calls") != 0):
            raise RuntimeError("A11 exact live paused state drift")
        after = json.loads((root / "round-159/state-after.json").read_text())
        for key in ("state_id", "current_sd_adapter", "current_sd_artifact_id", "next_round", "rollouts_consumed"):
            if live.get(key) != after.get(key):
                raise RuntimeError(f"A11 R159 live/after state drift: {key}")
        recovered = [row for row in live.get("controller_recoveries", []) if row.get("recovery_kind") == "CARRIER_HEALTH_DIAGNOSTIC_RESOLVED"]
        if not any(row.get("through_round") == 152 and row.get("resolution_receipt_sha256") == old_authority["carrier_health_resolution"]["sha256"] for row in recovered):
            raise RuntimeError("A11 historical R152 resolution not recorded")
        if any(row.get("through_round") == 159 for row in recovered):
            raise RuntimeError("A11 R159 resolution was already consumed")
    return doc


def validate_post_advisor_recovery(authority: dict, *, check_live_state: bool = True) -> dict | None:
    ref = authority.get("engineering_recovery")
    if ref is None:
        return None
    path = Path(str(ref.get("path") or ""))
    if not path.is_file() or _post_advisor_recovery_sha(path) != ref.get("sha256"):
        raise RuntimeError("post-advisor engineering recovery authority missing or drifted")
    doc = json.loads(path.read_text())
    if doc.get("schema") == POST_ADVISOR_A11_RECOVERY_SCHEMA:
        return validate_a11_terminal_seal(authority, doc, check_live_state=check_live_state)
    if doc.get("schema") in (POST_ADVISOR_A9_RECOVERY_SCHEMA, POST_ADVISOR_A10_RECOVERY_SCHEMA):
        return _validate_post_advisor_a9(authority, doc, path, check_live_state=check_live_state)
    is_a8 = doc.get("schema") == POST_ADVISOR_A8_RECOVERY_SCHEMA
    if doc.get("schema") not in (POST_ADVISOR_RECOVERY_SCHEMA, POST_ADVISOR_A8_RECOVERY_SCHEMA) or doc.get("status") != "AUTHORIZED_ENGINEERING_CONTINUATION":
        raise RuntimeError("post-advisor engineering recovery schema/status drift")
    if (doc.get("successor_execution_git_sha") != authority.get("execution_git_sha")
            or doc.get("target_output_root") != authority.get("output_root")
            or doc.get("predecessor_execution_git_sha") != "6b4ff4503adc9494c949f654521f09ee52ec28b4"):
        raise RuntimeError("post-advisor recovery execution/root lineage drift")
    if (doc.get("formal_rollout_replay") != 0 or doc.get("consumed_rollouts_before_recovery") != 128
            or doc.get("remaining_rollout_budget") != 896 or doc.get("final_panel_access_delta") != 0
            or doc.get("external_teacher_calls_delta") != 0 or doc.get("scientific_contract_changed") is not False):
        raise RuntimeError("post-advisor recovery protected budget drift")
    start_path = Path(str(doc.get("predecessor_formal_start_path") or ""))
    if not start_path.is_file() or _post_advisor_recovery_sha(start_path) != doc.get("predecessor_formal_start_sha256"):
        raise RuntimeError("post-advisor recovery A6 Formal Start drift")
    start = json.loads(start_path.read_text())
    if (start.get("status") != "FORMAL_START_COMMITTED"
            or start.get("campaign_id") != "20260928-2338-post-advisor-unanswered"
            or start.get("formal_start_committed") is not True
            or start.get("formal_task_consumption_may_begin") is not True
            or len(start.get("selected_gpu_uuids") or []) != 4):
        raise RuntimeError("post-advisor recovery A6 Formal Start identity drift")
    root = Path(doc["target_output_root"]).resolve()
    rr = root / "round-152"
    files = doc.get("predecessor_files")
    if not isinstance(files, dict) or len(files) != 180:
        raise RuntimeError("post-advisor recovery predecessor file manifest must contain the exact 180 sealed files")
    additional = doc.get("engineering_predecessor_files", {}) if is_a8 else {}
    if is_a8:
        expected_additional = {
            "container-attempts/transition/a002-gpu-gate.json",
            "container-attempts/transition/a002-intent.json",
            "container-attempts/transition/a002-launcher.json",
            "container-attempts/transition/a002.json",
            "container-attempts/transition/a002.log",
            f"sd-lora/replay-staging-{POST_ADVISOR_A7_EXECUTION_SHA[:12]}/openevo_sd_lora_replay.jsonl",
        }
        if (doc.get("engineering_predecessor_execution_git_sha") != POST_ADVISOR_A7_EXECUTION_SHA
                or doc.get("engineering_predecessor_activation_git_sha") != POST_ADVISOR_A7_ACTIVATION_SHA
                or doc.get("transition_output_parent_name") != "a8-transition-output"
                or not isinstance(additional, dict) or set(additional) != expected_additional):
            raise RuntimeError("post-advisor A8 exact A7 predecessor identity/inventory drift")
        a7_start_path = Path(str(doc.get("engineering_predecessor_formal_start_path") or ""))
        a7_handoff_path = Path(str(doc.get("engineering_predecessor_handoff_path") or ""))
        if (not a7_start_path.is_file() or _post_advisor_recovery_sha(a7_start_path) != doc.get("engineering_predecessor_formal_start_sha256")
                or not a7_handoff_path.is_file() or _post_advisor_recovery_sha(a7_handoff_path) != doc.get("engineering_predecessor_handoff_sha256")):
            raise RuntimeError("post-advisor A8 Formal Start/handoff predecessor drift")
        a7_start = json.loads(a7_start_path.read_text())
        a7_handoff = json.loads(a7_handoff_path.read_text())
        if (a7_start.get("status") != "FORMAL_START_COMMITTED"
                or a7_start.get("formal_task_consumption_may_begin") is not True
                or a7_start.get("campaign_id") != "20260928-2338-post-advisor-unanswered"
                or len(a7_start.get("selected_gpu_uuids") or []) != 4
                or a7_handoff.get("status") != "PASS"
                or a7_handoff.get("successor_execution_git_sha") != POST_ADVISOR_A7_EXECUTION_SHA
                or a7_handoff.get("producer_execution_git_sha") != doc["predecessor_execution_git_sha"]
                or a7_handoff.get("formal_rollout_replay") != 0):
            raise RuntimeError("post-advisor A8 predecessor Formal Start/handoff semantic drift")
        empty_output = rr / "sd-lora/bounded-state"
        if (doc.get("engineering_predecessor_empty_output_path") != str(empty_output)
                or not empty_output.is_dir() or empty_output.is_symlink()
                or empty_output.stat().st_uid != 1001 or any(empty_output.iterdir())):
            raise RuntimeError("post-advisor A8 predecessor output is not the exact empty UID1001 directory")
    required = {"state-before.json", "summary.json", "attempts.jsonl", "carrier-scope-round-binding.json",
                "evidence/evidence.json", "evidence/history-after.json", "text-memory/terminal-receipt.json",
                "text-memory/model-call-attempt.json", "text-memory/rejected-candidates.json",
                "sd-lora/dataset/selection.json", "sd-lora/dataset/manifest.json", "sd-lora/dataset/records.jsonl"}
    if not required.issubset(files) or len([name for name in files if name.startswith("shard-") and "/episodes/" in name]) != 128:
        raise RuntimeError("post-advisor recovery incomplete R152 scientific evidence inventory")
    for shard in range(4):
        prefix = f"shard-{shard:02d}/"
        if not {prefix + "attempts.jsonl", prefix + "summary.json",
                prefix + "gpu-before.json", prefix + "gpu-after.json"}.issubset(files):
            raise RuntimeError("post-advisor recovery shard receipt inventory incomplete")
    receipt_path = root / f"RECOVERY_HANDOFF-{doc['successor_execution_git_sha'][:12]}.json"
    if check_live_state and not receipt_path.is_file():
        observed_files = {str(fp.relative_to(rr)) for fp in rr.rglob("*") if fp.is_file()}
        allowed_files = set(files) | set(additional)
        pending_staged = rr / "sd-lora" / f"replay-staging-{doc['successor_execution_git_sha'][:12]}" / "openevo_sd_lora_replay.jsonl"
        if pending_staged.is_file():
            st = pending_staged.stat()
            if st.st_nlink != 1 or st.st_uid != 1001 or _post_advisor_recovery_sha(pending_staged) != doc["predecessor_replay_sha256"]:
                raise RuntimeError("post-advisor recovery pending staged replay drift")
            allowed_files.add(str(pending_staged.relative_to(rr)))
        if observed_files != allowed_files:
            raise RuntimeError("post-advisor recovery pre-handoff R152 file inventory drift")
    for name, expected in {**files, **additional}.items():
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError("post-advisor recovery unsafe evidence path")
        fp = rr / rel
        if not fp.is_file() or _post_advisor_recovery_sha(fp) != expected:
            raise RuntimeError(f"post-advisor recovery predecessor payload drift: {fp}")
    if is_a8:
        failed = json.loads((rr / "container-attempts/transition/a002.json").read_text())
        failed_log = (rr / "container-attempts/transition/a002.log").read_text()
        if (failed.get("status") != "ENGINEERING_FAILURE"
                or failed.get("execution_git_sha") != POST_ADVISOR_A7_EXECUTION_SHA
                or failed.get("round_index") != 152
                or failed.get("container_exit_code") != 1
                or "successor cannot restart: BLOCKED_PARTIAL_OUTPUT" not in failed_log):
            raise RuntimeError("post-advisor A8 A7 transition failure provenance drift")
        a7_staged = rr / f"sd-lora/replay-staging-{POST_ADVISOR_A7_EXECUTION_SHA[:12]}/openevo_sd_lora_replay.jsonl"
        if a7_staged.stat().st_uid != 1001 or a7_staged.stat().st_nlink != 1:
            raise RuntimeError("post-advisor A8 A7 staged replay ownership drift")
    from scripts.openevo_webshop import ceiling1_stage2_arm_controller as base
    before = base.load(rr / "state-before.json")
    summary = base.load(rr / "summary.json")
    base.validate_rollout_summary(summary, 152, before["state_id"])
    producers = base.validate_attempt_rows(rr, 152, before["state_id"], {doc["predecessor_execution_git_sha"]})
    if producers != {doc["predecessor_execution_git_sha"]}:
        raise RuntimeError("post-advisor recovery rollout producer mismatch")
    base.validate_evidence_result(base.load(rr / "evidence/evidence.json"), rr, 152,
                                  Path(before["evidence_history"]) if before.get("evidence_history") else None)
    base.validate_parametric_selection(base.load(rr / "sd-lora/dataset/selection.json"),
                                       rr / "sd-lora/dataset", rr / "attempts.jsonl", 152)
    terminal = base.load(rr / "text-memory/terminal-receipt.json")
    if terminal.get("status") != "NOOP" or terminal.get("round_index") != 152 or terminal.get("model_calls_consumed") != 2:
        raise RuntimeError("post-advisor recovery text-memory terminal drift")
    if before.get("state_id") != doc.get("predecessor_state_id") or before.get("post_advisor_short_fork_authority_sha256") != doc.get("predecessor_arm_authority_sha256"):
        raise RuntimeError("post-advisor recovery inherited state/authority mismatch")
    if check_live_state and not receipt_path.is_file():
        live = root / "execution-state.json"
        if not live.is_file() or _post_advisor_recovery_sha(live) != doc.get("predecessor_execution_state_sha256"):
            raise RuntimeError("post-advisor recovery live A6 control state drift before handoff")
    replay = Path(before["current_sd_adapter"]) / "openevo_sd_lora_replay.jsonl"
    if not replay.is_file() or _post_advisor_recovery_sha(replay) != doc.get("predecessor_replay_sha256"):
        raise RuntimeError("post-advisor recovery R151 replay drift")
    if receipt_path.is_file():
        handoff = base.load(receipt_path)
        staged = rr / "sd-lora" / f"replay-staging-{doc['successor_execution_git_sha'][:12]}" / "openevo_sd_lora_replay.jsonl"
        if not staged.is_file() or staged.stat().st_nlink != 1 or staged.stat().st_uid != 1001:
            raise RuntimeError("post-advisor recovery staged replay identity drift")
        expected_phase = {
            "schema": ("openevo.post-advisor.e1-r152-recovery-phase-seal.a8.202609301200" if is_a8
                       else "openevo.post-advisor.e1-r152-recovery-phase-seal.202609301000"),
            "status": "PASS", "round_index": 152,
            "producer_execution_git_sha": doc["predecessor_execution_git_sha"],
            "successor_execution_git_sha": doc["successor_execution_git_sha"],
            "state_before_sha256": files["state-before.json"],
            "attempts_sha256": files["attempts.jsonl"], "summary_sha256": files["summary.json"],
            "attempt_count": 128, "formal_rollout_replay": 0, "final_panel_access": 0,
            "recovery_authority_sha256": _post_advisor_recovery_sha(path),
            "staged_replay_path": str(staged),
            "staged_replay_sha256": doc["predecessor_replay_sha256"],
            "predecessor_execution_state_sha256": doc["predecessor_execution_state_sha256"],
        }
        if is_a8:
            expected_phase.update(
                engineering_predecessor_execution_git_sha=POST_ADVISOR_A7_EXECUTION_SHA,
                engineering_predecessor_formal_start_sha256=doc["engineering_predecessor_formal_start_sha256"],
                engineering_predecessor_handoff_sha256=doc["engineering_predecessor_handoff_sha256"],
                engineering_predecessor_failed_transition_sha256=additional["container-attempts/transition/a002.json"],
                transition_output_parent_name="a8-transition-output",
            )
        if handoff != expected_phase or _post_advisor_recovery_sha(staged) != doc["predecessor_replay_sha256"]:
            raise RuntimeError("post-advisor recovery handoff receipt drift")
        live = base.load(root / "execution-state.json")
        next_round = int(live.get("next_round", -1))
        if next_round < 152 or next_round > 160 or live.get("rollouts_consumed") != next_round * 128:
            raise RuntimeError("post-advisor recovery live controller accounting drift")
        if next_round == 152 and (live.get("state_id") != before["state_id"]
                                  or live.get("post_advisor_short_fork_authority_sha256") != doc["predecessor_arm_authority_sha256"]):
            raise RuntimeError("post-advisor recovery partial-round science state drift")
    return doc

def prepare_post_advisor_recovery(authority: dict) -> dict | None:
    doc = validate_post_advisor_recovery(authority)
    if doc is None:
        return None
    if doc.get("schema") == POST_ADVISOR_A11_RECOVERY_SCHEMA:
        return doc
    if doc.get("schema") in (POST_ADVISOR_A9_RECOVERY_SCHEMA, POST_ADVISOR_A10_RECOVERY_SCHEMA):
        from scripts.openevo_webshop import ceiling1_stage2_arm_controller as base
        root = Path(doc["target_output_root"]).resolve()
        receipt_path = root / f"RECOVERY_HANDOFF-{doc['successor_execution_git_sha'][:12]}.json"
        is_a10 = doc["schema"] == POST_ADVISOR_A10_RECOVERY_SCHEMA
        phase = {"schema": ("openevo.post-advisor.e1-r152-complete-a10-handoff.202609301900" if is_a10
                            else "openevo.post-advisor.e1-r152-complete-a9-handoff.202609301600"),
                 "status": "PASS", "round_index": 152, "next_round": 153,
                 "producer_execution_git_sha": doc["predecessor_execution_git_sha"],
                 "completed_by_execution_git_sha": POST_ADVISOR_A8_EXECUTION_SHA,
                 "successor_execution_git_sha": doc["successor_execution_git_sha"],
                 "round_complete_sha256": doc["sealed_r152_files"]["ROUND_COMPLETE.json"],
                 "pause_receipt_sha256": POST_ADVISOR_R152_PAUSE_SHA,
                 "resolution_receipt_sha256": doc["resolution_receipt_sha256"],
                 "recovery_authority_sha256": _post_advisor_recovery_sha(Path(authority["engineering_recovery"]["path"])),
                 "formal_rollout_replay": 0, "final_panel_access": 0}
        if is_a10:
            phase.update(
                engineering_predecessor_execution_git_sha=POST_ADVISOR_A9_EXECUTION_SHA,
                engineering_predecessor_formal_start_sha256=doc["engineering_predecessor_formal_start_sha256"],
                engineering_predecessor_launch_receipt_sha256=doc["engineering_predecessor_launch_receipt_sha256"],
                engineering_predecessor_formal_log_sha256=doc["engineering_predecessor_formal_log_sha256"],
            )
        base.write_once_json(receipt_path, phase)
        return phase
    from scripts.openevo_webshop import ceiling1_stage2_arm_controller as base
    root = Path(doc["target_output_root"]).resolve()
    rr = root / "round-152"
    receipt_path = root / f"RECOVERY_HANDOFF-{doc['successor_execution_git_sha'][:12]}.json"
    replay_source = Path(base.load(rr / "state-before.json")["current_sd_adapter"]) / "openevo_sd_lora_replay.jsonl"
    staged = rr / "sd-lora" / f"replay-staging-{doc['successor_execution_git_sha'][:12]}" / "openevo_sd_lora_replay.jsonl"
    staged.parent.mkdir(parents=True, exist_ok=True)
    if not staged.exists():
        with staged.open("xb") as out, replay_source.open("rb") as source:
            for block in iter(lambda: source.read(1 << 20), b""):
                out.write(block)
            out.flush(); os.fsync(out.fileno())
        os.chown(staged, 1001, 1001)
        staged.chmod(0o600)
    st = staged.stat()
    if not staged.is_file() or st.st_nlink != 1 or st.st_uid != 1001 or _post_advisor_recovery_sha(staged) != doc["predecessor_replay_sha256"]:
        raise RuntimeError("post-advisor recovery staged replay must be UID1001, single-link, exact SHA")
    phase = {"schema": ("openevo.post-advisor.e1-r152-recovery-phase-seal.a8.202609301200"
                        if doc["schema"] == POST_ADVISOR_A8_RECOVERY_SCHEMA
                        else "openevo.post-advisor.e1-r152-recovery-phase-seal.202609301000"), "status": "PASS",
             "round_index": 152, "producer_execution_git_sha": doc["predecessor_execution_git_sha"],
             "successor_execution_git_sha": doc["successor_execution_git_sha"],
             "state_before_sha256": doc["predecessor_files"]["state-before.json"],
             "attempts_sha256": doc["predecessor_files"]["attempts.jsonl"],
             "summary_sha256": doc["predecessor_files"]["summary.json"],
             "attempt_count": 128, "formal_rollout_replay": 0, "final_panel_access": 0,
             "recovery_authority_sha256": _post_advisor_recovery_sha(Path(authority["engineering_recovery"]["path"])),
             "staged_replay_path": str(staged), "staged_replay_sha256": _post_advisor_recovery_sha(staged),
             "predecessor_execution_state_sha256": doc["predecessor_execution_state_sha256"]}
    if doc["schema"] == POST_ADVISOR_A8_RECOVERY_SCHEMA:
        phase.update(
            engineering_predecessor_execution_git_sha=POST_ADVISOR_A7_EXECUTION_SHA,
            engineering_predecessor_formal_start_sha256=doc["engineering_predecessor_formal_start_sha256"],
            engineering_predecessor_handoff_sha256=doc["engineering_predecessor_handoff_sha256"],
            engineering_predecessor_failed_transition_sha256=doc["engineering_predecessor_files"]["container-attempts/transition/a002.json"],
            transition_output_parent_name="a8-transition-output",
        )
    base.write_once_json(receipt_path, phase)
    return phase
