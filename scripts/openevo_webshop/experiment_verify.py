#!/usr/bin/env python3
"""Read-only identity verification for Experiment Passport lineages."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from scripts.openevo_webshop.experiment_passport import (
    PassportError,
    iter_references,
    load_json,
    load_passport,
)
from scripts.openevo_webshop.experiment_registry import (
    RegistryError,
    load_current_registry,
    resolve_experiment,
    validate_registry,
)

ROOT = Path(__file__).resolve().parents[2]
SPEC_SCHEMA = "openevo.experiment-verification-spec.202609112330"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class VerifyError(ValueError):
    pass

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _get(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _layer(status: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {"status": status, "reason": reason, **extra}


def _repo_path(root: Path, rel: str) -> Path:
    path = (root / rel).resolve()
    base = root.resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise VerifyError(f"repository path escapes root: {rel}") from exc
    return path

def _git_blob(root: Path, revision: str, rel: str) -> bytes | None:
    result = subprocess.run(
        ["git", "-C", str(root), "show", f"{revision}:{rel}"],
        capture_output=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def _verify_reference(root: Path, ref: Mapping[str, Any]) -> tuple[str, str]:
    kind, rel, wanted = ref.get("kind"), ref.get("path"), ref.get("sha256")
    if kind == "repository_file" and isinstance(rel, str):
        path = _repo_path(root, rel)
        if not path.is_file():
            return "MISSING", f"repository asset missing: {rel}"
        if wanted and _sha256_path(path) != wanted:
            return "FAIL", f"repository asset hash mismatch: {rel}"
        return "PASS", f"repository asset verified: {rel}"
    if kind == "git_file" and isinstance(rel, str):
        revision = ref.get("revision")
        if not isinstance(revision, str):
            return "FAIL", f"git reference lacks immutable revision: {rel}"
        blob = _git_blob(root, revision, rel)
        if blob is None:
            return "MISSING", f"git object unavailable locally: {revision}:{rel}"
        if wanted and _sha256_bytes(blob) != wanted:
            return "FAIL", f"historical git asset hash mismatch: {revision}:{rel}"
        return "PASS", f"historical git asset verified: {revision}:{rel}"
    return "NOT_LOCAL", f"{kind} reference is not checked by static local verification"


def _verify_static_assets(root: Path, passport: Mapping[str, Any]) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    fail = missing = 0
    for location, ref in iter_references(passport):
        status, reason = _verify_reference(root, ref)
        if status == "FAIL":
            fail += 1
        elif status == "MISSING":
            missing += 1
        if status != "NOT_LOCAL":
            details.append({"location": location, "status": status, "reason": reason})
    science = passport["scientific_identity"]
    model = science["model"]
    asset = model["asset"]
    invariants: list[tuple[bool, str]] = [
        (asset.get("repository") == model.get("model_id"), "model asset repository matches model_id"),
        (asset.get("revision") == model.get("model_revision"), "model asset revision matches model_revision"),
        (science["seed_policy"]["schedule"] == science["task_manifest"], "seed schedule matches task manifest identity"),
    ]
    for ok, reason in invariants:
        details.append({"location": "semantic-invariant", "status": "PASS" if ok else "FAIL", "reason": reason})
        if not ok:
            fail += 1
    runtime_ref = passport["initial_execution_identity"]["runtime_asset_lock"]
    if runtime_ref.get("kind") == "repository_file" and isinstance(runtime_ref.get("path"), str):
        runtime_path = _repo_path(root, runtime_ref["path"])
        if runtime_path.is_file():
            runtime = load_json(runtime_path)
            expected = passport["initial_execution_identity"]["runtime_image_digest"]
            ok = runtime.get("image_digest") == expected
            details.append({"location": "runtime-image", "status": "PASS" if ok else "FAIL", "reason": "runtime asset image_digest matches Passport"})
            if not ok:
                fail += 1
    status = "FAIL" if fail else "MISSING" if missing else "PASS"
    reason = f"{fail} static identity failures, {missing} locally missing immutable objects" if (fail or missing) else "all locally checkable static identities pass"
    return _layer(status, reason, details=details)


def _verify_live_state(entry: Mapping[str, Any], passport: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    source = entry["state_source"]
    if source.get("type") != "server_json":
        return _layer("NOT_OBSERVED", "registered live source is not a directly observable server JSON"), None
    path = Path(str(source.get("path") or ""))
    if not path.is_file():
        return _layer("NOT_OBSERVED", "live state path is not visible from this host", path=str(path)), None
    try:
        state = load_json(path, max_bytes=16 * 1024 * 1024)
    except PassportError as exc:
        return _layer("FAIL", f"live state unreadable: {exc}", path=str(path)), None
    mismatches = []
    for field, wanted in (source.get("required_equal") or {}).items():
        observed = _get(state, str(field))
        if observed != wanted:
            mismatches.append({"field": field, "expected": wanted, "observed": observed})
    science = passport["scientific_identity"]
    budget = passport["budget_contract"]
    expected_config = budget["source"].get("sha256")
    if expected_config and state.get("config_sha256") != expected_config:
        mismatches.append({"field": "config_sha256", "expected": expected_config, "observed": state.get("config_sha256")})
    expected_activation = passport["initial_execution_identity"]["activation"].get("sha256")
    if expected_activation and state.get("stage2_activation_sha256") != expected_activation:
        mismatches.append({"field": "stage2_activation_sha256", "expected": expected_activation, "observed": state.get("stage2_activation_sha256")})
    rounds = state.get("rounds_complete")
    if isinstance(rounds, list):
        expected_rounds = list(range(len(rounds)))
        if rounds != expected_rounds:
            mismatches.append({"field": "rounds_complete", "expected": "contiguous 0..N-1", "observed": "non-contiguous"})
        expected_rollouts = len(rounds) * int(budget["rollouts_per_round"])
        if state.get("rollouts_consumed") != expected_rollouts:
            mismatches.append({"field": "rollouts_consumed", "expected": expected_rollouts, "observed": state.get("rollouts_consumed")})
        if state.get("next_round") != len(rounds):
            mismatches.append({"field": "next_round", "expected": len(rounds), "observed": state.get("next_round")})
    if isinstance(state.get("rollouts_consumed"), int) and state["rollouts_consumed"] > budget["rollout_limit"]:
        mismatches.append({"field": "rollouts_consumed", "expected": f"<= {budget['rollout_limit']}", "observed": state["rollouts_consumed"]})
    if state.get("external_teacher_calls", 0) > budget["external_teacher_call_limit"]:
        mismatches.append({"field": "external_teacher_calls", "expected": f"<= {budget['external_teacher_call_limit']}", "observed": state.get("external_teacher_calls")})
    panel_count = state.get("final_panel_access_count", 0)
    if state.get("status") != "COMPLETE" and panel_count != 0:
        mismatches.append({"field": "final_panel_access_count", "expected": 0, "observed": panel_count})
    if mismatches:
        return _layer("FAIL", "live state identity/accounting mismatch", path=str(path), mismatches=mismatches), state
    return _layer(
        "PASS",
        "live state matches Registry/Passport identity and bounded accounting",
        path=str(path),
        lifecycle_status=state.get("status"),
        next_round=state.get("next_round"),
        rollouts_consumed=state.get("rollouts_consumed"),
        current_execution_sha=state.get("twin_control_git_sha"),
    ), state


_PREV_KEYS = {"previous_control_sha", "previous_live_control_sha", "old_control_sha", "base_control_sha", "predecessor_control_sha", "previous_execution_git_sha"}
_SUCC_KEYS = {"successor_control_sha", "live_successor_control_sha", "new_control_sha", "successor_execution_git_sha"}


def _walk_items(value: Any):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key), item
            yield from _walk_items(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_items(item)


def _receipt_science_safe(value: Mapping[str, Any]) -> tuple[bool, str]:
    for key, item in _walk_items(value):
        low = key.lower()
        if low in {"scientific_semantics_changed", "science_semantics_changed"} and item is True:
            return False, "receipt declares scientific semantics changed"
        if low in {"formal_rollout_replay", "formal_task_consumption_delta", "external_teacher_delta", "final_panel_delta"}:
            if type(item) in (int, float) and item != 0:
                return False, f"receipt has nonzero {key}={item}"
    status = str(value.get("status") or "").upper()
    if any(token in status for token in ("FAIL", "REJECT", "NOT_APPLIED")):
        return False, f"receipt status is not an applied/pass authority: {status}"
    return True, "engineering-only receipt"


def _explicit_science_false(value: Mapping[str, Any]) -> bool:
    return any(key.lower() in {"scientific_semantics_changed", "science_semantics_changed"} and item is False for key, item in _walk_items(value))


def _receipt_pairs(value: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Extract predecessor/successor pairs only from the same mapping scope."""
    pairs: list[tuple[str, str]] = []
    previous = {item for key, item in value.items() if key in _PREV_KEYS and isinstance(item, str) and HEX40.fullmatch(item)}
    successor = {item for key, item in value.items() if key in _SUCC_KEYS and isinstance(item, str) and HEX40.fullmatch(item)}
    pairs.extend((p, s) for p in previous for s in successor if p != s)
    for item in value.values():
        if isinstance(item, Mapping):
            pairs.extend(_receipt_pairs(item))
        elif isinstance(item, list):
            for child in item:
                if isinstance(child, Mapping):
                    pairs.extend(_receipt_pairs(child))
    return list(dict.fromkeys(pairs))


def verify_execution_lineage_graph(initial_sha: str, current_sha: str, edges: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not HEX40.fullmatch(initial_sha) or not HEX40.fullmatch(current_sha):
        return _layer("FAIL", "execution SHA is malformed")
    adjacency: dict[str, list[Mapping[str, Any]]] = {}
    for edge in edges:
        previous = edge.get("previous")
        successor = edge.get("successor")
        safe = edge.get("science_changed") is False
        if not isinstance(previous, str) or not isinstance(successor, str) or not safe:
            continue
        adjacency.setdefault(previous, []).append(edge)
    seen = {initial_sha}
    queue = [initial_sha]
    parents: dict[str, Mapping[str, Any]] = {}
    while queue:
        node = queue.pop(0)
        for edge in adjacency.get(node, []):
            nxt = str(edge["successor"])
            if nxt in seen:
                continue
            seen.add(nxt)
            parents[nxt] = edge
            queue.append(nxt)
    if current_sha not in seen:
        return _layer("FAIL", "current execution SHA is not reachable from initial execution through safe receipts", reachable_count=len(seen))
    chain: list[Mapping[str, Any]] = []
    node = current_sha
    while node != initial_sha:
        edge = parents[node]
        chain.append(edge)
        node = str(edge["previous"])
    chain.reverse()
    return _layer("PASS", "current execution is a safe operational descendant of initial execution", transition_count=len(chain), chain=chain)


def _actual_control_sequence(passport: Mapping[str, Any], live_state: Mapping[str, Any]) -> tuple[list[str] | None, str | None]:
    run_root = passport.get("initial_execution_identity", {}).get("initial_run_root")
    rounds = live_state.get("rounds_complete")
    if not isinstance(run_root, str) or not Path(run_root).is_dir() or not isinstance(rounds, list):
        return None, "run root / committed-round sequence unavailable"
    controls: list[str] = []
    for round_index in rounds:
        if type(round_index) is not int or round_index < 0:
            return None, "rounds_complete contains an invalid index"
        state_path = Path(run_root) / f"round-{round_index:03d}" / "state-after.json"
        if not state_path.is_file():
            return None, f"committed state-after missing for round {round_index}"
        try:
            state = load_json(state_path, max_bytes=8 * 1024 * 1024)
        except PassportError as exc:
            return None, f"cannot read committed state-after round {round_index}: {exc}"
        control = state.get("twin_control_git_sha")
        if not isinstance(control, str) or not HEX40.fullmatch(control):
            return None, f"round {round_index} lacks a valid control SHA"
        if not controls or controls[-1] != control:
            controls.append(control)
    current = live_state.get("twin_control_git_sha")
    if isinstance(current, str) and HEX40.fullmatch(current) and (not controls or controls[-1] != current):
        controls.append(current)
    return controls, None


def verify_actual_execution_sequence(initial_sha: str, current_sha: str, controls: list[str], edges: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not controls or controls[0] != initial_sha or controls[-1] != current_sha:
        return _layer("FAIL", "actual control sequence endpoints disagree with Passport/live identity", controls=controls)
    by_pair: dict[tuple[str, str], Mapping[str, Any]] = {}
    for edge in edges:
        if edge.get("science_changed") is False:
            by_pair.setdefault((str(edge.get("previous")), str(edge.get("successor"))), edge)
    chain: list[Mapping[str, Any]] = []
    for previous, successor in zip(controls, controls[1:]):
        edge = by_pair.get((previous, successor))
        if edge is None:
            return _layer("FAIL", "actual live control transition lacks a same-science receipt", previous=previous, successor=successor)
        chain.append(edge)
    return _layer("PASS", "every control transition actually used by committed rounds is receipt-bound", transition_count=len(chain), chain=chain)


def _git_is_ancestor(root: Path, previous: str, successor: str) -> bool:
    if not HEX40.fullmatch(previous) or not HEX40.fullmatch(successor):
        return False
    result = subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", previous, successor], capture_output=True, check=False)
    return result.returncode == 0


def _reachable(initial: str, edges: list[Mapping[str, Any]]) -> set[str]:
    seen = {initial}
    changed = True
    while changed:
        changed = False
        for edge in edges:
            if edge.get("science_changed") is not False:
                continue
            previous, successor = edge.get("previous"), edge.get("successor")
            if previous in seen and isinstance(successor, str) and successor not in seen:
                seen.add(successor)
                changed = True
    return seen


def _sealed_boundary_predecessor(passport: Mapping[str, Any], successor: str, reachable: set[str]) -> dict[str, Any] | None:
    run_root = passport.get("initial_execution_identity", {}).get("initial_run_root")
    max_rounds = passport.get("budget_contract", {}).get("round_limit")
    if not isinstance(run_root, str) or not Path(run_root).is_dir() or not isinstance(max_rounds, int):
        return None
    root = Path(run_root)
    for round_index in range(1, max_rounds):
        before = root / f"round-{round_index:03d}" / "state-before.json"
        if not before.is_file():
            continue
        try:
            after_doc = load_json(root / f"round-{round_index-1:03d}" / "state-after.json", max_bytes=8 * 1024 * 1024)
            before_doc = load_json(before, max_bytes=8 * 1024 * 1024)
        except (OSError, PassportError):
            continue
        if before_doc.get("twin_control_git_sha") != successor:
            continue
        previous = after_doc.get("twin_control_git_sha")
        if previous not in reachable:
            continue
        keys = ("state_id", "rollouts_consumed", "next_round", "external_teacher_calls", "final_panel_access_count")
        if any(after_doc.get(key) != before_doc.get(key) for key in keys):
            continue
        return {
            "previous": previous, "successor": successor,
            "evidence": [str(root / f"round-{round_index-1:03d}" / "state-after.json"), str(before)],
            "proof": "authorized same-science successor + identical sealed boundary state",
        }
    return None


def _git_distance(root: Path, previous: str, successor: str) -> int | None:
    if not _git_is_ancestor(root, previous, successor):
        return None
    result = subprocess.run(["git", "-C", str(root), "rev-list", "--count", f"{previous}..{successor}"], text=True, capture_output=True, check=False)
    try:
        return int(result.stdout.strip()) if result.returncode == 0 else None
    except ValueError:
        return None


def _iter_authority_json(root: Path, limit: int = 512):
    tokens = ("HANDOFF", "BINDING", "MIGRATION", "ACTIVATION", "RECOVERY", "SUCCESSOR")
    count = 0
    for path in sorted(root.rglob("*.json")):
        if not any(token in path.name.upper() or token in str(path.parent.name).upper() for token in tokens):
            continue
        if count >= limit:
            raise VerifyError("authority receipt scan exceeded bounded file limit")
        count += 1
        try:
            if path.stat().st_size > 4 * 1024 * 1024:
                continue
            yield path, load_json(path, max_bytes=4 * 1024 * 1024)
        except (OSError, PassportError):
            continue


def _verify_execution_lineage(root: Path, passport: Mapping[str, Any], live_state: Mapping[str, Any] | None) -> dict[str, Any]:
    if live_state is None:
        return _layer("NOT_APPLICABLE", "live state is not observed")
    initial = passport["initial_execution_identity"].get("initial_experiment_sha")
    current = live_state.get("twin_control_git_sha")
    if not isinstance(initial, str) or not isinstance(current, str):
        return _layer("FAIL", "initial/current execution SHA is unavailable")
    if initial == current:
        return _layer("PASS", "live execution still equals initial execution identity", transition_count=0, chain=[])

    edges: list[dict[str, Any]] = []
    pending_claims: list[dict[str, str]] = []
    for item in live_state.get("controller_recoveries") or []:
        if not isinstance(item, Mapping):
            continue
        previous, successor = item.get("previous_control_sha"), item.get("successor_control_sha")
        receipt_path, wanted = item.get("recovery_receipt"), item.get("recovery_receipt_sha256")
        if not (isinstance(previous, str) and isinstance(successor, str)):
            continue
        if item.get("scientific_semantics_changed") is not False or item.get("formal_rollout_replay", 0) != 0:
            return _layer("FAIL", "live recovery ledger contains a science-changing or replaying successor", recovery_kind=item.get("recovery_kind"))
        if not isinstance(receipt_path, str) or not isinstance(wanted, str) or not Path(receipt_path).is_file():
            return _layer("FAIL", "live recovery ledger references a missing receipt", receipt=receipt_path)
        receipt = Path(receipt_path)
        if _sha256_path(receipt) != wanted:
            return _layer("FAIL", "live recovery receipt hash mismatch", receipt=receipt_path)
        try:
            receipt_doc = load_json(receipt, max_bytes=4 * 1024 * 1024)
        except PassportError as exc:
            return _layer("FAIL", f"live recovery receipt unreadable: {exc}", receipt=receipt_path)
        safe, reason = _receipt_science_safe(receipt_doc)
        if not safe or not _explicit_science_false(receipt_doc):
            return _layer("FAIL", f"live recovery receipt is not an explicit same-science authority: {reason}", receipt=receipt_path)
        if (previous, successor) in _receipt_pairs(receipt_doc):
            edges.append({"previous": previous, "successor": successor, "science_changed": False, "evidence": receipt_path, "sha256": wanted})
        else:
            pending_claims.append({"previous": previous, "successor": successor, "receipt": receipt_path})

    activation = passport["initial_execution_identity"].get("activation") or {}
    activation_path = activation.get("path") if isinstance(activation, Mapping) else None
    maintenance_root: Path | None = None
    if isinstance(activation_path, str) and Path(activation_path).is_absolute():
        ap = Path(activation_path)
        if len(ap.parents) >= 2:
            maintenance_root = ap.parents[1]
    single_successors: list[dict[str, Any]] = []
    if maintenance_root is not None and maintenance_root.is_dir():
        try:
            for path, doc in _iter_authority_json(maintenance_root):
                safe, _ = _receipt_science_safe(doc)
                if not safe or not _explicit_science_false(doc):
                    continue
                for previous, successor in _receipt_pairs(doc):
                    edges.append({
                        "previous": previous,
                        "successor": successor,
                        "science_changed": False,
                        "evidence": str(path),
                        "sha256": _sha256_path(path),
                    })
                status = str(doc.get("status") or "").upper()
                control = doc.get("control_sha")
                if (
                    isinstance(control, str)
                    and HEX40.fullmatch(control)
                    and "AUTHORIZED" in status
                    and doc.get("scientific_semantics_changed") is False
                    and doc.get("formal_rollout_replay", 0) == 0
                    and doc.get("model_calls_replayed", 0) == 0
                ):
                    single_successors.append({"successor": control, "evidence": str(path), "sha256": _sha256_path(path)})
        except VerifyError as exc:
            return _layer("FAIL", str(exc))

    explicit_pairs = {(str(e.get("previous")), str(e.get("successor"))) for e in edges if e.get("science_changed") is False}
    for claim in pending_claims:
        pair = (claim["previous"], claim["successor"])
        if pair not in explicit_pairs:
            return _layer("FAIL", "live recovery ledger successor lacks an exact same-science binding", receipt=claim["receipt"], previous=pair[0], successor=pair[1])
    reachable = _reachable(initial, edges)
    for item in single_successors:
        successor = str(item["successor"])
        if successor in reachable:
            continue
        boundary = _sealed_boundary_predecessor(passport, successor, reachable)
        if boundary is not None:
            edges.append({**boundary, "science_changed": False, "authority": item["evidence"], "authority_sha256": item["sha256"]})
            reachable = _reachable(initial, edges)
            continue
        candidates: list[tuple[int, str]] = []
        for previous in reachable:
            distance = _git_distance(root, previous, successor)
            if distance is not None and distance > 0:
                candidates.append((distance, previous))
        if not candidates:
            continue
        _, previous = min(candidates)
        edges.append({
            "previous": previous, "successor": successor, "science_changed": False,
            "evidence": item["evidence"], "sha256": item["sha256"],
            "proof": "authorized resource-safety successor + Git ancestry",
        })
        reachable = _reachable(initial, edges)
    controls, sequence_error = _actual_control_sequence(passport, live_state)
    if controls is None:
        return _layer("FAIL", f"actual execution sequence cannot be proven: {sequence_error}")
    return verify_actual_execution_sequence(initial, current, controls, edges)


def _overall(layers: Mapping[str, Mapping[str, Any]]) -> str:
    statuses = {name: layer.get("status") for name, layer in layers.items()}
    if any(status == "FAIL" for status in statuses.values()):
        return "FAIL"
    if any(status in {"MISSING", "NOT_OBSERVED"} for status in statuses.values()):
        return "MANUAL_REVIEW_REQUIRED"
    return "PASS"


def _registered_report(root: Path, experiment_or_alias: str) -> dict[str, Any]:
    layers: dict[str, dict[str, Any]] = {}
    try:
        registry, router = load_current_registry(root)
        errors = validate_registry(registry, root)
        if errors:
            raise RegistryError("; ".join(errors))
        entry = resolve_experiment(registry, experiment_or_alias)
        layers["REGISTRY"] = _layer("PASS", "Registry router, alias and Passport binding validate", router=router)
    except (RegistryError, PassportError) as exc:
        raise VerifyError(str(exc)) from exc
    passport_path = _repo_path(root, str(entry["passport_path"]))
    try:
        passport = load_passport(passport_path, root=root)
        layers["PASSPORT"] = _layer("PASS", "FROZEN Passport schema and internal fingerprints/hash validate")
    except PassportError as exc:
        layers["PASSPORT"] = _layer("FAIL", str(exc))
        passport = None
    if passport is None:
        for name in ("STATIC_ASSETS", "LIVE_STATE", "EXEC_LINEAGE", "ARTIFACTS"):
            layers[name] = _layer("NOT_APPLICABLE", "Passport validation failed")
        return {
            "schema": "openevo.experiment-verification.202609112330",
            "mutation": False,
            "experiment_id": entry["experiment_id"],
            "layers": layers,
            "overall": "FAIL",
            "identity": "FAIL",
            "safe_to_consider_continuation": "NO",
        }
    layers["STATIC_ASSETS"] = _verify_static_assets(root, passport)
    live_layer, live_state = _verify_live_state(entry, passport)
    layers["LIVE_STATE"] = live_layer
    layers["EXEC_LINEAGE"] = _verify_execution_lineage(root, passport, live_state) if live_layer["status"] == "PASS" else _layer("NOT_APPLICABLE", "live execution is not verified/observed")
    layers["ARTIFACTS"] = _layer("NOT_CHECKED", "provider availability is intentionally not network-checked by verify")
    overall = _overall(layers)
    identity = "PASS" if all(layers[name]["status"] == "PASS" for name in ("PASSPORT", "REGISTRY", "STATIC_ASSETS")) else "FAIL"
    resume_policy = str(passport["resume_policy"].get("policy") or "")
    if resume_policy in {"NO_RESUME", "NO_RESUME_HISTORICAL", "COMPLETE_NO_RESUME_NEEDED"}:
        safe = "NO"
    elif overall == "FAIL" or identity != "PASS":
        safe = "NO"
    elif live_layer["status"] != "PASS" or layers["EXEC_LINEAGE"]["status"] != "PASS":
        safe = "MANUAL_REVIEW"
    else:
        lifecycle = str(live_state.get("status") if live_state else "").upper()
        if lifecycle in {"STAGE2_SEALED", "SEALED", "COMPLETE", "COMPLETED"}:
            safe = "NO"
        elif lifecycle == "RUNNING":
            safe = "MANUAL_REVIEW"
        else:
            safe = "YES"
    return {
        "schema": "openevo.experiment-verification.202609112330",
        "mutation": False,
        "experiment_id": entry["experiment_id"],
        "requested_identity": experiment_or_alias,
        "passport_sha256": entry["passport_sha256"],
        "lifecycle_class": entry["lifecycle_class"],
        "resume_policy": resume_policy,
        "layers": layers,
        "overall": overall,
        "identity": identity,
        "safe_to_consider_continuation": safe,
    }


def _legacy_report(root: Path, experiment_id: str) -> dict[str, Any]:
    from scripts.openevo_webshop import experiment_console as console
    catalog_path = root / "configs/experiment/experiment-console-202609110009.json"
    catalog = console._read_json(catalog_path)
    errors = console.validate_catalog(root, catalog)
    if errors:
        raise VerifyError("legacy console catalog invalid: " + "; ".join(errors))
    row = next((item for item in catalog["experiments"] if item.get("id") == experiment_id), None)
    if row is None:
        raise VerifyError(f"unknown experiment identity or alias: {experiment_id}")
    observation = console.observe(root, row)
    authority_missing = [p for p in row.get("authority_paths") or [] if not _repo_path(root, str(p)).exists()]
    static_status = "MISSING" if authority_missing else "PASS"
    live_status = "FAIL" if observation.get("identity_trust") == "FAIL" else "NOT_OBSERVED" if observation.get("identity_trust") in {"NOT_OBSERVED", "INDEX_ONLY"} else "PASS"
    layers = {
        "PASSPORT": _layer("NOT_APPLICABLE", "legacy compatibility row has no backfilled Passport"),
        "REGISTRY": _layer("NOT_APPLICABLE", "legacy compatibility row is not yet in Experiment Registry"),
        "STATIC_ASSETS": _layer(static_status, "legacy authority paths present" if not authority_missing else "legacy authority path missing", missing=authority_missing),
        "LIVE_STATE": _layer(live_status, str(observation.get("effective_status")), observation_scope=observation.get("observation_scope")),
        "EXEC_LINEAGE": _layer("NOT_APPLICABLE", "no Passport execution-lineage contract exists for this legacy row"),
        "ARTIFACTS": _layer("NOT_CHECKED", "legacy artifact provider availability is not network-checked"),
    }
    policy = str((row.get("resume") or {}).get("policy") or "")
    safe = "NO" if policy in {"NO_RESUME", "NO_RESUME_HISTORICAL", "COMPLETE_NO_RESUME_NEEDED"} else "MANUAL_REVIEW"
    return {
        "schema": "openevo.experiment-verification.202609112330",
        "mutation": False,
        "experiment_id": experiment_id,
        "requested_identity": experiment_id,
        "legacy_compatibility": True,
        "resume_policy": policy,
        "layers": layers,
        "overall": "FAIL" if static_status == "MISSING" or live_status == "FAIL" else "MANUAL_REVIEW_REQUIRED",
        "identity": "FAIL" if static_status == "MISSING" or live_status == "FAIL" else "PASS",
        "safe_to_consider_continuation": safe,
    }


def verify_experiment(repo_root: Path, experiment_or_alias: str) -> dict[str, Any]:
    try:
        return _registered_report(repo_root, experiment_or_alias)
    except VerifyError as exc:
        if "unknown experiment identity or alias" not in str(exc):
            raise
        return _legacy_report(repo_root, experiment_or_alias)


def render_verification(report: Mapping[str, Any]) -> str:
    lines = [f"Experiment: {report['experiment_id']}"]
    for name in ("PASSPORT", "REGISTRY", "STATIC_ASSETS", "LIVE_STATE", "EXEC_LINEAGE", "ARTIFACTS"):
        layer = report["layers"][name]
        lines.append(f"{name}: {layer['status']} — {layer['reason']}")
    lines.extend([
        f"OVERALL: {report['overall']}",
        f"IDENTITY: {report['identity']}",
        f"SAFE_TO_CONSIDER_CONTINUATION: {report['safe_to_consider_continuation']}",
        "Read-only verification only. YES is not execution authorization; current campaign/release/live resource gates still apply.",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = verify_experiment(args.repo_root.expanduser().resolve(), args.experiment_id)
    except (VerifyError, RegistryError, PassportError) as exc:
        print(f"experiment-verify: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) if args.json else render_verification(report))
    return 0 if report["identity"] == "PASS" else 2


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
