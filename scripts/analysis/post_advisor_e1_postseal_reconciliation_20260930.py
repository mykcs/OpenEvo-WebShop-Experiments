#!/usr/bin/env python3
"""Read-only E1 postseal audit; optional append-only reconciliation receipt."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
REL_SCRIPT = SCRIPT.relative_to(REPO)
A6 = "6b4ff4503adc9494c949f654521f09ee52ec28b4"
A8 = "31a669ab9af34fdf5733849ea8ccfb7af37e7427"
A10 = "3181cb7440d48633685d9c675a1f4684e37500ce"
A11 = "32174473fc10b05756f680b4074f8cb9cda17883"
BASE = "a05d59e98452a56e46aac5519cb1b47dbaf878c5"
SEAL_SHA = "b03d0cbcb9e5fc89a8d8974952222aa5d35ffb083bc6a47239cfa30b2b893725"
STATE_SHA = "9a489bbac03eb72559989e4958bade00d203f96a8587d06b678844569d43e381"
ROOT = Path("/workspace/runs/post-advisor-unanswered-20260928/e1-beta-direction-formal-a6-6b4ff450")
OUTER = Path("/workspace/runs/post-advisor-unanswered-20260928/post-advisor-e1-beta-direction-a10-20260930")
RECEIPT = REPO / "docs/evidence/post-advisor-unanswered-20260928/E1_POSTSEAL_RECONCILIATION_20260930.json"
ARM_REL = "configs/experiment/activations/post-advisor-e1-a11-terminal-seal-authority-202609302100.json"
RELEASE_REL = "configs/experiment/activations/post-advisor-e1-a11-terminal-seal-release-202609302100.json"
PREREG_REL = "configs/experiment/preregistrations/post-advisor-e1-beta-direction-preregistration-202609282338.json"
A10_RELEASE_REL = "configs/experiment/activations/post-advisor-e1-a10-recovery-release-202609301900.json"
FAIL_SHA = {
    "launch-receipt-attempt-0001.json": "7731ff6c284c69ced12960b59dd401398bdaeba7d88333ad060b6799430c8705",
    "launch-receipt.json": "aeacc2557ab9f69caff0777b765af2162e5b2852a0ae648d04e1412f3cd9b006",
    "gpu-resource-receipt.json": "337ead58379d617bde820c5e373fb619f2a7f8e93a3ceb16bdfff9af4e0bba76",
}
VALIDATOR_SOURCES = {
    "scripts/openevo_webshop/bounded_effective_state_recovery_202609282338.py": "5a13c4d5ec28b63d9e266a328ffa4a0be3c176fea08ded62aa6c6d2740e3549d",
    "scripts/openevo_webshop/ceiling1_stage2_arm_controller.py": "c9fbca018adaa7c50f53c3698d7b0d36cb401fa95a82012063b2b850bec7eed8",
    "scripts/openevo_webshop/carrier_health_failure_policy_v1.py": "5cd5f04601e60b2be693b1526dd788b43e5e371bf620ec69759d1050999b1a5c",
    "scripts/openevo_webshop/ceiling1_stage2_reflector.py": "d43bb95f6811c91648e5f9e06911f4e389ed0d0736d2df1369d004ad2add747f",
    "scripts/openevo_webshop/carrier_scope_package_v1.py": "bc7359f2db2c7ba388e4ce250300fda061ad6c3f730396860014c45d15aa0eac",
    "scripts/openevo_webshop/build_carrier_health_v2.py": "15dae25602b5998c070e6bc43ac9e8dbdd693b05d2df38fc094f1d6d70f82d82",
    "scripts/openevo_webshop/carrier_contract_v2_identity.py": "2d2224aaa255ee19591bea14cdd1e5368feecfe73ba76f826727db5195719fa2",
    "scripts/openevo_webshop/carrier_contract_v2_adoption.py": "f3a02bf985888fb8113a2ac359ed8bc69105b7a38db2d1cee051eaa6a779aef5",
    "scripts/openevo_webshop/ceiling1_harness201_g8.py": "11ea0e28924ef22a157dcb1fe2b36524cefd8552d2d29962ea3d7c2939285296",
}
BASE_IMPORT_DEPS = set(VALIDATOR_SOURCES) - {
    "scripts/openevo_webshop/bounded_effective_state_recovery_202609282338.py",
    "scripts/openevo_webshop/ceiling1_stage2_arm_controller.py",
}


def sha(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"missing evidence: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    data = json.loads(path.read_bytes())
    if not isinstance(data, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return data


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError(reason)


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def bound(repo: Path, ref: dict) -> tuple[Path, dict]:
    rel = ref.get("path")
    require(isinstance(rel, str) and rel and not rel.startswith("/") and ".." not in Path(rel).parts,
            "invalid relative authority ref")
    path = repo / rel
    require(sha(path) == ref.get("sha256"), f"authority SHA drift: {path}")
    return path, load(path)


def validator_sources() -> dict:
    """Pin the A11 owner and every direct local module loaded by its base validator."""
    base_path = REPO / "scripts/openevo_webshop/ceiling1_stage2_arm_controller.py"
    direct = set()
    for node in ast.parse(base_path.read_text()).body:
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("scripts.openevo_webshop."):
            direct.add(node.module.replace(".", "/") + ".py")
    require(direct == BASE_IMPORT_DEPS, "base validator direct import graph drift")
    pinned = {}
    for rel, digest in VALIDATOR_SOURCES.items():
        path = REPO / rel
        require(sha(path) == digest, f"A11 validator source SHA drift: {rel}")
        committed = subprocess.check_output(["git", "-C", str(REPO), "show", f"{A11}:{rel}"])
        require(hashlib.sha256(committed).hexdigest() == digest, f"A11 source provenance drift: {rel}")
        pinned[rel] = {"sha256": digest, "source_git_sha": A11}
    return pinned


def build() -> dict:
    require(git("merge-base", BASE, "HEAD") == BASE, "candidate does not descend from PR #611 base")
    source_commit = git("log", "-1", "--format=%H", "--", str(REL_SCRIPT))
    require(len(source_commit) == 40 and git("show", f"{source_commit}:{REL_SCRIPT}") == SCRIPT.read_text().rstrip("\n"),
            "verifier bytes are not committed at source commit")
    sources = validator_sources()
    arm_path = REPO / ARM_REL
    release_path = REPO / RELEASE_REL
    prereg_path = REPO / PREREG_REL
    a10_release_path = REPO / A10_RELEASE_REL
    arm, release, prereg, a10_release = map(load, (arm_path, release_path, prereg_path, a10_release_path))
    require(arm.get("execution_git_sha") == A11 and arm.get("output_root") == str(ROOT), "A11 arm identity drift")
    require(release.get("execution_sha") == A11 and release.get("formal_task_consumption_allowed") is False,
            "A11 seal-only release drift")
    require(release.get("arm_authority") == {"path": ARM_REL, "sha256": sha(arm_path)}, "A11 arm ref drift")
    require(prereg.get("frozen_scientific_contract", {}).get("execution_sha") == A6, "frozen A6 contract drift")
    identity = release.get("experiment_identity")
    require(identity == a10_release.get("experiment_identity") and identity.get("experiment_id") == prereg.get("experiment_id"),
            "A10/A11 experiment or Passport identity drift")
    detail_ref = release.get("engineering_recovery")
    detail_path, detail = bound(REPO, detail_ref)
    require(arm.get("engineering_recovery", {}).get("sha256") == sha(detail_path), "A11 detail arm ref drift")
    require(detail.get("scientific_contract_changed") is False and detail.get("formal_rollout_replay") == 0
            and detail.get("already_consumed_rollouts") == 1024 and detail.get("remaining_rollout_budget") == 0
            and detail.get("predecessor_execution_git_sha") == A6
            and detail.get("engineering_predecessor_execution_git_sha") == A10
            and detail.get("successor_execution_git_sha") == A11, "A11 science/budget/lineage drift")
    a10_detail_path, a10_detail = bound(REPO, a10_release["engineering_recovery"])
    immutable_inputs = {
        "a10_recovery_detail": {"path": str(a10_detail_path), "sha256": sha(a10_detail_path)},
        "a10_arm_authority": {"path": detail["a10_arm_authority_path"], "sha256": detail["a10_arm_authority_sha256"]},
        "carrier_health_policy": {"path": a10_detail["carrier_health_policy_path"],
                                  "sha256": a10_detail["carrier_health_policy_sha256"]},
        "r152_resolution": {"path": a10_detail["resolution_receipt_path"],
                            "sha256": a10_detail["resolution_receipt_sha256"]},
        "r159_resolution": {"path": detail["resolution_receipt_path"],
                            "sha256": detail["resolution_receipt_sha256"]},
    }
    require(immutable_inputs["a10_arm_authority"]["sha256"] == "a999e0d6d3853098482aa468459a049b7d0b4b6031e4d739a384881f139d94aa"
            and immutable_inputs["carrier_health_policy"]["sha256"] == "856cc7b0f41ef4f0e698cb6565f12a0cba696a943446834de5c83aabfeda6377"
            and immutable_inputs["r152_resolution"]["sha256"] == "7768e7e32bee73d07a62bbe5789a990c1d016a21adc5f242d35dcdb52c67662c"
            and immutable_inputs["r159_resolution"]["sha256"] == "6e158d14548b3592831166c2bb2596e865c8cf0ced12e94e8fd68404446cd250",
            "immutable owner input identity drift")
    for name, ref in immutable_inputs.items():
        require(sha(Path(ref["path"])) == ref["sha256"], f"immutable owner input SHA drift: {name}")
    # The existing owner checks each of 1024 episode bytes, eight six-file round
    # manifests, state chain, real rollout producers, and health FAIL resolution.
    sys.path.insert(0, str(REPO))
    from scripts.openevo_webshop.bounded_effective_state_recovery_202609282338 import validate_a11_terminal_seal
    validate_a11_terminal_seal(arm, detail, check_live_state=False)

    seal_path = ROOT / "RANK128_SHORT_FORK_SEALED.json"
    state_path = ROOT / "execution-state.json"
    require(sha(seal_path) == SEAL_SHA and sha(state_path) == STATE_SHA, "postseal output SHA drift")
    seal, state = load(seal_path), load(state_path)
    require(seal.get("status") == "PASS" and seal.get("execution_git_sha") == A11
            and seal.get("post_advisor_short_fork_authority_sha256") == sha(arm_path)
            and seal.get("treatment_mode") == "MAGNITUDE_SCALAR_GATE_V1"
            and seal.get("persistent_state_rank") == 128 and seal.get("updates") == 8 and seal.get("noops") == 0
            and seal.get("terminal_carrier_health_status") == "FAIL_DIAGNOSTIC_RESOLVED"
            and seal.get("terminal_carrier_health_resolution_sha256") == detail["resolution_receipt_sha256"]
            and [row.get("round_index") for row in seal.get("rounds", [])] == list(range(152, 160))
            and seal.get("final_panel_access_count") == 0 and seal.get("external_teacher_calls") == 0,
            "A11 seal semantics drift")
    require(state.get("status") == "RANK128_SHORT_FORK_SEALED" and state.get("next_round") == 160
            and state.get("rollouts_consumed") == 20480 and state.get("state_id") == seal.get("final_state_id")
            and state.get("final_panel_access_count") == 0 and state.get("external_teacher_calls") == 0
            and not any("pause" in key or "blocker" in key for key in state), "postseal state/budget drift")
    recoveries = state.get("controller_recoveries") or []
    require(any(x.get("through_round") == 152 and x.get("resolution_receipt_sha256") == release["historical_r152_carrier_health_resolution"]["sha256"] for x in recoveries)
            and any(x.get("through_round") == 159 and x.get("resolution_receipt_sha256") == detail["resolution_receipt_sha256"] for x in recoveries),
            "diagnostic resolution chain drift")
    require(not list(OUTER.glob("round-*")), "A10 outer oe run unexpectedly contains scientific rounds")
    failures = {}
    for name, digest in FAIL_SHA.items():
        path = OUTER / name
        require(sha(path) == digest, f"A10 failed receipt SHA drift: {name}")
        data = load(path)
        if name.startswith("launch-receipt"):
            require(data.get("exit_code") == 1 and data.get("run_id") == OUTER.name,
                    f"A10 launch failure must remain exit 1: {name}")
            start = data.get("formal_start") or {}
            require(sha(Path(start.get("path", ""))) == str(start.get("sha256", "")).removeprefix("sha256:"),
                    f"A10 formal-start link drift: {name}")
        else:
            require(data.get("status") == "failed" and data.get("return_code") == 1,
                    "A10 GPU resource failure must stay failed")
        failures[name] = {"path": str(path), "sha256": digest}
    rounds = {}
    for r in range(152, 160):
        key = str(r)
        rr = ROOT / f"round-{r:03d}"
        complete = load(rr / "ROUND_COMPLETE.json")
        rollout = A6 if r == 152 else A10
        completion = A8 if r == 152 else A10
        require(complete.get("execution_git_sha") == completion, f"R{r} completion producer drift")
        after = load(rr / "state-after.json")
        sealed_round = seal["rounds"][r - 152]
        require(sealed_round.get("round_index") == r and sealed_round.get("state_id") == after.get("state_id")
                and sealed_round.get("adapter") == after.get("current_sd_adapter"),
                f"R{r} seal/state link drift")
        rows = detail["episode_shas"][key]
        require(len(rows) == 128, f"R{r} episode manifest drift")
        rounds[key] = {
            "rollout_producer_execution_sha": rollout,
            "round_complete_producer_execution_sha": completion,
            "episode_count": 128,
            "episode_shas_bound_by_a11_detail_sha256": sha(detail_path),
            "evidence": detail["round_evidence"][key],
        }
    require(set(rounds) == {str(r) for r in range(152, 160)}, "eight-round closure missing")
    return {
        "schema": "openevo.post-advisor.e1-postseal-reconciliation.20260930",
        "status": "E1_EXECUTION_SEALED_EVIDENCE_RECONCILED",
        "campaign_id": prereg["campaign_id"],
        "experiment_identity": identity,
        "frozen_preregistration": {"path": PREREG_REL, "sha256": sha(prereg_path), "execution_sha": A6},
        "a10_release": {"path": A10_RELEASE_REL, "sha256": sha(a10_release_path), "execution_sha": A10},
        "a11_release": {"path": RELEASE_REL, "sha256": sha(release_path), "execution_sha": A11},
        "a11_authority": {"path": ARM_REL, "sha256": sha(arm_path)},
        "a11_detail": {"path": str(detail_path), "sha256": sha(detail_path)},
        "validator_sources": sources,
        "immutable_validator_inputs": immutable_inputs,
        "science_output_root": str(ROOT),
        "a10_outer_oe_run_root": str(OUTER),
        "a10_failed_launch_receipts": failures,
        "rounds": rounds,
        "seal": {"path": str(seal_path), "sha256": SEAL_SHA, "producer_execution_sha": A11,
                 "terminal_carrier_health_status": "FAIL_DIAGNOSTIC_RESOLVED"},
        "postseal_execution_state": {"path": str(state_path), "sha256": STATE_SHA,
                                     "status": "RANK128_SHORT_FORK_SEALED", "next_round": 160},
        "r159_health": {"status": "FAIL", "ledger_sha256": detail["terminal_files"]["round-159/carrier-health-ledger.json"],
                        "pause_sha256": detail["terminal_files"]["carrier-health-pauses/round-159.json"],
                        "resolution_sha256": detail["resolution_receipt_sha256"]},
        "budget": {"e1_formal_rollouts_consumed": 1024, "e1_formal_rollouts_remaining": 0,
                   "formal_rollout_replay": 0, "final_panel_access": 0, "external_teacher_calls": 0},
        "future_e1_control": {"formal_task_consumption_allowed": False, "gpu_allocation_allowed": False,
                              "next_round": 160},
        "scientific_contract_changed": False,
        "claim_boundary": "E1 execution and provenance are sealed. R159 Carrier Health FAIL remains recorded; diagnostic resolution only permits terminal seal. Beta efficacy versus matched Bounded requires separate analysis. A10 outer exit-1 launches remain failures, not evidence of successful launches.",
        "verifier": {"path": str(REL_SCRIPT), "sha256": sha(SCRIPT), "source_git_sha": source_commit},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write-once", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    expected = build()
    rendered = (json.dumps(expected, sort_keys=True, indent=2) + "\n").encode()
    if args.write_once:
        require(RECEIPT.parent.is_dir(), "evidence owner directory missing")
        fd = os.open(RECEIPT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "wb") as out:
            out.write(rendered)
            out.flush()
            os.fsync(out.fileno())
    else:
        require(RECEIPT.is_file() and RECEIPT.read_bytes() == rendered, "committed postseal receipt drift")
    print(json.dumps({"status": "PASS", "receipt_path": str(RECEIPT),
                      "receipt_sha256": sha(RECEIPT), "episodes": 1024,
                      "formal_task_consumed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
