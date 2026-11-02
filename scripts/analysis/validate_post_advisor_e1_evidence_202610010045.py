#!/usr/bin/env python3
"""Read-only integrity check for a historical E1 evidence bundle, never a live gate.

Only files below the bundle are opened. Historical host paths, runtime modules,
and source URLs are descriptive provenance; this script never follows them.
PASS means preserved snapshot integrity and recorded-accounting consistency.
It does not independently revalidate server leaves or scientific efficacy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

BUNDLE = Path("docs/evidence/portable-runs/post-advisor-e1-sealed-202610010045")
SOURCE_SHA = "0a7e415f5309d9faf20807b74c730af2d381d61e"
RECEIPT_SHA = "11abe12cab607d40bf05a54cd243bd1bcd1eeacd37d8475016aeac8120a364a8"
BINDINGS = {
    "reconciliation": [],
    "preregistration": ["frozen_preregistration", "sha256"],
    "a10-release": ["a10_release", "sha256"],
    "a10-authority": ["immutable_validator_inputs", "a10_arm_authority", "sha256"],
    "a10-detail": ["immutable_validator_inputs", "a10_recovery_detail", "sha256"],
    "a11-release": ["a11_release", "sha256"],
    "a11-authority": ["a11_authority", "sha256"],
    "a11-detail": ["a11_detail", "sha256"],
    "r152-resolution": ["immutable_validator_inputs", "r152_resolution", "sha256"],
    "r159-resolution": ["immutable_validator_inputs", "r159_resolution", "sha256"],
    "health-policy": ["immutable_validator_inputs", "carrier_health_policy", "sha256"],
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def confined_file(root: Path, relative: str) -> Path:
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, "unsafe archive path")
    target = (root / path).resolve()
    require(target.is_relative_to(root.resolve()), "archive escaped bundle")
    require(target.is_file(), f"missing archive: {relative}")
    return target


def validate(bundle: Path) -> dict[str, object]:
    index = json.loads(confined_file(bundle, "index.json").read_bytes())
    require(index["schema"] == "openevo.post-advisor.e1-evidence-bundle.202610010045", "bundle schema drift")
    require(index["role"] == "immutable-evidence-only", "bundle role drift")
    require(index["execution_authority"] is False, "evidence cannot authorize execution")
    require(index["scientific_efficacy_assessed"] is False, "efficacy claim is unsupported")
    require(index["source_git_sha"] == SOURCE_SHA, "source head drift")
    require(index["reconciliation_sha256"] == RECEIPT_SHA, "reconciliation identity drift")
    require(index["external_reference_scope"], "missing external-reference limitation")
    entries = index["entries"]
    require(len(entries) == len(BINDINGS), "snapshot count drift")
    require({row["role"] for row in entries} == set(BINDINGS), "snapshot roles drift")
    payloads = {}
    for row in entries:
        role = row["role"]
        require(row["execution_authority"] is False, "snapshot cannot authorize execution")
        require(row["source_git_sha"] == SOURCE_SHA, "snapshot source head drift")
        require(row["receipt_binding"] == BINDINGS[role], "binding drift")
        expected_path = "reconciliation.json" if role == "reconciliation" else f"snapshots/{row['sha256']}.json"
        require(row["archive_path"] == expected_path, "archive path is not content-addressed")
        data = confined_file(bundle, row["archive_path"]).read_bytes()
        require(hashlib.sha256(data).hexdigest() == row["sha256"], f"snapshot digest drift: {role}")
        require(len(data) == row["size_bytes"], f"snapshot size drift: {role}")
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        require(blob == row["source_blob_sha"], f"source Git blob drift: {role}")
        require(row["source_url"] == f"https://github.com/mykcs/openevo-experiment/blob/{SOURCE_SHA}/{row['source_path']}", "source URL drift")
        payloads[role] = json.loads(data)
    receipt = payloads["reconciliation"]
    for row in entries:
        value = receipt
        for key in BINDINGS[row["role"]]:
            value = value[key]
        if row["role"] == "reconciliation":
            require(row["sha256"] == RECEIPT_SHA, "root receipt drift")
            require(row["consuming_receipt_sha256"] is None, "root receipt consumer drift")
            require(row["source_path"] == "docs/evidence/post-advisor-unanswered-20260928/E1_POSTSEAL_RECONCILIATION_20260930.json", "historical source path drift")
        else:
            require(row["sha256"] == value, f"receipt digest binding drift: {row['role']}")
            require(row["consuming_receipt_sha256"] == RECEIPT_SHA, "consuming receipt drift")
            reference = receipt
            for key in BINDINGS[row["role"]][:-1]:
                reference = reference[key]
            historical_path = reference["path"]
            expected_source = historical_path if historical_path.startswith("configs/") else "configs/" + historical_path.split("/configs/", 1)[1]
            require(row["source_path"] == expected_source, "historical source path drift")
    require(receipt["budget"] == {
        "e1_formal_rollouts_consumed": 1024, "e1_formal_rollouts_remaining": 0,
        "external_teacher_calls": 0, "final_panel_access": 0, "formal_rollout_replay": 0,
    }, "recorded budget drift")
    require(receipt["r159_health"]["status"] == "FAIL", "R159 failure must remain visible")
    require(receipt["seal"]["terminal_carrier_health_status"] == "FAIL_DIAGNOSTIC_RESOLVED", "seal is not a health PASS")
    require(receipt["future_e1_control"] == {
        "formal_task_consumption_allowed": False, "gpu_allocation_allowed": False, "next_round": 160,
    }, "sealed E1 control drift")
    detail = payloads["a11-detail"]
    rounds = {str(i) for i in range(152, 160)}
    require(set(receipt["rounds"]) == set(detail["round_evidence"]) == set(detail["episode_shas"]) == rounds, "round coverage drift")
    for number in sorted(rounds):
        record = receipt["rounds"][number]
        episodes = detail["episode_shas"][number]
        require(record["episode_count"] == len(episodes) == 128, f"episode count drift: {number}")
        require(all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in episodes.values()), "invalid episode hash")
        require(record["evidence"] == detail["round_evidence"][number], f"round manifest drift: {number}")
        require(record["episode_shas_bound_by_a11_detail_sha256"] == receipt["a11_detail"]["sha256"], "episode binding drift")
        rollout = "6b4ff4503adc9494c949f654521f09ee52ec28b4" if number == "152" else "3181cb7440d48633685d9c675a1f4684e37500ce"
        complete = "31a669ab9af34fdf5733849ea8ccfb7af37e7427" if number == "152" else rollout
        require(record["rollout_producer_execution_sha"] == rollout, "rollout producer drift")
        require(record["round_complete_producer_execution_sha"] == complete, "completion producer drift")
    return {
        "status": "PASS_EVIDENCE_BUNDLE_INTEGRITY", "snapshots": len(entries),
        "recorded_rounds": 8, "episode_hash_references": 1024,
        "server_leaves_revalidated": False, "scientific_efficacy_assessed": False,
        "execution_authority": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parents[2] / BUNDLE)
    args = parser.parse_args()
    try:
        result = validate(args.bundle)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "FAIL_EVIDENCE_BUNDLE_INTEGRITY", "error": str(exc)}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
