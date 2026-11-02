#!/usr/bin/env python3
"""Hash-bound CPU-only identity audit for the sealed Q17 R000-R121 snapshot.

Does not import the environment/model, rerun rollouts, change a selection, or read
any final panel. The historical metric pack is never rewritten.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

AUTHORITY_SHA256 = "df46c75aefeaf459418839875b98720c5a9b87d3f9d5afc61e941840c01f22dd"
FIRST_ROUND, LAST_ROUND = 0, 121

# Kept AST-identical to the frozen execution source; verify before auditing.
def task_identity(r:dict)->str: return f"{int(r['worker_seed'])}:{int(r['source_session_index'])}:{r['instruction_sha256']}"
def clean_exact(r:dict)->bool:
 return bool(float(r.get('final_task_score',-1))==1.0 and r.get('won') and r.get('identity_ok') and r.get('trace_complete') and not r.get('parser_or_fallback_failure') and not r.get('parser_repair_used') and not r.get('engineering_invalid') and not r.get('error'))

def verify_authority(path: Path, implementation: str) -> None:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != AUTHORITY_SHA256:
        raise ValueError("Frozen identity/clean-success source hash changed")
    def functions(text: str) -> dict[str, str]:
        return {n.name: ast.dump(n, include_attributes=False)
                for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)
                and n.name in {"task_identity", "clean_exact"}}
    if functions(data.decode()) != functions(implementation):
        raise ValueError("Audit predicates are not AST-identical to frozen science")

def parse_manifest(text: str) -> dict[str, str]:
    inputs: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid SHA256")
        if name.startswith("OUTPUT::"):
            continue
        if not re.fullmatch(r"round-[0-9]{3}/(ROUND_COMPLETE.json|attempts.jsonl|task-vector-geometry.json|sd-lora/dataset/selection.json|sd-lora/update.json)", name):
            raise ValueError("Input outside historical whitelist: " + name)
        if not FIRST_ROUND <= int(name[6:9]) <= LAST_ROUND or name in inputs:
            raise ValueError("Unsealed/out-of-scope or duplicate input: " + name)
        inputs[name] = digest
    return inputs

def compare_identities(selected: set[str], current: set[str], seen: set[str]) -> dict[str, Any]:
    return {"selected_equals_current_clean_exact": selected == current,
            "selected_subset_current_clean_exact": selected <= current,
            "selected_subset_cumulative_clean_exact": selected <= seen,
            "selected_current_overlap": sorted(selected & current),
            "selected_prior_only": sorted((selected - current) & seen),
            "selected_without_observed_clean_exact": sorted(selected - seen),
            "current_clean_exact_not_selected": sorted(current - selected)}

def audit(run_root: Path, manifest_text: str) -> dict[str, Any]:
    root = run_root.resolve(strict=True)
    expected = parse_manifest(manifest_text)
    checked, rounds, seen, errors = [], [], set(), []
    def read(relative: str) -> bytes:
        if relative not in expected:
            raise ValueError("Missing manifest binding: " + relative)
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("Input escapes run root: " + relative)
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected[relative]:
            raise ValueError("Input hash mismatch: " + relative)
        checked.append({"path": relative, "sha256": digest, "bytes": len(data)})
        return data
    for index in range(FIRST_ROUND, LAST_ROUND + 1):
        prefix = f"round-{index:03d}/"
        seal = json.loads(read(prefix + "ROUND_COMPLETE.json"))
        # JSONL records are LF-delimited; Unicode separators inside strings are data.
        attempts = [json.loads(line) for line in read(prefix + "attempts.jsonl").split(b"\n") if line.strip()]
        selection = json.loads(read(prefix + "sd-lora/dataset/selection.json"))
        summary = seal["rollout_summary"]
        if seal.get("status") != "PASS" or seal.get("round_index") != index:
            raise ValueError("Not the expected PASS sealed round")
        if len(attempts) != 128 or summary.get("attempt_count") != 128:
            raise ValueError("Historical round does not have exactly 128 attempts")
        if any(a.get("round_index") != index or a.get("final_panel_access") != 0 for a in attempts):
            raise ValueError("Attempt round/final-panel boundary mismatch")
        positions = [a["schedule_position"] for a in attempts]
        if len(set(positions)) != 128:
            raise ValueError("Duplicate rollout identity")
        current = {task_identity(a) for a in attempts if clean_exact(a)}
        clean_count = sum(clean_exact(a) for a in attempts)
        invalid_count = sum(bool(a.get("engineering_invalid")) for a in attempts)
        selected_list = selection.get("bound_task_identities", [])
        selected = set(selected_list)
        if len(selected_list) != len(selected) or len(selected) != selection.get("selected_task_count"):
            raise ValueError("Selection task identity/count inconsistency")
        seen.update(current)
        comparison = compare_identities(selected, current, seen)
        receipt_agrees = (len(current) == summary.get("clean_exact_success_task_count")
                          and clean_count == summary.get("clean_exact_success_count")
                          and invalid_count == summary.get("engineering_invalid_count"))
        if not receipt_agrees:
            errors.append({"round": index, "kind": "clean_exact_receipt_disagreement"})
        if not comparison["selected_subset_cumulative_clean_exact"]:
            errors.append({"round": index, "kind": "selected_identity_source_unproven"})
        rounds.append({"round": index, "attempt_count": len(attempts),
                       "clean_exact_attempt_count": clean_count,
                       "engineering_invalid_count": invalid_count,
                       "selected_task_count": len(selected),
                       "current_clean_exact_task_count": len(current),
                       "selected_task_identities": sorted(selected),
                       "current_clean_exact_task_identities": sorted(current),
                       "receipt_agrees": receipt_agrees, **comparison})
    return {"status": "PASS" if not errors else "HOLD", "round_start": FIRST_ROUND,
            "round_end": LAST_ROUND, "sealed_round_count": len(rounds),
            "rollout_count": sum(r["attempt_count"] for r in rounds),
            "engineering_invalid_count": sum(r["engineering_invalid_count"] for r in rounds),
            "exact_set_equality_rounds": sum(r["selected_equals_current_clean_exact"] for r in rounds),
            "current_set_subset_rounds": sum(r["selected_subset_current_clean_exact"] for r in rounds),
            "cumulative_set_subset_rounds": sum(r["selected_subset_cumulative_clean_exact"] for r in rounds),
            "selected_prior_only_count": sum(len(r["selected_prior_only"]) for r in rounds),
            "selected_unproven_count": sum(len(r["selected_without_observed_clean_exact"]) for r in rounds),
            "receipt_agreement_rounds": sum(r["receipt_agrees"] for r in rounds),
            "inputs": checked, "per_round": rounds, "errors": errors,
            "boundary": "Identity association only; not allocation efficacy, causation, full-GDR, or final evaluation."}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--authority-source", type=Path, required=True)
    args = parser.parse_args()
    verify_authority(args.authority_source, Path(__file__).read_text())
    result = audit(args.run_root, args.manifest.read_text())
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if result["status"] == "PASS" else 2

if __name__ == "__main__":
    raise SystemExit(main())
