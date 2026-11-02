#!/usr/bin/env python3
"""Zero-task engineering identity rebind for historical Carrier evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from scripts.openevo_webshop.carrier_contract_v2_identity import load_successor_identity  # legacy-revision-compat
from scripts.openevo_webshop.carrier_scope_package_v1 import load_package  # legacy-revision-compat
from scripts.openevo_webshop.carrier_health_failure_policy_v1 import load_policy, policy_identity  # legacy-revision-compat
from scripts.openevo_webshop.path_contract import load_and_validate

PREFLIGHT_SCHEMA = "openevo.carrier-engineering-preflight.202609300734"
ADOPTION_SCHEMA = "openevo.carrier-engineering-adoption.202609300734"
ROUTE_CONTRACTS = {
    "E3_RANK_CAPACITY": {
        "path": "configs/experiment/path-contracts/post-advisor-e3-rank-capacity-path-contract-202609300734.json",
        "sha256": "951161b186c5c22eef95a21d9762d1a8cd61c5983592bfdc5980299a8ec8e9c2",
        "schema": "openevo.e3-rank-capacity-checkout-path-contract.202609300734",
        "campaign_id": "20260928-2338-post-advisor-unanswered",
        "run_root": "/workspace/runs/post-advisor-unanswered-20260928",
    },
    "STAGE2_QWEN3": {
        "path": "configs/experiment/path-contracts/stage2-qwen3-path-contract-202609300734.json",
        "sha256": "8f818a4700c91f213e48ec505922fb6c5723e221cd4893e476335de0eb7fc4f2",
        "schema": "openevo.stage2-qwen3-checkout-path-contract.202609300734",
        "campaign_id": "202609030400-stage2-qwen3-engineering-successor",
        "run_root": "/workspace/runs/stage2-qwen3-engineering-202609300734",
    },
}
ROUTES = set(ROUTE_CONTRACTS)
SCOPE_EVIDENCE_INDEX = "docs/evidence/carrier-scope-package-1396472f16289fda5ac28f397ccc3e4394757ba4bb2083afd1e7247887726a8b/index.json"
SCOPE_EVIDENCE_SHA256 = "1396472f16289fda5ac28f397ccc3e4394757ba4bb2083afd1e7247887726a8b"
SCOPE_EVIDENCE_IDENTITY = "6a2f4c2f7131c5631c5a8680cd9a2f58094c8364100a4f67f1ba36f20b15e1ec"
SCOPE_PROVENANCE_PATH = "docs/evidence/carrier-scope-package-1396472f16289fda5ac28f397ccc3e4394757ba4bb2083afd1e7247887726a8b/PROVENANCE_202609300734.json"
SCOPE_PROVENANCE_SHA256 = "e834c96d58743c30e0a7979e5f37e634f50c2120b0e9f3bc77569dbec0cc34ff"
E3_PREDECESSOR = {
    "source_git_sha": "a84ad2e11fd057617d0ad6a0f7f615b6f7f00821",
    "release_git_sha": "cbafc5c94647fb0d20a40f189eeb3a2c45f8cb16",
    "profile_rebind_git_sha": "18ab77af3d3aaf39e11e8251cdb4655b9670f889",
    "inherited_activation_path": "configs/experiment/activations/post-advisor-e3-rank-capacity-activation-202609300248.json",
    "inherited_activation_sha256": "773428b780f920d1aad62fb12a6bacc5492c128217ee9c727e66760ec9dfa6b1",
    "release_path": "configs/experiment/activations/post-advisor-e3-rank-capacity-release-202609300248.json",
    "release_sha256": "e74cc69bfea7b3dd34b6cf89e3121002fb36f0d6c8cbd8adeeaeae6e453be882",
    "authority_path": "configs/experiment/activations/post-advisor-e3-rank-capacity-authority-202609300248.json",
    "authority_sha256": "cc99967989aaf8720d5f1ae3af3d91621066b531a79a700b9cfbafc2e23e9041",
    "required_successor_attempt": 9,
    "required_successor_execution_sha": "a84ad2e11fd057617d0ad6a0f7f615b6f7f00821",
    "required_successor_authority_sha": "18ab77af3d3aaf39e11e8251cdb4655b9670f889",
    "e1_current_campaign_sha256": "fce6a91875a8322fa382729d410e83a83a605609fcf82df9ecfff337bd02c2f2",
    "e1_reconciliation_sha256": "11abe12cab607d40bf05a54cd243bd1bcd1eeacd37d8475016aeac8120a364a8",
    "e1_seal_sha256": "b03d0cbcb9e5fc89a8d8974952222aa5d35ffb083bc6a47239cfa30b2b893725",
    "e1_release_sha256": "3eac9758b9f3ac8bf141bdfc611ef80cf31d93ca2010744141a0a68f4a2affde",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def historical_identity_and_adoption(
    repo: Path, historical_git_sha: str, identity_path: Path, adoption_path: Path
) -> tuple[str, str]:
    if sha(identity_path) != "5d42d64ef9b46b3f44c2194dff9004ba752f4db2c807a3b7656a35341dbde24c":
        raise ValueError("historical Carrier manifest file SHA drift")
    if sha(adoption_path) != "d9c2964a21d7d8e7565e62744e8716c6b92b9b844d47ca014d54e44ae83937a8":
        raise ValueError("historical Carrier adoption file SHA drift")
    identity = json.loads(identity_path.read_text())
    components = identity["identity_projection"]["components"]
    if len(components) != identity["component_file_count"] or len(components) != 25:
        raise ValueError("historical Carrier component closure drift")
    for rel, expected in components.items():
        blob = subprocess.check_output(["git", "show", f"{historical_git_sha}:{rel}"], cwd=repo)
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError(f"historical Carrier source mismatch: {rel}")
    receipt = json.loads(adoption_path.read_text())
    if receipt.get("status") != "PASS_INHERITED_COMMON_START_PROSPECTIVE_V2":  # legacy-revision-compat
        raise ValueError("historical Carrier adoption status drift")
    if receipt.get("carrier_contract_identity_sha256") != identity["carrier_contract_identity_sha256"]:
        raise ValueError("historical Carrier adoption identity drift")
    return sha(identity_path), sha(adoption_path)


def verify_e3_predecessor(repo: Path) -> dict[str, Any]:
    bound = E3_PREDECESSOR
    source = bound["source_git_sha"]
    release = bound["release_git_sha"]
    profile_rebind = bound["profile_rebind_git_sha"]
    for child, parent in ((release, source), (profile_rebind, release)):
        parents = subprocess.check_output(
            ["git", "rev-list", "--parents", "-n", "1", child], cwd=repo, text=True
        ).split()
        if parents != [child, parent]:
            raise ValueError("E3 predecessor source/B20/B21 lineage drift")
    changed = set(subprocess.check_output(
        ["git", "diff", "--name-only", release, profile_rebind], cwd=repo, text=True
    ).splitlines())
    if bound["inherited_activation_path"] in changed:
        raise ValueError("E3 B21 unexpectedly changed inherited activation")
    for label in ("inherited_activation", "release", "authority"):
        blob = subprocess.check_output(
            ["git", "show", f"{profile_rebind}:{bound[label + '_path']}"], cwd=repo
        )
        if hashlib.sha256(blob).hexdigest() != bound[label + "_sha256"]:
            raise ValueError(f"E3 B21 {label} blob drift")
        if label == "inherited_activation":
            state = json.loads(blob)
            if (state.get("status") != "PREPARED_NOT_ACTIVE"
                    or state.get("formal_task_consumption_allowed") is not False
                    or state.get("execution_sha") != "775249849c9c2fa63bd3ffe3a1946507a0be9502"
                    or state.get("authority_git_sha") != "a4d4b457b842ef90024661cef0940640ada40564"
                    or "attempt-0008" not in json.dumps(state.get("pending_evidence"))):
                raise ValueError("E3 B21 inherited activation is not stale C19 attempt eight")
    campaign_path = repo / "configs/experiment/current-campaign.json"
    if sha(campaign_path) != bound["e1_current_campaign_sha256"]:
        raise ValueError("E3 predecessor E1 postseal campaign bytes drift")
    campaign = json.loads(campaign_path.read_text())
    latest = campaign.get("latest_event") or {}
    if (campaign.get("status") != "execution-sealed"
            or campaign.get("formal_task_consumption_allowed") is not False
            or campaign.get("gpu_allocation_allowed") is not False
            or campaign.get("remaining_e1_rollout_budget") != 0
            or latest.get("already_consumed_rollouts") != 1024
            or latest.get("formal_rollout_replay") != 0
            or latest.get("reconciliation_sha256") != bound["e1_reconciliation_sha256"]
            or latest.get("seal_sha256") != bound["e1_seal_sha256"]
            or campaign.get("release_receipt_sha256") != bound["e1_release_sha256"]):
        raise ValueError("E3 predecessor E1 postseal scope drift")
    reconciliation_ref = campaign.get("reconciliation") or {}
    if (reconciliation_ref.get("path") != "docs/evidence/post-advisor-unanswered-20260928/E1_POSTSEAL_RECONCILIATION_20260930.json"
            or reconciliation_ref.get("sha256") != bound["e1_reconciliation_sha256"]):
        raise ValueError("E3 predecessor E1 reconciliation reference drift")
    reconciliation_path = repo / reconciliation_ref["path"]
    if sha(reconciliation_path) != bound["e1_reconciliation_sha256"]:
        raise ValueError("E3 predecessor E1 reconciliation bytes drift")
    reconciliation = json.loads(reconciliation_path.read_text())
    seal_ref = reconciliation.get("seal") or {}
    if (seal_ref.get("path") != "/workspace/runs/post-advisor-unanswered-20260928/e1-beta-direction-formal-a6-6b4ff450/RANK128_SHORT_FORK_SEALED.json"
            or seal_ref.get("sha256") != bound["e1_seal_sha256"]
            or sha(Path(seal_ref["path"])) != bound["e1_seal_sha256"]):
        raise ValueError("E3 predecessor E1 terminal seal bytes drift")
    release_ref = campaign.get("release") or {}
    if (release_ref.get("path") != "configs/experiment/activations/post-advisor-e1-a11-terminal-seal-release-202609302100.json"
            or release_ref.get("sha256") != bound["e1_release_sha256"]
            or sha(repo / release_ref["path"]) != bound["e1_release_sha256"]):
        raise ValueError("E3 predecessor E1 A11 release bytes drift")
    return {**bound, "status": "B21_RELEASE_SNAPSHOT_C20_ACTIVATION_UNCOMMITTED",
            "attempt_nine_qualification_validated": False,
            "formal_task_consumption_allowed": False,
            "execution_authority": False}


def validate_route_contract(repo: Path, route: str, path: Path) -> dict[str, Any]:
    expected = ROUTE_CONTRACTS.get(route)
    if expected is None:
        raise ValueError("unsupported Carrier successor route")
    if path.resolve() != (repo / expected["path"]).resolve() or sha(path) != expected["sha256"]:
        raise ValueError("Carrier successor route contract path/SHA drift")
    contract = json.loads(path.read_text())
    if (contract.get("schema") != expected["schema"]
            or contract.get("campaign_id") != expected["campaign_id"]
            or (contract.get("paths") or {}).get("run_root", {}).get("host_path") != expected["run_root"]
            or (contract.get("paths") or {}).get("experiment", {}).get("host_path") != str(repo)):
        raise ValueError("Carrier successor route contract scope drift")
    if route == "STAGE2_QWEN3" and contract["paths"]["run_root"].get("must_exist") is not False:
        raise ValueError("Stage2 successor requires an independent run root")
    return contract


def build_preflight(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve(strict=True)
    if args.route not in ROUTES:
        raise ValueError("unsupported Carrier successor route")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo, text=True).strip()
    if head != args.source_git_sha or dirty:
        raise ValueError("Carrier successor source checkout is not clean at its bound commit")
    identity = load_successor_identity(args.identity, args.source_map, repo)
    if identity.get("execution_authority") is not False:
        raise ValueError("Carrier source identity cannot authorize execution")
    old_identity_sha, old_adoption_sha = historical_identity_and_adoption(
        repo, args.historical_git_sha, args.historical_identity, args.historical_adoption
    )
    if (args.scope_package.resolve() != (repo / SCOPE_EVIDENCE_INDEX).resolve()
            or sha(args.scope_package) != SCOPE_EVIDENCE_SHA256):
        raise ValueError("Carrier historical scope evidence index path/SHA drift")
    scope = load_package(args.scope_package)
    if scope["package_identity_sha256"] != SCOPE_EVIDENCE_IDENTITY:
        raise ValueError("Carrier historical scope package identity drift")
    provenance_path = repo / SCOPE_PROVENANCE_PATH
    if sha(provenance_path) != SCOPE_PROVENANCE_SHA256:
        raise ValueError("Carrier scope archive provenance bytes drift")
    provenance = json.loads(provenance_path.read_text())
    if (provenance.get("evidence_only") is not True
            or provenance.get("execution_authority") is not False
            or provenance.get("replay_authorized") is not False
            or provenance.get("snapshot_index_path") != SCOPE_EVIDENCE_INDEX
            or provenance.get("index_file_sha256") != SCOPE_EVIDENCE_SHA256
            or provenance.get("package_identity_sha256") != SCOPE_EVIDENCE_IDENTITY
            or provenance.get("archive_file_count") != 161
            or provenance.get("round_count") != 160
            or (provenance.get("consuming_adoption_receipt") or {}).get("path") != str(args.historical_adoption)
            or (provenance.get("consuming_adoption_receipt") or {}).get("sha256") != sha(args.historical_adoption)):
        raise ValueError("Carrier scope archive provenance scope drift")
    old_adoption = json.loads(args.historical_adoption.read_text())
    if old_adoption.get("scope_package_identity_sha256") != scope["package_identity_sha256"]:
        raise ValueError("historical Carrier scope package drift")
    policy = load_policy(args.health_policy)
    projection = identity["identity_projection"]
    if policy_identity(policy) != projection["health_policy_identity_sha256"]:
        raise ValueError("Carrier health policy identity drift")
    if hashlib.sha256(scope["construction_policy_id"].encode()).hexdigest() != projection["scope_policy_identity_sha256"]:
        raise ValueError("Carrier scope policy identity drift")
    e3_predecessor = verify_e3_predecessor(repo) if args.route == "E3_RANK_CAPACITY" else None
    contract = validate_route_contract(repo, args.route, args.path_contract)
    if Path(contract["paths"]["experiment"]["host_path"]).resolve(strict=True) != repo:
        raise ValueError("Carrier successor path contract checkout mismatch")
    load_and_validate(args.path_contract)
    source_map = json.loads(args.source_map.read_text())
    for row in source_map["components"]:
        blob = subprocess.check_output(
            ["git", "show", f"{args.source_git_sha}:{row['original_path']}"], cwd=repo
        )
        if hashlib.sha256(blob).hexdigest() != row["sha256"]:
            raise ValueError(f"Carrier successor source commit drift: {row['original_path']}")
    contract_blob = subprocess.check_output(
        ["git", "show", f"{args.source_git_sha}:{args.path_contract.resolve().relative_to(repo)}"], cwd=repo
    )
    if hashlib.sha256(contract_blob).hexdigest() != sha(args.path_contract):
        raise ValueError("Carrier successor path contract commit drift")
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "PASS_ENGINEERING_IDENTITY",
        "route": args.route,
        "source_git_sha": args.source_git_sha,
        "historical_git_sha": args.historical_git_sha,
        "identity_manifest_sha256": sha(args.identity),
        "source_map_sha256": sha(args.source_map),
        "historical_identity_sha256": old_identity_sha,
        "historical_adoption_sha256": old_adoption_sha,
        "path_contract_sha256": sha(args.path_contract),
        "e3_predecessor": e3_predecessor,
        "scope_package_identity_sha256": scope["package_identity_sha256"],
        "scope_evidence_index_sha256": sha(args.scope_package),
        "scope_evidence_provenance_sha256": sha(provenance_path),
        "health_policy_identity_sha256": policy_identity(policy),
        "formal_task_consumption": 0,
        "external_teacher_calls": 0,
        "final_panel_access": 0,
        "execution_authority": False,
    }


def build_adoption(args: argparse.Namespace) -> dict[str, Any]:
    bindings = {}
    for route, path_contract, preflight in (
        ("E3_RANK_CAPACITY", args.e3_path_contract, args.e3_preflight),
        ("STAGE2_QWEN3", args.stage2_path_contract, args.stage2_preflight),
    ):
        if path_contract is None or preflight is None:
            raise ValueError(f"Carrier successor adoption missing input for {route}")
        route_args = argparse.Namespace(**{**vars(args), "route": route, "path_contract": path_contract})
        if json.loads(preflight.read_text()) != build_preflight(route_args):
            raise ValueError(f"Carrier successor preflight drift for {route}")
        bindings[route] = {"path_contract_sha256": sha(path_contract), "preflight_sha256": sha(preflight)}
    identity = load_successor_identity(args.identity, args.source_map, args.repo.resolve(strict=True))
    return {
        "schema": ADOPTION_SCHEMA,
        "status": "PASS_ENGINEERING_REBIND",
        "source_git_sha": args.source_git_sha,
        "carrier_contract_identity_sha256": identity["carrier_contract_identity_sha256"],
        "identity_manifest_sha256": sha(args.identity),
        "source_map_sha256": sha(args.source_map),
        "route_bindings": bindings,
        "formal_task_consumption": 0,
        "external_teacher_calls": 0,
        "final_panel_access": 0,
        "execution_authority": False,
    }


def write_immutable(path: Path, value: dict[str, Any]) -> None:
    rendered = json.dumps(value, sort_keys=True, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as handle:
            handle.write(rendered)
    except FileExistsError:
        if path.read_text() != rendered:
            raise ValueError(f"immutable Carrier successor receipt conflict: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--source-map", type=Path, required=True)
    parser.add_argument("--historical-identity", type=Path, required=True)
    parser.add_argument("--historical-adoption", type=Path, required=True)
    parser.add_argument("--historical-git-sha", required=True)
    parser.add_argument("--scope-package", type=Path, required=True)
    parser.add_argument("--health-policy", type=Path, required=True)
    parser.add_argument("--path-contract", type=Path)
    parser.add_argument("--e3-path-contract", type=Path)
    parser.add_argument("--stage2-path-contract", type=Path)
    parser.add_argument("--e3-preflight", type=Path)
    parser.add_argument("--stage2-preflight", type=Path)
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--route", choices=sorted(ROUTES))
    parser.add_argument("--adoption", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.adoption:
        expected = build_adoption(args)
    else:
        if args.route is None or args.path_contract is None:
            raise ValueError("Carrier successor preflight requires route and path contract")
        expected = build_preflight(args)
    if args.validate:
        if json.loads(args.output.read_text()) != expected:
            raise ValueError("Carrier successor preflight receipt drift")
    else:
        write_immutable(args.output, expected)
    print(json.dumps({"status": "PASS", "route": "ADOPTION" if args.adoption else args.route, "receipt_sha256": sha(args.output), "execution_authority": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
