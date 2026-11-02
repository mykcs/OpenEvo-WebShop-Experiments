#!/usr/bin/env python3
"""One-time bridge from historical SEED WebShop v2 evidence to successor handoff.

The generic successor-handoff machinery lives in ``successor_handoff.py``.  This
module supplies only the benchmark-specific historical bridge so a successor
Agent does not have to rediscover ``base-P1``, launch scripts, neighboring SEED
checkouts, adapter identity, task identity, or protocol evidence.

The migration is deliberately CPU-only.  It verifies immutable source/data
identity, carries already-archived predecessor evidence, optionally carries a
runtime qualification receipt only as a CACHE_CANDIDATE, captures a normal
``openevo.successor-handoff.v1`` receipt, and immediately bootstraps it.

It never grants GPU allocation or formal task-consumption authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.openevo_webshop import successor_handoff as sh


REPO_ROOT = Path(__file__).resolve().parents[2]
PREDECESSOR_PHASE = "seed-official-heldout-comparison-v1:v2"
SUCCESSOR_PHASE = "seed-webshop-public-code-reproduction-v1"
PINNED_SEED_COMMIT = "2cf2fadca3c5aba28da68e8e1405182ba8d90e6c"

# Main-repository evidence revision that added the repaired PRIMARY-v2 cell
# summaries (including the per-arm protocol fingerprints) and finalized the v2
# result wording/accounting.  This is evidence provenance, not runtime state.
PHASE1_EVIDENCE_REVISION = "4db98b084619d876739f125e9f438e3edac6c67e"

# Git blob object ids from jinyangwu/SEED at PINNED_SEED_COMMIT.  Recomputing
# Git blob ids locally proves byte-for-byte equality without requiring the
# materialized historical ``base-P1`` tree to contain .git metadata.
SOURCE_BLOBS: dict[str, str] = {
    "agent_system/environments/env_package/webshop/envs.py": (
        "c9b43e29a5643f63eee04cba3327623623815a9f"
    ),
    "agent_system/environments/env_package/webshop/projection.py": (
        "a3da8455501155f0d618ec0147e3ce079c53814a"
    ),
    "agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/"
    "web_agent_text_env.py": "8d18307010408f3ea81974221b76587a60f37b28",
    "agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/"
    "engine.py": "2e5eee3f5fcd96a4acd7defbeef4bef966324e4b",
    "agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/"
    "goal.py": "09ad26819db8b90ec09be24fa542c2fd387e725d",
}

DATA_SHA256: dict[str, str] = {
    "agent_system/environments/env_package/webshop/webshop/data/"
    "items_shuffle_1000.json": (
        "30a4765c3a327af72d9a9a95a6b2486d516f0fa1d3ecd83681901ce82a21b269"
    ),
    "agent_system/environments/env_package/webshop/webshop/data/"
    "items_ins_v2_1000.json": (
        "f88a36314a397b53b3d9c3fa5878e5f7b26d35019a51ec83fbedeca61a948f6f"
    ),
}

# These identities are copied from already-archived predecessor evidence rather
# than invented migration-local fingerprints:
# - PREREGISTRATION.md / results/qualification-receipt.json
# - v2/reconciliation-v2.json
# - cell-summaries/primary-v2-{base,sd-lora}-summary.json
ADAPTER_SHA256 = "47e63d417f85cb2defed5ff1ee934128b96e6a4f78a8f410fa052957d919190c"
CAMPAIGN_CONFIG_SHA256 = "2893addbc45c069f43b5e495abb081774fe8c1f2ca4bd43e8fe61120f05c1de0"
TASK_MANIFEST_FILE_SHA256 = "fe902f9610f4f43d7d183c6c57a96084bcb5bf07efacfe5df77942da97eb7635"
SELECTED_PANEL_DIGEST = "29d93cac4fa084e28ddad7747ca2e83623bc20af1129b44ebac8ef591ec96db3"
GOALS_LIST_DIGEST = "6257c56d133e465c4e8381e6651d5027fa840b011586d890d15bcc3926c446ed"
PRIMARY_V2_PROTOCOL_FINGERPRINTS: dict[str, str] = {
    "base": "a46142e7ee7a11ef8d40385ce526bf429fbc8112efd605865d9affae19e00476",
    "sd-lora": "819f8a93f99a573261a157733be6c44624759ad430c92639b8dd44279082eb1a",
}
GOALS_TOTAL = 6910
RUNTIME_QUAL_SCHEMA = "openevo.runtime-qualification.v1"


class MigrationError(RuntimeError):
    """Historical evidence cannot be migrated without weakening provenance."""


def git_blob_sha1(path: Path) -> str:
    """Return the Git blob object id for ``path`` without needing a git repo."""
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()  # noqa: S324 - Git object identity


def _require_file(path: Path, *, role: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise MigrationError(f"{role}: missing file: {resolved}")
    return resolved


def verify_pinned_seed_root(
    seed_root: Path,
    *,
    source_blobs: Mapping[str, str] = SOURCE_BLOBS,
    data_sha256: Mapping[str, str] = DATA_SHA256,
) -> dict[str, Any]:
    """Fail closed unless the materialized tree is the pinned task/data identity."""
    root = seed_root.expanduser().resolve()
    if not root.is_dir():
        raise MigrationError(f"seed root is missing: {root}")

    source_checks: list[dict[str, str]] = []
    for relative, expected_blob in source_blobs.items():
        path = _require_file(root / relative, role=f"source:{relative}")
        observed_blob = git_blob_sha1(path)
        if observed_blob != expected_blob:
            raise MigrationError(
                f"source:{relative}: pinned Git blob mismatch: "
                f"expected {expected_blob}, observed {observed_blob}"
            )
        source_checks.append(
            {
                "path": relative,
                "git_blob_sha1": observed_blob,
                "sha256": sh.sha256_file(path),
            }
        )

    data_checks: list[dict[str, str]] = []
    for relative, expected_hex in data_sha256.items():
        path = _require_file(root / relative, role=f"data:{relative}")
        expected = f"sha256:{expected_hex}"
        observed = sh.sha256_file(path)
        if observed != expected:
            raise MigrationError(
                f"data:{relative}: SHA-256 mismatch: "
                f"expected {expected}, observed {observed}"
            )
        data_checks.append({"path": relative, "sha256": observed})

    return {
        "seed_root": str(root),
        "pinned_seed_commit": PINNED_SEED_COMMIT,
        "source_checks": source_checks,
        "data_checks": data_checks,
    }


def _read_pass_qualification(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("schema") == RUNTIME_QUAL_SCHEMA
        and payload.get("status") == "PASS"
    )


def discover_runtime_qualification(seed_root: Path) -> Path | None:
    """Bounded lookup; ambiguity returns None instead of choosing by checkout name."""
    root = seed_root.expanduser().resolve()
    phase_root = root.parent.parent if root.parent.name == "seed-roots" else root.parent

    # Search only inside the predecessor phase root.  A neighboring checkout
    # (including one named 9996427) has no authority over source identity.
    valid = sorted(
        {
            path.resolve()
            for path in phase_root.glob("runtime-*v2*/**/qualification-receipt.json")
            if path.is_file() and _read_pass_qualification(path)
        }
    )
    return valid[0] if len(valid) == 1 else None


def build_spec(
    seed_root: Path,
    *,
    runtime_qualification: Path | None = None,
    source_blobs: Mapping[str, str] = SOURCE_BLOBS,
    data_sha256: Mapping[str, str] = DATA_SHA256,
) -> dict[str, Any]:
    root = seed_root.expanduser().resolve()
    verification = verify_pinned_seed_root(
        root, source_blobs=source_blobs, data_sha256=data_sha256
    )

    facts: list[dict[str, Any]] = [
        {
            "role": "seed-upstream-revision",
            "kind": "identity",
            "lifetime": "immutable",
            "value": PINNED_SEED_COMMIT,
        },
        {
            "role": "seed-source-faithfulness",
            "kind": "identity",
            "lifetime": "immutable",
            "value": {
                "upstream_commit": PINNED_SEED_COMMIT,
                "git_blob_sha1_by_path": dict(source_blobs),
            },
        },
        {
            "role": "frozen-adapter-sha256",
            "kind": "identity",
            "lifetime": "immutable",
            "value": f"sha256:{ADAPTER_SHA256}",
        },
        {
            "role": "predecessor-campaign-config-sha256",
            "kind": "identity",
            "lifetime": "immutable",
            "value": f"sha256:{CAMPAIGN_CONFIG_SHA256}",
        },
        {
            "role": "predecessor-task-manifest-sha256",
            "kind": "identity",
            "lifetime": "immutable",
            "value": f"sha256:{TASK_MANIFEST_FILE_SHA256}",
        },
        {
            "role": "predecessor-selected-panel-digest",
            "kind": "identity",
            "lifetime": "immutable",
            "value": f"sha256:{SELECTED_PANEL_DIGEST}",
        },
        {
            "role": "predecessor-goals-list-digest",
            "kind": "identity",
            "lifetime": "immutable",
            "value": f"sha256:{GOALS_LIST_DIGEST}",
        },
        {
            "role": "primary-v2-protocol-fingerprints",
            "kind": "identity",
            "lifetime": "immutable",
            "value": dict(PRIMARY_V2_PROTOCOL_FINGERPRINTS),
        },
        {
            "role": "webshop-goals-total",
            "kind": "identity",
            "lifetime": "immutable",
            "value": GOALS_TOTAL,
        },
    ]

    for relative in source_blobs:
        facts.append(
            {
                "role": f"seed-task-identity:{relative}",
                "kind": "file",
                "lifetime": "immutable",
                "path": str((root / relative).resolve()),
            }
        )

    for relative, expected_hex in data_sha256.items():
        facts.append(
            {
                "role": f"webshop-data:{Path(relative).name}",
                "kind": "file",
                "lifetime": "immutable",
                "path": str((root / relative).resolve()),
                "sha256": f"sha256:{expected_hex}",
            }
        )

    if runtime_qualification is not None:
        qualification = _require_file(
            runtime_qualification, role="runtime-qualification"
        )
        if not _read_pass_qualification(qualification):
            raise MigrationError(
                "runtime-qualification must be a PASS "
                f"{RUNTIME_QUAL_SCHEMA} receipt: {qualification}"
            )
        facts.append(
            {
                "role": "runtime-qualification",
                "kind": "receipt",
                "lifetime": "runtime-qualified",
                "path": str(qualification),
                "required_schema": RUNTIME_QUAL_SCHEMA,
                "required_status": "PASS",
            }
        )

    return {
        "schema": sh.SPEC_SCHEMA,
        "predecessor": {
            "phase_id": PREDECESSOR_PHASE,
            "evidence_revision": PHASE1_EVIDENCE_REVISION,
        },
        "successor": {"phase_id": SUCCESSOR_PHASE},
        "facts": facts,
        "migration_proof": verification,
    }


def handoff_required_roles(
    *,
    source_blobs: Mapping[str, str] = SOURCE_BLOBS,
    data_sha256: Mapping[str, str] = DATA_SHA256,
) -> tuple[str, ...]:
    """Return every immutable evidence role that the migrated handoff must retain."""
    identity_roles = (
        "seed-upstream-revision",
        "seed-source-faithfulness",
        "frozen-adapter-sha256",
        "predecessor-campaign-config-sha256",
        "predecessor-task-manifest-sha256",
        "predecessor-selected-panel-digest",
        "predecessor-goals-list-digest",
        "primary-v2-protocol-fingerprints",
        "webshop-goals-total",
    )
    source_roles = tuple(
        f"seed-task-identity:{relative}" for relative in source_blobs
    )
    data_roles = tuple(
        f"webshop-data:{Path(relative).name}" for relative in data_sha256
    )
    return identity_roles + source_roles + data_roles


def migrate(
    seed_root: Path,
    *,
    output_dir: Path,
    runtime_qualification: Path | None = None,
    auto_discover_runtime_qualification: bool = True,
    source_blobs: Mapping[str, str] = SOURCE_BLOBS,
    data_sha256: Mapping[str, str] = DATA_SHA256,
) -> dict[str, Any]:
    root = seed_root.expanduser().resolve()
    qualification = runtime_qualification
    if qualification is None and auto_discover_runtime_qualification:
        qualification = discover_runtime_qualification(root)

    output = output_dir.expanduser().resolve()
    spec_path = output / "seed-webshop-v2.successor-handoff-spec.json"
    handoff_path = output / "seed-webshop-v2.successor-handoff.json"
    bootstrap_path = output / "seed-webshop-v2.successor-bootstrap.json"

    spec = build_spec(
        root,
        runtime_qualification=qualification,
        source_blobs=source_blobs,
        data_sha256=data_sha256,
    )
    sh.atomic_json(spec_path, spec)
    receipt = sh.capture(spec_path, root=REPO_ROOT)
    sh.atomic_json(handoff_path, receipt)

    bootstrap = sh.bootstrap(
        handoff_path,
        root=REPO_ROOT,
        required_roles=handoff_required_roles(
            source_blobs=source_blobs,
            data_sha256=data_sha256,
        ),
    )
    sh.atomic_json(bootstrap_path, bootstrap)
    if bootstrap.get("status") != "READY_FOR_SUCCESSOR_PREP":
        raise MigrationError(
            "captured handoff did not bootstrap cleanly: "
            + json.dumps(bootstrap.get("failures", []), sort_keys=True)
        )

    return {
        "status": "MIGRATED",
        "seed_root": str(root),
        "spec": str(spec_path),
        "handoff": str(handoff_path),
        "handoff_sha256": bootstrap["handoff_sha256"],
        "bootstrap": str(bootstrap_path),
        "reused_roles": bootstrap["reused_roles"],
        "runtime_qualification": (
            str(qualification.resolve()) if qualification is not None else None
        ),
        "runtime_qualification_cache_candidates": bootstrap[
            "runtime_qualification_cache_candidates"
        ],
        "formal_task_consumption_allowed": False,
        "gpu_allocation_allowed": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "successor-handoffs" / "seed-webshop-v2",
    )
    parser.add_argument(
        "--runtime-qualification",
        type=Path,
        help=(
            "optional explicit predecessor PASS receipt; otherwise bounded "
            "auto-discovery is attempted"
        ),
    )
    parser.add_argument(
        "--no-runtime-qualification-discovery",
        action="store_true",
        help=(
            "capture immutable evidence only; current runtime qualification "
            "remains authoritative"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = migrate(
            args.seed_root,
            output_dir=args.output_dir,
            runtime_qualification=args.runtime_qualification,
            auto_discover_runtime_qualification=(
                not args.no_runtime_qualification_discovery
            ),
        )
    except (MigrationError, sh.HandoffError, OSError) as exc:
        print(f"[STALE] seed-webshop-v2 handoff migration: {exc}")
        return 2

    for role in result["reused_roles"]:
        print(f"[REUSE           ] {role}")
    for role in result["runtime_qualification_cache_candidates"]:
        print(f"[CACHE_CANDIDATE ] {role}")
    if result["runtime_qualification"] is None:
        print(
            "[RECHECK_AT_USE  ] runtime-qualification: predecessor receipt not "
            "unambiguously located; canonical runtime qualification remains authoritative"
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
