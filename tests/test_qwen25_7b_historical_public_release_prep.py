from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/governance/materialize_qwen25_7b_historical_public_release.py"
PREP = (
    ROOT
    / "docs/evidence/huggingface-stage2-checkpoint-decomposition-20260921"
    / "qwen25-7b-public-release-prep-202609212005"
)
RELEASE_RECEIPT = (
    ROOT
    / "docs/evidence/huggingface-stage2-checkpoint-decomposition-20260921"
    / "QWEN25_7B_PUBLIC_RELEASE_EXECUTION_202609212302.json"
)
HISTORICAL_PUBLIC_ALIAS = "example-account/openevo-ceiling1-qwen25-7b-historical-final-adapters"
CURRENT_PUBLIC_REPO = "openevo-recovery/openevo-ceiling1-qwen25-7b-historical-final-adapters"
PUBLIC_REVISION = "e18e0c7b02c247717f76c0b9fe2273c61b338fc4"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_materializer_has_no_provider_write_surface() -> None:
    source = SCRIPT.read_text()
    forbidden = (
        "create_repo(",
        "upload_file(",
        "upload_folder(",
        "move_repo(",
        "delete_repo(",
        "update_repo_settings(",
        "update_repo_visibility(",
    )
    for token in forbidden:
        assert token not in source


def test_release_manifest_is_owner_gated_and_exact() -> None:
    manifest = json.loads((PREP / "RELEASE_MANIFEST.json").read_text())
    assert manifest["status"] == "STAGED_METADATA_ONLY_OWNER_PUBLIC_RELEASE_GATE"
    assert manifest["publication_authorized"] is False
    assert manifest["provider_write_performed"] is False
    assert manifest["release_license"] == "UNRESOLVED_OWNER_SELECTION"
    assert manifest["source_provider"]["repo_id"].startswith("openevo-recovery/")
    assert manifest["source_provider"]["repo_type"] == "model"
    assert manifest["source_provider"]["revision"] == (
        "6849828d27d56f11423b0b417303564f4d36b2f6"
    )
    assert manifest["source_provider"]["visibility_expected"] == "private"
    assert manifest["upstream_base"]["revision"] == (
        "a09a35458c702b33eeacc393d103063234e8bc28"
    )
    assert manifest["upstream_base"]["required_file_identity"] == "9/9 exact SHA256 match"
    assert manifest["binary_bytes_total"] == 2_906_713_664
    assert manifest["package_bytes_with_configs"] == 2_906_715_273

    artifacts = {row["release_path"]: row for row in manifest["artifacts"]}
    assert artifacts["opsd/adapter_model.safetensors"]["sha256"] == (
        "f226725cad89f3f138423398168ce9e07d59e86a740a96e946ba1286f8d02190"
    )
    assert artifacts["stage2-final/adapter_model.safetensors"]["sha256"] == (
        "b0368f8c0dba79afa94e9e513c7b4f0023d1ebb95ab0d64adeca70734886199a"
    )


def test_metadata_only_materialization_is_binary_free(tmp_path: Path) -> None:
    out = tmp_path / "release"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(out)],
        cwd=ROOT,
        check=True,
    )
    assert (out / "README.md").is_file()
    assert (out / "PUBLIC_MANIFEST.json").is_file()
    assert (out / "SHA256SUMS.PLANNED").is_file()
    assert (out / "opsd/adapter_config.json").is_file()
    assert (out / "stage2-final/adapter_config.json").is_file()
    assert not list(out.rglob("*.safetensors"))

    receipt = json.loads((out / "LOCAL_MATERIALIZATION_RECEIPT.json").read_text())
    assert receipt["status"] == "PASS_METADATA_ONLY_NO_PROVIDER_WRITE"
    assert receipt["binary_download_performed"] is False
    assert receipt["provider_write_performed"] is False

    assert _sha256(out / "opsd/adapter_config.json") == (
        "8b38500148856b87ccaa1f9621bf0ef8ef340598a4bbba67187c272f7307fb05"
    )
    assert _sha256(out / "stage2-final/adapter_config.json") == (
        "c370668fdd320f1ce0f4f8d5d9f28f2d42aba54eb8b16f2937b8c0384da6de92"
    )


def test_public_readme_draft_preserves_claim_and_license_boundaries() -> None:
    text = (PREP / "PUBLIC_README_DRAFT.md").read_text()
    assert "DRAFT RELEASE METADATA" in text
    assert "Qwen2.5-7B historical long-run" in text
    assert "49.33 / 100" in text
    assert "58 / 128 = 45.31%" in text
    assert "paper-reported reference" in text
    assert "不能把数值差写成 paired causal comparison" in text
    assert "最终 release license / NOTICE 仍待 owner 明确选择" in text
    assert "69.9GB historical milestone checkpoint warehouse" in text


def test_planned_sha256sum_has_only_the_four_release_payload_files() -> None:
    lines = [
        line.strip()
        for line in (PREP / "SHA256SUMS.PLANNED").read_text().splitlines()
        if line.strip()
    ]
    assert len(lines) == 4
    assert lines[0].endswith("  opsd/adapter_config.json")
    assert lines[1].endswith("  opsd/adapter_model.safetensors")
    assert lines[2].endswith("  stage2-final/adapter_config.json")
    assert lines[3].endswith("  stage2-final/adapter_model.safetensors")


def test_metadata_only_output_is_public_safe(tmp_path: Path) -> None:
    out = tmp_path / "release-public-safe"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(out)],
        cwd=ROOT,
        check=True,
    )
    text = "\n".join(
        p.read_text(errors="ignore")
        for p in out.rglob("*")
        if p.is_file()
    )
    private_old = "example-account/" + "openevo-ceiling1-" + "v" + "2-archive"
    private_new = "openevo-recovery/" + "openevo-ceiling1-" + "v" + "2-archive"
    forbidden = [
        private_old,
        private_new,
        "kyo-myk/",
        "/data/home/",
        "/workspace/",
        "/workspace",
        "/workspace/model",
    ]
    assert all(item not in text for item in forbidden)

    public_manifest = json.loads((out / "PUBLIC_MANIFEST.json").read_text())
    assert public_manifest["private_source_provider_redacted"] is True
    assert "source_provider" not in public_manifest
    assert public_manifest["package_bytes_with_configs"] == 2_906_715_282
    opsd = json.loads((out / "opsd/adapter_config.json").read_text())
    assert opsd["base_model_name_or_path"] == "Qwen/Qwen2.5-7B-Instruct"


def test_current_registry_matches_verified_public_release() -> None:
    registry = json.loads(
        (ROOT / "docs/operations/publication/HUGGINGFACE_ASSET_REGISTRY.json").read_text()
    )
    prep = registry["historical_qwen25_7b_public_release_preparation_20260921"]
    assert prep["status"] == "PUBLIC_RELEASED_EXACT_VERIFIED"
    assert prep["package_bytes_with_configs"] == 2_906_715_282
    assert prep["proposed_public_repo_id"] == CURRENT_PUBLIC_REPO
    assert prep["public_repo_id"] == CURRENT_PUBLIC_REPO
    assert prep["legacy_public_alias"] == HISTORICAL_PUBLIC_ALIAS
    assert prep["immutable_revision"] == PUBLIC_REVISION
    assert prep["provider_repo_exists"] is True
    assert prep["visibility_change_authorized"] is True
    assert prep["provider_visibility"] == "public"
    assert prep["disclosure_state"] == "PUBLIC"
    assert prep["recovery_state"] == "RECOVERY_VERIFIED"
    assert prep["recoverability_class"] == "REMOTE_VERIFIED_EXACT"
    assert prep["release_license"] == "Apache-2.0"
    assert prep["anonymous_fresh_download_sha256"] == "PASS_BOTH_ADAPTERS"
    assert prep["two_stage_reconstruction_contract"] == "PASS"
    binding = registry["historical_qwen25_7b_recovery_binding_20260921"]
    assert binding["provider_visibility"] == "private"
    assert binding["public_release_status"] == prep["status"]

    assets = [row for row in registry["assets"] if row.get("repo_id") == CURRENT_PUBLIC_REPO]
    assert len(assets) == 1
    asset = assets[0]
    assert asset["verified_revision"] == PUBLIC_REVISION
    assert asset["provider_visibility"] == "public"
    assert asset["disclosure_state"] == "PUBLIC"
    assert asset["recovery_state"] == "RECOVERY_VERIFIED"
    assert asset["recoverability_class"] == "REMOTE_VERIFIED_EXACT"


def test_public_release_execution_receipt_closes_exact_readback_and_reconstruction() -> None:
    receipt = json.loads(RELEASE_RECEIPT.read_text())
    assert receipt["status"] == "PASS_PUBLIC_RELEASE_EXACT_AND_RECONSTRUCTIBLE"
    assert receipt["public_provider"]["repo_id"] == HISTORICAL_PUBLIC_ALIAS
    assert receipt["public_provider"]["immutable_revision"] == PUBLIC_REVISION
    assert receipt["public_provider"]["provider_visibility"] == "public"
    assert receipt["owner_gate"]["release_license"] == "Apache-2.0"
    assert receipt["owner_gate"]["deletion_authorized"] is False
    anon = receipt["anonymous_fresh_readback"]
    assert anon["status"] == "PASS"
    assert anon["opsd_full_stream_sha256"]["sha256"] == (
        "f226725cad89f3f138423398168ce9e07d59e86a740a96e946ba1286f8d02190"
    )
    assert anon["stage2_full_stream_sha256"]["sha256"] == (
        "b0368f8c0dba79afa94e9e513c7b4f0023d1ebb95ab0d64adeca70734886199a"
    )
    recon = receipt["reconstruction_contract"]
    assert recon["status"] == "PASS_TWO_STAGE_RECONSTRUCTION_CONTRACT"
    assert recon["upstream_base"]["historical_required_files_exact"] == "PASS_9_OF_9"
    assert recon["opsd_header"]["rank"] == 8
    assert recon["stage2_header"]["rank"] == 1144
    assert recon["opsd_header"]["shape_mismatch_count"] == 0
    assert recon["stage2_header"]["shape_mismatch_count"] == 0
    assert receipt["scientific_identity"]["passport_identity_action"] == (
        "NO_NEW_EXPERIMENT_PASSPORT_OR_CANONICAL_RUN_ID_INVENTED"
    )


def test_historical_7b_public_release_is_current_dag_backfill_without_invented_passport() -> None:
    router = json.loads((ROOT / "configs/experiment/current-passport-artifact-index.json").read_text())
    index = json.loads((ROOT / router["path"]).read_text())
    rows = [
        row for row in index["entries"]
        if row.get("operation_type") == "historical-7b.backfill.public-final-adapter-release"
    ]
    assert len(rows) == 1
    node = json.loads((ROOT / rows[0]["path"]).read_text())
    assert node["status"] == "PASS"
    assert node["provenance"]["experiment_id"] is None
    assert node["provenance"]["passport_sha256"] is None
    assert node["reuse"]["automatic_reuse_permitted"] is False
    views = node["provenance"]["external_views"]
    assert len(views) == 1
    assert views[0]["repository"] == HISTORICAL_PUBLIC_ALIAS
    assert views[0]["revision"] == PUBLIC_REVISION
