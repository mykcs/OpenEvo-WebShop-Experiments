from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/governance/restore_stage2_historical_3b_weights.py"


def test_optimized_mode_rejects_tampered_archive_member_and_path(tmp_path: Path) -> None:
    archive = tmp_path / "weights.tar.gz"
    rows = []
    with tarfile.open(archive, "w:gz") as tar:
        for number in range(62):
            legacy_member_prefix = "formal-vnext-v1"  # legacy-revision-compat: fixture mirrors the provider archive layout
            name = f"{legacy_member_prefix}/round-{number:03d}/weight.safetensors"
            payload = f"weight {number}".encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
            rows.append(
                {
                    "archive_member_path": name,
                    "current_relative_path": "stage2/3b/" + name,
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    good = {
        "source_archive_size_bytes": archive.stat().st_size,
        "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "reuse_weight_paths": rows,
    }

    def check(binding: dict) -> subprocess.CompletedProcess[str]:
        receipt = tmp_path / "binding.json"
        receipt.write_text(json.dumps(binding))
        return subprocess.run(
            [sys.executable, "-O", str(SCRIPT), "--archive", str(archive), "--binding", str(receipt)],
            capture_output=True,
            text=True,
            check=False,
        )

    assert check(good).returncode == 0
    wrong_archive = json.loads(json.dumps(good))
    wrong_archive["source_archive_sha256"] = "0" * 64
    result = check(wrong_archive)
    assert result.returncode != 0 and "archive SHA-256 mismatch" in result.stderr
    wrong_member = json.loads(json.dumps(good))
    wrong_member["reuse_weight_paths"][0]["sha256"] = "0" * 64
    result = check(wrong_member)
    assert result.returncode != 0 and "member digest mismatch" in result.stderr
    wrong_path = json.loads(json.dumps(good))
    wrong_path["reuse_weight_paths"][0]["current_relative_path"] = "../outside.safetensors"
    result = check(wrong_path)
    assert result.returncode != 0 and "unsafe target path" in result.stderr
