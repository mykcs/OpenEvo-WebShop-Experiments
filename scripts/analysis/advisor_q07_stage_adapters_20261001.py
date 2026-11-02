#!/usr/bin/env python3
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path("/workspace/maintenance/ADVISOR-Q07-TASK-VECTOR-CAUSAL-20261001")
WT = Path("/workspace/tmp/q07-sync-main-20261001")
MANIFEST = WT / "configs/experiment/manifests/advisor-q07-task-vector-causal-20261001.json"
STAGING = ROOT / "adapter-staging"
E4_CONFIG_SHA = "a8c00d1bf91be772497b54f1de9db8956b014d8f8211794c7e8d0efe366eca84"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

m = json.loads(MANIFEST.read_text())
STAGING.mkdir(parents=True, exist_ok=True)
records = {}

for cid, checkpoint in m["checkpoints"].items():
    if cid == "intact_w159":
        continue
    src = Path(checkpoint["adapter_path"])
    dst = STAGING / cid
    source_model = src / "adapter_model.safetensors"
    source_config = src / "adapter_config.json"
    if not source_model.is_file() or not source_config.is_file():
        raise RuntimeError(f"missing source adapter payload: {cid}")
    source_model_sha = sha(source_model)
    source_config_sha = sha(source_config)
    if source_model_sha != checkpoint["adapter_sha256"]:
        raise RuntimeError(f"source model SHA drift: {cid}")
    if checkpoint.get("output_rank") == 384 and source_config_sha != E4_CONFIG_SHA:
        raise RuntimeError(f"rank384 config SHA drift: {cid}: {source_config_sha}")

    dst.mkdir(parents=True, exist_ok=True)
    staged_model = dst / "adapter_model.safetensors"
    staged_config = dst / "adapter_config.json"
    copy_mode = "reuse"
    if staged_model.exists() or staged_config.exists():
        if not (
            staged_model.is_file()
            and staged_config.is_file()
            and sha(staged_model) == source_model_sha
            and sha(staged_config) == source_config_sha
        ):
            raise RuntimeError(f"existing staging mismatch: {cid}")
    else:
        copy_mode = "full_copy"
        shutil.copyfile(source_model, staged_model)
        shutil.copyfile(source_config, staged_config)

    os.chown(dst, 1001, 1001)
    os.chmod(dst, 0o755)
    os.chown(staged_model, 1001, 1001)
    os.chmod(staged_model, 0o600)
    os.chown(staged_config, 1001, 1001)
    os.chmod(staged_config, 0o644)

    staged_model_sha = sha(staged_model)
    staged_config_sha = sha(staged_config)
    if staged_model_sha != source_model_sha or staged_config_sha != source_config_sha:
        raise RuntimeError(f"staging hash mismatch: {cid}")

    records[cid] = {
        "source_path": str(src),
        "staging_path": str(dst),
        "copy_mode": copy_mode,
        "model_sha256": staged_model_sha,
        "config_sha256": staged_config_sha,
        "model_size_bytes": staged_model.stat().st_size,
    }

receipt = {
    "schema": "openevo.advisor.q07.adapter-staging.v1",
    "status": "PASS",
    "uid": 1001,
    "gid": 1001,
    "historical_source_mutated": False,
    "state_count": len(records),
    "records": records,
}
(ROOT / "ADAPTER_STAGING_RECEIPT.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n"
)
print(
    json.dumps(
        {
            "status": "PASS",
            "state_count": len(records),
            "logical_bytes": sum(x["model_size_bytes"] for x in records.values()),
        },
        sort_keys=True,
    )
)
