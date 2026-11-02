#!/usr/bin/env bash
set -euo pipefail

open_evo_root="${OPEN_EVO_ROOT:?set OPEN_EVO_ROOT to the external pinned OpenEvo checkout}"
expected_sha="${OPEN_EVO_EXPECTED_SHA:-796e6248a61112a61c880885844d3b0420c0b6ae}"
patch_file="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/patches" && pwd)/openevo-wandb-trainer-logging.patch"
check_only="${OPENEVO_WANDB_PATCH_CHECK_ONLY:-0}"
trainer_path="src/openevo/evolution/parametric/sd_lora_trainer.py"
service_path="src/openevo/evolution/parametric/trainer_service.py"
expected_trainer_blob="30d0c3df140b7d1e36a7b550ad7cf4bb77633b08"
expected_service_blob="d60b105a9d9e833d6fb6a14d6afa587e024936b4"

actual_sha="$(git -C "$open_evo_root" rev-parse HEAD)"
if [[ "$actual_sha" != "$expected_sha" ]]; then
  printf 'OpenEvo revision mismatch: expected %s, got %s\n' "$expected_sha" "$actual_sha" >&2
  exit 1
fi

# If the patch is already applied, the working-tree blobs intentionally differ
# from the pristine pin, so check that case before pristine blob verification.
if git -C "$open_evo_root" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
  echo "W&B trainer patch already applied at $actual_sha"
  exit 0
fi

actual_trainer_blob="$(git -C "$open_evo_root" hash-object "$trainer_path")"
actual_service_blob="$(git -C "$open_evo_root" hash-object "$service_path")"
if [[ "$actual_trainer_blob" != "$expected_trainer_blob" ]]; then
  printf 'OpenEvo trainer blob mismatch: expected %s, got %s\n' "$expected_trainer_blob" "$actual_trainer_blob" >&2
  exit 1
fi
if [[ "$actual_service_blob" != "$expected_service_blob" ]]; then
  printf 'OpenEvo trainer-service blob mismatch: expected %s, got %s\n' "$expected_service_blob" "$actual_service_blob" >&2
  exit 1
fi

if ! git -C "$open_evo_root" apply --check --whitespace=error-all "$patch_file"; then
  echo "W&B trainer patch does not apply cleanly at $actual_sha" >&2
  exit 1
fi

echo "W&B trainer patch cleanly applies at $actual_sha"
if [[ "$check_only" == "1" ]]; then
  echo "Check-only mode: no OpenEvo files modified"
  exit 0
fi

git -C "$open_evo_root" apply --whitespace=error-all "$patch_file"
git -C "$open_evo_root" diff --check
echo "Applied W&B trainer patch at $actual_sha"
