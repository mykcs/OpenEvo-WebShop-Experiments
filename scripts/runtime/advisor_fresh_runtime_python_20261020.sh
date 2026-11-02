#!/bin/sh
# CPU qualification interpreter for this exact immutable image/source family.
# Formal model execution remains exclusively behind oe run-campaign.
set -eu
fresh_exec_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
fresh_runtime=/workspace/maintenance/Q17-VALIDATION-CURVE-202609231600/runtime-ac130148
fresh_core=/workspace/control/advisor-fresh-core-594789-20261020-source
fresh_image=sha256:d0f8796010899ae41a7af8ae3f44f47b96dea73b297b5af124049b3ae88e2850
exec docker run --rm --pull=never --read-only --network none \
  --user 1001:1001 --cap-drop ALL --security-opt no-new-privileges \
  --mount type=bind,src=/workspace,dst=/workspace,readonly \
  --tmpfs /tmp:rw,exec,nosuid,nodev,size=1g,mode=1777 \
  --env CUDA_VISIBLE_DEVICES= --env NVIDIA_VISIBLE_DEVICES=void \
  --env PYTHONDONTWRITEBYTECODE=1 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env HOME=/tmp --env TMPDIR=/tmp --env HF_HOME=/tmp/hf --env XDG_CACHE_HOME=/tmp/cache \
  --env "OPENEVO_RUNTIME_IMAGE_ID=$fresh_image" \
  --env "PYTHONPATH=$fresh_core/src:$fresh_runtime:$fresh_runtime/vendor/openevo-webshop/src:$fresh_exec_root:$fresh_exec_root/vendor/openevo-webshop/src" \
  --entrypoint python3 "$fresh_image" "$@"
