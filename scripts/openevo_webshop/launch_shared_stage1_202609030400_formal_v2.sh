#!/usr/bin/env bash
set -euo pipefail
WT=${SHARED_STAGE1_WORKTREE:-/workspace/control/openevo-experiment-shared-min-deliberation-20260903}
SHA=${FORMAL_EXECUTION_SHA:?set FORMAL_EXECUTION_SHA to the exact committed execution source}
SEED=/workspace/assets/seed-webshop-v1
MODEL3=/workspace/models/qwen2.5-3b-instruct
MODEL17=/workspace/models/qwen3-1.7b
ROOT3=/workspace/runs/qwen25-3b-stage1-202609030400
ROOT17=/workspace/runs/qwen3-1p7b-stage1-202609030400
IMAGE=experiment-host/openevo-scientific:h144-runtime-v1
WRAPPER=scripts/openevo_webshop/shared_stage1_202609030400_formal_exec_v2.py
[[ "$(git -c safe.directory=$WT -C $WT rev-parse HEAD)" == "$SHA" ]]
[[ -z "$(git -c safe.directory=$WT -C $WT status --porcelain)" ]]
[[ "$(sha256sum $MODEL3/model-00001-of-00002.safetensors | awk "{print \$1}")" == 67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2 ]]
[[ "$(sha256sum $MODEL3/model-00002-of-00002.safetensors | awk "{print \$1}")" == a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d ]]
[[ "$(sha256sum $MODEL17/model-00001-of-00002.safetensors | awk "{print \$1}")" == 169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5 ]]
[[ "$(sha256sum $MODEL17/model-00002-of-00002.safetensors | awk "{print \$1}")" == 912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9 ]]
echo "FORMAL_EXECUTION_SOURCE_OK $SHA"
echo "3B_MODEL=$MODEL3"
echo "1P7B_MODEL=$MODEL17"
echo "WRAPPER=$WRAPPER"
echo "This launcher is execution plumbing only; it does not auto-start without a separate operator launch step."
