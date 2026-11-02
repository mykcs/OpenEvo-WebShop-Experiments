#!/usr/bin/env bash
set -euo pipefail

# Bounded, non-formal SEED route smoke. This validates the real manifest-bound
# trainer wiring with one short update. Set CHECKPOINT_MODE=persistent only for
# the explicit checkpoint-capacity probe; formal evaluation remains disabled.
# It must never be used as an H1.30 scientific result or formal denominator.

[[ "$(id -u)" == "0" ]] || { echo "control plane must run as root" >&2; exit 2; }
command -v docker >/dev/null || { echo "docker is required" >&2; exit 2; }
command -v jq >/dev/null || { echo "jq is required" >&2; exit 2; }

RUN_NAME="${OPENEVO_H130_SEED_SMOKE_RUN_NAME:-phase-h1.30-seed-manifest-route-smoke-v1-20260817}"
HOST_RUNS="${OPENEVO_H130_HOST_RUNS:-/workspace/runs}"
HOST_ROOT="${HOST_RUNS}/${RUN_NAME}"
HOST_OVERLAY="${OPENEVO_H130_SEED_OVERLAY:?set exact manifest-bound SEED overlay}"
HOST_MANIFEST="${OPENEVO_H130_SEED_MANIFEST:?set exact H1.30 training manifest}"
HOST_DATA="${OPENEVO_H130_SEED_DATA:?set exact manifest-bound train.parquet/val.parquet directory}"
HOST_CATALOG="${OPENEVO_H130_SEED_CATALOG:-/workspace/datasets/webshop-1000}"
HOST_INDEX="${OPENEVO_H130_SEED_INDEX:-/workspace/runtime-seed-index-9996427-v2/indexes_1k}"
HOST_MODEL="${OPENEVO_H130_SEED_MODEL:-/workspace/models/qwen2.5-3b-instruct}"
IMAGE="${OPENEVO_H130_SEED_IMAGE:-experiment-host/openevo-seed:qwen25-3b-webshop-20260817-rc15}"
IMAGE_DIGEST="${OPENEVO_H130_SEED_IMAGE_DIGEST:-sha256:1f049d117b376e4af225ef1915bedbcec88d31bdf7376d189f7682ab9d118885}"
CONTAINER_NAME="h130-seed-manifest-smoke"
CHECKPOINT_MODE="${OPENEVO_H130_SEED_CHECKPOINT_MODE:-ephemeral}"
RESUME_PATH="${OPENEVO_H130_SEED_RESUME_PATH:-}"
HOST_CHECKPOINT_ROOT="${OPENEVO_H130_SEED_CHECKPOINT_ROOT:-}"

GPU_UUIDS=(
  GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000
  GPU-00000000-0000-0000-0000-000000000000
)

die() { echo "H1.30 SEED route smoke failed: $*" >&2; exit 2; }

[[ "${CHECKPOINT_MODE}" == "ephemeral" || "${CHECKPOINT_MODE}" == "persistent" ]] || die "CHECKPOINT_MODE must be ephemeral or persistent"
if [[ -n "${RESUME_PATH}" ]]; then
  [[ -n "${HOST_CHECKPOINT_ROOT}" ]] || die "HOST_CHECKPOINT_ROOT is required for resume"
  [[ -d "${HOST_CHECKPOINT_ROOT}" ]] || die "checkpoint root is missing: ${HOST_CHECKPOINT_ROOT}"
fi
[[ "$(hostname)" == "experiment-host" ]] || die "unexpected host"
[[ -f "${HOST_MANIFEST}" && -f "${HOST_DATA}/train.parquet" && -f "${HOST_DATA}/val.parquet" ]] || die "manifest-bound data is incomplete"
[[ -d "${HOST_OVERLAY}" && -f "${HOST_OVERLAY}/h130-overlay-receipt.json" ]] || die "overlay receipt is missing"
[[ -d "${HOST_CATALOG}" && -d "${HOST_INDEX}" && -d "${HOST_MODEL}" ]] || die "catalog/index/model mount is missing"
[[ "$(docker image inspect "${IMAGE}" --format '{{.Id}}')" == "${IMAGE_DIGEST}" ]] || die "image digest mismatch"

jq -e '
  .schema == "openevo.webshop-h1.30-training-manifest.v1" and
  .status == "draft-not-authorized" and
  .formal_task_consumption_allowed == false and
  (.train_tasks | length == 16) and (.validation_tasks | length == 8)
' "${HOST_MANIFEST}" >/dev/null || die "training manifest is not fail-closed 16/8"
jq -e --arg sha "$(jq -r '.manifest_sha256' "${HOST_MANIFEST}")" '
  .training_manifest_sha256 == $sha and .status == "prepared-not-authorized"
' "${HOST_OVERLAY}/h130-overlay-receipt.json" >/dev/null || die "overlay is not bound to manifest"
jq -e --arg sha "$(jq -r '.manifest_sha256' "${HOST_MANIFEST}")" '
  .manifest_sha256 == $sha and .formal_task_consumption_allowed == false
' "${HOST_DATA}/h1.30-seed-parquet-receipt.json" >/dev/null || die "parquet is not bound to manifest"

gpu_gate() {
  local uuid row memory utilization apps
  for uuid in "${GPU_UUIDS[@]}"; do
    row="$(nvidia-smi -i "${uuid}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
    IFS=',' read -r memory utilization <<<"${row}"
    memory="${memory//[[:space:]]/}"
    utilization="${utilization//[[:space:]]/}"
    [[ "${memory}" =~ ^[0-9]+$ && "${memory}" -le 10 ]] || die "${uuid} memory=${memory} MiB"
    [[ "${utilization}" == "0" ]] || die "${uuid} utilization=${utilization}%"
    apps="$(nvidia-smi -i "${uuid}" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits)"
    [[ -z "${apps}" ]] || die "${uuid} has a compute process"
  done
}

mkdir -p "${HOST_ROOT}"
chown -R 1001:1001 "${HOST_ROOT}"
gpu_gate

TRAIN_GOALS="$(jq -c '[.train_tasks[].goal_idx]' "${HOST_MANIFEST}")"
VAL_GOALS="$(jq -c '[.validation_tasks[].goal_idx]' "${HOST_MANIFEST}")"
GPU_SPEC="$(IFS=,; echo "${GPU_UUIDS[*]}")"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

checkpoint_mount=()
if [[ -n "${HOST_CHECKPOINT_ROOT}" ]]; then
  checkpoint_mount=(-v "${HOST_CHECKPOINT_ROOT}:/workspace/checkpoint-in:ro")
fi

CONTAINER_SCRIPT="$(cat <<'CONTAINER_SCRIPT'
set -euo pipefail
train_goals="${H130_TRAIN_GOALS}"
val_goals="${H130_VAL_GOALS}"
cp -r /workspace/source-in/. /workspace/runtime/
mkdir -p /workspace/runtime/agent_system/environments/env_package/webshop/webshop/data
mkdir -p /workspace/runtime/agent_system/environments/env_package/webshop/webshop/search_engine/indexes_1k
cp -r /workspace/catalog-in/. /workspace/runtime/agent_system/environments/env_package/webshop/webshop/data/
cp -r /workspace/index-in/. /workspace/runtime/agent_system/environments/env_package/webshop/webshop/search_engine/indexes_1k/
export PYTHONPATH=/workspace/runtime
cd /workspace/runtime
checkpoint_args=(
  trainer.save_freq=-1
  trainer.test_freq=-1
  trainer.val_before_train=False
  trainer.total_epochs=1
  trainer.total_training_steps=1
  trainer.default_local_dir=/tmp/ckpt
  trainer.rollout_data_dir=/tmp/rollout
)
if [[ -n "${H130_RESUME_PATH}" ]]; then
  checkpoint_args=(
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.resume_mode=resume_path
    trainer.resume_from_path="${H130_RESUME_PATH}"
    trainer.val_before_train=True
    trainer.val_only=True
    trainer.total_epochs=1
    trainer.total_training_steps=1
    trainer.default_local_dir=/tmp/ckpt
    trainer.rollout_data_dir=/tmp/rollout
  )
elif [[ "${H130_CHECKPOINT_MODE}" == "persistent" ]]; then
  mkdir -p /workspace/out/checkpoints
  checkpoint_args=(
    trainer.save_freq=1
    trainer.test_freq=-1
    trainer.val_before_train=False
    trainer.total_epochs=1
    trainer.total_training_steps=1
    trainer.max_actor_ckpt_to_keep=1
    trainer.max_critic_ckpt_to_keep=1
    trainer.default_local_dir=/workspace/out/checkpoints
    trainer.rollout_data_dir=/workspace/out/rollout
  )
fi
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=seed \
  data.train_files=/workspace/data-in/train.parquet \
  data.val_files=/workspace/data-in/val.parquet \
  data.train_batch_size=8 data.val_batch_size=8 \
  data.max_prompt_length=512 data.max_response_length=32 \
  data.filter_overlong_prompts=True data.truncation=left data.return_raw_chat=True \
  actor_rollout_ref.model.path=/workspace/model \
  +actor_rollout_ref.model.override_config._attn_implementation=eager \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.opd_loss_coef=0.01 \
  actor_rollout_ref.actor.opd_gate_beta=5.0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
  actor_rollout_ref.rollout.max_model_len=1024 \
  actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  algorithm.use_kl_in_reward=False algorithm.gamma=0.95 \
  algorithm.seed.enable_analysis=False algorithm.seed.mode=mean_norm \
  env.history_length=2 env.env_name=Webshop env.seed=2026 env.max_steps=2 \
  env.rollout.n=1 env.resources_per_worker.num_cpus=0.1 \
  env.webshop.train_goal_indices="${train_goals}" \
  env.webshop.val_goal_indices="${val_goals}" \
  ray_init.num_cpus=32 \
  trainer.critic_warmup=0 trainer.logger="[console]" \
  trainer.project_name=h130-route-smoke trainer.experiment_name=manifest-bound-v1 \
  trainer.n_gpus_per_node=8 trainer.nnodes=1 \
  "${checkpoint_args[@]}"
CONTAINER_SCRIPT
)"

set +e
docker run --rm --name "${CONTAINER_NAME}" \
  --read-only --tmpfs /workspace/runtime:rw,nosuid,size=1g \
  --tmpfs /tmp:rw,exec,nosuid,size=4g --mount type=tmpfs,destination=/root,tmpfs-mode=0755 \
  --shm-size=64g --pids-limit 16384 --network host --gpus "\"device=${GPU_SPEC}\"" \
  --user 1001:1001 \
  -v "${HOST_OVERLAY}":/workspace/source-in:ro \
  -v "${HOST_DATA}":/workspace/data-in:ro \
  -v "${HOST_CATALOG}":/workspace/catalog-in:ro \
  -v "${HOST_INDEX}":/workspace/index-in:ro \
  -v "${HOST_MODEL}":/workspace/model:ro \
  -v "${HOST_MANIFEST}":/workspace/training-manifest.json:ro \
  -v "${HOST_ROOT}":/workspace/out:rw \
  -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/.cache -e HF_HOME=/tmp/hf \
  -e TRANSFORMERS_CACHE=/tmp/hf -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 \
  -e TOKENIZERS_PARALLELISM=false -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 -e RAYON_NUM_THREADS=1 \
  -e VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  -e H130_CHECKPOINT_MODE="${CHECKPOINT_MODE}" \
  -e H130_RESUME_PATH="${RESUME_PATH}" \
  -e H130_TRAIN_GOALS="${TRAIN_GOALS}" -e H130_VAL_GOALS="${VAL_GOALS}" \
  "${checkpoint_mount[@]}" \
  --entrypoint bash "${IMAGE}" -lc "${CONTAINER_SCRIPT}" \
  2>&1 | tee "${HOST_ROOT}/container.log"
CONTAINER_RC=${PIPESTATUS[0]}
set -e
ENDED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
LOG_SHA="$(sha256sum "${HOST_ROOT}/container.log" | awk '{print $1}')"
jq -n \
  --arg status "$(if [[ "${CONTAINER_RC}" == 0 ]]; then echo pass; else echo fail; fi)" \
  --arg started_at "${STARTED_AT}" --arg ended_at "${ENDED_AT}" \
  --arg manifest_sha256 "$(jq -r '.manifest_sha256' "${HOST_MANIFEST}")" \
  --arg log_sha256 "${LOG_SHA}" --argjson exit_code "${CONTAINER_RC}" \
  --arg checkpoint_persistence "${CHECKPOINT_MODE}" \
  --arg checkpoint_resume_path "${RESUME_PATH}" \
  '{schema:"openevo.webshop-h1.30-seed-manifest-route-smoke.v1",status:$status,formal_task_consumption_allowed:false,scientific_result:false,gpu_workload_launched:true,checkpoint_persistence:$checkpoint_persistence,checkpoint_resume_path:$checkpoint_resume_path,manifest_sha256:$manifest_sha256,gpu_count:8,started_at:$started_at,ended_at:$ended_at,exit_code:$exit_code,container_log_sha256:$log_sha256}' \
  > "${HOST_ROOT}/route-smoke-receipt.json"
chown 1001:1001 "${HOST_ROOT}/container.log" "${HOST_ROOT}/route-smoke-receipt.json"
gpu_gate
[[ "${CONTAINER_RC}" == 0 ]] || exit "${CONTAINER_RC}"
echo "H1.30 manifest-bound SEED route smoke passed; formal_task_consumption_allowed=false"
