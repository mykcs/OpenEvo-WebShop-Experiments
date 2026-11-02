#!/usr/bin/env bash
set -euo pipefail

# Build and identify the isolated SEED runtime. This script does not start a
# GPU container; launch authorization belongs to the experiment profile/gate.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
runtime_dir="${repo_root}/scripts/openevo_webshop/seed_runtime"
image="${IMAGE:-experiment-host/openevo-seed:qwen25-3b-webshop-20260817-rc15}"
expected_digest="${EXPECTED_IMAGE_DIGEST:-sha256:1f049d117b376e4af225ef1915bedbcec88d31bdf7376d189f7682ab9d118885}"

command -v docker >/dev/null || { echo "docker CLI is required" >&2; exit 2; }
docker build --pull=false --file "${runtime_dir}/Dockerfile" --tag "${image}" "${runtime_dir}"
observed_id="$(docker image inspect "${image}" --format '{{.Id}}')"
if [[ -n "${expected_digest}" && "${observed_id}" != "${expected_digest}" ]]; then
  echo "image identity mismatch: expected=${expected_digest} observed=${observed_id}" >&2
  exit 3
fi
docker image inspect "${image}" --format 'image={{.RepoTags}} id={{.Id}} size={{.Size}}'
docker image inspect "${image}" --format '{{json .RepoDigests}}'
echo "Build complete. Run environment_doctor.py and the single-GPU smoke before any formal task consumption."
