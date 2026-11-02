#!/usr/bin/env bash
set -euo pipefail

# No-Docker fallback. It is intentionally explicit and isolated; it never
# mutates the system Python or the validated OpenEvo image.
python_bin="${PYTHON_BIN:-python3.10}"
venv_dir="${VENV_DIR:-${HOME}/.cache/openevo/seed-vllm-0271}"
runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lock_path="${NO_DOCKER_LOCK:-${runtime_dir}/requirements-rc15-lock.txt}"
if [[ ! -x "$(command -v "${python_bin}" 2>/dev/null || true)" ]]; then
  echo "${python_bin} is required for the pinned Python 3.10 runtime" >&2
  exit 2
fi
for tool in java javac; do
  if [[ ! -x "$(command -v "${tool}" 2>/dev/null || true)" ]]; then
    echo "${tool} is required for the SEED/WebShop no-Docker route" >&2
    exit 2
  fi
done
if [[ -e "${venv_dir}" && "${REUSE_VENV:-0}" != "1" ]]; then
  echo "venv already exists: ${venv_dir}; set REUSE_VENV=1 only after auditing it" >&2
  exit 2
fi
if [[ ! -r "${lock_path}" ]]; then
  echo "no-Docker lock is missing: ${lock_path}" >&2
  exit 2
fi

venv_backend="python-venv"
if ! "${python_bin}" -c "import ensurepip" >/dev/null 2>&1; then
  if "${python_bin}" -c "import virtualenv" >/dev/null 2>&1; then
    venv_backend="user-virtualenv"
    "${python_bin}" -m virtualenv --no-download "${venv_dir}"
  else
    echo "${python_bin} lacks ensurepip and user virtualenv is unavailable" >&2
    echo "Install python3.10-venv through the host-owner repository or install virtualenv for this user" >&2
    exit 2
  fi
else
  "${python_bin}" -m venv "${venv_dir}"
fi
# shellcheck disable=SC1090
source "${venv_dir}/bin/activate"
python -m pip install --upgrade "pip==25.3"
python -m pip install "torch==2.13.0+cu130" --index-url https://download.pytorch.org/whl/cu130
python -m pip install --no-deps --upgrade "flashinfer-python==0.6.16.post4"
python -m pip install --no-deps --upgrade --index-url https://flashinfer.ai/whl "flashinfer-cubin==0.6.16.post4"
python -m pip install --no-deps --upgrade --index-url https://flashinfer.ai/whl/cu130 "flashinfer-jit-cache==0.6.16.post4+cu130"
python -m pip install --no-deps --requirement "${lock_path}"
python -m pip install --no-deps "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
python -m pip uninstall --yes fastapi-cli fastapi-cloud-cli pytest opencv-python-headless || true
python "${runtime_dir}/normalize_metadata.py"
python -m pip check
freeze_path="${venv_dir}/pip-freeze.txt"
python -m pip freeze > "${freeze_path}"
python - "${runtime_dir}/requirements.txt" "${freeze_path}" "${venv_dir}/venv-receipt.json" "${venv_backend}" "${lock_path}" <<'PY'
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

requirements = Path(sys.argv[1])
freeze = Path(sys.argv[2])
receipt = Path(sys.argv[3])
venv_backend = sys.argv[4]
lock = Path(sys.argv[5])

def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

payload = {
    "schema": "openevo.seed-venv-receipt.v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "python": subprocess.check_output([sys.executable, "--version"], text=True, stderr=subprocess.STDOUT).strip(),
    "python_executable": str(Path(sys.executable).resolve()),
    "requirements_sha256": digest(requirements),
    "lock_sha256": digest(lock),
    "pip_freeze_sha256": digest(freeze),
    "pip_freeze_path": str(freeze),
    "venv_backend": venv_backend,
    "formal_task_consumption_allowed": False,
}
receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
echo "Created ${venv_dir}; receipt=${venv_dir}/venv-receipt.json; formal task consumption remains disabled."
