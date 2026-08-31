#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${NAVRL_ENV_NAME:-NavRL}"
PYTHON_VERSION="${NAVRL_PYTHON_VERSION:-3.8}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

command -v conda >/dev/null || { echo "conda is required" >&2; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi
conda activate "${ENV_NAME}"
python -m pip install --upgrade pip
python -m pip install -r "${SCRIPT_DIR}/requirements-deployment.txt"
# The vendored 0.4 packages use the legacy setuptools develop path.  Disable
# build isolation so the Python 3.8/torch toolchain selected above is reused.
python -m pip install --no-cache-dir "setuptools<66" "importlib-metadata>=4.8,<7" zipp

TENSOR_DIR="${SCRIPT_DIR}/third_party/tensordict"
RL_DIR="${SCRIPT_DIR}/third_party/rl"
if [[ -f "${TENSOR_DIR}/setup.py" ]]; then python -m pip install --no-build-isolation -e "${TENSOR_DIR}"; fi
if [[ -f "${RL_DIR}/setup.py" ]]; then python -m pip install --no-build-isolation -e "${RL_DIR}"; fi

python - <<'PY'
import torch
import tensordict
import torchrl
print("NavRL deployment environment OK")
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
PY
