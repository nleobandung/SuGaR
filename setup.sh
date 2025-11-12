#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=${1:-sugar}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] Conda is required but was not found in PATH."
  exit 1
fi

echo "[INFO] Creating or updating the '${ENV_NAME}' environment from environment.yml..."
conda env remove --name "${ENV_NAME}" --yes >/dev/null 2>&1 || true
conda env create --name "${ENV_NAME}" --file "${PROJECT_ROOT}/environment.yml"

echo "[INFO] Upgrading pip inside '${ENV_NAME}'..."
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip

echo "[INFO] Installing CUDA rasterization extensions..."
conda run -n "${ENV_NAME}" python -m pip install --no-deps --no-build-isolation --editable "${PROJECT_ROOT}/gaussian_splatting/submodules/diff-gaussian-rasterization"
conda run -n "${ENV_NAME}" python -m pip install --no-deps --no-build-isolation --editable "${PROJECT_ROOT}/gaussian_splatting/submodules/simple-knn"

echo "[INFO] (Optional) Installing nvdiffrast... (Ctrl+C to skip)"
if ! [ -d "${PROJECT_ROOT}/nvdiffrast" ]; then
  git clone https://github.com/NVlabs/nvdiffrast "${PROJECT_ROOT}/nvdiffrast"
fi
conda run -n "${ENV_NAME}" python -m pip install --no-build-isolation --editable "${PROJECT_ROOT}/nvdiffrast"

echo "[INFO] Setting up environment variables for CUDA and PyTorch..."
# Get the conda environment path
ENV_PATH=$(conda info --envs | grep "^${ENV_NAME}" | awk '{print $NF}' || conda run -n "${ENV_NAME}" python -c "import sys; print(sys.prefix)")
ACTIVATE_DIR="${ENV_PATH}/etc/conda/activate.d"
mkdir -p "${ACTIVATE_DIR}"

# Create activation script that sets up CUDA and PyTorch paths
cat > "${ACTIVATE_DIR}/sugar_env.sh" <<EOF
#!/bin/bash
# Automatically set CUDA and PyTorch library paths when conda environment is activated

export CUDA_HOME="\$CONDA_PREFIX"
export PATH="\$CUDA_HOME/bin:\$PATH"

# Add PyTorch libraries to LD_LIBRARY_PATH (required for CUDA extensions)
PYTORCH_LIB_PATH="\$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib"
if [ -d "\$PYTORCH_LIB_PATH" ]; then
    export LD_LIBRARY_PATH="\$PYTORCH_LIB_PATH:\$CUDA_HOME/lib:\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
else
    export LD_LIBRARY_PATH="\$CUDA_HOME/lib:\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
fi

# Add SuGaR CUDA extension modules to PYTHONPATH (required for simple_knn and diff_gaussian_rasterization)
SUGAR_ROOT="${PROJECT_ROOT}"
if [ -d "\${SUGAR_ROOT}/gaussian_splatting/submodules/simple-knn" ]; then
    export PYTHONPATH="\${SUGAR_ROOT}/gaussian_splatting/submodules/simple-knn:\${SUGAR_ROOT}/gaussian_splatting/submodules/diff-gaussian-rasterization:\$PYTHONPATH"
fi
EOF

chmod +x "${ACTIVATE_DIR}/sugar_env.sh"

echo "[INFO] Environment ready!"
echo "[INFO] The environment variables (CUDA_HOME, PATH, LD_LIBRARY_PATH) will be automatically set when you run:"
echo "    conda activate ${ENV_NAME}"
echo ""
echo "[INFO] You can re-run this script to recreate the environment if dependencies change."