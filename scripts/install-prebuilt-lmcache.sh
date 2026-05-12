#!/usr/bin/env bash
# Install the pre-built sm_120 LMCache wheel into the vllm-stable-blackwell venv.
# Use this on a fresh RTX PRO 6000 rig instead of rebuilding from source —
# avoids the ~1.5 min build + the nvcc/cuda-13.0 toolchain dependency.
#
# Requires the wheel at ${LLM_HOME:-/opt/llm}/wheels/lmcache-0.4.4-sm120-cp312-cp312-linux_x86_64.whl
# (committed to the repo).
#
# For other architectures (sm_89, sm_90, ...) run bin/build-lmcache-sm120.sh
# after editing TORCH_CUDA_ARCH_LIST inside.
set -euo pipefail

VENV="${LLM_HOME:-/opt/llm}/venvs/vllm-stable-blackwell"
WHEEL="${LLM_HOME:-/opt/llm}/wheels/lmcache-0.4.4-sm120-cp312-cp312-linux_x86_64.whl"
LIBCUDART_SHIM="${LLM_HOME:-/opt/llm}/venvs/sglang-stable/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"

if [[ ! -f "$WHEEL" ]]; then
  echo "ERROR: wheel not found at $WHEEL"
  echo "Either commit it (from build-lmcache-sm120.sh output) or rebuild from source."
  exit 1
fi

if [[ ! -d "$VENV" ]]; then
  echo "ERROR: venv not found at $VENV"
  echo "Set up the vllm-stable-blackwell venv first."
  exit 1
fi

echo "Installing $WHEEL into $VENV..."
${HOME}/.local/share/mise/installs/uv/0.11.12/uv-x86_64-unknown-linux-musl/uv \
  pip install \
  --python "$VENV/bin/python" \
  --force-reinstall \
  "$WHEEL"

# Verify
echo ""
echo "=== verifying installed c_ops.so ==="
CO_OPS_SO="$VENV/lib/python3.12/site-packages/lmcache/c_ops.cpython-312-x86_64-linux-gnu.so"
if cuobjdump --list-elf "$CO_OPS_SO" 2>&1 | grep -q "sm_120"; then
  echo "✓ sm_120 cubins present in $CO_OPS_SO"
else
  echo "✗ ERROR: no sm_120 cubins — wheel didn't install correctly"
  cuobjdump --list-elf "$CO_OPS_SO" | head -5
  exit 1
fi

LD_LIBRARY_PATH="$LIBCUDART_SHIM" \
  "$VENV/bin/python" -c 'import lmcache; print("lmcache", lmcache.__version__, "ok")' \
  || { echo "✗ import failed"; exit 1; }

echo ""
echo "✓ LMCache sm_120 installed and importable."
echo "  Now you can serve a vLLM model with kv_offload + LMCacheConnectorV1."
