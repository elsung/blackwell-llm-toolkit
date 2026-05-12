#!/usr/bin/env bash
# Builds LMCache from source with sm_120 (Blackwell) CUDA kernels and
# installs into the vllm-stable-blackwell venv.
#
# Why: the pre-built lmcache-0.4.4-cu13 wheel ships c_ops.so with cubins
# for sm_70/75/80/86/89/90/100 and no PTX, so it crashes on Blackwell
# with cudaErrorNoKernelImage at lmc_ops.multi_layer_kv_transfer. This
# script forces TORCH_CUDA_ARCH_LIST=12.0 so nvcc emits
# -gencode arch=compute_120,code=sm_120 and the resulting wheel runs.
#
# Side-effect: saves the built wheel to engines/lmcache/dist/ for
# distribution to other RTX PRO 6000 (sm_120) machines without rebuilding.
#
# Build time on this hardware: ~1.5 min (16 cores, parallel nvcc).
set -euo pipefail

ENGINE_DIR="${LLM_HOME:-/opt/llm}/engines/lmcache"
VENV="${LLM_HOME:-/opt/llm}/venvs/vllm-stable-blackwell"
LMCACHE_TAG="v0.4.4"
LMCACHE_REPO="https://github.com/LMCache/LMCache.git"

# cu12 libcudart shim — same workaround the mjpansa profile uses.
# Required at IMPORT time, not just build time (LMCache issue #2843).
LIBCUDART_SHIM="${LLM_HOME:-/opt/llm}/venvs/sglang-stable/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"

if [[ ! -d "$ENGINE_DIR/.git" ]]; then
  echo "Cloning LMCache $LMCACHE_TAG..."
  git clone --depth=1 --branch "$LMCACHE_TAG" "$LMCACHE_REPO" "$ENGINE_DIR"
else
  cd "$ENGINE_DIR"
  echo "Updating to $LMCACHE_TAG..."
  git fetch --tags --depth=1 origin "+refs/tags/$LMCACHE_TAG:refs/tags/$LMCACHE_TAG"
  git checkout "$LMCACHE_TAG"
fi

cd "$ENGINE_DIR"

# Pre-built wheel artifact path. Keep these so a sister RTX PRO 6000
# can `uv pip install <whl>` without rebuilding.
mkdir -p dist

echo "Building LMCache wheel + installing into $VENV..."
PATH="$VENV/bin:/usr/local/cuda-13.0/bin:$PATH" \
TORCH_CUDA_ARCH_LIST="12.0" \
LD_LIBRARY_PATH="$LIBCUDART_SHIM" \
  ${HOME}/.local/share/mise/installs/uv/0.11.12/uv-x86_64-unknown-linux-musl/uv \
    pip install \
    --python "$VENV/bin/python" \
    --no-build-isolation \
    --force-reinstall \
    "$ENGINE_DIR"

# Also produce a portable wheel artifact for sharing across same-arch
# rigs. Drop it in $LLM_HOME/wheels/ so it's outside engines/ (which is
# gitignored) and can travel with the repo.
PATH="$VENV/bin:/usr/local/cuda-13.0/bin:$PATH" \
TORCH_CUDA_ARCH_LIST="12.0" \
LD_LIBRARY_PATH="$LIBCUDART_SHIM" \
  "$VENV/bin/python" setup.py bdist_wheel 2>&1 | tail -5

WHEELS_DIR="${LLM_HOME:-/opt/llm}/wheels"
mkdir -p "$WHEELS_DIR"
built_wheel="$(ls -t dist/lmcache-*.whl 2>/dev/null | head -1)"
if [[ -n "$built_wheel" ]]; then
  # Rename with explicit sm120 suffix to distinguish from PyPI cu13 wheels.
  target_name="$(basename "${built_wheel%-cp312-cp312-linux_x86_64.whl}-sm120-cp312-cp312-linux_x86_64.whl")"
  cp -f "$built_wheel" "$WHEELS_DIR/$target_name"
  echo "  Portable wheel: $WHEELS_DIR/$target_name"
fi

# Verify: the installed c_ops.so MUST have sm_120 cubins (and nothing else
# from our build).
echo ""
echo "=== verifying installed c_ops.so arch list ==="
CO_OPS_SO="$VENV/lib/python3.12/site-packages/lmcache/c_ops.cpython-312-x86_64-linux-gnu.so"
if cuobjdump --list-elf "$CO_OPS_SO" 2>&1 | grep -q "sm_120"; then
  echo "✓ sm_120 cubins present in $CO_OPS_SO"
  cuobjdump --list-elf "$CO_OPS_SO" | head -3
else
  echo "✗ ERROR: no sm_120 cubins found — build did not target Blackwell!"
  cuobjdump --list-elf "$CO_OPS_SO" | head -5
  exit 1
fi

# Verify: import works with the libcudart shim.
echo ""
echo "=== verifying import with libcudart shim ==="
LD_LIBRARY_PATH="$LIBCUDART_SHIM" \
  "$VENV/bin/python" -c 'import lmcache; print("lmcache", lmcache.__version__, "→ backend:", "c_ops" if "non_cuda" not in str(__import__("sys").modules.get("lmcache.c_ops", "non_cuda")) else "FALLBACK")' \
  || { echo "✗ ERROR: import failed"; exit 1; }
echo ""
echo "✓ LMCache sm_120 build complete and installed."
echo "  Source:      $ENGINE_DIR"
echo "  Wheel:       $(ls -1 $ENGINE_DIR/dist/lmcache-*.whl | head -1 || echo '(none built)')"
echo "  Installed:   $CO_OPS_SO"
