#!/usr/bin/env bash
# Builds llama.cpp PR #22673 (MTP) into engines/llama.cpp-mtp/build-cuda/.
# This is a SEPARATE binary from engines/llama.cpp/build-cuda/llama-server
# so production REAP profiles keep using upstream master and the MTP
# profile uses this binary explicitly.
set -euo pipefail

ENGINE_DIR="${LLM_HOME:-/opt/llm}/engines/llama.cpp-mtp"

if [[ ! -d "$ENGINE_DIR/.git" ]]; then
  echo "Cloning llama.cpp-mtp..."
  git clone --depth=1 https://github.com/ggml-org/llama.cpp.git "$ENGINE_DIR"
  cd "$ENGINE_DIR"
  git fetch --depth=20 origin pull/22673/head:pr-22673-mtp
  git checkout pr-22673-mtp
else
  cd "$ENGINE_DIR"
  echo "Updating from PR #22673..."
  git fetch --depth=20 origin pull/22673/head:pr-22673-mtp
  git checkout pr-22673-mtp
fi

mkdir -p build-cuda
cd build-cuda
cmake .. \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DLLAMA_CURL=OFF
cmake --build . --config Release -j "$(nproc)"

echo ""
echo "Built: $ENGINE_DIR/build-cuda/bin/llama-server"
"$ENGINE_DIR/build-cuda/bin/llama-server" --help 2>&1 | grep -iE 'spec-type|mtp' | head -5 \
  && echo "✓ MTP flags present" \
  || echo "⚠ MTP flags NOT in --help — check the PR branch"
