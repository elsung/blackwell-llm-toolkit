#!/usr/bin/env bash
# Builds cchuter's llama.cpp feat/v4-port-cuda branch into
# engines/llama.cpp-ds4/build-cuda/. This is the fork that supports
# DeepSeek-V4-Flash's V4 architecture + Lightning indexer + sinkhorn
# expert routing.
#
# Kept separate from engines/llama.cpp (production master) and
# engines/llama.cpp-mtp (PR #22673) so each fork stays pinned to its
# own ref and binary path.
set -euo pipefail

ENGINE_DIR="${LLM_HOME:-/opt/llm}/engines/llama.cpp-ds4"
BRANCH="feat/v4-port-cuda"
REPO="https://github.com/cchuter/llama.cpp.git"

if [[ ! -d "$ENGINE_DIR/.git" ]]; then
  echo "Cloning llama.cpp-ds4 ($BRANCH from cchuter)..."
  git clone --depth=1 --branch "$BRANCH" "$REPO" "$ENGINE_DIR"
else
  cd "$ENGINE_DIR"
  echo "Updating $BRANCH..."
  git fetch origin "$BRANCH"
  git checkout "$BRANCH"
  git reset --hard "origin/$BRANCH"
fi

cd "$ENGINE_DIR"
mkdir -p build-cuda
cd build-cuda

# Architectures: 89 (Ada / RTX 4090) + 120 (Blackwell / RTX PRO 6000) per the
# teamblobfish DS4 model card's recommended build command.
cmake .. \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="89;120" \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DLLAMA_CURL=OFF
cmake --build . --config Release -j "$(nproc)"

echo ""
echo "Built: $ENGINE_DIR/build-cuda/bin/llama-server"
"$ENGINE_DIR/build-cuda/bin/llama-server" --version 2>&1 | head -3 \
  && echo "✓ DS4 build OK" \
  || echo "⚠ DS4 build may have failed — check above"
