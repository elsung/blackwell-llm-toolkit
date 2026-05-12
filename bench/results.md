# Benchmark Results

Empirical numbers from **RTX PRO 6000 Blackwell Workstation Edition (96 GiB VRAM, sm_120)** + AMD EPYC 4565P 16-core / 32-thread + 96 GB DDR5 RAM, running Ubuntu 24.04 LTS, CUDA 13.0, kernel 6.17.0.

Method:
- **Decode** = median of 3 runs, 500-token completions, single stream
- **TTFT** = median of 3 streaming runs to first token
- **Prefill** = single 5,400-token prompt, time-to-first-completion-token
- **Concurrency N=k** = k concurrent requests of 500-token completions
- **Rapid bench** = 41-prompt suite (10 intelligence + 10 tool-use + 13 calibration + 3 orchestration + 5 creative). Lower is better only for "wall time"; everything else higher is better.

All numbers are **single-card** (no tensor parallelism).

---

## Production lineup (7 profiles)

| Profile | Engine | Quant | Decode tok/s | TTFT ms | Prefill tok/s @ 5.4k | N=4 agg | Intel | Tools | Cal | Orch | Ctx | Multimodal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **`nemotron3-nano-omni-30b-a3b-nvfp4-trtllm`** | TRT-LLM 1.3.0rc13 | NVFP4 (modelopt) | **269.81** | **24** | **62,407** | **705** | **10/10** | 8/10 | 12/13 | 3/3 | 8k | image+video+audio |
| `nemotron3-nano-30b-a3b-nvfp4-trtllm` | TRT-LLM 1.2.1 | NVFP4 (modelopt) | 249.15 | 29 | 24,094 | 682 | 9/10 | **10/10** | **13/13** | 3/3 | 8k | text only |
| `deepseek-v4-flash-iq2xxsxl-llamacpp-optane` | llama.cpp (cchuter fork) | IQ2_XXS-XL (GGUF) | 31 | 189 | 233 | — | 9/10 | **10/10** | **13/13** | 3/3 | 65k | text only |
| `minimax-m2.7-reap172b-q3ks-llamacpp-optane` | llama.cpp master | Q3_K_S (GGUF) | 117 | 23 | 2,547 | — | 7/10 | 8/10 | 10/13 | 3/3 | **196k** | text only |
| `minimax-m2.7-reap139b-q3km-llamacpp-optane` | llama.cpp master | Q3_K_M (GGUF) | 116 | — | — | — | (~7/10) | — | — | — | **196k** | text only |
| `minimax-m2.7-reap172b-w4a16-vllm-longctx` | vLLM 0.20.1 + LMCache | W4A16 (AutoRound) | 20-22 | — | 1k-5k | — | (~10/10) | — | — | — | **154k** | text only |
| `minimax-m2.7-reap172b-w4a16-vllm` | vLLM 0.20.1 | W4A16 (AutoRound) | 22-25 | — | 1,400-3,900 | — | **10/10** | (10/10) | 12/13 | — | 64k | text only |

Bold = best in column. Production rapid_bench results are aggregated from multiple runs; individual outlier runs noted in `docs/SUMMARY.md`.

### Model sources (Hugging Face)

The actual model checkpoints I downloaded and tested:

| Profile (above) | Hugging Face repo + variant | Quant | Ctx tested |
|---|---|---|---|
| `nemotron3-nano-omni-30b-a3b-nvfp4-trtllm` | [`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4) | NVFP4 (modelopt) | 8k |
| `nemotron3-nano-30b-a3b-nvfp4-trtllm` | [`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4) | NVFP4 (modelopt) | 8k |
| `deepseek-v4-flash-iq2xxsxl-llamacpp-optane` | [`teamblobfish/DeepSeek-V4-Flash-GGUF`](https://huggingface.co/teamblobfish/DeepSeek-V4-Flash-GGUF) — `IQ2_XXS-XL/` directory (78.6 GB across 2 shards) | IQ2_XXS-XL (GGUF) | 65k |
| `minimax-m2.7-reap172b-q3ks-llamacpp-optane` | [`exdysa/MiniMax-M2.7-REAP-172B-A10B-GGUF`](https://huggingface.co/exdysa/MiniMax-M2.7-REAP-172B-A10B-GGUF) — `Q3_K_S.gguf` (~70 GB single file) | Q3_K_S (GGUF) | 196k |
| `minimax-m2.7-reap139b-q3km-llamacpp-optane` | [`dervig/m51Lab-MiniMax-M2.7-REAP-139B-A10B-GGUF`](https://huggingface.co/dervig/m51Lab-MiniMax-M2.7-REAP-139B-A10B-GGUF) — `Q3_K_M.gguf` (~63 GB) | Q3_K_M (GGUF) | 196k |
| `minimax-m2.7-reap172b-w4a16-vllm-longctx` | [`MJPansa/MiniMax-M2.7-REAP-172B-A10B-AutoRound-W4A16`](https://huggingface.co/MJPansa/MiniMax-M2.7-REAP-172B-A10B-AutoRound-W4A16) + LMCache disk-tier offload to Optane SSD | W4A16 (AutoRound) | 154k |
| `minimax-m2.7-reap172b-w4a16-vllm` | same as above (without LMCache) | W4A16 (AutoRound) | 64k |

---

## Profile selection rubric

- **Vision / video / audio task?** → Omni V3 (the only multimodal option here)
- **Tool-heavy agentic?** → Text Nano (10/10 tools) or DS4-Flash (10/10 tools + 13/13 cal)
- **Long context 64k–154k with W4A16 max quality?** → mjpansa-longctx (LMCache offload to Optane)
- **Long context 196k with speed?** → exdysa REAP-172B (or REAP-139B for VRAM headroom)
- **General fast intelligent answer?** → Omni V3 (highest intel, fastest decode)

---

## Notable findings

### 1. MTP delivers exactly the PR-claimed 2× decode

Phase 2 of the project tested llama.cpp PR #22673 (Multi-Token Prediction) on Qwen3.6-27B Q6_K_XL via the `havenoammo/Qwen3.6-27B-MTP-UD-GGUF` model. Clean A/B with `--spec-type none` vs `--spec-type mtp --spec-draft-n-max 3 --spec-draft-ngl 999`:

| Metric | Baseline (spec=none) | MTP | Δ |
|---|---|---|---|
| Decode (tok/s, median 3) | 53.62 | 110.45 | **2.06×** |
| TTFT ms | 97 | 102 | identical |
| Prefill @ 5421 tok | 3246 | 2422 | 0.75× |
| Rapid intel | 6/10 | 6/10 | identical |
| Rapid wall time | 247s | 119s | 0.48× |

Quality preserved across all axes — MTP is a pure decode-time speculation technique, not a model surgery.

### 2. Ngram speculation gives ZERO win on MoE

Same llama.cpp binary (PR #22673), same MiniMax-M2.7-REAP-172B model, just `--spec-type ngram-cache --spec-draft-n-max 5`:

| Metric | Baseline | Ngram-cache | Δ |
|---|---|---|---|
| Decode (tok/s, median 3) | 117.02 | 115.96 | -0.9% (noise) |
| Concurrency N=2 agg | 176.7 | 170.5 | -3.5% (overhead) |

Why: MoE forward pass is already throughput-friendly (only 10B active of 172B). Speculation's verify-step still pays full forward cost; arithmetic doesn't pay back. **Ngram speculation is only worth it for dense models.**

### 3. LMCache + Optane gets MJPansa W4A16 from 64k → 154k context

Phase 3 of the project bumped vLLM's mjpansa profile to 160k max_model_len with KV-cache offload to Optane SSD via LMCache 0.4.4 (source-built for sm_120). Decode stays at 20-24 tok/s flat across the range:

| Effective ctx | Prefill (tok/s) | Decode (tok/s) |
|---|---|---|
| 64k | 1,216 | 24.21 |
| 100k | 2,341 | 22.22 |
| 116k | 5,605 | 20.16 |
| 154k | 2,733 | 20.28 |

**Counterintuitive: prefill speeds UP at longer prompts** because the `cpu_offload_gb=20` weight-streaming overhead amortizes over more tokens. Same pattern observed in Phase 1.5 mjpansa at 64k.

### 4. DS4-Flash IQ2_XXS-XL is highest agentic quality despite tiny quant

DeepSeek-V4-Flash 284B/13B-active MoE at IQ2_XXS-XL (~78 GB, fits 100% in VRAM) via cchuter's `feat/v4-port-cuda` llama.cpp fork. Despite ~2-bit equivalent quant:

- Intelligence: 9/10 (matches Omni V3, beats exdysa's 7/10)
- Tools: 10/10 (perfect)
- Calibration: 13/13 (perfect)
- Orchestration: 3/3 (perfect)

Decode is 31 tok/s — slow vs the Nemotron/MiniMax profiles, but quality is exceptional. The 233 tok/s prefill is the bottleneck for long-prompt usage.

### 5. TRT-LLM rc tags are weeks ahead of stable

The Nemotron Omni V3 multimodal handler (the headline of this whole repo) shipped in **v1.3.0rc13 (2026-04-29)** — 9 days after the v1.2.1 stable cut we started with. We initially attempted a manual port; research subagents discovered the upstream handler exists. **Always check rc tags before custom-handler engineering.**

### 6. NVFP4 stack is robust on sm_120 (mostly)

| Path | Status on sm_120 |
|---|---|
| TRT-LLM NemotronH + NVFP4 | ✅ works (rc13+) |
| TRT-LLM Qwen3-VL + NVFP4 | ✅ works |
| TRT-LLM Qwen 3.5/3.6 + NVFP4 | ❌ blocked upstream (GDN plugin pending, issue #11674) |
| TRT-LLM Gemma 4 + NVFP4 | ⚠️ AutoDeploy path only (PR #12710); classic convert_checkpoint won't accept |
| vLLM 0.20.1 + compressed-tensors NVFP4 | ✅ works (e.g., unsloth, igf-oeaw quants) |
| SGLang + NVFP4 multimodal | ❌ known multimodal compressed-tensors layer-strip bug |
| modelopt `hf_ptq.py --qformat nvfp4` | ✅ DIY-quantize any HF model, ~57 min for 27B-class with `--low_memory_mode` |

---

## Reproduction notes

All bench scripts hit OpenAI-compatible endpoints (default `http://127.0.0.1:8001` for llama.cpp, `8000` for vLLM, `9000` for TRT-LLM):

- **`rapid_bench.py`** — 41-prompt quality eval (intel/tools/cal/orch/creative). Uses `--url` flag to target any port.
- **`bench_harness.py`** — sustained decode + TTFT + prefill + concurrency. Has a `--prompt-tokens N` long-ctx mode.

---

## Contribute your results

Different Blackwell GPU? Different model? PR welcome — add a row to the table above with your hardware + the exact profile/launch config you used.
