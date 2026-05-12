# Blackwell LLM Toolkit

Reproducible recipes, configs, and **first-hand benchmarks** for running modern open-weight LLMs on **NVIDIA Blackwell** GPUs (RTX PRO 6000, RTX 50-series, B200/GB200). Focus on **NVFP4** quantization, multimodal models (image/video/audio), and the **TensorRT-LLM** + **vLLM** + **llama.cpp** + **LMCache** stack.

Everything in this repo is empirically validated on an **RTX PRO 6000 96 GB (sm_120)** as of 2026-05-11.

---

## 🎯 Headline result

**`nemotron3-nano-omni-30b-a3b-nvfp4-trtllm`** — [`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4) (30B / 3B-active MoE, multimodal) on TensorRT-LLM v1.3.0rc13, NVFP4 modelopt format, sm_120, tested at 8k context:

| Metric | Value |
|---|---|
| **Decode** | **269.81 tok/s** (single-stream, 500-tok generation) |
| **TTFT** | **24 ms** (median of 3 streaming runs) |
| **Prefill** | **62,407 tok/s** @ 5,426-token prompt |
| Concurrency N=4 | 705 tok/s aggregate |
| Intel score | 10/10 (rapid_bench) |
| Calibration | 12/13 |
| Orchestration | 3/3 |
| Tools | 8/10 (reasoning leak — see notes) |
| VRAM @ ctx=8k | 88 GB |
| **Modalities** | **image + video + audio + text** in one pass |

That's the **fastest and highest-quality model** measured on this box across all 7 production profiles benched in this repo — see [`bench/results.md`](bench/results.md) for the full table with Hugging Face links + quant + context-tested for each model.

**Bench methodology**: `rapid_bench.py` (41-prompt quality eval — 10 intelligence + 10 tool-use + 13 calibration + 3 orchestration + 5 creative) and `bench_harness.py` (sustained decode + TTFT + prefill + concurrency, with a `--prompt-tokens N` long-ctx mode). Both ship with the companion `runllm` launcher (in progress, separate release).

---

## 🚦 Quick start by GPU class

| Your GPU | What works directly | What needs adjustment |
|---|---|---|
| **RTX PRO 6000 / RTX 50-series (sm_120)** | Everything in `wheels/` + every `configs/trtllm/*.yaml` + the build scripts | Nothing — drop-in |
| **B200 / GB200 (sm_100)** | Recipes + research docs apply | Rebuild `wheels/lmcache-*.whl` via `scripts/build-lmcache-sm120.sh` after changing `CMAKE_CUDA_ARCHITECTURES` to `100`. TRT-LLM YAMLs may want `moe_config.backend: TRTLLM` instead of `CUTLASS`. |
| **Ada (RTX 4090, sm_89)** | Research docs + general recipes | NVFP4 has no hardware path — use FP8/AWQ/GPTQ. Rebuild wheels for sm_89. |
| **Hopper (H100/H200, sm_90)** | Research + recipes | FP8 native but NVFP4 emulated/slow. Use FP8 variants. |
| **Ampere (A100, sm_80)** | Research only | Stay on FP16/INT4-AWQ paths; NVFP4 unsupported. |

**NVFP4 hardware acceleration is Blackwell-only.** Anyone on Ada/Hopper/Ampere should look at the architecture deltas in [`docs/`](docs/) for ideas but use FP8 variants for the actual serving path.

---

## 📦 What's in this repo

```
blackwell-llm-toolkit/
├── README.md                          ← you are here
├── bench/results.md                   ← full 7-profile benchmark table
├── docs/
│   ├── SUMMARY.md                     ← strategic synthesis (read second)
│   └── architecture/                  ← architecture deltas vs prior gen
│       ├── omni-v3.md                  · NemotronH + RADIO + Parakeet
│       ├── qwen36.md                   · Qwen 3.5/3.6 vs Qwen3-VL (hybrid GDN)
│       └── gemma4-and-modelopt.md      · Gemma 4 vs Gemma 3, + modelopt NVFP4 recipe
├── configs/
│   ├── trtllm/
│   │   ├── nemotron-h-base.yaml       ← NemotronH (Mamba hybrid) launch knobs
│   │   └── nemotron-omni-v3-sm120.yaml ← Omni V3 sm_120-specific knobs
│   └── profiles/                       ← 6 production-tier profile TOMLs
├── scripts/
│   ├── build-lmcache-sm120.sh         ← source-build LMCache w/ sm_120 cubins
│   ├── install-prebuilt-lmcache.sh    ← install committed wheel (no nvcc needed)
│   ├── build-llama-cpp-mtp.sh         ← llama.cpp PR #22673 build
│   └── build-llama-cpp-ds4.sh         ← cchuter feat/v4-port-cuda (DeepSeek-V4-Flash)
└── wheels/
    ├── lmcache-0.4.4-sm120-cp312-cp312-linux_x86_64.whl  ← 11 MB, ready-to-install
    └── README.md                       ← compat matrix + verify command
```

---

## 🔧 Three things you can use today

### 1. LMCache for Blackwell sm_120 (the missing pre-built wheel)

The PyPI `lmcache==0.4.4-cu13` wheel ships cubins for sm_70/75/80/86/89/90/100 — **no sm_120, no PTX**. Crashes on Blackwell with `cudaErrorNoKernelImageForDevice` during `lmc_ops.multi_layer_kv_transfer`.

This repo has a working sm_120 wheel and a script to rebuild from source:

```bash
# Use the prebuilt wheel (sm_120 only; cp312 only)
LLM_HOME=/opt/llm bash scripts/install-prebuilt-lmcache.sh

# Or rebuild from source (1.5 min on 96 GB Blackwell)
LLM_HOME=/opt/llm bash scripts/build-lmcache-sm120.sh
```

After install, verify:
```bash
cuobjdump --list-elf $VLLM_VENV/lib/python3.12/site-packages/lmcache/c_ops.cpython-312-x86_64-linux-gnu.so | head -3
# Expect: sm_120.cubin entries
```

### 2. TRT-LLM v1.3.0rc13 launch configs for sm_120

The Nemotron Omni V3 (and any other Mamba-hybrid + DeepSeek-grouped-MoE model) needs three non-default launch knobs on sm_120 that aren't documented anywhere:

```yaml
# configs/trtllm/nemotron-omni-v3-sm120.yaml
kv_cache_config:
  enable_block_reuse: false       # Mamba hybrid requirement
  mamba_ssm_cache_dtype: float32  # mandatory for long sequences
moe_config:
  backend: CUTLASS                 # TRTLLM backend is B200-specific
```

Pass via `trtllm-serve serve ... --extra_llm_api_options <yaml>`.

### 3. Strategic architecture research

Before patching TRT-LLM's missing handlers manually, **check the latest rc tags first**. v1.2.1 stable doesn't have Omni V3; v1.3.0rc13 (released 9 days later) has it natively. That insight alone saved this project weeks of dead-end engineering.

The three reports in [`docs/architecture/`](docs/architecture/) cover:
- **Omni V3** — full architecture (NemotronH backbone, DeepSeek-V3-style grouped MoE, C-RADIOv4-H vision, Parakeet audio)
- **Qwen 3.5/3.6** — why these are NOT renamed Qwen3-VL (hybrid GatedDeltaNet, vLLM-native only, TRT-LLM blocked upstream)
- **Gemma 4 + modelopt NVFP4** — Gemma 4 deltas (dual head dims, PLE, MoE) + the `hf_ptq.py` DIY NVFP4 recipe (~57 min for 27B class)

---

## 🧰 Companion projects (separate repos, planned)

This repo is a focused dump of recipes and bench data. For full management tooling around launching/saving profiles, swapping engines, and DIY-quantizing models, watch for:

- **`runllm`** — bash launcher with TOML profile schema (one command per model + config)
- **`makellm`** — quantization + REAP-pruning toolchain wrapper

Both planned for separate releases. The profile TOMLs in `configs/profiles/` here are the schema `runllm` consumes.

---

## 🤝 Contributing

Bench results on other Blackwell GPUs (RTX 5090, B200, etc.) are very welcome. Open a PR adding a row to `bench/results.md` with your hardware, configs, and method. Same for other model families that work / don't work on sm_120.

If you hit a different sm_120 issue, open an issue with the error trace — the gotchas list at the bottom of `docs/SUMMARY.md` is the project's accumulated learnings.

---

## License

[Apache 2.0](LICENSE). The model weights referenced here have their own licenses (mostly permissive — Apache 2.0, MIT, Qwen license, Gemma terms); always check the original model card on Hugging Face before redistribution.

---

## Author

Built by [@elsung](https://github.com/elsung) (`elsung@gmail.com`) on a personal RTX PRO 6000 Blackwell workstation. If this saved you a day of debugging, ⭐ the repo or say hi on Reddit ([u/elsung](https://reddit.com/user/elsung)) or Discord.
