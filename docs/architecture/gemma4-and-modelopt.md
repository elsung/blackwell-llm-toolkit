# Gemma 4 Architecture + ModelOpt NVFP4 Research

_Date: 2026-05-11 | Target hardware: RTX PRO 6000 sm_120, 96 GB VRAM_

---

## 1. Gemma 4 Overview + Delta from Gemma 3

### Release

Google released the Gemma 4 family on **2026-04-02** under Apache 2.0 (Gemma 3 used Google's restricted custom license). Four sizes shipped: E2B, E4B, 26B-A4B (MoE), 31B dense.

Sources: [Google Blog](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/) | [HuggingFace Blog](https://huggingface.co/blog/gemma4)

### Architecture Delta: Dense 31B (Gemma 4) vs Gemma 3 27B

The 31B dense model shares the same fundamental decoder stack as Gemma 3 but introduces several new architectural mechanisms that Gemma 3 has no equivalent for:

| Feature | Gemma 3 | Gemma 4 31B / 26B-A4B |
|---|---|---|
| Attention type | Alternating local/global | Alternating `sliding_attention` / `full_attention` via explicit `layer_types` list |
| Head dim for full-attn | single `head_dim` | Separate `global_head_dim = 512`; local layers use `head_dim = 256` |
| K=V sharing | No | `attention_k_eq_v = True` on full-attention layers (v_proj is absent; k_proj output reused as V) |
| Per-layer scalar | No | `layer_scalar` buffer on every decoder layer |
| Per-layer embeddings (PLE) | No | Auxiliary `vocab_size_per_layer_input=262144`, `hidden_size_per_layer_input=256` residual fed into every layer |
| MoE block | No | `enable_moe_block` flag; 26B-A4B uses 128 experts, top-8, plus 1 shared expert, with a `Gemma4Router` that softmax-over-all → top-k → renormalize |
| Final logit softcapping | `final_logit_softcapping` (Gemma 2) removed in G3 | Reintroduced: `final_logit_softcapping = 30.0` |
| Context window | 128k | 256k (31B, 26B-A4B); 128k (E2B, E4B) |
| Multimodal | Vision only (VLM) | Vision + audio (E2B/E4B); Vision (31B/26B) |
| Reasoning ("thinking") | Not present | Configurable `<think>` mode |
| License | Google custom | Apache 2.0 |

### New `Gemma4TextConfig` Fields vs Gemma 3 TextConfig

Fields present in `Gemma4TextConfig` that have no Gemma 3 counterpart (from [HF transformers source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/configuration_gemma4.py)):

```
global_head_dim: int = 512          # head dim for full-attn layers
attention_k_eq_v: bool = False       # K=V sharing on global layers
layer_types: list[str] | None        # e.g. ["sliding_attention"]*5 + ["full_attention"]
num_global_key_value_heads: int      # override KV heads for global layers
num_kv_shared_layers: int = 0        # consecutive layers sharing KV projections
enable_moe_block: bool = False       # MoE vs dense FFN
num_experts: int | None              # 128 for 26B-A4B
top_k_experts: int | None            # 8 for 26B-A4B
moe_intermediate_size: int | None
use_double_wide_mlp: bool = False
vocab_size_per_layer_input: int      # PLE vocabulary
hidden_size_per_layer_input: int     # PLE hidden dim
use_bidirectional_attention: str | None  # "vision" or "all"
```

Fields **removed** from Gemma 3 in Gemma 4: `attn_logit_softcapping`, `query_pre_attn_scalar` (Gemma 2/3-specific).

### 26B-A4B MoE Details

- 26B total parameters, 3.8B active per forward pass
- 128 fine-grained experts + 1 shared expert per MoE block
- Top-8 routing with softmax-over-all-experts → top-k selection → renormalize
- `Gemma4Router` is a standalone `nn.Linear(hidden, num_experts)` outside the MoE block (not fused)
- Every decoder layer has a `layer_scalar` buffer (scalar multiplier applied to the attention/MoE output)
- MoE intermediate per expert: `gate_up_proj [num_experts, 2*intermediate, hidden]`, `down_proj [num_experts, hidden, intermediate]`, `per_expert_scale [num_experts]`

---

## 2. vLLM / HF Port Path for Gemma 4

### Transformers

Gemma 4 support was added in **transformers >= 5.5.0** (April 2026). The PR added a new `models/gemma4/` package rather than extending `gemma3`. The `model_type` string is `"gemma4"` (distinct from `"gemma3"`). Key transformers classes:

- `Gemma4TextModel`, `Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration`
- `Gemma4VisionModel`, `Gemma4AudioModel`
- `Gemma4Config` (wraps `Gemma4TextConfig` + `Gemma4VisionConfig` + `Gemma4AudioConfig`)

### vLLM

**PR #38826** by Luciano Martins (DeepMind DevRel) added Gemma 4 support. PR #38837 fixed a `Gemma4ToolParser.__init__()` missing `tools` parameter. Official Docker image: `vllm/vllm-openai:gemma4-cu130`.

What changed vs Gemma 3 in vLLM:
- New `Gemma4ForConditionalGeneration` model class wrapping an FP multimodal path
- `FusedMoE` block for the 26B-A4B MoE variant with custom `Gemma4Router`
- Attention dispatch: K=V sharing (no v_proj) on global layers; two separate rotary embeddings (local head_dim=256, global head_dim=512)
- TP=1 only currently for the MoE model (Expert Parallelism not yet landed; tracked in [vLLM #39595](https://github.com/vllm-project/vllm/issues/39595))
- `--tool-call-parser gemma4 --reasoning-parser gemma4` required at serve time

Source: [NVIDIA Developer Forums on Gemma4 vLLM PRs](https://forums.developer.nvidia.com/t/gemma-4-models-which-vllm-version-any-prs-spotted/365490) | [vLLM Issue #38868](https://github.com/vllm-project/vllm/issues/38868)

### TRT-LLM Status

TRT-LLM's classic conversion path (`examples/models/core/gemma/convert_checkpoint.py`) only handles `model_type == "gemma2"` and earlier Gemma 3 variants. **There is no Gemma 4 handler in the classic path.**

Gemma 4 support was added via **AutoDeploy** in [PR #12710](https://github.com/NVIDIA/TensorRT-LLM/pull/12710) (`[#12808][feat] AutoDeploy: Add Gemma4 Support`). Files changed:

```
tensorrt_llm/_torch/auto_deploy/models/custom/modeling_gemma4.py   # new, full custom impl
examples/auto_deploy/model_registry/configs/gemma4_moe.yaml         # new
examples/auto_deploy/model_registry/configs/gemma4_moe_base.yaml    # new
examples/auto_deploy/model_registry/models.yaml                      # entries added
```

The custom `modeling_gemma4.py` is a standalone self-contained reimplementation — it does **not** patch or extend `Gemma3ForCausalLM`. It includes its own `Gemma4TextConfig(model_type="gemma4_text")` that TRT-LLM resolves independently of transformers' version. The Gemma4Config wrapper sets `model_type = "gemma4"` to match HF checkpoint `config.json`.

**Blocker (as of 2026-05-11):** The builtin TRT-LLM runtime ships transformers 4.57.3; loading an NVFP4 Gemma4 checkpoint requires transformers >= 5.5.0. Forcing 5.5.0 inside the TRT-LLM container breaks other imports. Issue tracked at [TRT-LLM #12764](https://github.com/NVIDIA/TensorRT-LLM/issues/12764). Errors seen:
1. `AttributeError: list object has no attribute keys` — `extra_special_tokens` serialized as list not mapping
2. `ValueError: The checkpoint you are trying to load has model type 'gemma4' but Transformers does not recognize this architecture.`
3. `ImportError: cannot import name AutoModelForVision2Seq from transformers` when forcing 5.5.0

**AutoDeploy path (working):** Uses the standalone `modeling_gemma4.py` and does not rely on the transformers Gemma4 handler, so this path works with the container's current transformers version. Use:

```
python examples/auto_deploy/simple_deploy_and_run.py \
  --model google/gemma-4-26B-A4B-it \
  --config-id gemma4_moe
```

### Minimum Patch to Make TRT-LLM's Gemma3 Handler Accept Gemma 4

The Gemma3 classic handler will not work for Gemma 4 without non-trivial surgery because the architectures differ too much (no v_proj on global layers, MoE block, layer_types dispatch, per-layer scalar, PLE, different head dims). However, for the **dense 31B model only**, the following would be the minimum required:

1. Add `model_type == "gemma4"` as an alias in the `convert_checkpoint.py` model_type detection block alongside `"gemma3"`.
2. Map `global_head_dim` → `head_dim` in the TRT-LLM config for global attention layers (or unify to the larger value).
3. Strip the `layer_types` list and treat all layers as global (loses sliding window correctness but allows loading).
4. Set `final_logit_softcapping = config.final_logit_softcapping` (reintroduced from Gemma 2).
5. No MoE support — the 26B-A4B MoE model needs a FusedMoE block that does not exist in the Gemma3 handler.

**Recommendation: use the AutoDeploy path instead.** It is the official supported path, already merged, and handles both dense and MoE variants correctly.

---

## 3. ModelOpt NVFP4 Quantization Workflow

### Format Recap

NVFP4 (E2M1): 1 sign + 2 exponent + 1 mantissa. Block-wise with block size 16: each 16-element group gets an FP8 (E4M3) scale; a second FP32 per-tensor scale normalizes the full tensor. Effective storage: ~4.5 bits/value. Compression vs BF16: ~3.5x.

Reference: [NVIDIA NVFP4 Blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

### Tool: `hf_ptq.py` (NVIDIA Model-Optimizer)

Primary script: `examples/llm_ptq/hf_ptq.py` in [NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer).

**Basic NVFP4 command:**

```bash
python hf_ptq.py \
  --pyt_ckpt_path Qwen/Qwen3.6-27B \
  --qformat nvfp4 \
  --export_path ./qwen3.6-27b-nvfp4 \
  --calib_size 512 \
  --calib_max_seq_length 2048
```

**MoE-specific variant (recommended for 26B-A4B):**

```bash
python hf_ptq.py \
  --pyt_ckpt_path google/gemma-4-26B-A4B-it \
  --qformat nvfp4_experts_only \   # quantizes only expert MLP layers
  --export_path ./gemma4-26b-nvfp4 \
  --calib_size 512
```

**Available `--qformat` variants for NVFP4:**

| Format | What gets quantized |
|---|---|
| `nvfp4` | All linear layers (weights + activations) |
| `nvfp4_awq` | Lite AWQ calibration variant |
| `nvfp4_mse` | MSE sweep calibration (slower, higher quality) |
| `nvfp4_mlp_only` | Only MLP/MoE layers; attention stays BF16 |
| `nvfp4_experts_only` | Only `*mlp.experts*` and `*block_sparse_moe*` paths (NVIDIA's choice for Gemma4 MoE) |
| `nvfp4_omlp_only` | MLP + o_proj |
| `nvfp4_rotate` | With KV cache rotation |

**Memory-constrained flags:**

```bash
--low_memory_mode          # compresses weights during model load; prevents BF16+quant coexisting in VRAM
--use_seq_device_map       # sequential device map across GPUs; uses up to 80% of each
```

**KV cache quantization:**

```bash
--kv_cache_qformat fp8     # match NVIDIA's official quantization (FP8 KV cache)
```

### What `hf_quant_config.json` Looks Like (Producer Field)

From `nvidia/Gemma-4-31B-IT-NVFP4` (modelopt v0.37.0) and `nvidia/Gemma-4-26B-A4B-NVFP4` (modelopt v0.43.0):

```json
{
    "producer": {
        "name": "modelopt",
        "version": "0.37.0"
    },
    "quantization": {
        "quant_algo": "NVFP4",
        "kv_cache_quant_algo": "FP8",
        "group_size": 16,
        "exclude_modules": [
            "lm_head",
            "model.embed_vision*",
            "model.language_model.layers.*.self_attn*",
            "model.vision_tower*"
        ]
    }
}
```

**Critical for TRT-LLM:** `quant_method` as seen by vLLM/TRT-LLM is `"modelopt"` — set `--quantization modelopt` at serve time. The `producer.name` field in the JSON is the internal identifier; frameworks check for `quant_algo: "NVFP4"` inside the `quantization` block.

### Calibration Data

Default calibration mix in `hf_ptq.py`: `cnn_dailymail` + `nemotron-post-training-dataset-v2`. Typical calibration: 512 samples at 2048 tokens. Custom dataset possible via `--calib_dataset` flag.

### Memory Footprint During Calibration

**Default mode** (no `--low_memory_mode`): the full BF16 model is loaded into GPU/CPU memory first, then quantized layer by layer. For a 52 GB BF16 model (Qwen3.6-27B), peak VRAM usage is approximately 52–60 GB BF16 weights + ~8–12 GB activation buffers + ~15 GB quantized weights being built = **~75–85 GB peak**. This is tight but feasible on 96 GB with no other resident processes.

**`--low_memory_mode`** quantizes weights during the loading phase, preventing BF16 and quantized from fully coexisting. With this flag, a 52 GB BF16 source loads to approximately 20–25 GB NVFP4 equivalent with only a small rolling BF16 buffer. Recommended for 96 GB with headroom to spare.

**Real-world data point:** A community user quantized Qwen3.6-27B (51 GB BF16) to NVFP4 (26 GB) on a **single RTX PRO 6000 96 GB** in approximately **57 minutes** using llm-compressor (similar workflow), 512 samples, 4096 tokens each.

Source: [AEON-7 Qwen3.6-27B-NVFP4 GitHub](https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DFlash)

### MoE Calibration Notes

`mtq.quantize()` with `nvfp4_experts_only` skips attention layers entirely and calibrates only expert linear projections. The `sync_expert_weight_amax` utility handles routing-aware expert scale synchronization across TP ranks. MoE expert dimensions are handled correctly: `gate_up_proj [num_experts, 2*intermediate, hidden]` tensors are per-expert-blocked.

### Output Format

`hf_ptq.py` with `--export_hf True` (default for HF checkpoints) produces:
- `config.json` (unchanged from source, except `quantization_config` block added)
- `hf_quant_config.json` (shown above)
- `model-XXXXX-of-YYYYY.safetensors` — weights stored in mixed precision (BF16 non-quantized, INT8/U8 for quantized weights, F8_E4M3 for per-group scales)
- Directly loadable by vLLM (`--quantization modelopt`), TRT-LLM AutoDeploy, and SGLang (`--quantization modelopt_fp4`)

The checkpoint does **not** require a separate engine-build step like the old `trtllm-build` path — it is a standard HF checkpoint that inference frameworks load natively.

---

## 4. Concrete Next-Step Recommendations

### Priority Order for DIY NVFP4 on the 96 GB Box

**Target 1 (highest value, fastest): `Qwen/Qwen3.6-27B`**

- 51 GB BF16 → ~15 GB NVFP4 (weights-only `nvfp4_mlp_only`) or ~15 GB with attention excluded
- Already in the toolkit as a profile; NVFP4 would push decode speed from ~25 tok/s to ~70–90 tok/s on sm_120
- Command:
  ```bash
  python hf_ptq.py \
    --pyt_ckpt_path Qwen/Qwen3.6-27B \
    --qformat nvfp4 \
    --kv_cache_qformat fp8 \
    --export_path ${LLM_HOME}/models/nvfp4/qwen3.6-27b-nvfp4 \
    --calib_size 512 \
    --low_memory_mode
  ```
- Estimated time: ~1 hour on single 96 GB card
- Serve: `vllm serve ${LLM_HOME}/models/nvfp4/qwen3.6-27b-nvfp4 --quantization modelopt --max-model-len 131072`

**Target 2: `google/gemma-4-31B-it`**

- NVIDIA has already released `nvidia/Gemma-4-31B-IT-NVFP4` (modelopt v0.37.0, 21 GB on disk). **Download instead of DIY-quantizing.**
- 60 layers, self-attn excluded from quantization in NVIDIA's build
- Serve via TRT-LLM AutoDeploy (PR #12710 merged) or vLLM:
  ```bash
  vllm serve nvidia/Gemma-4-31B-IT-NVFP4 --quantization modelopt --tensor-parallel-size 1
  ```
- Requires vLLM with Gemma4 support (PR #38826), transformers >= 5.5.0

**Target 3: `google/gemma-4-26B-A4B-it` (MoE)**

- NVIDIA released `nvidia/Gemma-4-26B-A4B-NVFP4` (modelopt v0.43.0, 18.8 GB on disk). **Download, not DIY.**
- Use `nvfp4_experts_only` recipe (what NVIDIA used)
- MoE vLLM constraint: TP=1 only (issue [#39595](https://github.com/vllm-project/vllm/issues/39595))
- Serve:
  ```bash
  vllm serve nvidia/Gemma-4-26B-A4B-NVFP4 \
    --tool-call-parser gemma4 --reasoning-parser gemma4 \
    --enable-auto-tool-choice --trust-remote-code
  ```

**Target 4: Larger MoE (future)**

- `Qwen3-235B-A22B` → NVFP4 ~75 GB; fits in 96 GB with `--low_memory_mode` + `--use_seq_device_map`
- Would require multi-hour calibration run; viable but not the first experiment

### TRT-LLM Path Decision

Do **not** attempt to patch `Gemma3ForCausalLM` to accept `model_type=gemma4`. The divergence (K=V attention, global_head_dim, layer_types dispatch, per-layer scalars, PLEs, MoE block) is too large for a compatible shim. Instead:

1. Use `vllm serve --quantization modelopt` for Gemma4 NVFP4 HF checkpoints today (PR #38826 merged).
2. Use TRT-LLM AutoDeploy path (PR #12710) for TRT-LLM engine-based deployment — it uses the standalone `modeling_gemma4.py` that does not depend on transformers Gemma4 support.
3. Avoid the TRT-LLM builtin runtime with Gemma4 NVFP4 until the transformers version skew (4.57.3 vs 5.5.0) is resolved in an official release.

---

## Source URLs

- [Google Gemma 4 Blog](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
- [HuggingFace Gemma 4 Blog](https://huggingface.co/blog/gemma4)
- [HF Gemma4TextConfig source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/configuration_gemma4.py)
- [nvidia/Gemma-4-31B-IT-NVFP4](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4)
- [nvidia/Gemma-4-26B-A4B-NVFP4](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4)
- [vLLM Gemma4 Blog Post](https://vllm.ai/blog/gemma4)
- [vLLM Issue #38868: gemma4 model_type not recognized](https://github.com/vllm-project/vllm/issues/38868)
- [vLLM Gemma4 Usage Guide](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html)
- [TRT-LLM Issue #12764: Gemma4 NVFP4 runtime skew bug](https://github.com/NVIDIA/TensorRT-LLM/issues/12764)
- [TRT-LLM modeling_gemma4.py (AutoDeploy)](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/auto_deploy/models/custom/modeling_gemma4.py)
- [TRT-LLM gemma4_moe.yaml config](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/auto_deploy/model_registry/configs/gemma4_moe.yaml)
- [NVIDIA Model-Optimizer GitHub](https://github.com/NVIDIA/Model-Optimizer)
- [Model-Optimizer llm_ptq README](https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/llm_ptq/README.md)
- [NVIDIA NVFP4 Intro Blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [NVIDIA PTQ Blog](https://developer.nvidia.com/blog/optimizing-llms-for-performance-and-accuracy-with-post-training-quantization/)
- [AEON-7 Qwen3.6-27B NVFP4 on RTX PRO 6000 (57 min timing)](https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DFlash)
- [nvfp4-vllm: confirmed SM12.0 working](https://github.com/kelnei/nvfp4-vllm)
- [NVFP4 quantization (DGX Spark playbooks)](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/nvfp4-quantization)
- [Red Hat NVFP4 article](https://developers.redhat.com/articles/2026/02/04/accelerating-large-language-models-nvfp4-quantization)
- [Google Codelabs: Gemma4 + RTX 6000 Pro + vLLM](https://codelabs.developers.google.com/codelabs/cloud-run/cloud-run-gpu-rtx-pro-6000-gemma4-vllm)
