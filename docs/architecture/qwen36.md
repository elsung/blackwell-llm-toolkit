# Qwen3.5 / Qwen3.6 Architecture Research vs Qwen3-VL

**Research date:** 2026-05-11  
**Target model on disk:** `${LLM_HOME}/models/hf/Qwen__Qwen3.6-27B` (`qwen3_5`, `Qwen3_5ForConditionalGeneration`)

---

## 1. Family Overview

### Timeline and releases

| Model | Release date | Model type | Architecture class |
|-------|-------------|-----------|-------------------|
| Qwen3-VL (2B/8B/72B) | Late 2025 | `qwen3_vl` | `Qwen3VLForConditionalGeneration` |
| Qwen3.5 (0.8B–397B-A17B) | February 2026 | `qwen3_5` / `qwen3_5_moe` | `Qwen3_5ForConditionalGeneration` / `Qwen3_5MoeForConditionalGeneration` |
| Qwen3.6-35B-A3B | April 16, 2026 | `qwen3_5` (MoE) | `Qwen3_5MoeForConditionalGeneration` |
| Qwen3.6-27B | April 22, 2026 | `qwen3_5` | `Qwen3_5ForConditionalGeneration` |

The "3.6" branding is a fine-tuning/capability bump over 3.5 on the **same architecture**; both use `model_type: qwen3_5`. Qwen3.6-27B is the 3.5-27B architecture retrained with stronger agentic-coding data and a new `preserve_thinking` chat-template feature.

### What is officially new vs Qwen3-VL

1. **Hybrid attention backbone**: Qwen3.5/3.6 replaces all-softmax attention (Qwen3-VL) with a hybrid layer pattern — 3 Gated DeltaNet (linear attention) layers followed by 1 standard full-attention layer, repeating. For the 27B: 64 layers = 48 linear + 16 full. This is the **primary and fundamental architectural change**.
2. **Multi-Token Prediction (MTP) training**: weights include a 1-layer MTP head (`mtp_num_hidden_layers = 1`) for speculative decoding at serving time. Qwen3-VL has no MTP.
3. **Gated output on full-attention layers**: `attn_output_gate: true` with `output_gate_type: "swish"` on the full-attention layers. Not present in Qwen3-VL.
4. **Partial RoPE (`partial_rotary_factor: 0.25`)**: Only 25% of the head dimensions are rotary-encoded in the full-attention layers (head_dim=256, RoPE dim=64). Qwen3-VL uses full RoPE with `rope_scaling` dict and `rope_theta` as a flat field.
5. **MRoPE field name change**: `rope_parameters` (nested dict) replaces the flat `rope_scaling` + `rope_theta` pair used in Qwen3-VL.
6. **Larger vocab**: 248,320 tokens (vs 151,936 for Qwen3-VL).
7. **Vision encoder**: structurally identical to Qwen3-VL (same field names, same ViT design), sizes differ only for the 27B (depth 27, hidden_size 1152 vs depth 24, hidden_size 1024 for the 2B).

---

## 2. Config-Level Deltas

### Complete side-by-side (top-level fields)

| Field | Qwen3-VL-2B config | Qwen3.6-27B config | Notes |
|---|---|---|---|
| `architectures` | `["Qwen3VLForConditionalGeneration"]` | `["Qwen3_5ForConditionalGeneration"]` | Renamed class |
| `model_type` | `"qwen3_vl"` | `"qwen3_5"` | NEW type string |
| `image_token_id` | 151655 | 248056 | Different vocab |
| `video_token_id` | 151656 | 248057 | Different vocab |
| `vision_start_token_id` | 151652 | 248053 | Different vocab |
| `vision_end_token_id` | 151653 | 248054 | Different vocab |
| `language_model_only` | absent | `false` | New optional field |

### text_config: fields present in Qwen3.5/3.6 but ABSENT in Qwen3-VL

| New field | Value in Qwen3.6-27B | What it does |
|---|---|---|
| `layer_types` | 64-element list of `"linear_attention"` / `"full_attention"` | Explicit per-layer dispatch — the main gating signal for the hybrid runner |
| `linear_conv_kernel_dim` | `4` | Depthwise causal conv kernel size for GatedDeltaNet layers |
| `linear_key_head_dim` | `128` | QK head dim in linear attention |
| `linear_value_head_dim` | `128` | V head dim in linear attention |
| `linear_num_key_heads` | `16` | Number of QK heads in GDN layers |
| `linear_num_value_heads` | `48` | Number of V heads in GDN layers |
| `attn_output_gate` | `true` | Enables output gating on full-attention layers |
| `output_gate_type` | `"swish"` | Activation for that gate |
| `mamba_ssm_dtype` | `"float32"` | State dtype for SSM states in GDN |
| `mtp_num_hidden_layers` | `1` | MTP head layers (for speculative decoding) |
| `mtp_use_dedicated_embeddings` | `false` | Whether MTP head has its own embeddings |
| `partial_rotary_factor` | `0.25` | Fraction of head_dim that receives RoPE |
| `full_attention_interval` | `4` | Pattern period (inferred; every 4th layer is full) |
| `model_type` (inner) | `"qwen3_5_text"` | Sub-config model_type (was `"qwen3_vl_text"`) |

### text_config: fields CHANGED in Qwen3.5/3.6 vs Qwen3-VL

| Field | Qwen3-VL | Qwen3.5/3.6 27B | Notes |
|---|---|---|---|
| RoPE field name | `rope_scaling` (dict) + `rope_theta` (float, top-level) | `rope_parameters` (nested dict, includes theta) | Renamed + restructured |
| `rope_parameters.mrope_section` | `[24, 20, 20]` | `[11, 11, 10]` | Different section splits |
| `head_dim` | `128` | `256` | Full-attention head dim doubled |
| `num_attention_heads` | `16` | `24` | More heads in full-attention layers |
| `num_key_value_heads` | `8` | `4` | GQA ratio changed |
| `vocab_size` | `151,936` | `248,320` | Larger vocabulary |
| `hidden_size` | `2,048` (2B) | `5,120` | Scale |
| `num_hidden_layers` | `28` | `64` | Depth |

### vision_config: delta

The vision config field names are **identical** between Qwen3-VL and Qwen3.5/3.6. Values differ by scale:

| Field | Qwen3-VL-2B | Qwen3.6-27B |
|---|---|---|
| `model_type` | `"qwen3_vl"` | `"qwen3_5"` | ← only change that matters for dispatch |
| `depth` | 24 | 27 |
| `hidden_size` | 1024 | 1152 |
| `intermediate_size` | 4096 | 4304 |
| `out_hidden_size` | 2048 | 5120 |
| `deepstack_visual_indexes` | `[5, 11, 17]` | `[]` (empty) | Deepstack disabled in 3.6-27B |

The `deepstack_visual_indexes` field exists in both but is non-empty only in some Qwen3-VL variants. **This is likely the source of the `'NoneType' object does not support item assignment` TypeError** in TRT-LLM: the Qwen3-VL handler presumably indexes into this list expecting populated entries, but finds an empty list or mishandles its absence.

---

## 3. Forward-Pass / Weight-Layout Deltas

### Hybrid layer dispatch

Every layer in Qwen3.5/3.6 is one of two types determined by `layer_types[i]`:

- **`"linear_attention"` (GatedDeltaNet)**: uses projection matrices `in_proj_qkvz` (Q/K/V/Z fused), `in_proj_ba` (beta/alpha state projections), a depthwise causal conv1d (kernel=4), L2-normalized Q/K, a fixed-size matrix state (NOT a growing KV cache), and `out_proj`. No RoPE. Needs `A_log`, `dt_bias` per head.
- **`"full_attention"` (Gated Softmax Attention)**: standard GQA with partial RoPE (only 64 of 256 dims get positional encoding), plus an output gate controlled by `attn_output_gate` / `output_gate_type`. Projects are named normally but the output has a swish-gated multiplier.

Neither type exists in Qwen3-VL. TRT-LLM's `Qwen3VLModel` only handles the full-attention path.

### MTP head

Weight key prefix: `model.mtp_layers[0].*` — one transformer layer that predicts the next token in parallel during training. At inference time it's used as a draft head for speculative decoding (`--speculative-algo NEXTN`). TRT-LLM has no handler for this prefix.

### Module name changes (weight keys)

GatedDeltaNet layers emit weight keys such as:
```
model.layers.N.self_attn.in_proj_qkvz.weight
model.layers.N.self_attn.in_proj_ba.weight
model.layers.N.self_attn.A_log
model.layers.N.self_attn.dt_bias
model.layers.N.self_attn.conv1d.weight
model.layers.N.self_attn.conv1d.bias
```
Qwen3-VL uses `q_proj`, `k_proj`, `v_proj`, `o_proj` — entirely different projection names.

---

## 4. vLLM's Port Recipe

### PR reference

**vLLM PR #34110** — "Qwen3.5: Full support for the Qwen3.5 model family featuring GDN (Gated Delta Networks), with FP8 quantization, MTP speculative decoding"  
Merged in **vLLM v0.17.0** (February 2026).  
URL: https://github.com/vllm-project/vllm/pull/34110

### What vLLM added

1. **New file `vllm/model_executor/models/qwen3_5.py`** implementing:
   - `Qwen3_5ForCausalLM` — text-only dense path
   - `Qwen3_5ForConditionalGeneration` (inherits from `Qwen3VLForConditionalGeneration` AND `IsHybrid`) — multimodal path
   - `Qwen3_5MoeForConditionalGeneration` — MoE variant
   - `Qwen3_5MtpForConditionalGeneration` — MTP draft head
2. **Hybrid KV cache manager** — manages two different cache types simultaneously: standard KV tensors for full-attention layers, and fixed-size matrix states for GDN layers (using `MambaStateDtypeCalculator` / `MambaStateShapeCalculator`).
3. **Triton kernels from Flash Linear Attention** — `fused_recurrent_gated_delta_rule` for GDN forward/backward. Not needed for TRT-LLM inference but necessary to understand the op signature.
4. **Module fusing maps**: `packed_modules_mapping` covers `in_proj_qkvz` and `in_proj_ba` as merged column-parallel projections (analogous to `qkv_proj` in standard attention).
5. **`IsHybrid` mixin** — signals to the CUDA graph runner and decode loop that some layers use recurrent state rather than KV cache. TRT-LLM would need an equivalent signal.
6. **layer_types dispatch in forward()** — iterates layers, checks `config.text_config.layer_types[i]`, routes to either `Qwen3_5GatedDeltaNet` or `Qwen3_5Attention` module.

### Key vLLM config reading pattern (recipe to mirror)

```python
# In model __init__:
layer_types = config.text_config.layer_types  # list[str]
for i, layer_type in enumerate(layer_types):
    if layer_type == "linear_attention":
        self.layers.append(Qwen3_5GatedDeltaNetLayer(...))
    else:  # "full_attention"
        self.layers.append(Qwen3_5AttentionLayer(...))

# GDN layer reads:
conv_kernel = config.text_config.linear_conv_kernel_dim   # 4
kh          = config.text_config.linear_num_key_heads     # 16
vh          = config.text_config.linear_num_value_heads   # 48
k_head_dim  = config.text_config.linear_key_head_dim      # 128
v_head_dim  = config.text_config.linear_value_head_dim    # 128
ssm_dtype   = config.text_config.mamba_ssm_dtype          # "float32"

# Full-attention layer reads:
use_output_gate = config.text_config.attn_output_gate     # True
gate_act_type   = config.text_config.output_gate_type     # "swish"
partial_rope    = config.text_config.partial_rotary_factor # 0.25

# RoPE (new field name):
rope_params = config.text_config.rope_parameters          # dict
```

### Known vLLM bugs/issues post-merge (useful to know for TRT-LLM)

- [#35924](https://github.com/vllm-project/vllm/issues/35924): `in_proj_ba` fails Marlin MIN_THREAD_N=64 at TP>=4 (quantization edge case)
- [#38041](https://github.com/vllm-project/vllm/issues/38041): V2 model runner crashes on mixed attention (linear + full) — fixed in later release
- [#39273](https://github.com/vllm-project/vllm/issues/39273): ngram speculative decoding corrupts output on hybrid GDN models

---

## 5. Recommended TRT-LLM Patch Approach

TRT-LLM v1.2.1 has `Qwen3VLForConditionalGeneration` / `Qwen3VLModel` but these will fail on `qwen3_5` config for two reasons:
1. The model_type dispatcher rejects `"qwen3_5"`.
2. Even if aliased, the forward pass has no GDN path — every layer is routed to standard attention, which will crash or silently produce garbage on GDN weight shapes.

**This is NOT a simple rename/alias**. The GDN layers have completely different weight shapes and forward semantics.

### Minimum viable patch for TRT-LLM (text-only inference path, vision encoder pass-through)

#### Step 1: Register the new model_type

In `tensorrt_llm/models/modeling_utils.py` (or wherever TRT-LLM's `valid_types` list lives — cf. issue #11569 which showed the analogous problem for `qwen3_vl` itself):

```python
# Add "qwen3_5" alongside "qwen3_vl":
VALID_QWEN_TYPES = {"qwen", "qwen2", "qwen2_moe", "qwen2_vl", "qwen3", "qwen3_moe", "qwen3_vl", "qwen3_5"}
```

#### Step 2: Config adapter — read new field names without crashing

The TRT-LLM Qwen3-VL config reader almost certainly does `config["rope_scaling"]` and `config["rope_theta"]`. Qwen3.5/3.6 has `rope_parameters` instead. Add a shim:

```python
def _get_rope_config(text_cfg: dict) -> dict:
    if "rope_parameters" in text_cfg:          # qwen3_5 style
        rp = text_cfg["rope_parameters"]
        return {
            "rope_type": rp.get("rope_type", "default"),
            "mrope_interleaved": rp.get("mrope_interleaved", True),
            "mrope_section": rp.get("mrope_section", []),
            "rope_theta": rp.get("rope_theta", 10000000),
            "factor": rp.get("factor", 1.0),
        }
    else:                                       # qwen3_vl style
        return {**text_cfg.get("rope_scaling", {}), "rope_theta": text_cfg.get("rope_theta", 1000000)}
```

This addresses the `'NoneType' object does not support item assignment` crash: the crash happens because TRT-LLM's rope-config dict is built from `config["rope_scaling"]` which returns `None` for qwen3_5 configs (field doesn't exist → dict.get returns None → code tries to subscript it).

#### Step 3: `deepstack_visual_indexes` guard

The vision config has `"deepstack_visual_indexes": []` in Qwen3.6-27B (empty list, vs `[5, 11, 17]` in Qwen3-VL 2B). If TRT-LLM's vision builder does anything like `deepstack_visual_indexes[0]` without checking length, it will IndexError. Guard:

```python
dsi = vision_cfg.get("deepstack_visual_indexes", [])
if dsi:
    # do deepstack-specific path
else:
    # skip deepstack feature injection
```

#### Step 4: layer_types dispatch for text model

If TRT-LLM TensorRT engine only needs to handle the **full_attention layers** (e.g. for a partial/text-only demo), you can filter:

```python
layer_types = text_cfg.get("layer_types", None)
full_attn_layer_indices = [
    i for i, t in enumerate(layer_types) if t == "full_attention"
] if layer_types else list(range(num_hidden_layers))
```

However, this will not produce correct outputs — GDN layers occupy 75% of the network. The full fix requires implementing `Qwen3_5GatedDeltaNet` as a TRT plugin or using TRT-LLM's Mamba plugin (if compatible).

#### Step 5: MTP weight skip

The checkpoint has `model.mtp_layers[0].*` weights. TRT-LLM's weight loader will either error or ignore them. Add explicit skip:

```python
if "mtp_layers" in name:
    continue  # MTP head not used in standard inference
```

### Realistic assessment

Full native support (all 64 layers working correctly) requires:
- A TRT plugin or custom CUDA kernel for the recurrent GDN forward pass.
- A hybrid KV cache manager (existing KV cache for 16 full-attention layers + state buffers for 48 GDN layers).
- This is non-trivial — vLLM required a new model file of ~1,000+ lines.

**Minimum to get text generation working at all** (treating GDN layers as no-ops or as identity): Steps 1–3 above. Output quality will be completely wrong but it won't crash on config loading.

**For production quality**: Track TRT-LLM issue #11674 ("Support Qwen3.5 model in torch backend") — when that merges it will include the full GDN kernel integration.

---

## Sources

- [Qwen3.6-27B blog post](https://qwen.ai/blog?id=qwen3.6-27b)
- [Qwen/Qwen3.6-27B HuggingFace](https://huggingface.co/Qwen/Qwen3.6-27B)
- [Qwen/Qwen3.5-27B HuggingFace](https://huggingface.co/Qwen/Qwen3.5-27B)
- [HF Transformers Qwen3.5 docs](https://huggingface.co/docs/transformers/model_doc/qwen3_5) (transformers v5.8.0)
- [HF Transformers Qwen3-VL docs](https://huggingface.co/docs/transformers/main/model_doc/qwen3_vl)
- [Qwen3-VL-2B-Instruct config.json](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/main/config.json)
- [vLLM PR #34110](https://github.com/vllm-project/vllm/pull/34110) — Qwen3.5 initial support (merged v0.17.0)
- [vLLM issue #35391](https://github.com/vllm-project/vllm/issues/35391) — Qwen3_5ForConditionalGeneration not supported in v0.16.0
- [vLLM issue #35344](https://github.com/vllm-project/vllm/issues/35344) — Qwen3_5MoeForConditionalGeneration not supported
- [vLLM issue #35924](https://github.com/vllm-project/vllm/issues/35924) — GDN in_proj_ba Marlin TP>=4 bug
- [vLLM qwen3_5 model API docs](https://docs.vllm.ai/en/stable/api/vllm/model_executor/models/qwen3_5/)
- [vLLM blog: Qwen3-Next support](https://vllm.ai/blog/qwen3-next)
- [TRT-LLM issue #12321](https://github.com/NVIDIA/TensorRT-LLM/issues/12321) — Qwen3.5 support blocked by AutoModelForVision2Seq deprecation
- [TRT-LLM issue #11674](https://github.com/NVIDIA/TensorRT-LLM/issues/11674) — Support Qwen3.5 in torch backend
- [TRT-LLM issue #11569](https://github.com/NVIDIA/TensorRT-LLM/issues/11569) — Qwen3-VL valid_types AssertionError
- [GatedDeltaNet analysis gist](https://gist.github.com/justinchuby/0213aa253664fb72e9adb0089816de15)
- [Qwen3.5 GDN TP bug](https://github.com/vllm-project/vllm/issues/35924)
- [MarkTechPost Qwen3.6-27B release](https://www.marktechpost.com/2026/04/22/alibaba-qwen-team-releases-qwen3-6-27b-a-dense-open-weight-model-outperforming-397b-moe-on-agentic-coding-benchmarks/)
- [QwenLM/Qwen3.6 GitHub](https://github.com/QwenLM/Qwen3.6)
